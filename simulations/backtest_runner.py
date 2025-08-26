from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger

from services.signal_generator import SignalGenerator, TradingSignal

@dataclass
class BacktestTrade:
    """Container for individual backtest trades.

    Attributes:
        id: Unique trade id.
        symbol: Instrument symbol.
        direction: 'LONG' or 'SHORT'.
        entry_time: Timestamp of entry bar.
        entry_price: Executed entry price.
        size: Position size in *lots* (consistent with SignalGenerator).
        stop_loss: Stop price if set at entry (can be updated by strategy).
        take_profit: TP price if set at entry (can be updated by strategy).
        exit_time: Timestamp of exit bar (None until closed).
        exit_price: Executed exit price (None until closed).
        pnl: Profit/loss (priced in % of entry, not account currency).
        commission: Commission paid for this trade.
        slippage: Slippage cost for this trade.
    """

    id: str
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    size: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0


class BacktestRunner:
    """Minimal, readable backtest loop around the production SignalGenerator.

    The runner deliberately **reuses** `SignalGenerator` to avoid train/serve
    skew. It iterates bar-by-bar, feeds the *rolling* historical window into
    the model, and simulates order execution using next-bar open and bar
    high/low for TP/SL checks.

    Assumptions & Simplifications
    -----------------------------
    - Execution at next-bar *open* for entries and exits triggered by signals.
    - TP/SL are checked *within* the same bar using its high/low range.
    - PnL is reported as a percentage return relative to entry price.
    - No commissions/slippage modeled (add easily where indicated).

    Parameters
    ----------
    data_manager : DataManager
        Used only for potential persistence or auxiliary I/O in future; not
        strictly required for an in-memory run.
    websocket_broadcaster : WebSocketBroadcaster
        If provided and it exposes `broadcast_backtest_update`, the runner
        will send periodic progress snapshots.
    config : Optional[Dict[str, Any]]
        Optional runner-level overrides, e.g., starting_balance, update_every.
    """

    def __init__(
        self,
        data_manager: Any,
        websocket_broadcaster: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data_manager = data_manager
        self.websocket_broadcaster = websocket_broadcaster
        self.config = {
            "starting_balance": 10000.0,  # virtual account for metrics only
            "update_every": 200,           # how many bars between WS updates
            "allow_reverse": True,         # close & flip when opposite signal
            "commission_rate": 0.0001,     # 0.01% commission per trade
            "slippage_rate": 0.00005,      # 0.005% slippage per trade
        }
        if config:
            self.config.update(config)

        # Reuse production signal generator
        self.signal_generator = SignalGenerator()

    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    raise ValueError("Backtest `data` must have a DatetimeIndex or a 'timestamp' column.")
        return df

    async def run_backtest(
        self,
        symbol: str,
        data: pd.DataFrame,
        model: Any,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the backtest and return aggregate metrics.

        Parameters
        ----------
        symbol : str
            Instrument to backtest.
        data : pd.DataFrame
            OHLCV(+indicators) indexed by datetime. Must include at least
            `open`, `high`, `low`, `close`. Indicators (e.g., ATR) are used if
            present for better stop distances.
        model : Any
            Trained PPO model compatible with `PPOTradingAgent.predict`.
        session_id : Optional[str]
            Identifier for UI/backend correlation.

        Returns
        -------
        Dict[str, Any]
            Summary metrics and a compact trade ledger.
        """
        if data is None or data.empty:
            logger.warning("Backtest received empty data frame; aborting.")
            return {"success": False, "error": "empty_data"}
        else:
            data = self._ensure_datetime_index(data)

        # Attach the trained model to the signal generator
        self.signal_generator.ppo_agent.model = model

        # Working state
        equity = self.config["starting_balance"]
        peak_equity = equity
        drawdown_max = 0.0
        position: Optional[BacktestTrade] = None
        trades: List[BacktestTrade] = []
        returns: List[float] = []  # per-bar strategy returns

        # We need enough history for the feature extractor inside SignalGenerator.
        lookback = int(self.signal_generator.config['environment']['lookback_window'])
        if len(data) <= lookback + 2:
            logger.warning("Insufficient bars for backtest lookback window.")
            return {"success": False, "error": "insufficient_data"}

        # Iterate over bars; i points to the *close* of the current decision bar
        # Signals are executed at the **next** bar open (i+1), where available.
        for i in range(lookback, len(data) - 1):
            window = data.iloc[: i + 1]  # inclusive of bar i
            decision_bar = window.index[-1]
            next_bar = data.iloc[i + 1]

            try:
                signal: Optional[TradingSignal] = self.signal_generator.generate_signal(
                    symbol=symbol,
                    market_data=window,
                    current_tick=None,       # use last close as entry in generator
                    account_info=None,
                )
            except Exception as e:
                logger.error(f"Signal generation error at {decision_bar}: {e}")
                signal = None

            # Handle open position: check TP/SL within **next** bar's range
            if position is not None:
                hit_tp, hit_sl = False, False
                # Long position TP/SL conditions
                if position.direction == 'LONG':
                    if position.take_profit is not None and next_bar.high >= position.take_profit:
                        hit_tp = True
                    if position.stop_loss is not None and next_bar.low <= position.stop_loss:
                        hit_sl = True
                # Short position TP/SL conditions
                else:
                    if position.take_profit is not None and next_bar.low <= position.take_profit:
                        hit_tp = True
                    if position.stop_loss is not None and next_bar.high >= position.stop_loss:
                        hit_sl = True

                exit_reason = None
                if hit_tp or hit_sl:
                    exit_reason = 'TP' if hit_tp else 'SL'
                elif signal and self.config["allow_reverse"]:
                    # Opposite signal => exit and flip
                    if (position.direction == 'LONG' and signal.action == 'SELL') or (position.direction == 'SHORT' and signal.action == 'BUY'):
                        exit_reason = 'REVERSE'

                if exit_reason:
                    # Execute exit at next bar open
                    position.exit_time = next_bar.name
                    position.exit_price = float(next_bar.open)
                    
                    # Calculate transaction costs
                    commission = position.size * position.exit_price * self.config["commission_rate"]
                    slippage = position.size * position.exit_price * self.config["slippage_rate"]
                    position.commission += commission
                    position.slippage += slippage
                    
                    # PnL in % of entry
                    if position.direction == 'LONG':
                        ret = (position.exit_price - position.entry_price) / position.entry_price
                    else:
                        ret = (position.entry_price - position.exit_price) / position.entry_price
                    
                    # Subtract transaction costs from return
                    total_costs = (commission + slippage) / (position.size * position.entry_price)
                    position.pnl = ret - total_costs
                    
                    trades.append(position)

                    equity *= (1.0 + position.pnl)  # simple compounding on notional
                    peak_equity = max(peak_equity, equity)
                    drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                    drawdown_max = max(drawdown_max, drawdown)
                    returns.append(position.pnl)

                    # Close position
                    position = None

                    # Optional: when reversing, open the new one immediately below

            # Entry logic (either flat, or after closing above)
            if position is None and signal and signal.action in ('BUY', 'SELL'):
                # Execute at next bar open
                direction = 'LONG' if signal.action == 'BUY' else 'SHORT'
                
                # Calculate entry transaction costs
                entry_commission = signal.lot_size * next_bar.open * self.config["commission_rate"]
                entry_slippage = signal.lot_size * next_bar.open * self.config["slippage_rate"]
                
                entry = BacktestTrade(
                    id=signal.id,
                    symbol=symbol,
                    direction=direction,
                    entry_time=next_bar.name,
                    entry_price=float(next_bar.open),
                    size=signal.lot_size,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    commission=entry_commission,
                    slippage=entry_slippage,
                )
                position = entry

            # Periodic progress broadcast
            if (
                self.websocket_broadcaster 
                and hasattr(self.websocket_broadcaster, 'broadcast_performance_update')
                and (i % int(self.config['update_every']) == 0)
            ):
                try:
                    await self.websocket_broadcaster.broadcast_performance_update({
                        'session_id': session_id,
                        'symbol': symbol,
                        'type': 'backtest_progress',
                        'progress_index': i,
                        'progress_total': len(data),
                        'equity': equity,
                        'total_return': (equity / self.config['starting_balance']) - 1.0,
                        'max_drawdown': float(drawdown_max),
                        'num_trades': len(trades),
                        'win_rate': float(np.mean([1.0 if t.pnl and t.pnl > 0 else 0.0 for t in trades])) if trades else 0.0,
                        'open_position': None if position is None else {
                            'direction': position.direction,
                            'entry_price': position.entry_price,
                            'stop_loss': position.stop_loss,
                            'take_profit': position.take_profit,
                            'unrealized_pnl': self._calculate_unrealized_pnl(position, data.iloc[i + 1]) if i + 1 < len(data) else 0.0,
                        },
                        'timestamp': decision_bar.isoformat(),
                    })
                except Exception as e:
                    logger.debug(f"WebSocket backtest update failed (ignored): {e}")

        # If a position remains open at the end, close at the final close
        if position is not None:
            last_bar = data.iloc[-1]
            position.exit_time = last_bar.name
            position.exit_price = float(last_bar.close)
            
            # Calculate final transaction costs
            commission = position.size * position.exit_price * self.config["commission_rate"]
            slippage = position.size * position.exit_price * self.config["slippage_rate"]
            position.commission += commission
            position.slippage += slippage
            
            if position.direction == 'LONG':
                ret = (position.exit_price - position.entry_price) / position.entry_price
            else:
                ret = (position.entry_price - position.exit_price) / position.entry_price
            
            # Subtract transaction costs from return
            total_costs = (commission + slippage) / (position.size * position.entry_price)
            position.pnl = ret - total_costs
            
            trades.append(position)
            equity *= (1.0 + position.pnl)
            peak_equity = max(peak_equity, equity)
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            drawdown_max = max(drawdown_max, drawdown)
            returns.append(position.pnl)

        # Aggregate metrics
        total_return = (equity / self.config['starting_balance']) - 1.0
        win_rate = (
            np.mean([1.0 if t.pnl and t.pnl > 0 else 0.0 for t in trades])
            if trades else 0.0
        )
        # Simple Sharpe: mean/ std of per-trade returns (bar-level would also be fine)
        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 1e-12:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))  # daily-ish scaling

        # Additional metrics
        profit_factor = 0.0
        if trades:
            winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
            if losing_trades:
                total_wins = sum(t.pnl for t in winning_trades)
                total_losses = abs(sum(t.pnl for t in losing_trades))
                profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        results: Dict[str, Any] = {
            'success': True,
            'symbol': symbol,
            'session_id': session_id,
            'starting_balance': self.config['starting_balance'],
            'ending_balance': equity,
            'total_return': total_return,
            'max_drawdown': float(drawdown_max),
            'num_trades': len(trades),
            'win_rate': float(win_rate),
            'sharpe_like': sharpe,
            'profit_factor': float(profit_factor),
            'total_commission': sum(t.commission for t in trades),
            'total_slippage': sum(t.slippage for t in trades),
            'trades': [
                {
                    'id': t.id,
                    'symbol': t.symbol,
                    'direction': t.direction,
                    'entry_time': t.entry_time.isoformat(),
                    'entry_price': t.entry_price,
                    'exit_time': None if t.exit_time is None else t.exit_time.isoformat(),
                    'exit_price': t.exit_price,
                    'pnl_pct': t.pnl,
                    'size_lots': t.size,
                    'commission': t.commission,
                    'slippage': t.slippage,
                }
                for t in trades
            ],
        }

        # Final broadcast (if supported)
        if self.websocket_broadcaster and hasattr(self.websocket_broadcaster, 'broadcast_performance_update'):
            try:
                await self.websocket_broadcaster.broadcast_performance_update({
                    'session_id': session_id,
                    'symbol': symbol,
                    'type': 'backtest_completed',
                    'status': 'completed',
                    'results': {k: v for k, v in results.items() if k != 'trades'},
                    'timestamp': datetime.now().isoformat(),
                })
            except Exception as e:
                logger.debug(f"WebSocket final backtest update failed (ignored): {e}")

        logger.info(
            f"Backtest done: {symbol} | ret={total_return:.2%} | MDD={drawdown_max:.2%} | "
            f"trades={len(trades)} | win={win_rate:.1%}"
        )
        return results
    
    def _calculate_unrealized_pnl(self, position: BacktestTrade, current_bar: pd.Series) -> float:
        """Calculate unrealized PnL for open position."""
        current_price = float(current_bar.close)
        if position.direction == 'LONG':
            return (current_price - position.entry_price) / position.entry_price
        else:
            return (position.entry_price - current_price) / position.entry_price