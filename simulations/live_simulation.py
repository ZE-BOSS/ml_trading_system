import asyncio
from typing import Iterable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger

from services.mt5_client import MT5Client, TradeSignal, MarketTick
from services.signal_generator import SignalGenerator, TradingSignal

class LiveSimulation:
    """Async live (or paper) trading loop using `SignalGenerator`.

    This runner ties together the real-time data source (`MT5Client` and/or
    `DataManager`) with the signal generation logic and optional execution.

    Design Goals
    ------------
    - **Non-blocking**: uses asyncio; can be started/stopped cleanly by the
      orchestrator (`PPOTradingSystem`).
    - **Production parity**: reuses the same `SignalGenerator` used for
      backtesting and batch generation, reducing train/serve drift.
    - **Defensive execution**: interacts with `MT5Client` via feature detection
      (`hasattr`) so the runner stays robust even if MT5 APIs evolve.

    Lifecycle
    ---------
    - `start(...)` spawns a long-lived task `_run_loop()`.
    - `_run_loop()` periodically fetches the latest bars/tick, generates a
      signal, optionally executes it on MT5 (demo/real per flag), and emits
      updates over WebSocket.
    - `stop()` signals graceful cancellation and awaits task shutdown.
    """

    def __init__(
        self,
        mt5_client: MT5Client,
        signal_generator: SignalGenerator,
        data_manager: Any,
        websocket_broadcaster: Optional[Any] = None,
        poll_interval_seconds: int = 30,
        timeframe: str = "H1",
    ) -> None:
        self.mt5_client = mt5_client
        self.signal_generator = signal_generator
        self.data_manager = data_manager
        self.websocket_broadcaster = websocket_broadcaster
        self.poll_interval_seconds = poll_interval_seconds
        self.timeframe = timeframe

        self._task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._session_id: Optional[str] = None
        self._symbols: List[str] = []
        self._demo_mode: bool = True

    async def start(self, symbols: Iterable[str], session_id: str, demo_mode: bool = True) -> None:
        """Start the live loop as a background task.

        Parameters
        ----------
        symbols : Iterable[str]
            Symbols to process each cycle.
        session_id : str
            Correlation id used for UI and API.
        demo_mode : bool
            If True, execution (if supported) will target a demo account.
        """
        if self._running:
            logger.warning("LiveSimulation already running; ignoring start().")
            return

        self._symbols = list(symbols)
        self._session_id = session_id
        self._demo_mode = demo_mode
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"LiveSimulation started | symbols={self._symbols} | demo={demo_mode}")

    async def stop(self) -> None:
        """Request the live loop to stop and await task completion."""
        if not self._running:
            return
        self._running = False
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=self.poll_interval_seconds + 5)
            except Exception:
                self._task.cancel()
        logger.info("LiveSimulation stopped")

    async def _run_loop(self) -> None:
        """Main periodic loop that fetches data, generates signals, and executes."""
        while self._running:
            cycle_started = datetime.now()
            try:
                await self._process_all_symbols()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Live loop cycle error: {e}")

            # Sleep the remaining time until next poll
            elapsed = (datetime.now() - cycle_started).total_seconds()
            sleep_for = max(1.0, self.poll_interval_seconds - elapsed)
            await asyncio.sleep(sleep_for)

    async def _process_all_symbols(self) -> None:
        for symbol in self._symbols:
            try:
                await self._process_symbol(symbol)
            except Exception as e:
                logger.error(f"Symbol processing error [{symbol}]: {e}")

    async def _process_symbol(self, symbol: str) -> None:
        """Fetch latest data for `symbol`, generate a signal, and optionally execute.

        Data Retrieval Strategy
        -----------------------
        1) Try DataManager cache for the last N bars (fast local path).
        2) Fallback to MT5 client to fetch fresh recent bars.
        3) Request a current tick (best-effort) to price entries precisely.
        """
        # 1) Try cached bars
        market_data = await self.data_manager.get_historical_data(symbol, self.timeframe)

        # 2) Fallback to MT5 for recent bars if cache is missing or shallow
        if market_data is None or market_data.empty or len(market_data) < 300:
            if hasattr(self.mt5_client, 'get_historical_data'):
                try:
                    # Pull last ~1000 bars as a rolling context (implementation-defined)
                    market_data = self.mt5_client.get_historical_data(symbol, self.timeframe)
                except Exception as e:
                    logger.debug(f"MT5 historical fetch failed for {symbol}: {e}")
                    market_data = pd.DataFrame()

        if market_data is None or market_data.empty:
            logger.debug(f"No market data for {symbol}; skipping this cycle.")
            return

        # 3) Current tick (best-effort)
        current_tick: Optional[MarketTick] = None
        if hasattr(self.mt5_client, 'get_market_tick'):
            try:
                current_tick = self.mt5_client.get_market_tick(symbol)
            except Exception as e:
                logger.debug(f"Tick fetch failed for {symbol}: {e}")

        # Generate a trading signal
        signal = self.signal_generator.generate_signal(
            symbol=symbol,
            market_data=market_data,
            current_tick=current_tick,
            account_info=self._safe_account_info(),
        )

        if not signal or signal.action == 'HOLD':
            await self._emit_signal_update(symbol, signal)
            return

        # Convert to MT5 order request
        mt5_sig: Optional[TradeSignal] = signal.to_mt5_signal()

        # Route to execution in demo/real depending on MT5 client capabilities
        executed = False
        exec_error: Optional[str] = None
        if mt5_sig and hasattr(self.mt5_client, 'execute_trade'):
            try:
                executed = self.mt5_client.execute_trade(mt5_sig, demo=self._demo_mode)
            except Exception as e:
                exec_error = str(e)
        elif mt5_sig and hasattr(self.mt5_client, 'place_order'):
            try:
                executed = self.mt5_client.place_order(mt5_sig, demo=self._demo_mode)
            except Exception as e:
                exec_error = str(e)

        await self._emit_signal_update(symbol, signal, executed=executed, error=exec_error)

    async def _emit_signal_update(
        self,
        symbol: str,
        signal: Optional[TradingSignal],
        executed: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """Send a compact update to WebSocket listeners, when supported."""
        if not self.websocket_broadcaster:
            return

        payload = {
            'type': 'trading_signal',
            'session_id': self._session_id,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'executed': executed,
            'error': error,
        }
        if signal:
            payload.update({
                'action': signal.action,
                'confidence': float(signal.confidence),
                'lot_size': float(signal.lot_size),
                'entry_price': float(signal.entry_price),
                'stop_loss': None if signal.stop_loss is None else float(signal.stop_loss),
                'take_profit': None if signal.take_profit is None else float(signal.take_profit),
                'expected_return': None if signal.expected_return is None else float(signal.expected_return),
                'risk_score': None if signal.risk_score is None else float(signal.risk_score),
            })
        else:
            payload.update({'action': 'HOLD'})

        # Try several known broadcaster method names for compatibility
        for method_name in (
            'broadcast_trading_signal',
            'broadcast',
            'broadcast_message',
        ):
            if hasattr(self.websocket_broadcaster, method_name):
                try:
                    await getattr(self.websocket_broadcaster, method_name)(payload)
                    break
                except Exception as e:
                    logger.debug(f"WebSocket emit via {method_name} failed (ignored): {e}")

    def _safe_account_info(self) -> Optional[Dict[str, Any]]:
        """Fetch account info from MT5 client if available; otherwise a stub.

        The `SignalGenerator` can improve sizing with balance/equity info. If the
        client doesn't expose it, we return a minimal placeholder to keep logic
        deterministic.
        """
        info: Dict[str, Any] = {}
        if hasattr(self.mt5_client, 'get_account_info'):
            try:
                raw = self.mt5_client.get_account_info()
                if isinstance(raw, dict):
                    info = raw
            except Exception as e:
                logger.debug(f"Account info fetch failed: {e}")
        if not info:
            info = {'balance': 10_000.0, 'equity': 10_000.0}
        return info
