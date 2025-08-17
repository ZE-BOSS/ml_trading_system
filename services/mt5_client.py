"""
MetaTrader 5 Client
===================

This module handles the connection and communication with MetaTrader 5
for live market data, historical data, and trade execution.

Features:
- MT5 connection management
- Real-time market data streaming
- Historical data retrieval
- Order execution and management
- Position monitoring
- Account information

Author: PPO Trading System
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio
import threading
from loguru import logger
import yaml


@dataclass
class MarketTick:
    """Market tick data structure."""
    symbol: str
    time: datetime
    bid: float
    ask: float
    last: float
    volume: int
    spread: float
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2.0


@dataclass
class TradeSignal:
    """Trading signal structure."""
    symbol: str
    action: str  # 'BUY', 'SELL', 'CLOSE'
    lot_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = "PPO_Agent"
    magic_number: int = 234000


@dataclass
class Position:
    """Position information structure."""
    ticket: int
    symbol: str
    type: str  # 'BUY' or 'SELL'
    volume: float
    open_price: float
    current_price: float
    profit: float
    swap: float
    commission: float
    time: datetime


class MT5Client:
    """
    MetaTrader 5 client for market data and trading operations.
    
    This client handles all interactions with the MT5 terminal including
    data retrieval, order execution, and position management.
    """
    
    def __init__(self, config_path: str = "config/broker_config.yaml"):
        """
        Initialize MT5 client.
        
        Args:
            config_path: Path to broker configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.mt5_config = self.config['mt5']
        self.symbols_config = {s['name']: s for s in self.mt5_config['symbols']}
        self.risk_config = self.config['risk_management']
        
        # Connection state
        self.is_connected = False
        self.account_info = None
        
        # Data streaming
        self.is_streaming = False
        self.stream_thread = None
        self.tick_callback = None
        self.last_ticks = {}
        
        # Position tracking
        self.positions = {}
        self.open_orders = {}
        
        logger.info("MT5 Client initialized")
    
    def connect(self) -> bool:
        """
        Connect to MetaTrader 5 terminal.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Initialize MT5 connection
            if not mt5.initialize(
                path=self.mt5_config.get('path', ""),
                login=self.mt5_config['login'],
                password=self.mt5_config['password'],
                server=self.mt5_config['server'],
                timeout=self.mt5_config['timeout']
            ):
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Get account info
            self.account_info = mt5.account_info()
            if self.account_info is None:
                logger.error("Failed to get account information")
                return False
            
            # Verify symbols are available
            for symbol_name in self.symbols_config.keys():
                symbol_info = mt5.symbol_info(symbol_name)
                if symbol_info is None:
                    logger.warning(f"Symbol {symbol_name} not found")
                else:
                    # Enable symbol in Market Watch
                    mt5.symbol_select(symbol_name, True)
            
            self.is_connected = True
            logger.info(f"Connected to MT5: Account {self.account_info.login}, "
                       f"Balance: {self.account_info.balance}, "
                       f"Server: {self.account_info.server}")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from MetaTrader 5."""
        try:
            self.stop_streaming()
            mt5.shutdown()
            self.is_connected = False
            logger.info("Disconnected from MT5")
        except Exception as e:
            logger.error(f"Error during MT5 disconnect: {e}")
    
    def get_historical_data(self, 
                           symbol: str,
                           timeframe: str = "H1",
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           count: int = 1000) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1)
            start_date: Start date for data
            end_date: End date for data
            count: Number of bars to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_connected:
            logger.error("Not connected to MT5")
            return pd.DataFrame()
        
        try:
            # Map timeframe strings to MT5 constants
            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Get data
            if start_date and end_date:
                rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            else:
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data retrieved for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Rename columns to standard format
            df.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            }, inplace=True)
            
            logger.info(f"Retrieved {len(df)} bars for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            return pd.DataFrame()
    
    def get_current_tick(self, symbol: str) -> Optional[MarketTick]:
        """
        Get current market tick for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current market tick or None
        """
        if not self.is_connected:
            return None
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return MarketTick(
                symbol=symbol,
                time=datetime.fromtimestamp(tick.time),
                bid=tick.bid,
                ask=tick.ask,
                last=tick.last,
                volume=tick.volume,
                spread=(tick.ask - tick.bid)
            )
            
        except Exception as e:
            logger.error(f"Error getting tick for {symbol}: {e}")
            return None
    
    def start_streaming(self, symbols: List[str], callback: callable) -> bool:
        """
        Start real-time data streaming.
        
        Args:
            symbols: List of symbols to stream
            callback: Callback function for tick data
            
        Returns:
            True if streaming started successfully
        """
        if not self.is_connected:
            logger.error("Not connected to MT5")
            return False
        
        if self.is_streaming:
            logger.warning("Streaming already active")
            return True
        
        self.tick_callback = callback
        self.is_streaming = True
        
        # Start streaming thread
        self.stream_thread = threading.Thread(
            target=self._streaming_worker,
            args=(symbols,),
            daemon=True
        )
        self.stream_thread.start()
        
        logger.info(f"Started streaming for symbols: {symbols}")
        return True
    
    def stop_streaming(self) -> None:
        """Stop real-time data streaming."""
        self.is_streaming = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5.0)
        logger.info("Stopped market data streaming")
    
    def _streaming_worker(self, symbols: List[str]) -> None:
        """Worker thread for real-time data streaming."""
        logger.info("Data streaming worker started")
        
        while self.is_streaming:
            try:
                for symbol in symbols:
                    tick = self.get_current_tick(symbol)
                    if tick and self.tick_callback:
                        # Check if tick is new
                        last_tick = self.last_ticks.get(symbol)
                        if last_tick is None or tick.time > last_tick.time:
                            self.last_ticks[symbol] = tick
                            self.tick_callback(tick)
                
                time.sleep(0.1)  # 100ms polling interval
                
            except Exception as e:
                logger.error(f"Error in streaming worker: {e}")
                time.sleep(1)
        
        logger.info("Data streaming worker stopped")
    
    def execute_trade(self, signal: TradeSignal) -> Optional[Dict[str, Any]]:
        """
        Execute trading signal.
        
        Args:
            signal: Trading signal to execute
            
        Returns:
            Trade execution result or None if failed
        """
        if not self.is_connected:
            logger.error("Not connected to MT5")
            return None
        
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(signal.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {signal.symbol} not found")
                return None
            
            # Check risk limits
            if not self._check_risk_limits(signal):
                logger.warning(f"Trade rejected by risk management: {signal}")
                return None
            
            # Prepare trade request
            if signal.action == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = symbol_info.ask
            elif signal.action == "SELL":
                order_type = mt5.ORDER_TYPE_SELL
                price = symbol_info.bid
            else:
                logger.error(f"Unknown action: {signal.action}")
                return None
            
            # Build request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": signal.lot_size,
                "type": order_type,
                "price": price,
                "deviation": self.mt5_config['trading']['deviation'],
                "magic": signal.magic_number,
                "comment": signal.comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC
            }
            
            # Add stop loss and take profit if specified
            if signal.stop_loss:
                request["sl"] = signal.stop_loss
            if signal.take_profit:
                request["tp"] = signal.take_profit
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                logger.error(f"Order send failed: {mt5.last_error()}")
                return None
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Trade executed: {signal.action} {signal.lot_size} {signal.symbol} "
                           f"at {price}, ticket: {result.order}")
                
                return {
                    "success": True,
                    "ticket": result.order,
                    "price": result.price,
                    "volume": result.volume,
                    "symbol": signal.symbol,
                    "action": signal.action
                }
            else:
                logger.error(f"Trade failed: {result.retcode} - {result.comment}")
                return {
                    "success": False,
                    "error_code": result.retcode,
                    "error_message": result.comment
                }
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def close_position(self, ticket: int) -> bool:
        """
        Close position by ticket.
        
        Args:
            ticket: Position ticket number
            
        Returns:
            True if position closed successfully
        """
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.error(f"Position {ticket} not found")
                return False
            
            pos = position[0]
            
            # Prepare close request
            if pos.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(pos.symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(pos.symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": self.mt5_config['trading']['deviation'],
                "magic": pos.magic,
                "comment": f"Close position {ticket}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Position {ticket} closed successfully")
                return True
            else:
                logger.error(f"Failed to close position {ticket}: {result.comment if result else mt5.last_error()}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False
    
    def get_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of open positions
        """
        if not self.is_connected:
            return []
        
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                current_tick = self.get_current_tick(pos.symbol)
                current_price = current_tick.mid_price if current_tick else pos.price_open
                
                position = Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    type="BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                    volume=pos.volume,
                    open_price=pos.price_open,
                    current_price=current_price,
                    profit=pos.profit,
                    swap=pos.swap,
                    commission=pos.commission,
                    time=datetime.fromtimestamp(pos.time)
                )
                result.append(position)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current account information.
        
        Returns:
            Account information dictionary
        """
        if not self.is_connected:
            return None
        
        try:
            account = mt5.account_info()
            if account is None:
                return None
            
            return {
                "login": account.login,
                "balance": account.balance,
                "equity": account.equity,
                "profit": account.profit,
                "margin": account.margin,
                "margin_free": account.margin_free,
                "margin_level": account.margin_level,
                "currency": account.currency,
                "server": account.server,
                "leverage": account.leverage
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def _check_risk_limits(self, signal: TradeSignal) -> bool:
        """
        Check if trade signal passes risk management rules.
        
        Args:
            signal: Trading signal to check
            
        Returns:
            True if trade is allowed
        """
        try:
            # Get account info
            account = self.get_account_info()
            if not account:
                return False
            
            # Check maximum lot size
            symbol_config = self.symbols_config.get(signal.symbol, {})
            max_lot = symbol_config.get('max_lot', 1.0)
            if signal.lot_size > max_lot:
                logger.warning(f"Lot size {signal.lot_size} exceeds maximum {max_lot}")
                return False
            
            # Check maximum positions
            positions = self.get_positions()
            if len(positions) >= self.risk_config['max_open_positions']:
                logger.warning(f"Maximum positions ({self.risk_config['max_open_positions']}) reached")
                return False
            
            # Check daily loss limit
            daily_profit = sum(pos.profit for pos in positions)
            if daily_profit <= -self.risk_config['max_daily_loss']:
                logger.warning(f"Daily loss limit reached: {daily_profit}")
                return False
            
            # Check equity protection
            equity_threshold = account['balance'] * (1 - self.risk_config['equity_protection_percentage'])
            if account['equity'] < equity_threshold:
                logger.warning(f"Equity protection triggered: {account['equity']} < {equity_threshold}")
                return False
            
            # Check spread limits
            current_tick = self.get_current_tick(signal.symbol)
            if current_tick:
                spread_limit = symbol_config.get('spread_limit', 100)  # In points
                symbol_info = mt5.symbol_info(signal.symbol)
                if symbol_info:
                    spread_points = current_tick.spread / symbol_info.point
                    if spread_points > spread_limit:
                        logger.warning(f"Spread {spread_points} exceeds limit {spread_limit}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.
        
        Returns:
            True if market is open
        """
        try:
            now = datetime.now()
            
            # Check if it's weekend
            if self.config.get('market_hours', {}).get('exclude_weekends', True):
                if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    return False
            
            # Additional market hours checks can be added here
            return True
            
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False