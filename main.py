"""
Main Application Entry Point
============================

This is the main entry point for the PPO Trading System. It orchestrates
all components including training, live trading, WebSocket broadcasting,
and API services.

Usage:
    python main.py --mode train --symbol XAUUSD
    python main.py --mode live --demo
    python main.py --mode backtest --config custom_config.yaml

Author: PPO Trading System
"""

import argparse
import asyncio
import sys
import os
import signal
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from envs.trading_env import TradingEnvironment
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import yaml
import uuid
from loguru import logger
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ppo_agent import PPOTradingAgent
from services.mt5_client import MT5Client
from services.signal_generator import SignalGenerator
from services.websocket_server import WebSocketBroadcaster
from backend.data_manager import DataManager
from backend.api import TradingSystemAPI
from simulations.backtest_runner import BacktestRunner
from simulations.live_simulation import LiveSimulation


class PPOTradingSystem:
    """
    Main PPO Trading System orchestrator.
    
    This class manages all system components and coordinates their operation
    for training, backtesting, and live trading scenarios.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize the trading system.
        
        Args:
            config_path: Path to main configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize system components
        self.data_manager = DataManager()
        self.websocket_broadcaster = WebSocketBroadcaster()
        self.ppo_agent = None
        self.mt5_client = None
        self.signal_generator = None
        self.api_server = None
        self.backtest_runner = None
        self.live_simulation = None
        
        # System state
        self.is_running = False
        self.current_mode = None
        self.current_session_id = None
        
        # Setup logging
        self._setup_logging()
        
        logger.info("PPO Trading System initialized")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.get('app', {})
        log_level = log_config.get('log_level', 'INFO')
        
        # Configure loguru
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Add file logging
        logs_path = self.config.get('storage', {}).get('logs_path', './logs')
        os.makedirs(logs_path, exist_ok=True)
        
        logger.add(
            f"{logs_path}/trading_system.log",
            rotation="1 day",
            retention="30 days",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )
    
    async def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing PPO Trading System...")
            
            # Initialize data manager
            await self.data_manager.initialize()
            
            # Initialize WebSocket broadcaster
            await self.websocket_broadcaster.initialize()
            
            # Initialize PPO agent
            self.ppo_agent = PPOTradingAgent(
                websocket_broadcaster=self.websocket_broadcaster
            )
            
            # Initialize MT5 client
            self.mt5_client = MT5Client()
            
            # Initialize signal generator
            self.signal_generator = SignalGenerator()
            
            # Initialize backtest runner
            self.backtest_runner = BacktestRunner(
                data_manager=self.data_manager,
                websocket_broadcaster=self.websocket_broadcaster
            )
            
            # Initialize live simulation
            self.live_simulation = LiveSimulation(
                mt5_client=self.mt5_client,
                signal_generator=self.signal_generator,
                data_manager=self.data_manager,
                websocket_broadcaster=self.websocket_broadcaster
            )
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    
    def _normalize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure dataframe has lowercase OHLCV columns expected by the env."""
        if df is None or df.empty:
            return pd.DataFrame()

        rename_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'Adj Close': 'close',  # if coming from yfinance/others
            'Volume': 'volume', 'TickVolume': 'volume', 'tick_volume': 'volume',
            'BidOpen': 'open', 'BidHigh': 'high', 'BidLow': 'low', 'BidClose': 'close'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # If index is a column named 'time' or 'datetime', set it as index
        for tcol in ('time', 'datetime', 'date'):
            if tcol in df.columns:
                try:
                    df[tcol] = pd.to_datetime(df[tcol])
                    df = df.set_index(tcol)
                except Exception:
                    pass
                break

        # Sort and drop duplicates by index if index is datetime-like
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]

        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Data must contain columns: {required}")

        # Return only required columns to keep env happy
        return df[required]
    

    async def run_training(self, 
                          symbol: str,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          model_name: Optional[str] = None,
                          broadcast: bool = True) -> Dict[str, Any]:
        """
        Run model training.
        
        Args:
            symbol: Trading symbol
            start_date: Training data start date
            end_date: Training data end date
            model_name: Name for the trained model
            broadcast: Whether to broadcast training updates
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training for {symbol}")
        self.current_mode = 'training'
        self.current_session_id = str(uuid.uuid4())
        
        try:
            # Prepare training data
            if start_date and end_date:
                start_dt = datetime.fromisoformat(start_date)
                end_dt = datetime.fromisoformat(end_date)
            else:
                # Default to last 2 years
                end_dt = datetime.now()
                start_dt = end_dt - timedelta(days=730)
            
            # Get historical data (in real implementation, this would fetch from data provider)
            training_data = await self._get_or_download_data(symbol, start_dt, end_dt)
            
            if training_data.empty:
                raise ValueError(f"No training data available for {symbol}")
            
            logger.info(f"Training data prepared: {len(training_data)} samples")
            
            # Split data for training and validation
            split_point = int(len(training_data) * 0.8)
            train_data = training_data.iloc[:split_point]
            eval_data = training_data.iloc[split_point:]
            
            # Broadcast training start
            if broadcast:
                await self.websocket_broadcaster.broadcast_training_update({
                    'session_id': self.current_session_id,
                    'status': 'started',
                    'symbol': symbol,
                    'data_points': len(train_data),
                    'message': f'Training started for {symbol}'
                })
            
            # Run training
            results = self.ppo_agent.train(
                train_data=train_data,
                eval_data=eval_data,
                session_id=self.current_session_id
            )
            
            # Save training session to database
            await self.data_manager.save_training_session(self.current_session_id, {
                'symbol': symbol,
                'start_date': start_dt.isoformat(),
                'end_date': end_dt.isoformat(),
                'success': results.get('success', False),
                **results
            })
            
            # Broadcast training completion
            if broadcast:
                await self.websocket_broadcaster.broadcast_training_update({
                    'session_id': self.current_session_id,
                    'status': 'completed' if results.get('success') else 'failed',
                    'results': results,
                    'message': f'Training {"completed" if results.get("success") else "failed"} for {symbol}'
                })
            
            logger.info(f"Training completed for {symbol}: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
            # Broadcast training failure
            if broadcast:
                await self.websocket_broadcaster.broadcast_training_update({
                    'session_id': self.current_session_id,
                    'status': 'failed',
                    'error': str(e),
                    'message': f'Training failed for {symbol}: {e}'
                })
            
            return {'success': False, 'error': str(e)}
    
    async def run_backtest(self, 
                          symbol: str,
                          model_path: str,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run backtesting with trained model.
        
        Args:
            symbol: Trading symbol
            model_path: Path to trained model
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting backtest for {symbol} with model {model_path}")
        self.current_mode = 'backtest'
        self.current_session_id = str(uuid.uuid4())
        
        try:
            # Get backtest data
            if start_date and end_date:
                start_dt = datetime.fromisoformat(start_date)
                end_dt = datetime.fromisoformat(end_date)
            else:
                # Default to last 6 months
                end_dt = datetime.now()
                start_dt = end_dt - timedelta(days=180)
            
            backtest_data = await self._get_or_download_data(symbol, start_dt, end_dt)

            env = TradingEnvironment(data=backtest_data, config_path="config/model_config.yaml")
            env = Monitor(env)
            env = DummyVecEnv([lambda: env])

            # Load model
            if not self.ppo_agent.load_model(model_path, env):
                raise ValueError(f"Failed to load model from {model_path}")
            
            # Run backtest
            results = await self.backtest_runner.run_backtest(
                symbol=symbol,
                data=backtest_data,
                model=self.ppo_agent.model,
                session_id=self.current_session_id
            )
            
            logger.info(f"Backtest completed for {symbol}: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def run_live_trading(self, 
                              symbols: List[str],
                              model_path: str,
                              demo_mode: bool = True) -> None:
        """
        Run live trading with trained model.
        
        Args:
            symbols: List of trading symbols
            model_path: Path to trained model
            demo_mode: Whether to use demo account
        """
        logger.info(f"Starting live trading for {symbols} (demo: {demo_mode})")
        self.current_mode = 'live_trading'
        self.current_session_id = str(uuid.uuid4())
        
        try:
            # Load model
            if not self.signal_generator.load_model(model_path):
                raise ValueError(f"Failed to load model from {model_path}")
            
            # Connect to MT5
            if not self.mt5_client.connect():
                raise ConnectionError("Failed to connect to MetaTrader 5")
            
            # Start live simulation
            await self.live_simulation.start(
                symbols=symbols,
                session_id=self.current_session_id,
                demo_mode=demo_mode
            )
            
            logger.info("Live trading started successfully")
            
            # Keep running until stopped
            while self.is_running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Live trading failed: {e}")
            await self.websocket_broadcaster.broadcast_system_alert({
                'level': 'error',
                'message': f'Live trading failed: {e}',
                'timestamp': datetime.now().isoformat()
            })
        finally:
            # Cleanup
            if self.live_simulation:
                await self.live_simulation.stop()
            if self.mt5_client:
                self.mt5_client.disconnect()
    
    async def start_api_server(self) -> None:
        """Start the REST API server."""
        try:
            self.api_server = TradingSystemAPI()
            
            # Run API server (this would typically be done with uvicorn in production)
            logger.info("API server would be started here with uvicorn")
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
    
    async def start_websocket_server(self) -> None:
        """Start the WebSocket server."""
        try:
            # Start WebSocket server in background
            asyncio.create_task(self.websocket_broadcaster.start_server())
            logger.info("WebSocket server started")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
    
    async def _get_or_download_data(self, 
                                   symbol: str, 
                                   start_date: datetime,
                                   end_date: datetime) -> pd.DataFrame:
        """
        Get or download market data for the specified period.
        
        This is a placeholder implementation. In a real system, this would:
        1. Check if data exists in cache/database
        2. Download missing data from data provider
        3. Store downloaded data for future use
        """
        # Try to get from cache first
        data = await self.data_manager.get_historical_data(symbol, "H1")
        
        if not data.empty:
            # Filter by date range
            if hasattr(data.index, 'to_pydatetime'):
                mask = (data.index >= start_date) & (data.index <= end_date)
                data = data.loc[mask]
            
            if len(data) > 100:  # Minimum data requirement
                logger.info(f"Using cached data for {symbol}: {len(data)} samples")
                return data
        
        # If no cached data or insufficient, try to get from MT5
        if self.mt5_client:
            try:
                if not self.mt5_client.is_connected:
                    self.mt5_client.connect()
                
                data = self.mt5_client.get_historical_data(
                    symbol=symbol,
                    timeframe="H1",
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not data.empty:
                    # Cache the data
                    await self.data_manager.save_historical_data(symbol, data, "H1")
                    logger.info(f"Downloaded and cached data for {symbol}: {len(data)} samples")
                    return data
                    
            except Exception as e:
                logger.warning(f"Failed to download data from MT5: {e}")
        
        # Return empty DataFrame if no data available
        logger.warning(f"No data available for {symbol}")
        return pd.DataFrame()
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the system."""
        logger.info("Shutting down PPO Trading System...")
        
        self.is_running = False
        
        try:
            # Stop live simulation
            if self.live_simulation:
                await self.live_simulation.stop()
            
            # Disconnect MT5
            if self.mt5_client:
                self.mt5_client.disconnect()
            
            # Close data manager
            if self.data_manager:
                await self.data_manager.close()
            
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def run(self, mode: str, **kwargs) -> None:
        """
        Main run method that starts the system in specified mode.
        
        Args:
            mode: Operation mode ('train', 'backtest', 'live', 'api')
            **kwargs: Additional arguments for the specific mode
        """
        self.is_running = True
        self._setup_signal_handlers()
        
        try:
            # Initialize system
            if not await self.initialize():
                raise RuntimeError("System initialization failed")
            
            # Start WebSocket server
            await self.start_websocket_server()
            
            # Run based on mode
            if mode == 'train':
                results = await self.run_training(
                    symbol=kwargs.get('symbol', 'XAUUSDm'),
                    start_date=kwargs.get('start_date'),
                    end_date=kwargs.get('end_date'),
                    model_name=kwargs.get('model_name'),
                    broadcast=kwargs.get('broadcast', True)
                )
                print(f"Training results: {results}")
                
            elif mode == 'backtest':
                if not kwargs.get('model_path'):
                    raise ValueError("Model path required for backtesting")
                
                results = await self.run_backtest(
                    symbol=kwargs.get('symbol', 'XAUUSD'),
                    model_path=kwargs['model_path'],
                    start_date=kwargs.get('start_date'),
                    end_date=kwargs.get('end_date')
                )
                print(f"Backtest results: {results}")
                
            elif mode == 'live':
                if not kwargs.get('model_path'):
                    raise ValueError("Model path required for live trading")
                
                await self.run_live_trading(
                    symbols=kwargs.get('symbols', ['XAUUSD']),
                    model_path=kwargs['model_path'],
                    demo_mode=kwargs.get('demo', True)
                )
                
            elif mode == 'api':
                await self.start_api_server()
                
                # Keep running until stopped
                while self.is_running:
                    await asyncio.sleep(1)
                    
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in main run loop: {e}")
            raise
        finally:
            await self.shutdown()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PPO Trading System")
    
    parser.add_argument('--mode', 
                       choices=['train', 'backtest', 'live', 'api'],
                       required=True,
                       help='Operation mode')
    
    parser.add_argument('--symbol', 
                       default='XAUUSDm',
                       help='Trading symbol (default: XAUUSD)')
    
    parser.add_argument('--symbols',
                       nargs='+',
                       default=['XAUUSDm'],
                       help='Multiple trading symbols for live trading')
    
    parser.add_argument('--model-path',
                       help='Path to trained model file')
    
    parser.add_argument('--start-date',
                       help='Start date for data (YYYY-MM-DD)')
    
    parser.add_argument('--end-date',
                       help='End date for data (YYYY-MM-DD)')
    
    parser.add_argument('--demo',
                       action='store_true',
                       help='Use demo trading account')
    
    parser.add_argument('--config',
                       default='config/settings.yaml',
                       help='Configuration file path')
    
    parser.add_argument('--broadcast',
                       action='store_true',
                       default=True,
                       help='Enable WebSocket broadcasting')
    
    parser.add_argument('--model-name',
                       help='Name for the trained model')
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Create and run trading system
    system = PPOTradingSystem(config_path=args.config)
    
    await system.run(
        mode=args.mode,
        symbol=args.symbol,
        symbols=args.symbols,
        model_path=args.model_path,
        start_date=args.start_date,
        end_date=args.end_date,
        demo=args.demo,
        broadcast=args.broadcast,
        model_name=args.model_name
    )


if __name__ == "__main__":
    # Handle import for PANDAS-TA
    try:
        import pandas_ta
    except ImportError:
        print("Warning: PANDAS-TA not installed. Technical indicators may not work properly.")
        print("Install with: pip install PANDAS-TA")
    
    # Run the main application
    asyncio.run(main())