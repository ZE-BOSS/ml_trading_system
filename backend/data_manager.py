"""
Data Manager
============

This module handles all database operations, data caching, and data persistence
for the PPO Trading System. It manages both PostgreSQL and Redis connections.

Features:
- PostgreSQL database operations
- Redis caching layer
- Training session management
- Performance metrics storage
- Historical data management
- Real-time data caching

Author: PPO Trading System
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncpg
import redis.asyncio as redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from loguru import logger
import yaml


class DataManager:
    """
    Manages all data operations for the PPO Trading System.
    
    This class handles database connections, data storage, retrieval,
    and caching for optimal performance.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize the data manager.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.db_config = self.config['database']
        
        # Database connections
        self.postgres_pool = None
        self.redis_client = None
        self.engine = None
        self.session_factory = None
        
        # Connection status
        self._postgres_connected = False
        self._redis_connected = False
        
        logger.info("Data manager initialized")
    
    async def initialize(self) -> None:
        """Initialize database connections."""
        await self._connect_postgres()
        await self._connect_redis()
        await self._create_tables()
        
        logger.info("Data manager initialization completed")
    
    async def _connect_postgres(self) -> None:
        """Connect to PostgreSQL database."""
        try:
            pg_config = self.db_config['postgres']
            
            # Create connection pool
            dsn = f"postgresql://{pg_config['username']}:{pg_config['password']}@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
            
            self.postgres_pool = await asyncpg.create_pool(
                dsn,
                min_size=1,
                max_size=pg_config['pool_size']
            )
            
            # Create SQLAlchemy engine for compatibility
            self.engine = create_engine(dsn)
            self.session_factory = sessionmaker(bind=self.engine)
            
            self._postgres_connected = True
            logger.info("Connected to PostgreSQL database")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            self._postgres_connected = False
    
    async def _connect_redis(self) -> None:
        """Connect to Redis cache."""
        try:
            redis_config = self.db_config['redis']
            
            self.redis_client = redis.Redis(
                host=redis_config['host'],
                port=redis_config['port'],
                db=redis_config['db'],
                password=redis_config.get('password'),
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            self._redis_connected = True
            logger.info("Connected to Redis cache")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._redis_connected = False
    
    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        if not self._postgres_connected:
            return
        
        try:
            async with self.postgres_pool.acquire() as connection:
                # Training sessions table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS training_sessions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        session_id VARCHAR(255) UNIQUE NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        start_time TIMESTAMP DEFAULT NOW(),
                        end_time TIMESTAMP,
                        status VARCHAR(50) DEFAULT 'running',
                        model_path TEXT,
                        config_used JSONB,
                        final_performance JSONB,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                # Training metrics table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS training_metrics (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        session_id VARCHAR(255) NOT NULL REFERENCES training_sessions(session_id),
                        timestep INTEGER NOT NULL,
                        episode INTEGER,
                        reward FLOAT,
                        portfolio_value FLOAT,
                        total_return FLOAT,
                        win_rate FLOAT,
                        max_drawdown FLOAT,
                        sharpe_ratio FLOAT,
                        total_trades INTEGER,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        additional_metrics JSONB
                    );
                """)
                
                # Signals table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        signal_id VARCHAR(255) UNIQUE NOT NULL,
                        session_id VARCHAR(255),
                        symbol VARCHAR(20) NOT NULL,
                        action VARCHAR(10) NOT NULL,
                        confidence FLOAT NOT NULL,
                        lot_size FLOAT NOT NULL,
                        entry_price FLOAT NOT NULL,
                        stop_loss FLOAT,
                        take_profit FLOAT,
                        expected_return FLOAT,
                        risk_score FLOAT,
                        model_version VARCHAR(100),
                        features_used JSONB,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        executed BOOLEAN DEFAULT FALSE,
                        execution_result JSONB
                    );
                """)
                
                # Live trades table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS live_trades (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        signal_id VARCHAR(255) REFERENCES signals(signal_id),
                        ticket VARCHAR(50),
                        symbol VARCHAR(20) NOT NULL,
                        action VARCHAR(10) NOT NULL,
                        lot_size FLOAT NOT NULL,
                        entry_price FLOAT NOT NULL,
                        exit_price FLOAT,
                        stop_loss FLOAT,
                        take_profit FLOAT,
                        entry_time TIMESTAMP NOT NULL,
                        exit_time TIMESTAMP,
                        profit FLOAT,
                        commission FLOAT,
                        swap FLOAT,
                        status VARCHAR(20) DEFAULT 'open',
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                # Performance summary table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS performance_summary (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        session_id VARCHAR(255) NOT NULL,
                        period_start TIMESTAMP NOT NULL,
                        period_end TIMESTAMP NOT NULL,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        win_rate FLOAT DEFAULT 0,
                        total_profit FLOAT DEFAULT 0,
                        max_drawdown FLOAT DEFAULT 0,
                        sharpe_ratio FLOAT DEFAULT 0,
                        profit_factor FLOAT DEFAULT 0,
                        avg_trade_duration INTERVAL,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                # Create indexes for better performance
                await connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_training_metrics_session_timestep 
                    ON training_metrics(session_id, timestep);
                """)
                
                await connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signals_timestamp 
                    ON signals(timestamp DESC);
                """)
                
                await connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_live_trades_symbol_time 
                    ON live_trades(symbol, entry_time DESC);
                """)
            
            logger.info("Database tables created/verified successfully")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
    
    def is_connected(self) -> bool:
        """Check if database connections are active."""
        return self._postgres_connected
    
    async def close(self) -> None:
        """Close all database connections."""
        try:
            if self.postgres_pool:
                await self.postgres_pool.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    # Training session management
    async def save_training_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Save training session information."""
        if not self._postgres_connected:
            return False
        
        try:
            async with self.postgres_pool.acquire() as connection:
                await connection.execute("""
                    INSERT INTO training_sessions 
                    (session_id, symbol, model_path, final_performance, status, end_time)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (session_id) DO UPDATE SET
                        model_path = EXCLUDED.model_path,
                        final_performance = EXCLUDED.final_performance,
                        status = EXCLUDED.status,
                        end_time = EXCLUDED.end_time
                """, 
                    session_id,
                    session_data.get('symbol', 'unknown'),
                    session_data.get('model_path'),
                    json.dumps(session_data.get('final_performance', {})),
                    'completed' if session_data.get('success') else 'failed',
                    datetime.now()
                )
            
            logger.info(f"Training session {session_id} saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving training session: {e}")
            return False
    
    async def save_training_metrics(self, 
                                  session_id: str,
                                  metrics: Dict[str, Any]) -> bool:
        """Save training metrics for a session."""
        if not self._postgres_connected:
            return False
        
        try:
            async with self.postgres_pool.acquire() as connection:
                await connection.execute("""
                    INSERT INTO training_metrics 
                    (session_id, timestep, episode, reward, portfolio_value, 
                     total_return, win_rate, max_drawdown, sharpe_ratio, total_trades, additional_metrics)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                    session_id,
                    metrics.get('timestep', 0),
                    metrics.get('episode', 0),
                    metrics.get('reward', 0.0),
                    metrics.get('portfolio_value', 0.0),
                    metrics.get('total_return', 0.0),
                    metrics.get('win_rate', 0.0),
                    metrics.get('max_drawdown', 0.0),
                    metrics.get('sharpe_ratio', 0.0),
                    metrics.get('total_trades', 0),
                    json.dumps({k: v for k, v in metrics.items() if k not in [
                        'timestep', 'episode', 'reward', 'portfolio_value',
                        'total_return', 'win_rate', 'max_drawdown', 'sharpe_ratio', 'total_trades'
                    ]})
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving training metrics: {e}")
            return False
    
    async def get_training_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get training metrics for a session."""
        if not self._postgres_connected:
            return None
        
        try:
            async with self.postgres_pool.acquire() as connection:
                # Get session info
                session_row = await connection.fetchrow("""
                    SELECT * FROM training_sessions WHERE session_id = $1
                """, session_id)
                
                if not session_row:
                    return None
                
                # Get metrics
                metrics_rows = await connection.fetch("""
                    SELECT * FROM training_metrics 
                    WHERE session_id = $1 
                    ORDER BY timestep
                """, session_id)
                
                # Process metrics
                metrics_data = []
                for row in metrics_rows:
                    metric = dict(row)
                    if metric['additional_metrics']:
                        metric.update(metric['additional_metrics'])
                    metrics_data.append(metric)
                
                return {
                    'session_info': dict(session_row),
                    'metrics': metrics_data,
                    'summary': self._calculate_metrics_summary(metrics_data)
                }
                
        except Exception as e:
            logger.error(f"Error getting training metrics: {e}")
            return None
    
    def _calculate_metrics_summary(self, metrics_data: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from metrics data."""
        if not metrics_data:
            return {}
        
        df = pd.DataFrame(metrics_data)
        
        try:
            summary = {
                'total_timesteps': len(df),
                'final_portfolio_value': df['portfolio_value'].iloc[-1] if 'portfolio_value' in df else None,
                'final_total_return': df['total_return'].iloc[-1] if 'total_return' in df else None,
                'final_win_rate': df['win_rate'].iloc[-1] if 'win_rate' in df else None,
                'max_drawdown': df['max_drawdown'].max() if 'max_drawdown' in df else None,
                'best_sharpe_ratio': df['sharpe_ratio'].max() if 'sharpe_ratio' in df else None,
                'total_trades': df['total_trades'].iloc[-1] if 'total_trades' in df else None,
                'avg_reward': df['reward'].mean() if 'reward' in df else None,
                'reward_std': df['reward'].std() if 'reward' in df else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating metrics summary: {e}")
            return {}
    
    # Signal management
    async def save_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Save a trading signal."""
        if not self._postgres_connected:
            return False
        
        try:
            async with self.postgres_pool.acquire() as connection:
                await connection.execute("""
                    INSERT INTO signals 
                    (signal_id, session_id, symbol, action, confidence, lot_size,
                     entry_price, stop_loss, take_profit, expected_return, risk_score,
                     model_version, features_used)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                    signal_data['id'],
                    signal_data.get('session_id'),
                    signal_data['symbol'],
                    signal_data['action'],
                    signal_data['confidence'],
                    signal_data['lot_size'],
                    signal_data['entry_price'],
                    signal_data.get('stop_loss'),
                    signal_data.get('take_profit'),
                    signal_data.get('expected_return'),
                    signal_data.get('risk_score'),
                    signal_data.get('model_version'),
                    json.dumps(signal_data.get('features_used', []))
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            return False
    
    async def update_signal_execution(self, signal_id: str, execution_result: Dict[str, Any]) -> bool:
        """Update signal with execution result."""
        if not self._postgres_connected:
            return False
        
        try:
            async with self.postgres_pool.acquire() as connection:
                await connection.execute("""
                    UPDATE signals 
                    SET executed = TRUE, execution_result = $2
                    WHERE signal_id = $1
                """, signal_id, json.dumps(execution_result))
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating signal execution: {e}")
            return False
    
    # Historical data management
    async def save_historical_data(self, 
                                 symbol: str, 
                                 data: pd.DataFrame,
                                 timeframe: str = "H1") -> bool:
        """Save historical market data to cache."""
        if not self._redis_connected:
            return False
        
        try:
            # Serialize DataFrame to JSON
            data_json = data.to_json(orient='records', date_format='iso')
            
            # Store in Redis with expiration
            cache_key = f"historical:{symbol}:{timeframe}"
            expire_time = self.db_config['redis'].get('expire_time', 3600)
            
            await self.redis_client.setex(
                cache_key, 
                expire_time, 
                data_json
            )
            
            # Store metadata
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_time': data.index[0].isoformat() if hasattr(data.index[0], 'isoformat') else str(data.index[0]),
                'end_time': data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1]),
                'rows': len(data),
                'cached_at': datetime.now().isoformat()
            }
            
            await self.redis_client.setex(
                f"{cache_key}:metadata",
                expire_time,
                json.dumps(metadata)
            )
            
            logger.info(f"Cached historical data for {symbol} ({len(data)} rows)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")
            return False
    
    async def get_historical_data(self, 
                                symbol: str, 
                                timeframe: str = "H1",
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> pd.DataFrame:
        """Get historical market data from cache or database."""
        cache_key = f"historical:{symbol}:{timeframe}"
        
        # Try to get from Redis cache first
        if self._redis_connected:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    df = pd.DataFrame(json.loads(cached_data))
                    if not df.empty:
                        # Convert timestamp column back to datetime index if exists
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df.set_index('timestamp', inplace=True)
                        elif 'time' in df.columns:
                            df['time'] = pd.to_datetime(df['time'])
                            df.set_index('time', inplace=True)
                        
                        logger.info(f"Retrieved {len(df)} rows from cache for {symbol}")
                        return df
                        
            except Exception as e:
                logger.warning(f"Error getting cached data: {e}")
        
        # If not in cache, return empty DataFrame
        # In a real implementation, you would fetch from external data source here
        logger.warning(f"No cached data found for {symbol}:{timeframe}")
        return pd.DataFrame()
    
    # Live data caching
    async def cache_tick_data(self, symbol: str, tick_data: Dict[str, Any]) -> bool:
        """Cache real-time tick data."""
        if not self._redis_connected:
            return False
        
        try:
            cache_key = f"live_tick:{symbol}"
            await self.redis_client.setex(
                cache_key,
                300,  # 5 minutes expiration
                json.dumps(tick_data)
            )
            
            # Also maintain a list of recent ticks
            tick_list_key = f"tick_history:{symbol}"
            await self.redis_client.lpush(tick_list_key, json.dumps(tick_data))
            await self.redis_client.ltrim(tick_list_key, 0, 999)  # Keep last 1000 ticks
            await self.redis_client.expire(tick_list_key, 3600)  # 1 hour expiration
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching tick data: {e}")
            return False
    
    async def get_recent_ticks(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent tick data for symbol."""
        if not self._redis_connected:
            return []
        
        try:
            tick_list_key = f"tick_history:{symbol}"
            tick_data = await self.redis_client.lrange(tick_list_key, 0, limit - 1)
            
            return [json.loads(tick) for tick in tick_data]
            
        except Exception as e:
            logger.error(f"Error getting recent ticks: {e}")
            return []
    
    # Model performance tracking
    async def get_model_performance(self, model_name: str) -> Optional[Dict[str, float]]:
        """Get performance metrics for a model."""
        if not self._postgres_connected:
            return None
        
        try:
            async with self.postgres_pool.acquire() as connection:
                # Find the most recent session for this model
                row = await connection.fetchrow("""
                    SELECT final_performance 
                    FROM training_sessions 
                    WHERE model_path LIKE $1 
                    ORDER BY end_time DESC 
                    LIMIT 1
                """, f"%{model_name}%")
                
                if row and row['final_performance']:
                    return row['final_performance']
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return None
    
    # Database maintenance
    async def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up old data to prevent database bloat."""
        if not self._postgres_connected:
            return
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            async with self.postgres_pool.acquire() as connection:
                # Clean up old training metrics
                deleted_metrics = await connection.execute("""
                    DELETE FROM training_metrics 
                    WHERE timestamp < $1
                """, cutoff_date)
                
                # Clean up old signals
                deleted_signals = await connection.execute("""
                    DELETE FROM signals 
                    WHERE timestamp < $1
                """, cutoff_date)
                
                # Clean up completed training sessions older than cutoff
                deleted_sessions = await connection.execute("""
                    DELETE FROM training_sessions 
                    WHERE end_time < $1 AND status = 'completed'
                """, cutoff_date)
            
            logger.info(f"Cleanup completed: removed old data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")