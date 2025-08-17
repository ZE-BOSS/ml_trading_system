#!/usr/bin/env python3
"""
Database Setup Script
=====================

This script initializes the PostgreSQL database for the PPO Trading System.
It creates all necessary tables, indexes, and initial data.

Usage:
    python scripts/setup_database.py [--reset] [--config CONFIG_PATH]

Author: PPO Trading System
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import asyncpg
import yaml
from loguru import logger


class DatabaseSetup:
    """Database setup and initialization."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize database setup."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.db_config = config['database']['postgres']
        
    async def create_database_if_not_exists(self):
        """Create database if it doesn't exist."""
        # Connect to postgres database to create our database
        postgres_dsn = f"postgresql://{self.db_config['username']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/postgres"
        
        try:
            conn = await asyncpg.connect(postgres_dsn)
            
            # Check if database exists
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", 
                self.db_config['database']
            )
            
            if not exists:
                await conn.execute(f'CREATE DATABASE "{self.db_config["database"]}"')
                logger.info(f"Created database: {self.db_config['database']}")
            else:
                logger.info(f"Database already exists: {self.db_config['database']}")
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise
    
    async def setup_tables(self, reset: bool = False):
        """Setup all database tables."""
        dsn = f"postgresql://{self.db_config['username']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        
        try:
            conn = await asyncpg.connect(dsn)
            
            if reset:
                logger.info("Dropping existing tables...")
                await self.drop_tables(conn)
            
            logger.info("Creating tables...")
            await self.create_tables(conn)
            
            logger.info("Creating indexes...")
            await self.create_indexes(conn)
            
            logger.info("Setting up initial data...")
            await self.setup_initial_data(conn)
            
            await conn.close()
            logger.info("Database setup completed successfully!")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
    
    async def drop_tables(self, conn):
        """Drop all tables (for reset)."""
        tables = [
            'training_metrics',
            'signals', 
            'live_trades',
            'performance_summary',
            'training_sessions'
        ]
        
        for table in tables:
            await conn.execute(f'DROP TABLE IF EXISTS {table} CASCADE')
            logger.info(f"Dropped table: {table}")
    
    async def create_tables(self, conn):
        """Create all database tables."""
        
        # Training sessions table
        await conn.execute("""
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
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        logger.info("Created table: training_sessions")
        
        # Training metrics table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS training_metrics (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id VARCHAR(255) NOT NULL,
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
                additional_metrics JSONB,
                CONSTRAINT fk_training_metrics_session 
                    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
                    ON DELETE CASCADE
            );
        """)
        logger.info("Created table: training_metrics")
        
        # Signals table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                signal_id VARCHAR(255) UNIQUE NOT NULL,
                session_id VARCHAR(255),
                symbol VARCHAR(20) NOT NULL,
                action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'SELL', 'HOLD')),
                confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                lot_size FLOAT NOT NULL CHECK (lot_size > 0),
                entry_price FLOAT NOT NULL CHECK (entry_price > 0),
                stop_loss FLOAT CHECK (stop_loss > 0),
                take_profit FLOAT CHECK (take_profit > 0),
                expected_return FLOAT,
                risk_score FLOAT CHECK (risk_score >= 0 AND risk_score <= 1),
                model_version VARCHAR(100),
                features_used JSONB,
                timestamp TIMESTAMP DEFAULT NOW(),
                executed BOOLEAN DEFAULT FALSE,
                execution_result JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        logger.info("Created table: signals")
        
        # Live trades table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS live_trades (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                signal_id VARCHAR(255),
                ticket VARCHAR(50),
                symbol VARCHAR(20) NOT NULL,
                action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'SELL')),
                lot_size FLOAT NOT NULL CHECK (lot_size > 0),
                entry_price FLOAT NOT NULL CHECK (entry_price > 0),
                exit_price FLOAT CHECK (exit_price > 0),
                stop_loss FLOAT CHECK (stop_loss > 0),
                take_profit FLOAT CHECK (take_profit > 0),
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                profit FLOAT,
                commission FLOAT DEFAULT 0,
                swap FLOAT DEFAULT 0,
                status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'closed', 'cancelled')),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                CONSTRAINT fk_live_trades_signal 
                    FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
                    ON DELETE SET NULL
            );
        """)
        logger.info("Created table: live_trades")
        
        # Performance summary table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_summary (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id VARCHAR(255) NOT NULL,
                symbol VARCHAR(20),
                period_start TIMESTAMP NOT NULL,
                period_end TIMESTAMP NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                win_rate FLOAT DEFAULT 0 CHECK (win_rate >= 0 AND win_rate <= 1),
                total_profit FLOAT DEFAULT 0,
                max_drawdown FLOAT DEFAULT 0,
                sharpe_ratio FLOAT DEFAULT 0,
                profit_factor FLOAT DEFAULT 0,
                avg_trade_duration INTERVAL,
                roi FLOAT DEFAULT 0,
                volatility FLOAT DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        logger.info("Created table: performance_summary")
        
        # System logs table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                level VARCHAR(20) NOT NULL,
                message TEXT NOT NULL,
                module VARCHAR(100),
                function_name VARCHAR(100),
                line_number INTEGER,
                timestamp TIMESTAMP DEFAULT NOW(),
                additional_data JSONB
            );
        """)
        logger.info("Created table: system_logs")
        
        # Model metadata table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                model_name VARCHAR(255) NOT NULL,
                model_path TEXT NOT NULL,
                model_type VARCHAR(50) DEFAULT 'PPO',
                training_session_id VARCHAR(255),
                hyperparameters JSONB,
                performance_metrics JSONB,
                validation_results JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                is_active BOOLEAN DEFAULT FALSE,
                notes TEXT,
                CONSTRAINT fk_model_training_session 
                    FOREIGN KEY (training_session_id) REFERENCES training_sessions(session_id)
                    ON DELETE SET NULL
            );
        """)
        logger.info("Created table: model_metadata")
    
    async def create_indexes(self, conn):
        """Create database indexes for performance."""
        
        indexes = [
            # Training metrics indexes
            "CREATE INDEX IF NOT EXISTS idx_training_metrics_session_timestep ON training_metrics(session_id, timestep)",
            "CREATE INDEX IF NOT EXISTS idx_training_metrics_timestamp ON training_metrics(timestamp DESC)",
            
            # Signals indexes
            "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp ON signals(symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_signals_executed ON signals(executed, timestamp DESC)",
            
            # Live trades indexes
            "CREATE INDEX IF NOT EXISTS idx_live_trades_symbol_time ON live_trades(symbol, entry_time DESC)",
            "CREATE INDEX IF NOT EXISTS idx_live_trades_status ON live_trades(status, entry_time DESC)",
            "CREATE INDEX IF NOT EXISTS idx_live_trades_signal ON live_trades(signal_id)",
            
            # Performance summary indexes
            "CREATE INDEX IF NOT EXISTS idx_performance_summary_session ON performance_summary(session_id, period_end DESC)",
            "CREATE INDEX IF NOT EXISTS idx_performance_summary_symbol ON performance_summary(symbol, period_end DESC)",
            
            # System logs indexes
            "CREATE INDEX IF NOT EXISTS idx_system_logs_level_timestamp ON system_logs(level, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_module ON system_logs(module, timestamp DESC)",
            
            # Model metadata indexes
            "CREATE INDEX IF NOT EXISTS idx_model_metadata_active ON model_metadata(is_active, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_model_metadata_name ON model_metadata(model_name)"
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
            logger.info(f"Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
    
    async def setup_initial_data(self, conn):
        """Setup initial data and configurations."""
        
        # Create default system configuration
        await conn.execute("""
            INSERT INTO training_sessions (session_id, symbol, status, notes)
            VALUES ('system-init', 'SYSTEM', 'completed', 'Initial system setup')
            ON CONFLICT (session_id) DO NOTHING
        """)
        
        logger.info("Initial data setup completed")
    
    async def verify_setup(self):
        """Verify database setup is correct."""
        dsn = f"postgresql://{self.db_config['username']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        
        try:
            conn = await asyncpg.connect(dsn)
            
            # Check if all tables exist
            tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            
            table_names = [table['table_name'] for table in tables]
            expected_tables = [
                'training_sessions', 'training_metrics', 'signals', 
                'live_trades', 'performance_summary', 'system_logs', 'model_metadata'
            ]
            
            missing_tables = [t for t in expected_tables if t not in table_names]
            
            if missing_tables:
                logger.error(f"Missing tables: {missing_tables}")
                return False
            
            # Check if we can insert and query data
            test_session_id = 'test-setup-verification'
            await conn.execute("""
                INSERT INTO training_sessions (session_id, symbol, status, notes)
                VALUES ($1, 'TEST', 'completed', 'Setup verification test')
                ON CONFLICT (session_id) DO NOTHING
            """, test_session_id)
            
            result = await conn.fetchval("""
                SELECT session_id FROM training_sessions WHERE session_id = $1
            """, test_session_id)
            
            if result != test_session_id:
                logger.error("Database write/read test failed")
                return False
            
            # Clean up test data
            await conn.execute("""
                DELETE FROM training_sessions WHERE session_id = $1
            """, test_session_id)
            
            await conn.close()
            
            logger.info("Database verification completed successfully!")
            logger.info(f"Found tables: {', '.join(table_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Database verification failed: {e}")
            return False


async def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup PPO Trading System Database")
    parser.add_argument('--reset', action='store_true', help='Reset database (drop all tables)')
    parser.add_argument('--config', default='config/settings.yaml', help='Configuration file path')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing setup')
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    try:
        db_setup = DatabaseSetup(args.config)
        
        if args.verify_only:
            success = await db_setup.verify_setup()
            sys.exit(0 if success else 1)
        
        logger.info("Starting database setup...")
        
        # Create database if it doesn't exist
        await db_setup.create_database_if_not_exists()
        
        # Setup tables and indexes
        await db_setup.setup_tables(reset=args.reset)
        
        # Verify setup
        success = await db_setup.verify_setup()
        
        if success:
            logger.info("✅ Database setup completed successfully!")
        else:
            logger.error("❌ Database setup verification failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())