# PPO Reinforcement Learning Trading System

## Overview

This is a comprehensive trading system that uses Proximal Policy Optimization (PPO) reinforcement learning to generate trading signals in real-time. The system supports both backtesting and live trading with MT5 integration, real-time WebSocket broadcasting to mobile/web clients, and comprehensive data persistence.

## Features

- **PPO Reinforcement Learning Agent**: Custom trading environment with sophisticated reward functions
- **Real-time Signal Broadcasting**: WebSocket server for live updates to connected clients
- **MT5 Integration**: Direct connection to MetaTrader 5 for live market data and order execution
- **Data Persistence**: PostgreSQL database with Redis caching for optimal performance
- **Backtesting Engine**: Historical data analysis with detailed performance metrics
- **Mobile/Web Integration**: Real-time chart updates and trading notifications
- **Production-Ready Architecture**: Modular design with comprehensive error handling and logging

## Architecture

### Data Flow
1. **Training Phase**: Historical data → PPO Agent → Model checkpoints → Performance metrics
2. **Live Trading**: MT5 Live Data → Trained Agent → Signal Generation → Trade Execution → Broadcasting
3. **Broadcasting**: Real-time updates via WebSocket to connected mobile/web clients

### Technology Stack
- **Backend**: Python with FastAPI
- **Database**: PostgreSQL (primary), Redis (cache)
- **ML Framework**: Stable-Baselines3 (PPO)
- **Market Data**: MetaTrader 5 Python API
- **Real-time**: WebSocket with Socket.IO
- **Frontend**: React with Socket.IO client

## Quick Start

### 1. Environment Setup
```bash
# Clone and navigate to project
cd ppo-trading-system

# Install dependencies
pip install -r requirements.txt

# Setup database
python scripts/setup_database.py

# Configure settings
cp config/settings.example.yaml config/settings.yaml
# Edit config/settings.yaml with your MT5 credentials and database settings
```

### 2. Data Preparation
```bash
# Download historical data
python scripts/download_historical_data.py --symbol XAUUSD --days 365

# Preprocess data for training
python scripts/preprocess_data.py
```

### 3. Training
```bash
# Start training with real-time monitoring
python main.py --mode train --broadcast

# Monitor training progress at http://localhost:3000
```

### 4. Live Trading
```bash
# Start live trading (demo mode)
python main.py --mode live --demo

# Start live trading (real money - be careful!)
python main.py --mode live --real
```

## Configuration

All system configurations are managed through YAML files in the `config/` directory:

- `settings.yaml`: Main application settings
- `broker_config.yaml`: MT5 broker configurations
- `model_config.yaml`: PPO model hyperparameters

## API Documentation

The system provides REST API endpoints for monitoring and control:

- `GET /api/status` - System status and metrics
- `GET /api/models` - Available trained models
- `GET /api/performance/{session_id}` - Training/trading performance
- `POST /api/trading/start` - Start live trading
- `POST /api/trading/stop` - Stop live trading

WebSocket events:
- `training_update` - Real-time training progress
- `signal_generated` - New trading signal
- `trade_executed` - Trade execution confirmation
- `performance_update` - Updated performance metrics

## Project Structure

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system architecture documentation.

## Development

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for development guidelines and contribution instructions.

## License

MIT License - see LICENSE file for details.