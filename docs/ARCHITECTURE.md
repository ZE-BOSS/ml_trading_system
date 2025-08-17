# PPO Trading System Architecture

## System Overview

The PPO Trading System is a comprehensive reinforcement learning-based trading platform that uses Proximal Policy Optimization (PPO) to generate trading signals and execute trades in real-time. The system is designed with a modular architecture that separates concerns and allows for easy testing, maintenance, and scaling.

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mobile/Web    │    │   WebSocket     │    │   REST API      │
│   Clients       │◄──►│   Server        │    │   Server        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Cache    │    │   Main System   │    │   Database      │
│   (Redis)       │◄──►│   Orchestrator  │◄──►│   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MT5 Client    │    │   PPO Agent     │    │   Signal        │
│   (Live Data)   │◄──►│   (Training)    │◄──►│   Generator     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. PPO Agent (`models/ppo_agent.py`)
- **Purpose**: Implements the core PPO reinforcement learning algorithm
- **Key Features**:
  - Custom policy network architecture optimized for trading
  - Training with real-time progress broadcasting
  - Model checkpointing and versioning
  - Performance evaluation and validation
- **Dependencies**: Stable-Baselines3, PyTorch, Gymnasium

### 2. Trading Environment (`envs/trading_env.py`)
- **Purpose**: Gymnasium-compatible environment for training the PPO agent
- **Key Features**:
  - Realistic trading simulation with transaction costs
  - Comprehensive state representation with technical indicators
  - Flexible reward functions (profit-based, Sharpe ratio, custom)
  - Risk management and position sizing
- **Components**:
  - `StateFeatureExtractor`: Processes market data into ML features
  - `RewardCalculator`: Implements various reward strategies

### 3. Signal Generator (`services/signal_generator.py`)
- **Purpose**: Converts PPO predictions into executable trading signals
- **Key Features**:
  - Real-time inference from trained models
  - Confidence scoring and signal filtering
  - Risk-based position sizing
  - Stop-loss and take-profit calculation
- **Output**: Structured trading signals with risk metrics

### 4. MT5 Client (`services/mt5_client.py`)
- **Purpose**: Interfaces with MetaTrader 5 for market data and trade execution
- **Key Features**:
  - Real-time market data streaming
  - Historical data retrieval
  - Order execution and position management
  - Account monitoring and risk checks
- **Integration**: Direct MT5 Python API integration

### 5. WebSocket Server (`services/websocket_server.py`)
- **Purpose**: Real-time communication with client applications
- **Key Features**:
  - Training progress broadcasting
  - Live signal distribution
  - Trade execution notifications
  - System status updates
- **Technology**: Socket.IO with Redis pub/sub for scaling

### 6. Data Manager (`backend/data_manager.py`)
- **Purpose**: Handles all data persistence and caching operations
- **Key Features**:
  - PostgreSQL for relational data
  - Redis for real-time caching
  - Training session management
  - Performance metrics storage
- **Schema**: Comprehensive database schema for all system data

### 7. REST API (`backend/api.py`)
- **Purpose**: HTTP API for system control and monitoring
- **Key Features**:
  - Model management endpoints
  - Training control
  - Performance analytics
  - System health monitoring
- **Technology**: FastAPI with async support

## Data Flow Architecture

### Training Flow
1. **Data Preparation**: Historical market data → Feature extraction → Environment setup
2. **Model Training**: PPO agent learns trading strategies through reinforcement learning
3. **Progress Broadcasting**: Real-time training metrics via WebSocket
4. **Model Persistence**: Trained models saved with performance metadata
5. **Validation**: Backtesting on unseen data before deployment

### Live Trading Flow
1. **Market Data**: MT5 streams real-time tick data
2. **Feature Processing**: Raw data → Technical indicators → ML features
3. **Signal Generation**: Trained PPO model → Trading decisions
4. **Risk Management**: Position sizing, stop-loss calculation
5. **Order Execution**: Validated signals → MT5 trade execution
6. **Broadcasting**: Real-time updates to connected clients
7. **Performance Tracking**: Trade results stored in database

### Data Storage Architecture
```
PostgreSQL (Primary Database)
├── training_sessions      # Training session metadata
├── training_metrics       # Step-by-step training progress
├── signals               # Generated trading signals
├── live_trades          # Executed trade records
└── performance_summary  # Aggregated performance metrics

Redis (Cache Layer)
├── historical:{symbol}   # Cached market data
├── live_tick:{symbol}   # Real-time tick cache
├── tick_history:{symbol} # Recent tick history
└── ws_messages:{type}   # WebSocket message queue
```

## Configuration Management

The system uses YAML configuration files organized by purpose:

- `config/settings.yaml` - Main application settings
- `config/broker_config.yaml` - MT5 broker configurations
- `config/model_config.yaml` - PPO model hyperparameters

Configuration is hierarchical and supports environment-specific overrides.

## Security Architecture

### Authentication & Authorization
- API key-based authentication for REST endpoints
- Rate limiting to prevent abuse
- CORS configuration for web client access

### Data Security
- Database connection pooling with connection limits
- Redis password protection
- Sensitive configuration via environment variables

### Trading Security
- Multiple risk management layers
- Position size limits
- Maximum drawdown protection
- Demo mode for testing

## Scalability Considerations

### Horizontal Scaling
- Stateless API design allows multiple server instances
- Redis pub/sub enables WebSocket scaling across nodes
- Database connection pooling handles concurrent access

### Performance Optimization
- Redis caching for frequently accessed data
- Async/await throughout for non-blocking operations
- Efficient pandas operations for data processing
- Model inference optimization with batch processing

### Resource Management
- Configurable memory limits for data operations
- Automatic cleanup of old training data
- Efficient tick data storage with circular buffers

## Error Handling & Monitoring

### Logging
- Structured logging with loguru
- Multiple log levels and output destinations
- Automatic log rotation and retention

### Error Recovery
- Graceful degradation when components fail
- Automatic reconnection for network services
- Transaction rollback for database operations

### Health Monitoring
- System health checks via API endpoints
- Component status monitoring
- Performance metrics collection

## Development & Testing

### Code Organization
- Modular design with clear separation of concerns
- Comprehensive type hints throughout
- Extensive docstrings and comments

### Testing Strategy
- Unit tests for individual components
- Integration tests for system workflows
- Backtesting for strategy validation

### Deployment
- Docker containerization support
- Environment-specific configurations
- Automated database migrations

## Future Enhancements

### Planned Features
- Multi-model ensemble trading
- Advanced risk management strategies
- Additional broker integrations
- Enhanced mobile client features

### Technical Debt
- Implement proper model versioning
- Add comprehensive test coverage
- Optimize database queries
- Enhance error handling granularity

This architecture provides a solid foundation for a production-ready algorithmic trading system while maintaining flexibility for future enhancements and scaling.