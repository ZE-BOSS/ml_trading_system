# Development Guide

## Development Environment Setup

### Prerequisites
- Python 3.9 or higher
- MetaTrader 5 terminal installed
- PostgreSQL 12+ and Redis 6+
- Node.js 16+ (for frontend development)

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd ppo-trading-system
```

2. **Create Python virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install TA-Lib** (Technical Analysis Library)
```bash
# On Ubuntu/Debian
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib

# On macOS with Homebrew
brew install ta-lib
pip install TA-Lib

# On Windows
# Download and install TA-Lib from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.24‑cp39‑cp39‑win_amd64.whl  # Adjust for your Python version
```

5. **Setup databases**
```bash
# PostgreSQL
createdb ppo_trading
psql ppo_trading < scripts/init_database.sql

# Redis (should be running on default port 6379)
redis-cli ping  # Should return PONG
```

6. **Configure settings**
```bash
cp config/settings.example.yaml config/settings.yaml
cp config/broker_config.example.yaml config/broker_config.yaml
# Edit configuration files with your settings
```

## Project Structure

```
ppo-trading-system/
├── config/                 # Configuration files
│   ├── settings.yaml       # Main application settings
│   ├── broker_config.yaml  # MT5 broker settings
│   └── model_config.yaml   # PPO model configuration
├── envs/                   # Trading environment implementation
│   ├── trading_env.py      # Main Gymnasium environment
│   ├── state_features.py   # Feature extraction
│   └── reward_functions.py # Reward calculation
├── models/                 # ML model implementations
│   ├── ppo_agent.py        # PPO agent implementation
│   └── saved_models/       # Trained model storage
├── services/               # External service integrations
│   ├── mt5_client.py       # MetaTrader 5 client
│   ├── websocket_server.py # WebSocket broadcasting
│   └── signal_generator.py # Trading signal generation
├── backend/                # Backend services
│   ├── api.py              # REST API server
│   └── data_manager.py     # Database operations
├── simulations/            # Backtesting and simulation
│   ├── backtest_runner.py  # Backtesting engine
│   └── live_simulation.py  # Live trading simulation
├── tests/                  # Test suite
├── docs/                   # Documentation
├── logs/                   # Application logs
└── main.py                 # Main entry point
```

## Development Workflow

### 1. Feature Development

1. **Create feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Implement feature**
- Write code following project conventions
- Add comprehensive docstrings
- Include type hints
- Add appropriate logging

3. **Write tests**
```bash
# Run specific test
pytest tests/test_your_feature.py

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

4. **Test locally**
```bash
# Test training
python main.py --mode train --symbol XAUUSD --broadcast

# Test backtesting
python main.py --mode backtest --model-path models/saved_models/your_model.zip

# Test API
python main.py --mode api
```

### 2. Code Quality Standards

#### Python Code Style
- Follow PEP 8 guidelines
- Use Black for code formatting
- Use isort for import organization
- Use flake8 for linting

```bash
# Format code
black .
isort .

# Check linting
flake8 .
```

#### Type Hints
All functions should include proper type hints:

```python
from typing import Dict, List, Optional, Any
import pandas as pd

def process_data(data: pd.DataFrame, 
                symbol: str,
                config: Dict[str, Any]) -> Optional[List[float]]:
    """
    Process market data for model input.
    
    Args:
        data: Market data DataFrame
        symbol: Trading symbol
        config: Configuration dictionary
        
    Returns:
        Processed features or None if processing fails
    """
    # Implementation here
    pass
```

#### Documentation
- All classes and functions must have docstrings
- Use Google-style docstrings
- Include examples for complex functions
- Update README.md when adding major features

#### Error Handling
```python
import logging
from loguru import logger

try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Specific error occurred: {e}")
    # Handle gracefully
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise  # Re-raise if cannot handle
```

### 3. Testing Guidelines

#### Test Structure
```python
import pytest
from unittest.mock import Mock, patch
from your_module import YourClass

class TestYourClass:
    def setup_method(self):
        """Setup before each test method."""
        self.instance = YourClass()
    
    def test_specific_functionality(self):
        """Test specific functionality."""
        # Arrange
        input_data = {"key": "value"}
        expected_output = {"processed": "value"}
        
        # Act
        result = self.instance.process(input_data)
        
        # Assert
        assert result == expected_output
    
    @patch('your_module.external_dependency')
    def test_with_mocked_dependency(self, mock_dependency):
        """Test with mocked external dependencies."""
        mock_dependency.return_value = "mocked_result"
        result = self.instance.method_using_dependency()
        assert result is not None
```

#### Test Categories
1. **Unit Tests**: Test individual functions/classes in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows

#### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_trading_env.py

# Run tests with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=. --cov-report=html tests/
```

### 4. Database Development

#### Schema Changes
1. Create migration script in `scripts/migrations/`
2. Update data models if using ORM
3. Test migration on development database
4. Document changes in migration file

#### Working with PostgreSQL
```python
# Example database operation
async def create_trading_session(self, session_data: Dict[str, Any]) -> str:
    """Create new trading session."""
    async with self.postgres_pool.acquire() as connection:
        session_id = str(uuid.uuid4())
        await connection.execute("""
            INSERT INTO trading_sessions (id, symbol, config, created_at)
            VALUES ($1, $2, $3, NOW())
        """, session_id, session_data['symbol'], json.dumps(session_data['config']))
        return session_id
```

#### Redis Operations
```python
# Example Redis caching
async def cache_market_data(self, symbol: str, data: pd.DataFrame):
    """Cache market data in Redis."""
    cache_key = f"market_data:{symbol}"
    serialized_data = data.to_json()
    await self.redis_client.setex(cache_key, 3600, serialized_data)
```

### 5. WebSocket Development

#### Adding New Event Types
1. Define event schema in WebSocket server
2. Add broadcasting method
3. Update client-side handlers
4. Document event format

```python
# Server-side event broadcasting
await self.websocket_broadcaster.broadcast_custom_event({
    'type': 'new_event_type',
    'data': {
        'key': 'value'
    }
})

# Client-side handling (JavaScript)
socket.on('new_event_type', (data) => {
    console.log('Received event:', data);
    // Handle event
});
```

## Debugging

### 1. Logging Configuration
```python
from loguru import logger

# Add file logging
logger.add("debug.log", level="DEBUG", rotation="1 MB")

# Add specific module logging
logger.add("trading.log", filter=lambda record: "trading" in record["name"])
```

### 2. Common Debugging Scenarios

#### PPO Training Issues
```bash
# Enable detailed training logs
export LOG_LEVEL=DEBUG
python main.py --mode train --symbol XAUUSD
```

#### MT5 Connection Problems
```python
# Check MT5 connection
if not mt5.initialize():
    print(f"MT5 initialization failed: {mt5.last_error()}")

# Test symbol availability
symbol_info = mt5.symbol_info("XAUUSD")
print(f"Symbol info: {symbol_info}")
```

#### Database Connection Issues
```bash
# Test PostgreSQL connection
psql -h localhost -U ppo_user -d ppo_trading -c "SELECT 1;"

# Test Redis connection
redis-cli ping
```

### 3. Performance Profiling
```python
import cProfile
import pstats

# Profile function
pr = cProfile.Profile()
pr.enable()
your_function()
pr.disable()

# Save stats
stats = pstats.Stats(pr)
stats.sort_stats('tottime')
stats.print_stats(10)
```

## Deployment

### 1. Development Deployment
```bash
# Start all services
docker-compose up -d

# Run application
python main.py --mode api
```

### 2. Production Deployment
```bash
# Build production image
docker build -t ppo-trading-system .

# Deploy with environment variables
docker run -d --env-file .env ppo-trading-system
```

### 3. Environment Variables
```bash
# Required environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PASSWORD=your_password
export MT5_LOGIN=your_login
export MT5_PASSWORD=your_password
export MT5_SERVER=your_server
```

## Contributing

### 1. Code Review Process
1. Create pull request with detailed description
2. Ensure all tests pass
3. Code review by team member
4. Address feedback
5. Merge after approval

### 2. Git Conventions
```bash
# Commit message format
type(scope): description

# Examples
feat(trading): add new signal filtering algorithm
fix(api): resolve database connection timeout
docs(readme): update installation instructions
test(env): add unit tests for reward calculation
```

### 3. Release Process
1. Update version number
2. Update CHANGELOG.md
3. Create release tag
4. Deploy to staging
5. Deploy to production after testing

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure project root is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. Database Connection Timeouts
- Check connection pool settings
- Verify database server status
- Review network configuration

#### 3. MT5 Connection Failures
- Verify MT5 terminal is running
- Check account credentials
- Ensure symbols are available in Market Watch

#### 4. Memory Issues During Training
- Reduce batch size in model configuration
- Limit historical data size
- Monitor memory usage with system tools

### Getting Help

1. Check logs in `logs/` directory
2. Review configuration files
3. Consult documentation
4. Search existing issues
5. Create new issue with detailed information

## Best Practices

### 1. Performance
- Use async/await for I/O operations
- Implement proper caching strategies
- Optimize database queries
- Monitor memory usage

### 2. Security
- Never commit sensitive configuration
- Use environment variables for secrets
- Implement proper input validation
- Regular dependency updates

### 3. Maintainability
- Keep functions small and focused
- Write comprehensive tests
- Document complex algorithms
- Regular code refactoring

### 4. Monitoring
- Implement health checks
- Monitor system metrics
- Set up alerting for failures
- Regular performance reviews

This development guide provides the foundation for contributing to the PPO Trading System. Following these guidelines ensures code quality, system reliability, and maintainability.