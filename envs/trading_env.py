"""
PPO Trading Environment
======================

This module implements a custom Gymnasium environment for training PPO agents
on financial time series data. The environment handles:

- State representation with technical indicators
- Action space for trading decisions (Buy/Sell/Hold)
- Reward function based on portfolio performance
- Risk management and position sizing
- Real-time market simulation

Author: PPO Trading System
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
import yaml

from .state_features import StateFeatureExtractor
from .reward_functions import RewardCalculator


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for PPO agent training.
    
    This environment simulates a trading scenario where the agent makes
    buy/sell/hold decisions based on market data and technical indicators.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 config_path: str = "config/model_config.yaml",
                 initial_balance: float = 10000.0,
                 transaction_cost: float = 0.0001,
                 render_mode: Optional[str] = None):
        """
        Initialize the trading environment.
        
        Args:
            data: Market data with OHLCV columns
            config_path: Path to model configuration file
            initial_balance: Starting account balance
            transaction_cost: Cost per trade (as fraction of trade value)
            render_mode: Rendering mode for visualization
        """
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.render_mode = render_mode
        
        # Environment parameters
        self.lookback_window = self.config['environment']['lookback_window']
        self.action_config = self.config['environment']['action_space']
        self.reward_config = self.config['environment']['reward_function']
        
        # Initialize feature extractor and reward calculator
        self.feature_extractor = StateFeatureExtractor(self.config)
        self.reward_calculator = RewardCalculator(self.config)
        
        # Process data and extract features
        self._prepare_data()
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Trading state variables
        self.reset()
        
        logger.info(f"Trading environment initialized with {len(self.data)} data points")
    
    def _prepare_data(self) -> None:
        """Prepare and validate input data."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Extract features
        self.processed_data = self.feature_extractor.extract_features(self.data)
        
        # Remove NaN values that may result from indicator calculations
        self.processed_data = self.processed_data.dropna()
        
        if len(self.processed_data) < self.lookback_window:
            raise ValueError(f"Insufficient data: need at least {self.lookback_window} samples")
        
        logger.info(f"Data prepared: {len(self.processed_data)} samples, {len(self.processed_data.columns)} features")
    
    def _setup_spaces(self) -> None:
        """Setup action and observation spaces."""
        # Action space
        if self.action_config['type'] == 'discrete':
            self.action_space = spaces.Discrete(len(self.action_config['actions']))
        else:
            # Continuous action space
            bounds = self.action_config['continuous_bounds']
            self.action_space = spaces.Box(
                low=np.array([bounds['position_size'][0]]),
                high=np.array([bounds['position_size'][1]]),
                dtype=np.float32
            )
        
        # Observation space
        n_features = len(self.processed_data.columns)
        n_portfolio_features = 5  # balance, position, unrealized_pnl, total_trades, current_drawdown
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window * n_features + n_portfolio_features,),
            dtype=np.float32
        )
        
        logger.info(f"Action space: {self.action_space}")
        logger.info(f"Observation space: {self.observation_space.shape}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset trading state
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0  # Current position size
        self.position_entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.trade_history = []
        self.portfolio_values = [self.initial_balance]
        self.unrealized_pnl = 0.0
        
        # Performance tracking
        self.max_portfolio_value = self.initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action to take (0=Hold, 1=Buy, 2=Sell for discrete)
            
        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode is finished
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Execute action
        self._execute_action(action)
        
        # Move to next time step
        self.current_step += 1
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            self._get_current_price(),
            self.position,
            self.balance,
            self.unrealized_pnl,
            self.current_drawdown,
            self._get_portfolio_value()
        )
        
        # Update portfolio tracking
        self._update_portfolio_tracking()
        
        # Check if episode is done
        terminated = self.current_step >= len(self.processed_data) - 1
        truncated = self._check_early_termination()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> None:
        """Execute trading action."""
        current_price = self._get_current_price()
        
        if self.action_config['type'] == 'discrete':
            action_name = self.action_config['actions'][action]
            
            if action_name == 'buy' and self.position <= 0:
                self._open_position(1.0, current_price)
            elif action_name == 'sell' and self.position >= 0:
                self._open_position(-1.0, current_price)
            elif action_name == 'hold':
                # Update unrealized PnL for existing position
                if self.position != 0:
                    self.unrealized_pnl = (current_price - self.position_entry_price) * self.position
        else:
            # Continuous action space - action represents target position
            target_position = float(action[0])
            if target_position != self.position:
                self._adjust_position(target_position, current_price)
    
    def _open_position(self, direction: float, price: float) -> None:
        """Open a new position."""
        if self.position != 0:
            # Close existing position first
            self._close_position(price)
        
        # Calculate position size based on available balance
        position_sizing = self.config['backtesting']['position_sizing']
        
        if position_sizing['method'] == 'fixed_fraction':
            position_value = self.balance * position_sizing['fraction']
            position_size = (position_value / price) * direction
        elif position_sizing['method'] == 'fixed':
            position_size = position_sizing.get('size', 0.01) * direction
        else:
            position_size = 0.01 * direction  # Default small position
        
        # Apply maximum position limit
        max_position = position_sizing.get('max_position', 0.1)
        max_position_value = self.balance * max_position
        max_position_size = max_position_value / price
        
        if abs(position_size) > max_position_size:
            position_size = max_position_size * np.sign(position_size)
        
        # Execute trade
        trade_cost = abs(position_size) * price * self.transaction_cost
        
        if self.balance >= trade_cost:
            self.position = position_size
            self.position_entry_price = price
            self.balance -= trade_cost
            self.total_trades += 1
            self.unrealized_pnl = 0.0
            
            # Log trade
            trade_info = {
                'step': self.current_step,
                'action': 'BUY' if direction > 0 else 'SELL',
                'size': position_size,
                'price': price,
                'cost': trade_cost,
                'balance_after': self.balance
            }
            self.trade_history.append(trade_info)
    
    def _close_position(self, price: float) -> None:
        """Close current position."""
        if self.position == 0:
            return
        
        # Calculate profit/loss
        pnl = (price - self.position_entry_price) * self.position
        trade_cost = abs(self.position) * price * self.transaction_cost
        net_pnl = pnl - trade_cost
        
        # Update balance
        self.balance += self.position * price + net_pnl
        
        # Track winning trades
        if net_pnl > 0:
            self.winning_trades += 1
        
        # Log trade closure
        trade_info = {
            'step': self.current_step,
            'action': 'CLOSE',
            'size': self.position,
            'entry_price': self.position_entry_price,
            'exit_price': price,
            'pnl': net_pnl,
            'balance_after': self.balance
        }
        self.trade_history.append(trade_info)
        
        # Reset position
        self.position = 0.0
        self.position_entry_price = 0.0
        self.unrealized_pnl = 0.0
    
    def _adjust_position(self, target_position: float, price: float) -> None:
        """Adjust position to target size (for continuous action space)."""
        if target_position == self.position:
            return
        
        position_change = target_position - self.position
        trade_cost = abs(position_change) * price * self.transaction_cost
        
        if self.balance >= trade_cost:
            self.position = target_position
            if self.position != 0:
                # Weighted average entry price for position adjustments
                if self.position_entry_price == 0:
                    self.position_entry_price = price
                else:
                    total_position_value = self.position_entry_price * self.position + position_change * price
                    self.position_entry_price = total_position_value / self.position
            
            self.balance -= trade_cost
            self.total_trades += 1
    
    def _get_current_price(self) -> float:
        """Get current market price."""
        return float(self.processed_data.iloc[self.current_step]['close'])
    
    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        current_price = self._get_current_price()
        position_value = self.position * current_price if self.position != 0 else 0.0
        return self.balance + position_value
    
    def _update_portfolio_tracking(self) -> None:
        """Update portfolio performance metrics."""
        portfolio_value = self._get_portfolio_value()
        self.portfolio_values.append(portfolio_value)
        
        # Update maximum portfolio value
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        
        # Calculate current drawdown
        self.current_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        
        # Update maximum drawdown
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown
    
    def _check_early_termination(self) -> bool:
        """Check if episode should be terminated early."""
        # Terminate if maximum drawdown exceeded
        max_drawdown_stop = self.config['backtesting']['risk_management'].get('max_drawdown_stop', 0.5)
        if self.current_drawdown > max_drawdown_stop:
            return True
        
        # Terminate if balance too low
        if self.balance < self.initial_balance * 0.1:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current environment observation."""
        # Market features for lookback window
        start_idx = max(0, self.current_step - self.lookback_window)
        market_features = self.processed_data.iloc[start_idx:self.current_step].values
        
        # Pad if necessary
        if len(market_features) < self.lookback_window:
            padding = np.zeros((self.lookback_window - len(market_features), market_features.shape[1]))
            market_features = np.vstack([padding, market_features])
        
        market_features = market_features.flatten()
        
        # Portfolio features
        portfolio_value = self._get_portfolio_value()
        portfolio_features = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position,  # Current position
            self.unrealized_pnl / self.initial_balance,  # Normalized unrealized PnL
            self.total_trades / 1000.0,  # Normalized trade count
            self.current_drawdown  # Current drawdown
        ])
        
        observation = np.concatenate([market_features, portfolio_features]).astype(np.float32)
        return observation
    
    def _get_info(self) -> Dict:
        """Get additional environment information."""
        portfolio_value = self._get_portfolio_value()
        win_rate = self.winning_trades / max(1, self.total_trades)
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': portfolio_value,
            'total_return': (portfolio_value - self.initial_balance) / self.initial_balance,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'unrealized_pnl': self.unrealized_pnl
        }
    
    def render(self) -> None:
        """Render environment (optional visualization)."""
        if self.render_mode == 'human':
            info = self._get_info()
            print(f"Step: {info['step']}, Balance: ${info['balance']:.2f}, "
                  f"Portfolio: ${info['portfolio_value']:.2f}, "
                  f"Return: {info['total_return']:.2%}, "
                  f"Drawdown: {info['current_drawdown']:.2%}")