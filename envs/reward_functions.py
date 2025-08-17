"""
Reward Functions
================

This module implements various reward functions for the PPO trading agent.
The reward function is crucial for training the agent to make profitable trades.

Available reward strategies:
- Profit-based rewards
- Sharpe ratio-based rewards
- Risk-adjusted returns
- Custom composite rewards

Author: PPO Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from loguru import logger


class RewardCalculator:
    """Calculates rewards for the trading environment."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reward calculator.
        
        Args:
            config: Configuration dictionary from model_config.yaml
        """
        self.config = config
        self.reward_config = config['environment']['reward_function']
        self.reward_type = self.reward_config['type']
        self.params = self.reward_config['parameters']
        
        # Tracking variables
        self.previous_portfolio_value = None
        self.returns_history = []
        self.max_returns_history = 1000  # Keep last N returns for Sharpe calculation
        
        logger.info(f"Reward calculator initialized with type: {self.reward_type}")
    
    def calculate_reward(self, 
                        current_price: float,
                        position: float,
                        balance: float,
                        unrealized_pnl: float,
                        drawdown: float,
                        portfolio_value: float) -> float:
        """
        Calculate reward for the current step.
        
        Args:
            current_price: Current market price
            position: Current position size
            balance: Current account balance
            unrealized_pnl: Unrealized profit/loss
            drawdown: Current drawdown percentage
            portfolio_value: Total portfolio value
            
        Returns:
            Reward value for this step
        """
        if self.reward_type == 'profit_based':
            return self._profit_based_reward(portfolio_value, drawdown, position)
        elif self.reward_type == 'sharpe_based':
            return self._sharpe_based_reward(portfolio_value)
        elif self.reward_type == 'custom':
            return self._custom_reward(current_price, position, balance, unrealized_pnl, drawdown, portfolio_value)
        else:
            logger.warning(f"Unknown reward type: {self.reward_type}, using profit_based")
            return self._profit_based_reward(portfolio_value, drawdown, position)
    
    def _profit_based_reward(self, portfolio_value: float, drawdown: float, position: float) -> float:
        """
        Calculate profit-based reward.
        
        This reward function focuses on portfolio returns while penalizing drawdowns.
        """
        if self.previous_portfolio_value is None:
            self.previous_portfolio_value = portfolio_value
            return 0.0
        
        # Calculate step return
        step_return = (portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        
        # Base reward from profit
        profit_reward = step_return * self.params['profit_weight']
        
        # Drawdown penalty
        drawdown_penalty = drawdown * self.params['drawdown_penalty']
        
        # Transaction cost penalty (approximate)
        transaction_penalty = abs(position) * self.params['transaction_cost'] * self.params['penalty_weight']
        
        # Combine components
        total_reward = profit_reward - drawdown_penalty - transaction_penalty
        
        # Update tracking
        self.previous_portfolio_value = portfolio_value
        self.returns_history.append(step_return)
        
        # Keep only recent returns
        if len(self.returns_history) > self.max_returns_history:
            self.returns_history.pop(0)
        
        return float(total_reward)
    
    def _sharpe_based_reward(self, portfolio_value: float) -> float:
        """
        Calculate Sharpe ratio-based reward.
        
        This reward function uses the rolling Sharpe ratio as the reward signal.
        """
        if self.previous_portfolio_value is None:
            self.previous_portfolio_value = portfolio_value
            return 0.0
        
        # Calculate step return
        step_return = (portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        self.returns_history.append(step_return)
        
        # Keep only recent returns
        if len(self.returns_history) > self.max_returns_history:
            self.returns_history.pop(0)
        
        # Calculate Sharpe ratio if we have enough history
        if len(self.returns_history) < 30:  # Need at least 30 observations
            reward = step_return  # Use simple return for initial steps
        else:
            returns_array = np.array(self.returns_history)
            excess_returns = returns_array - (self.params['risk_free_rate'] / 252)  # Daily risk-free rate
            
            if np.std(excess_returns) > 0:
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
                reward = sharpe_ratio * 0.01  # Scale down the Sharpe ratio
            else:
                reward = 0.0
        
        self.previous_portfolio_value = portfolio_value
        return float(reward)
    
    def _custom_reward(self, 
                      current_price: float,
                      position: float,
                      balance: float,
                      unrealized_pnl: float,
                      drawdown: float,
                      portfolio_value: float) -> float:
        """
        Calculate custom composite reward.
        
        This combines multiple reward components for more sophisticated training.
        """
        if self.previous_portfolio_value is None:
            self.previous_portfolio_value = portfolio_value
            return 0.0
        
        # Base profit reward
        step_return = (portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        profit_reward = step_return * self.params['profit_weight']
        
        # Custom reward components
        custom_rewards = self.reward_config.get('custom_rewards', {})
        custom_reward_total = 0.0
        
        # Trend following reward
        if 'trend_following' in custom_rewards and len(self.returns_history) > 5:
            recent_returns = self.returns_history[-5:]
            trend_strength = np.mean(recent_returns)
            position_alignment = np.sign(position) * np.sign(trend_strength)
            trend_reward = position_alignment * abs(trend_strength) * custom_rewards['trend_following']
            custom_reward_total += trend_reward
        
        # Mean reversion reward
        if 'mean_reversion' in custom_rewards and len(self.returns_history) > 20:
            recent_returns = np.array(self.returns_history[-20:])
            z_score = (recent_returns[-1] - np.mean(recent_returns)) / (np.std(recent_returns) + 1e-8)
            # Reward betting against extreme moves
            mean_reversion_signal = -np.sign(z_score) if abs(z_score) > 1.5 else 0
            position_alignment = np.sign(position) * mean_reversion_signal
            mr_reward = position_alignment * custom_rewards['mean_reversion']
            custom_reward_total += mr_reward
        
        # Volatility targeting reward
        if 'volatility_targeting' in custom_rewards and len(self.returns_history) > 10:
            recent_volatility = np.std(self.returns_history[-10:])
            target_volatility = 0.02  # 2% daily volatility target
            vol_adjustment = 1.0 - abs(recent_volatility - target_volatility) / target_volatility
            vol_reward = vol_adjustment * custom_rewards['volatility_targeting']
            custom_reward_total += vol_reward
        
        # Risk penalties
        drawdown_penalty = drawdown * self.params['drawdown_penalty']
        transaction_penalty = abs(position) * self.params['transaction_cost'] * self.params['penalty_weight']
        
        # Combine all components
        total_reward = (profit_reward + custom_reward_total - 
                       drawdown_penalty - transaction_penalty)
        
        # Update tracking
        self.previous_portfolio_value = portfolio_value
        self.returns_history.append(step_return)
        
        if len(self.returns_history) > self.max_returns_history:
            self.returns_history.pop(0)
        
        return float(total_reward)
    
    def get_reward_breakdown(self) -> Dict[str, float]:
        """
        Get breakdown of reward components (for debugging/analysis).
        
        Returns:
            Dictionary with reward component values
        """
        # This would need to be called after calculate_reward to get meaningful values
        # For now, return empty dict
        return {}
    
    def reset(self) -> None:
        """Reset the reward calculator state."""
        self.previous_portfolio_value = None
        self.returns_history = []
        logger.info("Reward calculator reset")


class AdaptiveRewardCalculator(RewardCalculator):
    """
    Adaptive reward calculator that adjusts reward parameters based on performance.
    
    This can help with training stability by adapting the reward structure
    based on the agent's learning progress.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adaptation_enabled = True
        self.performance_history = []
        self.adaptation_frequency = 1000  # Steps between adaptations
        self.step_count = 0
        
    def calculate_reward(self, *args, **kwargs) -> float:
        """Calculate adaptive reward with parameter adjustment."""
        reward = super().calculate_reward(*args, **kwargs)
        
        self.step_count += 1
        
        # Adapt parameters periodically
        if (self.adaptation_enabled and 
            self.step_count % self.adaptation_frequency == 0 and
            len(self.returns_history) > 100):
            
            self._adapt_parameters()
        
        return reward
    
    def _adapt_parameters(self) -> None:
        """Adapt reward parameters based on recent performance."""
        if len(self.returns_history) < 50:
            return
        
        recent_returns = np.array(self.returns_history[-50:])
        
        # Analyze performance
        mean_return = np.mean(recent_returns)
        volatility = np.std(recent_returns)
        sharpe = mean_return / (volatility + 1e-8)
        
        # Adapt drawdown penalty based on recent volatility
        if volatility > 0.05:  # High volatility
            self.params['drawdown_penalty'] = min(self.params['drawdown_penalty'] * 1.1, 5.0)
        elif volatility < 0.01:  # Low volatility
            self.params['drawdown_penalty'] = max(self.params['drawdown_penalty'] * 0.9, 0.1)
        
        # Adapt profit weight based on Sharpe ratio
        if sharpe > 1.0:  # Good performance
            self.params['profit_weight'] = min(self.params['profit_weight'] * 1.05, 2.0)
        elif sharpe < -0.5:  # Poor performance
            self.params['profit_weight'] = max(self.params['profit_weight'] * 0.95, 0.5)
        
        logger.info(f"Adapted reward parameters: profit_weight={self.params['profit_weight']:.3f}, "
                   f"drawdown_penalty={self.params['drawdown_penalty']:.3f}")