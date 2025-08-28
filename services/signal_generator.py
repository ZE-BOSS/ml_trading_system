"""
Signal Generator
================

This module generates trading signals from PPO model predictions and
formats them for execution. It includes signal filtering, risk management,
and position sizing logic.

Features:
- PPO model inference
- Signal confidence scoring  
- Risk-based position sizing
- Signal filtering and validation
- Stop-loss and take-profit calculation

Author: PPO Trading System
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import uuid
from loguru import logger
import yaml
import torch

from models.ppo_agent import PPOTradingAgent
from services.mt5_client import TradeSignal, MarketTick
from envs.state_features import StateFeatureExtractor


@dataclass
class TradingSignal:
    """Enhanced trading signal with confidence and risk metrics."""
    id: str
    timestamp: datetime
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    lot_size: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    expected_return: Optional[float] = None
    risk_score: Optional[float] = None
    model_version: Optional[str] = None
    features_used: Optional[List[str]] = None
    
    def to_mt5_signal(self) -> TradeSignal:
        """Convert to MT5 trade signal format."""
        if self.action == 'HOLD':
            return None
        
        return TradeSignal(
            symbol=self.symbol,
            action=self.action,
            lot_size=self.lot_size,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            comment=f"PPO_Signal_{self.id[:8]}",
            magic_number=234000
        )


class SignalGenerator:
    """
    Generates trading signals from PPO model predictions.
    
    This class takes market data, runs it through the trained PPO model,
    and generates formatted trading signals with proper risk management.
    """
    
    def __init__(self, 
                 config_path: str = "config/model_config.yaml",
                 model_path: Optional[str] = None):
        """
        Initialize signal generator.
        
        Args:
            config_path: Path to model configuration
            model_path: Path to trained PPO model
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.signal_config = self.config.get('signal_generation', {})
        self.risk_config = self.config['backtesting']['risk_management']
        self.position_config = self.config['backtesting']['position_sizing']
        
        # Initialize components
        self.ppo_agent = PPOTradingAgent(config_path)
        self.feature_extractor = StateFeatureExtractor(self.config)
        
        # Load trained model if provided
        if model_path:
            self.load_model(model_path)
        
        # Signal filtering parameters
        self.min_confidence = self.signal_config.get('min_confidence', 0.6)
        self.max_signals_per_day = self.signal_config.get('max_signals_per_day', 10)
        self.signal_cooldown_minutes = self.signal_config.get('cooldown_minutes', 30)
        
        # State tracking
        self.last_signals = {}  # symbol -> last signal time
        self.daily_signal_count = {}  # date -> count
        self.market_state = {}  # symbol -> current market state
        
        logger.info("Signal generator initialized")
    
    def load_model(self, model_path: str) -> bool:
        """
        Load trained PPO model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            True if model loaded successfully
        """
        try:
            self.ppo_agent.load_model(model_path)
            logger.info(f"PPO model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate_signal(self, 
                       symbol: str,
                       market_data: pd.DataFrame,
                       current_tick: Optional[MarketTick] = None,
                       account_info: Optional[Dict] = None) -> Optional[TradingSignal]:
        """
        Generate trading signal for given market data.
        
        Args:
            symbol: Trading symbol
            market_data: Historical market data with technical indicators
            current_tick: Current market tick
            account_info: Current account information
            
        Returns:
            Trading signal or None if no signal generated
        """
        try:
            # Check if we can generate a signal for this symbol
            if not self._can_generate_signal(symbol):
                return None
            
            # Prepare features
            features = self._prepare_features(market_data)
            if features is None:
                logger.warning(f"Failed to prepare features for {symbol}")
                return None
            
            # Get model prediction
            action, confidence = self._get_model_prediction(features)
            if action is None:
                return None
            
            # Check confidence threshold
            if confidence < self.min_confidence:
                logger.debug(f"Signal confidence {confidence:.3f} below threshold {self.min_confidence}")
                return None
            
            # Calculate position sizing
            lot_size = self._calculate_position_size(
                symbol, confidence, account_info, current_tick
            )
            
            if lot_size <= 0:
                logger.debug(f"Position size calculation resulted in zero for {symbol}")
                return None
            
            # Calculate entry price
            entry_price = current_tick.mid_price if current_tick else market_data['close'].iloc[-1]
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_stop_take(
                symbol, action, entry_price, market_data, confidence
            )
            
            # Calculate expected return and risk
            expected_return = self._calculate_expected_return(
                action, entry_price, take_profit, stop_loss, confidence
            )
            risk_score = self._calculate_risk_score(symbol, market_data, confidence)
            
            # Create signal
            signal = TradingSignal(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                confidence=confidence,
                lot_size=lot_size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                expected_return=expected_return,
                risk_score=risk_score,
                model_version=self._get_model_version(),
                features_used=list(features.columns) if hasattr(features, 'columns') else None
            )
            
            # Update tracking
            self._update_signal_tracking(symbol, signal)
            
            logger.info(f"Generated signal: {action} {symbol} "
                       f"(confidence: {confidence:.3f}, size: {lot_size})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def generate_signals_batch(self, 
                              market_data_dict: Dict[str, pd.DataFrame],
                              current_ticks: Optional[Dict[str, MarketTick]] = None,
                              account_info: Optional[Dict] = None) -> List[TradingSignal]:
        """
        Generate signals for multiple symbols in batch.
        
        Args:
            market_data_dict: Dictionary mapping symbol to market data
            current_ticks: Current ticks for each symbol
            account_info: Account information
            
        Returns:
            List of generated trading signals
        """
        signals = []
        
        for symbol, market_data in market_data_dict.items():
            current_tick = current_ticks.get(symbol) if current_ticks else None
            
            signal = self.generate_signal(
                symbol, market_data, current_tick, account_info
            )
            
            if signal:
                signals.append(signal)
        
        # Sort by confidence (highest first)
        signals.sort(key=lambda s: s.confidence, reverse=True)
        
        logger.info(f"Generated {len(signals)} signals from {len(market_data_dict)} symbols")
        return signals
    
    def _can_generate_signal(self, symbol: str) -> bool:
        """Check if we can generate a signal for this symbol."""
        now = datetime.now()
        today = now.date()
        
        # Check daily signal limit
        if today in self.daily_signal_count:
            if self.daily_signal_count[today] >= self.max_signals_per_day:
                return False
        
        # Check cooldown period
        if symbol in self.last_signals:
            last_signal_time = self.last_signals[symbol]
            time_diff = now - last_signal_time
            if time_diff.total_seconds() < self.signal_cooldown_minutes * 60:
                return False
        
        return True
    
    def _prepare_features(self, market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for model prediction with enhanced NaN handling."""
        try:
            # Extract features
            features_df = self.feature_extractor.extract_features(market_data)
            
            # Enhanced NaN detection and logging
            nan_count = features_df.isnull().sum().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in features before processing")
                # Log which columns have NaN values
                nan_columns = features_df.columns[features_df.isnull().any()].tolist()
                logger.warning(f"Columns with NaN values: {nan_columns}")
                
                # Fill NaN values with appropriate methods
                for col in features_df.columns:
                    if features_df[col].isnull().any():
                        # Use forward fill, then backward fill for time series data
                        features_df[col] = features_df[col].fillna(method='ffill').fillna(method='bfill')
                        # If still NaN, fill with mean
                        if features_df[col].isnull().any():
                            features_df[col] = features_df[col].fillna(features_df[col].mean())
            
            # Check for infinite values
            inf_count = np.isinf(features_df.values).sum()
            if inf_count > 0:
                logger.warning(f"Found {inf_count} infinite values in features")
                features_df = features_df.replace([np.inf, -np.inf], np.nan)
                features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            # Normalize features
            features_normalized = self.feature_extractor._normalize_features(features_df)
            
            # Check again after normalization
            if np.any(np.isnan(features_normalized.values)) or np.any(np.isinf(features_normalized.values)):
                logger.error("NaN or Inf values still present after normalization")
                return None
            
            # Rest of the feature preparation logic
            lookback_window = self.config['environment']['lookback_window']
            if len(features_normalized) < lookback_window:
                logger.warning("Insufficient data for feature extraction")
                return None
            
            recent_features = features_normalized.iloc[-lookback_window:].values
            observation = recent_features.flatten()
            
            # Add portfolio features
            portfolio_features = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
            full_observation = np.concatenate([observation, portfolio_features])
            
            # Check final observation for issues
            if np.any(np.isnan(full_observation)) or np.any(np.isinf(full_observation)):
                logger.error("Final observation contains NaN or Inf values")
                return None
            
            # Check if values are within reasonable range
            if np.any(np.abs(full_observation) > 100):
                logger.warning(f"Observation values are too large: min={full_observation.min()}, max={full_observation.max()}")
                # Clip extreme values to prevent model issues
                full_observation = np.clip(full_observation, -10, 10)
            
            logger.info(f"Observation range: [{full_observation.min():.4f}, {full_observation.max():.4f}]")
            
            return full_observation.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
        
    def _get_model_prediction(self, features: np.ndarray) -> Tuple[Optional[str], float]:
        """Get model prediction and confidence."""
        try:
            if self.ppo_agent.model is None:
                logger.error("No model loaded")
                return None, 0.0
            
            # Get action from model
            action_idx, _ = self.ppo_agent.predict(features, deterministic=True)
            
            # Map action index to action name
            action_names = self.config['environment']['action_space']['actions']
            action = action_names[action_idx[0]]
            
            # Calculate confidence (simplified - could be enhanced with actual model probabilities)
            # For now, use a heuristic based on action consistency over recent predictions
            confidence = self._calculate_action_confidence(features, action_idx[0])
            
            return action, confidence
            
        except Exception as e:
            logger.error(f"Error getting model prediction: {e}")
            return None, 0.0
    
    def _calculate_action_confidence(self, features: np.ndarray, action_idx: int) -> float:
        """Calculate confidence score for the predicted action."""
        try:
            # Try to get the actual action probabilities from the model
            if self.ppo_agent.model is not None:
                # Get the model's policy
                policy = self.ppo_agent.model.policy
                
                # Convert features to tensor with the correct data type
                obs_tensor = torch.as_tensor(features, dtype=torch.float32)
                
                # Move to the same device as the policy
                device = next(policy.parameters()).device
                obs_tensor = obs_tensor.to(device)
                
                # Get action distribution
                with torch.no_grad():
                    latent_pi, _, _ = policy.forward(obs_tensor)
                    distribution = policy._get_action_dist_from_latent(latent_pi)
                    
                    # Get probabilities for all actions
                    if hasattr(distribution, 'distribution'):
                        probs = distribution.distribution.probs.cpu().numpy()
                        
                        # Use the probability of the selected action as confidence
                        action_prob = probs[0, action_idx]
                        
                        # Base confidence on the action probability
                        base_confidence = float(action_prob)
                    else:
                        # Fallback if we can't get probabilities
                        base_confidence = 0.7
            else:
                base_confidence = 0.7
            
            # Adjust confidence based on market volatility (lower volatility = higher confidence)
            # Assuming volatility is the 26th feature (index 25)
            volatility = abs(features[0, 25]) if features.shape[1] > 25 else 0
            volatility_factor = 1.0 / (1.0 + volatility)
            
            # Adjust confidence based on trend strength (stronger trend = higher confidence for trend-following actions)
            # Assuming price_change is the 23rd feature (index 22)
            trend_strength = abs(features[0, 22]) if features.shape[1] > 22 else 0
            trend_factor = 1.0 + (0.5 * trend_strength)  # Up to 50% boost for strong trends
            
            # Combine factors
            confidence = base_confidence * volatility_factor * trend_factor
            
            # Ensure confidence is within reasonable bounds
            confidence = np.clip(confidence, 0.1, 0.95)
            
            logger.debug(f"Confidence calculation: base={base_confidence:.3f}, "
                        f"vol_factor={volatility_factor:.3f}, trend_factor={trend_factor:.3f}, "
                        f"final={confidence:.3f}")
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            # Fallback confidence calculation based on observation quality
            try:
                # Check if observation values are reasonable
                obs_range = np.max(features) - np.min(features)
                if obs_range > 100:  # Very large range indicates issues
                    return 0.3  # Low confidence for problematic observations
                else:
                    return 0.7  # Medium confidence as fallback
            except:
                return 0.5  # Default fallback
            
    def _calculate_position_size(self, 
                                symbol: str,
                                confidence: float,
                                account_info: Optional[Dict],
                                current_tick: Optional[MarketTick]) -> float:
        """Calculate position size based on risk management rules."""
        try:
            if not account_info or not current_tick:
                return 0.01  # Default minimum size
            
            method = self.position_config['method']
            
            if method == 'fixed':
                return self.position_config.get('size', 0.01)
            
            elif method == 'fixed_fraction':
                fraction = self.position_config['fraction']
                balance = account_info.get('balance', 10000)
                position_value = balance * fraction
                lot_size = position_value / current_tick.mid_price / 100000  # Assuming standard lot
                
                # Adjust by confidence
                lot_size *= confidence
                
                # Apply limits
                max_position = self.position_config.get('max_position', 0.1)
                max_lot_value = balance * max_position
                max_lot_size = max_lot_value / current_tick.mid_price / 100000
                
                return min(lot_size, max_lot_size, 1.0)  # Cap at 1 lot
            
            elif method == 'volatility_target':
                # Implement volatility-based sizing
                return 0.01  # Placeholder
            
            else:
                return 0.01
                
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01
    
    def _calculate_stop_take(self, 
                           symbol: str,
                           action: str,
                           entry_price: float,
                           market_data: pd.DataFrame,
                           confidence: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""
        try:
            if action == 'HOLD':
                return None, None
            
            # Calculate ATR for volatility-based stops
            if 'atr_14' in market_data.columns:
                atr = market_data['atr_14'].iloc[-1]
                if pd.isna(atr) or atr <= 0:
                    # Fallback to price-based calculation
                    recent_prices = market_data['close'].tail(14)
                    atr = recent_prices.std()
            else:
                # Fallback to simple price-based calculation
                recent_prices = market_data['close'].tail(14)
                atr = recent_prices.std()
            
            # Ensure ATR is valid
            if pd.isna(atr) or atr <= 0:
                atr = entry_price * 0.01  # 1% fallback
            
            # Base stop and target distances
            stop_distance = atr * 2.0  # 2x ATR stop
            take_distance = atr * 3.0  # 3x ATR target (1.5:1 reward/risk ratio)
            
            # Adjust based on confidence
            stop_distance *= (1 / confidence)  # Tighter stops for higher confidence
            take_distance *= confidence      # More aggressive targets for higher confidence
            
            if action == 'BUY':
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + take_distance
            else:  # SELL
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - take_distance
            
            # Ensure stops are reasonable (not negative prices)
            stop_loss = max(stop_loss, entry_price * 0.01) if action == 'BUY' else stop_loss
            take_profit = max(take_profit, entry_price * 0.01) if action == 'SELL' else take_profit
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating stop/take levels: {e}")
            return None, None
    
    def _calculate_expected_return(self, 
                                 action: str,
                                 entry_price: float,
                                 take_profit: Optional[float],
                                 stop_loss: Optional[float],
                                 confidence: float) -> Optional[float]:
        """Calculate expected return for the signal."""
        try:
            if action == 'HOLD' or not take_profit or not stop_loss:
                return None
            
            if action == 'BUY':
                profit = take_profit - entry_price
                loss = entry_price - stop_loss
            else:  # SELL
                profit = entry_price - take_profit
                loss = stop_loss - entry_price
            
            # Simple expected return calculation
            # Assuming win probability based on confidence
            win_prob = confidence
            loss_prob = 1 - confidence
            
            expected_return = (win_prob * profit) - (loss_prob * loss)
            return expected_return / entry_price  # Return as percentage
            
        except Exception as e:
            logger.error(f"Error calculating expected return: {e}")
            return None
    
    def _calculate_risk_score(self, 
                            symbol: str,
                            market_data: pd.DataFrame,
                            confidence: float) -> float:
        """Calculate risk score for the signal."""
        try:
            # Base risk score from volatility
            if 'volatility' in market_data.columns:
                volatility = market_data['volatility'].iloc[-1]
            else:
                recent_returns = market_data['close'].pct_change().tail(20)
                volatility = recent_returns.std()
            
            # Normalize volatility to 0-1 scale
            risk_score = min(volatility * 100, 1.0)
            
            # Adjust by confidence (lower confidence = higher risk)
            risk_score *= (1 / confidence)
            
            return np.clip(risk_score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    def _get_model_version(self) -> str:
        """Get model version identifier."""
        # This could be enhanced to track actual model versions
        return "PPO_v1.0"
    
    def _update_signal_tracking(self, symbol: str, signal: TradingSignal) -> None:
        """Update signal tracking for rate limiting."""
        now = datetime.now()
        today = now.date()
        
        # Update last signal time for symbol
        self.last_signals[symbol] = now
        
        # Update daily signal count
        if today not in self.daily_signal_count:
            self.daily_signal_count[today] = 0
        self.daily_signal_count[today] += 1
        
        # Clean old tracking data
        self._cleanup_tracking_data()
    
    def _cleanup_tracking_data(self) -> None:
        """Clean up old tracking data to prevent memory issues."""
        now = datetime.now()
        cutoff_date = now.date() - timedelta(days=7)
        
        # Remove old daily counts
        self.daily_signal_count = {
            date: count for date, count in self.daily_signal_count.items()
            if date >= cutoff_date
        }
        
        # Remove old signal times (older than 24 hours)
        cutoff_time = now - timedelta(hours=24)
        self.last_signals = {
            symbol: time for symbol, time in self.last_signals.items()
            if time >= cutoff_time
        }
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get signal generation statistics."""
        today = datetime.now().date()
        
        return {
            'signals_today': self.daily_signal_count.get(today, 0),
            'max_signals_per_day': self.max_signals_per_day,
            'active_symbols': len(self.last_signals),
            'min_confidence_threshold': self.min_confidence,
            'cooldown_minutes': self.signal_cooldown_minutes
        }