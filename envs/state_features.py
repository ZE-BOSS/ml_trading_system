"""
State Feature Extractor
=======================

This module extracts and processes features for the trading environment state.
It handles technical indicators, price features, and time-based features.

Features include:
- Price data (OHLCV)
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- Time features (hour, day of week, etc.)
- Normalized and scaled features

Author: PPO Trading System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from loguru import logger
import pandas_ta as ta


class StateFeatureExtractor:
    """Extracts and processes features for the trading environment state."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration dictionary from model_config.yaml
        """
        self.config = config
        self.feature_config = config['environment']['features']
        
        # Feature scaling parameters (learned during first extraction)
        self.feature_stats = {}
        self.is_fitted = False
        
        logger.info("State feature extractor initialized")
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from the input data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with extracted features
        """
        logger.info("Extracting features from market data")
        
        features_df = data.copy()
        
        # Add technical indicators
        features_df = self._add_technical_indicators(features_df)
        
        # Add time features
        features_df = self._add_time_features(features_df)
        
        # Normalize features
        features_df = self._normalize_features(features_df)
        
        # Select only the configured features
        features_df = self._select_features(features_df)
        
        logger.info(f"Feature extraction complete. Shape: {features_df.shape}")
        
        logger.debug(f"Feature Data: {features_df}")
        return features_df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        indicators = self.feature_config.get('technical_indicators', [])
        
        for indicator in indicators:
            try:
                if ta:
                    # Use pandas_ta if available
                    if indicator == 'sma_5':
                        df['sma_5'] = ta.sma(df['close'], length=5)
                    elif indicator == 'sma_20':
                        df['sma_20'] = ta.sma(df['close'], length=20)
                    elif indicator == 'sma_50':
                        df['sma_50'] = ta.sma(df['close'], length=50)
                    elif indicator == 'ema_12':
                        df['ema_12'] = ta.ema(df['close'], length=12)
                    elif indicator == 'ema_26':
                        df['ema_26'] = ta.ema(df['close'], length=26)
                    elif indicator == 'rsi_14':
                        df['rsi_14'] = ta.rsi(df['close'], length=14)
                    elif indicator == 'macd':
                        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
                        df['macd'] = macd_df['MACD_12_26_9']
                        df['macd_signal'] = macd_df['MACDs_12_26_9']
                        df['macd_hist'] = macd_df['MACDh_12_26_9']
                    elif indicator in ['bb_upper', 'bb_lower']:
                        bbands = ta.bbands(df['close'], length=20, std=2)
                        df['bb_upper'] = bbands['BBU_20_2.0']
                        df['bb_middle'] = bbands['BBM_20_2.0']
                        df['bb_lower'] = bbands['BBL_20_2.0']
                    elif indicator == 'atr_14':
                        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
                    elif indicator == 'stoch_k':
                        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
                        df['stoch_k'] = stoch['STOCHk_14_3_3']
                        df['stoch_d'] = stoch['STOCHd_14_3_3']
                    elif indicator == 'williams_r':
                        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
                    elif indicator == 'cci_14':
                        df['cci_14'] = ta.cci(df['high'], df['low'], df['close'], length=14)
                    elif indicator == 'adx_14':
                        df['adx_14'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
                    elif indicator == 'obv':
                        df['obv'] = ta.obv(df['close'], df['volume'])
                else:
                    # Fallback implementations using pandas
                    if indicator == 'sma_5':
                        df['sma_5'] = df['close'].rolling(window=5).mean()
                    elif indicator == 'sma_20':
                        df['sma_20'] = df['close'].rolling(window=20).mean()
                    elif indicator == 'sma_50':
                        df['sma_50'] = df['close'].rolling(window=50).mean()
                    elif indicator == 'ema_12':
                        df['ema_12'] = df['close'].ewm(span=12).mean()
                    elif indicator == 'ema_26':
                        df['ema_26'] = df['close'].ewm(span=26).mean()
                    elif indicator == 'rsi_14':
                        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
                    elif indicator == 'macd':
                        macd, signal = self._calculate_macd(df['close'])
                        df['macd'] = macd
                        df['macd_signal'] = signal
                    elif indicator in ['bb_upper', 'bb_lower']:
                        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
                        df['bb_upper'] = bb_upper
                        df['bb_middle'] = bb_middle
                        df['bb_lower'] = bb_lower
                    elif indicator == 'atr_14':
                        df['atr_14'] = self._calculate_atr(df['high'], df['low'], df['close'], 14)
                    
            except Exception as e:
                logger.warning(f"Failed to calculate indicator {indicator}: {e}")
        
        # Add price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        if ta:
            df['volume_sma'] = ta.sma(df['volume'], length=20)
        else:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility features
        df['volatility'] = df['price_change'].rolling(window=20).std()
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI using pandas fallback."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD using pandas fallback."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> tuple:
        """Calculate Bollinger Bands using pandas fallback."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate ATR using pandas fallback."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = data.copy()
        
        time_features = self.feature_config.get('time_features', [])
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            # common timestamp column names to try
            ts_candidates = [c for c in ('timestamp', 'time', 'date', 'datetime') if c in df.columns]
            coerced = False
            for col in ts_candidates:
                try:
                    df.index = pd.to_datetime(df[col])
                    coerced = True
                    break
                except Exception:
                    continue

            if not coerced:
                # Try to coerce the existing index
                try:
                    df.index = pd.to_datetime(df.index)
                    coerced = True
                except Exception:
                    coerced = False

            if not coerced:
                logger.warning("Could not coerce a DatetimeIndex from data. "
                            "Time features will be omitted. Provide a DatetimeIndex or a 'timestamp' column.")
                
        for feature in time_features:
            try:
                if feature == 'hour_of_day':
                    df['hour_of_day'] = df.index.hour
                elif feature == 'day_of_week':
                    df['day_of_week'] = df.index.dayofweek
                elif feature == 'day_of_month':
                    df['day_of_month'] = df.index.day
                elif feature == 'month_of_year':
                    df['month_of_year'] = df.index.month
                elif feature == 'is_weekend':
                    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
                elif feature == 'is_month_end':
                    df['is_month_end'] = df.index.is_month_end.astype(int)
                elif feature == 'is_quarter_end':
                    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
                    
            except Exception as e:
                logger.warning(f"Failed to calculate time feature {feature}: {e}")
        
        # Cyclical encoding for time features
        if 'hour_of_day' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        
        if 'day_of_week' in df.columns:
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using z-score normalization."""
        df = data.copy()
        
        # Features that should not be normalized (already in good range)
        skip_normalization = ['is_weekend', 'is_month_end', 'is_quarter_end', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        
        # Features that should be normalized
        normalize_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                            if col not in skip_normalization]
        
        if not self.is_fitted:
            # Calculate statistics on first run
            self.feature_stats = {}
            for feature in normalize_features:
                if feature in df.columns:
                    values = df[feature].dropna()
                    if len(values) > 0:
                        self.feature_stats[feature] = {
                            'mean': values.mean(),
                            'std': values.std() if values.std() > 0 else 1.0
                        }
            self.is_fitted = True
            logger.info(f"Feature normalization parameters fitted for {len(self.feature_stats)} features")
        
        # Apply normalization
        for feature, stats in self.feature_stats.items():
            if feature in df.columns:
                df[feature] = (df[feature] - stats['mean']) / stats['std']
        
        return df
    
    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select only the configured features."""
        # Get all configured features
        price_features = self.feature_config.get('price_features', [])
        technical_indicators = self.feature_config.get('technical_indicators', [])
        time_features = self.feature_config.get('time_features', [])
        
        # Add derived features
        derived_features = ['price_change', 'high_low_ratio', 'volume_ratio', 'volatility', 'price_position']
        
        # Do NOT add cyclical_features manually, they are already created in _add_time_features
        all_features = (price_features + technical_indicators + time_features + derived_features)
        
        # Deduplicate while preserving order
        all_features = list(dict.fromkeys(all_features))
        
        # Select only existing features
        existing_features = [f for f in all_features if f in data.columns]
        
        if not existing_features:
            logger.warning("No configured features found in data, using all numeric columns")
            existing_features = list(data.select_dtypes(include=[np.number]).columns)
        
        logger.info(f"Selected {len(existing_features)} features: {existing_features}")
        logger.debug(f"All Selected Features: {all_features}")
        return data[existing_features]
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        Calculate feature importance if the model supports it.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(feature_names, importances))
            else:
                logger.warning("Model does not support feature importance calculation")
                return {}
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def reset_normalization(self) -> None:
        """Reset normalization parameters (useful for retraining)."""
        self.feature_stats = {}
        self.is_fitted = False
        logger.info("Feature normalization parameters reset")
