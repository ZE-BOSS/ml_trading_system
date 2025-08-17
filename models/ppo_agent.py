"""
PPO Agent Implementation
========================

This module implements the Proximal Policy Optimization (PPO) agent for trading.
It uses Stable-Baselines3 for the core PPO implementation with custom modifications
for trading environments.

Features:
- Custom feature extractor for market and portfolio inputs
- Trading-specific observation preprocessing
- Model checkpointing and loading
- Performance monitoring and callbacks

Author: PPO Trading System
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from loguru import logger
import yaml

from envs.trading_env import TradingEnvironment


class TradingFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that splits observations into market and portfolio parts,
    embeds them separately, then combines them into a fixed-size feature vector.
    """

    def __init__(self, observation_space: gym.Space, portfolio_features_dim: int = 5):
        super().__init__(observation_space, features_dim=256)
        self.portfolio_features_dim = portfolio_features_dim

        total_obs_dim = int(observation_space.shape[0])
        market_dim = total_obs_dim - self.portfolio_features_dim

        # Market pathway
        self.market_extractor = nn.Sequential(
            nn.Linear(market_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Portfolio pathway
        self.portfolio_extractor = nn.Sequential(
            nn.Linear(self.portfolio_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Fuse pathways
        self.combined_extractor = nn.Sequential(
            nn.Linear(128 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        market_features = obs[:, :-self.portfolio_features_dim]
        portfolio_features = obs[:, -self.portfolio_features_dim:]
        market_embed = self.market_extractor(market_features)
        portfolio_embed = self.portfolio_extractor(portfolio_features)
        combined = torch.cat([market_embed, portfolio_embed], dim=1)
        return self.combined_extractor(combined)


class TradingCallback(BaseCallback):
    """Custom callback for training monitoring and early stopping."""

    def __init__(self,
                 websocket_broadcaster=None,
                 performance_tracker=None,
                 verbose: int = 1):
        super().__init__(verbose)
        self.websocket_broadcaster = websocket_broadcaster
        self.performance_tracker = performance_tracker

        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []

        self.best_mean_reward = -np.inf
        self.best_model_path = None

    def _on_step(self) -> bool:
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'portfolio_value' in info:
                self.portfolio_values.append(info['portfolio_value'])
                if self.websocket_broadcaster and self.num_timesteps % 100 == 0:
                    self._broadcast_training_update(info)
        return True

    def _on_rollout_end(self) -> None:
        if len(self.portfolio_values) > 0:
            mean_portfolio_value = np.mean(self.portfolio_values[-100:])
            if mean_portfolio_value > self.best_mean_reward:
                self.best_mean_reward = mean_portfolio_value
                if hasattr(self.model, 'save'):
                    self.best_model_path = os.path.join(self.model.logger.dir, 'best_model')
                    self.model.save(self.best_model_path)
                    logger.info(f"New best model saved with portfolio value: {mean_portfolio_value:.2f}")

    def _broadcast_training_update(self, info: Dict) -> None:
        try:
            update_data = {
                'type': 'training_update',
                'timestep': self.num_timesteps,
                'portfolio_value': info.get('portfolio_value', 0),
                'total_return': info.get('total_return', 0),
                'win_rate': info.get('win_rate', 0),
                'max_drawdown': info.get('max_drawdown', 0),
                'total_trades': info.get('total_trades', 0)
            }
            if self.websocket_broadcaster:
                self.websocket_broadcaster.broadcast_training_update(update_data)
        except Exception as e:
            logger.warning(f"Failed to broadcast training update: {e}")


class PPOTradingAgent:
    """
    Main PPO trading agent class that handles training, evaluation, and inference.
    Uses SB3's MlpPolicy with a custom features extractor that is aware of portfolio
    features at the tail of the observation vector.
    """

    def __init__(self,
                 config_path: str = "config/model_config.yaml",
                 websocket_broadcaster=None,
                 performance_tracker=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.websocket_broadcaster = websocket_broadcaster
        self.performance_tracker = performance_tracker

        self.model_params = self.config['ppo']
        self.training_params = self.config['training']

        self.model = None
        self.env = None
        self.vec_env: Optional[VecNormalize] = None

        logger.info("PPO Trading Agent initialized")

    def _build_eval_vec_env(self, eval_data: pd.DataFrame) -> VecNormalize:
        """
        Build an evaluation VecEnv wrapped exactly like the training env,
        and copy over normalization statistics so obs are normalized identically.
        """
        eval_env = TradingEnvironment(data=eval_data, config_path="config/model_config.yaml")
        eval_env = Monitor(eval_env)
        eval_vec = DummyVecEnv([lambda: eval_env])
        # Do not update stats during evaluation
        eval_vec_norm = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, training=False)
        if isinstance(self.vec_env, VecNormalize):
            # Reuse the obs normalization stats from training
            eval_vec_norm.obs_rms = self.vec_env.obs_rms
        else:
            logger.warning("Training env is not VecNormalize; proceeding without shared normalization stats.")
        return eval_vec_norm

    def create_model(self,
                     train_data: pd.DataFrame,
                     model_path: Optional[str] = None) -> None:
        """Create or load the PPO model."""
        self.env = TradingEnvironment(
            data=train_data,
            config_path="config/model_config.yaml"
        )

        self.env = Monitor(self.env)
        train_vec = DummyVecEnv([lambda: self.env])
        self.vec_env = VecNormalize(train_vec, norm_obs=True, norm_reward=True)

        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            self.model = PPO.load(
                model_path,
                env=self.vec_env,
                custom_objects={'learning_rate': self.model_params['learning_rate']}
            )
            logger.info("PPO model created successfully")
            return

        logger.info("Creating new PPO model")

        # Always use MlpPolicy, inject custom feature extractor when enabled
        # Make a shallow copy to avoid mutating the YAML-loaded dict
        policy_kwargs = dict(self.model_params.get("policy_kwargs", {}))

        # Auto-fix activation_fn if string
        act_fn = policy_kwargs.get("activation_fn")
        if isinstance(act_fn, str):
            act_map = {
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "sigmoid": nn.Sigmoid,
                "leaky_relu": nn.LeakyReLU
            }
            mapped = act_map.get(act_fn.lower())
            if mapped is None:
                logger.warning(f"Unknown activation_fn '{act_fn}', defaulting to ReLU")
                mapped = nn.ReLU
            policy_kwargs["activation_fn"] = mapped

        if self.model_params.get("use_custom_policy", True):
            extractor_kwargs = policy_kwargs.pop('features_extractor_kwargs', {}) or {}
            extractor_kwargs.update({'portfolio_features_dim': 5})
            policy_kwargs.update({
                'features_extractor_class': TradingFeatureExtractor,
                'features_extractor_kwargs': extractor_kwargs
            })

        self.model = PPO(
            policy='MlpPolicy',
            env=self.vec_env,
            learning_rate=self.model_params['learning_rate'],
            n_steps=self.model_params['n_steps'],
            batch_size=self.model_params['batch_size'],
            n_epochs=self.model_params['n_epochs'],
            gamma=self.model_params['gamma'],
            gae_lambda=self.model_params['gae_lambda'],
            clip_range=self.model_params['clip_range'],
            ent_coef=self.model_params['ent_coef'],
            vf_coef=self.model_params['vf_coef'],
            max_grad_norm=self.model_params['max_grad_norm'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            device='auto'
        )

        logger.info("PPO model created successfully")

    def train(self,
              train_data: pd.DataFrame,
              eval_data: Optional[pd.DataFrame] = None,
              session_id: Optional[str] = None) -> Dict[str, Any]:
        """Train the PPO agent."""
        if self.model is None:
            self.create_model(train_data)

        logger.info(f"Starting PPO training with {len(train_data)} samples")

        callbacks = [
            TradingCallback(
                websocket_broadcaster=self.websocket_broadcaster,
                performance_tracker=self.performance_tracker
            ),
            CheckpointCallback(
                save_freq=self.training_params['save_freq'],
                save_path='./models/saved_models/checkpoints/',
                name_prefix='ppo_trading'
            )
        ]

        # Ensure eval env is wrapped exactly like training (VecNormalize) and shares stats
        if eval_data is not None:
            eval_vec_norm = self._build_eval_vec_env(eval_data)
            callbacks.append(EvalCallback(
                eval_vec_norm,
                eval_freq=self.training_params['eval_freq'],
                n_eval_episodes=self.training_params['n_eval_episodes'],
                best_model_save_path='./models/saved_models/',
                log_path='./logs/eval/',
                verbose=1
            ))

        total_timesteps = self.training_params['total_timesteps']

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )

            # Save final model + normalization stats
            model_path = f"./models/saved_models/ppo_final_{session_id or 'default'}.zip"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)

            # Save VecNormalize statistics for later inference/eval consistency
            try:
                vecnorm_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
                if isinstance(self.vec_env, VecNormalize):
                    self.vec_env.save(vecnorm_path)
                    logger.info(f"Saved VecNormalize stats to {vecnorm_path}")
            except Exception as e:
                logger.warning(f"Could not save VecNormalize stats: {e}")

            logger.info(f"Training completed. Model saved to {model_path}")

            training_cb = callbacks[0]
            return {
                'success': True,
                'model_path': model_path,
                'total_timesteps': total_timesteps,
                'best_model_path': getattr(training_cb, 'best_model_path', None),
                'final_performance': {
                    'mean_reward': training_cb.best_mean_reward,
                    'total_episodes': len(training_cb.episode_rewards)
                }
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'success': False, 'error': str(e)}

    def predict(self,
                observation: np.ndarray,
                deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make prediction using trained model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call create_model() or train() first.")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action, None

    def evaluate(self,
                 eval_data: pd.DataFrame,
                 n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call create_model() or train() first.")

        # Build eval VecEnv that mirrors training + reuse normalization stats
        eval_vec_norm = self._build_eval_vec_env(eval_data)

        episode_returns, episode_lengths, win_rates, max_drawdowns = [], [], [], []

        for episode in range(n_episodes):
            # VecEnv reset: returns obs only
            obs = eval_vec_norm.reset()
            done = [False]
            ep_ret = 0.0
            ep_len = 0
            last_info: Dict[str, Any] = {}

            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, done, infos = eval_vec_norm.step(action)
                ep_ret += float(rewards[0])
                ep_len += 1
                # infos is a list of dicts (one per env); keep last non-empty
                if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
                    last_info = infos[0] or last_info

            episode_returns.append(ep_ret)
            episode_lengths.append(ep_len)
            win_rates.append(last_info.get('win_rate', 0.0))
            max_drawdowns.append(last_info.get('max_drawdown', 0.0))

            logger.info(f"Evaluation episode {episode + 1}/{n_episodes}: Return={ep_ret:.2f}, Length={ep_len}")

        return {
            'mean_return': float(np.mean(episode_returns)),
            'std_return': float(np.std(episode_returns)),
            'mean_length': float(np.mean(episode_lengths)),
            'mean_win_rate': float(np.mean(win_rates)),
            'mean_max_drawdown': float(np.mean(max_drawdowns)),
            'sharpe_ratio': float(np.mean(episode_returns) / (np.std(episode_returns) + 1e-8))
        }

    def save_model(self, path: str) -> None:
        """Save the current model."""
        if self.model is None:
            raise ValueError("No model to save")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)
        # Also save VecNormalize stats if available
        if isinstance(self.vec_env, VecNormalize):
            try:
                self.vec_env.save(os.path.join(os.path.dirname(path) or ".", "vecnormalize.pkl"))
            except Exception as e:
                logger.warning(f"Could not save VecNormalize stats on save_model: {e}")
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load a saved model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        # Note: caller should set/create env + VecNormalize before predict/learn
        self.model = PPO.load(path)
        logger.info(f"Model loaded from {path}")
