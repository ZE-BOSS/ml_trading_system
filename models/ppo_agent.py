import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

from loguru import logger

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.trading_env import TradingEnvironment


# Filenames for saved sidecar artifacts
VECNORM_FILENAME = "vecnormalize.pkl"
OBS_CONFIG_FILENAME = "obs_config.yaml"
POLICY_KWARGS_FILENAME = "policy_kwargs.yaml"


class TradingFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, portfolio_features_dim: int = 5):
        super().__init__(observation_space, features_dim=256)
        self.portfolio_features_dim = portfolio_features_dim

        total_obs_dim = int(observation_space.shape[0])
        market_dim = total_obs_dim - self.portfolio_features_dim
        if market_dim <= 0:
            raise ValueError("Observation space too small for configured portfolio_features_dim")

        self.market_extractor = nn.Sequential(
            nn.Linear(market_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.portfolio_extractor = nn.Sequential(
            nn.Linear(self.portfolio_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

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
    def __init__(self, websocket_broadcaster=None, performance_tracker=None, verbose: int = 1):
        super().__init__(verbose)
        self.websocket_broadcaster = websocket_broadcaster
        self.performance_tracker = performance_tracker

        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []

        self.best_mean_reward = -np.inf
        self.best_model_path = None

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        if infos and isinstance(infos, (list, tuple)) and len(infos) > 0:
            info = infos[0]
            if isinstance(info, dict) and 'portfolio_value' in info:
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
                    try:
                        log_dir = getattr(self.model.logger, "dir", None)
                        if log_dir:
                            self.best_model_path = os.path.join(log_dir, 'best_model')
                            self.model.save(self.best_model_path)
                            logger.info(f"New best model saved with portfolio value: {mean_portfolio_value:.2f}")
                    except Exception as e:
                        logger.warning(f"Could not save best model automatically: {e}")

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

        self.model: Optional[PPO] = None
        self.env: Optional[gym.Env] = None
        self.vec_env: Optional[VecNormalize] = None

        # Track policy kwargs used and a serializable snapshot
        self._policy_kwargs_used: Optional[Dict[str, Any]] = None
        self._policy_kwargs_serializable: Optional[Dict[str, Any]] = None

        logger.info("PPO Trading Agent initialized")

    @staticmethod
    def _read_yaml_if_exists(path: str) -> Optional[Dict[str, Any]]:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to read yaml at {path}: {e}")
        return None

    @staticmethod
    def _ensure_vec_env(env: Optional[gym.Env]) -> Optional[VecEnv]:
        if env is None:
            return None

        if isinstance(env, VecEnv):
            return env

        base = env
        try:
            underlying = getattr(base, "env", None)
            if underlying is not None and isinstance(underlying, gym.Env):
                base = base
        except Exception:
            pass

        try:
            monitored = base if isinstance(base, Monitor) else Monitor(base)
            return DummyVecEnv([lambda: monitored])
        except Exception as exc:
            raise ValueError(f"Failed to wrap provided env into VecEnv: {exc}")

    @staticmethod
    def _unwrap_env_for_obs_signature(env: Optional[gym.Env]):
        if env is None:
            return None

        if isinstance(env, DummyVecEnv):
            try:
                inner = getattr(env, "envs", None)
                if inner and len(inner) > 0:
                    return inner[0]
            except Exception:
                pass

        try:
            inner = getattr(env, "env", None)
            if inner is not None:
                return inner
        except Exception:
            pass

        return env

    def _map_activation_fn(self, act):
        if isinstance(act, str):
            act_map = {
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "sigmoid": nn.Sigmoid,
                "leaky_relu": nn.LeakyReLU
            }
            return act_map.get(act.lower(), nn.ReLU)
        if isinstance(act, type) and issubclass(act, nn.Module):
            return act
        return nn.ReLU

    def _serialize_policy_kwargs(self, pk: Dict[str, Any]) -> Dict[str, Any]:
        serial = {}
        serial['net_arch'] = pk.get('net_arch')
        act = pk.get('activation_fn')
        if isinstance(act, type):
            serial['activation_fn'] = act.__name__.lower()
        else:
            serial['activation_fn'] = str(act).lower() if act is not None else None

        fe = pk.get('features_extractor_class')
        serial['features_extractor_class'] = fe.__name__ if hasattr(fe, '__name__') else str(fe)
        serial['features_extractor_kwargs'] = pk.get('features_extractor_kwargs', {})
        serial['ortho_init'] = pk.get('ortho_init', None)

        other = {k: v for k, v in pk.items() if k not in ['net_arch', 'activation_fn', 'features_extractor_class', 'features_extractor_kwargs', 'ortho_init']}
        serial['other'] = other
        serial['version'] = 1
        return serial

    def _deserialize_policy_kwargs(self, serialized: Dict[str, Any]) -> Dict[str, Any]:
        if not serialized:
            return {}
        pk: Dict[str, Any] = {}
        if 'net_arch' in serialized and serialized['net_arch'] is not None:
            pk['net_arch'] = serialized['net_arch']
        act_name = serialized.get('activation_fn')
        if act_name:
            pk['activation_fn'] = self._map_activation_fn(act_name)
        fe_name = serialized.get('features_extractor_class')
        if fe_name and fe_name.lower().startswith('tradingfeatureextractor'):
            pk['features_extractor_class'] = TradingFeatureExtractor
        pk['features_extractor_kwargs'] = serialized.get('features_extractor_kwargs', {}) or {}
        if 'ortho_init' in serialized:
            pk['ortho_init'] = serialized.get('ortho_init')
        other = serialized.get('other', {}) or {}
        pk.update(other)
        return pk

    def _build_eval_vec_env(self, eval_data: pd.DataFrame, model_dir: Optional[str] = None) -> VecNormalize:
        expected_signature = None
        if model_dir:
            sig_path = os.path.join(model_dir, OBS_CONFIG_FILENAME)
            expected_signature = self._read_yaml_if_exists(sig_path)

        eval_env = TradingEnvironment(
            data=eval_data,
            config_path="config/model_config.yaml",
            **({"expected_signature": expected_signature} if expected_signature is not None else {})
        )
        eval_env = Monitor(eval_env)
        eval_vec = DummyVecEnv([lambda: eval_env])

        eval_vec_norm = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, training=False)

        if isinstance(self.vec_env, VecNormalize) and getattr(self.vec_env, "obs_rms", None) is not None:
            eval_vec_norm.obs_rms = self.vec_env.obs_rms

        return eval_vec_norm

    def create_model(self, train_data: pd.DataFrame, model_path: Optional[str] = None) -> None:
        self.env = TradingEnvironment(
            data=train_data,
            config_path="config/model_config.yaml"
        )
        self.env = Monitor(self.env)
        train_vec = DummyVecEnv([lambda: self.env])
        self.vec_env = VecNormalize(train_vec, norm_obs=True, norm_reward=True)

        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")

            model_dir = os.path.dirname(model_path)
            policy_kwargs_path = os.path.join(model_dir, POLICY_KWARGS_FILENAME)
            saved_policy_kwargs = self._read_yaml_if_exists(policy_kwargs_path)

            custom_objects = {"learning_rate": self.model_params.get('learning_rate')}
            if saved_policy_kwargs:
                try:
                    deserialized = self._deserialize_policy_kwargs(saved_policy_kwargs)
                    # Ensure feature extractor present
                    deserialized.setdefault("features_extractor_class", TradingFeatureExtractor)
                    deserialized.setdefault("features_extractor_kwargs", {"portfolio_features_dim": 5})
                    custom_objects["policy_kwargs"] = deserialized
                except Exception as e:
                    logger.warning(f"Failed to deserialize saved policy kwargs: {e}")

            try:
                self.model = PPO.load(model_path, env=self.vec_env, custom_objects=custom_objects)
                # store policy kwargs used if available
                self._policy_kwargs_used = getattr(self.model, 'policy_kwargs', custom_objects.get('policy_kwargs'))
                if self._policy_kwargs_used:
                    self._policy_kwargs_serializable = self._serialize_policy_kwargs(self._policy_kwargs_used)
                logger.info("PPO model loaded successfully for continued training")
                return
            except Exception as e:
                logger.error(f"Failed to load model for continued training: {e}")
                raise

        logger.info("Creating new PPO model")

        policy_kwargs = dict(self.model_params.get("policy_kwargs", {}) or {})

        act_fn = policy_kwargs.get("activation_fn")
        if isinstance(act_fn, str):
            policy_kwargs["activation_fn"] = self._map_activation_fn(act_fn)

        if self.model_params.get("use_custom_policy", True):
            extractor_kwargs = policy_kwargs.pop('features_extractor_kwargs', {}) or {}
            extractor_kwargs.update({'portfolio_features_dim': 5})
            policy_kwargs.update({
                'features_extractor_class': TradingFeatureExtractor,
                'features_extractor_kwargs': extractor_kwargs
            })

        # store used policy kwargs for later serialization
        self._policy_kwargs_used = policy_kwargs
        self._policy_kwargs_serializable = self._serialize_policy_kwargs(policy_kwargs)

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

    def train(self, train_data: pd.DataFrame, eval_data: Optional[pd.DataFrame] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        if self.model is None:
            self.create_model(train_data)

        logger.info(f"Starting PPO training with {len(train_data)} samples")

        callbacks = [
            TradingCallback(
                websocket_broadcaster=self.websocket_broadcaster,
                performance_tracker=self.performance_tracker
            ),
            CheckpointCallback(
                save_freq=self.training_params.get('save_freq', 5000),
                save_path='./models/saved_models/checkpoints/',
                name_prefix='ppo_trading'
            )
        ]

        if eval_data is not None:
            eval_vec_norm = self._build_eval_vec_env(eval_data)
            callbacks.append(EvalCallback(
                eval_vec_norm,
                eval_freq=self.training_params.get('eval_freq', 10000),
                n_eval_episodes=self.training_params.get('n_eval_episodes', 5),
                best_model_save_path='./models/saved_models/',
                log_path='./logs/eval/',
                verbose=1
            ))

        total_timesteps = self.training_params['total_timesteps']

        try:
            self.model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

            model_path = f"./models/saved_models/ppo_final_{session_id or 'default'}.zip"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)

            try:
                vecnorm_path = os.path.join(os.path.dirname(model_path), VECNORM_FILENAME)
                if isinstance(self.vec_env, VecNormalize):
                    self.vec_env.save(vecnorm_path)
                    logger.info(f"Saved VecNormalize stats to {vecnorm_path}")
            except Exception as e:
                logger.warning(f"Could not save VecNormalize stats: {e}")

            try:
                base_env = self._unwrap_env_for_obs_signature(self.env)
                obs_sig_fn = getattr(base_env, "obs_signature", None)
                if callable(obs_sig_fn):
                    obs_sig = obs_sig_fn()
                    with open(os.path.join(os.path.dirname(model_path), OBS_CONFIG_FILENAME), "w") as f:
                        yaml.safe_dump(obs_sig, f)
            except Exception as e:
                logger.warning(f"Could not save {OBS_CONFIG_FILENAME}: {e}")

            try:
                # prefer serializable snapshot created earlier
                policy_to_save = self._policy_kwargs_serializable or self._serialize_policy_kwargs(self.config['ppo'].get('policy_kwargs', {}))
                with open(os.path.join(os.path.dirname(model_path), POLICY_KWARGS_FILENAME), "w") as f:
                    yaml.safe_dump(policy_to_save, f)
            except Exception as e:
                logger.warning(f"Could not save {POLICY_KWARGS_FILENAME}: {e}")

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

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.model is None:
            raise ValueError("Model not loaded. Call create_model() or train() first.")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action, None

    def evaluate(self, eval_data: pd.DataFrame, n_episodes: int = 10) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not loaded. Call create_model() or train() first.")

        model_dir = None
        eval_vec_norm = self._build_eval_vec_env(eval_data, model_dir=model_dir)

        episode_returns, episode_lengths, win_rates, max_drawdowns = [], [], [], []

        for episode in range(n_episodes):
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
        if self.model is None:
            raise ValueError("No model to save")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self.model.save(path)

        if isinstance(self.vec_env, VecNormalize):
            try:
                self.vec_env.save(os.path.join(os.path.dirname(path) or ".", VECNORM_FILENAME))
            except Exception as e:
                logger.warning(f"Could not save VecNormalize stats on save_model: {e}")

        try:
            base_env = self._unwrap_env_for_obs_signature(self.env)
            obs_sig_fn = getattr(base_env, "obs_signature", None)
            if callable(obs_sig_fn):
                obs_sig = obs_sig_fn()
                with open(os.path.join(os.path.dirname(path) or ".", OBS_CONFIG_FILENAME), "w") as f:
                    yaml.safe_dump(obs_sig, f)
        except Exception as e:
            logger.warning(f"Could not save obs_config.yaml: {e}")

        try:
            policy_to_save = self._policy_kwargs_serializable or self._serialize_policy_kwargs(self.config['ppo'].get('policy_kwargs', {}))
            with open(os.path.join(os.path.dirname(path) or ".", POLICY_KWARGS_FILENAME), "w") as f:
                yaml.safe_dump(policy_to_save, f)
        except Exception as e:
            logger.warning(f"Could not save policy_kwargs.yaml: {e}")

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str, env: Optional[gym.Env] = None) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        model_dir = os.path.dirname(path)

        policy_kwargs_path = os.path.join(model_dir, POLICY_KWARGS_FILENAME)
        policy_kwargs_serial = self._read_yaml_if_exists(policy_kwargs_path) or {}
        # deserialize into runtime objects
        try:
            policy_kwargs = self._deserialize_policy_kwargs(policy_kwargs_serial)
        except Exception as e:
            logger.warning(f"Failed to deserialize policy kwargs from file: {e}")
            policy_kwargs = {}

        policy_kwargs.setdefault("features_extractor_class", TradingFeatureExtractor)
        policy_kwargs.setdefault("features_extractor_kwargs", {"portfolio_features_dim": 5})

        vecnorm_path = os.path.join(model_dir, VECNORM_FILENAME)
        if os.path.exists(vecnorm_path):
            if env is None:
                raise ValueError(
                    f"Model appears to use VecNormalize (found {VECNORM_FILENAME}). Provide a real environment with matching observation signature when loading."
                )
            env_vec = self._ensure_vec_env(env)

            obs_sig_path = os.path.join(model_dir, OBS_CONFIG_FILENAME)
            expected_sig = self._read_yaml_if_exists(obs_sig_path)
            if expected_sig is not None:
                underlying = self._unwrap_env_for_obs_signature(env)
                obs_sig_fn = getattr(underlying, "obs_signature", None)
                if callable(obs_sig_fn):
                    current_sig = obs_sig_fn()
                    if int(current_sig.get("obs_dim", -1)) != int(expected_sig.get("obs_dim", -1)):
                        raise RuntimeError(
                            f"Saved model obs_dim={expected_sig.get('obs_dim')} does not match provided env obs_dim={current_sig.get('obs_dim')}. Create an environment with matching features/lookback or retrain the model."
                        )
            self.vec_env = VecNormalize.load(vecnorm_path, env_vec)
            self.vec_env.training = False
            self.vec_env.norm_reward = False
            env = self.vec_env
            logger.info(f"Loaded VecNormalize stats from {vecnorm_path}")
        elif env is not None:
            self.vec_env = env

        try:
            self.model = PPO.load(path, env=env, custom_objects={"policy_kwargs": policy_kwargs})
            # record policy kwargs used
            self._policy_kwargs_used = getattr(self.model, 'policy_kwargs', policy_kwargs)
            if self._policy_kwargs_used:
                self._policy_kwargs_serializable = self._serialize_policy_kwargs(self._policy_kwargs_used)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model due to: {e}")
            raise
