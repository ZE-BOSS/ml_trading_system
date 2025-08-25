"""
PPO Trading Environment
=======================

Custom Gymnasium environment tailored for training and evaluating PPO agents
on financial time series (OHLCV) data.

This implementation emphasizes robustness and reproducibility by:
 - validating input data early and clearly
 - exposing an observation signature (obs_signature) that can be persisted and
   used by model save/load utilities to detect feature/lookback drift
 - providing clear, documented behavior for action/observation spaces
 - separating feature extraction and reward calculation (pluggable)
 - ensuring deterministic shapes (padding the lookback window) so observation
   vectors are stable across episodes

Design notes / rationale (deep thinking summary):
 - The agent, VecNormalize, and saved policy weights must agree on the
   observation dimensionality. To make that explicit we provide obs_signature()
   which returns (lookback_window, n_features, obs_dim, feature_names, ...).
   The model-saving logic should persist this signature (obs_config.yaml)
   and the loading logic should validate it against the environment used for
   evaluation/backtesting. This prevents silent shape mismatches and cryptic
   PyTorch "size mismatch" errors.
 - The environment expects a dataframe with standard OHLCV columns. Any
   technical indicator generation happens in StateFeatureExtractor (pluggable),
   which returns processed_data with columns corresponding to features used
   by the policy.
 - We avoid creating "fake" or empty environments for model loading: if the
   caller needs to load vecnormalize stats they must provide a real env that
   matches the obs_signature saved with the model.
 - Observation layout (flattened) is: [ market_window_features_flattened, portfolio_features ],
   where portfolio_features has fixed length = 5. This ordering must be honored
   by the agent's feature extractor (TradingFeatureExtractor).

Author: PPO Trading System
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from loguru import logger
import yaml
from pathlib import Path

from .state_features import StateFeatureExtractor
from .reward_functions import RewardCalculator


class TradingEnvironment(gym.Env):
    """
    High-level behavior:
    - The environment maintains a processed_data DataFrame produced by the
    StateFeatureExtractor. processed_data must have no NaNs and supply at least
    `lookback_window` rows for the environment to operate.
    - At each step the agent receives a flattened observation consisting of:
        [ flattened lookback_window x n_features market block, 5 portfolio stats ]
    The last 5 entries are the portfolio features:
        [ normalized_balance, position_size, normalized_unrealized_pnl, normalized_total_trades, current_drawdown ]
    - Action space can be discrete (e.g. Hold/Buy/Sell) or continuous (target position).
    - The environment computes rewards using a pluggable RewardCalculator.
    - The environment exposes obs_signature() for persistence/validation.
    """

    METADATA = {"render_modes": ["human"]}

    def __init__(self,
                data: pd.DataFrame,
                config_path: str = "config/model_config.yaml",
                initial_balance: float = 10_000.0,
                transaction_cost: float = 0.0001,
                render_mode: Optional[str] = None,
                expected_signature: Optional[Dict[str, Any]] = None):
        super().__init__()

        # load main config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # copy input
        self.raw_data = data.copy() if data is not None else pd.DataFrame()
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.render_mode = render_mode

        # environment params from config as defaults
        env_cfg = self.config.get("environment", {})
        self.action_config: Dict[str, Any] = env_cfg.get("action_space", {"type": "discrete", "actions": ["hold", "buy", "sell"]})
        self.reward_config: Dict[str, Any] = env_cfg.get("reward_function", {})

        # backtest config
        self.backtest_cfg: Dict[str, Any] = self.config.get("backtesting", {})

        # feature extractor and reward calculator
        self.feature_extractor = StateFeatureExtractor(self.config)
        self.reward_calculator = RewardCalculator(self.config)

        # expected signature for strict validation if provided
        self._expected_signature = expected_signature

        # load persisted observation signature if present
        obs_config_path = Path("models/saved_models/obs_config.yaml")
        if obs_config_path.exists():
            with open(obs_config_path, "r") as f:
                obs_config = yaml.safe_load(f) or {}
        else:
            obs_config = {}

        # persistent observation params, fall back to config-derived defaults
        self.feature_names: List[str] = obs_config.get("feature_names", [])
        self.n_features: int = int(obs_config.get("n_features", len(self.feature_names))) if obs_config else 0
        self.lookback_window: int = int(obs_config.get("lookback_window", env_cfg.get("lookback_window", 100)))
        self.portfolio_features_dim: int = int(obs_config.get("portfolio_features_dim", 5))
        # compute obs_dim if present, otherwise compute from lookback and n_features (n_features may be 0 until features extracted)
        self.obs_dim: int = int(obs_config.get("obs_dim", self.lookback_window * max(1, self.n_features) + self.portfolio_features_dim))

        # prepare data and validate against obs_config when available
        self._prepare_data()

        # if feature names were not provided by obs_config, derive them now
        if not self.feature_names:
            self.feature_names = list(self.processed_data.columns)
            self.n_features = int(len(self.feature_names))
            self.obs_dim = int(self.lookback_window * self.n_features + self.portfolio_features_dim)

        # build action and observation spaces
        self._setup_spaces()

        # reset internal state
        self.reset()

        logger.info(f"Trading environment initialized with {len(self.raw_data)} data points")

    # observation signature
    def obs_signature(self) -> Dict[str, Any]:
        return {
            "lookback_window": int(self.lookback_window),
            "n_features": int(self.n_features),
            "feature_names": list(self.feature_names),
            "portfolio_features_dim": int(self.portfolio_features_dim),
            "obs_dim": int(self.obs_dim)
        }

    # data preparation and validation
    def _prepare_data(self) -> None:
        required_columns = ["open", "high", "low", "close", "volume"]
        if not isinstance(self.raw_data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")

        missing = [c for c in required_columns if c not in self.raw_data.columns]
        if missing:
            raise ValueError(f"Data must contain columns: {required_columns}. Missing: {missing}")

        # extract features
        self.processed_data = self.feature_extractor.extract_features(self.raw_data)

        # drop NaNs from indicator warmup
        self.processed_data = self.processed_data.dropna(axis=0, how="any").reset_index(drop=True)

        if len(self.processed_data) < self.lookback_window:
            raise ValueError(f"Insufficient data after feature extraction: need at least lookback_window ({self.lookback_window}) rows, got {len(self.processed_data)}")

        # if obs_config provided feature_names, ensure processed_data contains them
        if self.feature_names:
            missing_feats = [f for f in self.feature_names if f not in self.processed_data.columns]
            if missing_feats:
                raise ValueError(f"Processed data missing features required by obs_config.yaml: {missing_feats}")
            # ensure n_features matches
            self.n_features = int(len(self.feature_names))
        else:
            # derive feature names from processed_data
            self.feature_names = list(self.processed_data.columns)
            self.n_features = int(len(self.feature_names))

        # recompute obs_dim from final values
        self.obs_dim = int(self.lookback_window * self.n_features + self.portfolio_features_dim)

        # validate against expected signature if supplied
        if self._expected_signature:
            if int(self._expected_signature.get("obs_dim", -1)) != int(self.obs_dim):
                raise ValueError(
                    f"Observation signature mismatch. Expected obs_dim={self._expected_signature.get('obs_dim')}, "
                    f"but environment produces obs_dim={self.obs_dim}"
                )

        logger.info(f"Data prepared: {len(self.processed_data)} samples, {len(self.processed_data.columns)} features")

    # spaces setup
    def _setup_spaces(self) -> None:
        # action space
        if self.action_config.get("type", "discrete") == "discrete":
            actions = list(self.action_config.get("actions", ["hold", "buy", "sell"]))
            self.action_space = spaces.Discrete(len(actions))
        else:
            bounds = self.action_config.get("continuous_bounds", {"position_size": [-1.0, 1.0]})
            low = float(bounds["position_size"][0])
            high = float(bounds["position_size"][1])
            self.action_space = spaces.Box(
                low=np.array([low], dtype=np.float32),
                high=np.array([high], dtype=np.float32),
                dtype=np.float32,
            )

        # observation space
        obs_dim = int(self.lookback_window * self.n_features + self.portfolio_features_dim)
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        logger.info(f"Action space: {self.action_space}")
        logger.info(f"Observation space: {self.observation_space.shape}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step: int = int(self.lookback_window)
        self.balance: float = float(self.initial_balance)
        self.position: float = 0.0
        self.position_entry_price: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.trade_history: List[Dict[str, Any]] = []
        self.portfolio_values: List[float] = [self.initial_balance]
        self.unrealized_pnl: float = 0.0

        self.max_portfolio_value: float = self.initial_balance
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._execute_action(action)
        self.current_step += 1

        reward = self.reward_calculator.calculate_reward(
            current_price=self._get_current_price(),
            position=self.position,
            balance=self.balance,
            unrealized_pnl=self.unrealized_pnl,
            drawdown=self.current_drawdown,
            portfolio_value=self._get_portfolio_value()
        )

        self._update_portfolio_tracking()

        terminated = (self.current_step >= (len(self.processed_data) - 1))
        truncated = self._check_early_termination()

        obs = self._get_observation()
        info = self._get_info()
        return obs, float(reward), bool(terminated), bool(truncated), info

    # actions
    def _execute_action(self, action: Any) -> None:
        current_price = self._get_current_price()

        if self.action_config.get("type", "discrete") == "discrete":
            if not isinstance(action, (int, np.integer)):
                if isinstance(action, (list, tuple, np.ndarray)):
                    action = int(np.asarray(action).flatten()[0])
                else:
                    action = int(action)

            actions = self.action_config.get("actions", ["hold", "buy", "sell"])
            idx = int(action) % len(actions)
            action_name = actions[idx]

            if action_name == "buy" and self.position <= 0:
                self._open_position(1.0, current_price)
            elif action_name == "sell" and self.position >= 0:
                self._open_position(-1.0, current_price)
            elif action_name == "hold":
                if self.position != 0.0:
                    self.unrealized_pnl = (current_price - self.position_entry_price) * self.position
        else:
            if isinstance(action, (list, tuple, np.ndarray)):
                target_position = float(np.asarray(action).flatten()[0])
            else:
                target_position = float(action)
            if not math.isfinite(target_position):
                return
            if target_position != self.position:
                self._adjust_position(target_position, current_price)

    def _open_position(self, direction: float, price: float) -> None:
        if self.position != 0.0:
            self._close_position(price)

        sizing = self.backtest_cfg.get("position_sizing", {"method": "fixed_fraction", "fraction": 0.01, "max_position": 0.1})

        if sizing.get("method", "fixed_fraction") == "fixed_fraction":
            fraction = float(sizing.get("fraction", 0.01))
            position_value = self.balance * fraction
            position_size = (position_value / price) * float(direction)
        elif sizing.get("method") == "fixed":
            position_size = float(sizing.get("size", 0.01)) * float(direction)
        else:
            position_size = 0.01 * float(direction)

        max_pos = float(sizing.get("max_position", 0.1))
        max_pos_val = self.balance * max_pos
        max_pos_size = max_pos_val / price if price > 0 else max_pos_val
        if abs(position_size) > max_pos_size:
            position_size = math.copysign(max_pos_size, position_size)

        trade_cost = abs(position_size) * price * self.transaction_cost

        if self.balance >= trade_cost:
            self.position = float(position_size)
            self.position_entry_price = float(price)
            self.balance -= float(trade_cost)
            self.total_trades += 1
            self.unrealized_pnl = 0.0

            self.trade_history.append({
                "step": self.current_step,
                "action": "BUY" if direction > 0 else "SELL",
                "size": self.position,
                "price": price,
                "cost": trade_cost,
                "balance_after": self.balance
            })

    def _close_position(self, price: float) -> None:
        if self.position == 0.0:
            return

        pnl = (price - self.position_entry_price) * self.position
        trade_cost = abs(self.position) * price * self.transaction_cost
        net_pnl = pnl - trade_cost

        self.balance += self.position * price + net_pnl

        if net_pnl > 0:
            self.winning_trades += 1

        self.trade_history.append({
            "step": self.current_step,
            "action": "CLOSE",
            "size": self.position,
            "entry_price": self.position_entry_price,
            "exit_price": price,
            "pnl": net_pnl,
            "balance_after": self.balance
        })

        self.position = 0.0
        self.position_entry_price = 0.0
        self.unrealized_pnl = 0.0

    def _adjust_position(self, target_position: float, price: float) -> None:
        if target_position == self.position:
            return

        delta = target_position - self.position
        trade_cost = abs(delta) * price * self.transaction_cost
        if self.balance >= trade_cost:
            if self.position == 0.0 and target_position != 0.0:
                self.position_entry_price = price
            elif self.position != 0.0 and target_position != 0.0:
                prev_val = self.position * self.position_entry_price
                added_val = delta * price
                new_pos = self.position + delta
                if new_pos != 0:
                    self.position_entry_price = (prev_val + added_val) / new_pos
                else:
                    self.position_entry_price = 0.0

            self.position = float(target_position)
            self.balance -= float(trade_cost)
            self.total_trades += 1

    # pricing and portfolio helpers
    def _get_current_price(self) -> float:
        try:
            return float(self.processed_data.iloc[self.current_step]["close"])
        except Exception:
            return float(self.processed_data.iloc[self.current_step].iloc[-1])

    def _get_portfolio_value(self) -> float:
        current_price = self._get_current_price()
        position_value = self.position * current_price if self.position != 0.0 else 0.0
        return float(self.balance + position_value)

    def _update_portfolio_tracking(self) -> None:
        pv = self._get_portfolio_value()
        self.portfolio_values.append(pv)

        if pv > self.max_portfolio_value:
            self.max_portfolio_value = pv

        if self.max_portfolio_value > 0:
            self.current_drawdown = (self.max_portfolio_value - pv) / float(self.max_portfolio_value)
        else:
            self.current_drawdown = 0.0

        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown

    def _check_early_termination(self) -> bool:
        risk_cfg = self.backtest_cfg.get("risk_management", {})
        max_drawdown_stop = float(risk_cfg.get("max_drawdown_stop", 0.5))
        if self.current_drawdown > max_drawdown_stop:
            return True

        if self.balance < (0.1 * self.initial_balance):
            return True

        return False

    # observation and info
    def _get_observation(self) -> np.ndarray:
        start_idx = max(0, self.current_step - self.lookback_window)

        # select features in the order defined by feature_names
        df = self.processed_data
        if all(f in df.columns for f in self.feature_names):
            window = df[self.feature_names].iloc[start_idx:self.current_step]
            n_cols = len(self.feature_names)
        else:
            # fallback to all columns if mismatch
            window = df.iloc[start_idx:self.current_step]
            n_cols = window.shape[1]

        # pad if needed
        if len(window) < self.lookback_window:
            n_missing = self.lookback_window - len(window)
            pad_shape = (n_missing, n_cols)
            pad = np.zeros(pad_shape, dtype=np.float32)
            window_vals = np.vstack([pad, window.values.astype(np.float32)])
        else:
            window_vals = window.values.astype(np.float32)

        flat_market = window_vals.flatten()

        portfolio_features = np.array([
            self.balance / float(self.initial_balance),
            self.position,
            (self.unrealized_pnl / float(self.initial_balance)) if self.initial_balance != 0 else 0.0,
            float(self.total_trades) / 1000.0,
            float(self.current_drawdown)
        ], dtype=np.float32)

        obs = np.concatenate([flat_market.astype(np.float32), portfolio_features], axis=0)

        # ensure obs length matches expected obs_dim, pad or trim if necessary to keep deterministic shape
        expected_len = int(self.observation_space.shape[0]) if hasattr(self, "observation_space") else int(self.obs_dim)
        if obs.shape[0] < expected_len:
            pad_len = expected_len - obs.shape[0]
            obs = np.concatenate([obs, np.zeros(pad_len, dtype=np.float32)], axis=0)
        elif obs.shape[0] > expected_len:
            obs = obs[:expected_len]

        return obs

    def _get_info(self) -> Dict[str, Any]:
        pv = self._get_portfolio_value()
        win_rate = (float(self.winning_trades) / float(max(1, self.total_trades))) if self.total_trades > 0 else 0.0
        return {
            "step": int(self.current_step),
            "balance": float(self.balance),
            "position": float(self.position),
            "portfolio_value": float(pv),
            "total_return": float((pv - self.initial_balance) / float(self.initial_balance)),
            "total_trades": int(self.total_trades),
            "winning_trades": int(self.winning_trades),
            "win_rate": float(win_rate),
            "max_drawdown": float(self.max_drawdown),
            "current_drawdown": float(self.current_drawdown),
            "unrealized_pnl": float(self.unrealized_pnl)
        }

    def render(self) -> None:
        if self.render_mode == "human":
            info = self._get_info()
            print(
                f"[Step {info['step']}] Balance: ${info['balance']:.2f} | "
                f"Portfolio: ${info['portfolio_value']:.2f} | Return: {info['total_return']:.2%} | "
                f"Drawdown: {info['current_drawdown']:.2%} | Position: {info['position']:.4f}"
            )

    @classmethod
    def build_for_model_loading(cls,
                                data: pd.DataFrame,
                                config_path: str,
                                expected_signature: Dict[str, Any]) -> "TradingEnvironment":
        env = cls(data=data, config_path=config_path, expected_signature=expected_signature)
        sig = env.obs_signature()
        if sig.get("obs_dim") != expected_signature.get("obs_dim"):
            raise ValueError(
                f"obs_dim mismatch building env_for_model_loading: expected {expected_signature.get('obs_dim')}, got {sig.get('obs_dim')}"
            )
        return env
