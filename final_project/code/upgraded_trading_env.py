import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class UpgradedTradingEnv(gym.Env):
    """
    Upgraded Trading Environment (Final Project Requirements A & B)
    - Upgrade A: Richer state features (rolling volatility and 50-day MA distance)
    - Upgrade B: 5 position-sizing actions (-1.0, -0.5, 0.0, 0.5, 1.0)
    """
    def __init__(
        self,
        returns,
        episode_length=100,
        history_len=5,
        transaction_cost=0.1,
        reward_scale=100.0,
    ):
        super().__init__()
        self.returns = np.asarray(returns, dtype=np.float64)
        assert len(self.returns) > max(50, episode_length + 1), "Not enough data for 50-day MA features."
        
        self.episode_length = episode_length
        self.history_len = history_len
        self.transaction_cost = transaction_cost
        self.reward_scale = reward_scale
        
        # Precompute the advanced features globally to avoid runtime overhead
        # Feature 1: Rolling Volatility (20-day window)
        # Feature 2: Distance from 50-day Moving Average (cumulative log return trend)
        self.rolling_vol = np.zeros_like(self.returns)
        self.ma_dist = np.zeros_like(self.returns)
        
        cum_log_prices = np.cumsum(self.returns) # Simulate a structural price track
        
        for i in range(len(self.returns)):
            # Volatility (strict historical lookback)
            start_v = max(0, i - 19)
            self.rolling_vol[i] = np.std(self.returns[start_v:i+1]) if i > 0 else 0.0
            
            # Moving average distance (strict historical lookback)
            start_m = max(0, i - 49)
            ma = np.mean(cum_log_prices[start_m:i+1])
            self.ma_dist[i] = cum_log_prices[i] - ma

        # Obs dim = history_len (returns) + 1 (vol) + 1 (ma_dist) + 1 (current position) = history_len + 3
        obs_dim = self.history_len + 3
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
        )
        
        # Upgrade B: 5 actions -> 0: full long, 1: half long, 2: flat, 3: half short, 4: full short
        self.action_space = spaces.Discrete(5)
        self.action_to_position = {0: 1.0, 1: 0.5, 2: 0.0, 3: -0.5, 4: -1.0}
        
        self.position = None
        self.step_count = None
        self._start = None
        self._price_history = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Ensure we start late enough that rolling features are fully populated
        last_valid_start = len(self.returns) - self.episode_length - 1
        self._start = int(self.np_random.integers(50, last_valid_start))
        
        self.position = 0.0
        self.step_count = 0
        self._price_history = deque([0.0] * self.history_len, maxlen=self.history_len)
        return self._get_obs(), {}

    def step(self, action):
        current_idx = self._start + self.step_count
        price_return = float(self.returns[current_idx])
        self._price_history.append(price_return)
        
        new_position = self.action_to_position[int(action)]
        
        pnl = self.position * price_return * self.reward_scale
        cost = self.transaction_cost * abs(new_position - self.position)
        reward = pnl - cost
        
        self.position = new_position
        self.step_count += 1
        
        terminated = self.step_count >= self.episode_length
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        current_idx = self._start + self.step_count - 1 if self.step_count > 0 else self._start
        history = np.array(list(self._price_history), dtype=np.float32)
        
        vol_feat = np.array([self.rolling_vol[current_idx]], dtype=np.float32)
        ma_feat = np.array([self.ma_dist[current_idx]], dtype=np.float32)
        pos_feat = np.array([self.position], dtype=np.float32)
        
        return np.concatenate([history, vol_feat, ma_feat, pos_feat]).astype(np.float32)