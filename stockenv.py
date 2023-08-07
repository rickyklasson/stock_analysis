import gymnasium as gym
import numpy as np
import pandas as pd
import random
from gymnasium import spaces
from pathlib import Path

NBR_ACTIONS = 3  # BUY, SELL, PASS = 1, -1, 0
OBS_WINDOW = 60


class StockEnv(gym.Env):
    def __init__(self, files: list[Path]):
        super(StockEnv).__init__()
        self.balance = 0.0  # Running balance during a trading period. Used as reward.
        self.files = random.sample(files, k=len(files))  # Files to use during training.
        self.file_idx = 0  # Index of current file in list self.files.
        self.timestamp_idx = OBS_WINDOW  # Index of timestamp in current file.
        self.df = None  # Current dataframe used for training.
        self.max_share_price = 0.0
        self.max_volume = 0.0
        self.nbr_shares = 0
        self.price = 0.0
        self.volume = 0

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(NBR_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=1, shape=(OBS_WINDOW * 2,), dtype=np.float64)

    def _get_info(self, action):
        return {
            'action': action,
            'balance': self.balance,
            'price': self.price,
            'volume': self.volume,
            'file': str(self.files[self.file_idx]),
            'max_share_price': self.max_share_price,
            'max_volume': self.max_volume,
            'timestamp_idx': self.timestamp_idx,
        }

    def _get_observation(self):
        close = np.array(self.df.loc[self.timestamp_idx - OBS_WINDOW: self.timestamp_idx - 1,
                         'close'].values / self.max_share_price)
        volume = np.array(self.df.loc[self.timestamp_idx - OBS_WINDOW: self.timestamp_idx - 1,
                          'volume'].values / self.max_volume)

        obs = np.concatenate((close, volume))

        return obs

    def _act(self, action):
        # Update price and previous price.
        self.price = self.df.loc[self.timestamp_idx, 'close']
        self.volume = self.df.loc[self.timestamp_idx, 'volume']
        prev_price = self.df.loc[self.timestamp_idx - 1, 'close']

        # Update balance based on currently held shares and the price difference from previous sample.
        sample_diff = (self.price - prev_price) / self.max_share_price
        self.balance += self.nbr_shares * sample_diff

        if action == 2:
            # SELL
            self.nbr_shares -= 1
        elif action == 1:
            # BUY
            self.nbr_shares += 1

    def _next_state(self) -> bool:
        self.timestamp_idx += 1

        return self.timestamp_idx == len(self.df.index)

    def step(self, action):
        self._act(action)

        info = self._get_info(action)
        terminated = self._next_state()
        obs = self._get_observation()
        reward = self.balance

        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        self.balance = 0.0
        self.timestamp_idx = OBS_WINDOW
        self.file_idx += 1
        self.df = pd.read_csv(self.files[self.file_idx])
        self.max_share_price = max(self.df['close'])
        self.max_volume = max(self.df['volume'])
        self.nbr_shares = 0

        obs = self._get_observation()
        info = self._get_info(0)

        return obs, info
