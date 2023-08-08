import gymnasium as gym
import numpy as np
import pandas as pd
import random

from finta import TA
from gymnasium import spaces
from pathlib import Path

from gymnasium.core import RenderFrame

NBR_ACTIONS = 2  # BUY, SELL = 0, 1
OBS_WINDOW = 10


class StockEnv(gym.Env):
    def __init__(self, files: list[Path]):
        super(StockEnv).__init__()
        self.balance = 0.0  # Running balance during a trading period. Used as reward.
        self.files = random.sample(files, k=len(files))  # Shuffle files used during training.
        self.file_idx = 0  # Index of current file in list self.files.
        self.timestamp_idx = OBS_WINDOW  # Index of timestamp in current file.
        self.df = None  # Current dataframe used for training.
        self.nbr_shares = 0
        self.price = 0.0
        self.volume = 0

        self.action_space = spaces.Discrete(NBR_ACTIONS)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_WINDOW, 4), dtype=np.float64)

    def _get_info(self, action, reward):
        return {
            'action': action,
            'price': self.price,
            'reward': reward,
            'balance': self.balance,
            'volume': self.volume,
            'file': str(self.files[self.file_idx]),
            'timestamp_idx': self.timestamp_idx,
        }

    def _get_observation(self):
        obs_df = self.df[['close', 'volume', 'SMA', 'RSI']].copy()
        obs_df = obs_df.iloc[self.timestamp_idx - OBS_WINDOW: self.timestamp_idx, :]

        return obs_df.to_numpy()

    def _act(self, action) -> float:
        # Update price and previous price.
        self.price = self.df.loc[self.timestamp_idx, 'close']
        self.volume = self.df.loc[self.timestamp_idx, 'volume']
        prev_price = self.df.loc[self.timestamp_idx - 1, 'close']

        # Update balance based on currently held shares and the price difference from previous sample.
        sample_diff = self.price - prev_price
        self.balance += self.nbr_shares * sample_diff

        # Flat reward at center, large reward for buying very low and selling very high.
        reward = 800 * (self.price - 0.5) ** 3
        if action == 1:
            # SELL
            self.nbr_shares -= 1
        elif action == 0:
            # BUY
            self.nbr_shares += 1

            # Flip reward for buying so that high reward is given at low price.
            reward = -reward

        return reward

    def _next_state(self) -> bool:
        self.timestamp_idx += 1

        return self.timestamp_idx == len(self.df.index)

    def _prepare_df(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_csv(self.files[self.file_idx])

        # Normalize close and volume columns before calculating indicators.
        df['close'] = (df['close'] - min(df['close'])) / (max(df['close']) - min(df['close']))
        df['volume'] = (df['volume'] - min(df['volume'])) / (max(df['volume']) - min(df['volume']))

        df['SMA'] = TA.SMA(df, period=20)
        df['RSI'] = TA.RSI(df, period=14)

        df['SMA'].fillna(df['close'], inplace=True)
        df['RSI'].fillna(0.0, inplace=True)

        # Rescale RSI from range 0..100 -> 0..1
        df['RSI'] = df['RSI'] / 100.0

        return df

    def step(self, action):
        reward = self._act(action)

        info = self._get_info(action, reward)
        terminated = self._next_state()
        obs = self._get_observation()

        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        self.balance = 0.0
        self.timestamp_idx = OBS_WINDOW

        self.file_idx += 1
        if self.file_idx >= len(self.files):
            self.file_idx = 0

        self.df = self._prepare_df()
        self.nbr_shares = 0

        obs = self._get_observation()
        info = self._get_info(0, 0)

        return obs, info

    def render(self):
        pass
