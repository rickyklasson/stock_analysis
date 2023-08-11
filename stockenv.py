import gymnasium as gym
import numpy as np
import pandas as pd
import random

from enum import Enum
from finta import TA
from gymnasium import spaces
from pathlib import Path

NBR_ACTIONS = 2  # SELL=0, BUY=1
OBS_WINDOW = 60


class Actions(Enum):
    Sell = 0
    Buy = 1


class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class StockEnv(gym.Env):
    def __init__(self, files: list[Path]):
        super(StockEnv).__init__()
        self.files = random.sample(files, k=len(files))  # Shuffle files used during training.
        self.file_idx = 0  # Index of current file in list self.files.
        self.current_tick = OBS_WINDOW  # Index of timestamp in current file.
        self.last_trade_tick = None
        self.df = None  # Current dataframe used for training.
        self.total_reward = 0.0
        self.position = None
        self.trade_fee = 0.0

        self.action_space = spaces.Discrete(NBR_ACTIONS)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_WINDOW, 5), dtype=np.float64)

    def _prepare_df(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_csv(self.files[self.file_idx])

        df['RSI'] = TA.RSI(df, period=14)
        df['OBV'] = TA.OBV(df)
        df['SMI'] = TA.STOCH(df)

        df['RSI'].fillna(0.0, inplace=True)
        df['OBV'].fillna(0.0, inplace=True)
        df['SMI'].fillna(0.0, inplace=True)

        # Rescale RSI from range 0..100 -> 0..1
        df['RSI'] = df['RSI'] / 100.0
        df['SMI'] = df['SMI'] / 100.0

        return df

    def _get_observation(self):
        obs_df = self.df[['close', 'volume', 'RSI', 'SMI', 'OBV']].copy()
        obs_df = obs_df.iloc[self.current_tick - OBS_WINDOW: self.current_tick, :]

        # Normalize columns.
        obs_df['close'] = (obs_df['close'] - min(obs_df['close'])) / (max(obs_df['close']) - min(obs_df['close']))
        obs_df['volume'] = (obs_df['volume'] - min(obs_df['volume'])) / (max(obs_df['volume']) - min(obs_df['volume']))
        obs_df['RSI'] = (obs_df['RSI'] - min(obs_df['RSI'])) / (max(obs_df['RSI']) - min(obs_df['RSI']))
        obs_df['OBV'] = (obs_df['OBV'] - min(obs_df['OBV'])) / (max(obs_df['OBV']) - min(obs_df['OBV']))
        obs_df['SMI'] = (obs_df['SMI'] - min(obs_df['SMI'])) / (max(obs_df['SMI']) - min(obs_df['SMI']))

        return obs_df.to_numpy()

    def _calculate_reward(self, action):
        step_reward = 0.0

        trade = False
        if ((action == Actions.Buy.value and self.position == Positions.Short) or
                (action == Actions.Sell.value and self.position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.df['close'][self.current_tick]
            last_trade_price = self.df['close'][self.last_trade_tick]
            price_diff = current_price - last_trade_price

            if self.position == Positions.Long:
                step_reward += price_diff

        return step_reward

    def step(self, action):
        terminated = False

        step_reward = self._calculate_reward(action)
        self.total_reward += step_reward

        trade = False
        if ((action == Actions.Buy.value and self.position == Positions.Short) or
                (action == Actions.Sell.value and self.position == Positions.Long)):
            trade = True

        if trade:
            self.position = self.position.opposite()
            self.last_trade_tick = self.current_tick

        observation = self._get_observation()
        info = dict(
            file=str(self.files[self.file_idx]),
            tick=self.current_tick,
            total_reward=self.total_reward,
            position=self.position.value,
            price=self.df['close'][self.current_tick]
        )

        self.current_tick += 1

        if self.current_tick == len(self.df.index):
            terminated = True

        return observation, step_reward, terminated, False, info

    def reset(self, seed=None, options=None):
        self.current_tick = OBS_WINDOW
        self.last_trade_tick = OBS_WINDOW - 1
        self.position = Positions.Short
        self.total_reward = 0.

        self.file_idx += 1
        if self.file_idx >= len(self.files):
            self.file_idx = 0

        self.df = self._prepare_df()

        period_info = {}

        return self._get_observation(), period_info

    def render(self):
        pass
