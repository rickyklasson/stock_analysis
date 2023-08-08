import indicator
import pandas as pd
import random

from abc import ABC, abstractmethod


class Strategy(ABC):
    _indicators: list  # The indicators used for this strategy

    @property
    @abstractmethod
    def indicators(self) -> list:
        return self._indicators

    @abstractmethod
    def act(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def indicator_labels(self) -> list:
        pass


class BaseStrategy(Strategy):
    def __init__(self):
        self._indicators = []

    @property
    def indicators(self) -> list:
        return self._indicators

    def indicator_labels(self) -> list:
        return [str(ind) for ind in self._indicators]

    def act(self, df: pd.DataFrame) -> int:
        """Returns number of shares to buy(>0) or sell(<0)."""
        return 0

    def __str__(self):
        return f'{"_".join(self.indicator_labels())}'


class StrategyRNG(BaseStrategy):
    FIRST_TRADE_IDX = 30
    SMA_20 = 0  # Index of indicator.
    SMA_60 = 1  # Index of indicator.

    def __init__(self):
        super().__init__()
        self._indicators = [indicator.SMA(20), indicator.SMA(60)]

    def act(self, df: pd.DataFrame) -> int:
        """Returns number of shares to buy(>0) or sell(<0)."""
        if df.shape[0] < StrategySMA.FIRST_TRADE_IDX:
            return 0

        rng = random.randint(0, 99)

        if rng < 5:
            return 1
        elif rng > 90:
            return -1
        else:
            return 0


class StrategySMA(BaseStrategy):
    FIRST_TRADE_IDX = 60
    SMA_SHORT = 0  # Index of indicator.
    SMA_LONG = 1  # Index of indicator.

    def __init__(self):
        super().__init__()
        self._indicators = [indicator.SMA(10), indicator.SMA(80)]

    def act(self, df: pd.DataFrame) -> int:
        """Returns number of shares to buy(>0) or sell(<0)."""
        if df.shape[0] < StrategySMA.FIRST_TRADE_IDX:
            return 0

        sma_short_last = df.iloc[-1][str(self._indicators[StrategySMA.SMA_SHORT])]
        sma_long_last = df.iloc[-1][str(self._indicators[StrategySMA.SMA_LONG])]
        sma_short_second_last = df.iloc[-2][str(self._indicators[StrategySMA.SMA_SHORT])]
        sma_long_second_last = df.iloc[-2][str(self._indicators[StrategySMA.SMA_LONG])]

        if sma_short_last < sma_long_last and sma_short_second_last > sma_long_second_last:
            # SMA 20 crossed SMA 60 downward on this sample -> SELL
            return -1
        elif sma_short_last > sma_long_last and sma_short_second_last < sma_long_second_last:
            # SMA 20 crossed SMA 60 upward on this sample -> BUY
            return 1

        return 0


class StrategyEMA(BaseStrategy):
    EMA_SHORT = 0
    EMA_LONG = 1

    def __init__(self):
        super().__init__()
        self._indicators = [indicator.EMA(0.1), indicator.EMA(0.02)]

    def act(self, df: pd.DataFrame) -> int:
        """Returns number of shares to buy(>0) or sell(<0)."""
        if df.shape[0] < StrategySMA.FIRST_TRADE_IDX:
            return 0

        ema_short_last = df.iloc[-1][str(self._indicators[StrategyEMA.EMA_SHORT])]
        ema_long_last = df.iloc[-1][str(self._indicators[StrategyEMA.EMA_LONG])]
        ema_short_second_last = df.iloc[-2][str(self._indicators[StrategyEMA.EMA_SHORT])]
        ema_long_second_last = df.iloc[-2][str(self._indicators[StrategyEMA.EMA_LONG])]

        if ema_short_last < ema_long_last and ema_short_second_last > ema_long_second_last:
            # SMA 20 crossed SMA 60 downward on this sample -> SELL
            return -1
        elif ema_short_last > ema_long_last and ema_short_second_last < ema_long_second_last:
            # SMA 20 crossed SMA 60 upward on this sample -> BUY
            return 1

        return 0
