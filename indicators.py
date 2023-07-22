from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


class Indicator(ABC):
    @abstractmethod
    def indicate(self, data: pd.Series):
        pass


@dataclass
class SMA(Indicator):
    samples: int

    def indicate(self, data: pd.Series) -> pd.Series:
        """Simple moving average indicator. Returns indicator values as pd.Series of same length as data."""
        data_sma = data.rolling(self.samples, min_periods=1).mean()
        return data_sma

    def __str__(self):
        return f'SMA_{self.samples}'


@dataclass
class EMA(Indicator):
    alpha: float

    def indicate(self, data: pd.Series) -> pd.Series:
        """Exponential moving average indicator. Returns indicator values as pd.Series of same length as data."""
        data_ema: pd.Series = data.ewm(alpha=self.alpha).mean()

        return data_ema

    def __str__(self):
        return f'EMA_{self.alpha:.2f}'


@dataclass
class BB(Indicator):
    samples: int
    std_dev: float

    def indicate(self, data: pd.Series):
        """Bollinger band created indentically to SMA but offset by std_dev standard deviations. Returns indicator
        values as pd.Series of same length as data.
        """
        data_sma = data.rolling(self.samples, min_periods=1).mean()
        data_std_offset = data.rolling(self.samples, min_periods=1).std() * self.std_dev
        data_std_offset.fillna(0, inplace=True)

        return data_sma + data_std_offset

    def __str__(self):
        return f'BB_{self.samples}_{self.std_dev:.2f}'
