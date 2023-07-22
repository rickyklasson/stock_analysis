import indicators
import numpy as np
import pandas as pd



class TestIndicators:
    @staticmethod
    def test_sma():
        sma = indicators.SMA(2)
        data = pd.Series(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

        # SMA with 2 sample mean.
        indication = sma.indicate(data)

        print('---- SMA ----')
        print(f'Data: {list(data)}')
        print(f'SMA 2: {list(indication)}')

    @staticmethod
    def test_ema():
        ema = indicators.EMA(0.5)
        data = pd.Series(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

        indication = ema.indicate(data)

        print('---- EMA ----')
        print(f'Data: {list(data)}')
        print(f'EMA 0.5: {list(indication)}')

    @staticmethod
    def test_bb():
        bb = indicators.BB(3, 1)
        data = pd.Series(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

        indication = bb.indicate(data)

        print('---- BB ----')
        print(f'Data: {list(data)}')
        print(f'BB 3, +1 std: {list(indication)}')


if __name__ == '__main__':
    TestIndicators.test_sma()
    TestIndicators.test_ema()
    TestIndicators.test_bb()
