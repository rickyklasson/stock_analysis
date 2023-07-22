import argparse
import indicators
import pandas as pd
import random

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from pprint import PrettyPrinter

DATA_CLEAN: Path = Path('./data/clean')
PP = PrettyPrinter(indent=2)


class Actions:
    PASS = 0
    BUY = 1
    SELL = 2


class Actuator:
    @staticmethod
    def do_action(df: pd.DataFrame):
        """Takes df with columns 'date', 'time', 'close' and indicator columns."""
        rng = random.randint(1, 100)

        if rng < 5:
            return Actions.BUY
        elif rng > 95:
            return Actions.SELL
        else:
            return Actions.PASS


@dataclass
class Strategy:
    """Defines a trading strategy."""
    actuator: Actuator
    indicators: list  # The indicators used for this strategy
    start_trade: int  # Start allowing trades at this data sample index


@dataclass
class Trade:
    action: bool  # 0=sold, 1=bought.
    date_time: datetime  # Date & time of trade.
    nbr: int  # Number of traded stocks.
    price: float  # Price for single stock.
    symbol: str  # Symbol of traded stock.


@dataclass
class TradingPeriod:
    data_file: Path
    result: float
    date_time_start: datetime  # Start of period.
    date_time_end: datetime  # End of period.
    trades: list[Trade]


class Simulator:
    WINDOW = 10

    @staticmethod
    def prepare_dataframe(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
        # Keep the 'date', 'time', 'close' and add the indicator columns.
        df_ret = pd.DataFrame()

        df_ret['date'] = df['date']
        df_ret['time'] = df['time']
        df_ret['close'] = df['close']

        for indicator in strategy.indicators:
            df_ret[str(indicator)] = indicator.indicate(pd.Series(df['close']))

        print(df_ret)
        return df_ret

    @staticmethod
    def run_period(strategy: Strategy, data_file) -> TradingPeriod:
        """Simulates single period."""
        data_clean: pd.DataFrame = pd.read_csv(data_file)

        # Create DataFrame section with indicator values.
        data_in = Simulator.prepare_dataframe(strategy, data_clean)

        # Read datetimes. Only supports single day periods.
        date: str = data_in['date'][0]
        time_start: str = data_in['time'][0]
        time_end: str = data_in['time'].iloc[-1]
        date_time_start = datetime.strptime(f'{date} {time_start}', '%Y-%m-%d %H:%M:%S')
        date_time_end = datetime.strptime(f'{date} {time_end}', '%Y-%m-%d %H:%M:%S')

        trading_period = TradingPeriod(data_file, 0.0, date_time_start, date_time_end, [])

        # Actuate the strategy at each window.
        for row_idx in range(len(data_in.index)):
            start_idx = max(0, row_idx - Simulator.WINDOW)
            df_section: pd.DataFrame = data_in.iloc[start_idx:row_idx, :]



    @staticmethod
    def run(strategy: Strategy, data_files: list[Path]):
        """Runs simulation of strategy over list of files. Produces list of 'TradingPeriods', one for each file."""
        for data_file in data_files:
            Simulator.run_period(strategy, data_file)


def main(args):
    strategy = Strategy(
        actuator=Actuator(),
        indicators=[indicators.SMA(20), indicators.SMA(60)],
        start_trade=60,
    )

    data_files = list(DATA_CLEAN.glob('1min/AAPL/2023/04/03.csv'))  # '**/*.csv'
    Simulator.run(strategy, data_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Stock data collector', description='Gathers and cleans data.')

    main(parser.parse_args())
