import argparse
import indicator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import random

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from pprint import PrettyPrinter

DATA_CLEAN: Path = Path('./data/clean')
DATA_SIMULATED: Path = Path('./data/simulated')
PP = PrettyPrinter(indent=2)


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


class StrategySMA(Strategy):
    """Defines a trading strategy."""
    FIRST_TRADE_IDX = 80
    SMA_20 = 0  # Index of indicator.
    SMA_60 = 1  # Index of indicator.

    def __init__(self):
        self._indicators = [indicator.SMA(20), indicator.SMA(60)]

    @property
    def indicators(self) -> list:
        return self._indicators

    def indicator_labels(self) -> list:
        return [str(ind) for ind in self._indicators]

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

    def __str__(self):
        return f'strategy_{"_".join(self.indicator_labels())}'


class Simulator:
    WINDOW = 100  # TODO: Should this be hardcoded or up to the strategy to decide?

    @staticmethod
    def prepare_dataframe(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
        # Keep the 'date', 'time', 'close' and add the indicator columns.
        df_ret = pd.DataFrame()

        df_ret['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df_ret['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
        df_ret['close'] = df['close']

        for ind in strategy.indicators:
            df_ret[str(ind)] = ind.indicate(pd.Series(df['close']))

        return df_ret

    @staticmethod
    def validate_actions(data_in: pd.DataFrame) -> pd.DataFrame:
        """Validates the 'action' column so that we can only sell when we have stocks."""
        data_valid = data_in.copy(deep=True)
        nr_shares = 0

        for index, row in data_valid.iterrows():
            action = row['action']

            if action > 0:
                nr_shares += action
            elif action < 0:
                if abs(action) > nr_shares:
                    action = -nr_shares
                    data_valid.at[index, 'action'] = action
                nr_shares += action

        return data_valid

    @staticmethod
    def run_period(strategy: Strategy, data_path: Path) -> pd.DataFrame:
        """Simulates single period. Returns DataFrame with additional 'actions' column."""
        data_clean: pd.DataFrame = pd.read_csv(data_path)

        # Create DataFrame section with indicator values.
        data_out = Simulator.prepare_dataframe(strategy, data_clean)

        actions = []
        # Actuate the strategy at each window.
        for row_idx in range(len(data_out.index)):
            start_idx = max(0, row_idx - Simulator.WINDOW)
            df_section: pd.DataFrame = data_out.iloc[start_idx:row_idx, :]

            action = strategy.act(df_section)
            actions.append(action)

        # Add strategy actions as new dataframe column.
        data_out['action'] = pd.Series(actions)

        return data_out

    @staticmethod
    def run(strategy: Strategy, data_files: list[Path]):
        """Runs simulation of strategy over list of files. Produces list of 'TradingPeriods', one for each file."""
        print(f'-- SIMULATION START --')
        print(f'{str(strategy)}')
        for data_path in data_files:
            print(f'Simulating: {data_path}')

            data_period: pd.DataFrame = Simulator.run_period(strategy, data_path)
            data_valid: pd.DataFrame = Simulator.validate_actions(data_period)

            path_simulated = DATA_SIMULATED.joinpath(str(strategy)).joinpath(*data_path.parts[2:])
            path_simulated.parent.mkdir(parents=True, exist_ok=True)

            data_valid.to_csv(path_simulated, index=False)


@dataclass
class Trade:
    nbr_shares: int  # Positive=BUY, negative=SELL.
    price: float


@dataclass
class Result:
    balance: float
    trades: list[Trade]


class Analyzer:
    @staticmethod
    def evaluate(df: pd.DataFrame) -> Result:
        """Evalutes simulated data."""
        trades = []

        for idx, row in df.iterrows():
            action = row['action']
            if action == 0:
                continue

            trade = Trade(nbr_shares=action, price=row['close'])
            trades.append(trade)

        # TODO: Continue work...


        return Result(0.0, [])  # TODO: Return proper result instance.

    @staticmethod
    def show_graph(df: pd.DataFrame):
        """Graphs columns in dataframe where actions are highlighted with points in graph."""
        # date, time, close, [indicators], actions
        ax = df.plot(x='time', y=list(range(2, len(df.columns) - 1)), kind='line', zorder=1)

        # Plot actions conditionally.
        cmap, norm = mcolors.from_levels_and_colors(levels=[-1, 0, 2], colors=['#ff1934', '#1cd100'])
        ax.scatter(df['time'], df['close'], s=abs(df['action'] * 15), c=df['action'], cmap=cmap, norm=norm, zorder=2)
        plt.show()


def main(args):
    if args.evaluate:
        # TODO: Handle evaluation of multiple files.
        data_files = list(DATA_SIMULATED.glob(f'**/{args.evaluate}*.csv'))
        Analyzer.evaluate(pd.read_csv(data_files[0]))

    if args.graph:
        # TODO: Handle graphing when multiple strategy folders exist.
        data_files = list(DATA_SIMULATED.glob(f'**/{args.graph}*.csv'))
        Analyzer.show_graph(pd.read_csv(data_files[0]))

    if args.run_simulation:
        strategy = StrategySMA()
        data_files = list(DATA_CLEAN.glob('**/*.csv'))
        Simulator.run(strategy, data_files)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Stock data collector', description='Gathers and cleans data.')

    parser.add_argument('-e', '--evaluate', type=str,
                        help='Evaluate the simulation results for the specific symbol and date. E.g.: '
                             'NVDA/2023/05/01')
    parser.add_argument('-g', '--graph', type=str,
                        help='Graph the simulation results for the specific symbol and date. E.g.: '
                             'AAPL/2023/04/03')
    parser.add_argument('-r', '--run-simulation', action='store_true', help='Simulate the active strategy.')

    main(parser.parse_args())
