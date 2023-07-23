import argparse
import indicator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import random

from abc import ABC, abstractmethod
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
        return f'Strategy: Evaluate {self.indicator_labels()} and act: Random.'


def validate_actions(data_in: pd.DataFrame) -> pd.DataFrame:
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


class Simulator:
    WINDOW = 100

    @staticmethod
    def prepare_dataframe(strategy: Strategy, df: pd.DataFrame) -> pd.DataFrame:
        # Keep the 'date', 'time', 'close' and add the indicator columns.
        df_ret = pd.DataFrame()

        df_ret['date'] = df['date']
        df_ret['time'] = df['time']
        df_ret['close'] = df['close']

        for ind in strategy.indicators:
            df_ret[str(ind)] = ind.indicate(pd.Series(df['close']))

        return df_ret

    @staticmethod
    def run_period(strategy: Strategy, data_file) -> pd.DataFrame:
        """Simulates single period. Returns DataFrame with additional 'actions' column."""
        data_clean: pd.DataFrame = pd.read_csv(data_file)

        # Create DataFrame section with indicator values.
        data_out = Simulator.prepare_dataframe(strategy, data_clean)

        # Read datetimes. Only supports single day periods.
        # date: str = data_in['date'][0]
        # time_start: str = data_in['time'][0]
        # time_end: str = data_in['time'].iloc[-1]
        # date_time_start = datetime.strptime(f'{date} {time_start}', '%Y-%m-%d %H:%M:%S')
        # date_time_end = datetime.strptime(f'{date} {time_end}', '%Y-%m-%d %H:%M:%S')

        actions = []

        # Actuate the strategy at each window.
        for row_idx in range(len(data_out.index)):
            start_idx = max(0, row_idx - Simulator.WINDOW)
            df_section: pd.DataFrame = data_out.iloc[start_idx:row_idx, :]

            actions.append(strategy.act(df_section))

        data_out['action'] = pd.Series(actions)

        return data_out

    @staticmethod
    def run(strategy: Strategy, data_files: list[Path]):
        """Runs simulation of strategy over list of files. Produces list of 'TradingPeriods', one for each file."""
        print(f'-- SIMULATION START --')
        print(f'{str(strategy)}')
        for data_file in data_files:
            print(f'Simulating: {data_file}')

            data_period: pd.DataFrame = Simulator.run_period(strategy, data_file)
            data_valid = validate_actions(data_period)

            path_simulated = DATA_SIMULATED.joinpath(*data_file.parts[2:])
            path_simulated.parent.mkdir(parents=True, exist_ok=True)

            data_valid.to_csv(path_simulated, index=False)


class Analyzer:
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
    if args.run_simulation:
        strategy = StrategySMA()
        data_files = list(DATA_CLEAN.glob('**/*.csv'))
        Simulator.run(strategy, data_files)

    if args.graph:
        data_file = DATA_SIMULATED / f'1min/{args.graph}.csv'
        Analyzer.show_graph(pd.read_csv(data_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Stock data collector', description='Gathers and cleans data.')

    parser.add_argument('-r', '--run-simulation', action='store_true', help='Simulate the active strategy.')
    parser.add_argument('-g', '--graph', type=str,
                        help='Graph the simulation results for the specific symbol and date. E.g.: '
                             'AAPL/2023/04/03')

    main(parser.parse_args())
