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
PP = PrettyPrinter(indent=2, depth=1)


def filter_files(file_paths: list[Path], filter_strs: list[str]) -> list[Path]:
    filtered_data_files = []

    for f in file_paths:
        for arg in filter_strs:
            if arg in str(f):
                filtered_data_files.append(f)
                break

    return filtered_data_files


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

        nbr_rows = len(data_valid.index)
        for idx, row in data_valid.iterrows():
            action = row['action']

            if idx == nbr_rows - 1:
                # Force sell all held stocks at final sample point.
                data_valid.at[idx, 'action'] = -nr_shares
            elif action > 0:
                nr_shares += action
            elif action < 0:
                if abs(action) > nr_shares:
                    action = -nr_shares
                    data_valid.at[idx, 'action'] = action
                nr_shares += action

        return data_valid

    @staticmethod
    def run_period(strategy: Strategy, data_path: Path) -> pd.DataFrame:
        """Simulates single period. Returns DataFrame with additional 'actions' column."""
        data_clean: pd.DataFrame = pd.read_csv(data_path)

        # Create DataFrame section with indicator values.
        data_out = Simulator.prepare_dataframe(strategy, data_clean)

        actions: list[int] = []
        nbr_rows = len(data_out.index)
        # Actuate the strategy at each window.
        for row_idx in range(nbr_rows):
            start_idx = max(0, row_idx - Simulator.WINDOW)
            df_section: pd.DataFrame = data_out.iloc[start_idx:row_idx, :]

            # At final row, force sell held stocks to finish day without stocks in account.
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
    perc_yield: float  # The yield for simulated period as nbr of stocks gained. E.g. 0.3 = +0.3 * stock price.
    trades: list[Trade]  # List of trades performed during simulated period.
    sim_file: Path = None


class Analyzer:
    @staticmethod
    def calculate_balance(trades: list[Trade]):
        running_balance = 0.0

        for trade in trades:
            running_balance -= trade.nbr_shares * trade.price

        return running_balance

    @staticmethod
    def evaluate(files: list[Path]):
        results: list[Result] = []

        for f in files:
            df = pd.read_csv(f)

            result = Analyzer.evaluate_sim(df)
            result.sim_file = f

            results.append(result)

        PP.pprint(results)

        print(f'Evaluation complete...')

    @staticmethod
    def evaluate_sim(df: pd.DataFrame) -> Result:
        """Evalutes simulated data."""
        trades = []

        for idx, row in df.iterrows():
            action = row['action']
            if action == 0:
                continue

            trade = Trade(nbr_shares=action, price=row['close'])
            trades.append(trade)

        day_initial_price = df['close'][0]
        balance = Analyzer.calculate_balance(trades)
        perc_yield = balance / day_initial_price

        return Result(perc_yield, trades)

    @staticmethod
    def show_graph(df: pd.DataFrame, title: str):
        """Graphs columns in dataframe where actions are highlighted with points in graph."""
        # date, time, close, [indicators], actions
        ax = df.plot(x='time', y=list(range(2, len(df.columns) - 1)), kind='line', zorder=1)

        # Plot actions conditionally.
        cmap, norm = mcolors.from_levels_and_colors(levels=[-100, 0, 100], colors=['#ff1934', '#1cd100'])
        ax.scatter(df['time'], df['close'],
                   s=abs(df['action'] * 20),
                   c=df['action'],
                   cmap=cmap,
                   norm=norm,
                   zorder=2)
        plt.title(title)
        plt.show()

    @staticmethod
    def show_graphs(data_files: list[Path]):
        for f in data_files:
            Analyzer.show_graph(pd.read_csv(f), str(f))


def main(args):
    if args.evaluate is not None:
        # TODO: Add function to calculate expected value if stocks were owned from start to finish.

        data_files = list(DATA_SIMULATED.glob('**/*.csv'))

        # Filter data files if arguments are given.
        if args.evaluate:
            data_files = filter_files(data_files, args.evaluate)
        Analyzer.evaluate(data_files)

    if args.graph:
        # TODO: Handle graphing when multiple strategy folders exist.
        data_files = list(DATA_SIMULATED.glob('**/*.csv'))
        data_files = filter_files(data_files, args.graph)
        Analyzer.show_graphs(data_files)

    # When run_simulation is not None, the argument has been given but may still be empty.
    if args.run_simulation is not None:
        strategy = StrategySMA()
        data_files = list(DATA_CLEAN.glob('**/*.csv'))

        # Filter data files if arguments are given.
        if args.run_simulation:
            data_files = filter_files(data_files, args.run_simulation)

        Simulator.run(strategy, data_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Stock data collector', description='Gathers and cleans data.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-e', '--evaluate', type=str, nargs='*',
                        help='Evaluate the simulation results for the specific symbol and date. E.g.: '
                             'NVDA/2023/05/01')
    parser.add_argument('-g', '--graph', type=str, nargs='+',
                        help='Graph the simulation results. Supply string selectors to filter files. E.g.: '
                             'AAPL\\2023\\04\\03')
    parser.add_argument('-r', '--run-simulation', type=str, nargs='*',
                        help='Simulate the active strategy. Supply string selectors to filter files. E.g.\n'
                             '\tTSLA NVDA\t\t to only select Tesla and Nvidia stock, or\n'
                             '\tTSLA\\2023 MSFT\t\t to only select Tesla 2023 and all Microsoft files.    ')

    main(parser.parse_args())
