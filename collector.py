import argparse
import pandas as pd
import requests
import time

from config import config, stock_symbols
from pathlib import Path
from pprint import PrettyPrinter

DATA_RAW: Path = Path('./data/raw')
DATA_CLEAN: Path = Path('./data/clean')
PP = PrettyPrinter(indent=2)


class DataGatherer:
    @staticmethod
    def get_csv_data(symbol: str, month: str) -> str | None:
        """See https://www.alphavantage.co/documentation/ for alpha vantage API docs."""

        url_base = 'https://www.alphavantage.co/query'
        url_params = {
            'apikey': config['API_KEY'],
            'datatype': 'csv',
            'extended_hours': 'false',
            'function': 'TIME_SERIES_INTRADAY',
            'interval': '1min',
            'month': month,
            'outputsize': 'full',
            'symbol': symbol,
        }

        r = requests.get(url_base, url_params)

        if r.status_code == 200:
            return r.text
        else:
            return None

    @staticmethod
    def symbol_data_to_file(symbol: str, month: str) -> bool:
        print(f'Fetching data: {symbol} ({month})')

        out_folder: Path = DATA_RAW / '1min' / symbol
        out_file: Path = out_folder / f'{month}.csv'

        if out_file.exists():
            return False

        csv_str = DataGatherer.get_csv_data(symbol, month)
        if csv_str is None:
            return False

        out_folder.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w', newline='') as outfile:
            outfile.write(csv_str)

        return True


class DataCleaner:
    @staticmethod
    def clean_file(filepath: Path):
        """Data is cleaned in the following steps:
            1) Reverse data such that it is in chronological order from top to bottom.
            2) Split data into one file for each day. New data path should be: './data/clean/1min/AAPL/2023/05/01.csv
        """
        print(f'Cleaning file: {filepath}')

        filename = filepath.stem  # Filename is on format YYYY-MM
        symbol = filepath.parent.stem
        year, month = filename.split('-')

        # Read data and adjust type of timestamp column.
        df: pd.DataFrame = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

        # Reverse data to make it chronological from top to bottom. Reset index to enumerate from top to bottom.
        df = df.iloc[::-1]
        df = df.reset_index(drop=True)

        # Replace 'timestamp' column with 'date' and 'time' column to simplify grouping.
        df[['date', 'time']] = df['timestamp'].str.split(' ', expand=True)
        df = df.drop('timestamp', axis=1)

        # Rearrange to have date and time be the first two dataframe columns.
        df.insert(0, 'time', df.pop('time'))
        df.insert(0, 'date', df.pop('date'))

        # Group dataframe by timestamp date and extract the dates.
        group_by = df.groupby(['date'])
        dates = group_by.groups.keys()

        # Create folder for cleaned data.
        data_folder_clean = DATA_CLEAN / '1min' / symbol / year / month
        data_folder_clean.mkdir(parents=True, exist_ok=True)

        # Get each sub-dataframe and write to file.
        for date in dates:
            data_file_path = data_folder_clean / f'{str(date).split("-")[2]}.csv'

            sub_df: pd.DataFrame = group_by.get_group(date)
            sub_df.to_csv(data_file_path, index=False)

    @staticmethod
    def clean_data():
        """Cleans all raw data files in folder: ./data/raw/**/*"""
        p = Path(DATA_RAW)
        filepaths = list(p.glob('**/*.csv'))

        for p in filepaths:
            DataCleaner.clean_file(p)


def main(args):
    if args.gather:
        for symbol in stock_symbols:
            if DataGatherer.symbol_data_to_file(symbol, '2023-05'):
                time.sleep(12)

    if args.clean:
        DataCleaner.clean_data()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Stock data collector', description='Gathers and cleans data.')
    parser.add_argument('-c', '--clean', action='store_true', help='Cleans data if set.')
    parser.add_argument('-g', '--gather', action='store_true', help='List of stock symbols. E.g. TSLA AAPL')

    main(parser.parse_args())
