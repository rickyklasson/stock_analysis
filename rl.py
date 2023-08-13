import argparse
import pandas as pd

from pathlib import Path
from pprint import PrettyPrinter
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from stockenv import StockEnv, Trade

DATA_CLEAN = Path('./data/clean')
DATA_SIMULATED = Path('./data/simulated')
PP = PrettyPrinter(indent=2, depth=1, width=180)


def trades_to_actions(trades: list[Trade], nr_rows: int) -> pd.Series:
    actions = []
    buy_ticks = [trade.buy_tick for trade in trades]
    sell_ticks = [trade.sell_tick for trade in trades]

    for i in range(nr_rows):
        if i in buy_ticks:
            actions.append(1)
        elif i in sell_ticks:
            actions.append(-1)
        else:
            actions.append(0)

    return pd.Series(actions)


def main(args):
    if args.check_env:
        data_files = list(DATA_CLEAN.glob('**/*.csv'))
        env = StockEnv(data_files)
        check_env(env)

    if args.double_check_env:
        # Double check env by printing info from a short period of random actions.
        episodes = 1
        data_files = list(DATA_CLEAN.glob('**/TSLA/2022/12/01.csv'))
        env = StockEnv(data_files)

        for ep in range(episodes):
            terminated = False
            _, period_info = env.reset()

            while not terminated:
                random_action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(random_action)

                print(f'{reward=}, {terminated=}')
                PP.pprint(info)

            _, period_info = env.reset()
            PP.pprint(period_info)

    if args.train:
        model_name = 'PPO_SMA_RSI_TSLA_2022_12_RSI_CLOSE_VOLUME_SMI_OBV_LONG'
        models_dir = Path('./ml_models') / model_name
        log_dir = Path('./ml_logs') / model_name

        models_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        data_files = list(DATA_CLEAN.glob('**/TSLA/2022/12/*.csv'))
        env = StockEnv(data_files)
        env.reset()

        print('---- Model training start ----')
        print('Files included in training:')
        PP.pprint(data_files)
        print(f'Model name: {model_name}')

        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=str(log_dir), device='cpu')
        #model = PPO.load('ml_models/PPO_SMA_RSI_TSLA_2022_12_RSI_CLOSE_VOLUME_SMI_OBV/1240000.zip', env=env)

        timesteps = 10000
        for i in range(1, 191):
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
            model.save(models_dir / f'{timesteps * i}')

    if args.simulate:
        data_files = list(DATA_CLEAN.glob('**/TSLA/2022/12/02.csv'))
        env = StockEnv(data_files)
        model = PPO.load(args.simulate, env=env)
        obs, _ = env.reset()

        episodes = 0
        while True:
            action = model.predict(obs)[0]
            obs, reward, terminated, _, info = env.step(action)

            if terminated:
                _, period_info = env.reset()
                PP.pprint(period_info)

                orig_df = pd.read_csv(period_info['file_path'])
                orig_df['action'] = trades_to_actions(period_info['trade_history'], len(orig_df.index))

                model_name = Path(args.simulate).parent.name
                sim_file = DATA_SIMULATED / model_name / Path('/'.join(Path(period_info['file_path']).parts[2:]))
                sim_file.parent.mkdir(parents=True, exist_ok=True)
                orig_df.to_csv(sim_file, index=False, float_format='%.3f')

                episodes += 1
                if episodes == len(data_files):
                    break

    if args.evaluate:
        # Evaluate model over a few episodes.
        data_files = list(DATA_CLEAN.glob('**/TSLA/2022/12/01.csv'))
        env = StockEnv(data_files)
        model = PPO.load(args.evaluate, env=env)

        reward_mean, reward_std = evaluate_policy(model, env, n_eval_episodes=1)
        print(f'{reward_mean=}, {reward_std=}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Stock data collector', description='Gathers and cleans data.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-ch', '--check-env', action='store_true', help='Ensure correct custom gym Env implementation.')
    parser.add_argument('-dch', '--double-check-env', action='store_true',
                        help='Double check env by running a few episodes.')
    parser.add_argument('-t', '--train', action='store_true', help='Train RL model on stock data.')
    parser.add_argument('-s', '--simulate', type=str,
                        help='Simulate supplied model and generate simulated data.')
    parser.add_argument('-ev', '--evaluate', type=str,
                        help='Evaluate model by running a few episodes.')
    main(parser.parse_args())
