import argparse
import pandas as pd

from pathlib import Path
from pprint import PrettyPrinter
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from stockenv import StockEnv

DATA_CLEAN = Path('./data/clean')
DATA_SIMULATED = Path('./data/simulated')
PP = PrettyPrinter(indent=2, depth=2)


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
            env.reset()

            while not terminated:
                random_action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(random_action)

                print(f'{reward=}, {terminated=}')
                PP.pprint(info)

    if args.train:
        model_id = 'PPO_SMA_RSI_TSLA_2022_12_01_RSI_CLOSE_VOLUME_SMI_OBV_TEMP'
        models_dir = Path('./ml_models') / model_id
        log_dir = Path('./ml_logs') / model_id

        models_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        data_files = list(DATA_CLEAN.glob('**/TSLA/2022/12/01.csv'))
        env = StockEnv(data_files)
        env.reset()

        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=str(log_dir), device='cpu')
        #model = PPO.load('ml_models/PPO_SMA_RSI_TSLA_2022_12_RSI_CLOSE_VOLUME_SMI_OBV/1240000.zip', env=env)

        timesteps = 10000
        for i in range(1, 1001):
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
            model.save(models_dir / f'{timesteps * i}')

    if args.simulate:
        data_files = list(DATA_CLEAN.glob('**/TSLA/2022/12/01.csv'))
        env = StockEnv(data_files)
        model = PPO.load(args.simulate, env=env)

        obs, _ = env.reset()
        episodes = 0
        while True:
            action = model.predict(obs)
            obs, reward, terminated, _, info = env.step(action)

            if terminated:
                _, period_info = env.reset()
                PP.pprint(period_info)

                #sim_file = DATA_SIMULATED / model_name / Path('/'.join(Path(info['file']).parts[2:]))
                #write_to_sim_file(Path(info['file']), sim_file, actions)

                episodes += 1
                if episodes == len(data_files):
                    break

    if args.evaluate:
        # Evaluate model over a few episodes.
        data_files = list(DATA_CLEAN.glob('**/TSLA/2022/10/*.csv'))
        env = StockEnv(data_files)
        model = PPO.load('ml_models/saved/PPO_SMA_RSI_TSLA_2022_12_2M_ticks.zip', env=env)

        reward_mean, reward_std = evaluate_policy(model, env, n_eval_episodes=20)
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
    parser.add_argument('-ev', '--evaluate', action='store_true',
                        help='Evaluate model by running a few episodes.')
    main(parser.parse_args())
