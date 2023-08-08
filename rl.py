import argparse
import gymnasium as gym

from pathlib import Path
from pprint import PrettyPrinter
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import RecurrentPPO
from stockenv import StockEnv

DATA_CLEAN = Path('./data/clean')
PP = PrettyPrinter(indent=2, depth=1)


def main(args):
    if args.example:
        env = gym.make("CartPole-v1", render_mode="rgb_array")

        model = A2C("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=20_000)

        vec_env = model.get_env()
        obs = vec_env.reset()
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render("human")
            # VecEnv resets automatically
            # if done:
            #   obs = vec_env.reset()

    if args.check_env:
        # TODO: Divide data files into traning and validation data.
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
            obs = env.reset()

            while not terminated:
                random_action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(random_action)

                print(f'{reward=}, {terminated=}')
                PP.pprint(info)

    if args.train:
        model_id = 'LSTM_SMA_RSI'
        models_dir = Path('./ml_models') / model_id
        log_dir = Path('./ml_logs') / model_id

        models_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        data_files = list(DATA_CLEAN.glob('**/TSLA/2022/12/01.csv'))
        env = StockEnv(data_files)
        env.reset()

        # Load previous best model.
        #model_path = Path('ml_models') / 'win_60_samples_1M_close_and_volume_950k_start' / '1200000.zip'

        model = RecurrentPPO('MlpLstmPolicy', env, verbose=1, tensorboard_log=str(log_dir))
        #model = PPO.load(model_path, env)

        timesteps = 10000
        for i in range(1, 51):
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
            model.save(models_dir / f'{timesteps * i}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Stock data collector', description='Gathers and cleans data.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-ex', '--example', action='store_true', help='Run the RL gym example code.')
    parser.add_argument('-ch', '--check-env', action='store_true', help='Ensure correct custom gym Env implementation.')
    parser.add_argument('-dch', '--double-check-env', action='store_true',
                        help='Double check env by running a few episodes.')
    parser.add_argument('-t', '--train', action='store_true', help='Train RL model on stock data.')
    main(parser.parse_args())
