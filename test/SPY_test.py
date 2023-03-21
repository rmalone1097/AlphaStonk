import sys
import os
import csv
import pandas as pd
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from utils.data_utils import *
from utils.data_utils import *
from pathlib import Path
from envs.stock_env_test import StockEnv
from ray.rllib.algorithms.algorithm import Algorithm

tickers = ['SPY']
cande_length = 1
start_stamp = 928761600
end_stamp = 1676581140
data_path = Path.home() / 'Documents' / 'Datasets'
_, _, full_test_df, obs_test_df = prepare_state_df(tickers, data_path, 2206200)

short_df = pd.read_pickle(Path.home() / 'data' / 'SPY_minute_2022-08-22_built.pkl')
print(short_df)

env = StockEnv(config = {'full_df': full_test_df, 'obs_df': obs_test_df, 'tickers': tickers})
algo_path = Path.home() / 'ray_results'/'PPO'/'PPO_StockEnv_280fa_00000_0_2023-03-01_06-13-17'/'checkpoint_002500'
roi_file_name = 'SPY_AAPL_BAC_PPO_roi.csv'
portfolio_file_name = 'SPY_AAPL_BAC_PPO_portfolio.csv'

def test_algo(algo_path, env, roi_file_name, portfolio_file_name):
    roi_list = []
    portfolio_list = []
    algo = Algorithm.from_checkpoint(algo_path)

    episode_reward = 0
    done=False
    obs = env.reset()
    while not done:
        action = algo.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        roi_list.append(env.total_roi)
    return roi_list

def read_csv(path):
    with open(Path.home() / 'Desktop' / 'SPY_PPO_roi.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data[0]

def plot_roi_list(full_test_df, roi_list):
    plt.plot([round(float(i), 2) for i in roi_list], label='Bot')
    plt.plot(np.linspace(0, len(full_test_df), len(full_test_df)), [(price - full_test_df['SPY_close'].iloc[0]) / full_test_df['SPY_close'].iloc[0] * 100 for price in full_test_df['SPY_close']], label='SPY')
    #plt.plot(np.linspace(0, len(full_test_df), len(full_test_df)), [(price - full_test_df['AAPL_close'].iloc[0]) / full_test_df['AAPL_close'].iloc[0] * 100 for price in full_test_df['AAPL_close']], label='AAPL')
    #plt.plot(np.linspace(0, len(full_test_df), len(full_test_df)), [(price - full_test_df['BAC_close'].iloc[0]) / full_test_df['BAC_close'].iloc[0] * 100 for price in full_test_df['BAC_close']], label='BAC')

    plt.legend(loc=1)
    plt.show()

def old_plot_roi_list(full_test_df, roi_list):
    plt.plot([round(float(i), 2) for i in roi_list], label='Bot')
    #plt.plot(np.linspace(0, len(short_df), len(short_df)), [(price - full_test_df['close'].iloc[0]) / full_test_df['close'].iloc[0] * 100 for price in full_test_df['close']], label='SPY')

    plt.legend(loc=1)
    plt.show()

def plot_portfolio(portfolio_list):
    #plt.plot((np.linspace(0, len(data2[0]), len(data2[0]))), [float(i) for i in portfolio_list])
    plt.show()

if __name__ == "__main__":
    test_algo(algo_path, env, roi_file_name, portfolio_file_name)
    #roi_list = read_csv(Path.home() / 'Desktop' / 'SPY_PPO_roi.csv')
    #plot_roi_list(full_test_df, roi_list)