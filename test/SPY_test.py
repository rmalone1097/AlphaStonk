import sys
import os
import csv
import pandas as pd
import numpy as np
import gym
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from utils.data_utils import *
from pathlib import Path
from envs.stock_env_test import StockEnv
from ray.rllib.algorithms.algorithm import Algorithm

tickers = ['SPY', 'AAPL', 'BAC']
cande_length = 1
start_stamp = 928761600
end_stamp = 1676581140
data_path = Path.home() / 'data'
_, _, full_test_df, obs_test_df = prepare_state_df(tickers, data_path, 2206200)

def test_algo(algo_path, env):
    roi_list = []
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

with open(Path.home() / 'Git' /  'AlphaStonk' / 'test' / 'SPY_AAPL_BAC_PPO_results.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

with open(Path.home() / 'Git' /  'AlphaStonk' / 'test' / 'SPY_AAPL_BAC_PPO_portfolio.csv', newline='') as l:
    reader2 = csv.reader(l)
    data2 = list(reader2)

with open(Path.home() / 'Git' /  'AlphaStonk' / 'test' / 'results.csv', newline='') as f:
    reader = csv.reader(f)
    data3 = list(reader)

def plot_roi_list(roi_list):
    plt.plot([round(float(i), 2) for i in roi_list], label='Bot')
    plt.plot(np.linspace(0, len(full_test_df), len(full_test_df)), [(price - full_test_df['SPY_close'].iloc[0]) / full_test_df['SPY_close'].iloc[0] * 100 for price in full_test_df['SPY_close']], label='SPY')
    plt.plot(np.linspace(0, len(full_test_df), len(full_test_df)), [(price - full_test_df['AAPL_close'].iloc[0]) / full_test_df['AAPL_close'].iloc[0] * 100 for price in full_test_df['AAPL_close']], label='AAPL')
    plt.plot(np.linspace(0, len(full_test_df), len(full_test_df)), [(price - full_test_df['BAC_close'].iloc[0]) / full_test_df['BAC_close'].iloc[0] * 100 for price in full_test_df['BAC_close']], label='BAC')

    plt.legend(loc=1)
    plt.show()

def plot_portfolio(portfolio_list):
    plt.plot((np.linspace(0, len(data2[0]), len(data2[0]))), [float(i) for i in portfolio_list])
    plt.show()

plot_roi_list(data[0])
#plot_roi_list(data3[0])
#plot_portfolio(data2[0])