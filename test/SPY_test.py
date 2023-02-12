import sys
import os
import csv
import pandas as pd
import numpy as np
import gym
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from utils.data_utils import add_indicators
from pathlib import Path
from envs.stock_env_test import StockEnv
from ray.rllib.algorithms.algorithm import Algorithm

pickle_dir = Path.home() / 'Git' /  'AlphaStonk' / 'utils'
df = pd.read_pickle(pickle_dir / 'SPY_minute_2022-08-22_built.pkl')
test_df = add_indicators(df)
test_df = test_df.fillna(0)

env = StockEnv(test_df)
algo_path = Path.home() / "ray_results" / "PPO" / "PPO_StockEnv_aa5c4_00000_0_2023-02-11_12-45-41" / "checkpoint_000003"

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

with open(Path.home() / 'Git' /  'AlphaStonk' / 'test' / 'results.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

print(data)

def plot_roi_list(roi_list):
    plt.plot(roi_list[0])
    plt.show()

plot_roi_list(data)