import sys
import os
import pandas as pd
import numpy as np
import gym
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from envs.stock_env_test import StockEnv
from ray.rllib.algorithms.algorithm import Algorithm

test_df = 0
env = StockEnv(env_config={"df": test_df})
algo = Algorithm.from_checkpoint(Path.home() / "ray_results" / "SPY_PPO_MLP_woVIX_2012" / "checkpoint")

def test_algo(algo, env, test_df):
    tensor = test_df.to_numpy()
    roi_list = []

    episode_reward = 0
    done=False
    obs = env.reset()
    while not done:
        action = algo.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        roi_list.append(env.total_roi)