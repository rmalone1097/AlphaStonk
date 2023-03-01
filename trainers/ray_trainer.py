from typing import Dict, Tuple
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tensorflow as tf

import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.registry import get_trainable_cls

from envs.stock_env import StockEnv
from utils.data_utils import prepare_state_df

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--stop-iters", type=int, default=6000, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=10000000, help="Number of timesteps to train."
)
'''parser.add_argument(
    "--stop-reward", type=float, default=600.0, help="Reward at which we stop training."
)'''

tickers = ['SPY', 'AAPL', 'BAC']
cande_length = 1
start_stamp = 928761600
end_stamp = 1676581140
data_path = Path.home() / 'data'
full_train_df, obs_train_df, _, _ = prepare_state_df(tickers, data_path, 2206200)
print(full_train_df)
print(obs_train_df)

if __name__ == "__main__":

    ray.init(num_gpus=1)
    args = parser.parse_args()
    config = (
        PPOConfig()
        .environment(StockEnv, env_config={"full_df": full_train_df,
                                           "obs_df": obs_train_df,
                                           "tickers": tickers})
        .framework(args.framework)
        .training()
        .rollouts(num_rollout_workers=20)
        .resources(num_gpus=1)
    )
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total" : args.stop_timesteps
    }
    tuner = tune.Tuner(
        "PPO", param_space=config.to_dict(), run_config=air.RunConfig(
            stop=stop, 
            verbose=1, 
            checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True)
        )
    )
    results = tuner.fit()