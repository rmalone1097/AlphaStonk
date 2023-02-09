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
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.registry import get_trainable_cls

from stock_env import StockEnv
from utils.data_utils import add_indicators

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
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

pickle_dir = Path.home()
df = pd.read_pickle(pickle_dir / 'SPY_minute_2012-08-22_built_gcp.pkl')
trading_df = add_indicators(df)
trading_df = trading_df.fillna(0)

if __name__ == "__main__":

    ray.init(num_gpus=1)
    args = parser.parse_args()
    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(StockEnv, env_config={"df": trading_df})
        .framework(args.framework)
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=1)
    )
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total" : args.stop_timesteps
    }
    tuner = tune.Tuner(
        args.run, param_space=config.to_dict(), run_config=air.RunConfig(stop=stop, verbose=1)
    )
    tuner.fit()