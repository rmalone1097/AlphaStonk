from typing import Dict, Tuple
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig

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
    "--stop-iters", type=int, default=600, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=200000, help="Number of timesteps to train."
)
'''parser.add_argument(
    "--stop-reward", type=float, default=600.0, help="Reward at which we stop training."
)'''

pickle_dir = Path('C:/users/water/documents/datasets/stock_data')
df = pd.read_pickle(pickle_dir / 'SPY_minute_2012-08-22_built_gcp.pkl')
trading_df = add_indicators(df)
trading_df = trading_df.fillna(0)
env = StockEnv(trading_df)

if __name__ == "__main__":
    ray.init()
    args = parser.parse_args()
    config = {
        "environment": env,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "framework": args.framework
    }
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total" : args.stop_timesteps
    }
    resources = PPO.default_resource_request(config)
    tuner = tune.Tuner(
        "PPO", param_space=config, run_config=air.RunConfig(stop=stop, verbose=1)
    )
    tuner.fit()