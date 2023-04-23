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

from envs.stock_env_train import StockEnv
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
        .training(
            gamma=0.99,
            entropy_coeff=0.001,
            num_sgd_iter=10,
            vf_loss_coeff=1e-5,
            model={
                "use_attention": True,
                "max_seq_len": 30,
                "attention_num_transformer_units": 1,
                "attention_dim":32,
                "attention_memory_inference": 10,
                "attention_memory_training":10,
                "attention_num_heads": 1,
                "attention_head_dim": 32,
                "attention_position_wise_mlp_dim": 32,
            },
        )
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
            checkpoint_config=air.CheckpointConfig(checkpoint_score_attribute='episode_reward_mean',
                                                   checkpoint_score_order='max',
                                                   num_to_keep=5)
        )
    )
    results = tuner.fit()