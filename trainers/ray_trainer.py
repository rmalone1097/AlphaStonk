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
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy import Policy
from ray.rllib.env import BaseEnv
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker

from policies.ray_models import *
from envs.stock_env import BaseEnv
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

tickers = ['SPY']
cande_length = 1
start_stamp = 928761600
end_stamp = 1676581140
data_path = Path.home() / 'data'
full_train_df, obs_train_df, _, _ = prepare_state_df(tickers, data_path, 2206200)
print(full_train_df)
print(obs_train_df)

class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],episode: MultiAgentEpisode, **kwargs):
        print("episode {} started".format(episode.episode_id))
        episode.user_data["total_roi"] = []
        episode.hist_data["total_roi"] = []

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, **kwargs):
        total_roi = abs(base_env.total_roi)
        episode.user_data["total_roi"].append(total_roi)

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode, **kwargs):
        total_roi = np.mean(episode.user_data["total_roi"])
        print("episode {} ended with length {} and pole angles {}".format(episode.episode_id, episode.length, total_roi))
        
        episode.custom_metrics["total_roi"] = total_roi
        episode.hist_data["total_roi"] = episode.user_data["total_roi"]

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        print("returned sample batch of size {}".format(samples.count))

    def on_train_result(self, trainer, result: dict, **kwargs):
        print("trainer.train() result: {} -> {} episodes".format(trainer, result["episodes_this_iter"]))
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

    def on_postprocess_trajectory(self, worker: RolloutWorker, episode: MultiAgentEpisode, agent_id: str, policy_id: str, policies: Dict[str, Policy], postprocessed_batch: SampleBatch, original_batches: Dict[str, SampleBatch], **kwargs):
        print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1

if __name__ == "__main__":

    ModelCatalog.register_custom_model("simple_cnn", SimpleCNN)

    ray.init(num_gpus=1)
    args = parser.parse_args()
    config = (
        PPOConfig()
        .environment(BaseEnv, env_config={"full_df": full_train_df,
                                           "obs_df": obs_train_df,
                                           "tickers": tickers,
                                           "print": True})
        .framework(args.framework)
        .training(
            model={
                "custom_model": "simple_cnn",
            }
        )
        .rollouts(num_rollout_workers=20)
        .resources(num_gpus=1)
        .callbacks(MyCallbacks)
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