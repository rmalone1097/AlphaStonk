import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from envs.stock_env import StockEnv

from ray.rllib.algorithms.algorithm import Algorithm

test_df = 0
algo = Algorithm.from_checkpoint(Path.home() / "ray_results" / "PPO" / "PPO_StockEnv_a8b6a_00000_0_2023-02-10_13-01-08" / "checkpoint_000003")

def test_algo(algo, test_df):
    tensor = test_df.to_numpy()
    position_log = 0

    for i in range(tensor.shape[0]):
        obs = tensor[i, 2:20]
        action = algo.compute_single_action(obs)
