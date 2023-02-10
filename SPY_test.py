from pathlib import Path
from stock_env import StockEnv

from ray.rllib.algorithms.algorithm import Algorithm

algo = Algorithm.from_checkpoint(Path.home() / "ray_results" / "PPO" / "PPO_StockEnv_a8b6a_00000_0_2023-02-10_13-01-08" / "checkpoint_000003")