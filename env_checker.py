from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from data_utils import *
from stock_env import StockEnv
import time

import os

df = df_builder('SPY_minute_2020-08-17.csv')
env = StockEnv(df)
check_env(env)