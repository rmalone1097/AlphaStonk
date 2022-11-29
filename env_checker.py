from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from data_utils import *
from stock_env import StockEnv
import time
import os

from data_utils import add_indicators

pickle_dir = 'C:\\Users\\water\\Documents\\datasets\\stock_data'

df = pd.read_pickle(pickle_dir + '\\SPY_minute_2012-08-22_built.pkl')
trading_df = add_indicators(df)

env = StockEnv(trading_df)
check_env(env)