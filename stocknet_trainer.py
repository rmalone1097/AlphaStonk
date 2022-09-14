from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from data_utils import *
from stock_env import StockEnv
import time
import os

from CNN_custom_policy import CustomCNN

pickle_dir = 'C:\\Users\\water\\documents\\datasets\\stock_data\\'
df = pd.read_pickle(pickle_dir + 'SPY_minute_2012-08-22_built.pkl')
trading_df = add_indicators(df)
env = StockEnv(trading_df)
#check_env(env)

cwd = os.getcwd()
models_dir = cwd + '\\models\\PPOflat' + str(time.time())
logs_dir = cwd + '\\logs\\PPOflat'+ str(time.time())

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

env.reset()

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=1024)
)

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=logs_dir, policy_kwargs=policy_kwargs)

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.cum_rew_1 = 0
    
    def _on_rollout_end(self) -> None:
        self.logger.record("rollout/cum_rew_1", self.cum_rew_1)

        self.cum_rew_1 = 0

    def _on_step(self) -> bool:
        self.cum_rew_1 += self.training_env.get_attr("reward")[0]
        net_worth = self.training_env.get_attr('net_worth')[0]
        self.logger.record('variables/net_worth', net_worth)
        self.logger.record('variables/action', self.training_env.get_attr('action')[0])
        self.logger.record('variables/current_price', self.training_env.get_attr('current_price')[0])
        #self.logger.record('reward', reward)
        return True

TIMESTEPS = 196560*5
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f'PPOflat', callback=TensorboardCallback())
model.save(models_dir + '\\' + str(TIMESTEPS))