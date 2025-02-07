from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import CSVOutputFormat
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from utils.data_utils import *
from stock_env import StockEnv
import time
import os
from pathlib import Path

from policies.CNN_custom_policy import CustomCNN
#from policies.custom_combined_extractor import CustomCombinedExtractor
from policies.omni_scale_CNN import CustomCombinedExtractor

#pickle_dir = Path.home()
#pickle_dir = Path('C:/users/water/desktop')
pickle_dir = Path('C:/users/water/documents/datasets/stock_data')
df = pd.read_pickle(pickle_dir / 'SPY_minute_2012-08-22_built_gcp.pkl')
trading_df = add_indicators(df)
trading_df = trading_df.fillna(0)
env = StockEnv(trading_df)
#check_env(env)

cwd = os.getcwd()
models_dir = pickle_dir / Path('models\\PPOflat' + str(time.time()))
logs_dir = pickle_dir / Path('logs\\PPOflat'+ str(time.time()))

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

env.reset()

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=100)
)

#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logs_dir)
#model = PPO.load(cwd + '\\models\\PPOflat2epoch\\982800', env=env)
model = PPO('MultiInputPolicy', env, ent_coef=0.001, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=logs_dir, batch_size=64, seed=4)

print(model.policy)

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.cum_rew_1 = 0
    
    def _on_rollout_end(self) -> None:
        self.logger.record("rollout/cum_rew_1", self.cum_rew_1)

        self.cum_rew_1 = 0

    def _on_step(self) -> bool:
        self.cum_rew_1 += self.training_env.get_attr("reward")[0]
        self.logger.record('variables/reward', self.training_env.get_attr('reward')[0])
        self.logger.record('variables/action', self.training_env.get_attr('action')[0])
        self.logger.record('variables/current_price', self.training_env.get_attr('current_price')[0])
        self.logger.record('variables/win_ratio', self.training_env.get_attr('win_ratio')[0])
        self.logger.record('variables/long_ratio', self.training_env.get_attr('long_ratio')[0])
        self.logger.record('variables/streak', self.training_env.get_attr('streak')[0])
        self.logger.record('variables/holding_time', self.training_env.get_attr('holding_time')[0])
        #self.logger.record('variables/roi', self.training_env.get_attr('roi')[0])
        self.logger.record('variables/total_roi', self.training_env.get_attr('total_roi')[0])
        self.logger.record('variables/average_roi', self.training_env.get_attr('average_roi')[0])
        self.logger.record('variables/zero_ratio', self.training_env.get_attr('zero_ratio')[0])
        self.logger.record('variables/average_holding_time', self.training_env.get_attr('average_holding_time')[0])
        self.logger.record('variables/average_long_roi', self.training_env.get_attr('average_long_roi')[0])
        self.logger.record('variables/average_short_roi', self.training_env.get_attr('average_short_roi')[0])
        #self.logger.record('reward', reward)
        return True

# Configuration of CSV logger
#csv_logger = configure(folder = '.', format_strings=["stdout", "csv", "tensorboard"])
#model.set_logger(csv_logger)

TIMESTEPS = 100000
for i in range(1, 100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f'PPOflat', callback=TensorboardCallback())
    model.save(models_dir / str(time.time()))