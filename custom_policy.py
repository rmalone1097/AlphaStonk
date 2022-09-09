import gym
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNNLSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim:int=1024):
        super(CustomCNNLSTM, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnnlstm = nn.Sequential(
            nn.Conv1d(n_input_channels)
        )