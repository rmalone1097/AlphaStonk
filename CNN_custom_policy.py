import gym
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim:int=1024):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[1]
        #print(type(n_input_channels))
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, 32, 1, padding='same'),
            nn.Tanh(),
            nn.MaxPool1d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1950*32, 1024),
            nn.ReLU()
        )

        self.linear = nn.Sequential(nn.Linear(1950*32, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations  = torch.permute(observations, (0, 2, 1))
        return self.cnn(observations)
        #return self.linear(self.cnn(obs))