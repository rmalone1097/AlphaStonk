import gym
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim:int=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[1]
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, 32, kernel_size=7, padding='same'),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )

        self.linear = nn.Sequential(
            nn.Linear(1950*64 + n_input_channels, features_dim), 
            nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations  = torch.permute(observations, (0, 2, 1))
        # Flatten last observation and add it to features
        last_obs = observations[:, :, -1]
        #print(last_obs.shape)
        features = self.cnn(observations)
        features = torch.cat((features, last_obs), 1)
        #print(features.shape)
        return self.linear(features)
        #return self.linear(self.cnn(obs))