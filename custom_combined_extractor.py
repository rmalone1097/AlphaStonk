import gym
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=1024):
        # features_dim is specifically for the output dim of the CNN. Will have to be added to output
        # for vector extractor for total featuers_dim, which is self.features_dim later on
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space['slice'].shape[1]
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, 32, kernel_size=7, padding='same'),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Flatten()
        )

        self.linear = nn.Sequential(
            nn.Linear(observation_space['slice'].shape[0]*64, features_dim), 
            nn.ReLU())

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "slice":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(self.cnn, self.linear)
                total_concat_size += features_dim
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == 'slice':
                observations['slice']  = torch.permute(observations['slice'], (0, 2, 1))
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)