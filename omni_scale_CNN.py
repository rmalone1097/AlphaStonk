import gym
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# List of prime numbers as int
with open('prime_numbers.txt') as f:
    lines = f.readlines()
prime_list = lines[0].split(",")
prime_list = [int(n) for n in prime_list]

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=1024, prime_list=prime_list):
        # features_dim is specifically for the output dim of the CNN. Will have to be added to output
        # for vector extractor for total featuers_dim, which is self.features_dim later on
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

        n_input_features = observation_space['slice'].shape[0]
        n_input_channels = observation_space['slice'].shape[1]

        self.kernels_1 = nn.ModuleList()
        self.kernels_2 = nn.ModuleList(nn.Conv1d(1, 1, kernel_size=1, padding='same'), nn.Conv1d(1, 1, kernel_size=2, padding='same'))
        conv_1_length = 0

        for p in prime_list:
            if p >= n_input_features // 2:
                conv = nn.Conv1d(1, 1, kernel_size=p, padding='same')
                self.kernels_1.append(conv)
                conv_1_length += n_input_features - p + 1
        
        # Not including convolutions
        self.os_block_1 = nn.Sequential(
            nn.BatchNorm1d(conv_1_length),
            nn.ReLU()
        )

        conv_2_length = (n_input_features - 1 + 1) + (n_input_features - 2 + 1)
        self.os_block_2 = nn.Sequential(
            nn.BatchNorm1d(conv_2_length),
            nn.ReLU
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
                extractors[key] = nn.Sequential(self.linear)
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

                # Batch, length, channel
                observations['slice']  = torch.permute(observations['slice'], (0, 2, 1))
                obs_clone = torch.clone(observations['slice'])

                # iterate through channels and implement OS block on all
                n_channels = obs_clone.size(dim=2)
                for  i in range (n_channels):
                    x = [k(obs_clone[:, :, i]) for k in self.kernels_1]
                    x = torch.cat(x, dim=1)
                    x = self.os_block_1(x)
                    x = [k(x) for k in self.kernels_1]
                    x = torch.cat(x, dim=1)
                    x = self.os_block_1(x)
                    x = [k(x) for k in self.kernels_2]
                    x = torch.cat(x, dim=1)
                    x = self.os_block_2(x)

                    # os_out should have shape [batch, length]
                    if not os_out:
                        os_out = x
                    else:
                        os_out = torch.cat((os_out, x), dim=1)
                
                # Take concatenated output from previous os block and run it though a new os block
                x = [k(os_out) for k in self.kernels_1]
                x = torch.cat(x, dim=1)
                x = self.os_block_1(x)
                x = [k(x) for k in self.kernels_1]
                x = torch.cat(x, dim=1)
                x = self.os_block_1(x)
                x = [k(x) for k in self.kernels_2]
                x = torch.cat(x, dim=1)
                x = self.os_block_2(x)

            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)