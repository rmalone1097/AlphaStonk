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

        self.n_input_features = observation_space['slice'].shape[0]
        self.n_input_channels = observation_space['slice'].shape[1]

        self.rf = 60

        self.kernels_1 = nn.ModuleList()
        conv_1_length = 0

        for p in prime_list:
            if p <= self.rf // 2:
                conv = nn.Conv1d(1, 1, kernel_size=p, padding='same')
                self.kernels_1.append(conv)
                conv_1_length += self.n_input_features - p + 1
        
        self.kernels_2 = nn.ModuleList()
        conv_2_length = 0

        for p in prime_list:
            if p <= self.rf // 2:
                conv = nn.Conv1d(1, 1, kernel_size=p, padding='same')
                self.kernels_2.append(conv)
                conv_2_length += conv_1_length - p + 1
        
        self.kernels_3 = nn.ModuleList([nn.Conv1d(1, 1, kernel_size=1, padding='same'), nn.Conv1d(1, 1, kernel_size=2, padding='same')])
        conv_3_length = conv_2_length + conv_2_length - 1

        concat_1 = conv_3_length * self.n_input_channels

        self.kernels_4 = nn.ModuleList()
        conv_4_length = 0

        for p in prime_list:
            if p <= self.rf // 2:
                conv = nn.Conv1d(1, 1, kernel_size=p, padding='same')
                self.kernels_4.append(conv)
                conv_4_length += concat_1 - p + 1
        
        self.kernels_5 = nn.ModuleList()
        conv_5_length = 0

        for p in prime_list:
            if p <= self.rf // 2:
                conv = nn.Conv1d(1, 1, kernel_size=p, padding='same')
                self.kernels_5.append(conv)
                conv_5_length += conv_4_length - p + 1
        
        self.kernels_6 = nn.ModuleList([nn.Conv1d(1, 1, kernel_size=1, padding='same'), nn.Conv1d(1, 1, kernel_size=2, padding='same')])
        conv_6_length = conv_5_length + conv_5_length - 1
        
        # Not including convolutions
        self.os_block_1 = nn.Sequential(
            nn.BatchNorm1d(conv_1_length),
            nn.ReLU()
        )

        self.os_block_2 = nn.Sequential(
            nn.BatchNorm1d(conv_2_length),
            nn.ReLU()
        )

        self.os_block_3 = nn.Sequential(
            nn.BatchNorm1d(conv_3_length),
            nn.ReLU()
        )

        self.os_block_4 = nn.Sequential(
            nn.BatchNorm1d(conv_4_length),
            nn.ReLU()
        )

        self.os_block_5 = nn.Sequential(
            nn.BatchNorm1d(conv_5_length),
            nn.ReLU()
        )

        self.os_block_6 = nn.Sequential(
            nn.BatchNorm1d(conv_6_length),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(conv_6_length, features_dim), 
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
                for i in range (self.n_input_channels):
                    x = [k(obs_clone[:, :, i]) for k in self.kernels_1]
                    x = torch.cat(x, dim=1)
                    x = self.os_block_1(x)
                    x = [k(x) for k in self.kernels_2]
                    x = torch.cat(x, dim=1)
                    x = self.os_block_2(x)
                    x = [k(x) for k in self.kernels_3]
                    x = torch.cat(x, dim=1)
                    x = self.os_block_3(x)

                    # os_out should have shape [batch, length]
                    if not os_out:
                        os_out = x
                    else:
                        os_out = torch.cat((os_out, x), dim=1)
                
            # Take concatenated output from previous os block and run it though a new os block
            x = [k(os_out) for k in self.kernels_4]
            x = torch.cat(x, dim=1)
            x = self.os_block_4(x)
            x = [k(x) for k in self.kernels_5]
            x = torch.cat(x, dim=1)
            x = self.os_block_5(x)
            x = [k(x) for k in self.kernels_6]
            x = torch.cat(x, dim=1)
            x = self.os_block_6(x)

            observations['slice'] = x

            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)