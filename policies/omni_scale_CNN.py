import gym
import torch
import torch.nn as nn

from policies.wensi import *
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# List of prime numbers as int
with open(r'utils//prime_numbers.txt') as f:
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

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "slice":
                #TODO: What are quarter_or_half and parameter_number_of_layer_list?
                input_shape = self.n_input_features
                n_class = features_dim
                start_kernel_size = 1
                max_kernel_size = 389
                quarter_or_half = 4
                parameter_number_of_layer_list = [106*128,103*128*256 + 3*256*128]
                receptive_field_shape= min(int(input_shape/quarter_or_half),max_kernel_size)
                layer_parameter_list = generate_layer_parameter_list(start_kernel_size,receptive_field_shape,parameter_number_of_layer_list,in_channel = 1)

                model = OS_CNN(parameter_number_of_layer_list, layer_parameter_list, n_class, self.n_input_channels, True)
                extractors[key] = model
                total_concat_size += features_dim

                print('Slice shape: ', subspace.shape)
            elif key == "vector":
                # Run through a simple MLP
                #TODO: More than one linear layer?
                hidden_nodes = subspace.shape[0]
                extractors[key] = nn.Linear(subspace.shape[0], hidden_nodes)
                total_concat_size += hidden_nodes

                print('Vector shape: ', subspace.shape)

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

            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)