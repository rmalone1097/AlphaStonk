import numpy as np
import gymnasium as gym
from policies.wensi import *

from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch

torch, nn = try_import_torch()

class SimpleCNN(TorchModelV2, nn.Module):
        def __init__(
                self,
                obs_space: gym.spaces.Space,
                action_space: gym.spaces.Space,
                num_outputs: int,
                model_config: ModelConfigDict,
                name: str,
        ):
                print('Using SimpleCNN policy')
                # num placeholder for CNN output calculation
                TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
                nn.Module.__init__(self)

                self.num_tickers = 3
                logit_count = 1 + 2*self.num_tickers
                input_rows = 780
                self.input_features = 5
                num_filters = 16
                vector_length = 6 + 16*self.num_tickers
                output_features = 256

                self.cnn = nn.Sequential(
                        nn.Conv1d(self.input_features, num_filters, kernel_size=7, padding='same'),
                        nn.ReLU(),
                        nn.Conv1d(num_filters, num_filters, kernel_size=5, padding='same'),
                        nn.ReLU(),
                        nn.Conv1d(num_filters, num_filters, kernel_size=3, padding='same'),
                        nn.ReLU(),
                        nn.Conv1d(num_filters, num_filters, kernel_size=3, padding='same'),
                        nn.ReLU(),
                        nn.Flatten()
                )
                self.FC_slice = nn.Sequential(
                        nn.Linear(input_rows*num_filters, output_features),
                        nn.Tanh(),
                        nn.Linear(output_features, output_features),
                        nn.Tanh()
                )
                self.FC_vector = nn.Sequential(
                        nn.Linear(vector_length, output_features),
                        nn.Tanh(),
                        nn.Linear(output_features, output_features),
                        nn.Tanh()
                )
                self.logits_net = nn.Sequential(
                        nn.Linear(output_features+output_features*self.num_tickers, logit_count)
                )
                self.value_net = nn.Sequential(
                        nn.Linear(output_features+output_features*self.num_tickers, 1)
                )

        
        @override(TorchModelV2)
        def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType):
                obs_slice = torch.permute(input_dict['obs']['slice'], (0, 2, 1))
                print(obs_slice.shape)

                slice_out_list = []
                # Iterate through tickers and run separate slices through os_cnn
                for i in range(self.num_tickers):
                        slice_out_list.append(self.FC_slice(self.cnn(obs_slice[:, i*self.input_features:(i+1)*self.input_features, :])))
                slices_output = torch.cat((slice_out_list), dim=1)

                vector_output = self.FC_vector(input_dict['obs']['vector'])
                self._features = torch.cat((slices_output, vector_output), dim=1)
                self._logits = self.logits_net(self._features)

                return self._logits, state
        
        @override(TorchModelV2)
        def value_function(self) -> TensorType:
                out = self.value_net(self._features).squeeze(1)
                
                return out

class osCNN(TorchModelV2, nn.Module):
        def __init__(
                self,
                obs_space: gym.spaces.Space,
                action_space: gym.spaces.Space,
                num_outputs: int,
                model_config: ModelConfigDict,
                name: str
        ):
                print('Using osCNN policy')
                TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
                nn.Module.__init__(self)

                # Hard coding these for now, will have to make dynamic at some point
                input_rows = 780
                self.input_features = 5
                self.num_tickers = 3
                logit_count = 1 + 2*self.num_tickers
                vector_length = 6 + 16*self.num_tickers
                output_os_features = 256
                start_kernel_size = 1
                max_kernel_size = 197
                quarter_or_half = 4
                parameter_starter = 37

                output_features = 256

                parameter_number_of_layer_list = [parameter_starter*128,parameter_starter*128*256]
                receptive_field_shape= min(int(input_rows/quarter_or_half),max_kernel_size)
                layer_parameter_list = generate_layer_parameter_list(start_kernel_size,receptive_field_shape,parameter_number_of_layer_list,in_channel = 1)

                # Input to OS CNN has 5 features per stock
                self.os_cnn = OS_CNN(parameter_number_of_layer_list, layer_parameter_list, output_os_features, self.input_features, True)
                self.FC_slice = nn.Sequential(
                        nn.Linear(output_os_features, output_features),
                        nn.Tanh(),
                        nn.Linear(output_features, output_features),
                        nn.Tanh()
                )
                self.FC_vector = nn.Sequential(
                        nn.Linear(vector_length, output_features),
                        nn.Tanh(),
                        nn.Linear(output_features, output_features),
                        nn.Tanh()
                )
                self.logits_net = nn.Sequential(
                        nn.Linear(output_features+output_features*self.num_tickers, logit_count)
                )
                self.value_net = nn.Sequential(
                        nn.Linear(output_features+output_features*self.num_tickers, 1)
                )
        
        @override(TorchModelV2)
        def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType):
                # Batch, channel, length
                obs_slice = torch.permute(input_dict['obs']['slice'], (0, 2, 1))

                slice_out_list = []
                # Iterate through tickers and run separate slices through os_cnn
                for i in range(self.num_tickers):
                        slice_out_list.append(self.os_cnn(obs_slice[:, i*self.input_features:(i+1)*self.input_features, :]))
                slices_output = torch.cat((slice_out_list), dim=1)

                vector_output = self.FC_vector(input_dict['obs']['vector'])
                self._features = torch.cat((slices_output, vector_output), dim=1)
                self._logits = self.logits_net(self._features)

                return self._logits, state

        @override(TorchModelV2)
        def value_function(self) -> TensorType:
                out = self.value_net(self._features).squeeze(1)
                
                return out