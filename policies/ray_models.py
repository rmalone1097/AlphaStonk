import numpy as np
import gymnasium as gym

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

                input_rows = 780
                input_features = 5
                num_filters = 16
                vector_length = 22
                output_features = 256

                self.cnn = nn.Sequential(
                        nn.Conv1d(input_features, num_filters, kernel_size=7, padding='same'),
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
                        nn.Linear(output_features*2, 3)
                )
                self.value_net = nn.Sequential(
                        nn.Linear(output_features*2, 1)
                )

        
        @override(TorchModelV2)
        def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType):
                obs_slice = input_dict['obs']['slice']
                obs_vector = input_dict['obs']['vector']

                obs_slice = torch.permute(input_dict['obs']['slice'], (0, 2, 1))
                slice_output = self.FC_slice(self.cnn(obs_slice))
                vector_output = self.FC_vector(input_dict['obs']['vector'])
                self._features = torch.cat((slice_output, vector_output), dim=1)
                self._logits = self.logits_net(self._features)

                return self._logits, state
        
        @override(TorchModelV2)
        def value_function(self) -> TensorType:
                out = self.value_net(self._features).squeeze(1)
                
                return out