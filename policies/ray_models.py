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

                input_rows = obs_space.shape[0]
                input_features = 5
                num_filters = 16
                vector_length = 21

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
                self.FC = nn.Sequential(
                        nn.Linear(input_rows*num_filters + vector_length, num_outputs)
                )
        
        @override(TorchModelV2)
        def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType):
                obs_slice = input_dict['obs']['slice']
                obs_vector = input_dict['obs']['vector']

                obs_slice = torch.permute(obs_slice, (0, 2, 1))
                cnn_output = self.cnn(obs_slice)
                logits = self.FC(torch.cat((cnn_output, obs_vector), dim=1))

                return logits, state