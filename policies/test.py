import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_features):
        super(CustomModel, self).__init__()
        self.num_features = num_features
        
        # CNN layers for time series data
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Fully connected layers for value network
        self.fc_value = nn.Linear(32, 64)
        self.fc_value_out = nn.Linear(64, 1)
        
        # Fully connected layers for logits network
        self.fc_logits = nn.Linear(32 + num_features, 64)
        self.fc_logits_out = nn.Linear(64, num_outputs)
        
    def forward(self, inputs):
        time_series_data = inputs[:, :-self.num_features]  # Extract time series data
        features = inputs[:, -self.num_features:]  # Extract additional features
        
        # Process time series data with CNN
        time_series_data = time_series_data.unsqueeze(1)  # Add channel dimension
        x = self.conv1(time_series_data)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers for value network
        value = F.relu(self.fc_value(x))
        value = self.fc_value_out(value)
        
        # Concatenate features with CNN output for logits network
        x = torch.cat((x, features), dim=1)
        
        # Fully connected layers for logits network
        logits = F.relu(self.fc_logits(x))
        logits = self.fc_logits_out(logits)
        
        return logits, value

