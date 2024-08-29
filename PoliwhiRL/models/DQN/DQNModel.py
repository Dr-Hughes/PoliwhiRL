# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class DeepQNetworkModel(nn.Module):
    def __init__(self, input_shape, action_size, lstm_size=32, fc_size=64):
        super(DeepQNetworkModel, self).__init__()
        self.input_shape = input_shape
        self.action_size = action_size
        self.lstm_size = lstm_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of flattened features after conv layers
        conv_out_size = self._get_conv_out_size(input_shape)

        # Fully connected layer before LSTM
        self.fc_pre = nn.Linear(conv_out_size, fc_size)

        # LSTM layer
        self.lstm = nn.LSTM(fc_size, lstm_size, batch_first=True)

        # Fully connected layers after LSTM
        self.fc_post = nn.Sequential(
            nn.Linear(lstm_size, fc_size), nn.ReLU(), nn.Linear(fc_size, action_size)
        )

    def _get_conv_out_size(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x, hidden_state):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Reshape and process with conv layers
        x = x.view(batch_size * seq_len, *self.input_shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size * seq_len, -1)

        x = F.relu(self.fc_pre(x))
        x = x.view(batch_size, seq_len, -1)

        x, hidden_state = self.lstm(x, hidden_state)

        x = self.fc_post(x.contiguous().view(batch_size * seq_len, -1))
        x = x.view(batch_size, seq_len, self.action_size)

        return x, hidden_state

    def init_hidden(self, batch_size):
        return (
            torch.zeros(1, batch_size, self.lstm_size),
            torch.zeros(1, batch_size, self.lstm_size),
        )
