# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class TransformerDQN(nn.Module):
    def __init__(self, input_shape, action_size, d_model=64, nhead=4, num_layers=2):
        super(TransformerDQN, self).__init__()
        self.action_size = action_size
        self.input_shape = input_shape

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # Calculate the size of flattened features after conv layers
        conv_out_size = self._get_conv_out_size(input_shape)

        # Linear layer to project to d_model dimensions
        self.fc_pre = nn.Linear(conv_out_size, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output layer
        self.fc_out = nn.Linear(d_model, action_size)

    def _get_conv_out_size(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        return int(torch.prod(torch.tensor(o.shape)))

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        # Reshape for conv layers
        x = x.view(batch_size * seq_len, c, h, w)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the output
        x = x.view(batch_size * seq_len, -1)

        # Project to d_model dimensions
        x = self.fc_pre(x)

        # Reshape back to (batch_size, seq_len, d_model)
        x = x.view(batch_size, seq_len, -1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through Transformer
        x = self.transformer_encoder(x)

        # Output layer for each step in the sequence
        q_values = self.fc_out(x)

        return q_values


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x
