# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from PoliwhiRL.models.CNN.GameBoy import GameBoyOptimizedCNN
from PoliwhiRL.models.transformers.positional_encoding import PositionalEncoding


class FlexibleInputLayer(nn.Module):
    def __init__(self, input_shape, d_model):
        super(FlexibleInputLayer, self).__init__()
        self.input_shape = input_shape
        self.d_model = d_model

        if len(input_shape) == 3:  # [C, H, W]
            self.cnn = GameBoyOptimizedCNN(input_shape, d_model)
        elif len(input_shape) == 2:  # [X, Y]
            self.fc = nn.Linear(input_shape[0] * input_shape[1], d_model)
        else:
            raise ValueError("Unsupported input shape")

    def forward(self, x):
        if len(self.input_shape) == 3:
            return self.cnn(x)
        else:
            return torch.relu(self.fc(x.view(x.size(0), -1)))


class PPOTransformer(nn.Module):
    def __init__(self, input_shape, action_size, d_model=32, nhead=4, num_layers=2):
        super(PPOTransformer, self).__init__()
        self.action_size = action_size
        self.input_shape = input_shape
        self.d_model = d_model

        self.flexible_input = FlexibleInputLayer(input_shape, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc_actor = nn.Linear(d_model, action_size)
        self.fc_critic = nn.Linear(d_model, 1)

    def forward(self, x):
        batch_size, seq_len = x.size()[:2]
        x = x.view(batch_size * seq_len, *self.input_shape)
        x = self.flexible_input(x)
        x = x.view(batch_size, seq_len, self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # Use the last output of the sequence for both actor and critic
        x = x[:, -1, :]

        action_probs = torch.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)

        return action_probs, value
