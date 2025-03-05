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


class ExplorationEncoder(nn.Module):
    def __init__(self, d_model, history_length=5):
        super(ExplorationEncoder, self).__init__()
        # Input features: visit count + history indicators
        input_features = 1 + history_length
        self.fc1 = nn.Linear(input_features, 16)
        self.fc2 = nn.Linear(16, 32)
        self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
        self.fc_out = nn.Linear(32, d_model)

    def forward(self, x):
        # x shape: [batch_size, num_locations, 1+history_length]
        # where first column is visit count, remaining columns are history indicators
        _ = x.size(0)
        x = torch.relu(self.fc1(x))  # [batch_size, num_locations, 16]
        x = torch.relu(self.fc2(x))  # [batch_size, num_locations, 32]

        # Self-attention over locations
        attn_output, _ = self.attention(x, x, x)

        # Global pooling over locations
        x = attn_output.mean(dim=1)  # [batch_size, 32]

        # Project to d_model
        x = self.fc_out(x)  # [batch_size, d_model]
        return x


class PPOTransformer(nn.Module):
    def __init__(
        self, input_shape, action_size, d_model=32, nhead=4, num_layers=2, **kwargs
    ):
        super(PPOTransformer, self).__init__()
        self.action_size = action_size
        self.input_shape = input_shape
        self.d_model = d_model

        self.flexible_input = FlexibleInputLayer(input_shape, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)

        # Get history length from config or use default
        history_length = kwargs.get("ppo_exploration_history_length", 5)
        # Exploration memory encoder
        self.exploration_encoder = ExplorationEncoder(
            d_model, history_length=history_length
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc_actor = nn.Linear(
            d_model * 2, action_size
        )  # *2 for concatenated exploration features
        self.fc_critic = nn.Linear(
            d_model * 2, 1
        )  # *2 for concatenated exploration features

    def forward(self, x, exploration_tensor=None):
        batch_size, seq_len = x.size()[:2]
        x = x.view(batch_size * seq_len, *self.input_shape)
        x = self.flexible_input(x)
        x = x.view(batch_size, seq_len, self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # Use the last output of the sequence
        x = x[:, -1, :]

        # Process exploration memory if provided
        if exploration_tensor is not None:
            # Ensure exploration_tensor is the right shape [batch_size, num_locations, 1+history_length]
            if (
                len(exploration_tensor.shape) == 3
                and exploration_tensor.shape[0] == batch_size
            ):
                exploration_features = self.exploration_encoder(exploration_tensor)
                # Concatenate with transformer output
                combined_features = torch.cat([x, exploration_features], dim=1)
            else:
                # If exploration tensor is not properly shaped, just duplicate the transformer output
                combined_features = torch.cat([x, x], dim=1)
        else:
            # If no exploration tensor, just duplicate the transformer output
            combined_features = torch.cat([x, x], dim=1)

        action_probs = torch.softmax(self.fc_actor(combined_features), dim=-1)
        value = self.fc_critic(combined_features)

        return action_probs, value
