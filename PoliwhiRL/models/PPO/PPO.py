# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer


class RobustLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(RobustLayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(-1, keepdim=True) + self.eps)
        x = (x - mean) / std
        x = x.clamp(-10, 10)  # Clip values to prevent extremes
        return self.weight * x + self.bias


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (RobustLayerNorm, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)


class FeatureNN(nn.Module):
    def __init__(self, input_dim, vision=True, base_channels=32, num_conv_layers=3):
        super(FeatureNN, self).__init__()
        self.vision = vision

        if self.vision:
            layers = []
            in_channels = input_dim[0]
            for i in range(num_conv_layers):
                out_channels = base_channels * (2**i)
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            stride=1 if i == 0 else 2,
                            padding=1,
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                    ]
                )
                in_channels = out_channels

            layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # Global average pooling
            layers.append(nn.Flatten())

            self.feature_layer = nn.Sequential(*layers)
        else:
            self.feature_layer = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )

        # Apply weight initialization
        self.apply(init_weights)

    def feature_size(self, input_dim):
        if self.vision:
            return self.feature_layer(torch.zeros(1, *input_dim)).view(1, -1).size(1)
        else:
            return 1024  # The output size of the last linear layer

    def forward(self, x):
        return self.feature_layer(x)


class PPOModel(nn.Module):
    def __init__(
        self,
        input_dim,
        num_actions,
        vision=True,
        num_transformer_layers=4,
        lstm_hidden_size=256,
    ):
        super(PPOModel, self).__init__()
        self.vision = vision
        self.FeatureCNN = FeatureNN(input_dim, vision)
        self.num_actions = num_actions
        feature_size = self.FeatureCNN.feature_size(input_dim)

        self.layer_norm1 = RobustLayerNorm(feature_size)

        encoder_layer = TransformerEncoderLayer(
            d_model=feature_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        self.layer_norm2 = RobustLayerNorm(feature_size)

        self.lstm = nn.LSTM(feature_size, lstm_hidden_size, batch_first=True)

        self.layer_norm3 = RobustLayerNorm(lstm_hidden_size)

        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128), nn.ReLU(), nn.Linear(128, num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        self.apply(init_weights)

    def forward(self, x, hidden_state=None):
        if self.vision:
            batch_size, seq_len, channels, height, width = x.size()
            x = x.view(batch_size * seq_len, channels, height, width)
        else:
            batch_size, seq_len, input_dim = x.size()
            x = x.view(batch_size * seq_len, input_dim)

        x = x.float() / 255.0
        features = self.FeatureCNN(x)
        features = features.clamp(-10, 10)  # Clip values

        features = features.view(batch_size, seq_len, -1)
        residual = features
        features = self.layer_norm1(features)

        features = self.transformer_encoder(features)
        features = features.clamp(-10, 10)  # Clip values

        features = features + residual
        features = self.layer_norm2(features)

        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(features)
        else:
            lstm_out, hidden_state = self.lstm(features, hidden_state)

        lstm_out = self.layer_norm3(lstm_out)
        temperature = 1.0  # Adjust this value as needed
        logits = self.actor(lstm_out) / temperature
        logits = logits.clamp(-10, 10)  # Clip values
        action_probs = F.softmax(logits, dim=-1)

        value_estimates = self.critic(lstm_out)

        return action_probs, value_estimates, hidden_state
