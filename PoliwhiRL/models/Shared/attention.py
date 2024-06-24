# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x):
        attention_scores = self.attention_layer(x)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_features = x * attention_weights
        return weighted_features
