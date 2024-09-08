# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class CuriosityModel(nn.Module):
    def __init__(self, state_shape, action_size):
        super(CuriosityModel, self).__init__()
        self.action_size = action_size
        self.state_encoder = nn.Sequential(
            nn.Linear(state_shape[0], 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(64 + action_size, 128), nn.ReLU(), nn.Linear(128, 64)
        )

    def forward(self, state, action):
        encoded_state = self.state_encoder(state)
        action_one_hot = F.one_hot(action, num_classes=self.action_size).float()
        predicted_next_state = self.forward_model(
            torch.cat([encoded_state, action_one_hot], dim=1)
        )
        return predicted_next_state
