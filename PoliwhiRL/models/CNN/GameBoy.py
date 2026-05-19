# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F


def _make_norm(num_channels, max_groups=8):
    """GroupNorm with a sensible group count. BatchNorm is a known PPO
    pitfall: rollout-time batches (size N) differ from update-time batches
    (size W*N), and the running stats accumulate across mismatched dists,
    silently biasing the policy. GroupNorm is batch-size-independent."""
    num_groups = min(max_groups, num_channels)
    # Walk down until num_channels is divisible by num_groups.
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class GameBoyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GameBoyBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        # Attribute is `norm` rather than `bn` so old BatchNorm-keyed
        # checkpoints get cleanly skipped on a non-strict load.
        self.norm = _make_norm(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class GameBoyOptimizedCNN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(GameBoyOptimizedCNN, self).__init__()
        self.block1 = GameBoyBlock(input_shape[0], 16)
        self.block2 = GameBoyBlock(16, 32)
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            sample_output = self.block2(self.block1(sample_input))
            self.flat_features = sample_output.view(1, -1).size(1)
        self.fc = nn.Linear(self.flat_features, output_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(-1, self.flat_features)
        return F.relu(self.fc(x))
