import torch
import torch.nn as nn
from torch.nn import functional as F

class GameBoyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GameBoyBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

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