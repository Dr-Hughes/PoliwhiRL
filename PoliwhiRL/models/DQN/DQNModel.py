import math
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
        
        # Calculate the size of the flattened output
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

class TransformerDQN(nn.Module):
    def __init__(self, input_shape, action_size, d_model=128, nhead=8, num_layers=8):
        super(TransformerDQN, self).__init__()
        self.action_size = action_size
        self.input_shape = input_shape
        self.d_model = d_model

        self.cnn = GameBoyOptimizedCNN(input_shape, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(d_model, action_size)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        q_values = self.fc_out(x)
        return q_values

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x