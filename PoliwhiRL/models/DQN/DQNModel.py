import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn

class TransformerDQN(nn.Module):
    def __init__(self, input_shape, action_size, n_heads=4, n_layers=2, d_model=128, debug_dir="debug"):
        super(TransformerDQN, self).__init__()
        self.debug_counter = 0
        self.debug_dir = debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)
        self.action_size = action_size
        
        # Convolutional layers (same as before)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of flattened features after conv layers
        conv_out_size = self._get_conv_out_size(input_shape)
        
        # Linear layer to project to d_model dimensions
        self.fc_pre = nn.Linear(conv_out_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, action_size)

    def _get_conv_out_size(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x, debug=False):
        batch_size, seq_len = x.size(0), x.size(1)
        
        if seq_len == 1 and debug:
            subplot = self._save_debug_image(x)

        # Process with conv layers
        x = x.view(batch_size * seq_len, *x.size()[2:])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, seq_len, -1)
        
        # Project to d_model dimensions
        x = self.fc_pre(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer expects shape (seq_len, batch, features)
        x = x.permute(1, 0, 2)
        
        # Pass through Transformer
        x = self.transformer_encoder(x)
        
        # Take the last sequence element for Q-values
        x = x[-1]
        
        # Output layer
        q_values = self.fc_out(x)

        if seq_len == 1 and debug:
            self._save_action_probabilities(q_values, subplot)
        
        return q_values
    
    def _save_debug_image(self, x):
        i = -1  # Use the last timestep's image
        img = x[i, 0].cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min())

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title("Game Screen")

        plt.tight_layout()
        self.debug_counter += 1

        if self.debug_counter % 100 == 0:
            print(f"Saved debug image {self.debug_counter}")
        
        return plt.subplot(1, 2, 2)

    def _save_action_probabilities(self, q_values, subplot=None):
        probs = F.softmax(q_values, dim=-1).cpu().detach().numpy()
        actions = ["", "a", "b", "left", "right", "up", "down", "start", "select"]
        action_names = [f"{actions[i]}" for i in range(self.action_size)]

        if subplot is None:
            plt.figure(figsize=(10, 5))

        plt.bar(action_names, probs[0])  # Use the last timestep's probabilities
        plt.title("Action Probabilities")
        plt.xlabel("Actions")
        plt.ylabel("Probability")
        #plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(f"{self.debug_dir}/action_probs_{self.debug_counter}.png")
        plt.close()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

