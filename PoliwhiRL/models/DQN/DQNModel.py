import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os

class DeepQNetworkModel(nn.Module):
    def __init__(
        self,
        input_shape,
        action_size,
        lstm_size=64,
        fc_size=128,
        debug_dir="debug_images",
    ):
        super(DeepQNetworkModel, self).__init__()
        self.input_shape = input_shape
        self.action_size = action_size
        self.lstm_size = lstm_size
        self.debug_dir = debug_dir
        self.debug_counter = 0

        # Create debug directory if it doesn't exist
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of flattened features after conv layers
        conv_out_size = self._get_conv_out_size(input_shape)

        # Fully connected layer before LSTM
        self.fc_pre = nn.Linear(conv_out_size, fc_size)

        # LSTM layer
        self.lstm = nn.LSTM(fc_size, lstm_size, batch_first=True)

        # Fully connected layers after LSTM
        self.fc_post = nn.Sequential(
            nn.Linear(lstm_size, fc_size), nn.ReLU(), nn.Linear(fc_size, action_size)
        )

    def _get_conv_out_size(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x, hidden_state, debug=True):
        batch_size = x.size(0)
        seq_len = x.size(1)

        if debug and batch_size == 1:
            subplot = self._save_debug_image(x)

        # Reshape and process with conv layers
        x = x.view(batch_size * seq_len, *self.input_shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size * seq_len, -1)

        x = F.relu(self.fc_pre(x))
        x = x.view(batch_size, seq_len, -1)

        lstm_out, hidden_state = self.lstm(x, hidden_state)

        q_values = self.fc_post(lstm_out.contiguous().view(batch_size * seq_len, -1))
        q_values = q_values.view(batch_size, seq_len, self.action_size)

        if debug and batch_size == 1:
            self._save_action_probabilities(q_values, subplot=subplot)

        return q_values, hidden_state

    def init_hidden(self, batch_size):
        return (
            torch.zeros(1, batch_size, self.lstm_size),
            torch.zeros(1, batch_size, self.lstm_size),
        )

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

        plt.bar(action_names, probs[0, -1])  # Use the last timestep's probabilities
        plt.title("Action Probabilities")
        plt.xlabel("Actions")
        plt.ylabel("Probability")
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(f"{self.debug_dir}/action_probs_{self.debug_counter}.png")
        plt.close()