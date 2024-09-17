# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ICMModule:
    def __init__(self, input_shape, action_size, config):
        self.device = torch.device(config["device"])
        self.icm = ICM(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.icm.parameters(), lr=config["learning_rate"])
        self.curiosity_weight = config["curiosity_weight"]
        self.icm_loss_scale = config["icm_loss_scale"]

    def compute_intrinsic_reward(self, state, next_state, action):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)

        with torch.no_grad():
            _, pred_next_state_feature, encoded_next_state = self.icm(
                state_tensor, next_state_tensor, action_tensor
            )
            intrinsic_reward = (
                self.curiosity_weight
                * torch.mean(
                    torch.square(pred_next_state_feature - encoded_next_state)
                ).item()
            )

        return intrinsic_reward

    def update(self, states, next_states, actions):

        pred_actions, pred_next_state_features, encoded_next_states = self.icm(
            states, next_states, actions
        )

        inverse_loss = F.cross_entropy(pred_actions, actions)
        forward_loss = F.mse_loss(pred_next_state_features, encoded_next_states)

        icm_loss = (inverse_loss + forward_loss) * self.icm_loss_scale

        self.optimizer.zero_grad()
        icm_loss.backward()
        self.optimizer.step()

        return icm_loss.item()

    def save(self, path):
        torch.save(
            {
                "icm_state_dict": self.icm.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.icm.load_state_dict(checkpoint["icm_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class ICM(nn.Module):
    def __init__(self, input_shape, action_size):
        super(ICM, self).__init__()
        self.feature_size = 256
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                64 * ((input_shape[1] - 1) // 4) * ((input_shape[2] - 1) // 4),
                self.feature_size,
            ),
        )
        self.inverse_model = nn.Linear(self.feature_size * 2, action_size)
        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_size + action_size, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size),
        )

    def forward(self, state, next_state, action):
        encoded_state = self.encoder(state)
        encoded_next_state = self.encoder(next_state)

        # Inverse Model
        inverse_input = torch.cat([encoded_state, encoded_next_state], dim=1)
        predicted_action = self.inverse_model(inverse_input)

        # Forward Model
        forward_input = torch.cat(
            [
                encoded_state,
                F.one_hot(action, num_classes=self.inverse_model.out_features).float(),
            ],
            dim=1,
        )
        predicted_next_state_feature = self.forward_model(forward_input)

        return predicted_action, predicted_next_state_feature, encoded_next_state
