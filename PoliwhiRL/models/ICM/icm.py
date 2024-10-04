# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.serialization


class ICMModule:
    def __init__(self, input_shape, action_size, config):
        self.device = torch.device(config["device"])
        self.icm = FlexibleICM(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(
            self.icm.parameters(), lr=config["icm_learning_rate"]
        )
        self.curiosity_weight = config["icm_curiosity_weight"]
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
        torch.save(self.icm.state_dict(), f"{path}_icm.pth")
        torch.save(self.optimizer.state_dict(), f"{path}_optimizer.pth")

        # Save additional parameters
        additional_params = {
            "curiosity_weight": self.curiosity_weight,
            "icm_loss_scale": self.icm_loss_scale,
        }
        torch.save(additional_params, f"{path}_params.pth")

    def load(self, path):
        try:
            icm_state = torch.load(
                f"{path}_icm.pth", map_location=self.device, weights_only=True
            )
            self.icm.load_state_dict(icm_state)

            optimizer_state = torch.load(
                f"{path}_optimizer.pth", map_location=self.device, weights_only=True
            )
            self.optimizer.load_state_dict(optimizer_state)

            torch.serialization.add_safe_globals(["numpy", "np"])
            additional_params = torch.load(
                f"{path}_params.pth", map_location=self.device, weights_only=False
            )
            self.curiosity_weight = additional_params["curiosity_weight"]
            self.icm_loss_scale = additional_params["icm_loss_scale"]
        except FileNotFoundError:
            print(f"No ICM checkpoint found at {path}, using initial values.")
        except Exception as e:
            print(f"Error loading ICM model: {e}")
            print("Using initial values.")


class FlexibleEncoder(nn.Module):
    def __init__(self, input_shape, feature_size):
        super(FlexibleEncoder, self).__init__()
        self.input_shape = input_shape
        self.feature_size = feature_size

        if len(input_shape) == 3:  # [C, H, W]
            self.encoder = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(
                    64 * ((input_shape[1] - 1) // 4) * ((input_shape[2] - 1) // 4),
                    feature_size,
                ),
            )
        elif len(input_shape) == 2:  # [X, Y]
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_shape[0] * input_shape[1], 256),
                nn.ReLU(),
                nn.Linear(256, feature_size),
            )
        else:
            raise ValueError("Unsupported input shape")

    def forward(self, x):
        return self.encoder(x)


class FlexibleICM(nn.Module):
    def __init__(self, input_shape, action_size):
        super(FlexibleICM, self).__init__()
        self.feature_size = 256
        self.encoder = FlexibleEncoder(input_shape, self.feature_size)
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
