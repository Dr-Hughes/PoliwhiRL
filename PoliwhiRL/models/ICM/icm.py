import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Linear(64 * ((input_shape[1] - 1) // 4) * ((input_shape[2] - 1) // 4), self.feature_size)
        )
        self.inverse_model = nn.Linear(self.feature_size * 2, action_size)
        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_size + action_size, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size)
        )

    def forward(self, state, next_state, action):
        encoded_state = self.encoder(state)
        encoded_next_state = self.encoder(next_state)
        
        # Inverse Model
        inverse_input = torch.cat([encoded_state, encoded_next_state], dim=1)
        predicted_action = self.inverse_model(inverse_input)
        
        # Forward Model
        forward_input = torch.cat([encoded_state, F.one_hot(action, num_classes=self.inverse_model.out_features).float()], dim=1)
        predicted_next_state_feature = self.forward_model(forward_input)
        
        return predicted_action, predicted_next_state_feature, encoded_next_state
