# -*- coding: utf-8 -*-
"""Auxiliary prediction tasks for better representation learning"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NextFramePredictor(nn.Module):
    """Predicts the next visual frame given current state and action"""

    def __init__(self, input_channels=3, action_space_size=9, hidden_dim=128):
        """
        Initialize next frame predictor

        Args:
            input_channels: Number of input image channels
            action_space_size: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.input_channels = input_channels
        self.action_space_size = action_space_size

        # Action embedding
        self.action_embedding = nn.Embedding(action_space_size, hidden_dim)

        # Frame encoder
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # State-action fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 * 16 + hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Frame decoder
        self.frame_decoder = nn.Sequential(nn.Linear(256, 128 * 16), nn.ReLU())

        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, current_frame, action):
        """
        Predict next frame

        Args:
            current_frame: Current visual frame [batch, channels, height, width]
            action: Action taken [batch]

        Returns:
            Predicted next frame
        """
        # Encode current frame
        frame_features = self.frame_encoder(current_frame)
        frame_features = frame_features.view(frame_features.size(0), -1)

        # Embed action
        action_features = self.action_embedding(action)

        # Fuse state and action
        combined = torch.cat([frame_features, action_features], dim=1)
        fused_features = self.fusion(combined)

        # Decode to next frame
        decoded = self.frame_decoder(fused_features)
        decoded = decoded.view(-1, 128, 4, 4)

        next_frame = self.upsample(decoded)

        return next_frame


class RewardPredictor(nn.Module):
    """Predicts immediate reward given state and action"""

    def __init__(self, state_dim, action_space_size=9, hidden_dim=128):
        """
        Initialize reward predictor

        Args:
            state_dim: Dimension of state representation
            action_space_size: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.action_embedding = nn.Embedding(action_space_size, hidden_dim)

        self.predictor = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state, action):
        """
        Predict reward

        Args:
            state: State representation [batch, state_dim]
            action: Action taken [batch]

        Returns:
            Predicted reward [batch, 1]
        """
        action_features = self.action_embedding(action)
        combined = torch.cat([state, action_features], dim=1)
        reward = self.predictor(combined)
        return reward


class DonePredictor(nn.Module):
    """Predicts episode termination given state and action"""

    def __init__(self, state_dim, action_space_size=9, hidden_dim=128):
        """
        Initialize done predictor

        Args:
            state_dim: Dimension of state representation
            action_space_size: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.action_embedding = nn.Embedding(action_space_size, hidden_dim)

        self.predictor = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, state, action):
        """
        Predict episode termination probability

        Args:
            state: State representation [batch, state_dim]
            action: Action taken [batch]

        Returns:
            Termination probability [batch, 1]
        """
        action_features = self.action_embedding(action)
        combined = torch.cat([state, action_features], dim=1)
        done_prob = self.predictor(combined)
        return done_prob


class InverseDynamicsModel(nn.Module):
    """Predicts action taken given current and next state"""

    def __init__(self, state_dim, action_space_size=9):
        """
        Initialize inverse dynamics model

        Args:
            state_dim: Dimension of state representation
            action_space_size: Number of possible actions
        """
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(state_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_size),
        )

    def forward(self, current_state, next_state):
        """
        Predict action

        Args:
            current_state: Current state [batch, state_dim]
            next_state: Next state [batch, state_dim]

        Returns:
            Action logits [batch, action_space_size]
        """
        combined = torch.cat([current_state, next_state], dim=1)
        action_logits = self.predictor(combined)
        return action_logits


class ForwardDynamicsModel(nn.Module):
    """Predicts next state given current state and action"""

    def __init__(self, state_dim, action_space_size=9, hidden_dim=128):
        """
        Initialize forward dynamics model

        Args:
            state_dim: Dimension of state representation
            action_space_size: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.action_embedding = nn.Embedding(action_space_size, hidden_dim)

        self.predictor = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim),
        )

    def forward(self, current_state, action):
        """
        Predict next state

        Args:
            current_state: Current state [batch, state_dim]
            action: Action taken [batch]

        Returns:
            Predicted next state [batch, state_dim]
        """
        action_features = self.action_embedding(action)
        combined = torch.cat([current_state, action_features], dim=1)
        next_state = self.predictor(combined)
        return next_state


class StateContrastiveModel(nn.Module):
    """Contrastive learning for state representations"""

    def __init__(self, state_dim, projection_dim=128, temperature=0.1):
        """
        Initialize contrastive model

        Args:
            state_dim: Dimension of state representation
            projection_dim: Dimension of projection head
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        self.temperature = temperature

        self.projection_head = nn.Sequential(
            nn.Linear(state_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, states):
        """
        Project states to contrastive space

        Args:
            states: State representations [batch, state_dim]

        Returns:
            Projected states [batch, projection_dim]
        """
        return F.normalize(self.projection_head(states), dim=1)

    def contrastive_loss(self, anchor, positive, negatives):
        """
        Compute contrastive loss

        Args:
            anchor: Anchor state [projection_dim]
            positive: Positive state [projection_dim]
            negatives: Negative states [num_negatives, projection_dim]

        Returns:
            Contrastive loss
        """
        # Cosine similarity
        pos_sim = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0))
        neg_sim = F.cosine_similarity(anchor.unsqueeze(0), negatives)

        # Contrastive loss
        logits = torch.cat([pos_sim, neg_sim]) / self.temperature
        labels = torch.zeros(1, dtype=torch.long, device=anchor.device)

        loss = F.cross_entropy(logits.unsqueeze(0), labels)
        return loss


class AuxiliaryTaskManager(nn.Module):
    """Manages multiple auxiliary tasks for representation learning"""

    def __init__(self, config):
        """
        Initialize auxiliary task manager

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config

        # Task configuration
        self.use_next_frame = config.get("use_next_frame_prediction", True)
        self.use_reward_pred = config.get("use_reward_prediction", True)
        self.use_done_pred = config.get("use_done_prediction", True)
        self.use_inverse_dynamics = config.get("use_inverse_dynamics", True)
        self.use_forward_dynamics = config.get("use_forward_dynamics", True)
        self.use_contrastive = config.get("use_contrastive_learning", True)

        # Task weights
        self.task_weights = {
            "next_frame": config.get("next_frame_weight", 1.0),
            "reward": config.get("reward_prediction_weight", 1.0),
            "done": config.get("done_prediction_weight", 0.5),
            "inverse_dynamics": config.get("inverse_dynamics_weight", 0.5),
            "forward_dynamics": config.get("forward_dynamics_weight", 0.5),
            "contrastive": config.get("contrastive_weight", 0.5),
        }

        # Model dimensions
        state_dim = config.get("state_representation_dim", 256)
        action_space_size = config.get("action_space_size", 9)
        input_channels = config.get("input_channels", 3)

        # Initialize task modules
        if self.use_next_frame:
            self.next_frame_predictor = NextFramePredictor(
                input_channels, action_space_size
            )

        if self.use_reward_pred:
            self.reward_predictor = RewardPredictor(state_dim, action_space_size)

        if self.use_done_pred:
            self.done_predictor = DonePredictor(state_dim, action_space_size)

        if self.use_inverse_dynamics:
            self.inverse_dynamics = InverseDynamicsModel(state_dim, action_space_size)

        if self.use_forward_dynamics:
            self.forward_dynamics = ForwardDynamicsModel(state_dim, action_space_size)

        if self.use_contrastive:
            self.contrastive_model = StateContrastiveModel(state_dim)

    def compute_auxiliary_losses(self, batch_data):
        """
        Compute all auxiliary losses

        Args:
            batch_data: Dictionary containing:
                - current_states: Current state representations
                - next_states: Next state representations
                - current_frames: Current visual frames (if available)
                - next_frames: Next visual frames (if available)
                - actions: Actions taken
                - rewards: Rewards received
                - dones: Episode termination flags

        Returns:
            Dictionary of losses and total auxiliary loss
        """
        losses = {}
        total_loss = 0

        # Extract batch data
        current_states = batch_data.get("current_states")
        next_states = batch_data.get("next_states")
        current_frames = batch_data.get("current_frames")
        next_frames = batch_data.get("next_frames")
        actions = batch_data.get("actions")
        rewards = batch_data.get("rewards")
        dones = batch_data.get("dones")

        # Next frame prediction
        if (
            self.use_next_frame
            and current_frames is not None
            and next_frames is not None
        ):
            pred_next_frames = self.next_frame_predictor(current_frames, actions)
            frame_loss = F.mse_loss(pred_next_frames, next_frames)
            losses["next_frame"] = frame_loss
            total_loss += self.task_weights["next_frame"] * frame_loss

        # Reward prediction
        if self.use_reward_pred and current_states is not None and rewards is not None:
            pred_rewards = self.reward_predictor(current_states, actions)
            reward_loss = F.mse_loss(pred_rewards.squeeze(), rewards)
            losses["reward"] = reward_loss
            total_loss += self.task_weights["reward"] * reward_loss

        # Done prediction
        if self.use_done_pred and current_states is not None and dones is not None:
            pred_dones = self.done_predictor(current_states, actions)
            done_loss = F.binary_cross_entropy(pred_dones.squeeze(), dones.float())
            losses["done"] = done_loss
            total_loss += self.task_weights["done"] * done_loss

        # Inverse dynamics
        if (
            self.use_inverse_dynamics
            and current_states is not None
            and next_states is not None
            and actions is not None
        ):
            pred_actions = self.inverse_dynamics(current_states, next_states)
            inverse_loss = F.cross_entropy(pred_actions, actions)
            losses["inverse_dynamics"] = inverse_loss
            total_loss += self.task_weights["inverse_dynamics"] * inverse_loss

        # Forward dynamics
        if (
            self.use_forward_dynamics
            and current_states is not None
            and next_states is not None
            and actions is not None
        ):
            pred_next_states = self.forward_dynamics(current_states, actions)
            forward_loss = F.mse_loss(pred_next_states, next_states)
            losses["forward_dynamics"] = forward_loss
            total_loss += self.task_weights["forward_dynamics"] * forward_loss

        # Contrastive learning (simplified - would need proper positive/negative sampling)
        if self.use_contrastive and current_states is not None:
            # Simple contrastive loss using temporal neighbors as positives
            if len(current_states) > 1:
                projections = self.contrastive_model(current_states)
                # Use next state as positive, others as negatives
                if len(projections) >= 2:
                    anchor = projections[0]
                    positive = projections[1]
                    negatives = (
                        projections[2:] if len(projections) > 2 else projections[0:1]
                    )

                    if len(negatives) > 0:
                        contrastive_loss = self.contrastive_model.contrastive_loss(
                            anchor, positive, negatives
                        )
                        losses["contrastive"] = contrastive_loss
                        total_loss += (
                            self.task_weights["contrastive"] * contrastive_loss
                        )

        losses["total_auxiliary"] = total_loss
        return losses

    def get_predictions(self, current_state, action, current_frame=None):
        """
        Get predictions for debugging/visualization

        Args:
            current_state: Current state representation
            action: Action to take
            current_frame: Current visual frame (optional)

        Returns:
            Dictionary of predictions
        """
        predictions = {}

        with torch.no_grad():
            if self.use_reward_pred:
                pred_reward = self.reward_predictor(
                    current_state.unsqueeze(0), action.unsqueeze(0)
                )
                predictions["reward"] = pred_reward.item()

            if self.use_done_pred:
                pred_done = self.done_predictor(
                    current_state.unsqueeze(0), action.unsqueeze(0)
                )
                predictions["done_prob"] = pred_done.item()

            if self.use_next_frame and current_frame is not None:
                pred_frame = self.next_frame_predictor(
                    current_frame.unsqueeze(0), action.unsqueeze(0)
                )
                predictions["next_frame"] = pred_frame.squeeze(0)

        return predictions
