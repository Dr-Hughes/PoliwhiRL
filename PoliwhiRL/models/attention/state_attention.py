# -*- coding: utf-8 -*-
"""Attention mechanisms for focusing on important state features"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialAttention(nn.Module):
    """Spatial attention for visual game states"""

    def __init__(self, in_channels, reduction_ratio=8):
        """
        Initialize spatial attention module

        Args:
            in_channels: Number of input channels
            reduction_ratio: Channel reduction ratio for efficiency
        """
        super().__init__()
        self.in_channels = in_channels

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid()
        )

    def forward(self, x):
        """
        Apply spatial attention to input

        Args:
            x: Input tensor [batch_size, channels, height, width]

        Returns:
            (attended_features, attention_maps)
        """
        # Channel attention
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att

        # Spatial attention
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_input)

        # Apply attention
        attended = x_channel * spatial_att

        attention_maps = {
            "channel_attention": channel_att,
            "spatial_attention": spatial_att,
        }

        return attended, attention_maps


class FeatureAttention(nn.Module):
    """Attention mechanism for arbitrary feature vectors"""

    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        """
        Initialize feature attention

        Args:
            feature_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert (
            feature_dim % num_heads == 0
        ), "feature_dim must be divisible by num_heads"

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x, mask=None):
        """
        Apply feature attention

        Args:
            x: Input features [batch_size, seq_len, feature_dim]
            mask: Optional attention mask

        Returns:
            (attended_features, attention_weights)
        """
        batch_size, seq_len, _ = x.shape

        # Linear transformations
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        attended = torch.matmul(attention_weights, V)

        # Concatenate heads
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.feature_dim)
        )

        # Residual connection and layer norm
        output = self.layer_norm(x + attended)

        return output, attention_weights.mean(dim=1)  # Average over heads


class StateAttentionModule(nn.Module):
    """Complete state attention module for game states"""

    def __init__(self, config):
        """
        Initialize state attention module

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.use_spatial = config.get("use_spatial_attention", True)
        self.use_feature = config.get("use_feature_attention", True)

        # Spatial attention for visual inputs
        if self.use_spatial:
            visual_channels = config.get("visual_channels", 3)
            self.spatial_attention = SpatialAttention(visual_channels)

        # Feature attention for processed features
        if self.use_feature:
            feature_dim = config.get("attention_feature_dim", 256)
            num_heads = config.get("attention_num_heads", 4)
            self.feature_attention = FeatureAttention(feature_dim, num_heads)

        # Importance scoring for different state components
        self.importance_scorer = nn.Sequential(
            nn.Linear(config.get("state_components", 4), 64),
            nn.ReLU(),
            nn.Linear(64, config.get("state_components", 4)),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        visual_state=None,
        feature_state=None,
        exploration_memory=None,
        coordinates=None,
    ):
        """
        Apply attention to different state components

        Args:
            visual_state: Visual game state [batch, channels, height, width]
            feature_state: Processed feature state [batch, seq_len, features]
            exploration_memory: Exploration memory tensor
            coordinates: Current coordinates (x, y, map)

        Returns:
            Dictionary with attended features and attention maps
        """
        results = {}
        attention_maps = {}

        # Spatial attention on visual state
        if visual_state is not None and self.use_spatial:
            attended_visual, spatial_maps = self.spatial_attention(visual_state)
            results["visual"] = attended_visual
            attention_maps["spatial"] = spatial_maps

        # Feature attention on processed features
        if feature_state is not None and self.use_feature:
            attended_features, feature_weights = self.feature_attention(feature_state)
            results["features"] = attended_features
            attention_maps["features"] = feature_weights

        # Compute importance scores for state components
        component_features = []

        if visual_state is not None:
            # Visual importance: variance in spatial attention
            if "spatial" in attention_maps:
                visual_importance = attention_maps["spatial"]["spatial_attention"].var()
            else:
                visual_importance = torch.tensor(0.5)
            component_features.append(visual_importance)
        else:
            component_features.append(torch.tensor(0.0))

        if exploration_memory is not None:
            # Memory importance: how much new information vs old
            memory_novelty = torch.mean(exploration_memory[:, 0])  # Visit counts
            memory_importance = torch.sigmoid(
                -memory_novelty + 5
            )  # Lower visits = higher importance
            component_features.append(memory_importance)
        else:
            component_features.append(torch.tensor(0.0))

        if coordinates is not None:
            # Position importance: distance from goal or waypoints
            # Simplified: random importance for now
            position_importance = torch.tensor(0.3)
            component_features.append(position_importance)
        else:
            component_features.append(torch.tensor(0.0))

        # Action history importance
        action_importance = torch.tensor(0.2)
        component_features.append(action_importance)

        # Compute component importance scores
        component_tensor = torch.stack(component_features).unsqueeze(0)
        importance_scores = self.importance_scorer(component_tensor)

        results["importance_scores"] = importance_scores
        results["attention_maps"] = attention_maps

        return results

    def get_attention_summary(self, attention_results):
        """Get a summary of what the attention mechanism is focusing on"""
        summary = {}

        if "importance_scores" in attention_results:
            scores = attention_results["importance_scores"][0]
            component_names = ["visual", "memory", "position", "actions"]

            for i, (name, score) in enumerate(zip(component_names, scores)):
                summary[f"{name}_importance"] = score.item()

        if "attention_maps" in attention_results:
            maps = attention_results["attention_maps"]

            if "spatial" in maps:
                spatial_att = maps["spatial"]["spatial_attention"]
                # Find the most attended region
                flat_attention = spatial_att.flatten()
                max_idx = torch.argmax(flat_attention)
                h, w = spatial_att.shape[-2:]
                max_y, max_x = divmod(max_idx.item(), w)
                summary["most_attended_pixel"] = (max_x, max_y)
                summary["attention_concentration"] = torch.max(flat_attention).item()

        return summary


class AttentionGuidedPolicy(nn.Module):
    """Policy network that uses attention to focus on important features"""

    def __init__(self, base_policy, attention_module):
        """
        Initialize attention-guided policy

        Args:
            base_policy: Base policy network
            attention_module: State attention module
        """
        super().__init__()
        self.base_policy = base_policy
        self.attention_module = attention_module

        # Attention integration layer
        self.attention_integration = nn.Sequential(
            nn.Linear(4, 64),  # 4 importance scores
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )

    def forward(self, state, exploration_memory=None, coordinates=None):
        """
        Forward pass with attention

        Args:
            state: Game state
            exploration_memory: Exploration memory tensor
            coordinates: Current coordinates

        Returns:
            (policy_output, attention_info)
        """
        # Apply attention to state
        if len(state.shape) == 4:  # Visual state
            attention_results = self.attention_module(
                visual_state=state,
                exploration_memory=exploration_memory,
                coordinates=coordinates,
            )
            attended_state = attention_results.get("visual", state)
        else:  # Feature state
            attention_results = self.attention_module(
                feature_state=state.unsqueeze(1),  # Add seq dimension
                exploration_memory=exploration_memory,
                coordinates=coordinates,
            )
            attended_state = attention_results.get(
                "features", state.unsqueeze(1)
            ).squeeze(1)

        # Get base policy output
        base_output = self.base_policy(attended_state)

        # Integrate attention information
        if "importance_scores" in attention_results:
            importance_features = self.attention_integration(
                attention_results["importance_scores"]
            )

            # Modulate policy output based on attention
            if hasattr(base_output, "shape") and len(base_output.shape) > 1:
                modulation = importance_features.expand_as(base_output)
                modulated_output = base_output + 0.1 * modulation  # Small modulation
            else:
                modulated_output = base_output
        else:
            modulated_output = base_output

        attention_info = {
            "attention_results": attention_results,
            "attention_summary": self.attention_module.get_attention_summary(
                attention_results
            ),
        }

        return modulated_output, attention_info
