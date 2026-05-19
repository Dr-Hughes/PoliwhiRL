# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from PoliwhiRL.models.CNN.GameBoy import GameBoyBlock
from PoliwhiRL.models.transformers.positional_encoding import PositionalEncoding


def _orthogonal_init(module, gain):
    """Orthogonal weight init with zero bias — standard PPO practice for
    keeping the policy near-uniform and the value head well-conditioned
    at initialisation."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class GameBoyCNN(nn.Module):
    """ResNet-style CNN for GameBoy screen images."""

    def __init__(self, input_shape, output_dim):
        super().__init__()
        # input_shape = (C, H, W)
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
        return torch.relu(self.fc(x))


class TransformerXLBlock(nn.Module):
    """Transformer-XL block: caches a fixed-size, detached window of prior
    inputs and concatenates it onto the current chunk for attention context."""

    def __init__(self, d_model, n_heads, mem_len, dropout=0.1):
        super().__init__()
        self.mem_len = mem_len
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mem):
        extended = x if mem is None else torch.cat([mem, x], dim=1)

        attn_out, _ = self.attn(extended, extended, extended)
        out = attn_out[:, -x.size(1) :, :]

        out = self.norm1(x + out)
        ff_out = self.ffn(out)
        out = self.norm2(out + ff_out)

        # Cap memory at mem_len so it doesn't grow unbounded across calls.
        new_mem = extended[:, -self.mem_len :, :].detach()
        return out, new_mem


class PPOTransformer(nn.Module):
    """
    Screen Images -> CNN -> Latent Embedding -> TransformerXL -> Actor/Critic heads.

    Memory is passed in as an argument (per-layer list of (B, mem_len, d_model)
    tensors) rather than stored on the module. Callers manage lifecycle: reset
    at episode start, carry across rollout steps, snapshot per transition for
    replay at update time.
    """

    def __init__(
        self,
        input_shape,
        action_size,
        d_model=128,
        n_heads=8,
        num_layers=4,
        dropout=0.1,
        mem_len=16,
        **kwargs
    ):
        super().__init__()
        self.action_size = action_size
        self.input_shape = input_shape
        self.d_model = d_model
        self.num_layers = num_layers
        self.mem_len = mem_len

        self.cnn = GameBoyCNN(input_shape, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerXLBlock(d_model, n_heads, mem_len, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc_actor = nn.Linear(d_model, action_size)
        self.fc_critic = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        # Default orthogonal init for trunk weights (hidden gain = sqrt(2)),
        # then override the heads with their canonical gains.
        hidden_gain = math.sqrt(2)
        for module in self.modules():
            if module is self.fc_actor or module is self.fc_critic:
                continue
            _orthogonal_init(module, hidden_gain)
        # Actor head: small gain (~0.01) → near-uniform initial action probs,
        # so the policy explores at the start rather than committing.
        _orthogonal_init(self.fc_actor, gain=0.01)
        # Critic head: unit gain — value outputs should start near zero.
        _orthogonal_init(self.fc_critic, gain=1.0)

    def init_mems(self, batch_size, device):
        return [
            torch.zeros(batch_size, self.mem_len, self.d_model, device=device)
            for _ in range(self.num_layers)
        ]

    def forward(self, x, mems=None):
        batch_size, seq_len = x.size()[:2]

        if mems is None:
            mems = self.init_mems(batch_size, x.device)

        x = x.reshape(batch_size * seq_len, *self.input_shape)
        x = self.cnn(x)
        x = x.reshape(batch_size, seq_len, self.d_model)
        x = self.pos_encoder(x)

        new_mems = []
        for block, mem in zip(self.transformer_blocks, mems):
            x, nm = block(x, mem)
            new_mems.append(nm)

        x = x[:, -1, :]

        action_probs = torch.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)

        return action_probs, value, new_mems
