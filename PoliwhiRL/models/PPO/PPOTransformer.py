# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from PoliwhiRL.models.CNN.GameBoy import GameBoyBlock
from PoliwhiRL.models.transformers.positional_encoding import PositionalEncoding


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
    """Transformer XL block with recurrent memory."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mem=None):
        # Concatenate memory with current sequence
        if mem is not None:
            extended = torch.cat([mem, x], dim=1)
        else:
            extended = x

        # Self-attention
        attn_out, _ = self.attn(extended, extended, extended)

        # Only keep the output corresponding to the current sequence positions
        out = attn_out[:, -x.size(1):, :]

        # Residual + norm
        out = self.norm1(x + out)

        # Feed-forward
        ff_out = self.ffn(out)
        out = self.norm2(out + ff_out)

        # New memory: detach and keep the last mem_len tokens of the extended input
        new_mem = extended[:, -extended.size(1):].detach()  # keep full extended as mem
        return out, new_mem


class PPOTransformer(nn.Module):
    """
    Screen Images -> CNN -> Latent Embedding -> TransformerXL -> Actor/Critic heads
    """

    def __init__(self, input_shape, action_size, d_model=128, n_heads=8, num_layers=4, dropout=0.1, **kwargs):
        super().__init__()
        self.action_size = action_size
        self.input_shape = input_shape
        self.d_model = d_model
        self.mem_len = kwargs.get("mem_len", 16)  # context length for XL memory

        # CNN encoder
        self.cnn = GameBoyCNN(input_shape, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)

        # Transformer XL blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerXLBlock(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])

        # Persistent memory (not a parameter, just a buffer we manage)
        self.register_buffer("memory", torch.zeros(1, self.mem_len, d_model))

        # Actor & Critic heads
        self.fc_actor = nn.Linear(d_model, action_size)
        self.fc_critic = nn.Linear(d_model, 1)

    def forward(self, x):
        batch_size, seq_len = x.size()[:2]
        # Flatten batch x seq into single dimension for CNN
        x = x.view(batch_size * seq_len, *self.input_shape)

        # CNN -> latent embeddings
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, self.d_model)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer XL blocks with memory
        mem = self.memory.expand(batch_size, -1, -1)
        for block in self.transformer_blocks:
            x, mem = block(x, mem)

        # Update persistent memory
        self.memory.data = mem.data[:1]

        # Use last token output
        x = x[:, -1, :]

        action_probs = torch.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)

        return action_probs, value
