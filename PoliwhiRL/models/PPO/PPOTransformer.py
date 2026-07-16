# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from PoliwhiRL.models.CNN.GameBoy import GameBoyBlock


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
    inputs and concatenates it onto the current chunk for attention context.

    The trainer feeds ONE new frame per call (sequence_length=1) with the
    memory carried across steps, so the cache holds the last ``mem_len``
    genuine env steps — this is what gives the model a within-episode
    memory horizon of ``mem_len`` steps. (Feeding sliding windows longer
    than 1 also works, but fills the cache with overlapping duplicates of
    the same frames and shrinks the effective horizon to ~mem_len/seq_len
    steps — the configuration bug this design replaces.)

    Attention over the cache is content-based and order-free on its own;
    ``age_emb`` (zero-init, learned) is added by slot age before attention
    so the model CAN represent recency — "just came from there" vs "was
    there a while ago" — which pure content matching cannot.
    """

    # Headroom over mem_len for the current chunk's slots in age_emb.
    _AGE_HEADROOM = 16

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
        # Learned per-age additive embedding, indexed by slot age (last
        # entry = the current frame, first = the oldest memory slot).
        # Zero-init: a no-op at initialisation, so recency information is
        # opt-in for the optimiser rather than injected noise. Applied to
        # the attention INPUT only — the cache stores un-aged activations
        # and each forward re-ages them by their current age.
        self.age_emb = nn.Parameter(
            torch.zeros(mem_len + self._AGE_HEADROOM, d_model)
        )

    def forward(self, x, mem):
        extended = x if mem is None else torch.cat([mem, x], dim=1)

        length = extended.size(1)
        if length > self.age_emb.size(0):
            raise ValueError(
                f"sequence too long for age embedding: {x.size(1)} new + "
                f"{self.mem_len} mem slots > mem_len + {self._AGE_HEADROOM}"
            )
        attn_in = extended + self.age_emb[-length:]

        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        out = attn_out[:, -x.size(1) :, :]

        out = self.norm1(x + out)
        ff_out = self.ffn(out)
        out = self.norm2(out + ff_out)

        # Cap memory at mem_len so it doesn't grow unbounded across calls.
        new_mem = extended[:, -self.mem_len :, :].detach()
        return out, new_mem


class RAMEncoder(nn.Module):
    """MLP over the normalised RAM vector. Produces a per-step embedding the
    same way the CNN produces a per-step image embedding. Both then get
    concatenated and projected to d_model before the transformer."""

    def __init__(self, ram_dim, output_dim, hidden_dim=128):
        super().__init__()
        # Two hidden layers for better representation of the high-dimensional
        # RAM vector (138+ features including derived flags). The wider first
        # layer captures cross-feature interactions; the second projects to
        # output_dim for fusion with the CNN branch. Default hidden_dim=128
        # matches d_model so the RAM branch has equal capacity to image.
        self.net = nn.Sequential(
            nn.Linear(ram_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PPOTransformer(nn.Module):
    """
    (Screen Image, RAM vector) -> dual encoders -> fused d_model -> TransformerXL -> Actor/Critic heads.

    Memory is passed in as an argument (per-layer list of (B, mem_len, d_model)
    tensors) rather than stored on the module. Callers manage lifecycle: reset
    at episode start, carry across rollout steps, snapshot per transition for
    replay at update time.

    Temporal structure: the trainer feeds one frame per step, so the
    per-layer memory holds the previous ``mem_len`` genuine env steps and
    the model's within-episode context is ``mem_len`` steps deep. There is
    no absolute positional encoding — with single-frame chunks it would be
    a constant — recency comes from the blocks' learned age embeddings,
    and coarser history (recent maps, steps-since-novel-cell, run-wide
    cell visit counts) arrives explicitly through the RAM feature vector.
    Fresh-episode memories are zero tensors: blank slots the model learns
    to ignore, traded off deliberately against variable-length memory
    (fixed shapes keep the rollout buffer's mems snapshot/replay simple).
    """

    def __init__(
        self,
        input_shape,
        action_size,
        ram_dim,
        d_model=128,
        d_ram=128,
        n_heads=4,
        num_layers=2,
        dropout=0.1,
        mem_len=64,
        **kwargs
    ):
        super().__init__()
        self.action_size = action_size
        self.input_shape = input_shape
        self.ram_dim = int(ram_dim)
        self.d_model = d_model
        self.d_ram = d_ram
        self.num_layers = num_layers
        self.mem_len = mem_len

        # Image branch produces (B*T, d_model). RAM branch produces (B*T, d_ram).
        # Concat -> Linear(d_model + d_ram, d_model) fuses them into the
        # trunk's working width without changing the transformer geometry.
        self.cnn = GameBoyCNN(input_shape, d_model)
        self.ram_encoder = RAMEncoder(self.ram_dim, d_ram)
        self.fuse = nn.Linear(d_model + d_ram, d_model)

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

    def forward(self, x_image, x_ram, mems=None, action_mask=None):
        """Args:
        x_image:     (B, seq_len, C, H, W) float — screen sequences.
        x_ram:       (B, seq_len, ram_dim) float — RAM vector sequences.
        mems:        per-layer list of (B, mem_len, d_model) or None.
        action_mask: (B, action_size) float, optional. ``1`` = allowed,
                     ``0`` = blocked. Applied to actor logits *before*
                     softmax via a large negative additive shift, so the
                     resulting categorical distribution places zero mass
                     on blocked actions and entropy / log-prob calculations
                     stay self-consistent across rollout and update phases.
                     Callers derive it from the current-frame RAM via
                     ``environment.action_mask.compute_action_mask``.
        """
        batch_size, seq_len = x_image.size()[:2]

        if mems is None:
            mems = self.init_mems(batch_size, x_image.device)

        img = x_image.reshape(batch_size * seq_len, *self.input_shape)
        img = self.cnn(img)  # (B*T, d_model)
        ram = x_ram.reshape(batch_size * seq_len, self.ram_dim)
        ram = self.ram_encoder(ram)  # (B*T, d_ram)

        fused = self.fuse(torch.cat([img, ram], dim=-1))  # (B*T, d_model)
        x = fused.reshape(batch_size, seq_len, self.d_model)

        new_mems = []
        for block, mem in zip(self.transformer_blocks, mems):
            x, nm = block(x, mem)
            new_mems.append(nm)

        x = x[:, -1, :]

        logits = self.fc_actor(x)
        if action_mask is not None:
            # Additive penalty on blocked actions. Using -1e9 rather than
            # -inf keeps the gradient finite on the rare edge case where
            # every action is masked (defensive — shouldn't happen, but a
            # NaN backprop here would be catastrophic).
            logits = logits + (action_mask - 1.0) * 1e9
        action_probs = torch.softmax(logits, dim=-1)
        value = self.fc_critic(x)

        return action_probs, value, new_mems
