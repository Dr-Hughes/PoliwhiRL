# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from PoliwhiRL.models.CNN.GameBoy import TileGridCNN


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


class TransformerXLBlock(nn.Module):
    """Transformer-XL block: caches a fixed-size, DETACHED window of prior
    inputs and concatenates it onto the current chunk for attention context.

    Two regimes (2026-07-16 BPTT rebuild):
      * ROLLOUT / inference — ONE frame per call, memory carried across
        steps, so the cache holds the last ``mem_len`` genuine env steps.
      * TRAINING — a contiguous SEGMENT of ``seq_len`` frames per call, with
        the (detached) memory that preceded the segment as context. Gradient
        now flows across the segment's ``seq_len`` positions, so the model
        genuinely learns temporal structure (real BPTT). Detaching the
        cross-segment memory is standard Transformer-XL; the within-segment
        BPTT is what segment_length=1 (the old config) never had.

    Attention is CAUSAL: position q attends only to keys k <= q (all memory
    slots, plus segment positions up to and including its own). Without this,
    a segment position would see its own future, leaking information into the
    policy/value at time t — fine when seq_len==1 (one query, causal == full)
    but wrong for real segments.

    ``age_emb`` (zero-init, learned) is added by slot age before attention so
    the model CAN represent recency — "just came from there" vs "was there a
    while ago" — which pure content matching cannot.
    """

    # Headroom over mem_len for the current segment's slots in age_emb —
    # bounds the max trainable ``sequence_length`` (segment length).
    _AGE_HEADROOM = 64

    def __init__(self, d_model, n_heads, mem_len, dropout=0.1):
        super().__init__()
        self.mem_len = mem_len
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
        # Learned per-age additive embedding, indexed by slot age (last
        # entry = the current frame, first = the oldest memory slot).
        # Zero-init: a no-op at initialisation, so recency information is
        # opt-in for the optimiser rather than injected noise. Applied to
        # the attention INPUT only — the cache stores un-aged activations
        # and each forward re-ages them by their current age.
        self.age_emb = nn.Parameter(torch.zeros(mem_len + self._AGE_HEADROOM, d_model))

    def forward(self, x, mem, reset_after=None):
        # Replay a segment one frame at a time while retaining its graph.  The
        # age embedding is relative to the current frame, so applying it once
        # to a whole segment gave a different age to its early frames than the
        # single-frame rollout sees.  This loop is therefore intentionally the
        # same computation as rollout inference; ``history`` is detached only
        # at the segment boundary, preserving within-segment BPTT.
        history = x.new_zeros(x.size(0), 0, x.size(-1)) if mem is None else mem
        outputs = []
        for position in range(x.size(1)):
            current = x[:, position : position + 1]
            extended = torch.cat([history, current], dim=1)
            length = extended.size(1)
            if length > self.age_emb.size(0):
                raise ValueError(
                    f"sequence too long for {self.mem_len} memory slots > "
                    f"mem_len + {self._AGE_HEADROOM}"
                )
            attn_in = extended + self.age_emb[-length:]
            attn_out, _ = self.attn(attn_in, attn_in, attn_in)
            out = self.norm1(current + attn_out[:, -1:])
            out = self.norm2(out + self.ffn(out))
            outputs.append(out)
            history = extended[:, -self.mem_len :]
            if reset_after is not None:
                reset = reset_after[:, position].to(dtype=torch.bool)
                history = history.masked_fill(reset[:, None, None], 0.0)

        return torch.cat(outputs, dim=1), history.detach()


class RAMEncoder(nn.Module):
    """MLP over the normalised RAM vector. Produces a per-step embedding the
    same way the CNN produces a per-step image embedding. Both then get
    concatenated and projected to d_model before the transformer."""

    def __init__(self, ram_dim, output_dim, hidden_dim=128, input_norm=False):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_norm = bool(input_norm)
        # Two hidden layers for better representation of the high-dimensional
        # RAM vector (138+ features including derived flags). The wider first
        # layer captures cross-feature interactions; the second projects to
        # output_dim for fusion with the CNN branch. Default hidden_dim=128
        # matches d_model so the RAM branch has equal capacity to image.
        layers = []
        if self.input_norm:
            # RISK, accepted knowingly and gated behind ``ram_input_norm`` so it
            # can be A/B'd without a code change. The RAM vector is 222 wide and
            # largely SPARSE BINARY -- battle-type one-hots, derived event
            # flags, menu state -- alongside [0,1]-normalised fields, and
            # LayerNorm normalises ACROSS the feature dimension per sample. When
            # most entries are 0 the per-sample mean sits near 0 and the std is
            # set by the handful of nonzero flags, so the norm can AMPLIFY those
            # flags (and the zero/nonzero pattern around them) rather than
            # stabilise the branch.
            layers.append(nn.LayerNorm(ram_dim))
        layers.extend(
            [
                nn.Linear(ram_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
                nn.GELU(),
            ]
        )
        # With input_norm=False this is the same Sequential in the same order, so
        # the state_dict keys ram_encoder.net.0..5 are unchanged; with the norm on
        # -- PPOTransformer's default -- every index shifts by one.
        self.net = nn.Sequential(*layers)

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
        ram_hidden_dim=192,
        ram_input_norm=True,
        **kwargs,
    ):
        super().__init__()
        self.action_size = action_size
        self.input_shape = input_shape
        self.ram_dim = int(ram_dim)
        self.d_model = d_model
        self.d_ram = d_ram
        self.num_layers = num_layers
        self.mem_len = mem_len
        # Kept as attributes so triage tooling can read the RAM encoder's shape
        # off the live module instead of re-deriving it from config.
        self.ram_hidden_dim = int(ram_hidden_dim)
        self.ram_input_norm = bool(ram_input_norm)

        # Image branch produces (B*T, d_model). RAM branch produces (B*T, d_ram).
        # Concat -> Linear(d_model + d_ram, d_model) fuses them into the
        # trunk's working width without changing the transformer geometry.
        # TileGridCNN's stem patchifies one 8x8 Game Boy tile per feature cell, so
        # the grid the trunk is fed is locked to the game's own tile grid. Only
        # the RAM ENCODER's hidden width is tunable: d_ram is what feeds ``fuse``,
        # so widening ram_hidden_dim leaves the fusion layer -- and the trunk
        # geometry -- untouched.
        self.cnn = TileGridCNN(input_shape, d_model)
        self.ram_encoder = RAMEncoder(
            self.ram_dim,
            d_ram,
            hidden_dim=self.ram_hidden_dim,
            input_norm=self.ram_input_norm,
        )
        self.fuse = nn.Linear(d_model + d_ram, d_model)

        self.transformer_blocks = nn.ModuleList(
            [TransformerXLBlock(d_model, n_heads, mem_len, dropout) for _ in range(num_layers)]
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

    def forward(self, x_image, x_ram, mems=None, reset_after=None):
        """Args:
        x_image:     (B, seq_len, C, H, W) float — screen sequences.
        x_ram:       (B, seq_len, ram_dim) float — RAM vector sequences.
        mems:        per-layer list of (B, mem_len, d_model) or None.
        reset_after:  (B, seq_len) bool, optional. A true entry clears each
                      layer's recurrent history after scoring that position,
                      matching collection's memory reset on episode end.

        Returns per-position outputs: action_probs (B, seq_len, action_size)
        and value (B, seq_len, 1). Callers that only need the current frame
        (rollout, probes) index ``[:, -1]``; the PPO update uses all
        positions so gradient flows across the segment (BPTT).
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
            x, nm = block(x, mem, reset_after=reset_after)
            new_mems.append(nm)

        # Keep ALL positions (B, seq_len, d_model) so the PPO loss gets a
        # gradient at every step of the segment.
        logits = self.fc_actor(x)  # (B, seq_len, action_size)
        action_probs = torch.softmax(logits, dim=-1)  # (B, seq_len, action_size)
        value = self.fc_critic(x)  # (B, seq_len, 1)

        return action_probs, value, new_mems
