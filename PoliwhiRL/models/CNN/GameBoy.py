# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F


def _make_norm(num_channels, max_groups=8):
    """GroupNorm with a sensible group count. BatchNorm is a known PPO
    pitfall: rollout-time batches (size N) differ from update-time batches
    (size W*N), and the running stats accumulate across mismatched dists,
    silently biasing the policy. GroupNorm is batch-size-independent."""
    num_groups = min(max_groups, num_channels)
    # Walk down until num_channels is divisible by num_groups.
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class TileGridResidualBlock(nn.Module):
    """Constant-resolution residual block: one 3x3 conv + GroupNorm wrapped in an
    identity skip. Every body conv is stride 1 so the feature grid stays locked
    to the tile grid the stem established; the single downsample lives in
    TileGridCNN.down, where 2x2 tiles become the 16px metatile a player step
    moves."""

    def __init__(self, channels):
        super(TileGridResidualBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm = _make_norm(channels)

    def forward(self, x):
        h = F.relu(self.norm(self.conv(x)))
        return F.relu(x + h)


class TileGridCNN(nn.Module):
    """Tile-grid-aligned CNN -- the only image encoder this project builds.

    The stride-2 3x3 conv this stem replaces ran on a half-scaled frame, so its
    feature grid was aligned to neither the 8x8 Game Boy tile grid nor the 16x16
    metatile grid, and every later layer mixed partial tiles. Here the stem
    is a LINEAR PATCHIFY with kernel == stride == 4 scaled px == one 8x8 tile at
    scaling_factor 0.5: one output vector per tile, no overlap, and deliberately
    no norm and no activation (a norm would mix statistics across tiles, an
    activation would throw away the signed patch content before the body sees
    it).

    Alignment is measured, not assumed: the patch grid sits on the tile grid at
    offset (0,0) on 28/28 settled frames across 10/10 maps (SCX/SCY are 0 mod
    16), and the camera does NOT clamp on small maps -- it drives SCX/SCY
    negative and holds the player at a fixed screen cell -- so the alignment
    holds on interiors too.

    Receptive field: the patchify stem is exactly 1 tile, and the three stride-1
    3x3 body blocks add 2 scaled px of radius each, so after the body it is
    4 + 3*8 = 28 scaled px = exactly 7 tiles.

    FLATTEN, do not pool. The screen is centred on the player and the camera
    never clamps, so a cell's grid position means a fixed direction-and-distance
    from the player; global pooling would discard exactly the information the
    alignment buys.
    """

    # One 8x8 Game Boy tile at scaling_factor 0.5 (160->80 and 144->72 are both
    # exact integer halvings), i.e. 4x4 scaled px.
    TILE_PX = 4

    def __init__(self, input_shape, output_dim):
        super(TileGridCNN, self).__init__()
        channels = int(input_shape[0])
        height = int(input_shape[1])
        width = int(input_shape[2])
        if height % self.TILE_PX != 0 or width % self.TILE_PX != 0:
            raise ValueError(
                "TileGridCNN cannot patchify input_shape "
                f"{(channels, height, width)}: the tile-grid stem needs each spatial side to "
                f"be a multiple of {self.TILE_PX} (one Game Boy tile at scaling_factor 0.5) "
                "and does not silently pad."
            )
        # Linear patchify: one vector per Game Boy tile, no norm, no activation.
        self.stem = nn.Conv2d(
            channels, 32, kernel_size=self.TILE_PX, stride=self.TILE_PX, padding=0
        )
        self.blocks = nn.ModuleList([TileGridResidualBlock(32) for _ in range(3)])
        # After this one feature cell == one player step (a 16px metatile).
        self.down = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.down_norm = _make_norm(64)
        self.squeeze = nn.Conv2d(64, 16, kernel_size=1)
        self.squeeze_norm = _make_norm(16)
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            sample_output = self._features(sample_input)
            self.flat_features = sample_output.view(1, -1).size(1)
        self.fc = nn.Linear(self.flat_features, output_dim)

    def _features(self, x):
        """stem -> body -> down -> squeeze, i.e. everything but the head. Split
        out of forward so the two alignment properties this design exists for --
        tile isolation of the stem and translation equivariance of stem+body --
        can be asserted on the grid itself."""
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.down_norm(self.down(x)))
        return F.relu(self.squeeze_norm(self.squeeze(x)))

    def forward(self, x):
        # The screen arrives as uint8 0-255 and conv WEIGHT gradients scale with
        # the input magnitude, which at 0-255 dominated the shared grad-norm clip
        # and starved the trunk and heads (2026-10-23 review F10). Normalise once,
        # here, for every caller.
        x = x / 255.0
        x = self._features(x)
        x = x.view(-1, self.flat_features)
        return F.relu(self.fc(x))
