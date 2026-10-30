# -*- coding: utf-8 -*-
"""TileGridCNN -- the only image encoder -- and the properties it exists for.

The stride-2 3x3 conv it replaces opened on a half-scaled frame, so its feature
grid was aligned to neither the 8x8 Game Boy tile grid nor the 16x16 metatile
grid. TileGridCNN's stem is a linear patchify with kernel == stride == 4 scaled
px == exactly one tile at scaling_factor 0.5, and the patch grid is measured to
sit on the tile grid at offset (0,0) on 28/28 settled frames across 10/10 maps
(SCX/SCY are 0 mod 16, and the camera does not clamp on small maps). That buys
two properties, pinned here:

  * TILE ISOLATION -- one tile's pixels cannot reach another tile's patch vector,
    because the stem's 4x4 windows are disjoint. This holds at the tensor level;
    whether it holds for the OBSERVATION depends on the resize filter, which is
    why get_screen_image hardcodes BOX: at scaling_factor 0.5 that is an exact
    2x2 area average, so nothing crosses a tile boundary.
  * POSITION INDEPENDENCE -- the same tiles at a different grid position give
    the same features, because every body conv is stride 1 with weights shared
    across cells, and the head FLATTENS rather than pools (the screen is centred
    on the player and the camera never clamps, so a cell's grid position means a
    fixed direction-and-distance from the player).

It also pins that the DEFAULTS are this design: a config mentioning no encoder
key at all builds the 818,421-parameter tile model.
"""

import unittest

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from PoliwhiRL.environment.gym_env import PyBoyEnvironment
from PoliwhiRL.models.CNN.GameBoy import TileGridCNN, _make_norm
from PoliwhiRL.models.PPO.PPOTransformer import PPOTransformer
from PoliwhiRL.models.PPO.ppo_model_implementation import PPOModel

# scaling_factor 0.5 applied to a 160x144 RGB screen: one 8x8 Game Boy tile is
# 4x4 scaled px, so a frame is an 18x20 tile grid.
INPUT_SHAPE = (3, 72, 80)
TILE_GRID = (18, 20)
TILE_PX = 4
RAM_DIM = 222
ACTION_SIZE = 8
# Trunk geometry used by the parameter-count pins.
GEOM = dict(d_model=128, d_ram=128, n_heads=4, num_layers=2, mem_len=128, dropout=0.0)
# Body radius in CELLS: three stride-1 3x3 residual blocks, one cell each.
BODY_RADIUS = 3


def _tile_cnn(output_dim=128, input_shape=INPUT_SHAPE):
    torch.manual_seed(0)
    return TileGridCNN(input_shape, output_dim).eval()


def _transformer(ram_hidden_dim=192, ram_input_norm=True, input_shape=INPUT_SHAPE):
    torch.manual_seed(0)
    return PPOTransformer(
        input_shape,
        ACTION_SIZE,
        ram_dim=RAM_DIM,
        ram_hidden_dim=ram_hidden_dim,
        ram_input_norm=ram_input_norm,
        **GEOM,
    ).eval()


def _forward(model, batch=3, seq_len=1):
    """One full PPOTransformer forward on uint8-range screen + RAM sequences."""
    image = torch.randint(0, 256, (batch, seq_len, *INPUT_SHAPE)).float()
    ram = torch.rand(batch, seq_len, RAM_DIM)
    mems = model.init_mems(batch, torch.device("cpu"))
    with torch.no_grad():
        return model(image, ram, mems)


def _conv_output_shapes(model, batch=2):
    """(C, H, W) of every Conv2d output, in module-definition order.

    Captured with forward hooks on the real module tree rather than by naming
    private attributes: delete, reorder or re-width a layer and this sequence
    changes, so the shape assertions fail instead of silently passing."""
    captured = []
    hooks = [
        conv.register_forward_hook(lambda _m, _i, out: captured.append(tuple(out.shape[1:])))
        for conv in model.modules()
        if isinstance(conv, nn.Conv2d)
    ]
    try:
        with torch.no_grad():
            model(torch.zeros(batch, *INPUT_SHAPE))
    finally:
        for hook in hooks:
            hook.remove()
    return captured


def _body_features(model, x):
    """stem + the three residual blocks, i.e. the constant-resolution grid."""
    h = model.stem(x)
    for block in model.blocks:
        h = block(h)
    return h


def _param_groups(model):
    """Per-group parameter counts, second level deep under the branch prefixes."""
    groups = {}
    for name, param in model.named_parameters():
        parts = name.split(".")
        branches = ("cnn", "ram_encoder", "transformer_blocks")
        key = ".".join(parts[:2]) if parts[0] in branches else parts[0]
        groups[key] = groups.get(key, 0) + param.numel()
    return groups


def _breakdown(model):
    groups = _param_groups(model)
    lines = [f"      {key}: {count:,}" for key, count in groups.items()]
    lines.append(f"      TOTAL: {sum(groups.values()):,}")
    return "\n".join(lines)


class TestForwardContract(unittest.TestCase):
    """(a) The CNN produces a (B, d_model) image embedding and drives the full
    transformer forward."""

    def test_tile_embedding_and_forward(self):
        self._check(_transformer())

    def _check(self, model):
        batch = 3
        image = torch.randint(0, 256, (batch, *INPUT_SHAPE)).float()
        with torch.no_grad():
            embedding = model.cnn(image)
        self.assertEqual(tuple(embedding.shape), (batch, GEOM["d_model"]))
        # The forward contract: ReLU'd and finite from a uint8-range input.
        self.assertTrue(torch.all(embedding >= 0), "image embedding is not ReLU'd")
        self.assertTrue(torch.all(torch.isfinite(embedding)))
        probs, value, mems = _forward(model, batch)
        self.assertEqual(tuple(probs.shape), (batch, 1, ACTION_SIZE))
        self.assertEqual(tuple(value.shape), (batch, 1, 1))
        self.assertEqual(len(mems), GEOM["num_layers"])
        for mem in mems:
            self.assertEqual(tuple(mem.shape), (batch, GEOM["mem_len"], GEOM["d_model"]))
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(batch, 1), atol=1e-5))


class TestTileGridIntermediateShapes(unittest.TestCase):
    """(b) The exact grid the tile variant walks: one cell per tile after the
    stem, one cell per player step after the single downsample."""

    def test_conv_output_shapes_in_order(self):
        shapes = _conv_output_shapes(_tile_cnn())
        stem = (32, *TILE_GRID)  # one 4x4-px patch per 8x8 Game Boy tile
        body = [(32, *TILE_GRID)] * 3  # stride 1: the grid never moves
        down = [(64, 9, 10)]  # 2x2 tiles = one 16px metatile = one player step
        squeeze = [(16, 9, 10)]
        self.assertEqual(shapes, [stem] + body + down + squeeze)

    def test_head_flattens_rather_than_pools(self):
        model = _tile_cnn()
        self.assertEqual(model.flat_features, 16 * 9 * 10)
        heads = [m for m in model.modules() if isinstance(m, nn.Linear)]
        self.assertEqual(len(heads), 1, "expected exactly one Linear head")
        self.assertEqual((heads[0].in_features, heads[0].out_features), (1440, 128))
        self.assertNotIsInstance(model.squeeze, nn.AdaptiveAvgPool2d)

    def test_stem_is_a_linear_patchify(self):
        model = _tile_cnn()
        stem = model.stem
        self.assertEqual(tuple(stem.weight.shape), (32, 3, TILE_PX, TILE_PX))
        self.assertEqual(stem.kernel_size, (TILE_PX, TILE_PX))
        self.assertEqual(stem.stride, (TILE_PX, TILE_PX))
        self.assertEqual(stem.padding, (0, 0))
        # No norm and no activation between the input and the stem output.
        self.assertNotIsInstance(model.stem, nn.Sequential)


class TestParameterCounts(unittest.TestCase):
    """(c) The measured total for the one architecture, built from its own
    defaults -- if the design moves, this number moves."""

    TILE_TOTAL = 818_421
    TOLERANCE = 200

    def test_tile_variant_total(self):
        model = _transformer()
        total = sum(p.numel() for p in model.parameters())
        self.assertLessEqual(
            abs(total - self.TILE_TOTAL),
            self.TOLERANCE,
            f"tile PPOTransformer has {total:,} params, expected {self.TILE_TOTAL:,} "
            f"+/- {self.TOLERANCE}. Per-group breakdown:\n{_breakdown(model)}",
        )


class TestStemTileIsolation(unittest.TestCase):
    """(d) kernel == stride == 4 with padding 0 makes every stem cell read a
    DISJOINT 4x4 scaled-px window, i.e. exactly one 8x8 Game Boy tile. Content
    in one tile therefore cannot reach another tile's patch vector. This is the
    property the observation's hardcoded BOX resize preserves; a leaking filter
    would break it before the tensor ever reached the stem."""

    PATCH = (5, 7)

    def _inputs(self):
        black = torch.zeros(1, 3, *INPUT_SHAPE[1:])
        white = black.clone()
        row, col = self.PATCH
        rows = slice(row * TILE_PX, (row + 1) * TILE_PX)
        cols = slice(col * TILE_PX, (col + 1) * TILE_PX)
        white[:, :, rows, cols] = 255.0
        return black, white

    def _other_cells(self):
        row, col = self.PATCH
        mask = torch.ones(1, 32, *TILE_GRID, dtype=torch.bool)
        mask[:, :, row, col] = False
        return mask

    def test_one_tile_cannot_reach_another_patch(self):
        model = _tile_cnn()
        black, white = self._inputs()
        with torch.no_grad():
            delta = model.stem(white) - model.stem(black)
        mask = self._other_cells()
        # Differencing cancels the stem's bias, so this pins the conv's window
        # disjointness rather than the init.
        self.assertEqual(float(delta[mask].abs().max()), 0.0)
        row, col = self.PATCH
        self.assertGreater(float(delta[:, :, row, col].abs().max()), 0.0)

    def test_bias_free_stem_is_exactly_zero_off_tile(self):
        # PPOTransformer's orthogonal init zeroes every conv bias, so for a
        # trained model the plain statement holds: black in, exactly zero out
        # everywhere except the lit tile's cell.
        model = _tile_cnn()
        nn.init.zeros_(model.stem.bias)
        black, white = self._inputs()
        with torch.no_grad():
            out_black = model.stem(black)
            out_white = model.stem(white)
        self.assertTrue(torch.all(out_black == 0.0))
        self.assertTrue(torch.all(out_white[self._other_cells()] == 0.0))
        row, col = self.PATCH
        self.assertGreater(float(out_white[:, :, row, col].abs().max()), 0.0)

    def test_stem_is_exactly_position_independent(self):
        """The same tile content at two different grid positions gives
        BITWISE-identical stem cells. The stem has no norm and no padding, so
        nothing couples the two placements -- exactness the body cannot offer
        (its GroupNorm statistics are global, so any border difference reaches
        every cell)."""
        model = _tile_cnn()
        generator = torch.Generator().manual_seed(7)
        pattern = torch.randint(0, 256, (3, 5 * TILE_PX, 5 * TILE_PX), generator=generator)
        placed_a = torch.zeros(1, 3, *INPUT_SHAPE[1:])
        placed_b = torch.zeros(1, 3, *INPUT_SHAPE[1:])
        placed_a[0, :, 8:28, 12:32] = pattern.float()
        placed_b[0, :, 40:60, 44:64] = pattern.float()
        with torch.no_grad():
            out_a = model.stem(placed_a)
            out_b = model.stem(placed_b)
        self.assertTrue(torch.equal(out_b[:, :, 10:15, 11:16], out_a[:, :, 2:7, 3:8]))


class TestTranslationEquivariance(unittest.TestCase):
    """(e) Shift the tile grid by exactly one tile and the post-body feature map
    shifts by exactly one cell -- the position-independence the redesign buys.

    The input is shifted with ``torch.roll`` so the tile content is preserved
    rather than lost off the edge. Two borders are then unavoidable: the body's
    3-cell receptive-field radius (one cell per stride-1 residual block), and
    one more cell on the trailing edge for the shift itself. What is left over
    inside that border is NOT zero: the body's GroupNorm statistics are computed
    over the whole map, so the zero-padded edge cells -- whose content the roll
    changes -- perturb the mean/variance every cell is normalised by. The
    assertions below therefore bound that residual against the feature scale and
    against the uncompensated (wrong-offset) difference, which is what makes the
    one-cell offset a real measurement rather than a tolerance that anything
    passes."""

    def _maps(self, model, x, cells):
        with torch.no_grad():
            original = _body_features(model, x)
            rolled = torch.roll(x, shifts=(4 * cells, 4 * cells), dims=(2, 3))
            shifted = _body_features(model, rolled)
        return original, shifted

    def test_one_tile_shift_moves_the_feature_map_one_cell(self):
        model = _tile_cnn()
        generator = torch.Generator().manual_seed(1)
        x = torch.randint(0, 256, (1, 3, *INPUT_SHAPE[1:]), generator=generator).float() / 255.0
        original, shifted = self._maps(model, x, 1)
        height, width = TILE_GRID
        radius = BODY_RADIUS
        lo_i, hi_i = radius, height - 2 - radius
        lo_j, hi_j = radius, width - 2 - radius
        compensated = shifted[:, :, lo_i + 1 : hi_i + 2, lo_j + 1 : hi_j + 2]
        interior = original[:, :, lo_i : hi_i + 1, lo_j : hi_j + 1]
        self.assertEqual(tuple(compensated.shape), tuple(interior.shape))
        # The same slices with NO offset compensation: the control that proves
        # the shift really is one cell.
        uncompensated = shifted[:, :, lo_i : hi_i + 1, lo_j : hi_j + 1]
        scale = float(original.abs().max())
        interior_diff = float((compensated - interior).abs().max())
        control_diff = float((uncompensated - interior).abs().max())
        detail = (
            f"scale={scale:.4f} interior_diff={interior_diff:.4e} "
            f"control_diff={control_diff:.4e} (measured 6.74 / 1.08e-01 / 4.92)"
        )
        self.assertLess(interior_diff, 0.05 * scale, detail)
        self.assertGreater(control_diff, 0.30 * scale, detail)
        self.assertGreater(control_diff, 5 * interior_diff, detail)

    def test_two_tile_shift_moves_the_feature_map_two_cells(self):
        model = _tile_cnn()
        generator = torch.Generator().manual_seed(2)
        x = torch.randint(0, 256, (1, 3, *INPUT_SHAPE[1:]), generator=generator).float() / 255.0
        original, shifted = self._maps(model, x, 2)
        height, width = TILE_GRID
        radius = BODY_RADIUS
        lo_i, hi_i = radius, height - 3 - radius
        lo_j, hi_j = radius, width - 3 - radius
        compensated = shifted[:, :, lo_i + 2 : hi_i + 3, lo_j + 2 : hi_j + 3]
        interior = original[:, :, lo_i : hi_i + 1, lo_j : hi_j + 1]
        self.assertEqual(tuple(compensated.shape), tuple(interior.shape))
        scale = float(original.abs().max())
        interior_diff = float((compensated - interior).abs().max())
        self.assertLess(
            interior_diff,
            0.05 * scale,
            f"interior_diff={interior_diff:.4e} scale={scale:.4f}",
        )


class TestArchitectureGuards(unittest.TestCase):
    """(f) The stem refuses an input that is not a whole number of tiles rather
    than silently padding it."""

    def test_non_tile_multiple_input_is_rejected(self):
        for shape in ((3, 70, 80), (3, 72, 78), (3, 74, 82)):
            with self.assertRaises(ValueError, msg=f"{shape} was accepted"):
                TileGridCNN(shape, 128)

    def test_rejection_names_the_shape_and_the_requirement(self):
        with self.assertRaises(ValueError) as caught:
            TileGridCNN((3, 70, 80), 128)
        message = str(caught.exception)
        self.assertIn("(3, 70, 80)", message)
        self.assertIn("multiple of 4", message)

    def test_guard_reaches_through_the_transformer(self):
        with self.assertRaises(ValueError):
            _transformer(input_shape=(3, 70, 80))

    def test_the_only_image_encoder_is_the_tile_grid_one(self):
        torch.manual_seed(0)
        model = PPOTransformer(INPUT_SHAPE, ACTION_SIZE, ram_dim=RAM_DIM, **GEOM).eval()
        self.assertIsInstance(model.cnn, TileGridCNN)


class TestMakeNorm(unittest.TestCase):
    """(h) Every width the tile CNN uses gets a valid GroupNorm with the full
    8 groups -- no walk-down at 32, 64 or 16 channels."""

    def test_group_counts_are_valid(self):
        for channels in (32, 64, 16):
            norm = _make_norm(channels)
            self.assertIsInstance(norm, nn.GroupNorm)
            self.assertEqual(norm.num_channels, channels)
            self.assertEqual(norm.num_groups, min(8, channels))
            self.assertEqual(channels % norm.num_groups, 0)
            with torch.no_grad():
                out = norm(torch.randn(2, channels, 3, 3))
            self.assertEqual(tuple(out.shape), (2, channels, 3, 3))

    def test_tile_cnn_norms_all_use_eight_groups(self):
        model = _tile_cnn()
        norms = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
        self.assertEqual([m.num_channels for m in norms], [32, 32, 32, 64, 16])
        for norm in norms:
            self.assertEqual(norm.num_groups, 8)


class _FakeScreen:
    def __init__(self, image):
        self.image = image


class _FakePyBoy:
    def __init__(self, image):
        self.screen = _FakeScreen(image)


class _FakeEnv:
    """Just enough of PyBoyEnvironment for get_screen_image: a screen image and
    the two config keys it reads. Lets the REAL method be called unbound, so the
    filter assertion is about the production code path rather than a copy of it,
    and no emulator has to boot."""

    def __init__(self, image, scaling_factor):
        self.pyboy = _FakePyBoy(image)
        self.config = {"use_grayscale": False, "scaling_factor": scaling_factor}


class TestObservationResizeFilter(unittest.TestCase):
    """(i) get_screen_image downsamples with BOX, which at scaling_factor 0.5 is
    an exact 2x2 area average: one all-white tile becomes exactly one all-white
    4x4 patch and nothing else is touched. The filter is no longer selectable,
    so this pins the hardcoded value through the real method."""

    TILE = (6, 9)

    def _observe(self, screen, scaling_factor=0.5):
        return PyBoyEnvironment.get_screen_image(_FakeEnv(Image.fromarray(screen), scaling_factor))

    def _white_tile_screen(self):
        """A 160x144 uint8 screen with one all-white 8x8 tile."""
        screen = np.zeros((144, 160, 3), dtype=np.uint8)
        row, col = self.TILE
        screen[row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8, :] = 255
        return screen

    def _split(self, out):
        """The lit tile's 4x4 patch and every other pixel of channel 0."""
        row, col = self.TILE
        plane = out[0]
        rows = slice(row * 4, (row + 1) * 4)
        cols = slice(col * 4, (col + 1) * 4)
        outside = np.ones(plane.shape, dtype=bool)
        outside[rows, cols] = False
        return plane[rows, cols], plane[outside]

    def test_box_gives_one_all_white_patch_and_nothing_else(self):
        out = self._observe(self._white_tile_screen())
        self.assertEqual(out.shape, (3, 72, 80))
        self.assertEqual(out.dtype, np.uint8)
        cell, outside = self._split(out)
        self.assertEqual(cell.shape, (4, 4))
        self.assertTrue(np.all(cell == 255), "the lit tile did not survive as one full patch")
        self.assertEqual(int(np.count_nonzero(outside)), 0)

    def test_downscale_is_an_exact_two_by_two_area_average(self):
        """The half of "it is BOX" that a white tile cannot show: NEAREST also
        passes that one. A tile whose every 2x2 block is three black pixels plus
        one 252 pixel has a block mean of exactly 63, so only an area average
        reproduces it -- nearest picks 0 or 252, bilinear and lanczos weight the
        block unevenly and leak into their neighbours."""
        screen = np.zeros((144, 160, 3), dtype=np.uint8)
        row, col = self.TILE
        rows = slice(row * 8, (row + 1) * 8)
        cols = slice(col * 8, (col + 1) * 8)
        screen[rows, cols, :][1::2, 1::2] = 252  # one lit px per 2x2 block
        out = self._observe(screen)
        cell, outside = self._split(out)
        self.assertTrue(np.all(cell == 63), f"expected an exact 2x2 mean of 63, got {cell}")
        self.assertEqual(int(np.count_nonzero(outside)), 0)

    def test_scaling_factor_one_skips_the_resize(self):
        out = self._observe(self._white_tile_screen(), scaling_factor=1.0)
        self.assertEqual(out.shape, (3, 144, 160))
        self.assertEqual(int(np.count_nonzero(out[0])), 64)


def _model_config(**overrides):
    """A minimal but complete PPOModel config. It mentions none of the removed
    selection keys; the shapes are tiny unless an override says otherwise."""
    config = {
        "device": "cpu",
        "ram_obs_dim": 16,
        "ppo_learning_rate": 1e-4,
        "ppo_gamma": 0.99,
        "ppo_epsilon": 0.2,
        "ppo_value_loss_coef": 0.5,
        "ppo_entropy_coef": 0.01,
        "ppo_entropy_coef_decay": 0.0,
        "ppo_entropy_coef_min": 0.01,
        "ppo_entropy_anneal_enabled": False,
        "ppo_clip_value_loss": True,
        "ppo_max_grad_norm": 0.5,
        "ppo_gae_lambda": 0.95,
        "ppo_adam_eps": 1e-5,
        "ppo_lr_schedule": "constant",
        "num_rollouts": 4,
        "reset_optimizer_on_load": True,
        "d_ram": 16,
        "d_model": 16,
        "n_heads": 2,
        "num_layers": 1,
        "mem_len": 4,
        "ppo_dropout": 0.0,
    }
    config.update(overrides)
    return config


class TestConfigThreading(unittest.TestCase):
    """CHANGE 3: the two RAM-encoder keys still reach the network from the
    config, and their defaults are the new design."""

    def test_ram_encoder_keys_reach_the_network(self):
        model = PPOModel((3, 8, 8), 8, _model_config(ram_hidden_dim=64, ram_input_norm=False))
        net = model.actor_critic
        self.assertIsInstance(net.cnn, TileGridCNN)
        self.assertEqual(net.ram_hidden_dim, 64)
        self.assertFalse(net.ram_input_norm)
        self.assertNotIsInstance(net.ram_encoder.net[0], nn.LayerNorm)
        # d_ram feeds fuse, so widening the RAM encoder must NOT widen it.
        self.assertEqual((net.fuse.in_features, net.fuse.out_features), (32, 16))

    def test_ram_encoder_defaults_are_the_new_design(self):
        net = PPOModel((3, 8, 8), 8, _model_config()).actor_critic
        self.assertIsInstance(net.cnn, TileGridCNN)
        self.assertEqual(net.ram_hidden_dim, 192)
        self.assertTrue(net.ram_input_norm)
        self.assertIsInstance(net.ram_encoder.net[0], nn.LayerNorm)
        self.assertEqual(net.ram_encoder.net[0].normalized_shape, (16,))
        self.assertEqual((net.fuse.in_features, net.fuse.out_features), (32, 16))
        # The norm occupies index 0, so the Linears sit at 1/3/5 (the GELUs at
        # 2/4 carry no parameters). Pinning the indices keeps a silent reorder
        # from looking like a successful warm start.
        ram_keys = sorted(k for k in net.state_dict() if k.startswith("ram_encoder."))
        self.assertEqual(
            ram_keys,
            [
                "ram_encoder.net.0.bias",
                "ram_encoder.net.0.weight",
                "ram_encoder.net.1.bias",
                "ram_encoder.net.1.weight",
                "ram_encoder.net.3.bias",
                "ram_encoder.net.3.weight",
                "ram_encoder.net.5.bias",
                "ram_encoder.net.5.weight",
            ],
        )
        self.assertEqual(tuple(net.state_dict()["ram_encoder.net.1.weight"].shape), (192, 16))
        self.assertEqual(tuple(net.state_dict()["ram_encoder.net.3.weight"].shape), (192, 192))
        self.assertEqual(tuple(net.state_dict()["ram_encoder.net.5.weight"].shape), (16, 192))


class TestDefaultsAreTheNewDesign(unittest.TestCase):
    """The guarantee this slice exists for: a config that mentions NONE of the
    removed keys -- and no encoder-shape key at all -- builds the measured tile
    model, because there is nothing left to select."""

    def test_a_bare_config_builds_the_818_421_parameter_model(self):
        config = _model_config(ram_obs_dim=RAM_DIM)
        config.update({key: value for key, value in GEOM.items() if key != "dropout"})
        config["ppo_dropout"] = GEOM["dropout"]
        for removed in ("cnn_arch", "obs_resize_filter", "ram_hidden_dim", "ram_input_norm"):
            self.assertNotIn(removed, config, f"{removed} is still in the test config")
        net = PPOModel(INPUT_SHAPE, ACTION_SIZE, config).actor_critic
        self.assertIsInstance(net.cnn, TileGridCNN)
        self.assertEqual(net.ram_hidden_dim, 192)
        self.assertTrue(net.ram_input_norm)
        total = sum(p.numel() for p in net.parameters())
        self.assertLessEqual(
            abs(total - TestParameterCounts.TILE_TOTAL),
            TestParameterCounts.TOLERANCE,
            f"defaults build {total:,} params, expected "
            f"{TestParameterCounts.TILE_TOTAL:,} +/- {TestParameterCounts.TOLERANCE}. "
            f"Per-group breakdown:\n{_breakdown(net)}",
        )


if __name__ == "__main__":
    unittest.main()
