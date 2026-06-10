# -*- coding: utf-8 -*-
"""Phase-1 action mask: confirms the mask is derived correctly from the
script / UI state bytes, applied consistently at action sampling time,
and propagates through the model's softmax."""
import unittest

import torch

from PoliwhiRL.environment.action_mask import (
    ACTION_SIZE,
    NOOP, A, B, LEFT, RIGHT, UP, DOWN, START, SELECT,
    compute_action_mask,
    compute_action_mask_from_byte_state,
)
from PoliwhiRL.environment.gym_env import RAM_FEATURE_INDEX


# --------- mask-from-bytes (helper exists for fixture writing) ----------

class TestByteStateMask(unittest.TestCase):
    def test_dialog_state_allows_only_noop_a_b(self):
        # script_active=1, text_box=1 → dialog → noop, A, B only.
        m = compute_action_mask_from_byte_state(d438_byte=255, cf07_byte=7)
        self.assertEqual(m[NOOP], 1.0)
        self.assertEqual(m[A], 1.0)
        self.assertEqual(m[B], 1.0)
        for a in (LEFT, RIGHT, UP, DOWN, START, SELECT):
            self.assertEqual(m[a], 0.0, f"action {a} should be masked in dialog")

    def test_menu_overlay_allows_directional_and_ab(self):
        # script_active=1, text_box=0 → menu / keyboard.
        m = compute_action_mask_from_byte_state(d438_byte=255, cf07_byte=0)
        for a in (NOOP, A, B, LEFT, RIGHT, UP, DOWN):
            self.assertEqual(m[a], 1.0, f"action {a} should be allowed in menu overlay")
        for a in (START, SELECT):
            self.assertEqual(m[a], 0.0, f"action {a} should be blocked in menu overlay")

    def test_walking_default_blocks_start_select(self):
        # Indoor walking baseline (cf07=5).
        m = compute_action_mask_from_byte_state(d438_byte=0, cf07_byte=5)
        for a in (NOOP, A, B, LEFT, RIGHT, UP, DOWN):
            self.assertEqual(m[a], 1.0)
        for a in (START, SELECT):
            self.assertEqual(m[a], 0.0)

    def test_walking_with_menus_opt_in(self):
        m = compute_action_mask_from_byte_state(
            d438_byte=0, cf07_byte=5, allow_menus_walking=True
        )
        for a in (START, SELECT):
            self.assertEqual(m[a], 1.0, "opt-in should unmask menu actions during walking")

    def test_menus_opt_in_does_not_leak_into_dialog(self):
        """Even with allow_menus_walking=True, start/select must stay
        blocked when a script is running."""
        m = compute_action_mask_from_byte_state(
            d438_byte=255, cf07_byte=7, allow_menus_walking=True
        )
        self.assertEqual(m[START], 0.0)
        self.assertEqual(m[SELECT], 0.0)


# --------- mask-from-RAM-vector (the real path used in training) --------

class TestRAMVectorMask(unittest.TestCase):
    """Build the indicators a real RAM observation would carry, then check
    compute_action_mask returns the same shape as the byte-based helper."""

    def _ram_with(self, script_active, text_box):
        # Build a zero RAM vector of the right size and set just the two
        # bits the mask cares about. Anything else is irrelevant.
        ram_dim = max(RAM_FEATURE_INDEX.values()) + 1
        vec = torch.zeros((1, ram_dim))
        vec[0, RAM_FEATURE_INDEX["script_active"]] = float(script_active)
        vec[0, RAM_FEATURE_INDEX["ui_state_text_box"]] = float(text_box)
        return vec

    def test_dialog_via_ram(self):
        mask = compute_action_mask(self._ram_with(script_active=1, text_box=1))
        self.assertEqual(mask.shape, (1, ACTION_SIZE))
        expected = compute_action_mask_from_byte_state(d438_byte=255, cf07_byte=7)
        self.assertEqual(mask[0].tolist(), expected)

    def test_walking_via_ram(self):
        mask = compute_action_mask(self._ram_with(script_active=0, text_box=0))
        expected = compute_action_mask_from_byte_state(d438_byte=0, cf07_byte=0)
        self.assertEqual(mask[0].tolist(), expected)

    def test_batched(self):
        # Stack a dialog row and a walking row.
        ram = torch.cat([
            self._ram_with(script_active=1, text_box=1),
            self._ram_with(script_active=0, text_box=0),
        ], dim=0)
        mask = compute_action_mask(ram)
        self.assertEqual(mask.shape, (2, ACTION_SIZE))
        # Row 0 is dialog → only noop/A/B allowed.
        self.assertEqual(mask[0, LEFT].item(), 0.0)
        self.assertEqual(mask[0, A].item(), 1.0)
        # Row 1 is walking → directional allowed.
        self.assertEqual(mask[1, LEFT].item(), 1.0)
        self.assertEqual(mask[1, START].item(), 0.0)


# --------- mask propagated through PPOTransformer ------------------------

class TestTransformerMask(unittest.TestCase):
    """End-to-end: feed the model a mask, confirm masked actions get zero
    softmax probability."""

    def setUp(self):
        from PoliwhiRL.models.PPO.PPOTransformer import PPOTransformer
        # Small model for the test — just needs to forward cleanly.
        self.model = PPOTransformer(
            input_shape=(1, 8, 8),
            action_size=ACTION_SIZE,
            ram_dim=10,
            d_model=16,
            d_ram=8,
            n_heads=2,
            num_layers=1,
            mem_len=4,
        )
        self.model.eval()

    def test_masked_actions_get_zero_probability(self):
        batch, seq_len = 2, 3
        x_image = torch.zeros((batch, seq_len, 1, 8, 8))
        x_ram = torch.zeros((batch, seq_len, 10))
        # Dialog mask: only noop/A/B.
        mask = torch.zeros((batch, ACTION_SIZE))
        mask[:, NOOP] = 1.0
        mask[:, A] = 1.0
        mask[:, B] = 1.0

        with torch.no_grad():
            probs, _, _ = self.model(x_image, x_ram, action_mask=mask)

        # Every masked action should have ~0 probability.
        for a in (LEFT, RIGHT, UP, DOWN, START, SELECT):
            self.assertLess(probs[:, a].max().item(), 1e-6,
                            f"action {a} should be ~0 prob when masked")
        # And the unmasked ones must sum to ~1.
        unmasked_sum = probs[:, [NOOP, A, B]].sum(dim=-1)
        self.assertTrue(torch.allclose(unmasked_sum, torch.ones(batch), atol=1e-5))

    def test_unmasked_model_still_normalises(self):
        """No mask passed → normal softmax over all 9 actions."""
        batch, seq_len = 1, 2
        x_image = torch.zeros((batch, seq_len, 1, 8, 8))
        x_ram = torch.zeros((batch, seq_len, 10))
        with torch.no_grad():
            probs, _, _ = self.model(x_image, x_ram)
        self.assertEqual(probs.shape, (1, ACTION_SIZE))
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
