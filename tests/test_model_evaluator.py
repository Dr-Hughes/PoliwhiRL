# -*- coding: utf-8 -*-
"""Regression test for the inference-mode action sampler.

_sample_action feeds a full sequence_length window through the model each
call (unlike the training rollout path, which feeds one frame at a time),
so the forward pass returns (N, L, A) rather than (N, 1, A). The sampler
must act on the LAST position in that window (the current frame), the same
slice vec_ppo_agent.py's rollout-collection path takes before sampling —
not the whole window, which crashed torch.multinomial outright (a tensor
with L*A elements cannot be reduced to a single scalar action).
"""
import unittest

import torch

from PoliwhiRL.evaluator.model_evaluator import _sample_action


class _FakeModel:
    """Stub exposing just what _sample_action calls: _action_mask_for and
    actor_critic(...). Returns a deterministic, distinguishable
    distribution per sequence position so the test can prove which
    position was actually sampled from."""

    def __init__(self, probs_per_position):
        # probs_per_position: (L, A) tensor, one action distribution per
        # window position (position L-1 is "current").
        self._probs = probs_per_position
        self.device = torch.device("cpu")

    def _action_mask_for(self, ram_tensor):
        return None

    def actor_critic(self, state_tensor, ram_tensor, mems, action_mask=None):
        batch = state_tensor.shape[0]
        action_probs = self._probs.unsqueeze(0).expand(batch, -1, -1).clone()
        return action_probs, None, mems


class TestSampleActionActsOnLastPosition(unittest.TestCase):
    def test_does_not_crash_on_a_full_sequence_window(self):
        """Pre-fix: action_probs[0] had shape (L, A), and
        torch.multinomial(..., 1) on a 2D tensor returns (L, 1), which
        .item() cannot convert to a scalar — this reproduces that crash
        if the fix regresses."""
        num_actions = 9
        seq_len = 16
        probs = torch.full((seq_len, num_actions), 1.0 / num_actions)
        model = _FakeModel(probs)
        state_arr = torch.zeros(seq_len, 3, 4, 4).numpy()
        ram_arr = torch.zeros(seq_len, 5).numpy()

        action, _ = _sample_action(model, state_arr, ram_arr, mems=None)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, num_actions)

    def test_samples_from_the_last_position_not_an_earlier_one(self):
        num_actions = 4
        seq_len = 3
        probs = torch.zeros(seq_len, num_actions)
        probs[0, 0] = 1.0  # earlier positions: deterministic action 0
        probs[1, 1] = 1.0
        probs[2, 3] = 1.0  # last (current) position: deterministic action 3
        model = _FakeModel(probs)
        state_arr = torch.zeros(seq_len, 3, 4, 4).numpy()
        ram_arr = torch.zeros(seq_len, 5).numpy()

        action, _ = _sample_action(model, state_arr, ram_arr, mems=None)
        self.assertEqual(action, 3)


if __name__ == "__main__":
    unittest.main()