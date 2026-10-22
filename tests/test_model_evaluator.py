# -*- coding: utf-8 -*-
"""Regression tests for the inference-mode action sampler.

_sample_action feeds a full sequence_length window through the model each
call (unlike the training rollout path, which feeds one frame at a time),
so the forward pass returns (N, L, A) rather than (N, 1, A). The sampler
must act on the LAST position in that window (the current frame), the same
slice vec_ppo_agent.py's rollout-collection path takes before sampling —
not the whole window, which crashed torch.multinomial outright (a tensor
with L*A elements cannot be reduced to a single scalar action).

_sample_action must also delegate to PPOModel.get_action rather than
reimplementing the forward pass itself — a prior version called
model.actor_critic directly and silently skipped get_action's
_behaviour_policy step (the stuck_action_temperature escalation that
training relies on to break multi-step navigation stalls), making
standalone inference runs get stuck well short of milestones live training
reaches most of the time. See TestSampleActionDelegatesToGetAction below.
"""
import unittest

import torch

from PoliwhiRL.evaluator.model_evaluator import _sample_action


class _FakeModel:
    """Stub exposing just what PPOModel.get_action calls: _action_mask_for
    and actor_critic(...). get_action itself mirrors the real
    PPOModel.get_action's slicing/sampling contract (minus
    normalise/behaviour-policy, irrelevant to the slicing tests below) so
    _sample_action's delegation to it is exercised faithfully. Returns a
    deterministic, distinguishable distribution per sequence position so
    the test can prove which position was actually sampled from."""

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

    def get_action(self, state_sequence, ram_sequence, mems=None, assisted=True):
        state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0)
        ram_tensor = torch.FloatTensor(ram_sequence).unsqueeze(0)
        action_mask = self._action_mask_for(ram_tensor)
        action_probs, _, new_mems = self.actor_critic(
            state_tensor, ram_tensor, mems, action_mask=action_mask
        )
        action_probs = action_probs[:, -1]
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs[0, action] + 1e-10).item()
        return action, log_prob, new_mems


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


class TestSampleActionDelegatesToGetAction(unittest.TestCase):
    """_sample_action must call model.get_action (which applies
    _behaviour_policy / stuck_action_temperature by default), not
    reimplement the forward pass and bypass it."""

    def test_calls_get_action_not_actor_critic_directly(self):
        calls = []

        class _SpyModel:
            device = torch.device("cpu")

            def get_action(self, state_sequence, ram_sequence, mems=None, assisted=True):
                calls.append((state_sequence, ram_sequence, mems, assisted))
                return 2, -0.1, "new_mems_sentinel"

            def actor_critic(self, *args, **kwargs):
                raise AssertionError(
                    "_sample_action must not call actor_critic directly — "
                    "it must delegate to get_action so _behaviour_policy "
                    "(stuck_action_temperature) still applies."
                )

        state_arr = torch.zeros(4, 3, 4, 4).numpy()
        ram_arr = torch.zeros(4, 5).numpy()
        action, mems = _sample_action(_SpyModel(), state_arr, ram_arr, mems="prev_mems")

        self.assertEqual(len(calls), 1)
        self.assertEqual(action, 2)
        self.assertEqual(mems, "new_mems_sentinel")
        self.assertTrue(calls[0][3], "assisted must default to True")


if __name__ == "__main__":
    unittest.main()