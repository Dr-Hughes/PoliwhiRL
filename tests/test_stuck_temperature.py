# -*- coding: utf-8 -*-
"""Behaviour-time stuck-exploration temperature (VecPPOAgent).

Pinned behaviours of ``_apply_stuck_temperature``:

- Off (coefficient 0) returns the policy distribution unchanged.
- Below the stall threshold the distribution is unchanged (temperature 1).
- Deep in a stall the distribution flattens toward uniform (entropy rises),
  capped at the configured maximum temperature.
- Masked actions stay at probability 0 no matter how high the temperature —
  tempering must never resurrect a disallowed action.
"""
import math
import types
import unittest

import torch

from PoliwhiRL.agents.PPO.vec_ppo_agent import VecPPOAgent


def _stub(coef=2.0, max_temp=3.0, threshold=256.0, ssnc_idx=0):
    return types.SimpleNamespace(
        stuck_temperature=coef,
        stuck_temperature_max=max_temp,
        stuck_temperature_threshold=threshold,
        _ssnc_idx=ssnc_idx,
    )


def _ram_with_ssnc(steps, ram_dim=4, batch=1, idx=0):
    """Build (batch, seq=1, ram_dim) with steps_since_novel_cell encoded as
    log1p(steps)/6 at feature index ``idx`` (matches gym_env scaling)."""
    t = torch.zeros((batch, 1, ram_dim))
    t[:, -1, idx] = math.log1p(steps) / 6.0
    return t


def _apply(stub, probs, ram, mask):
    return VecPPOAgent._apply_stuck_temperature(stub, probs, ram, mask)


class TestStuckTemperature(unittest.TestCase):
    def test_off_returns_unchanged(self):
        stub = _stub(coef=0.0)
        probs = torch.tensor([[0.7, 0.2, 0.1, 0.0]])
        out = _apply(stub, probs, _ram_with_ssnc(9999), torch.ones_like(probs))
        self.assertTrue(torch.allclose(out, probs))

    def test_below_threshold_unchanged(self):
        stub = _stub(threshold=256.0)
        probs = torch.tensor([[0.7, 0.2, 0.1, 0.0]])
        mask = torch.ones_like(probs)
        out = _apply(stub, probs, _ram_with_ssnc(10), mask)
        self.assertTrue(torch.allclose(out, probs, atol=1e-6))

    def test_deep_stall_flattens_distribution(self):
        stub = _stub(coef=2.0, max_temp=3.0, threshold=256.0)
        probs = torch.tensor([[0.85, 0.10, 0.03, 0.02]])
        mask = torch.ones_like(probs)
        out = _apply(stub, probs, _ram_with_ssnc(100000), mask)

        def entropy(p):
            return float(-(p * torch.log(p + 1e-12)).sum())

        self.assertGreater(entropy(out), entropy(probs))
        # Still a valid distribution.
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)
        # Argmax preserved (tempering flattens, doesn't reorder).
        self.assertEqual(int(out.argmax()), int(probs.argmax()))

    def test_masked_action_stays_zero(self):
        stub = _stub(coef=2.0, max_temp=3.0, threshold=256.0)
        # Policy leaks a tiny clamped mass onto the masked action (index 3),
        # exactly what the live code's clamp(1e-10, 1.0) produces.
        probs = torch.tensor([[0.7, 0.2, 0.1, 1e-10]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        out = _apply(stub, probs, _ram_with_ssnc(100000), mask)
        self.assertEqual(float(out[0, 3]), 0.0)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)

    def test_temperature_capped(self):
        # Even at absurd stall depth the flattening is bounded by max_temp:
        # with T=max, the result equals normalize(probs ** (1/max)).
        stub = _stub(coef=100.0, max_temp=3.0, threshold=256.0)
        probs = torch.tensor([[0.85, 0.10, 0.03, 0.02]])
        mask = torch.ones_like(probs)
        out = _apply(stub, probs, _ram_with_ssnc(10**6), mask)
        expected = probs.pow(1.0 / 3.0)
        expected = expected / expected.sum()
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
