# -*- coding: utf-8 -*-
"""Tests for RunningMeanStd and RewardScaler (no emulator needed)."""
import unittest
import numpy as np

from PoliwhiRL.utils import RunningMeanStd, RewardScaler


class TestRunningMeanStd(unittest.TestCase):
    def test_default_var_is_one(self):
        rms = RunningMeanStd()
        self.assertAlmostEqual(rms.var, 1.0, places=6)

    def test_update_matches_numpy_on_single_batch(self):
        rms = RunningMeanStd(epsilon=1e-12)  # near-zero pseudo-count
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500) * 7.0 + 3.0
        rms.update(x)
        self.assertAlmostEqual(rms.mean, float(x.mean()), places=4)
        self.assertAlmostEqual(rms.var, float(x.var()), places=4)

    def test_streaming_update_converges_to_numpy(self):
        rms = RunningMeanStd(epsilon=1e-12)
        rng = np.random.default_rng(1)
        chunks = [rng.standard_normal(100) * 5.0 - 2.0 for _ in range(20)]
        for c in chunks:
            rms.update(c)
        full = np.concatenate(chunks)
        self.assertAlmostEqual(rms.mean, float(full.mean()), places=3)
        self.assertAlmostEqual(rms.var, float(full.var()), places=3)

    def test_empty_update_is_noop(self):
        rms = RunningMeanStd()
        before = rms.state_dict()
        rms.update(np.array([]))
        after = rms.state_dict()
        self.assertEqual(before, after)

    def test_state_roundtrip(self):
        rms = RunningMeanStd()
        rms.update(np.linspace(-1, 1, 50))
        state = rms.state_dict()
        rms2 = RunningMeanStd()
        rms2.load_state_dict(state)
        self.assertEqual(rms.mean, rms2.mean)
        self.assertEqual(rms.var, rms2.var)
        self.assertEqual(rms.count, rms2.count)

    def test_merge_state_dict_matches_combined_update(self):
        """merge_state_dict (pre-aggregated stats) must agree with feeding
        the same two batches through update() directly on one instance —
        this is what lets per-worker DomainRewardBalancer estimates be
        combined without re-streaming raw samples."""
        rng = np.random.default_rng(2)
        a = rng.standard_normal(300) * 3.0 + 1.0
        b = rng.standard_normal(500) * 3.0 + 1.0

        combined = RunningMeanStd(epsilon=1e-12)
        combined.update(a)
        combined.update(b)

        rms_a = RunningMeanStd(epsilon=1e-12)
        rms_a.update(a)
        rms_b = RunningMeanStd(epsilon=1e-12)
        rms_b.update(b)
        rms_a.merge_state_dict(rms_b.state_dict())

        self.assertAlmostEqual(combined.mean, rms_a.mean, places=6)
        self.assertAlmostEqual(combined.var, rms_a.var, places=6)
        self.assertAlmostEqual(combined.count, rms_a.count, places=6)

    def test_merge_state_dict_ignores_zero_count_state(self):
        rms = RunningMeanStd()
        rms.update(np.array([5.0, 5.0, 5.0]))
        before = rms.state_dict()
        rms.merge_state_dict({"mean": 999.0, "var": 999.0, "count": 0.0})
        self.assertEqual(rms.state_dict(), before)


class TestRewardScaler(unittest.TestCase):
    def test_running_return_resets_on_done(self):
        scaler = RewardScaler(gamma=0.9, num_envs=2)
        # Two envs; env 0 done at step 1.
        scaler.observe(rewards=[1.0, 1.0], dones=[False, False])
        # env 0: running = 0*0.9 + 1 = 1
        # env 1: running = 0*0.9 + 1 = 1
        self.assertAlmostEqual(scaler.running_returns[0], 1.0, places=6)
        self.assertAlmostEqual(scaler.running_returns[1], 1.0, places=6)

        scaler.observe(rewards=[2.0, 2.0], dones=[True, False])
        # env 0: running = 0.9*1 + 2 = 2.9, then reset -> 0
        # env 1: running = 0.9*1 + 2 = 2.9
        self.assertAlmostEqual(scaler.running_returns[0], 0.0, places=6)
        self.assertAlmostEqual(scaler.running_returns[1], 2.9, places=6)

    def test_scale_factor_shrinks_after_large_rewards(self):
        scaler = RewardScaler(gamma=0.99, num_envs=1)
        scale_initial = scaler.scale_factor()
        for _ in range(200):
            scaler.observe(rewards=[100.0], dones=[False])
        scale_after = scaler.scale_factor()
        self.assertLess(scale_after, scale_initial)
        # With consistent large rewards the running return grows so the
        # scale factor should be well below 1.
        self.assertLess(scale_after, 0.1)

    def test_state_roundtrip(self):
        s = RewardScaler(gamma=0.9, num_envs=4)
        s.observe(rewards=[1, 2, 3, 4], dones=[False, False, False, False])
        s.observe(rewards=[1, 2, 3, 4], dones=[False, True, False, False])
        state = s.state_dict()
        s2 = RewardScaler(gamma=0.9, num_envs=4)
        s2.load_state_dict(state)
        np.testing.assert_array_almost_equal(s.running_returns, s2.running_returns)
        self.assertAlmostEqual(s.scale_factor(), s2.scale_factor(), places=8)


if __name__ == "__main__":
    unittest.main()
