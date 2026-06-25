# -*- coding: utf-8 -*-
"""Workstream F: entropy floor / LR schedule / plateau-signal coverage."""
import unittest
from collections import deque

import torch

from PoliwhiRL.models.PPO.ppo_model_implementation import PPOModel
from PoliwhiRL.agents.PPO.vec_ppo_agent import VecPPOAgent


class TestEntropyConstantMode(unittest.TestCase):
    def _model(self, **cfg):
        m = PPOModel.__new__(PPOModel)
        m.config = {"num_rollouts": 100, **cfg}
        m.entropy_coef = 0.04
        m.entropy_min = 0.005
        m._entropy_reset_offset = 0
        return m

    def test_constant_when_anneal_disabled(self):
        m = self._model(ppo_entropy_anneal_enabled=False)
        self.assertAlmostEqual(m._get_entropy_coef(0), 0.04)
        self.assertAlmostEqual(m._get_entropy_coef(99), 0.04)

    def test_anneals_when_enabled(self):
        m = self._model(ppo_entropy_anneal_enabled=True)
        self.assertAlmostEqual(m._get_entropy_coef(0), 0.04, places=4)
        self.assertAlmostEqual(m._get_entropy_coef(100), 0.005, places=4)


class TestLRScheduleModes(unittest.TestCase):
    def _model(self, **cfg):
        m = PPOModel.__new__(PPOModel)
        m.config = {"num_rollouts": 10, "ppo_scheduler_t_max": 10,
                    "ppo_lr_min": 1e-5, **cfg}
        lin = torch.nn.Linear(2, 2)
        m.optimizer = torch.optim.Adam(lin.parameters(), lr=1e-3)
        return m

    def test_constant_lr_stays_flat(self):
        m = self._model(ppo_lr_schedule="constant")
        m._setup_lr_scheduler()
        lr0 = m.optimizer.param_groups[0]["lr"]
        for _ in range(5):
            m.scheduler.step()
        self.assertAlmostEqual(m.optimizer.param_groups[0]["lr"], lr0)

    def test_cosine_lr_decays(self):
        m = self._model(ppo_lr_schedule="cosine")
        m._setup_lr_scheduler()
        lr0 = m.optimizer.param_groups[0]["lr"]
        for _ in range(10):
            m.scheduler.step()
        self.assertLess(m.optimizer.param_groups[0]["lr"], lr0)

    def test_stage_load_resets_lr_to_configured_peak(self):
        # The cross-stage LR bug: after optimizer.load_state_dict the
        # param-group lr is the PREVIOUS stage's decayed value, and a fresh
        # cosine adopts it as its base — stage LRs compounded downward
        # (3e-4 → 2.9e-4 → 1.4e-4 → ...). _reset_base_lr must restore the
        # configured peak before the scheduler is rebuilt.
        m = self._model(ppo_lr_schedule="cosine")
        m.learning_rate = 3e-4
        # Simulate a loaded optimizer that ended the previous stage decayed.
        for group in m.optimizer.param_groups:
            group["lr"] = 5e-5
        m._reset_base_lr()
        m._setup_lr_scheduler()
        self.assertAlmostEqual(m.optimizer.param_groups[0]["lr"], 3e-4)
        self.assertAlmostEqual(m.scheduler.base_lrs[0], 3e-4)


class _StubModel:
    def __init__(self):
        self.offset = None

    def set_entropy_offset(self, off):
        self.offset = off


class TestPlateauSignal(unittest.TestCase):
    def _agent(self, signal, series_key, series, n_goals=0):
        a = VecPPOAgent.__new__(VecPPOAgent)
        a.num_rollouts = 100
        a.num_envs = 1
        a.episode = 100
        a.stage_start_episode = 0
        a.rollout_idx = 50
        a.n_goals = n_goals
        a._entropy_last_reset_ep = 0
        a._entropy_reset_count = 0
        a.model = _StubModel()
        a.config = {
            "entropy_plateau_reset": True,
            "entropy_plateau_signal": signal,
            "entropy_reset_window_fraction": 0.5,   # window 50
            "entropy_reset_min_fraction": 0.0,
            "entropy_reset_debounce_fraction": 0.0,
            "entropy_reset_rewind_fraction": 0.1,
            "entropy_reset_max_count": 3,
        }
        a.episode_data = {series_key: list(series)}
        return a

    def test_unique_maps_flat_triggers_reset(self):
        # 60 flat values → trend test sees no improvement → reset.
        a = self._agent("unique_maps", "episode_unique_maps", [3.0] * 60)
        a._check_entropy_plateau()
        self.assertIsNotNone(a.model.offset)
        self.assertEqual(a._entropy_reset_count, 1)

    def test_unique_maps_rising_does_not_reset(self):
        a = self._agent("unique_maps", "episode_unique_maps",
                        [float(i) for i in range(60)])  # strictly rising
        a._check_entropy_plateau()
        self.assertIsNone(a.model.offset)

    def test_reset_count_capped(self):
        a = self._agent("unique_maps", "episode_unique_maps", [3.0] * 60)
        a._entropy_reset_count = 3  # already at cap
        a._check_entropy_plateau()
        self.assertIsNone(a.model.offset)

    def test_goals_bootstrap_guard(self):
        # goals signal, all zero → bootstrap guard prevents reset.
        a = self._agent("goals", "episode_goals_total", [0] * 60, n_goals=1)
        a._check_entropy_plateau()
        self.assertIsNone(a.model.offset)

    def test_goals_bouncing_below_target_triggers_reset(self):
        # The stage-4 regression: per-episode goals bounce between ladder
        # rungs (1..3) below target 4, ceiling established before the
        # window and never advanced since. The old strict flat test
        # (max==min) could never fire on this; the ceiling test must.
        series = ([1, 2, 3] * 20)[:60]
        a = self._agent("goals", "episode_goals_total", series, n_goals=4)
        a._check_entropy_plateau()
        self.assertIsNotNone(a.model.offset)
        self.assertEqual(a._entropy_reset_count, 1)

    def test_goals_new_ceiling_in_window_blocks_reset(self):
        # Ceiling advanced within the window (2 → 3) → still climbing.
        series = [1, 2] * 25 + ([1, 2, 3] * 4)[:10]
        a = self._agent("goals", "episode_goals_total", series, n_goals=4)
        a._check_entropy_plateau()
        self.assertIsNone(a.model.offset)

    def test_goals_prior_stage_history_is_ignored(self):
        # episode_data carries the previous stage's series (ceiling 3); only
        # 30 episodes belong to this stage (< window 50) → no reset.
        a = self._agent("goals", "episode_goals_total",
                        [3] * 70 + [1] * 30, n_goals=4)
        a.stage_start_episode = 70
        a._check_entropy_plateau()
        self.assertIsNone(a.model.offset)

    def test_solved_stage_blocks_reset(self):
        # A flat exploration signal would normally trigger a reset, but a stage
        # that was already solved must NOT re-inject exploration (regression /
        # reward-hacking, not under-exploration).
        a = self._agent("unique_maps", "episode_unique_maps", [3.0] * 60)
        a._stage_solved = True
        a._check_entropy_plateau()
        self.assertIsNone(a.model.offset)
        self.assertEqual(a._entropy_reset_count, 0)


class TestRealEpisodeBudget(unittest.TestCase):
    """Plateau/controller windows must be sized in completed episodes, not
    num_rollouts × num_envs (which over-counts by episode_length/update_freq)."""

    def _agent(self, cfg):
        a = VecPPOAgent.__new__(VecPPOAgent)
        a.num_rollouts = 600
        a.num_envs = 16
        a.config = cfg
        return a

    def test_divides_by_rollouts_per_episode(self):
        # 600 rollouts × 128 steps / 1024 ep_len = 75 eps/env × 16 = 1200.
        a = self._agent({"ppo_update_frequency": 128, "episode_length": 1024})
        self.assertEqual(a._real_episode_budget(), 1200)

    def test_fallback_when_timing_absent(self):
        # No timing config → legacy product (keeps stub tests stable).
        a = self._agent({})
        self.assertEqual(a._real_episode_budget(), 600 * 16)


if __name__ == "__main__":
    unittest.main()
