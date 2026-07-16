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
            # These tests pin the legacy schedule-rewind ACTION; the
            # detection logic is shared with the servo path (tested in
            # TestEntropyServo below).
            "entropy_servo_enabled": False,
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

    def test_short_stage_budget_never_resets_with_default_floor(self):
        # A stage whose OWN total budget is only ~25 completed episodes
        # (e.g. a much-longer-episode stage) can never reach the default
        # 50-episode window floor — a genuinely flat signal still can't
        # trigger the plateau/boost mechanism at all, silently disabling
        # the exact safety net that broke a prior run out of its "camp the
        # first milestone" equilibrium. Fixed via entropy_reset_window_floor
        # (see the next test).
        a = self._agent("unique_maps", "episode_unique_maps", [3.0] * 25)
        a.num_rollouts = 25
        a._check_entropy_plateau()
        self.assertIsNone(a.model.offset)

    def test_entropy_reset_window_floor_enables_short_stage_resets(self):
        a = self._agent("unique_maps", "episode_unique_maps", [3.0] * 25)
        a.num_rollouts = 25
        a.config["entropy_reset_window_floor"] = 10
        a._check_entropy_plateau()
        self.assertIsNotNone(a.model.offset)
        self.assertEqual(a._entropy_reset_count, 1)


class _ServoStubModel:
    def __init__(self):
        self.coef = None
        self.offset = None

    def set_entropy_coef(self, value):
        self.coef = value

    def set_entropy_offset(self, off):
        self.offset = off


class TestEntropyServo(unittest.TestCase):
    """Closed-loop control of MEASURED policy entropy — the coefficient is
    the controlled variable, the rollout's mean behaviour entropy is the
    measured one. Deadband [entropy_target_low, entropy_target_high]."""

    def _agent(self, measured, coef=0.02, boost_until=0, **cfg_overrides):
        a = VecPPOAgent.__new__(VecPPOAgent)
        a.episode = 100
        a.model = _ServoStubModel()
        a._entropy_coef = coef
        a._entropy_boost_until_ep = boost_until
        a._rollout_entropy_sum = measured
        a._rollout_entropy_n = 1
        a.config = {
            "entropy_servo_enabled": True,
            "entropy_target_low": 0.6,
            "entropy_target_high": 1.2,
            "entropy_servo_eta": 0.3,
        }
        a.config.update(cfg_overrides)
        return a

    def test_above_band_lowers_coefficient(self):
        # The stage-3 failure mode: measured ~1.8 nats (near-uniform).
        a = self._agent(measured=1.8)
        a._update_entropy_servo()
        self.assertLess(a._entropy_coef, 0.02)
        self.assertEqual(a.model.coef, a._entropy_coef)

    def test_below_band_raises_coefficient(self):
        a = self._agent(measured=0.2)
        a._update_entropy_servo()
        self.assertGreater(a._entropy_coef, 0.02)

    def test_inside_band_is_a_deadband(self):
        a = self._agent(measured=0.9)
        a._update_entropy_servo()
        self.assertAlmostEqual(a._entropy_coef, 0.02, places=9)

    def test_boost_raises_floor_to_band_top(self):
        # measured 0.9 is inside the normal band, but while boosted the
        # floor is the band top (1.2) → coefficient must rise.
        a = self._agent(measured=0.9, boost_until=200)
        a._update_entropy_servo()
        self.assertGreater(a._entropy_coef, 0.02)

    def test_boost_never_pushes_past_band_top(self):
        # Already above the band top while boosted → pressure goes DOWN,
        # not up (pressure past the band dissolves the policy).
        a = self._agent(measured=1.8, boost_until=200)
        a._update_entropy_servo()
        self.assertLess(a._entropy_coef, 0.02)

    def test_coefficient_clamped(self):
        a = self._agent(measured=5.0, coef=1.5e-4)
        for _ in range(50):
            a._rollout_entropy_sum = 5.0
            a._rollout_entropy_n = 1
            a._update_entropy_servo()
        # Floor is 1e-3: the servo climbs multiplicatively, so recovery
        # speed from a collapse is set by how deep the coefficient sank.
        self.assertGreaterEqual(a._entropy_coef, 1e-3)

    def test_hold_cut_while_entropy_already_descending(self):
        # Anti-windup: entropy above the band but the smoothed trend is
        # already falling toward it — cutting further is how the 2026-07-11
        # run pinned the floor with entropy still at 1.4 and had no braking
        # authority left for the undershoot. The cut must HOLD.
        a = self._agent(measured=1.6)
        a._entropy_measured_ema = 1.9
        a._update_entropy_servo()
        self.assertAlmostEqual(a._entropy_coef, 0.02, places=9)

    def test_cut_resumes_when_descent_stalls(self):
        # Same starting point, but the trend has flattened out above the
        # band — the hold must release and correction resume.
        a = self._agent(measured=1.8)
        a._entropy_measured_ema = 1.8
        a._update_entropy_servo()
        self.assertLess(a._entropy_coef, 0.02)

    def test_hold_raise_while_entropy_already_recovering(self):
        a = self._agent(measured=0.55)
        a._entropy_measured_ema = 0.35
        a._update_entropy_servo()
        self.assertAlmostEqual(a._entropy_coef, 0.02, places=9)

    def test_raise_applies_while_entropy_still_falling(self):
        # Below the band AND still falling — full correction, no hold.
        a = self._agent(measured=0.4)
        a._entropy_measured_ema = 0.6
        a._update_entropy_servo()
        self.assertGreater(a._entropy_coef, 0.02)

    def test_disabled_servo_is_inert(self):
        a = self._agent(measured=1.8, entropy_servo_enabled=False)
        a._update_entropy_servo()
        self.assertAlmostEqual(a._entropy_coef, 0.02, places=9)
        self.assertIsNone(a.model.coef)

    def test_plateau_reset_boosts_servo_instead_of_rewinding(self):
        a = VecPPOAgent.__new__(VecPPOAgent)
        a.num_rollouts = 100
        a.num_envs = 1
        a.episode = 100
        a.stage_start_episode = 0
        a.rollout_idx = 50
        a.n_goals = 0
        a._entropy_last_reset_ep = 0
        a._entropy_reset_count = 0
        a._entropy_boost_until_ep = 0
        a.model = _ServoStubModel()
        a.config = {
            "entropy_plateau_reset": True,
            "entropy_plateau_signal": "unique_maps",
            "entropy_reset_window_fraction": 0.5,
            "entropy_reset_min_fraction": 0.0,
            "entropy_reset_debounce_fraction": 0.1,
            "entropy_reset_max_count": 3,
            "entropy_servo_enabled": True,
        }
        a.episode_data = {"episode_unique_maps": [3.0] * 60}
        a._check_entropy_plateau()
        self.assertGreater(a._entropy_boost_until_ep, a.episode)
        self.assertIsNone(a.model.offset)  # no schedule rewind
        self.assertEqual(a._entropy_reset_count, 1)


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


class TestEntropyStagnationWeighting(unittest.TestCase):
    """Post-mortem (2026-07-14): the flat batch-mean entropy in
    _compute_ppo_losses hid a single stuck env behind 15 healthy ones, same
    limitation as the rollout-level servo but at the loss itself.
    ppo_entropy_stagnation_boost weights each timestep's entropy by its own
    steps_since_novel_cell RAM feature before averaging, so exploration
    pressure concentrates on transitions that are actually stalled."""

    def _model_and_data(self, boost, stagnation_values):
        from PoliwhiRL.models.PPO.ppo_model_implementation import (
            _STEPS_SINCE_NOVEL_CELL_IDX,
        )
        from PoliwhiRL.environment.gym_env import RAM_FEATURE_KEYS

        m = PPOModel.__new__(PPOModel)
        m.config = {}
        m.entropy_stagnation_boost = boost
        m.epsilon = 0.2
        m.value_loss_coef = 0.5
        m.clip_value_loss = False
        m.action_mask_enabled = False
        m._get_entropy_coef = lambda step: 1.0

        batch = len(stagnation_values)
        action_size = 3
        # Identical distribution for every row: isolates the weighting
        # effect from any per-row entropy variation.
        probs_row = torch.tensor([0.5, 0.3, 0.2])
        fixed_probs = probs_row.unsqueeze(0).repeat(batch, 1)
        fixed_values = torch.zeros(batch)
        m.actor_critic = lambda *a, **kw: (fixed_probs, fixed_values, None)

        ram_dim = len(RAM_FEATURE_KEYS)
        ram_states = torch.zeros(batch, 2, ram_dim)
        for i, v in enumerate(stagnation_values):
            ram_states[i, -1, _STEPS_SINCE_NOVEL_CELL_IDX] = v

        actions = torch.zeros(batch, dtype=torch.long)
        old_log_probs = torch.log(fixed_probs[:, 0] + 1e-10)
        data = {
            "states": torch.zeros(batch, 1),
            "ram_states": ram_states,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "returns": torch.zeros(batch),
            "advantages": torch.zeros(batch),
        }
        return m, data

    def _entropy(self, m, data):
        m._compute_ppo_losses(data, step=0)
        return m.last_update_diag["policy_entropy"]

    def test_zero_boost_reproduces_flat_batch_mean(self):
        m, data = self._model_and_data(boost=0.0, stagnation_values=[0.0, 5.0])
        probs_row = torch.tensor([0.5, 0.3, 0.2])
        expected = float(-(probs_row * torch.log(probs_row + 1e-10)).sum())
        self.assertAlmostEqual(self._entropy(m, data), expected, places=5)

    def test_uniform_stagnation_does_not_change_entropy(self):
        """All envs equally (non-)stalled -> boost has nothing to lean on."""
        m, data = self._model_and_data(boost=2.0, stagnation_values=[0.0, 0.0])
        m0, data0 = self._model_and_data(boost=0.0, stagnation_values=[0.0, 0.0])
        self.assertAlmostEqual(self._entropy(m, data), self._entropy(m0, data0), places=5)

    def test_boost_weights_toward_stalled_transitions(self):
        """Stagnation values [0, 1] with boost=1.0 -> weights [1, 2] -> the
        weighted mean is exactly 1.5x the per-row entropy (both rows share
        the same distribution, isolating the weighting arithmetic)."""
        m, data = self._model_and_data(boost=1.0, stagnation_values=[0.0, 1.0])
        probs_row = torch.tensor([0.5, 0.3, 0.2])
        per_row_entropy = float(-(probs_row * torch.log(probs_row + 1e-10)).sum())
        expected = per_row_entropy * 1.5
        self.assertAlmostEqual(self._entropy(m, data), expected, places=5)


if __name__ == "__main__":
    unittest.main()
