# -*- coding: utf-8 -*-
"""Unit coverage for best-checkpoint selection logic.

``_should_update_best`` selects on goal-success rate (directed stages),
recent run-first discoveries (free-play), or reward (fallback before any
success).

The agent is built via ``__new__`` so we exercise the pure decision logic
without constructing the transformer / vec env.
"""
import unittest
from collections import deque

from PoliwhiRL.agents.PPO.vec_ppo_agent import VecPPOAgent
from PoliwhiRL.environment.gym_env import RAM_OBS_DIM


def _agent(
    n_goals,
    config=None,
    success_flags=(),
    rewards=(),
    discovery_episodes=(),
    episode=100,
):
    a = VecPPOAgent.__new__(VecPPOAgent)
    a.config = {"best_success_window": 5, "best_success_min_episodes": 5}
    if config:
        a.config.update(config)
    a.n_goals = n_goals
    a.episode = episode
    a.best_reward = float("-inf")
    a.best_success_rate = -1.0
    a.best_discoveries = -1
    a.checkpoint_recordings = {
        flag_num: {"episode": ep, "flag": flag_num}
        for flag_num, ep in enumerate(discovery_episodes, start=1)
    }
    ma = deque(maxlen=5)
    ma.extend(rewards)
    a.episode_data = {
        "moving_avg_reward": ma,
        "discovery_log": [
            {"episode": ep, "type": "flag", "key": 1, "step": 0}
            for ep in discovery_episodes
        ],
    }
    win = deque(maxlen=5)
    win.extend(success_flags)
    a._goal_success_window = win
    return a


class TestShouldUpdateBest(unittest.TestCase):
    def test_no_decision_until_window_full(self):
        a = _agent(1, rewards=[1, 2])  # only 2 of maxlen 5
        self.assertFalse(a._should_update_best())

    def test_directed_selects_on_success_rate(self):
        a = _agent(
            1, rewards=[10, 10, 10, 10, 10], success_flags=[1, 1, 0, 1, 0]
        )  # sr = 0.6
        self.assertTrue(a._should_update_best())
        self.assertAlmostEqual(a.best_success_rate, 0.6)
        # A later window with lower success rate must NOT beat it, even if
        # reward is higher.
        a.episode_data["moving_avg_reward"] = deque([999] * 5, maxlen=5)
        a._goal_success_window = deque([1, 0, 0, 0, 0], maxlen=5)  # sr = 0.2
        self.assertFalse(a._should_update_best())

    def test_directed_falls_back_to_reward_before_any_success(self):
        a = _agent(1, rewards=[5, 5, 5, 5, 5], success_flags=[0, 0, 0, 0, 0])
        self.assertTrue(a._should_update_best())  # reward fallback
        self.assertEqual(a.best_reward, 5.0)
        # No improvement in reward → no update.
        self.assertFalse(a._should_update_best())

    def test_freeplay_selects_on_recent_discoveries(self):
        # Two run-first discoveries inside the window (episodes 96-100,
        # window 5, current episode 100) beat the initial -1.
        a = _agent(0, rewards=[9999] * 5, discovery_episodes=[97, 99])
        self.assertTrue(a._should_update_best())
        self.assertEqual(a.best_discoveries, 2)
        # Higher reward but no NEW discoveries in a later window → not
        # better (raw reward must never drive free-play selection).
        a.episode = 200
        a.episode_data["moving_avg_reward"] = deque([1e9] * 5, maxlen=5)
        self.assertFalse(a._should_update_best())

    def test_freeplay_first_full_window_always_writes_best(self):
        # Zero discoveries ever: 0 > -1 → best/ still written once, so the
        # next stage always has a checkpoint to load.
        a = _agent(0, rewards=[1] * 5)
        self.assertTrue(a._should_update_best())
        self.assertEqual(a.best_discoveries, 0)
        self.assertFalse(a._should_update_best())  # but only once

    def test_freeplay_keeps_cumulative_honest_checkpoint_high_water(self):
        # First-honest checkpoints remain meaningful after the rolling window;
        # every newly reached story checkpoint must be eligible for best/.
        a = _agent(0, rewards=[1] * 5, discovery_episodes=[10, 11, 12])
        self.assertTrue(a._should_update_best())
        self.assertEqual(a.best_discoveries, 3)

    def test_freeplay_ignores_seeded_discoveries(self):
        a = _agent(0, rewards=[1] * 5, discovery_episodes=[])
        a.episode_data["discovery_log"] = [
            {"episode": 99, "type": "flag", "key": 31, "step": 100, "seeded": True},
        ]
        self.assertTrue(a._should_update_best())
        self.assertEqual(a.best_discoveries, 0)


class TestLoadModelWindowRebuild(unittest.TestCase):
    """load_model's moving-average deques must always rebuild at the
    CURRENTLY configured best_success_window — not silently keep whatever
    maxlen the checkpoint's deque object happened to unpickle with.

    Regression coverage: a stage transition to a much shorter-episode-
    count stage (e.g. 10x longer episodes, correspondingly fewer of them)
    needs a smaller best_success_window to ever write best/ at all
    (_should_update_best gates on this buffer reaching maxlen). Before the
    fix, `isinstance(value, deque)` was True for an unpickled checkpoint
    deque, so the OLD stage's window silently survived the reload.
    """

    def _make_agent(self, best_success_window):
        from main import load_default_config

        config = load_default_config()
        config.update(
            {
                "device": "cpu",
                "sequence_length": 1,
                "ram_obs_dim": RAM_OBS_DIM,
                "best_success_window": best_success_window,
                "best_success_min_episodes": best_success_window,
                "save_checkpoint": False,
                "checkpoint": None,
                "probe_enabled": False,
                "record": False,
            }
        )
        return VecPPOAgent((3, 36, 40), 9, config)

    def test_window_shrinks_on_reload_with_a_smaller_config(self):
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            big = self._make_agent(best_success_window=100)
            big.episode_data["moving_avg_reward"].extend(range(50))
            big.episode = 50
            big.save_model(temp_dir)

            small = self._make_agent(best_success_window=5)
            small.load_model(temp_dir)

            self.assertEqual(small.episode_data["moving_avg_reward"].maxlen, 5)
            # Content is preserved (tail of what was loaded), just re-bounded.
            self.assertEqual(
                list(small.episode_data["moving_avg_reward"]),
                list(range(45, 50)),
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_frontier_pool_and_checkpoint_recordings_survive_reload(self):
        import os
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            snapshot = os.path.join(temp_dir, "egg.state")
            with open(snapshot, "wb") as f:
                f.write(b"state")
            original = self._make_agent(best_success_window=5)
            original.frontier_pool = [
                {
                    "path": snapshot,
                    "kind": "flag",
                    "flag": 30,
                    "map_count": 0,
                    "bank": 26,
                    "num": 10,
                    "x": 3,
                    "y": 4,
                }
            ]
            original.checkpoint_recordings = {
                30: {"title": "Got Mystery Egg From Mr Pokemon", "status": "recorded"}
            }
            original.best_success_rate = 0.75
            original.best_discoveries = 3
            original.best_reward = 123.0
            original.total_rollouts = 456
            original.save_model(temp_dir)

            loaded = self._make_agent(best_success_window=5)
            loaded.config["checkpoint"] = temp_dir
            loaded.load_model(temp_dir)

            self.assertEqual(len(loaded.frontier_pool), 1)
            self.assertEqual(loaded.frontier_pool[0]["flag"], 30)
            self.assertEqual(loaded.checkpoint_recordings[30]["status"], "recorded")
            self.assertTrue(loaded._loaded_checkpoint)
            self.assertEqual(loaded.best_success_rate, 0.75)
            self.assertEqual(loaded.best_discoveries, 3)
            self.assertEqual(loaded.best_reward, 123.0)
            self.assertEqual(loaded.total_rollouts, 456)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_new_curriculum_stage_resets_selection_high_water(self):
        import os
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            previous = self._make_agent(best_success_window=5)
            previous.best_success_rate = 1.0
            previous.best_discoveries = 8
            previous.best_reward = 999.0
            previous.save_model(temp_dir)

            next_stage = self._make_agent(best_success_window=5)
            next_stage.config["checkpoint"] = os.path.join(temp_dir, "next-stage")
            next_stage.load_model(temp_dir)

            self.assertEqual(next_stage.best_success_rate, -1.0)
            self.assertEqual(next_stage.best_discoveries, -1)
            self.assertEqual(next_stage.best_reward, float("-inf"))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
