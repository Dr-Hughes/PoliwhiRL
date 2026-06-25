# -*- coding: utf-8 -*-
"""Unit coverage for best-checkpoint selection logic.

``_should_update_best`` selects on goal-success rate (directed stages),
intrinsic exploration (free-play), or reward (fallback before any success).

The agent is built via ``__new__`` so we exercise the pure decision logic
without constructing the transformer / vec env.
"""
import unittest
from collections import deque

from PoliwhiRL.agents.PPO.vec_ppo_agent import VecPPOAgent


def _agent(n_goals, config=None, success_flags=(), rewards=(), unique_maps=()):
    a = VecPPOAgent.__new__(VecPPOAgent)
    a.config = {"best_success_window": 5, "best_success_min_episodes": 5}
    if config:
        a.config.update(config)
    a.n_goals = n_goals
    a.best_reward = float("-inf")
    a.best_success_rate = -1.0
    a.best_intrinsic = float("-inf")
    ma = deque(maxlen=5)
    ma.extend(rewards)
    a.episode_data = {
        "moving_avg_reward": ma,
        "episode_unique_maps": list(unique_maps),
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
        a = _agent(1, rewards=[10, 10, 10, 10, 10],
                   success_flags=[1, 1, 0, 1, 0])  # sr = 0.6
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

    def test_freeplay_selects_on_unique_maps(self):
        a = _agent(0, rewards=[9999] * 5, unique_maps=[2, 2, 3, 4, 4])
        self.assertTrue(a._should_update_best())  # intrinsic, not reward
        self.assertGreater(a.best_intrinsic, 0)
        # Higher reward but lower exploration → not better.
        a.episode_data["moving_avg_reward"] = deque([1e9] * 5, maxlen=5)
        a.episode_data["episode_unique_maps"] = [1, 1, 1, 1, 1]
        self.assertFalse(a._should_update_best())


if __name__ == "__main__":
    unittest.main()
