# -*- coding: utf-8 -*-
"""Integration tests for snapshot capture/restore on the real emulator.

Exercises the full path the vec worker uses: capture a snapshot mid-episode
(emulator save-state + reward seed facts), then restore it — in the SAME
env and in a FRESH env (the cross-worker case) — and check the restored
episode is a clean fresh episode that cannot re-pay the walked path.
"""
import pickle
import unittest

from PoliwhiRL.environment.gym_env import PyBoyEnvironment
from main import load_default_config


class TestSnapshotCaptureRestore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_default_config()
        cls.config["episode_length"] = 30
        cls.config["goals"] = [{"type": "map", "map_bank": 24, "map_num": 4}]
        cls.config["terminate_on_goal_complete"] = True

    def test_capture_restore_same_env(self):
        env = PyBoyEnvironment(self.config)
        try:
            env.reset()
            for _ in range(5):
                env.step(0)
            snap = env._capture_snapshot("frontier", rung=0)
            self.assertIsInstance(snap["pyboy_state"], bytes)
            self.assertGreater(len(snap["pyboy_state"]), 0)
            self.assertIn("facts", snap)

            walked_cells = set(
                env.reward_calculator._novel_cells_this_episode
            )
            walked_maps = set(env.reward_calculator.explored_maps)
            episode_before = env.episode

            obs = env.restore_snapshot(snap)
            self.assertIsNotNone(obs)
            self.assertIn("image", obs)
            self.assertIn("ram", obs)
            # Fresh episode: full step budget, new episode counter.
            self.assertEqual(env.steps, 0)
            self.assertEqual(env.episode, episode_before + 1)
            self.assertFalse(env.done)
            # The walked path cannot re-pay: novelty sets seeded from facts.
            rc = env.reward_calculator
            self.assertTrue(
                walked_cells.issubset(rc._novel_cells_this_episode)
            )
            self.assertTrue(walked_maps.issubset(rc.explored_maps))
            # The restored episode steps normally.
            _, _, done, _ = env.step(0)
            self.assertIsInstance(done, bool)
            self.assertEqual(env.steps, 1)
        finally:
            env.close()

    def test_capture_restore_pickled_in_fresh_env(self):
        # The cross-worker case: snapshot serialized by one process,
        # restored by another env instance.
        env_a = PyBoyEnvironment(self.config)
        try:
            env_a.reset()
            for _ in range(5):
                env_a.step(0)
            snap = env_a._capture_snapshot("goal", rung=1)
            blob = pickle.dumps(snap)
        finally:
            env_a.close()

        env_b = PyBoyEnvironment(self.config)
        try:
            env_b.reset()
            obs = env_b.restore_snapshot(pickle.loads(blob))
            self.assertIsNotNone(obs)
            self.assertEqual(env_b.steps, 0)
            env_b.step(0)
        finally:
            env_b.close()

    def test_restore_rejects_fully_complete_state(self):
        env = PyBoyEnvironment(self.config)
        try:
            env.reset()
            for _ in range(3):
                env.step(0)
            snap = env._capture_snapshot("goal", rung=1)
            # Forge facts that complete the stage's only goal — the restore
            # must reject the state (returns None → caller falls back to a
            # plain reset).
            snap["facts"]["map_goals_fired"] = [(24, 4)]
            self.assertIsNone(env.restore_snapshot(snap))
            # Env recovers with a plain reset (the worker's fallback).
            obs = env.reset()
            self.assertIsNotNone(obs)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
