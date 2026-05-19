# -*- coding: utf-8 -*-
"""Tests for action replay, story flags, and the best-actions dump.

The pure-Python tests run without an emulator. The end-of-file class
spins up real PyBoy instances to verify the vec env worker protocol.
"""
import os
import json
import tempfile
import shutil
import unittest
import numpy as np

from PoliwhiRL.environment.gym_env import (
    PyBoyEnvironment,
    RAM_OBS_DIM,
    STORY_FLAGS_NUM_BYTES,
    _build_ram_vector,
)
from PoliwhiRL.environment import VecPyBoyEnv
from PoliwhiRL.environment.vec_env import _load_actions_file
from main import load_default_config


class TestRamVectorIncludesStoryFlags(unittest.TestCase):
    def test_total_dim_includes_story_flags(self):
        # 21 base features + 256 story flag bytes
        self.assertEqual(RAM_OBS_DIM, 21 + STORY_FLAGS_NUM_BYTES)

    def test_build_ram_vector_appends_flag_bytes(self):
        env_vars = {
            "X": 5, "Y": 6,
            "map_num": 7, "map_bank": 1, "room": 0, "warp_number": 0,
            "party_info": (10, 50, 100),
            "money": 0,
            "pokedex_seen": 0, "pokedex_owned": 0,
            "collision_down": 0, "collision_up": 0,
            "collision_left": 0, "collision_right": 0,
            "story_flags": np.arange(STORY_FLAGS_NUM_BYTES, dtype=np.int64).astype(
                np.uint8
            ),
        }
        target = (0.0, 0.0, 0.0, 0.0)
        vec = _build_ram_vector(env_vars, target, explored_tile_count=0)
        self.assertEqual(vec.shape, (RAM_OBS_DIM,))
        # The trailing 256 entries should be the normalised story-flag bytes.
        tail = vec[-STORY_FLAGS_NUM_BYTES:]
        expected = (np.arange(STORY_FLAGS_NUM_BYTES) % 256) / 255.0
        self.assertTrue(np.allclose(tail, expected, atol=1e-6))

    def test_wrong_story_flags_length_raises(self):
        env_vars = {
            "X": 0, "Y": 0, "map_num": 0, "map_bank": 0, "room": 0, "warp_number": 0,
            "party_info": (0, 0, 0), "money": 0,
            "pokedex_seen": 0, "pokedex_owned": 0,
            "collision_down": 0, "collision_up": 0,
            "collision_left": 0, "collision_right": 0,
            "story_flags": np.zeros(10, dtype=np.uint8),  # wrong length
        }
        with self.assertRaises(ValueError):
            _build_ram_vector(env_vars, (0, 0, 0, 0), 0)


class TestLoadActionsFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_load_basic(self):
        path = os.path.join(self.tmp, "actions.steps")
        with open(path, "w") as f:
            f.write("1\n2\n3\n")
        self.assertEqual(_load_actions_file(path), [1, 2, 3])

    def test_load_with_comments_and_blanks(self):
        path = os.path.join(self.tmp, "actions.steps")
        with open(path, "w") as f:
            f.write("# header comment\n1\n\n2\n#another\n3\n")
        self.assertEqual(_load_actions_file(path), [1, 2, 3])

    def test_missing_file_returns_empty(self):
        # Missing file should warn and return [] rather than crash.
        self.assertEqual(_load_actions_file(os.path.join(self.tmp, "no.steps")), [])


class TestEnvStoryFlagsLive(unittest.TestCase):
    """Smoke test that the real env produces a RAM vector of correct shape
    and the story-flag region exists at the expected addresses."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = load_default_config()
        self.config["erase"] = False
        self.env = PyBoyEnvironment(self.config)

    def tearDown(self):
        self.env.close()
        shutil.rmtree(self.temp_dir)

    def test_observation_has_full_ram_vector(self):
        obs = self.env.reset()
        self.assertEqual(obs["ram"].shape, (RAM_OBS_DIM,))

    def test_story_flags_extracted(self):
        variables = self.env.ram.get_variables()
        self.assertIn("story_flags", variables)
        self.assertEqual(variables["story_flags"].shape, (STORY_FLAGS_NUM_BYTES,))


class TestReplayActions(unittest.TestCase):
    """Verify env.replay_actions walks Rewards forward and clears per-episode
    counters without touching curriculum state."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = load_default_config()
        self.config["episode_length"] = 50
        self.config["erase"] = False
        self.env = PyBoyEnvironment(self.config)

    def tearDown(self):
        self.env.close()
        shutil.rmtree(self.temp_dir)

    def test_replay_resets_per_episode_counters(self):
        self.env.reset()
        # Replay a handful of no-op actions. Per-episode counters should be
        # cleared, env step counter should be 0. Exploration set is
        # *preserved* across the replay boundary so re-walking visited
        # tiles doesn't double-reward.
        explored_before = set(self.env.reward_calculator.explored_tiles)
        obs = self.env.replay_actions([0, 0, 0, 0])
        self.assertEqual(self.env.steps, 0)
        self.assertEqual(self.env._fitness, 0)
        self.assertEqual(self.env.reward_calculator.steps, 0)
        self.assertEqual(self.env.reward_calculator.cumulative_reward, 0)
        # explored_tiles is preserved (and grows if the replay visited
        # new tiles).
        explored_after = set(self.env.reward_calculator.explored_tiles)
        self.assertTrue(explored_after.issuperset(explored_before))
        self.assertIsInstance(obs, dict)
        self.assertIn("image", obs)
        self.assertIn("ram", obs)

    def test_replay_empty_is_noop(self):
        self.env.reset()
        obs_before = self.env.get_observation()
        obs_after = self.env.replay_actions([])
        self.assertTrue(np.array_equal(obs_before["image"], obs_after["image"]))


class TestVecEnvReplayPool(unittest.TestCase):
    """Spawn a small vec env with an action-replay pool and confirm the
    workers actually replay on reset (rather than starting from raw state)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = load_default_config()
        self.config["episode_length"] = 20
        self.config["erase"] = False
        # Write a small .steps file. 3 no-op actions are enough to confirm
        # the worker accepted the replay and applied it without exploding.
        self.replay_path = os.path.join(self.temp_dir, "warm.steps")
        with open(self.replay_path, "w") as f:
            f.write("# warm-start replay\n0\n0\n0\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_vec_env_loads_and_applies_replay_pool(self):
        cfg = dict(self.config)
        cfg["action_replay_paths"] = [self.replay_path]
        vec = VecPyBoyEnv(cfg, num_envs=2)
        try:
            # Both workers should have replay_index=0 (round-robin over a
            # 1-element pool).
            self.assertEqual(vec.replay_indices, [0, 0])
            self.assertEqual(len(vec.action_replay_paths), 1)
            self.assertEqual(len(vec.replay_action_sequences[0]), 3)
            # Reset should still return a dict obs (replay applied transparently).
            obs = vec.reset()
            self.assertIn("image", obs)
            self.assertIn("ram", obs)
            self.assertEqual(obs["image"].shape[0], 2)
        finally:
            vec.close()

    def test_no_replay_paths_means_no_op(self):
        # Without action_replay_paths the vec env should behave like before.
        vec = VecPyBoyEnv(self.config, num_envs=2)
        try:
            self.assertEqual(vec.action_replay_paths, [])
            self.assertEqual(vec.replay_indices, [-1, -1])
        finally:
            vec.close()


if __name__ == "__main__":
    unittest.main()
