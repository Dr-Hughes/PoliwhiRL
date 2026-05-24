# -*- coding: utf-8 -*-
"""Tests for action replay, story flags, the multi-trajectory .steps
format, and the per-checkpoint trajectory dump.

The pure-Python tests run without an emulator. The end-of-file class
spins up real PyBoy instances to verify the vec env worker protocol.
"""
import os
import tempfile
import shutil
import unittest
import numpy as np

from PoliwhiRL.environment.gym_env import (
    PyBoyEnvironment,
    RAM_OBS_DIM,
    STORY_FLAGS_NUM_BYTES,
    _BASE_RAM_LEN,
    _DERIVED_FLAG_KEYS,
    _DERIVED_FLAG_TABLE,
    _extract_derived_flags,
    _build_ram_vector,
    N_LOC_GOALS_RAM_IDX,
    N_POK_GOALS_RAM_IDX,
)
from PoliwhiRL.environment import VecPyBoyEnv
from PoliwhiRL.environment.vec_env import (
    _load_actions_file,
    _load_replay_pool,
    write_actions_file,
)
from main import load_default_config


# Target tuple shape: (target_x, target_y, target_map, target_map_bank,
# has_active_target). Keep this constant in one place so test files don't
# drift from the env's `get_current_target_vector` contract.
DEFAULT_TARGET = (0.0, 0.0, 0.0, 0.0, 0.0)


def _make_env_vars(story_flags=None):
    """Helper to build a minimal env_vars dict for _build_ram_vector tests."""
    if story_flags is None:
        story_flags = np.zeros(STORY_FLAGS_NUM_BYTES, dtype=np.uint8)
    return {
        "X": 5, "Y": 6,
        "map_num": 7, "map_bank": 1, "room": 0, "warp_number": 0,
        "party_info": (1, 10, 50, 100),
        "money": 0,
        "pokedex_seen": 0, "pokedex_owned": 0,
        "collision_down": 0, "collision_up": 0,
        "collision_left": 0, "collision_right": 0,
        "story_flags": story_flags,
        "battle_type": 0,
        "johto_badges": 0,
        "player_state": 0,
        "key_items_count": 0,
        "game_hour": 12,
        "bgm_id": 0,
    }


def _build(env_vars, target=DEFAULT_TARGET, explored=0, loc_goals=0, pok_goals=0, lvl_goals=0):
    return _build_ram_vector(
        env_vars,
        target,
        explored_tile_count=explored,
        n_location_goals_completed=loc_goals,
        n_pokedex_goals_completed=pok_goals,
        n_level_goals_completed=lvl_goals,
    )


class TestRamVectorIncludesStoryFlags(unittest.TestCase):
    def test_total_dim_includes_all_sections(self):
        expected = _BASE_RAM_LEN + len(_DERIVED_FLAG_KEYS)
        self.assertEqual(RAM_OBS_DIM, expected)

    def test_build_ram_vector_shape(self):
        env_vars = _make_env_vars(
            np.arange(STORY_FLAGS_NUM_BYTES, dtype=np.int64).astype(np.uint8)
        )
        vec = _build(env_vars)
        self.assertEqual(vec.shape, (RAM_OBS_DIM,))

    def test_derived_flags_extracted_from_story_flags(self):
        sf = np.zeros(STORY_FLAGS_NUM_BYTES, dtype=np.uint8)
        sf[2] = 0x01  # has_cut: byte 2, bit 0
        env_vars = _make_env_vars(sf)
        vec = _build(env_vars)
        derived = vec[_BASE_RAM_LEN:_BASE_RAM_LEN + len(_DERIVED_FLAG_KEYS)]
        self.assertAlmostEqual(float(derived[0]), 1.0)

    def test_derived_flags_position(self):
        env_vars = _make_env_vars()
        vec = _build(env_vars)
        derived_slice = vec[_BASE_RAM_LEN: _BASE_RAM_LEN + len(_DERIVED_FLAG_KEYS)]
        self.assertEqual(len(derived_slice), len(_DERIVED_FLAG_KEYS))
        self.assertTrue(np.allclose(derived_slice, 0.0))

    def test_raw_features_in_base(self):
        env_vars = _make_env_vars()
        env_vars["battle_type"] = 2
        env_vars["johto_badges"] = 0xFF
        env_vars["player_state"] = 4
        env_vars["key_items_count"] = 10
        env_vars["game_hour"] = 18
        env_vars["bgm_id"] = 42
        vec = _build(env_vars)
        base_end = vec[:_BASE_RAM_LEN]
        # Last 6 entries of base section are the priority-1 raw features.
        self.assertAlmostEqual(base_end[-6], 2 / 255.0, places=5)  # battle_type
        self.assertAlmostEqual(base_end[-5], 0xFF / 255.0, places=5)  # johto_badges
        self.assertAlmostEqual(base_end[-4], 4 / 255.0, places=5)  # player_state
        self.assertAlmostEqual(base_end[-3], 10 / 25.0, places=5)  # key_items_count
        self.assertAlmostEqual(base_end[-2], 18 / 255.0, places=5)  # game_hour
        self.assertAlmostEqual(base_end[-1], 42 / 255.0, places=5)  # bgm_id

    def test_wrong_story_flags_length_raises(self):
        env_vars = _make_env_vars(np.zeros(10, dtype=np.uint8))
        with self.assertRaises(ValueError):
            _build(env_vars)

    def test_goal_progress_features_present(self):
        # Raw goal-count features sit at the named indices.
        env_vars = _make_env_vars()
        vec = _build(env_vars, loc_goals=3, pok_goals=2)
        self.assertEqual(float(vec[N_LOC_GOALS_RAM_IDX]), 3.0)
        self.assertEqual(float(vec[N_POK_GOALS_RAM_IDX]), 2.0)

    def test_target_map_bank_present(self):
        # Target tuple's bank field threads through to its own RAM slot.
        env_vars = _make_env_vars()
        target = (10.0, 5.0, 50.0, 25.0, 1.0)
        vec = _build(env_vars, target=target)
        # target_map_bank sits between target_map and has_active_target.
        # We don't hard-code an index here — just verify it survives.
        from PoliwhiRL.environment.gym_env import RAM_FEATURE_INDEX
        bank_idx = RAM_FEATURE_INDEX["target_map_bank"]
        self.assertAlmostEqual(float(vec[bank_idx]), 25.0 / 255.0, places=5)


class TestExtractDerivedFlags(unittest.TestCase):
    def test_flag_25_has_starter(self):
        flags = np.zeros(STORY_FLAGS_NUM_BYTES, dtype=np.uint8)
        flags[3] = 0b0000_0010  # byte 3, bit 1 = flag 25
        result = _extract_derived_flags(flags)
        self.assertAlmostEqual(result["has_starter"], 1.0)

    def test_flag_16_has_cut(self):
        flags = np.zeros(STORY_FLAGS_NUM_BYTES, dtype=np.uint8)
        flags[2] = 0b0000_0001  # byte 2, bit 0 = flag 16
        result = _extract_derived_flags(flags)
        self.assertAlmostEqual(result["has_cut"], 1.0)

    def test_flag_53_rocket_hideout(self):
        flags = np.zeros(STORY_FLAGS_NUM_BYTES, dtype=np.uint8)
        flags[6] = 0b0010_0000  # byte 6, bit 5 = flag 53
        result = _extract_derived_flags(flags)
        self.assertAlmostEqual(result["rocket_cleared_hideout"], 1.0)

    def test_flag_1611_ilex_gate(self):
        flags = np.zeros(STORY_FLAGS_NUM_BYTES, dtype=np.uint8)
        flags[201] = 0b0000_1000  # byte 201, bit 3 = flag 1611
        result = _extract_derived_flags(flags)
        self.assertAlmostEqual(result["ilex_gate_clear"], 1.0)

    def test_all_zero_when_no_flags_set(self):
        flags = np.zeros(STORY_FLAGS_NUM_BYTES, dtype=np.uint8)
        result = _extract_derived_flags(flags)
        for name, val in result.items():
            self.assertAlmostEqual(val, 0.0, msg=f"{name} should be 0")

    def test_derived_flag_keys_match_table(self):
        self.assertEqual(len(_DERIVED_FLAG_KEYS), len(_DERIVED_FLAG_TABLE))
        for key, (_, name) in zip(_DERIVED_FLAG_KEYS, _DERIVED_FLAG_TABLE):
            self.assertEqual(key, name)


class TestLoadActionsFile(unittest.TestCase):
    """Tests for the multi-trajectory .steps file format. The loader
    returns a list[list[int]] and tolerates the legacy single-trajectory
    form (no `# trajectory` markers).
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_legacy_single_trajectory_format(self):
        # Plain file with no trajectory markers should be treated as a
        # one-trajectory file.
        path = os.path.join(self.tmp, "actions.steps")
        with open(path, "w") as f:
            f.write("1\n2\n3\n")
        self.assertEqual(_load_actions_file(path), [[1, 2, 3]])

    def test_legacy_with_comments_and_blanks(self):
        path = os.path.join(self.tmp, "actions.steps")
        with open(path, "w") as f:
            f.write("# header comment\n1\n\n2\n#another\n3\n")
        self.assertEqual(_load_actions_file(path), [[1, 2, 3]])

    def test_multi_trajectory_format(self):
        path = os.path.join(self.tmp, "actions.steps")
        with open(path, "w") as f:
            f.write(
                "# trajectory 0\n"
                "1\n2\n3\n"
                "# trajectory 1\n"
                "4\n5\n"
                "# trajectory 2\n"
                "# length=2\n"
                "6\n7\n"
            )
        self.assertEqual(_load_actions_file(path), [[1, 2, 3], [4, 5], [6, 7]])

    def test_missing_file_returns_empty(self):
        self.assertEqual(_load_actions_file(os.path.join(self.tmp, "no.steps")), [])

    def test_write_round_trip(self):
        path = os.path.join(self.tmp, "actions.steps")
        write_actions_file(
            path,
            [[1, 2, 3], [4, 5, 6, 7]],
            metadata=[{"length": 3}, {"length": 4, "note": "second"}],
        )
        loaded = _load_actions_file(path)
        self.assertEqual(loaded, [[1, 2, 3], [4, 5, 6, 7]])

    def test_write_skips_empty_trajectories(self):
        path = os.path.join(self.tmp, "actions.steps")
        write_actions_file(path, [[], [1, 2], []])
        self.assertEqual(_load_actions_file(path), [[1, 2]])

    def test_load_replay_pool_concatenates_files(self):
        a = os.path.join(self.tmp, "a.steps")
        b = os.path.join(self.tmp, "b.steps")
        write_actions_file(a, [[1, 2], [3, 4]])
        # Legacy single-trajectory file mixes in cleanly.
        with open(b, "w") as f:
            f.write("# legacy\n5\n6\n")
        paths, trajectories = _load_replay_pool([a, b])
        self.assertIn(a, paths)
        self.assertIn(b, paths)
        # Order: a's trajectories first, then b's.
        self.assertEqual(trajectories, [[1, 2], [3, 4], [5, 6]])

    def test_load_replay_pool_glob_expansion(self):
        for i in range(3):
            path = os.path.join(self.tmp, f"chunk_{i}.steps")
            write_actions_file(path, [[i, i + 1]])
        paths, trajectories = _load_replay_pool(
            [os.path.join(self.tmp, "chunk_*.steps")]
        )
        # All three files matched; trajectories pooled.
        self.assertEqual(len(trajectories), 3)
        # Sorted by filename, so first should be [0, 1].
        self.assertEqual(trajectories[0], [0, 1])


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

    def test_new_raw_features_present(self):
        variables = self.env.ram.get_variables()
        for key in ["battle_type", "johto_badges", "player_state",
                     "key_items_count", "game_hour", "bgm_id"]:
            self.assertIn(key, variables, f"{key} missing from get_variables")
            self.assertIsInstance(variables[key], int)

    def test_initial_goal_progress_is_zero(self):
        # Fresh env: no goals completed.
        obs = self.env.reset()
        self.assertEqual(float(obs["ram"][N_LOC_GOALS_RAM_IDX]), 0.0)
        self.assertEqual(float(obs["ram"][N_POK_GOALS_RAM_IDX]), 0.0)


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

    def test_replay_resets_per_episode_counters_only(self):
        self.env.reset()
        # Replay a handful of no-op actions. Per-episode counters should be
        # cleared, env step counter should be 0. Exploration set is
        # *preserved* across the replay boundary so re-walking visited
        # tiles doesn't double-reward. Goal-progress counters are likewise
        # preserved.
        explored_before = set(self.env.reward_calculator.explored_tiles)
        goals_before = self.env.reward_calculator.N_goals
        loc_before = self.env.reward_calculator.n_location_goals_completed()
        pok_before = self.env.reward_calculator.n_pokedex_goals_completed()

        obs = self.env.replay_actions([0, 0, 0, 0])

        self.assertEqual(self.env.steps, 0)
        self.assertEqual(self.env._fitness, 0)
        self.assertEqual(self.env.reward_calculator.steps, 0)
        self.assertEqual(self.env.reward_calculator.cumulative_reward, 0)
        # Curriculum + exploration state preserved (and possibly grown).
        explored_after = set(self.env.reward_calculator.explored_tiles)
        self.assertTrue(explored_after.issuperset(explored_before))
        self.assertGreaterEqual(self.env.reward_calculator.N_goals, goals_before)
        self.assertGreaterEqual(
            self.env.reward_calculator.n_location_goals_completed(), loc_before
        )
        self.assertGreaterEqual(
            self.env.reward_calculator.n_pokedex_goals_completed(), pok_before
        )

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
    workers actually replay on reset (rather than starting from raw state).
    Uses the multi-trajectory pool API.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = load_default_config()
        self.config["episode_length"] = 20
        self.config["erase"] = False
        # Multi-trajectory file with two short no-op routes. Each worker
        # samples uniformly between them on each reset.
        self.replay_path = os.path.join(self.temp_dir, "warm.steps")
        write_actions_file(
            self.replay_path,
            [[0, 0, 0], [0, 0]],
            metadata=[{"length": 3}, {"length": 2}],
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_vec_env_loads_pool_from_replay_path(self):
        cfg = dict(self.config)
        cfg["action_replay_paths"] = [self.replay_path]
        vec = VecPyBoyEnv(cfg, num_envs=2)
        try:
            self.assertEqual(len(vec.action_replay_paths), 1)
            # Pool contains both trajectories from the file.
            self.assertEqual(len(vec.replay_trajectories), 2)
            self.assertEqual(vec.replay_trajectories[0], [0, 0, 0])
            self.assertEqual(vec.replay_trajectories[1], [0, 0])
            # Reset should still return a dict obs (replay applied transparently).
            obs = vec.reset()
            self.assertIn("image", obs)
            self.assertIn("ram", obs)
            self.assertEqual(obs["image"].shape[0], 2)
        finally:
            vec.close()

    def test_no_replay_paths_means_no_op(self):
        vec = VecPyBoyEnv(self.config, num_envs=2)
        try:
            self.assertEqual(vec.action_replay_paths, [])
            self.assertEqual(vec.replay_trajectories, [])
        finally:
            vec.close()

    def test_multiple_files_concatenate_into_pool(self):
        # File A: 1 trajectory. File B: 2 trajectories. Pool should hold 3.
        path_a = os.path.join(self.temp_dir, "a.steps")
        path_b = os.path.join(self.temp_dir, "b.steps")
        write_actions_file(path_a, [[0, 0]])
        write_actions_file(path_b, [[0, 0, 0], [0]])
        cfg = dict(self.config)
        cfg["action_replay_paths"] = [path_a, path_b]
        vec = VecPyBoyEnv(cfg, num_envs=2)
        try:
            self.assertEqual(len(vec.replay_trajectories), 3)
        finally:
            vec.close()

    def test_set_replay_pool_hot_swaps(self):
        # Workers should accept a new pool after init via set_replay_pool.
        cfg = dict(self.config)
        cfg["action_replay_paths"] = [self.replay_path]
        vec = VecPyBoyEnv(cfg, num_envs=2)
        try:
            vec.set_replay_pool([[0, 0, 0, 0]])
            self.assertEqual(vec.replay_trajectories, [[0, 0, 0, 0]])
            # And reset still works after the swap.
            obs = vec.reset()
            self.assertEqual(obs["image"].shape[0], 2)
        finally:
            vec.close()


if __name__ == "__main__":
    unittest.main()
