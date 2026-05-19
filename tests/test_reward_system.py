# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
import numpy as np
from PoliwhiRL.reward_evaluator import evaluate_reward_system
from PoliwhiRL.environment.rewards import Rewards
from main import load_default_config, load_user_config, merge_configs


class TestRewardSystem(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_loc = "./configs/evaluate_reward_system.json"
        self.config = load_default_config()
        self.user_config = load_user_config(self.config_loc)
        self.config = merge_configs(self.config, self.user_config)
        self.config["output_path"] = self.temp_dir

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_reward_system_functionality_1_goal(self):
        """Test that reward system works correctly with 1 goal"""
        self.config["N_goals_target"] = 1
        rewards = evaluate_reward_system(self.config)
        # Test that rewards are computed (non-empty list)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        # Test that all rewards are finite numbers
        self.assertTrue(all(np.isfinite(r) for r in rewards))

    def test_reward_system_functionality_2_goals(self):
        """Test that reward system works correctly with 2 goals"""
        self.config["N_goals_target"] = 2
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))

    def test_reward_system_functionality_3_goals(self):
        """Test that reward system works correctly with 3 goals"""
        self.config["N_goals_target"] = 3
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        # Test that negative total is reasonable for inefficient play
        total_reward = np.sum(rewards)
        self.assertGreater(total_reward, -1000)

    def test_reward_system_functionality_4_goals(self):
        """Test that reward system works correctly with 4 goals"""
        self.config["N_goals_target"] = 4
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        total_reward = np.sum(rewards)
        self.assertGreater(total_reward, -1000)

    def test_reward_system_functionality_5_goals(self):
        """Test that reward system works correctly with 5 goals"""
        self.config["N_goals_target"] = 5
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        total_reward = np.sum(rewards)
        self.assertGreater(total_reward, -1000)

    def test_reward_system_functionality_6_goals(self):
        """Test that reward system works correctly with 6 goals"""
        self.config["N_goals_target"] = 6
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        total_reward = np.sum(rewards)
        self.assertGreater(total_reward, -1000)

    def test_reward_system_functionality_7_goals(self):
        """Test that reward system works correctly with 7 goals"""
        self.config["N_goals_target"] = 7
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        total_reward = np.sum(rewards)
        self.assertGreater(total_reward, -1000)

    def test_reward_scaling_and_clipping(self):
        """Test that rewards are properly scaled and clipped"""
        config = load_default_config()
        config["N_goals_target"] = 1
        config["episode_length"] = 100
        config["location_goals"] = [[[1, 1, 1]]]
        config["pokedex_goals"] = {}

        reward_system = Rewards(config)

        self.assertEqual(reward_system.clip, 1000)

        self.assertLessEqual(reward_system.step_penalty, 0)
        self.assertGreaterEqual(reward_system.step_penalty, -1.0)

        self.assertGreater(reward_system.goal_reward, 0)
        self.assertGreater(reward_system.sequence_bonus, 0)
        self.assertGreater(reward_system.checkpoint_bonus, 0)
        self.assertGreater(reward_system.all_goals_bonus, 0)

    def test_reward_goal_achievement(self):
        """Test that goal achievement produces positive rewards"""
        config = load_default_config()
        config["N_goals_target"] = 1
        config["episode_length"] = 1000
        config["location_goals"] = [[[5, 5, 1]]]
        config["pokedex_goals"] = {}

        reward_system = Rewards(config)

        # Simulate reaching the goal location
        env_vars = {
            "X": 5,
            "Y": 5,
            "map_num": 1,
            "room": 0,
            "pokedex_seen": 0,
            "pokedex_owned": 0,
        }

        reward, done = reward_system.calculate_reward(env_vars, "A")

        # Should get a positive reward for achieving goal
        self.assertGreater(reward, 0)
        # Should complete the episode since break_on_goal defaults to True
        self.assertTrue(done)

    def test_reward_step_penalty_is_constant(self):
        """Step penalty is fixed across an episode (no time dependency)."""
        config = load_default_config()
        config["N_goals_target"] = 1
        config["episode_length"] = 100
        config["punish_steps"] = True
        config["location_goals"] = [[[999, 999, 999]]]  # Unreachable goal
        config["pokedex_goals"] = {}

        reward_system = Rewards(config)

        env_vars = {
            "X": 1,
            "Y": 1,
            "map_num": 1,
            "room": 0,
            "pokedex_seen": 0,
            "pokedex_owned": 0,
        }

        early_reward, _ = reward_system.calculate_reward(env_vars, "A")
        for _ in range(40):
            reward_system.calculate_reward(env_vars, "A")
        mid_reward, _ = reward_system.calculate_reward(env_vars, "A")

        self.assertEqual(mid_reward, early_reward)
        self.assertEqual(float(early_reward), float(reward_system.step_penalty))

    def test_reward_step_penalty_disabled(self):
        """punish_steps=False zeroes out the per-step penalty."""
        config = load_default_config()
        config["N_goals_target"] = 1
        config["episode_length"] = 100
        config["punish_steps"] = False
        config["location_goals"] = [[[999, 999, 999]]]
        config["pokedex_goals"] = {}

        reward_system = Rewards(config)
        self.assertEqual(reward_system.step_penalty, 0)

        env_vars = {
            "X": 1, "Y": 1, "map_num": 1, "room": 0,
            "pokedex_seen": 0, "pokedex_owned": 0,
        }
        reward, _ = reward_system.calculate_reward(env_vars, "A")
        self.assertEqual(float(reward), 0.0)


class TestRewardsBranches(unittest.TestCase):
    """Direct unit tests on Rewards class branches not exercised end-to-end."""

    def _base_config(self):
        config = load_default_config()
        config["episode_length"] = 100
        config["punish_steps"] = False  # isolate goal/bonus rewards
        config["pokedex_goals"] = {}
        return config

    def _env_vars(self, x=1, y=1, map_num=1, room=0, seen=0, owned=0):
        return {
            "X": x, "Y": y, "map_num": map_num, "room": room,
            "pokedex_seen": seen, "pokedex_owned": owned,
        }

    def test_sequential_goals_must_be_in_order(self):
        """Reaching goal 2 before goal 1 gives no reward."""
        config = self._base_config()
        config["N_goals_target"] = 2
        config["location_goals"] = [[[1, 1, 1]], [[2, 2, 1]]]
        config["require_sequential"] = True

        rewards = Rewards(config)

        # Land on goal 2 first — should not advance.
        reward, _ = rewards.calculate_reward(self._env_vars(x=2, y=2), "a")
        self.assertEqual(float(reward), 0.0)
        self.assertEqual(rewards.N_goals, 0)

        # Land on goal 1 — should advance.
        reward, _ = rewards.calculate_reward(self._env_vars(x=1, y=1), "a")
        self.assertGreater(float(reward), 0)
        self.assertEqual(rewards.N_goals, 1)

    def test_sequential_bonus_added(self):
        config = self._base_config()
        config["N_goals_target"] = 5  # avoid all-goals bonus on first hit
        config["location_goals"] = [
            [[1, 1, 1]], [[2, 2, 1]], [[3, 3, 1]], [[4, 4, 1]], [[5, 5, 1]],
        ]
        config["require_sequential"] = True
        config["checkpoint_goals"] = [99]  # avoid checkpoint bonus

        rewards = Rewards(config)
        reward, _ = rewards.calculate_reward(self._env_vars(x=1, y=1), "a")
        expected = rewards.goal_reward + rewards.sequence_bonus
        self.assertEqual(float(reward), float(expected))

    def test_checkpoint_bonus_triggers_at_milestone(self):
        config = self._base_config()
        config["N_goals_target"] = 5
        config["location_goals"] = [
            [[1, 1, 1]], [[2, 2, 1]], [[3, 3, 1]], [[4, 4, 1]], [[5, 5, 1]],
        ]
        config["require_sequential"] = False
        config["checkpoint_goals"] = [2]

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(x=1, y=1), "a")  # goal 1
        reward, _ = rewards.calculate_reward(self._env_vars(x=2, y=2), "a")  # goal 2

        # Second goal should include the checkpoint bonus.
        self.assertGreaterEqual(float(reward), float(rewards.checkpoint_bonus))

    def test_all_goals_bonus_and_break_on_goal(self):
        config = self._base_config()
        config["N_goals_target"] = 1
        config["location_goals"] = [[[1, 1, 1]]]
        config["break_on_goal"] = True
        config["checkpoint_goals"] = [99]

        rewards = Rewards(config)
        reward, done = rewards.calculate_reward(self._env_vars(x=1, y=1), "a")
        self.assertTrue(done)
        # Reward should include goal + sequence + all_goals bonus.
        self.assertGreaterEqual(
            float(reward),
            float(rewards.goal_reward + rewards.all_goals_bonus),
        )

    def test_button_penalty_for_start_select(self):
        config = self._base_config()
        config["N_goals_target"] = 1
        config["location_goals"] = [[[999, 999, 999]]]

        rewards = Rewards(config)
        r_start, _ = rewards.calculate_reward(self._env_vars(), "start")
        rewards2 = Rewards(config)
        r_select, _ = rewards2.calculate_reward(self._env_vars(), "select")
        rewards3 = Rewards(config)
        r_a, _ = rewards3.calculate_reward(self._env_vars(), "a")

        self.assertEqual(float(r_start), float(rewards.button_penalty))
        self.assertEqual(float(r_select), float(rewards2.button_penalty))
        self.assertEqual(float(r_a), 0.0)

    def test_exploration_reward_first_visit_only(self):
        config = self._base_config()
        config["N_goals_target"] = 1
        config["location_goals"] = [[[999, 999, 999]]]
        config["exploration_reward"] = 1.0

        rewards = Rewards(config)
        r1, _ = rewards.calculate_reward(self._env_vars(x=5, y=5), "a")
        r2, _ = rewards.calculate_reward(self._env_vars(x=5, y=5), "a")
        r3, _ = rewards.calculate_reward(self._env_vars(x=6, y=5), "a")

        self.assertEqual(float(r1), 1.0)
        self.assertEqual(float(r2), 0.0)
        self.assertEqual(float(r3), 1.0)

    def test_pokedex_seen_reward(self):
        config = self._base_config()
        config["N_goals_target"] = 99  # don't trigger all-goals
        config["location_goals"] = [[[999, 999, 999]]]

        rewards = Rewards(config)
        # Bump seen from 0 -> 3 in one step.
        r, _ = rewards.calculate_reward(self._env_vars(seen=3), "a")
        self.assertEqual(float(r), float(rewards.pokedex_seen_reward))

        # Subsequent step with same seen count — no extra pokedex reward.
        r2, _ = rewards.calculate_reward(self._env_vars(seen=3), "a")
        self.assertEqual(float(r2), 0.0)

    def test_pokedex_goal_threshold_increments_n_goals(self):
        config = self._base_config()
        config["N_goals_target"] = 99
        config["location_goals"] = [[[999, 999, 999]]]
        config["pokedex_goals"] = {"seen": 2}

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(seen=1), "a")
        self.assertEqual(rewards.N_goals, 0)
        rewards.calculate_reward(self._env_vars(seen=2), "a")
        # Crossing threshold removes the goal and increments N_goals.
        self.assertEqual(rewards.N_goals, 1)
        self.assertNotIn("seen", rewards.pokedex_goals)

    def test_reward_clipping(self):
        config = self._base_config()
        config["N_goals_target"] = 1
        config["location_goals"] = [[[1, 1, 1]]]
        config["goal_reward"] = 100000
        config["all_goals_bonus"] = 100000

        rewards = Rewards(config)
        r, _ = rewards.calculate_reward(self._env_vars(x=1, y=1), "a")
        self.assertEqual(float(r), float(rewards.clip))

    def test_set_goals_accepts_dict_format(self):
        config = self._base_config()
        config["N_goals_target"] = 1
        config["location_goals"] = [[{"x": 3, "y": 4, "map": 5}]]

        rewards = Rewards(config)
        # Dict format should have been normalised to [x, y, map].
        goal = list(rewards.location_goals.values())[0]
        self.assertEqual(goal, [[3, 4, 5]])

        # And the goal should be reachable as a location.
        r, _ = rewards.calculate_reward(self._env_vars(x=3, y=4, map_num=5), "a")
        self.assertGreater(float(r), 0)

    def test_set_goals_rejects_unknown_format(self):
        config = self._base_config()
        config["N_goals_target"] = 1
        config["location_goals"] = [[42]]
        with self.assertRaises(ValueError):
            Rewards(config)


if __name__ == "__main__":
    unittest.main()
