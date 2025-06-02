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
        self.assertGreater(total_reward, -100)  # With -0.2 penalty per step over 278 steps

    def test_reward_system_functionality_4_goals(self):
        """Test that reward system works correctly with 4 goals"""
        self.config["N_goals_target"] = 4
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        total_reward = np.sum(rewards)
        self.assertGreater(total_reward, -100)  # With -0.2 penalty per step

    def test_reward_system_functionality_5_goals(self):
        """Test that reward system works correctly with 5 goals"""
        self.config["N_goals_target"] = 5
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        total_reward = np.sum(rewards)
        self.assertGreater(total_reward, -100)  # With -0.2 penalty per step

    def test_reward_system_functionality_6_goals(self):
        """Test that reward system works correctly with 6 goals"""
        self.config["N_goals_target"] = 6
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        total_reward = np.sum(rewards)
        self.assertGreater(total_reward, -100)  # With -0.2 penalty per step

    def test_reward_system_functionality_7_goals(self):
        """Test that reward system works correctly with 7 goals"""
        self.config["N_goals_target"] = 7
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        total_reward = np.sum(rewards)
        self.assertGreater(total_reward, -100)  # With -0.2 penalty per step

    def test_reward_scaling_and_clipping(self):
        """Test that rewards are properly scaled and clipped"""
        config = load_default_config()
        config["N_goals_target"] = 1
        config["episode_length"] = 100
        config["location_goals"] = [[[1, 1, 1]]]
        config["pokedex_goals"] = {}

        reward_system = Rewards(config)

        # Test clipping works
        self.assertEqual(reward_system.clip, 50.0)

        # Test penalty values are reasonable
        self.assertLessEqual(reward_system.step_penalty, 0)
        self.assertGreaterEqual(reward_system.step_penalty, -1.0)

        # Test goal rewards are positive
        self.assertGreater(reward_system.goal_reward_max, 0)
        self.assertGreater(reward_system.goal_reward_min, 0)

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

    def test_reward_step_penalty_progression(self):
        """Test that step penalties increase over time"""
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

        # Take several steps and check penalty increases
        early_reward, _ = reward_system.calculate_reward(env_vars, "A")

        # Advance to middle of episode
        for _ in range(40):
            reward_system.calculate_reward(env_vars, "A")

        mid_reward, _ = reward_system.calculate_reward(env_vars, "A")

        # Mid-episode penalty should be more negative than early penalty
        self.assertLess(mid_reward, early_reward)


if __name__ == "__main__":
    unittest.main()
