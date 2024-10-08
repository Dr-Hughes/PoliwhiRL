# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
import numpy as np
from PoliwhiRL.reward_evaluator import evaluate_reward_system
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

    def test_positive_sum_1_goal(self):
        self.config["N_goals_target"] = 1
        rewards = evaluate_reward_system(self.config)
        self.assertGreater(np.sum(rewards), 0)

    def test_positive_sum_2_goals(self):
        self.config["N_goals_target"] = 2
        rewards = evaluate_reward_system(self.config)
        self.assertGreater(np.sum(rewards), 0)

    def test_positive_sum_3_goals(self):
        self.config["N_goals_target"] = 3
        rewards = evaluate_reward_system(self.config)
        self.assertGreater(np.sum(rewards), 0)

    def test_positive_sum_4_goals(self):
        self.config["N_goals_target"] = 4
        rewards = evaluate_reward_system(self.config)
        self.assertGreater(np.sum(rewards), 0)

    def test_positive_sum_5_goals(self):
        self.config["N_goals_target"] = 5
        rewards = evaluate_reward_system(self.config)
        self.assertGreater(np.sum(rewards), 0)

    def test_positive_sum_6_goals(self):
        self.config["N_goals_target"] = 6
        rewards = evaluate_reward_system(self.config)
        self.assertGreater(np.sum(rewards), 0)

    def test_positive_sum_7_goals(self):
        self.config["N_goals_target"] = 7
        rewards = evaluate_reward_system(self.config)
        self.assertGreater(np.sum(rewards), 0)


if __name__ == "__main__":
    unittest.main()
