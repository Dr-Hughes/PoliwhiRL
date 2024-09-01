# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
from PoliwhiRL.reward_evaluator import evaluate_reward_system
from main import load_default_config, load_user_config, merge_configs


class TestRewardSystem(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_loc = "./configs/evaluate_reward_system.json"
        self.config = load_default_config()
        self.user_config = load_user_config(self.config_loc)
        self.config = merge_configs(self.config, self.user_config)
        self.config['output_path'] = self.temp_dir

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_positive_sum(self):
        rewards = evaluate_reward_system(self.config)
        self.assertGreater(sum(rewards), 0)



if __name__ == "__main__":
    unittest.main()
