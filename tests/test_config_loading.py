import unittest
import json
import os
from unittest.mock import patch, mock_open
from main import load_default_config, load_user_config, merge_configs

class TestConfigLoading(unittest.TestCase):

    def setUp(self):
        self.default_config = {
            "model": "DQN",
            "device": "cpu",
            "erase": False
        }
        self.user_config = {
            "model": "PPO",
            "learning_rate": 0.001
        }

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_default_config(self, mock_file, mock_exists):
        mock_exists.return_value = True
        mock_file.return_value.__enter__().read.return_value = json.dumps(self.default_config)
        
        result = load_default_config()
        
        self.assertEqual(result, self.default_config)
        mock_exists.assert_called_once_with("./configs/default_config.json")
        mock_file.assert_called_once_with("./configs/default_config.json", "r")

    @patch("os.path.exists")
    def test_load_default_config_file_not_exists(self, mock_exists):
        mock_exists.return_value = False
        
        result = load_default_config()
        
        self.assertEqual(result, {})
        mock_exists.assert_called_once_with("./configs/default_config.json")

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_user_config(self, mock_file, mock_exists):
        mock_exists.return_value = True
        mock_file.return_value.__enter__().read.return_value = json.dumps(self.user_config)
        
        result = load_user_config("user_config.json")
        
        self.assertEqual(result, self.user_config)
        mock_exists.assert_called_once_with("user_config.json")
        mock_file.assert_called_once_with("user_config.json", "r")

    @patch("os.path.exists")
    def test_load_user_config_file_not_exists(self, mock_exists):
        mock_exists.return_value = False
        
        result = load_user_config("non_existent_config.json")
        
        self.assertEqual(result, {})
        mock_exists.assert_called_once_with("non_existent_config.json")

    def test_merge_configs(self):
        merged_config = merge_configs(self.default_config, self.user_config)
        
        expected_config = {
            "model": "PPO",
            "device": "cpu",
            "erase": False,
            "learning_rate": 0.001
        }
        
        self.assertEqual(merged_config, expected_config)

if __name__ == "__main__":
    unittest.main()