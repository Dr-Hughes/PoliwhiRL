# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
import os
import numpy as np
from collections import deque
from PoliwhiRL.agents.PPO import PPOAgent
from PoliwhiRL.environment.gym_env import PyBoyEnvironment
from main import load_default_config


class TestActionDiversityLogging(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = load_default_config()
        self.config["model"] = "PPO"
        self.config["erase"] = False
        self.config["checkpoint"] = f"{self.temp_dir}/checkpoint"
        self.config["results_dir"] = f"{self.temp_dir}/results"
        self.config["device"] = "cpu"
        self.config["episode_length"] = 100
        self.config["record"] = False
        self.config["report_episode"] = False
        self.config["N_goals_target"] = 1
        self.config["location_goals"] = [[[1, 1, 1]]]
        self.config["pokedex_goals"] = {}
        self.config["tqdm_worker_id"] = "test"

        # Create environment and agent
        self.env = PyBoyEnvironment(self.config)
        self.state_shape = self.env.output_shape()
        self.action_size = self.env.action_space.n
        self.agent = PPOAgent(self.state_shape, self.action_size, self.config)

    def tearDown(self):
        self.env.close()
        shutil.rmtree(self.temp_dir)

    def test_action_diversity_logging_file_creation(self):
        """Test that action diversity logging creates the expected log files"""
        # Simulate some button presses
        for i in range(30):  # More than 25 to trigger logging
            self.agent.episode_data["buttons_pressed"].append(i % self.action_size)

        # Set up agent state for logging
        self.agent.episode = 5
        self.agent.steps = 25

        # Create dummy state sequence and exploration tensor
        state_seq_arr = np.random.rand(self.agent.sequence_length, *self.state_shape)
        exploration_tensor = np.random.rand(10, 84, 84)  # Dummy exploration memory

        # Call the logging function
        self.agent._log_action_diversity_to_file(0, state_seq_arr, exploration_tensor)

        # Check that log file was created
        expected_log_dir = f"{self.config['results_dir']}/action_logs"
        expected_log_file = f"{expected_log_dir}/agent_test_actions.log"

        self.assertTrue(os.path.exists(expected_log_dir))
        self.assertTrue(os.path.exists(expected_log_file))

        # Check that log file contains content
        with open(expected_log_file, "r") as f:
            content = f.read()
            self.assertIn("Episode:", content)
            self.assertIn("Step:", content)
            self.assertIn("Action:", content)
            self.assertIn("ActionEntropy:", content)
            self.assertIn("ProbEntropy:", content)

    def test_action_diversity_entropy_calculation(self):
        """Test entropy calculations in action diversity logging"""
        # Set up agent with known action pattern
        action_pattern = [0, 0, 0, 1, 1, 2] * 5  # Some repetition
        for action in action_pattern:
            self.agent.episode_data["buttons_pressed"].append(action)

        self.agent.episode = 1
        self.agent.steps = 25

        state_seq_arr = np.random.rand(self.agent.sequence_length, *self.state_shape)
        exploration_tensor = np.random.rand(10, 84, 84)

        # Call logging function - should not raise exceptions
        try:
            self.agent._log_action_diversity_to_file(
                0, state_seq_arr, exploration_tensor
            )
            logging_successful = True
        except Exception as e:
            logging_successful = False
            print(f"Logging failed: {e}")

        self.assertTrue(logging_successful)

    def test_action_diversity_warning_flags(self):
        """Test that warning flags are correctly identified"""
        # Create a scenario that should trigger warnings
        # Fill with mostly the same action (should trigger ACTION_COLLAPSE)
        action_pattern = [0] * 25  # All same action
        for action in action_pattern:
            self.agent.episode_data["buttons_pressed"].append(action)

        self.agent.episode = 1
        self.agent.steps = 25

        state_seq_arr = np.random.rand(self.agent.sequence_length, *self.state_shape)
        exploration_tensor = np.random.rand(10, 84, 84)

        self.agent._log_action_diversity_to_file(0, state_seq_arr, exploration_tensor)

        # Read the log file and check for warnings
        log_file = f"{self.config['results_dir']}/action_logs/agent_test_actions.log"
        with open(log_file, "r") as f:
            content = f.read()
            # Should contain warning flags for low entropy or action collapse
            self.assertIn("WARNINGS:", content)

    def test_action_diversity_button_name_mapping(self):
        """Test that actions are correctly mapped to button names"""
        # Test each button
        button_names = ["A", "B", "Right", "Left", "Up", "Down", "Start", "Select"]

        for action_id in range(min(len(button_names), self.action_size)):
            # Clear previous button presses
            self.agent.episode_data["buttons_pressed"] = deque(maxlen=100)
            for _ in range(30):
                self.agent.episode_data["buttons_pressed"].append(action_id)

            self.agent.episode = 1
            self.agent.steps = 25

            state_seq_arr = np.random.rand(
                self.agent.sequence_length, *self.state_shape
            )
            exploration_tensor = np.random.rand(10, 84, 84)

            self.agent._log_action_diversity_to_file(
                action_id, state_seq_arr, exploration_tensor
            )

            # Check that the correct button name appears in the log
            log_file = (
                f"{self.config['results_dir']}/action_logs/agent_test_actions.log"
            )
            with open(log_file, "r") as f:
                content = f.read()
                self.assertIn(f"Action: {button_names[action_id]}", content)

    def test_action_diversity_no_exploration_tensor(self):
        """Test logging when exploration tensor is None"""
        # Fill with some actions
        for i in range(30):
            self.agent.episode_data["buttons_pressed"].append(i % 4)

        self.agent.episode = 1
        self.agent.steps = 25

        state_seq_arr = np.random.rand(self.agent.sequence_length, *self.state_shape)

        # Call with None exploration tensor
        try:
            self.agent._log_action_diversity_to_file(0, state_seq_arr, None)
            logging_successful = True
        except Exception as e:
            logging_successful = False
            print(f"Logging with None exploration tensor failed: {e}")

        self.assertTrue(logging_successful)

    def test_training_progress_logging(self):
        """Test the training progress logging for hang detection"""
        # Test that progress logging works
        try:
            self.agent._log_training_progress("Test message for hang detection")
            logging_successful = True
        except Exception as e:
            logging_successful = False
            print(f"Progress logging failed: {e}")

        self.assertTrue(logging_successful)

        # Check that hang detection log file was created
        hang_log_dir = f"{self.config['results_dir']}/hang_logs"
        hang_log_file = f"{hang_log_dir}/agent_test_hang_detection.log"

        self.assertTrue(os.path.exists(hang_log_dir))
        self.assertTrue(os.path.exists(hang_log_file))

        # Check log content
        with open(hang_log_file, "r") as f:
            content = f.read()
            self.assertIn("Agent test - PROGRESS:", content)
            self.assertIn("Test message for hang detection", content)

    def test_entropy_coefficient_tracking(self):
        """Test that entropy coefficient is correctly tracked in logs"""
        # Set up minimal action history
        for i in range(30):
            self.agent.episode_data["buttons_pressed"].append(i % 3)

        # Test at different episode numbers to see entropy decay
        episodes_to_test = [1, 50, 100]

        for episode in episodes_to_test:
            self.agent.episode = episode
            self.agent.steps = 25

            state_seq_arr = np.random.rand(
                self.agent.sequence_length, *self.state_shape
            )
            exploration_tensor = np.random.rand(10, 84, 84)

            # Clear previous log entries for this test
            log_file = (
                f"{self.config['results_dir']}/action_logs/agent_test_actions.log"
            )
            if os.path.exists(log_file):
                os.remove(log_file)

            self.agent._log_action_diversity_to_file(
                0, state_seq_arr, exploration_tensor
            )

            # Check that entropy coefficient is logged
            with open(log_file, "r") as f:
                content = f.read()
                self.assertIn("EntropyCoef:", content)
                # Extract the entropy coefficient value
                lines = content.strip().split("\n")
                if lines:
                    last_line = lines[-1]
                    self.assertIn(f"Episode: {episode:4d}", last_line)

    def test_action_probabilities_logging(self):
        """Test that action probability distributions are logged correctly"""
        # Set up action history
        for i in range(30):
            self.agent.episode_data["buttons_pressed"].append(i % 4)

        self.agent.episode = 1
        self.agent.steps = 25

        state_seq_arr = np.random.rand(self.agent.sequence_length, *self.state_shape)
        exploration_tensor = np.random.rand(10, 84, 84)

        self.agent._log_action_diversity_to_file(0, state_seq_arr, exploration_tensor)

        # Check that action probabilities are in the log
        log_file = f"{self.config['results_dir']}/action_logs/agent_test_actions.log"
        with open(log_file, "r") as f:
            content = f.read()
            self.assertIn("ActionProbs:", content)
            # Should contain probability values in brackets
            self.assertIn("[", content)
            self.assertIn("]", content)

    def test_recent_actions_context_logging(self):
        """Test that recent actions context is properly logged"""
        # Create a specific pattern of recent actions
        recent_pattern = [0, 1, 2, 0, 1, 2, 3, 3, 3, 3]  # Last few actions
        for action in [0, 1] * 10 + recent_pattern:  # Fill history + recent pattern
            self.agent.episode_data["buttons_pressed"].append(action)

        self.agent.episode = 1
        self.agent.steps = 25

        state_seq_arr = np.random.rand(self.agent.sequence_length, *self.state_shape)
        exploration_tensor = np.random.rand(10, 84, 84)

        self.agent._log_action_diversity_to_file(0, state_seq_arr, exploration_tensor)

        # Check that recent actions are logged
        log_file = f"{self.config['results_dir']}/action_logs/agent_test_actions.log"
        with open(log_file, "r") as f:
            content = f.read()
            self.assertIn("RecentActions:", content)
            # Should show the recent pattern
            for action in recent_pattern:
                self.assertIn(str(action), content)


if __name__ == "__main__":
    unittest.main()
