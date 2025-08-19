# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
import os
import numpy as np
from PoliwhiRL.replay.enhanced_exploration_memory import EnhancedExplorationMemory
from PoliwhiRL.replay.exploration_memory import ExplorationMemory
from PoliwhiRL.agents.PPO import PPOAgent
from PoliwhiRL.environment.gym_env import PyBoyEnvironment
from main import load_default_config


class TestEnhancedExplorationMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.action_space_size = 8
        self.history_length = 5
        self.memory = EnhancedExplorationMemory(
            max_size=50,
            history_length=self.history_length,
            use_memory=True,
            action_space_size=self.action_space_size,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_enhanced_memory_initialization(self):
        """Test that enhanced exploration memory initializes correctly"""
        self.assertEqual(self.memory.max_size, 50)
        self.assertEqual(self.memory.history_length, self.history_length)
        self.assertEqual(self.memory.action_space_size, self.action_space_size)
        self.assertTrue(self.memory.use_memory)
        self.assertIsNotNone(self.memory.hash_visits)
        self.assertIsNotNone(self.memory.state_transitions)

    def test_add_transition_functionality(self):
        """Test adding transitions to enhanced exploration memory"""
        state = np.random.rand(84, 84, 3).astype(np.float32)
        next_state = np.random.rand(84, 84, 3).astype(np.float32)
        action = 2
        reward = 1.5
        coordinates = (10, 15, 1)  # x, y, map

        # Add transition
        self.memory.add_transition(state, action, next_state, reward, coordinates)

        # Check that memory was updated
        self.assertGreater(len(self.memory.hash_visits), 0)
        self.assertGreater(self.memory.total_transitions, 0)

    def test_exploration_bonus_calculation(self):
        """Test exploration bonus calculation"""
        # Add some transitions first
        for i in range(10):
            state = np.random.rand(84, 84, 3).astype(np.float32)
            next_state = np.random.rand(84, 84, 3).astype(np.float32)
            action = i % self.action_space_size
            reward = 1.0
            coordinates = (i, i, 1)

            self.memory.add_transition(state, action, next_state, reward, coordinates)

        # Test exploration bonus for a new state
        new_state = np.random.rand(84, 84, 3).astype(np.float32)
        state_hash = self.memory._compute_hash(new_state)
        bonus = self.memory.get_exploration_bonus(state_hash)

        self.assertIsInstance(bonus, float)
        self.assertGreaterEqual(bonus, 0.0)

    def test_hash_computation(self):
        """Test state hash computation"""
        state1 = np.random.rand(84, 84, 3).astype(np.float32)
        state2 = np.random.rand(84, 84, 3).astype(np.float32)

        hash1 = self.memory._compute_hash(state1)
        hash2 = self.memory._compute_hash(state2)

        self.assertIsInstance(hash1, str)
        self.assertIsInstance(hash2, str)
        # Different states should have different hashes (very likely)
        self.assertNotEqual(hash1, hash2)

        # Same state should have same hash
        hash1_repeat = self.memory._compute_hash(state1)
        self.assertEqual(hash1, hash1_repeat)

    def test_memory_tensor_generation(self):
        """Test memory tensor generation for model input"""
        # Add some screens first
        for i in range(10):
            screen = np.random.rand(84, 84, 3).astype(np.float32)
            self.memory.add_screen(screen)

        memory_tensor = self.memory.get_memory_tensor()

        if memory_tensor is not None:
            self.assertIsInstance(memory_tensor, np.ndarray)
            # Should have correct shape for memory: (max_size, 1 + history_length)
            expected_shape = (50, 1 + self.history_length)
            self.assertEqual(memory_tensor.shape, expected_shape)

    def test_save_and_load_functionality(self):
        """Test saving and loading enhanced exploration memory"""
        # Add some data
        for i in range(5):
            state = np.random.rand(84, 84, 3).astype(np.float32)
            next_state = np.random.rand(84, 84, 3).astype(np.float32)
            action = i % self.action_space_size
            reward = float(i)
            coordinates = (i, i + 1, 1)

            self.memory.add_transition(state, action, next_state, reward, coordinates)

        # Save to file
        save_path = f"{self.temp_dir}/test_memory.pkl"
        self.memory.save_to_file(save_path)

        self.assertTrue(os.path.exists(save_path))

        # Create new memory and load
        new_memory = EnhancedExplorationMemory(
            max_size=50,
            history_length=self.history_length,
            use_memory=True,
            action_space_size=self.action_space_size,
        )

        new_memory.load_from_file(save_path)

        # Check that data was loaded
        self.assertEqual(len(new_memory.hash_visits), len(self.memory.hash_visits))

    def test_reset_functionality(self):
        """Test memory reset functionality"""
        # Add some data
        for i in range(5):
            screen = np.random.rand(84, 84, 3).astype(np.float32)
            self.memory.add_screen(screen)

        # Ensure there's some data
        self.assertGreater(len(self.memory.memory), 0)

        # Reset memory
        self.memory.reset()

        # Check that episode-specific memory was cleared
        self.assertEqual(len(self.memory.memory), 0)
        self.assertEqual(len(self.memory.recent_hashes), 0)

    def test_exploration_memory_comparison(self):
        """Test that enhanced memory provides different behavior than standard memory"""
        # Create standard exploration memory for comparison
        standard_memory = ExplorationMemory(
            max_size=50, history_length=self.history_length, use_memory=True
        )

        # Both should have basic functionality
        screen = np.random.rand(84, 84, 3).astype(np.float32)

        self.memory.add_screen(screen)
        standard_memory.add_screen(screen)

        # Enhanced memory should have additional methods
        self.assertTrue(hasattr(self.memory, "add_transition"))
        self.assertTrue(hasattr(self.memory, "get_exploration_bonus"))
        self.assertFalse(hasattr(standard_memory, "add_transition"))
        self.assertFalse(hasattr(standard_memory, "get_exploration_bonus"))


class TestHangDetectionFeatures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = load_default_config()
        self.config["model"] = "PPO"
        self.config["results_dir"] = f"{self.temp_dir}/results"
        self.config["device"] = "cpu"
        self.config["tqdm_worker_id"] = "hang_test"
        self.config["episode_length"] = 50
        self.config["record"] = False
        self.config["report_episode"] = False

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_hang_detection_log_creation(self):
        """Test that hang detection logs are created properly"""
        # Create environment and agent for testing
        env = PyBoyEnvironment(self.config)
        try:
            state_shape = env.output_shape()
            action_size = env.action_space.n
            agent = PPOAgent(state_shape, action_size, self.config)

            # Test progress logging
            agent._log_training_progress("Test hang detection message")

            # Check that hang detection log was created
            expected_log_dir = f"{self.config['results_dir']}/hang_logs"
            expected_log_file = f"{expected_log_dir}/agent_hang_test_hang_detection.log"

            self.assertTrue(os.path.exists(expected_log_dir))
            self.assertTrue(os.path.exists(expected_log_file))

            # Check log content
            with open(expected_log_file, "r") as f:
                content = f.read()
                self.assertIn("Agent hang_test - PROGRESS:", content)
                self.assertIn("Test hang detection message", content)
                # Should have timestamp
                self.assertRegex(content, r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]")

        finally:
            env.close()

    def test_hang_detection_error_handling(self):
        """Test that hang detection logging handles errors gracefully"""
        # Create environment and agent
        env = PyBoyEnvironment(self.config)
        try:
            state_shape = env.output_shape()
            action_size = env.action_space.n
            agent = PPOAgent(state_shape, action_size, self.config)

            # Test with various message types
            test_messages = [
                "Normal message",
                "",  # Empty message
                "Message with special chars: !@#$%^&*()",
                "Very long message " + "x" * 1000,
            ]

            for message in test_messages:
                try:
                    agent._log_training_progress(message)
                    success = True
                except Exception as e:
                    success = False
                    print(f"Failed with message '{message[:50]}...': {e}")

                self.assertTrue(success, f"Logging failed for message: {message[:50]}")

        finally:
            env.close()

    def test_multiple_hang_log_entries(self):
        """Test that multiple hang detection entries accumulate correctly"""
        env = PyBoyEnvironment(self.config)
        try:
            state_shape = env.output_shape()
            action_size = env.action_space.n
            agent = PPOAgent(state_shape, action_size, self.config)

            # Log multiple messages
            messages = [
                "Starting training",
                "Episode 1 complete",
                "Episode 2 complete",
                "Training finished",
            ]

            for message in messages:
                agent._log_training_progress(message)

            # Check that all messages are in the log
            log_file = f"{self.config['results_dir']}/hang_logs/agent_hang_test_hang_detection.log"
            with open(log_file, "r") as f:
                content = f.read()

                for message in messages:
                    self.assertIn(message, content)

                # Should have multiple lines
                lines = content.strip().split("\n")
                self.assertEqual(len(lines), len(messages))

        finally:
            env.close()

    def test_hang_detection_with_different_worker_ids(self):
        """Test hang detection with different worker IDs"""
        worker_ids = ["agent_0", "agent_1", "main"]

        for worker_id in worker_ids:
            config = self.config.copy()
            config["tqdm_worker_id"] = worker_id

            env = PyBoyEnvironment(config)
            try:
                state_shape = env.output_shape()
                action_size = env.action_space.n
                agent = PPOAgent(state_shape, action_size, config)

                agent._log_training_progress(f"Test message from {worker_id}")

                # Check that correct log file was created
                expected_log_file = f"{config['results_dir']}/hang_logs/agent_{worker_id}_hang_detection.log"
                self.assertTrue(os.path.exists(expected_log_file))

                # Check content
                with open(expected_log_file, "r") as f:
                    content = f.read()
                    self.assertIn(f"Agent {worker_id} - PROGRESS:", content)
                    self.assertIn(f"Test message from {worker_id}", content)

            finally:
                env.close()


class TestModelIntegrationFeatures(unittest.TestCase):
    """Test integration of enhanced features with PPO model"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = load_default_config()
        self.config["model"] = "PPO"
        self.config["results_dir"] = f"{self.temp_dir}/results"
        self.config["device"] = "cpu"
        self.config["episode_length"] = 50
        self.config["record"] = False
        self.config["report_episode"] = False
        self.config["use_enhanced_exploration_memory"] = True
        self.config["N_goals_target"] = 1
        self.config["location_goals"] = [[[1, 1, 1]]]
        self.config["pokedex_goals"] = {}

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_enhanced_memory_integration(self):
        """Test that enhanced exploration memory integrates correctly with PPO agent"""
        env = PyBoyEnvironment(self.config)
        try:
            state_shape = env.output_shape()
            action_size = env.action_space.n
            agent = PPOAgent(state_shape, action_size, self.config)

            # Check that enhanced memory was created
            self.assertIsInstance(agent.exploration_memory, EnhancedExplorationMemory)
            self.assertTrue(hasattr(agent.exploration_memory, "add_transition"))
            self.assertTrue(hasattr(agent.exploration_memory, "get_exploration_bonus"))

        finally:
            env.close()

    def test_standard_memory_fallback(self):
        """Test that standard memory is used when enhanced is disabled"""
        config = self.config.copy()
        config["use_enhanced_exploration_memory"] = False

        env = PyBoyEnvironment(config)
        try:
            state_shape = env.output_shape()
            action_size = env.action_space.n
            agent = PPOAgent(state_shape, action_size, config)

            # Check that standard memory was created
            self.assertIsInstance(agent.exploration_memory, ExplorationMemory)
            self.assertFalse(hasattr(agent.exploration_memory, "add_transition"))

        finally:
            env.close()

    def test_exploration_memory_persistence(self):
        """Test that exploration memory can be saved and loaded with agent"""
        env = PyBoyEnvironment(self.config)
        try:
            state_shape = env.output_shape()
            action_size = env.action_space.n
            agent = PPOAgent(state_shape, action_size, self.config)

            # Add some data to exploration memory
            for i in range(5):
                screen = np.random.rand(84, 84, 3).astype(np.float32)
                agent.exploration_memory.add_screen(screen)

                if hasattr(agent.exploration_memory, "add_transition"):
                    # Enhanced memory
                    state = screen
                    next_state = np.random.rand(84, 84, 3).astype(np.float32)
                    action = i % action_size
                    reward = 1.0
                    coordinates = (i, i, 1)
                    agent.exploration_memory.add_transition(
                        state, action, next_state, reward, coordinates
                    )

            # Save agent (should include exploration memory)
            checkpoint_path = f"{self.temp_dir}/test_checkpoint"
            agent.save_model(checkpoint_path)

            # Create new agent and load
            new_agent = PPOAgent(state_shape, action_size, self.config)
            new_agent.load_model(checkpoint_path)

            # Check that exploration memory was loaded (basic test)
            self.assertIsInstance(
                new_agent.exploration_memory, type(agent.exploration_memory)
            )

        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
