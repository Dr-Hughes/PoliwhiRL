# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
import os
from PoliwhiRL.agents.PPO.parallel_runner import PPOParallelRunner, run_single_agent
from PoliwhiRL.agents.PPO import PPOAgent
from PoliwhiRL.environment.gym_env import PyBoyEnvironment
from main import load_default_config


class TestPPOParallelRunner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = load_default_config()
        self.config["model"] = "PPO"
        self.config["erase"] = False
        self.config["checkpoint"] = f"{self.temp_dir}/checkpoint"
        self.config["results_dir"] = f"{self.temp_dir}/results"
        self.config["record_path"] = f"{self.temp_dir}/records"
        self.config["export_state_loc"] = f"{self.temp_dir}/states"
        self.config["device"] = "cpu"
        self.config["num_episodes"] = 2  # Very short for testing
        self.config["episode_length"] = 50  # Very short for testing
        self.config["ppo_num_agents"] = 2
        self.config["ppo_iterations"] = 1
        self.config["record"] = False
        self.config["report_episode"] = False
        self.config["N_goals_target"] = 1
        self.config["use_curriculum"] = False
        self.config["location_goals"] = [[[1, 1, 1]]]
        self.config["pokedex_goals"] = {}

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_parallel_runner_initialization(self):
        """Test that PPOParallelRunner initializes correctly"""
        runner = PPOParallelRunner(self.config)

        self.assertEqual(runner.num_agents, 2)
        self.assertEqual(runner.iterations, 1)
        self.assertEqual(runner.base_checkpoint, self.config["checkpoint"])
        self.assertIsNotNone(runner.hang_summary_file)

    def test_performance_score_calculation(self):
        """Test the composite performance scoring system"""
        runner = PPOParallelRunner(self.config)

        # Test with good performance metrics
        good_result = {
            "performance_metrics": {
                "avg_reward": 5.0,
                "avg_length": 25.0,  # Efficient (half of max episode length)
                "recent_rewards": [4.0, 5.0, 6.0, 5.0, 4.0],
            }
        }

        good_score = runner.calculate_composite_score(good_result)
        self.assertGreater(good_score, 0)

        # Test with poor performance metrics
        poor_result = {
            "performance_metrics": {
                "avg_reward": -2.0,
                "avg_length": 50.0,  # Inefficient (full episode length)
                "recent_rewards": [-3.0, -2.0, -1.0, -2.0, -3.0],
            }
        }

        poor_score = runner.calculate_composite_score(poor_result)
        self.assertLess(poor_score, good_score)

    def test_get_shared_model_state_without_checkpoint(self):
        """Test shared model state when no checkpoint exists"""
        runner = PPOParallelRunner(self.config)

        actor_critic_state, icm_state = runner.get_shared_model_state(0)

        # Should return None when no checkpoint exists
        self.assertIsNone(actor_critic_state)
        self.assertIsNone(icm_state)

    def test_get_shared_model_state_with_checkpoint(self):
        """Test shared model state loading from existing checkpoint"""
        # Create a dummy checkpoint first
        os.makedirs(self.config["checkpoint"], exist_ok=True)

        # Create a simple PPO agent and save it to create checkpoint files
        env = PyBoyEnvironment(self.config)
        try:
            state_shape = env.output_shape()
            num_actions = env.action_space.n
            agent = PPOAgent(state_shape, num_actions, self.config)
            agent.save_model(self.config["checkpoint"])
        finally:
            env.close()

        runner = PPOParallelRunner(self.config)

        actor_critic_state, icm_state = runner.get_shared_model_state(0)

        # Should successfully load states
        self.assertIsNotNone(actor_critic_state)
        self.assertIsNotNone(icm_state)
        self.assertIsInstance(actor_critic_state, dict)
        self.assertIsInstance(icm_state, dict)

    def test_single_agent_function_basic(self):
        """Test that run_single_agent function works in basic case"""
        # This test runs in the main process, so it should work
        agent_id = 0
        iteration = 0
        cumulative_episodes = 0

        result = run_single_agent(
            agent_id,
            self.config,
            None,
            None,
            iteration,
            self.config["checkpoint"],
            cumulative_episodes,
        )

        # Should return a valid result
        self.assertIsNotNone(result)
        self.assertIn("checkpoint_path", result)
        self.assertIn("episode", result)
        self.assertIn("performance_metrics", result)

        # Check performance metrics structure
        metrics = result["performance_metrics"]
        self.assertIn("recent_rewards", metrics)
        self.assertIn("recent_lengths", metrics)
        self.assertIn("avg_reward", metrics)
        self.assertIn("avg_length", metrics)

    def test_entropy_logging(self):
        """Test entropy status logging functionality"""
        # Create a dummy agent for testing entropy logging
        env = PyBoyEnvironment(self.config)
        try:
            state_shape = env.output_shape()
            num_actions = env.action_space.n
            agent = PPOAgent(state_shape, num_actions, self.config)
        finally:
            env.close()

        runner = PPOParallelRunner(self.config)

        # This should not raise an exception
        runner.log_entropy_status(agent)

        # Test entropy calculation
        current_entropy = agent.model._get_entropy_coef(0)
        self.assertIsInstance(current_entropy, float)
        self.assertGreater(current_entropy, 0)

    def test_model_averaging_functionality(self):
        """Test the model averaging functionality with dummy agents"""
        # Create multiple dummy checkpoints
        env = PyBoyEnvironment(self.config)
        try:
            state_shape = env.output_shape()
            num_actions = env.action_space.n
        finally:
            env.close()

        # Create dummy results with checkpoints
        results = []
        for i in range(2):
            checkpoint_path = f"{self.temp_dir}/agent_{i}"
            os.makedirs(checkpoint_path, exist_ok=True)

            # Create and save an agent
            agent = PPOAgent(state_shape, num_actions, self.config)
            agent.save_model(checkpoint_path)

            result = {
                "checkpoint_path": checkpoint_path,
                "episode": 2,
                "performance_metrics": {
                    "recent_rewards": [1.0, 2.0],
                    "recent_lengths": [25, 30],
                    "avg_reward": 1.5,
                    "avg_length": 27.5,
                },
            }
            results.append(result)

        runner = PPOParallelRunner(self.config)

        # Test model averaging
        averaged_agent = runner.average_models(results, state_shape, num_actions)

        self.assertIsNotNone(averaged_agent)
        self.assertIsInstance(averaged_agent, PPOAgent)
        self.assertEqual(averaged_agent.episode, runner.total_episodes_run)

    def test_smart_aggregation_strategies(self):
        """Test different aggregation strategies based on performance distribution"""
        env = PyBoyEnvironment(self.config)
        try:
            state_shape = env.output_shape()
            num_actions = env.action_space.n
        finally:
            env.close()

        # Create results with varying performance
        results = []
        performance_scores = [5.0, 3.0, 1.0]  # High variance

        for i, score in enumerate(performance_scores):
            checkpoint_path = f"{self.temp_dir}/agent_{i}"
            os.makedirs(checkpoint_path, exist_ok=True)

            agent = PPOAgent(state_shape, num_actions, self.config)
            agent.save_model(checkpoint_path)

            result = {
                "checkpoint_path": checkpoint_path,
                "episode": 2,
                "performance_metrics": {
                    "recent_rewards": [score],
                    "recent_lengths": [25],
                    "avg_reward": score,
                    "avg_length": 25.0,
                },
            }
            results.append(result)

        runner = PPOParallelRunner(self.config)

        # Test that smart aggregation completes without error
        runner.smart_model_aggregation(results, state_shape, num_actions)

        # Verify checkpoint was created
        self.assertTrue(os.path.exists(f"{self.config['checkpoint']}/actor_critic.pth"))


class TestPPOParallelRunnerAdvanced(unittest.TestCase):
    """Advanced tests that require more setup"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = load_default_config()
        self.config["model"] = "PPO"
        self.config["erase"] = False
        self.config["checkpoint"] = f"{self.temp_dir}/checkpoint"
        self.config["results_dir"] = f"{self.temp_dir}/results"
        self.config["device"] = "cpu"
        self.config["num_episodes"] = 1
        self.config["episode_length"] = 20
        self.config["ppo_num_agents"] = 2
        self.config["ppo_iterations"] = 1
        self.config["record"] = False
        self.config["report_episode"] = False
        self.config["N_goals_target"] = 1
        self.config["use_curriculum"] = False

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_performance_weighted_averaging(self):
        """Test performance-weighted model averaging"""
        env = PyBoyEnvironment(self.config)
        try:
            state_shape = env.output_shape()
            num_actions = env.action_space.n
        finally:
            env.close()

        # Create scored results (score, result) pairs
        scored_results = []
        for i, score in enumerate([3.0, 1.0]):  # Different performance scores
            checkpoint_path = f"{self.temp_dir}/agent_{i}"
            os.makedirs(checkpoint_path, exist_ok=True)

            agent = PPOAgent(state_shape, num_actions, self.config)
            agent.save_model(checkpoint_path)

            result = {
                "checkpoint_path": checkpoint_path,
                "episode": 1,
                "performance_metrics": {
                    "recent_rewards": [score],
                    "recent_lengths": [20],
                    "avg_reward": score,
                    "avg_length": 20.0,
                },
            }
            scored_results.append((score, result))

        runner = PPOParallelRunner(self.config)

        averaged_agent = runner.performance_weighted_average(
            scored_results, state_shape, num_actions
        )

        self.assertIsNotNone(averaged_agent)
        self.assertIsInstance(averaged_agent, PPOAgent)

    def test_top_k_averaging(self):
        """Test top-k model averaging strategy"""
        env = PyBoyEnvironment(self.config)
        try:
            state_shape = env.output_shape()
            num_actions = env.action_space.n
        finally:
            env.close()

        # Create scored results
        scored_results = []
        for i in range(3):
            checkpoint_path = f"{self.temp_dir}/agent_{i}"
            os.makedirs(checkpoint_path, exist_ok=True)

            agent = PPOAgent(state_shape, num_actions, self.config)
            agent.save_model(checkpoint_path)

            result = {
                "checkpoint_path": checkpoint_path,
                "episode": 1,
                "performance_metrics": {
                    "recent_rewards": [float(i + 1)],
                    "recent_lengths": [20],
                    "avg_reward": float(i + 1),
                    "avg_length": 20.0,
                },
            }
            scored_results.append((float(i + 1), result))

        runner = PPOParallelRunner(self.config)

        # Test top-2 averaging
        averaged_agent = runner.top_k_average(
            scored_results, state_shape, num_actions, k=2
        )

        self.assertIsNotNone(averaged_agent)
        self.assertIsInstance(averaged_agent, PPOAgent)


if __name__ == "__main__":
    unittest.main()
