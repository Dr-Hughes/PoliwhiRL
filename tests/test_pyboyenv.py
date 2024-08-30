# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
import numpy as np
from PoliwhiRL.environment.gym_env import PyBoyEnvironment
from main import load_default_config


class TestPyBoyEnvironment(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        self.config = load_default_config()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_environment_initialization(self):
        env = PyBoyEnvironment(self.config)
        self.assertIsNotNone(env)
        self.assertEqual(env.action_space.n, 9)  # 9 actions defined in the gym_env.py
        self.assertEqual(env.steps, 0)
        self.assertEqual(env.episode, 0)
        env.close()

    def test_bad_environment_initialization(self):
        self.config["rom_path"] = "non_existent_file"
        with self.assertRaises(FileNotFoundError):
            PyBoyEnvironment(self.config)

    def test_reset(self):
        env = PyBoyEnvironment(self.config)
        scale = self.config.get("scaling_factor", 1.0)
        observation = env.reset()
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (144 * scale, 160 * scale, 3))
        self.assertEqual(env.steps, 0)
        self.assertEqual(env.episode, 1)
        env.close()

    def test_step(self):
        env = PyBoyEnvironment(self.config)
        scale = self.config.get("scaling_factor", 1.0)
        env.reset()
        observation, reward, done, _ = env.step(0)  # Take a "no action" step
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (144 * scale, 160 * scale, 3))
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertEqual(env.steps, 1)
        env.close()

    def test_no_vision(self):
        self.config["vision"] = False
        env = PyBoyEnvironment(self.config)
        env.reset()
        observation, reward, done, _ = env.step(0)  # Take a "no action" step
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (18, 20))
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertEqual(env.steps, 1)
        env.close()

    def test_bw_vision(self):
        self.config["use_grayscale"] = True
        env = PyBoyEnvironment(self.config)
        self.config["scaling_factor"] = 1.0
        env.reset()
        observation, reward, done, _ = env.step(0)  # Take a "no action" step
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (144, 160, 1))
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertEqual(env.steps, 1)
        env.close()

    def test_scaling_vision(self):
        self.config["scaling_factor"] = 0.5
        env = PyBoyEnvironment(self.config)
        env.reset()
        observation, reward, done, _ = env.step(0)  # Take a "no action" step
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (72, 80, 3))
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertEqual(env.steps, 1)
        env.close()

    def test_bw_scaling_vision(self):
        self.config["scaling_factor"] = 0.5
        self.config["use_grayscale"] = True
        env = PyBoyEnvironment(self.config)
        env.reset()
        observation, reward, done, _ = env.step(0)  # Take a "no action" step
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (72, 80, 1))
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertEqual(env.steps, 1)
        env.close()

    def test_episode_length(self):
        env = PyBoyEnvironment(self.config)
        env.reset()
        for n in range(self.config["episode_length"]):
            _, _, done, _ = env.step(0)
        self.assertTrue(done)
        self.assertEqual(env.steps, self.config["episode_length"])
        env.close()

    


if __name__ == "__main__":
    unittest.main()
