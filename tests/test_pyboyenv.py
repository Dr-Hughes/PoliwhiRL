# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
import numpy as np
from PoliwhiRL.environment.gym_env import PyBoyEnvironment, RAM_OBS_DIM
from main import load_default_config


def _check_obs(test, obs, expected_image_shape):
    """Shared assertion: dict observation with the expected image shape and
    a 1-D RAM vector of the canonical width."""
    test.assertIsInstance(obs, dict)
    test.assertIn("image", obs)
    test.assertIn("ram", obs)
    test.assertEqual(obs["image"].shape, expected_image_shape)
    test.assertEqual(obs["ram"].shape, (RAM_OBS_DIM,))
    test.assertEqual(obs["ram"].dtype, np.float32)


class TestPyBoyEnvironment(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = load_default_config()
        self.config["episode_length"] = 25
        self.config["erase"] = False  # just in case

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_environment_initialization(self):
        env = PyBoyEnvironment(self.config)
        self.assertIsNotNone(env)
        self.assertEqual(env.action_space.n, 9)  # 9 actions defined in the gym_env.py
        self.assertEqual(env.steps, 0)
        self.assertEqual(env.episode, -1)
        self.assertEqual(env.ram_observation_shape(), (RAM_OBS_DIM,))
        env.close()

    def test_bad_environment_initialization(self):
        self.config["rom_path"] = "non_existent_file"
        with self.assertRaises(FileNotFoundError):
            PyBoyEnvironment(self.config)

    def test_reset(self):
        env = PyBoyEnvironment(self.config)
        scale = self.config["scaling_factor"]
        observation = env.reset()
        env.reset()
        _check_obs(self, observation, (3, 144 * scale, 160 * scale))
        self.assertEqual(env.steps, 0)
        self.assertEqual(env.episode, 1)
        env.close()

    def test_step(self):
        env = PyBoyEnvironment(self.config)
        scale = self.config["scaling_factor"]
        self.config["use_grayscale"] = False
        env.reset()
        observation, reward, done, _ = env.step(0)  # Take a "no action" step
        _check_obs(self, observation, (3, 144 * scale, 160 * scale))
        self.assertIsInstance(done, bool)
        self.assertEqual(env.steps, 1)
        env.close()

    def test_no_vision(self):
        self.config["vision"] = False
        env = PyBoyEnvironment(self.config)
        env.reset()
        observation, reward, done, _ = env.step(0)  # Take a "no action" step
        _check_obs(self, observation, (18, 20))
        self.assertIsInstance(done, bool)
        self.assertEqual(env.steps, 1)
        env.close()

    def test_bw_vision(self):
        self.config["use_grayscale"] = True
        env = PyBoyEnvironment(self.config)
        self.config["scaling_factor"] = 1.0
        env.reset()
        observation, reward, done, _ = env.step(0)  # Take a "no action" step
        _check_obs(self, observation, (1, 144, 160))
        self.assertIsInstance(done, bool)
        self.assertEqual(env.steps, 1)
        env.close()

    def test_scaling_vision(self):
        self.config["scaling_factor"] = 0.5
        self.config["use_grayscale"] = False
        env = PyBoyEnvironment(self.config)
        env.reset()
        observation, reward, done, _ = env.step(0)  # Take a "no action" step
        _check_obs(self, observation, (3, 72, 80))
        self.assertIsInstance(done, bool)
        self.assertEqual(env.steps, 1)
        env.close()

    def test_bw_scaling_vision(self):
        self.config["scaling_factor"] = 0.5
        self.config["use_grayscale"] = True
        env = PyBoyEnvironment(self.config)
        env.reset()
        observation, reward, done, _ = env.step(0)  # Take a "no action" step
        _check_obs(self, observation, (1, 72, 80))
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

    def test_save_load_gym_state(self):
        env = PyBoyEnvironment(self.config)
        save_loc = self.temp_dir + "/test.pkl"
        env.reset()
        for n in range(self.config["episode_length"]):
            _, _, _, _ = env.step(0)
        env.save_gym_state(save_loc)

        env_new = PyBoyEnvironment(self.config)
        env_new.load_gym_state(save_loc)

        self.assertEqual(env.steps, self.config["episode_length"])
        self.assertEqual(env_new.steps, self.config["episode_length"])

        env.close()
        env_new.close()


if __name__ == "__main__":
    unittest.main()
