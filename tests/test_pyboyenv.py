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
        # Channel-first (1, 18, 20), matching get_screen_image()'s (C, H, W)
        # convention — a bare (18, 20) 2-tuple (the pre-fix shape) gets
        # misread as 18 channels of a 20-wide, channel-less image and
        # raises a Conv2d shape error at model construction (see
        # test_game_area_input_shape_builds_a_valid_cnn below).
        self.config["vision"] = False
        env = PyBoyEnvironment(self.config)
        env.reset()
        observation, reward, done, _ = env.step(0)  # Take a "no action" step
        _check_obs(self, observation, (1, 18, 20))
        self.assertEqual(observation["image"].dtype, np.uint8)
        self.assertEqual(env.output_shape(), (1, 18, 20))
        self.assertIsInstance(done, bool)
        self.assertEqual(env.steps, 1)
        env.close()

    def test_game_area_input_shape_builds_a_valid_cnn(self):
        """Regression guard for the exact crash this shape fix resolves:
        GameBoyCNN(input_shape, ...) does `torch.zeros(1, *input_shape)`
        then a Conv2d(in_channels=input_shape[0]); with a channel-less
        (18, 20) shape that Conv2d saw 1 channel where it expected 18 and
        raised at construction time, before training ever started."""
        from PoliwhiRL.models.PPO.PPOTransformer import GameBoyCNN
        import torch

        self.config["vision"] = False
        env = PyBoyEnvironment(self.config)
        try:
            shape = env.output_shape()
            obs = env.reset()
        finally:
            env.close()
        cnn = GameBoyCNN(shape, 128)
        out = cnn(torch.from_numpy(obs["image"]).float().unsqueeze(0))
        self.assertEqual(out.shape, (1, 128))

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

    def test_enable_record_log_format_writes_csv_not_pngs(self):
        """Periodic full-episode recordings use the lightweight CSV log
        (see open_trajectory_log) — no per-step PNGs, one trajectory.csv
        with a row per valid step."""
        import glob
        import os

        env = PyBoyEnvironment(self.config)
        env.reset()
        env.enable_record(
            "N_goals_0/ep_0", use_episode_number=False, record_format="log"
        )
        for _ in range(5):
            env.step(0)
        env.close()

        ep_dir = os.path.join(self.config["record_path"], "N_goals_0/ep_0")
        csv_path = os.path.join(ep_dir, "trajectory.csv")
        self.assertTrue(os.path.exists(csv_path))
        self.assertEqual(glob.glob(os.path.join(ep_dir, "*.png")), [])

        with open(csv_path) as f:
            lines = f.readlines()
        self.assertEqual(lines[0].strip(), "step,x,y,map,bank,room,battlestate,playerstate,warp,btn,reward")
        # At least one data row was written (RAM-state validity gating may
        # drop the very first frame or two, same as the PNG path).
        self.assertGreater(len(lines), 1)
        shutil.rmtree(self.config["record_path"], ignore_errors=True)

    def test_enable_record_default_still_writes_pngs(self):
        """Checkpoint replays don't pass record_format, so the default must
        stay "images" (PNG per step) — unchanged behaviour."""
        import glob
        import os

        env = PyBoyEnvironment(self.config)
        env.reset()
        env.enable_record("N_goals_0/ep_1", use_episode_number=False)
        for _ in range(5):
            env.step(0)
        env.close()

        ep_dir = os.path.join(self.config["record_path"], "N_goals_0/ep_1")
        self.assertFalse(os.path.exists(os.path.join(ep_dir, "trajectory.csv")))
        self.assertGreater(len(glob.glob(os.path.join(ep_dir, "*.png"))), 0)
        shutil.rmtree(self.config["record_path"], ignore_errors=True)

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
