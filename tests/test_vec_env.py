# -*- coding: utf-8 -*-
"""Smoke tests for the multiprocessing vec env wrapper and the end-to-end
vec training step. These spawn real PyBoy instances, so they require the
ROM and state files to be present in emu_files/.
"""
import unittest
import tempfile
import shutil
import os
import numpy as np
import torch

from PoliwhiRL.environment import VecPyBoyEnv
from PoliwhiRL.agents.PPO import VecPPOAgent
from main import load_default_config


class TestVecPyBoyEnv(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = load_default_config()
        self.config["episode_length"] = 8
        self.config["erase"] = False

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_two_envs_reset_and_step(self):
        vec = VecPyBoyEnv(self.config, num_envs=2)
        try:
            obs = vec.reset()
            self.assertEqual(obs.shape[0], 2)
            obs_shape = obs.shape[1:]
            self.assertEqual(tuple(obs_shape), tuple(vec.output_shape()))
            self.assertEqual(vec.action_size, 9)

            # A handful of steps; envs auto-reset on done.
            for _ in range(5):
                actions = np.array([0, 1])
                next_obs, rewards, dones = vec.step(actions)
                self.assertEqual(next_obs.shape, (2,) + tuple(obs_shape))
                self.assertEqual(rewards.shape, (2,))
                self.assertEqual(dones.shape, (2,))
                self.assertEqual(rewards.dtype, np.float32)
                self.assertEqual(dones.dtype, np.bool_)
        finally:
            vec.close()

    def test_close_is_idempotent(self):
        vec = VecPyBoyEnv(self.config, num_envs=2)
        vec.close()
        # Second close should not raise.
        vec.close()


class TestVecPPOAgentSmoke(unittest.TestCase):
    """One-rollout end-to-end smoke test on the real env."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = load_default_config()
        self.config.update(
            {
                "device": "cpu",
                "num_envs": 2,
                "num_rollouts": 1,
                "episode_length": 6,
                "sequence_length": 3,
                "ppo_update_frequency": 6,
                "ppo_epochs": 2,
                "ppo_target_kl": None,
                "ppo_clip_value_loss": True,
                "report_episode": False,
                "save_checkpoint": False,
                "checkpoint": None,
                "results_dir": os.path.join(self.temp_dir, "Results"),
                "checkpoint_frequency": 999,
                "record_frequency": 999,
                "N_goals_target": 2,
                "vision": True,
                "scaling_factor": 0.5,
                "use_grayscale": False,
                "erase": False,
                "load_checkpoint": "",
                "record": False,
            }
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_single_rollout_updates_model(self):
        # Probe input shape once to mirror dispatch in PPO.py.
        from PoliwhiRL.environment import PyBoyEnvironment
        env = PyBoyEnvironment(self.config)
        try:
            state_shape = env.output_shape()
            num_actions = env.action_space.n
        finally:
            env.close()

        agent = VecPPOAgent(state_shape, num_actions, self.config)

        # Snapshot params, run training, confirm at least one param changed.
        params_before = [p.detach().clone() for p in agent.model.actor_critic.parameters()]
        agent.train_agent()
        params_after = list(agent.model.actor_critic.parameters())

        changed = any(
            not torch.equal(a, b) for a, b in zip(params_before, params_after)
        )
        self.assertTrue(changed, "Expected at least one parameter to update.")


if __name__ == "__main__":
    unittest.main()
