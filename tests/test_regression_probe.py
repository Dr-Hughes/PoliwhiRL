# -*- coding: utf-8 -*-
"""Smoke test for the regression-probe mechanism (real PyBoy instances,
mirrors tests/test_vec_env.py's TestVecPPOAgentSmoke pattern).

Pinned behaviours:

- With probe_enabled and probe_frequency=1, a single rollout triggers
  exactly one probe event, appending to episode_data["probe_success_rate"]
  / ["probe_rollout_idx"].
- The probe env is created lazily and closed by train_agent()'s cleanup.
- probe_enabled defaults to False and doesn't run when unset (checked via
  the existing TestVecPPOAgentSmoke config in test_vec_env.py already
  passing with no probe keys set at all).
- The long-horizon probe (long_probe_enabled) is an independent on/off
  switch with its own frequency and episode_length, recording to its own
  episode_data["long_probe_*"] keys without disturbing the regular probe.
"""
import unittest
import tempfile
import shutil
import os

from PoliwhiRL.agents.PPO import VecPPOAgent
from PoliwhiRL.environment.gym_env import RAM_OBS_DIM
from main import load_default_config


class TestRegressionProbeSmoke(unittest.TestCase):
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
                "n_goals_target": 0,
                "goals": [],
                "vision": True,
                "scaling_factor": 0.5,
                "use_grayscale": False,
                "erase": False,
                "load_checkpoint": "",
                "record": False,
                "ram_obs_dim": RAM_OBS_DIM,
                # Probe config: tiny budget so this stays a fast unit test.
                "probe_enabled": True,
                "probe_frequency": 1,
                "probe_episodes": 1,
                "probe_episode_length": 4,
                "probe_goals": [
                    {"type": "pokedex", "kind": "owned", "threshold": 1},
                ],
                "probe_label": "test_probe",
            }
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_probe_runs_once_per_frequency(self):
        from PoliwhiRL.environment import PyBoyEnvironment

        env = PyBoyEnvironment(self.config)
        try:
            state_shape = env.output_shape()
            num_actions = env.action_space.n
        finally:
            env.close()

        agent = VecPPOAgent(state_shape, num_actions, self.config)
        agent.train_agent()

        rates = agent.episode_data["probe_success_rate"]
        rollouts = agent.episode_data["probe_rollout_idx"]
        self.assertEqual(len(rates), 1)
        self.assertEqual(len(rollouts), 1)
        self.assertIn(rates[0], (0.0, 1.0))  # 1 probe episode -> binary rate
        # Probe env was created lazily and cleaned up by train_agent().
        self.assertIsNotNone(agent._probe_env)
        # Long-horizon probe wasn't enabled — must stay untouched.
        self.assertEqual(agent.episode_data["long_probe_success_rate"], [])
        self.assertIsNone(agent._long_probe_env)


class TestLongHorizonProbeSmoke(unittest.TestCase):
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
                "n_goals_target": 0,
                "goals": [],
                "vision": True,
                "scaling_factor": 0.5,
                "use_grayscale": False,
                "erase": False,
                "load_checkpoint": "",
                "record": False,
                "ram_obs_dim": RAM_OBS_DIM,
                # Regular probe OFF, long-horizon probe ON — the two must
                # be independently switchable.
                "probe_enabled": False,
                "long_probe_enabled": True,
                "long_probe_frequency": 1,
                "long_probe_episodes": 1,
                "long_probe_episode_length": 5,
                "probe_goals": [
                    {"type": "pokedex", "kind": "owned", "threshold": 1},
                ],
                "probe_label": "test_probe",
            }
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_long_probe_runs_independently_of_regular_probe(self):
        from PoliwhiRL.environment import PyBoyEnvironment

        env = PyBoyEnvironment(self.config)
        try:
            state_shape = env.output_shape()
            num_actions = env.action_space.n
        finally:
            env.close()

        agent = VecPPOAgent(state_shape, num_actions, self.config)
        agent.train_agent()

        self.assertEqual(len(agent.episode_data["long_probe_success_rate"]), 1)
        self.assertEqual(len(agent.episode_data["long_probe_rollout_idx"]), 1)
        self.assertIsNotNone(agent._long_probe_env)
        # Regular probe was off — must stay untouched.
        self.assertEqual(agent.episode_data["probe_success_rate"], [])
        self.assertIsNone(agent._probe_env)


if __name__ == "__main__":
    unittest.main()
