# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
import os
import torch
import numpy as np
from PoliwhiRL.environment.gym_env import PyBoyEnvironment
from PoliwhiRL.models.PPO.PPOTransformer import PPOTransformer
from PoliwhiRL.agents.PPO import PPOAgent
from main import load_default_config


class TestPPOModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = load_default_config()
        self.config["model"] = "PPO"
        self.config["erase"] = False
        self.config["checkpoint"] = f"{self.temp_dir}"
        self.config["results_dir"] = self.temp_dir
        self.config["device"] = "cpu"
        # Disable curriculum-resume resets so save/load round-trip preserves
        # optimizer and scheduler state for the assertions in test_model_save_load.
        self.config["reset_lr_scheduler_on_load"] = False
        self.config["reset_optimizer_on_load"] = False
        self.env = PyBoyEnvironment(self.config)
        self.state_shape = (
            self.env.get_screen_size()
            if self.config["vision"]
            else self.env.get_game_area().shape
        )
        self.input_shape = self.state_shape
        self.action_size = self.env.action_space.n

    def tearDown(self):
        self.env.close()
        shutil.rmtree(self.temp_dir)

    def test_agent_initialization(self):
        agent = PPOAgent(self.input_shape, self.action_size, self.config)
        self.assertIsInstance(agent.model.actor_critic, PPOTransformer)
        self.assertEqual(agent.action_size, self.action_size)

    def test_model_forward_pass(self):
        model = PPOTransformer(self.input_shape, self.action_size)
        batch_size = 1
        seq_len = 8
        dummy_input = torch.randn(batch_size, seq_len, *self.input_shape)
        action_probs, state_values, new_mems = model(dummy_input)
        self.assertEqual(action_probs.shape, (batch_size, self.action_size))
        self.assertEqual(state_values.shape, (batch_size, 1))
        self.assertEqual(len(new_mems), model.num_layers)
        self.assertEqual(
            new_mems[0].shape, (batch_size, model.mem_len, model.d_model)
        )

    def test_agent_action_selection(self):
        agent = PPOAgent(self.input_shape, self.action_size, self.config)
        state = self.env.reset()
        state_sequence = [state] * agent.sequence_length
        action, log_prob, new_mems = agent.model.get_action(np.array(state_sequence))
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_size)
        self.assertIsInstance(log_prob, float)
        self.assertEqual(len(new_mems), agent.model.actor_critic.num_layers)

    def test_episode_storage(self):
        agent = PPOAgent(self.input_shape, self.action_size, self.config)
        state = self.env.reset()
        next_state, reward, done, _ = self.env.step(0)
        mems = agent.model.init_mems(batch_size=1)
        action, log_prob, _ = agent.model.get_action(
            np.array([state] * agent.sequence_length), mems
        )
        agent.memory.store_transition(
            state, next_state, action, reward, done, log_prob, mems
        )
        self.assertEqual(len(agent.memory), 1)

    def test_model_save_load(self):
        agent = PPOAgent(self.input_shape, self.action_size, self.config)
        agent.save_model(self.config["checkpoint"])

        self.assertTrue(os.path.exists(f"{self.config['checkpoint']}/actor_critic.pth"))
        self.assertTrue(os.path.exists(f"{self.config['checkpoint']}/optimizer.pth"))
        self.assertTrue(os.path.exists(f"{self.config['checkpoint']}/scheduler.pth"))
        self.assertTrue(os.path.exists(f"{self.config['checkpoint']}/info.pth"))

        new_agent = PPOAgent(self.input_shape, self.action_size, self.config)
        new_agent.load_model(self.config["checkpoint"])

        sample_param_name = list(agent.model.actor_critic.state_dict().keys())[0]
        self.assertTrue(
            torch.all(
                torch.eq(
                    agent.model.actor_critic.state_dict()[sample_param_name],
                    new_agent.model.actor_critic.state_dict()[sample_param_name],
                )
            )
        )

        self.assertEqual(
            agent.model.optimizer.state_dict()["param_groups"],
            new_agent.model.optimizer.state_dict()["param_groups"],
        )

        self.assertEqual(
            agent.model.scheduler.state_dict(), new_agent.model.scheduler.state_dict()
        )

        self.assertEqual(agent.episode, new_agent.episode)

    def test_ppo_losses_computation(self):
        agent = PPOAgent(self.input_shape, self.action_size, self.config)
        batch_size = 10
        seq_len = agent.sequence_length
        dummy_states = torch.randn(batch_size, seq_len, *self.input_shape)
        dummy_actions = torch.randint(0, agent.action_size, (batch_size,))
        dummy_rewards = torch.rand(batch_size)
        dummy_dones = torch.zeros(batch_size, dtype=torch.bool)
        dummy_old_log_probs = torch.rand(batch_size)

        batch_data = {
            "states": dummy_states,
            "actions": dummy_actions,
            "rewards": dummy_rewards,
            "dones": dummy_dones,
            "old_log_probs": dummy_old_log_probs,
        }

        actor_loss, critic_loss, entropy_loss = agent.model._compute_ppo_losses(
            batch_data, 1
        )
        self.assertIsInstance(actor_loss, torch.Tensor)
        self.assertIsInstance(critic_loss, torch.Tensor)
        self.assertIsInstance(entropy_loss, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
