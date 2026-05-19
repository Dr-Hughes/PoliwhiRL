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
        self.assertEqual(new_mems[0].shape, (batch_size, model.mem_len, model.d_model))

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

        actor_loss, critic_loss, entropy_loss, approx_kl = agent.model._compute_ppo_losses(
            batch_data, 1
        )
        self.assertIsInstance(actor_loss, torch.Tensor)
        self.assertIsInstance(critic_loss, torch.Tensor)
        self.assertIsInstance(approx_kl, float)
        self.assertIsInstance(entropy_loss, torch.Tensor)

    def test_compute_returns_discount(self):
        """returns[t] = r[t] + gamma * returns[t+1], zeroed at done boundaries."""
        agent = PPOAgent(self.input_shape, self.action_size, self.config)
        gamma = agent.model.gamma
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        dones = torch.tensor([False, False, False, False])

        returns = agent.model._compute_returns(rewards, dones)

        # Compute expected by reverse recursion.
        expected = torch.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t].item() + gamma * running
            expected[t] = running
        self.assertTrue(torch.allclose(returns, expected, atol=1e-6))

    def test_compute_returns_done_resets_bootstrap(self):
        """A True done at step t blocks bootstrap from t+1."""
        agent = PPOAgent(self.input_shape, self.action_size, self.config)
        gamma = agent.model.gamma
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        dones = torch.tensor([False, True, False, False])

        returns = agent.model._compute_returns(rewards, dones)

        # Step 3: return = 4
        # Step 2: return = 3 + gamma * 4
        # Step 1: done=True so running *= 0 before adding: return = 2
        # Step 0: return = 1 + gamma * 2
        self.assertAlmostEqual(returns[3].item(), 4.0, places=5)
        self.assertAlmostEqual(returns[2].item(), 3.0 + gamma * 4.0, places=5)
        self.assertAlmostEqual(returns[1].item(), 2.0, places=5)
        self.assertAlmostEqual(returns[0].item(), 1.0 + gamma * 2.0, places=5)

    def test_compute_gae_matches_manual_recursion(self):
        agent = PPOAgent(self.input_shape, self.action_size, self.config)
        agent.config["ppo_gae_lambda"] = 0.95
        gamma = agent.model.gamma
        gae_lambda = 0.95

        rewards = torch.tensor([1.0, 0.5, -0.3, 2.0])
        values = torch.tensor([0.1, 0.2, 0.3, 0.4])
        dones = torch.tensor([False, False, False, False])

        returns, advantages = agent.model._compute_gae(rewards, values, dones)

        # Standard GAE recursion with zero tail bootstrap. The last advantage
        # also propagates back through earlier gae terms.
        expected_adv = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1].item() if t + 1 < len(rewards) else 0.0
            delta = rewards[t].item() + gamma * next_value - values[t].item()
            gae = delta + gamma * gae_lambda * gae
            expected_adv[t] = gae

        self.assertTrue(torch.allclose(advantages, expected_adv, atol=1e-6))
        self.assertTrue(torch.allclose(returns, advantages + values, atol=1e-6))

    def test_compute_gae_uses_tail_bootstrap(self):
        agent = PPOAgent(self.input_shape, self.action_size, self.config)
        agent.config["ppo_gae_lambda"] = 0.95
        gamma = agent.model.gamma
        gae_lambda = 0.95

        rewards = torch.tensor([1.0, 0.5, -0.3, 2.0])
        values = torch.tensor([0.1, 0.2, 0.3, 0.4])
        dones = torch.tensor([False, False, False, False])
        tail_v = 1.5

        returns, advantages = agent.model._compute_gae(
            rewards, values, dones, last_value=tail_v
        )

        expected_adv = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1].item() if t + 1 < len(rewards) else tail_v
            delta = rewards[t].item() + gamma * next_value - values[t].item()
            gae = delta + gamma * gae_lambda * gae
            expected_adv[t] = gae

        self.assertTrue(torch.allclose(advantages, expected_adv, atol=1e-6))

    def test_compute_returns_uses_tail_bootstrap_when_not_done(self):
        agent = PPOAgent(self.input_shape, self.action_size, self.config)
        gamma = agent.model.gamma

        rewards = torch.tensor([1.0, 2.0])
        dones = torch.tensor([False, False])
        tail_v = 5.0

        returns = agent.model._compute_returns(rewards, dones, last_value=tail_v)

        # Last step: return = r + gamma * tail_v (not done, so bootstrap applies)
        # First step: return = r0 + gamma * returns[1]
        self.assertAlmostEqual(returns[1].item(), 2.0 + gamma * tail_v, places=5)
        self.assertAlmostEqual(
            returns[0].item(), 1.0 + gamma * (2.0 + gamma * tail_v), places=5
        )

    def test_compute_returns_ignores_tail_bootstrap_when_done(self):
        agent = PPOAgent(self.input_shape, self.action_size, self.config)
        gamma = agent.model.gamma

        rewards = torch.tensor([1.0, 2.0])
        dones = torch.tensor([False, True])
        tail_v = 5.0

        returns = agent.model._compute_returns(rewards, dones, last_value=tail_v)

        # Last step is terminal: ~done zeros the bootstrap regardless of tail_v.
        self.assertAlmostEqual(returns[1].item(), 2.0, places=5)
        self.assertAlmostEqual(returns[0].item(), 1.0 + gamma * 2.0, places=5)

    def test_entropy_coef_decays_to_min(self):
        agent = PPOAgent(self.input_shape, self.action_size, self.config)
        coef0 = agent.model._get_entropy_coef(0)
        coef10 = agent.model._get_entropy_coef(10)
        coef_huge = agent.model._get_entropy_coef(10_000_000)

        # Start equals initial coef.
        self.assertAlmostEqual(coef0, agent.model.entropy_coef, places=6)
        # Decay strictly reduces (assuming decay < 1).
        if agent.model.entropy_decay < 1.0:
            self.assertLess(coef10, coef0)
        # Floored at entropy_min.
        self.assertGreaterEqual(coef_huge, agent.model.entropy_min)
        self.assertAlmostEqual(coef_huge, agent.model.entropy_min, places=6)

    def test_forward_action_probs_are_distribution(self):
        """Softmax output sums to ~1 per batch entry."""
        model = PPOTransformer(self.input_shape, self.action_size)
        batch_size = 4
        seq_len = 8
        x = torch.randn(batch_size, seq_len, *self.input_shape)
        probs, _, _ = model(x)
        sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(batch_size), atol=1e-5))
        self.assertTrue((probs >= 0).all())


if __name__ == "__main__":
    unittest.main()
