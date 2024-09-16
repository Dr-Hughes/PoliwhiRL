# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
import os
import torch
from PoliwhiRL.environment.gym_env import PyBoyEnvironment
from PoliwhiRL.models.DQN import TransformerDQN
from PoliwhiRL.agents.DQN import DQNPokemonAgent


from main import load_default_config


class TestDQNModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = load_default_config()
        self.config["erase"] = False  # just in case
        self.config["checkpoint"] = f"{self.temp_dir}"
        self.config["results_dir"] = self.temp_dir
        self.config["device"] = "cpu"
        self.env = PyBoyEnvironment(self.config)
        self.state_shape = (
            self.env.get_screen_size()
            if self.config["vision"]
            else self.env.get_game_area().shape
        )
        self.input_shape = self.state_shape
        self.action_size = self.env.action_space.n

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_agent_initialization(self):
        agent = DQNPokemonAgent(
            self.input_shape, self.action_size, self.config, load_checkpoint=False
        )
        self.assertIsInstance(agent.model, TransformerDQN)
        self.assertIsInstance(agent.target_model, TransformerDQN)
        self.assertEqual(agent.action_size, self.action_size)

    def test_model_forward_pass(self):
        model = TransformerDQN(self.input_shape, self.action_size)
        batch_size = 1
        seq_len = 10
        dummy_input = torch.randn(batch_size, seq_len, *self.input_shape)
        output = model(dummy_input)
        self.assertEqual(output.shape, (batch_size, seq_len, self.action_size))

    def test_agent_action_selection(self):
        agent = DQNPokemonAgent(
            self.input_shape, self.action_size, self.config, load_checkpoint=False
        )
        state = self.env.reset()
        action = agent.get_action(agent.model, [state], 0.1)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_size)

    def test_replay_buffer_addition(self):
        agent = DQNPokemonAgent(
            self.input_shape, self.action_size, self.config, load_checkpoint=False
        )
        state = self.env.reset()
        next_state, _, done, _ = self.env.step(0)
        done = False
        for _ in range(self.config["sequence_length"] - 1):
            agent.replay_buffer.add(state, 0, 0, next_state, done)
        agent.replay_buffer.add(state, 0, 0, next_state, True)
        self.assertEqual(len(agent.replay_buffer), 1)

    def test_model_save_load(self):
        agent = DQNPokemonAgent(
            self.input_shape, self.action_size, self.config, load_checkpoint=False
        )
        agent.save_model(self.config["checkpoint"])

        self.assertTrue(os.path.exists(self.config["checkpoint"] + "/model.pth"))
        self.assertTrue(os.path.exists(self.config["checkpoint"] + "/optimizer.pth"))

        new_agent = DQNPokemonAgent(
            self.input_shape, self.action_size, self.config, load_checkpoint=True
        )
        self.assertTrue(
            torch.all(
                torch.eq(
                    agent.model.state_dict()["fc_out.weight"],
                    new_agent.model.state_dict()["fc_out.weight"],
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
