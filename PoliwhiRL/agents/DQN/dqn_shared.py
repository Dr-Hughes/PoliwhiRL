# -*- coding: utf-8 -*-
from collections import deque
import os
import torch
import torch.nn.functional as F
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.base_agent import BaseAgent
import numpy as np


class DQNSharedAgent(BaseAgent):
    def __init__(self):
        pass

    def step(self, env, state_sequence, model, temperature=0.0):
        action = self.get_action(model, state_sequence, temperature)
        next_state, reward, done, _ = env.step(action)
        return action, reward, next_state, done

    def get_action(self, model, state_sequence, temperature=1.0):

        if temperature <= 0:
            return np.random.randint(0, model.action_size)

        state_sequence = (
            torch.FloatTensor(np.array(state_sequence[-1]))
            .unsqueeze(0).unsqueeze(0)
            .to(next(model.parameters()).device)
        )
        with torch.no_grad():
            q_values = model(state_sequence)
        q_values = q_values[0, -1, :]
        if temperature > 0:
            probs = F.softmax(q_values / temperature, dim=0)
            return torch.multinomial(probs, 1).item()
        return q_values.argmax().item()

    def run_episode(
        self,
        model,
        config,
        temperature,
        record_loc=None,
        load_path=None,
        save_path=None,
    ):
        env = Env(config)
        if load_path is not None:
            state = env.load_gym_state(
                load_path, config["episode_length"], config["N_goals_target"]
            )
        else:
            state = env.reset()

        if record_loc is not None:
            env.enable_record(record_loc, False)
        sequence_length = config["sequence_length"]
        done = False
        episode_experiences = []

        state_sequence = deque([state], maxlen=sequence_length)

        while not done:
            action, reward, next_state, done = self.step(
                env, list(state_sequence), model, temperature
            )
            episode_experiences.append((state, action, reward, next_state, done))
            state_sequence.append(next_state)
            state = next_state

        if save_path is not None:
            env.save_gym_state(save_path)
        steps = env.steps
        return episode_experiences, temperature, steps

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/model.pth")
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pth")

    def load_model(self):
        try:
            model_state = torch.load(
                f"{self.checkpoint}/model.pth",
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(model_state)
            optimizer_state = torch.load(
                f"{self.checkpoint}/optimizer.pth",
                map_location=self.device,
                weights_only=True,
            )
            self.optimizer.load_state_dict(optimizer_state)
            print(f"Loaded model from {self.checkpoint}")
        except FileNotFoundError:
            print("No model found, training from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training from scratch.")
