# -*- coding: utf-8 -*-
from collections import deque
import torch
import torch.nn.functional as F
import math
from PoliwhiRL.environment import PyBoyEnvironment as Env
import numpy as np
from .curiosity import CuriosityModel


class BaselineAgent:
    def __init__(self):
        self.curiosity_model = None

    def setup_curiosity(self, state_shape, action_size):
        self.curiosity_model = CuriosityModel(state_shape, action_size)
        self.curiosity_optimizer = torch.optim.Adam(
            self.curiosity_model.parameters(), lr=0.0001
        )

    def train_curiosity(self, states, actions, next_states):
        predicted_next_states = self.curiosity_model(states, actions)
        curiosity_loss = F.mse_loss(predicted_next_states, next_states)

        with torch.no_grad():
            intrinsic_rewards = F.mse_loss(
                predicted_next_states, next_states, reduction="none"
            ).mean(dim=1)

        self.curiosity_optimizer.zero_grad()
        curiosity_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.curiosity_model.parameters(), max_norm=10.0)
        self.curiosity_optimizer.step()
        return curiosity_loss.item(), intrinsic_rewards

    def step(self, env, state_sequence, model, temperature=0.0):
        action = self.get_action(model, state_sequence, temperature)
        next_state, reward, done, _ = env.step(action)
        return action, reward, next_state, done

    def get_action(self, model, state_sequence, temperature=1.0):
        state_sequence = torch.FloatTensor(np.array(state_sequence)).unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            q_values = model(state_sequence)
        q_values = q_values[0, -1, :]
        if temperature > 0:
            probs = F.softmax(q_values / temperature, dim=0)
            return torch.multinomial(probs, 1).item()
        return q_values.argmax().item()

    def compute_curiosity(self, curiosity_model, state, next_state, action):
        with torch.no_grad():
            predicted_next_state = curiosity_model(state, action)
            curiosity = F.mse_loss(
                predicted_next_state, next_state.detach(), reduction="none"
            ).mean(dim=1)
        return curiosity

    def get_action_curiosity(self, model, curiosity_model, state, epsilon=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state)
        q_values = q_values[0, -1, :]

        if np.random.random() < epsilon:
            # Explore using curiosity
            curiosity_values = torch.zeros_like(q_values)
            for action in range(len(q_values)):
                next_state_pred = curiosity_model(state, torch.tensor([action]))
                curiosity_values[action] = self.compute_curiosity(
                    curiosity_model, state, next_state_pred, torch.tensor([action])
                )
            return curiosity_values.argmax().item()
        else:
            # Exploit using Q-values
            return q_values.argmax().item()

    def get_cyclical_temperature(
        self, temperature_cycle_length, min_temperature, max_temperature, i
    ):
        cycle_progress = (i % temperature_cycle_length) / temperature_cycle_length
        return (
            min_temperature
            + (max_temperature - min_temperature)
            * (math.cos(cycle_progress * 2 * math.pi) + 1)
            / 2
        )

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
            state = env.load_gym_state(load_path)
        else:
            state = env.reset()

        if record_loc is not None:
            env.enable_record(record_loc, False)
        sequence_length = config["sequence_length"]
        done = False
        episode_experiences = []

        # Initialize state sequence with initial state repeated
        state_sequence = deque([state], maxlen=sequence_length)

        while not done:
            action, reward, next_state, done = self.step(
                env, list(state_sequence), model, temperature
            )

            # Store only the current state, action, reward, next_state, and done flag
            episode_experiences.append((state, action, reward, next_state, done))

            # Update state sequence for next iteration
            state_sequence.append(next_state)
            state = next_state

        if save_path is not None:
            env.save_gym_state(save_path)

        return episode_experiences, temperature
