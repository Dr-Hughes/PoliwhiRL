# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import math
from PoliwhiRL.environment import PyBoyEnvironment as Env
import numpy as np

class BaselineAgent:

    def step(self, env, state, model, temperature=0.0):
        action = self.get_action(model, state, temperature)
        next_state, reward, done, _ = env.step(action)
        return action, reward, next_state, done

    def get_action(self, model, state, temperature=1.0):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state)
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
                curiosity_values[action] = self.compute_curiosity(curiosity_model, state, next_state_pred, torch.tensor([action]))
            return curiosity_values.argmax().item()
        else:
            # Exploit using Q-values
            return q_values.argmax().item()


    def get_cyclical_temperature(self, 
        temperature_cycle_length, min_temperature, max_temperature, i
    ):
        cycle_progress = (i % temperature_cycle_length) / temperature_cycle_length
        return (
            min_temperature
            + (max_temperature - min_temperature)
            * (math.cos(cycle_progress * 2 * math.pi) + 1)
            / 2
        )

    def run_episode(self, model, config, temperature, record_loc=None):
        env = Env(config)
        state = env.reset()
        if record_loc is not None:
            env.enable_record(record_loc, False)
        done = False
        episode_experiences = []

        while not done:
            action, reward, next_state, done = self.step(env, state, model, temperature)
            episode_experiences.append((state, action, reward, next_state, done))
            state = next_state

        return episode_experiences, temperature
