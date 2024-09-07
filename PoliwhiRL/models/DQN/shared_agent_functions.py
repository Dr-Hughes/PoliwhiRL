# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import math
from PoliwhiRL.environment import PyBoyEnvironment as Env


def step(env, state, model, temperature=0.0):
    action = get_action(model, state, temperature)
    next_state, reward, done, _ = env.step(action)
    return action, reward, next_state, done


def get_action(model, state, temperature=1.0):
    state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state)
    q_values = q_values[0, -1, :]
    if temperature > 0:
        probs = F.softmax(q_values / temperature, dim=0)
        return torch.multinomial(probs, 1).item()
    return q_values.argmax().item()


def get_cyclical_temperature(
    temperature_cycle_length, min_temperature, max_temperature, i
):
    cycle_progress = (i % temperature_cycle_length) / temperature_cycle_length
    return (
        min_temperature
        + (max_temperature - min_temperature)
        * (math.cos(cycle_progress * 2 * math.pi) + 1)
        / 2
    )


def run_episode(model, config, temperature, record_loc=None):
    env = Env(config)
    state = env.reset()
    if record_loc is not None:
        env.enable_record(record_loc)
    done = False
    episode_experiences = []

    while not done:
        action, reward, next_state, done = step(env, state, model, temperature)
        episode_experiences.append((state, action, reward, next_state, done))
        state = next_state

    return episode_experiences
