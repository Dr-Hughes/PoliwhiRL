# -*- coding: utf-8 -*-
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.models.DQN.DQNModel import TransformerDQN


def init_shared_model(model):
    device = torch.device("cpu")
    state_shape = model.input_shape
    action_size = model.action_size
    shared_model = TransformerDQN(state_shape, action_size).to(device)
    shared_model.load_state_dict(model.state_dict())
    shared_model.share_memory()
    return shared_model


def gather_experiences(base_model, config, temperatures):
    shared_model = init_shared_model(base_model)
    num_agents = len(temperatures)
    experiences = []
    with mp.Pool(processes=num_agents) as pool:
        # Run episodes in parallel
        results = pool.starmap(
            run_episode,
            [(shared_model, config, temperature) for temperature in temperatures],
        )

        # Collect results
        for episode_experiences in results:
            experiences.extend(episode_experiences)
    return experiences


def run_episode(shared_model, config, temperature):
    env = Env(config)
    state = env.reset()
    done = False
    episode_reward = 0
    episode_experiences = []

    while not done:
        # Select action
        action = get_action(shared_model, state, temperature)

        # Take action in environment
        next_state, reward, done, _ = env.step(action)

        # Store experience
        episode_experiences.append((state, action, reward, next_state, done))

        state = next_state
        episode_reward += reward

    return episode_experiences


def get_action(model, state, temperature=1.0):
    state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state)
    q_values = q_values[0, -1, :]
    probs = F.softmax(q_values / temperature, dim=0)
    return torch.multinomial(probs, 1).item()
