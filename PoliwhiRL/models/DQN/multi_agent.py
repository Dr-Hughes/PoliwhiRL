# -*- coding: utf-8 -*-
import torch
import torch.multiprocessing as mp
from .shared_agent_functions import run_episode
from PoliwhiRL.models.DQN.DQNModel import TransformerDQN


def init_shared_model(model):
    device = torch.device("cpu")
    state_shape = model.input_shape
    action_size = model.action_size
    shared_model = TransformerDQN(state_shape, action_size).to(device)
    shared_model.load_state_dict(model.state_dict())
    shared_model.share_memory()
    return shared_model


def run_parallel_agents(base_model, config, temperatures, record_loc=None):
    shared_model = init_shared_model(base_model)
    num_agents = len(temperatures)
    episode_experiences = []
    with mp.Pool(processes=num_agents) as pool:
        # Run episodes in parallel
        results = pool.starmap(
            run_episode,
            [
                (
                    shared_model,
                    config,
                    temperature,
                    None if record_loc is None else f"{record_loc}_{temperature}",
                )
                for temperature in temperatures
            ],
        )
        for episode in results:
            episode_experiences.append(episode)
    return episode_experiences
