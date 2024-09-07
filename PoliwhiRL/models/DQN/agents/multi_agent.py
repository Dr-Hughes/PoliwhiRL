# -*- coding: utf-8 -*-
import torch
import torch.multiprocessing as mp
from PoliwhiRL.models.DQN.DQNModel import TransformerDQN
from .baseline import BaselineAgent

class ParallelAgentRunner(BaselineAgent):
    def __init__(self, base_model):
        self.shared_model = self._init_shared_model(base_model)
        self.update_shared_model(base_model)

    def _init_shared_model(self, base_model):
        device = torch.device("cpu")
        state_shape = base_model.input_shape
        action_size = base_model.action_size
        shared_model = TransformerDQN(state_shape, action_size).to(device)
        shared_model.share_memory()
        return shared_model

    def update_shared_model(self, base_model):
        self.shared_model.load_state_dict(base_model.state_dict())

    def run_agents(self, config, temperatures, record_loc=None):
        num_agents = len(temperatures)
        episode_experiences = []

        with mp.Pool(processes=num_agents) as pool:
            # Run episodes in parallel
            results = pool.starmap(
                self.run_episode,
                [
                    (
                        self.shared_model,
                        config,
                        round(temperature, 2),
                        (
                            None
                            if record_loc is None
                            else f"{record_loc}_{round(temperature, 2)}"
                        ),
                    )
                    for temperature in temperatures
                ],
            )

        for episode in results:
            episode_experiences.append(episode)

        return episode_experiences
