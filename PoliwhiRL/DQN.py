# -*- coding: utf-8 -*-
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.DQN import DQNPokemonAgent


def setup_and_train_DQN(config):
    env = Env(config)
    try:
        state_shape = (
            env.get_screen_size() if config["vision"] else env.get_game_area().shape
        )
        num_actions = env.action_space.n

        agent = DQNPokemonAgent(state_shape, num_actions, config)
        agent.run_ciriculum(1, config["N_goals_target"], 200)
    finally:
        # Ensure environment is properly closed
        env.close()
