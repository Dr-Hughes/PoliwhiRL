# -*- coding: utf-8 -*-
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.PPO import PPOAgent


def setup_and_train_PPO(config):
    env = Env(config)
    state_shape = (
        env.get_screen_size() if config["vision"] else env.get_game_area().shape
    )
    num_actions = env.action_space.n

    agent = PPOAgent(state_shape, num_actions, config)
    agent.run_curriculum(1, config["N_goals_target"], 200)
