# -*- coding: utf-8 -*-
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.PPO import PPOAgent


def setup_and_train_PPO(config):
    env = Env(config)
    state_shape = env.output_shape()
    num_actions = env.action_space.n

    agent = PPOAgent(state_shape, num_actions, config)

    if config["load_checkpoint"] != "":
        agent.load_model(config["load_checkpoint"], config["load_checkpoint_num"])

    if config["use_curriculum"]:
        agent.run_curriculum(1, config["N_goals_target"], 600)
    else:
        agent.train_agent()
