# -*- coding: utf-8 -*-
def setup_and_train_PPO(config):
    from PoliwhiRL.environment import PyBoyEnvironment as Env
    from PoliwhiRL.agents.PPO import PPOAgent

    env = Env(config)
    try:
        state_shape = env.output_shape()
        num_actions = env.action_space.n

        agent = PPOAgent(state_shape, num_actions, config)
        if config["load_checkpoint"]:
            agent.load_model(config["load_checkpoint"])

        agent.train_agent()

        return agent.model
    finally:
        env.close()
