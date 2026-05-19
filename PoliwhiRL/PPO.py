# -*- coding: utf-8 -*-
def setup_and_train_PPO(config):
    from PoliwhiRL.environment import PyBoyEnvironment as Env
    from PoliwhiRL.agents.PPO import PPOAgent, VecPPOAgent

    num_envs = int(config.get("num_envs", 1))

    # Probe shape/action size with a single short-lived env. The vec wrapper
    # spins up its own workers; the single-env path reuses this env if N=1.
    env = Env(config)
    try:
        state_shape = env.output_shape()
        num_actions = env.action_space.n
    finally:
        env.close()

    if num_envs > 1:
        agent = VecPPOAgent(state_shape, num_actions, config)
    else:
        agent = PPOAgent(state_shape, num_actions, config)

    if config["load_checkpoint"]:
        agent.load_model(config["load_checkpoint"])

    agent.train_agent()
    return agent.model
