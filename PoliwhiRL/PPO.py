# -*- coding: utf-8 -*-
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.PPO import PPOAgent, PPOParallelRunner


def setup_and_train_PPO(config):
    num_agents = config["ppo_num_agents"]

    if num_agents > 1:
        # Use parallel runner based on DQN pattern
        print(f"Using PPO parallel runner with {num_agents} agents")
        if config["load_checkpoint"] != "":
            print(f"Multi-agent will load checkpoint from: {config['load_checkpoint']}")

        parallel_runner = PPOParallelRunner(config)
        parallel_runner.train()
        return None  # Model is saved to checkpoint
    else:
        # Single agent training
        env = Env.create_with_shared_temp(config)
        try:
            state_shape = env.output_shape()
            num_actions = env.action_space.n

            agent_config = config.copy()
            agent_config["shared_temp_dir"] = env.temp_dir

            agent = PPOAgent(state_shape, num_actions, agent_config)
            if config["load_checkpoint"] != "":
                agent.load_model(config["load_checkpoint"])

            if config["use_curriculum"]:
                agent.run_curriculum(1, config["N_goals_target"], 600)
            else:
                agent.train_agent()

            return agent.model
        finally:
            env.close()
