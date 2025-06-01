# -*- coding: utf-8 -*-
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.PPO import PPOAgent, MultiAgentPPO
from PoliwhiRL.agents.PPO.ppo_multi_agent_improved import ImprovedMultiAgentPPO


def setup_and_train_PPO(config):
    # Use shared temp directory for single agent as well to reduce file creation
    env = Env.create_with_shared_temp(config)
    try:
        state_shape = env.output_shape()
        num_actions = env.action_space.n
        num_agents = config["ppo_num_agents"]

        if num_agents > 1:
            # Use improved multi-agent system with agent reuse and shared temp directories
            use_improved_multiagent = config.get("use_improved_multiagent", True)

            if use_improved_multiagent:
                print(
                    "Using improved multi-agent PPO with agent reuse and shared temp directories"
                )
                if config["load_checkpoint"] != "":
                    print(
                        f"Multi-agent will load checkpoint from: {config['load_checkpoint']}"
                    )
                multi_agent_ppo = ImprovedMultiAgentPPO(config)
            else:
                print("Using legacy multi-agent PPO (creates more temporary files)")
                if config["load_checkpoint"] != "":
                    print(
                        f"Multi-agent will load checkpoint from: {config['load_checkpoint']}"
                    )
                multi_agent_ppo = MultiAgentPPO(config)

            final_model = multi_agent_ppo.train()
            return final_model
        else:
            # Single agent with shared temp directory support
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
        # Ensure environment is properly closed
        env.close()
