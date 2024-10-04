from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.PPO import PPOAgent, MultiAgentPPO

def setup_and_train_PPO(config):
    env = Env(config)
    state_shape = env.output_shape()
    num_actions = env.action_space.n
    num_agents = config["ppo_num_agents"]

    if num_agents > 1:
        multi_agent_ppo = MultiAgentPPO(config)
        final_model = multi_agent_ppo.train()
        return final_model
    else:
        agent = PPOAgent(state_shape, num_actions, config)
        if config["load_checkpoint"] != "":
            agent.load_model(config["load_checkpoint"])
        
        if config["use_curriculum"]:
            agent.run_curriculum(1, config["N_goals_target"], 600)
        else:
            agent.train_agent()
        
        return agent.model