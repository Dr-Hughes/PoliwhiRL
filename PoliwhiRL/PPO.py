# -*- coding: utf-8 -*-
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.PPO import PPOAgent
from PoliwhiRL.models.PPO import PPOModel
import torch.multiprocessing as mp


def run_agent(i, state_shape, num_actions, config):

    config["checkpoint"] = f"{config['checkpoint']}_{i}"
    config["record_path"] = f"{config['record_path']}_{i}"
    config["export_state_loc"] = f"{config['export_state_loc']}_{i}"
    config["results_dir"] = f"{config['results_dir']}_{i}"

    agent = PPOAgent(state_shape, num_actions, config)
    if config["load_checkpoint"] != "":
        agent.load_model(config["load_checkpoint"])
    if config["use_curriculum"]:
        agent.run_curriculum(1, config["N_goals_target"], 600)
    else:
        agent.train_agent()


def average_models(model_paths, input_shape, action_size, config):
    # Initialize a new model to hold the averaged parameters
    averaged_model = PPOModel(input_shape, action_size, config)

    # Dictionary to hold the sum of parameters
    summed_params = {}

    # Count of models
    model_count = 0

    for path in model_paths:
        model = PPOModel(input_shape, action_size, config)
        model.load(path)
        model_count += 1

        # Sum up the parameters
        for name, param in model.actor_critic.named_parameters():
            if name not in summed_params:
                summed_params[name] = param.data.clone()
            else:
                summed_params[name] += param.data

    # Average the parameters
    for name, param in averaged_model.actor_critic.named_parameters():
        param.data = summed_params[name] / model_count

    return averaged_model


def combine_parallel_agents(config, num_agents, total_episodes_run):
    # Paths to the saved models
    model_paths = [
        f"{config['checkpoint']}_{i}/model_{config['N_goals_target']}_ep_{config['num_episodes']}"
        for i in range(num_agents)
    ]

    # Get input_shape and action_size from the environment
    env = Env(config)
    input_shape = env.output_shape()
    action_size = env.action_space.n

    # Average the models
    averaged_model = average_models(model_paths, input_shape, action_size, config)
    agent = PPOAgent(input_shape, action_size, config)
    agent.model = averaged_model
    agent.episode = total_episodes_run

    # Save the averaged model
    averaged_model_path = f"{config['checkpoint']}_averaged"
    agent.save_model(averaged_model_path)

    print(f"Averaged model saved to {averaged_model_path}")

    return averaged_model


def setup_and_train_PPO(config):
    env = Env(config)
    state_shape = env.output_shape()
    num_actions = env.action_space.n
    num_agents = config["ppo_num_agents"]
    total_episodes_run = config["start_episode"]
    if num_agents > 1:
        with mp.Pool(processes=num_agents) as pool:
            pool.starmap(
                run_agent,
                [(i, state_shape, num_actions, config) for i in range(num_agents)],
            )
        total_episodes_run += config["num_episodes"]
        _ = combine_parallel_agents(config, num_agents, total_episodes_run)

    else:
        # Single agent training (unchanged)
        agent = PPOAgent(state_shape, num_actions, config)
        if config["load_checkpoint"] != "":
            agent.load_model(config["load_checkpoint"], config["load_checkpoint_num"])

        if config["use_curriculum"]:
            agent.run_curriculum(1, config["N_goals_target"], 600)
        else:
            agent.train_agent()
