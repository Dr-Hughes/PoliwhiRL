# -*- coding: utf-8 -*-
import torch
import torch.multiprocessing as mp
import os
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.PPO import PPOAgent
from tqdm.auto import tqdm as auto_tqdm


class MultiAgentPPO:
    def __init__(self, config):
        self.config = config
        self.num_agents = config["ppo_num_agents"]
        self.iterations = config["ppo_iterations"]
        self.total_episodes_run = config["start_episode"]

    def run_agent(self, i, state_shape, num_actions, og_checkpoint):
        # Set environment variable for this worker to identify itself in tqdm
        os.environ["TQDM_WORKER_ID"] = str(i)

        config = self.config.copy()
        config["checkpoint"] = f"{config['checkpoint']}_{i}"
        config["record_path"] = f"{config['record_path']}_{i}"
        config["export_state_loc"] = f"{config['export_state_loc']}_{i}"
        config["results_dir"] = f"{config['results_dir']}_{i}"

        # Add worker-specific tqdm configuration
        config["tqdm_position"] = i
        config["tqdm_worker_id"] = i
        config["tqdm_desc_prefix"] = f"Agent {i}"

        agent = PPOAgent(state_shape, num_actions, config)
        agent.load_model(og_checkpoint)

        if config["use_curriculum"]:
            agent.run_curriculum(1, config["N_goals_target"], 600)
        else:
            agent.train_agent()

    def average_models(self, agent_paths, input_shape, action_size):
        averaged_agent = PPOAgent(input_shape, action_size, self.config)

        actor_critic_params = []
        icm_params = []
        optimizer_states = []
        scheduler_states = []
        all_episode_data = []

        agent_count = 0

        for path in agent_paths:
            agent = PPOAgent(input_shape, action_size, self.config)
            agent.load_model(path)
            agent_count += 1

            actor_critic_params.append(
                {
                    name: param.data
                    for name, param in agent.model.actor_critic.named_parameters()
                }
            )
            icm_params.append(
                {
                    name: param.data
                    for name, param in agent.model.icm.icm.named_parameters()
                }
            )
            optimizer_states.append(agent.model.optimizer.state_dict())
            scheduler_states.append(agent.model.scheduler.state_dict())

            # Collect episode data
            all_episode_data.append(agent.get_episode_data())

        # Average actor_critic parameters
        for name in actor_critic_params[0].keys():
            averaged_param = (
                sum(params[name] for params in actor_critic_params) / agent_count
            )
            averaged_agent.model.actor_critic.get_parameter(name).data.copy_(
                averaged_param
            )

        # Average ICM parameters
        for name in icm_params[0].keys():
            averaged_param = sum(params[name] for params in icm_params) / agent_count
            averaged_agent.model.icm.icm.get_parameter(name).data.copy_(averaged_param)

        # Average optimizer state
        averaged_optimizer_state = optimizer_states[0]
        for key in averaged_optimizer_state["state"].keys():
            for param_key in averaged_optimizer_state["state"][key].keys():
                if isinstance(
                    averaged_optimizer_state["state"][key][param_key], torch.Tensor
                ):
                    averaged_optimizer_state["state"][key][param_key] = (
                        sum(
                            state["state"][key][param_key] for state in optimizer_states
                        )
                        / agent_count
                    )
        averaged_agent.model.optimizer.load_state_dict(averaged_optimizer_state)

        # Average scheduler state
        averaged_scheduler_state = scheduler_states[0]
        for key, value in averaged_scheduler_state.items():
            if isinstance(value, torch.Tensor):
                averaged_scheduler_state[key] = (
                    sum(state[key] for state in scheduler_states) / agent_count
                )
        averaged_agent.model.scheduler.load_state_dict(averaged_scheduler_state)

        # Combine episode data
        averaged_agent.set_episode_data(self.combine_episode_data(all_episode_data))

        return averaged_agent

    def combine_episode_data(self, all_episode_data):
        # Initialize combined data with the structure of episode_data
        combined_data = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_losses": [],
            "episode_icm_losses": [],
            "moving_avg_reward": [],
            "moving_avg_length": [],
            "moving_avg_loss": [],
            "moving_avg_icm_loss": [],
            "buttons_pressed": [],
            # Add a new field to store individual agent data
            "individual_agent_data": all_episode_data,
        }

        # For the averaged model, we'll use the data from the first agent
        # This ensures we don't mix statistics between agents
        if all_episode_data:
            first_agent_data = all_episode_data[0]
            for key in combined_data:
                if key != "individual_agent_data":  # Skip the new field
                    if isinstance(first_agent_data[key], list):
                        combined_data[key] = first_agent_data[key].copy()
                    else:
                        combined_data[key] = first_agent_data[key].copy()

        return combined_data

    def combine_parallel_agents(self, iteration, og_checkpoint):
        agent_paths = [
            f"{self.config['checkpoint']}_{i}" for i in range(self.num_agents)
        ]

        env = Env(self.config)
        input_shape = env.output_shape()
        action_size = env.action_space.n

        # Average the models but preserve individual agent statistics
        averaged_agent = self.average_models(agent_paths, input_shape, action_size)
        averaged_agent.episode = self.total_episodes_run

        # Save the averaged model
        averaged_model_path = og_checkpoint
        averaged_agent.save_model(averaged_model_path)

        # Also save individual agent models with their own statistics
        for i in range(self.num_agents):
            agent_path = f"{self.config['checkpoint']}_{i}"
            # Load the individual agent
            agent = PPOAgent(input_shape, action_size, self.config)
            agent.load_model(agent_path)

            # Update the episode count but keep its own statistics
            agent.episode = self.total_episodes_run

            # Save the updated agent
            agent.save_model(agent_path)

        return averaged_agent.model

    def train(self):
        env = Env(self.config)
        state_shape = env.output_shape()
        num_actions = env.action_space.n
        og_checkpoint = self.config["checkpoint"]

        for iteration in auto_tqdm(
            range(self.iterations), desc="Iterations", position=0
        ):
            print(f"\nStarting iteration {iteration + 1}/{self.iterations}")
            print(f"{'=' * 50}")
            print(f"Running {self.num_agents} parallel agents...")

            # Create a process pool with the specified number of agents
            with mp.Pool(processes=self.num_agents) as pool:
                pool.starmap(
                    self.run_agent,
                    [
                        (i, state_shape, num_actions, og_checkpoint)
                        for i in range(self.num_agents)
                    ],
                )

            self.total_episodes_run += self.config["num_episodes"]
            print(f"\nCombining models from {self.num_agents} agents...")
            averaged_model = self.combine_parallel_agents(iteration, og_checkpoint)
            print(
                f"Iteration {iteration + 1} completed. Total episodes run: {self.total_episodes_run}"
            )

        print("\nAll iterations completed.")
        return averaged_model
