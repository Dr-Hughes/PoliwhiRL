# -*- coding: utf-8 -*-
import torch
import os
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.PPO import PPOAgent
from PoliwhiRL.agents.PPO.agent_pool import AgentPool
from tqdm.auto import tqdm as auto_tqdm
from PoliwhiRL.utils.shared_model import get_shared_model_manager
from PoliwhiRL.utils.resource_manager import get_resource_pool, periodic_cleanup_thread
import threading


class ImprovedMultiAgentPPO:
    """
    Improved Multi-Agent PPO that reuses agents and shared temporary directories
    to reduce temporary file creation and improve performance.
    """

    def __init__(self, config):
        self.config = config
        self.num_agents = config["ppo_num_agents"]
        self.iterations = config["ppo_iterations"]
        self.total_episodes_run = config["start_episode"]
        self.agent_pool = None
        self.shared_model_manager = get_shared_model_manager()
        self.resource_pool = get_resource_pool()

    def train(self):
        """Main training loop using agent pool for efficient resource management"""
        env = Env.create_with_shared_temp(self.config)

        # Start periodic cleanup thread
        cleanup_thread = threading.Thread(
            target=periodic_cleanup_thread,
            args=(self.resource_pool, 30),  # Cleanup every 30 seconds
            daemon=True,
        )
        cleanup_thread.start()

        try:
            state_shape = env.output_shape()
            num_actions = env.action_space.n
            og_checkpoint = self.config["checkpoint"]

            # Initialize agent pool
            self.agent_pool = AgentPool(self.config, state_shape, num_actions)

            print(
                f"Starting improved multi-agent training with {self.num_agents} agents"
            )
            print("Using shared temporary directories to reduce file creation")
            print("Periodic resource cleanup enabled (every 30 seconds)")

            for iteration in auto_tqdm(
                range(self.iterations), desc="Iterations", position=0
            ):
                print(f"\nStarting iteration {iteration + 1}/{self.iterations}")
                print("=" * 50)

                # Run parallel iteration using agent pool
                self.agent_pool.run_parallel_iteration(og_checkpoint, iteration)

                # Update total episodes count
                self.total_episodes_run += self.config["num_episodes"]

                # Combine and average models from all agents
                averaged_model = self.combine_parallel_agents(iteration, og_checkpoint)

                # Periodic agent pool recycling (every 10 iterations)
                if iteration > 0 and (iteration + 1) % 10 == 0:
                    print("\nRecycling agent pool to free resources...")
                    self.agent_pool.cleanup()
                    self.agent_pool = AgentPool(self.config, state_shape, num_actions)
                    print("Agent pool recycled successfully")

                # Force garbage collection periodically
                if (iteration + 1) % 5 == 0:
                    import gc

                    gc.collect()
                    print("Garbage collection completed")

                print(
                    f"Iteration {iteration + 1} completed. Total episodes run: {self.total_episodes_run}"
                )

            print("\nAll iterations completed.")
            return averaged_model

        finally:
            # Ensure environment is properly closed
            env.close()

            # Clean up agent pool
            if self.agent_pool:
                self.agent_pool.cleanup()

    def combine_parallel_agents(self, iteration, og_checkpoint):
        """Combine and average models from all agents"""
        agent_paths = [
            f"{self.config['checkpoint']}_{i}" for i in range(self.num_agents)
        ]

        env = Env.create_with_shared_temp(self.config)
        try:
            input_shape = env.output_shape()
            action_size = env.action_space.n

            # Average the models (only model parameters, not episode data)
            averaged_agent = self.average_models(agent_paths, input_shape, action_size)

            # Save only the model parameters to the shared checkpoint
            averaged_model_path = og_checkpoint
            self.save_model_only(averaged_agent, averaged_model_path)

            return averaged_agent.model
        finally:
            # Ensure environment is properly closed
            env.close()

    def average_models(self, agent_paths, input_shape, action_size):
        """Average only the model parameters from all agents"""
        averaged_agent = PPOAgent(input_shape, action_size, self.config)

        actor_critic_params = []
        icm_params = []
        optimizer_states = []
        scheduler_states = []
        agent_count = 0

        for path in agent_paths:
            if not os.path.exists(path):
                continue

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

        if agent_count == 0:
            print("No valid agent checkpoints found to average.")
            return averaged_agent

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

        # Set episode count
        averaged_agent.episode = self.total_episodes_run

        return averaged_agent

    def save_model_only(self, agent, path):
        """Save only the model parameters, not the episode data or other statistics"""
        os.makedirs(path, exist_ok=True)

        # Use file locking for safe checkpoint writing
        with self.resource_pool.file_lock(path):
            # Save only the model parameters
            torch.save(
                agent.model.actor_critic.state_dict(), f"{path}/actor_critic.pth"
            )
            torch.save(agent.model.optimizer.state_dict(), f"{path}/optimizer.pth")
            torch.save(agent.model.scheduler.state_dict(), f"{path}/scheduler.pth")
            agent.model.icm.save(f"{path}/icm")

            # Save minimal info.pth with just the episode count
            info = {
                "episode": agent.episode,
                "best_reward": 0,
                "episode_data": {},  # Empty episode data
            }
            torch.save(info, f"{path}/info.pth")

    def cleanup(self):
        """Clean up resources"""
        if self.agent_pool:
            self.agent_pool.cleanup()
        # Note: Don't cleanup shared model manager here as it may be used by other instances

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except Exception:
            pass
