# -*- coding: utf-8 -*-
import torch
import torch.multiprocessing as mp
import os
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.PPO import PPOAgent
from tqdm.auto import tqdm as auto_tqdm
import time


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
        # Create consistent paths for each agent that persist across iterations
        agent_checkpoint = f"{config['checkpoint']}_{i}"
        agent_record_path = f"{config['record_path']}_{i}"
        agent_export_state_loc = f"{config['export_state_loc']}_{i}"
        agent_results_dir = f"{config['results_dir']}_{i}"

        config["checkpoint"] = agent_checkpoint
        config["record_path"] = agent_record_path
        config["export_state_loc"] = agent_export_state_loc
        config["results_dir"] = agent_results_dir

        # Add worker-specific tqdm configuration
        config["tqdm_position"] = i
        config["tqdm_worker_id"] = i
        config["tqdm_desc_prefix"] = f"Agent {i}"

        agent = PPOAgent(state_shape, num_actions, config)

        # First, check if this agent has its own checkpoint and load it
        agent_specific_checkpoint_exists = os.path.exists(agent_checkpoint)
        if agent_specific_checkpoint_exists:
            agent.load_model(agent_checkpoint)

        # Then, if a shared model exists, only replace the model parameters
        shared_model_exists = os.path.exists(og_checkpoint)
        if shared_model_exists:
            self.update_agent_model_only(agent, og_checkpoint)
        if config["use_curriculum"]:
            agent.run_curriculum(1, config["N_goals_target"], ["N_goals_increment"])
        else:
            agent.train_agent()

    def update_agent_model_only(self, agent, shared_checkpoint_path):
        """
        Update only the model parameters of an agent from a shared checkpoint,
        preserving all other agent data (episode data, statistics, etc.)
        """
        try:
            # Load the shared model parameters
            shared_actor_critic = torch.load(
                f"{shared_checkpoint_path}/actor_critic.pth",
                map_location=agent.device,
                weights_only=True,
            )
            shared_optimizer = torch.load(
                f"{shared_checkpoint_path}/optimizer.pth",
                map_location=agent.device,
                weights_only=True,
            )
            shared_scheduler = torch.load(
                f"{shared_checkpoint_path}/scheduler.pth",
                map_location=agent.device,
                weights_only=True,
            )

            # Update only the model parameters
            agent.model.actor_critic.load_state_dict(shared_actor_critic)
            agent.model.optimizer.load_state_dict(shared_optimizer)
            agent.model.scheduler.load_state_dict(shared_scheduler)
            agent.model.icm.load(f"{shared_checkpoint_path}/icm")

            return True
        except Exception as e:
            print(f"Error updating agent model: {e}")
            return False

    def average_models(self, agent_paths, input_shape, action_size):
        """
        Average only the model parameters from all agents.
        Does not combine or store episode data.
        """
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
        """
        Save only the model parameters, not the episode data or other statistics.
        """
        os.makedirs(path, exist_ok=True)

        # Save only the model parameters
        torch.save(agent.model.actor_critic.state_dict(), f"{path}/actor_critic.pth")
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

    def combine_parallel_agents(self, iteration, og_checkpoint):
        agent_paths = [
            f"{self.config['checkpoint']}_{i}" for i in range(self.num_agents)
        ]

        env = Env(self.config)
        input_shape = env.output_shape()
        action_size = env.action_space.n

        # Average the models (only model parameters, not episode data)
        averaged_agent = self.average_models(agent_paths, input_shape, action_size)

        # Save only the model parameters to the shared checkpoint
        averaged_model_path = og_checkpoint
        self.save_model_only(averaged_agent, averaged_model_path)

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

            # Instead of using pool.starmap, we'll create and manage processes manually
            # to implement the timeout mechanism
            processes = []
            process_start_times = {}
            finished_agents = set()
            first_agent_finished_time = None
            timeout_seconds = 30  # 30 second timeout after first agent finishes

            # Create and start processes for each agent
            for i in range(self.num_agents):
                process = mp.Process(
                    target=self.run_agent,
                    args=(i, state_shape, num_actions, og_checkpoint),
                )
                process.start()
                processes.append(process)
                process_start_times[process.pid] = time.time()

            # Monitor processes and implement timeout
            all_finished = False
            while not all_finished:
                all_finished = True
                for i, process in enumerate(processes):
                    if process is None:
                        continue  # Already handled this process

                    if not process.is_alive():
                        # Process has finished
                        if process.pid not in finished_agents:
                            finished_agents.add(process.pid)

                            # Record when the first agent finishes
                            if first_agent_finished_time is None:
                                first_agent_finished_time = time.time()
                    else:
                        # Process is still running
                        all_finished = False

                        # Check if we need to kill this process due to timeout
                        if (
                            first_agent_finished_time is not None
                            and time.time() - first_agent_finished_time
                            > timeout_seconds
                        ):

                            # Calculate how long this process has been running
                            run_time = time.time() - process_start_times[process.pid]

                            print(
                                f"Agent {i} exceeded timeout ({run_time:.2f} seconds). Terminating process."
                            )

                            # Try to get information about what the process is doing
                            try:
                                import psutil

                                p = psutil.Process(process.pid)
                                print(f"Process was executing: {p.cmdline()}")
                                print(f"Process status: {p.status()}")

                                # On Unix systems, we might be able to get more detailed info
                                if hasattr(p, "open_files"):
                                    open_files = p.open_files()
                                    if open_files:
                                        print(f"Open files: {open_files}")

                            except Exception as e:
                                print(f"Could not get process details: {e}")

                            # Kill the process
                            process.terminate()

                            # Wait a bit for termination, then force kill if needed
                            try:
                                process.join(
                                    5
                                )  # Give it 5 seconds to terminate gracefully
                                if process.is_alive():
                                    print(
                                        "Process didn't terminate gracefully. Force killing."
                                    )
                                    process.kill()
                            except Exception as e:
                                print(f"Error killing process: {e}")

                            processes[i] = None  # Mark as handled

                # Small sleep to prevent CPU hogging
                time.sleep(0.1)

            self.total_episodes_run += self.config["num_episodes"]
            averaged_model = self.combine_parallel_agents(iteration, og_checkpoint)
            print(
                f"Iteration {iteration + 1} completed. Total episodes run: {self.total_episodes_run}"
            )

        print("\nAll iterations completed.")
        return averaged_model
