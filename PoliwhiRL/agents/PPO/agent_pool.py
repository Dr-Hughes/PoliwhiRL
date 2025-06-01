# -*- coding: utf-8 -*-
import os
import torch
import torch.multiprocessing as mp
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.PPO import PPOAgent
import time
import signal
import atexit
import gc
from PoliwhiRL.utils.resource_manager import get_resource_pool, ProcessMonitor

# Set multiprocessing start method to spawn to avoid pickle issues
if mp.get_start_method(allow_none=True) != "spawn":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set, continue
        pass


class AgentPool:
    """Manages a pool of persistent PPO agents for multi-agent training"""

    def __init__(self, config, state_shape, num_actions):
        self.config = config
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.num_agents = config["ppo_num_agents"]
        self.agents = []
        self.processes = []
        self.shared_temp_dir = None
        self.is_initialized = False
        self.process_monitor = ProcessMonitor()
        self.resource_pool = get_resource_pool()

        # Register cleanup
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle signals to ensure cleanup"""
        self.cleanup()
        exit(1)

    def initialize_pool(self):
        """Initialize the agent pool with persistent agents"""
        if self.is_initialized:
            return

        print(f"Initializing agent pool with {self.num_agents} agents...")

        # Create shared temporary directory for all agents
        env = Env.create_with_shared_temp(self.config)
        self.shared_temp_dir = env.temp_dir
        env.close()

        # Create agent configurations
        for i in range(self.num_agents):
            agent_config = self.config.copy()

            # Create consistent paths for each agent
            agent_checkpoint = f"{agent_config['checkpoint']}_{i}"
            agent_record_path = f"{agent_config['record_path']}_{i}"
            agent_export_state_loc = f"{agent_config['export_state_loc']}_{i}"
            agent_results_dir = f"{agent_config['results_dir']}_{i}"

            agent_config["record"] = False if i != 0 else agent_config["record"]
            agent_config["checkpoint"] = agent_checkpoint
            agent_config["record_path"] = agent_record_path
            agent_config["export_state_loc"] = agent_export_state_loc
            agent_config["results_dir"] = agent_results_dir
            agent_config["tqdm_position"] = i
            agent_config["tqdm_worker_id"] = i
            agent_config["tqdm_desc_prefix"] = f"Agent {i}"
            agent_config["shared_temp_dir"] = self.shared_temp_dir

            self.agents.append(
                {
                    "id": i,
                    "config": agent_config,
                    "process": None,
                    "manager": None,
                    "agent_instance": None,
                }
            )

        self.is_initialized = True
        print("Agent pool initialized successfully")

    def create_persistent_agent(self, agent_info, og_checkpoint):
        """Create a persistent agent that can be reused across iterations"""
        agent_id = agent_info["id"]
        config = agent_info["config"]

        print(f"Creating persistent agent {agent_id}")

        # Create environment with shared temp directory
        env = Env(config, shared_temp_dir=self.shared_temp_dir)

        try:
            # Create agent
            agent = PPOAgent(self.state_shape, self.num_actions, config)

            # Load existing checkpoint if available
            agent_checkpoint = config["checkpoint"]
            if os.path.exists(agent_checkpoint):
                agent.load_model(agent_checkpoint)

            # Load shared model if available
            if os.path.exists(og_checkpoint):
                AgentPool.update_agent_model_only_static(agent, og_checkpoint)

            # Load curriculum checkpoint if specified in config
            if config.get("load_checkpoint") and config["load_checkpoint"] != "":
                curriculum_checkpoint = config["load_checkpoint"]
                if os.path.exists(curriculum_checkpoint):
                    print(
                        f"Agent {agent_id} loading curriculum checkpoint from: {curriculum_checkpoint}"
                    )
                    agent.load_model(curriculum_checkpoint)
                else:
                    print(
                        f"Warning: Curriculum checkpoint not found at {curriculum_checkpoint}"
                    )

            agent_info["agent_instance"] = agent
            agent_info["env"] = env

            return agent

        except Exception as e:
            print(f"Error creating agent {agent_id}: {e}")
            env.close()
            raise

    def reset_agent_for_iteration(self, agent_info, og_checkpoint):
        """Reset an existing agent for a new iteration"""
        agent_id = agent_info["id"]
        agent = agent_info["agent_instance"]

        print(f"Resetting agent {agent_id} for new iteration")

        try:
            # Reset agent state
            agent.reset_tracking()

            # Update model from shared checkpoint if available
            if os.path.exists(og_checkpoint):
                self.update_agent_model_only(agent, og_checkpoint)

            return agent

        except Exception as e:
            print(f"Error resetting agent {agent_id}: {e}")
            raise

    @staticmethod
    def run_agent_iteration_wrapper(
        serializable_info,
        og_checkpoint,
        is_first_iteration,
        state_shape,
        num_actions,
        process_monitor_queue,
    ):
        """Wrapper method that creates agent within subprocess to avoid pickle issues"""
        agent_id = serializable_info["id"]
        config = serializable_info["config"]

        # Disable tqdm in subprocess to avoid weak reference issues
        config["report_episode"] = False

        print(f"Starting agent {agent_id} in subprocess (PID: {os.getpid()})")

        # Send process info to monitor
        process_monitor_queue.put(("register", agent_id, os.getpid()))

        agent = None
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Add small delay to prevent race conditions
                import time

                time.sleep(agent_id * 0.1)  # Stagger agent starts

                # Create agent within subprocess
                agent = PPOAgent(state_shape, num_actions, config)

                # Load existing checkpoint if available
                agent_checkpoint = config["checkpoint"]
                if os.path.exists(agent_checkpoint):
                    print(
                        f"Agent {agent_id} loading checkpoint from: {agent_checkpoint}"
                    )
                    agent.load_model(agent_checkpoint)

                # Load shared model if available
                if os.path.exists(og_checkpoint):
                    print(
                        f"Agent {agent_id} loading shared model from: {og_checkpoint}"
                    )
                    AgentPool.update_agent_model_only_static(agent, og_checkpoint)

                # Load curriculum checkpoint if specified in config
                if config.get("load_checkpoint") and config["load_checkpoint"] != "":
                    curriculum_checkpoint = config["load_checkpoint"]
                    if os.path.exists(curriculum_checkpoint):
                        print(
                            f"Agent {agent_id} loading curriculum checkpoint from: {curriculum_checkpoint}"
                        )
                        agent.load_model(curriculum_checkpoint)
                    else:
                        print(
                            f"Warning: Curriculum checkpoint not found at {curriculum_checkpoint}"
                        )

                # Run training
                print(f"Agent {agent_id} starting training...")

                if config["use_curriculum"]:
                    agent.run_curriculum(
                        1, config["N_goals_target"], config["N_goals_increment"]
                    )
                else:
                    agent.train_agent()

                print(f"Agent {agent_id} completed training successfully")
                break  # Success, exit retry loop

            except Exception as e:
                retry_count += 1
                print(
                    f"Error during agent {agent_id} training (attempt {retry_count}/{max_retries}): {e}"
                )

                if retry_count < max_retries:
                    print(f"Agent {agent_id} retrying in {retry_count} seconds...")
                    time.sleep(retry_count)  # Exponential backoff

                    # Cleanup failed agent
                    if agent is not None:
                        try:
                            del agent
                            agent = None
                        except Exception:
                            pass
                    gc.collect()
                else:
                    print(f"Agent {agent_id} failed after {max_retries} attempts")
                    import traceback

                    traceback.print_exc()
                    raise

        # Cleanup in subprocess
        if agent is not None:
            try:
                # Force cleanup of any resources
                del agent
            except Exception:
                pass

        # Force garbage collection
        gc.collect()
        print(f"Agent {agent_id} subprocess cleanup completed")

    def run_agent_iteration(self, agent_info, og_checkpoint, is_first_iteration=False):
        """Run a single iteration for an agent (legacy method for non-multiprocessing use)"""
        if is_first_iteration or agent_info["agent_instance"] is None:
            agent = self.create_persistent_agent(agent_info, og_checkpoint)
        else:
            agent = self.reset_agent_for_iteration(agent_info, og_checkpoint)

        config = agent_info["config"]

        try:
            if config["use_curriculum"]:
                agent.run_curriculum(
                    1, config["N_goals_target"], config["N_goals_increment"]
                )
            else:
                agent.train_agent()

        except Exception as e:
            print(f"Error during agent {agent_info['id']} training: {e}")
            raise

    def run_parallel_iteration(self, og_checkpoint, iteration_num=0):
        """Run a parallel iteration using the agent pool"""
        if not self.is_initialized:
            self.initialize_pool()

        is_first_iteration = iteration_num == 0
        processes = []

        # Create queue for process monitoring
        process_monitor_queue = mp.Queue()

        print(
            f"Starting parallel iteration {iteration_num} with {self.num_agents} agents"
        )

        # Start processes for each agent
        for agent_info in self.agents:
            # Only pass serializable data to the subprocess
            serializable_info = {
                "id": agent_info["id"],
                "config": agent_info["config"],
            }
            process = mp.Process(
                target=self.run_agent_iteration_wrapper,
                args=(
                    serializable_info,
                    og_checkpoint,
                    is_first_iteration,
                    self.state_shape,
                    self.num_actions,
                    process_monitor_queue,
                ),
            )
            process.start()
            processes.append(process)
            agent_info["process"] = process

        # Monitor processes with dynamic timeout
        self._monitor_processes_with_dynamic_timeout(processes, process_monitor_queue)

        print(f"Parallel iteration {iteration_num} completed")

    def _monitor_processes_with_dynamic_timeout(self, processes, monitor_queue):
        """Monitor processes with dynamic timeout based on progress"""
        finished_agents = set()
        first_agent_finished_time = None
        base_timeout = self.config.get("agent_timeout", 180)
        start_time = time.time()
        health_check_delay = 5  # Wait 5 seconds before starting health checks

        while True:
            all_finished = True

            # Process monitor queue messages
            while not monitor_queue.empty():
                try:
                    msg = monitor_queue.get_nowait()
                    if msg[0] == "register":
                        _, agent_id, pid = msg
                        self.process_monitor.register_process(agent_id, pid)
                    elif msg[0] == "heartbeat":
                        _, agent_id, episodes = msg
                        self.process_monitor.update_heartbeat(agent_id, episodes)
                except Exception:
                    pass

            for i, process in enumerate(processes):
                if process is None:
                    continue

                if not process.is_alive():
                    if process.pid not in finished_agents:
                        finished_agents.add(process.pid)
                        if first_agent_finished_time is None:
                            first_agent_finished_time = time.time()
                        print(f"Agent {i} completed training")
                else:
                    all_finished = False

                    # Only start health checks after initial delay
                    if time.time() - start_time > health_check_delay:
                        # Update resource usage
                        self.process_monitor.update_resource_usage(i)

                        # Check agent health
                        if not self.process_monitor.check_agent_health(i):
                            print(f"Agent {i} unhealthy. Terminating...")
                            process.terminate()
                            try:
                                process.join(5)
                                if process.is_alive():
                                    process.kill()
                            except Exception as e:
                                print(f"Error terminating unhealthy process: {e}")
                            processes[i] = None
                            continue

                    # Check dynamic timeout
                    if first_agent_finished_time is not None:
                        dynamic_timeout = self.process_monitor.get_timeout_for_agent(
                            i, base_timeout
                        )
                        if time.time() - first_agent_finished_time > dynamic_timeout:
                            print(
                                f"Agent {i} exceeded dynamic timeout ({dynamic_timeout}s). Terminating..."
                            )
                            process.terminate()
                            try:
                                process.join(5)
                                if process.is_alive():
                                    process.kill()
                            except Exception as e:
                                print(f"Error terminating process: {e}")
                            processes[i] = None

            if all_finished:
                break

            # Print status every 10 seconds
            if int(time.time()) % 10 == 0:
                status = self.process_monitor.get_all_agents_status()
                if status:
                    print("\nAgent Status:")
                    for agent_id, info in status.items():
                        if info["is_alive"]:
                            print(
                                f"  Agent {agent_id}: Episodes={info['episodes']}, "
                                + f"Memory={info['memory_mb']:.1f}MB, CPU={info['cpu_percent']:.1f}%"
                            )

            time.sleep(0.1)

    @staticmethod
    def update_agent_model_only_static(agent, shared_checkpoint_path):
        """Static method to update only the model parameters of an agent from a shared checkpoint"""
        try:
            resource_pool = get_resource_pool()

            # Use file locking for safe checkpoint reading
            with resource_pool.file_lock(shared_checkpoint_path):
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

                agent.model.actor_critic.load_state_dict(shared_actor_critic)
                agent.model.optimizer.load_state_dict(shared_optimizer)
                agent.model.scheduler.load_state_dict(shared_scheduler)
                agent.model.icm.load(f"{shared_checkpoint_path}/icm")

            return True
        except Exception as e:
            print(f"Error updating agent model: {e}")
            return False

    def update_agent_model_only(self, agent, shared_checkpoint_path):
        """Update only the model parameters of an agent from a shared checkpoint"""
        return AgentPool.update_agent_model_only_static(agent, shared_checkpoint_path)

    def cleanup(self):
        """Clean up all agents and processes"""
        print("Cleaning up agent pool...")

        # Terminate any running processes
        for agent_info in self.agents:
            if agent_info.get("process") and agent_info["process"].is_alive():
                try:
                    agent_info["process"].terminate()
                    agent_info["process"].join(5)
                    if agent_info["process"].is_alive():
                        agent_info["process"].kill()
                except Exception as e:
                    print(f"Error terminating agent process: {e}")

        # Close environments
        for agent_info in self.agents:
            if agent_info.get("env"):
                try:
                    agent_info["env"].close()
                except Exception as e:
                    print(f"Error closing agent environment: {e}")

        self.agents.clear()
        self.processes.clear()
        print("Agent pool cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except Exception:
            pass
