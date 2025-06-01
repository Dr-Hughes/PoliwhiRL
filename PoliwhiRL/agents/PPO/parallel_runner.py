# -*- coding: utf-8 -*-
import os
import torch
import torch.multiprocessing as mp
import numpy as np
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.PPO import PPOAgent
from tqdm.auto import tqdm as auto_tqdm


def run_single_agent(
    agent_id,
    config,
    shared_model_state,
    icm_state,
    iteration,
    base_checkpoint,
    cumulative_episodes,
):
    """
    Run a single PPO agent training episode.
    This function runs in a separate process.
    """
    import time
    import traceback

    # Create hang detection log file
    hang_log_dir = f"{config.get('results_dir', 'stage_1')}/hang_logs"
    os.makedirs(hang_log_dir, exist_ok=True)
    hang_log_file = f"{hang_log_dir}/agent_{agent_id}_hang_detection.log"

    def log_progress(stage, details=""):
        """Log progress to detect where hangs occur"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(hang_log_file, "a") as f:
            f.write(f"[{timestamp}] Agent {agent_id} - {stage}: {details}\n")

    try:
        log_progress(
            "STARTING", f"iteration {iteration}, episode {cumulative_episodes}"
        )

        # Update config for this specific agent
        agent_config = config.copy()
        agent_config["checkpoint"] = f"{base_checkpoint}_{agent_id}"
        agent_config["record_path"] = f"{config['record_path']}_{agent_id}"
        agent_config["export_state_loc"] = f"{config['export_state_loc']}_{agent_id}"
        agent_config["results_dir"] = f"{config['results_dir']}_{agent_id}"
        agent_config["record"] = False if agent_id != 0 else config["record"]
        agent_config["report_episode"] = False  # Disable tqdm in subprocess
        # Set the starting episode number to continue from previous iterations
        agent_config["start_episode"] = cumulative_episodes
        # Set the worker ID for debug prints
        agent_config["tqdm_worker_id"] = agent_id

        log_progress("CONFIG_SETUP", "Agent configuration prepared")
        print(
            f"Agent {agent_id} starting (iteration {iteration}, starting from episode {cumulative_episodes})"
        )

        # Create environment
        log_progress("ENV_CREATING", "About to create environment")
        env = Env(agent_config)
        log_progress("ENV_CREATED", "Environment created successfully")

        log_progress("AGENT_SETUP", "Getting environment state shape")
        state_shape = env.output_shape()
        num_actions = env.action_space.n

        # Create agent
        log_progress("AGENT_CREATING", f"Creating PPO agent with shape {state_shape}")
        agent = PPOAgent(state_shape, num_actions, agent_config)
        log_progress("AGENT_CREATED", "PPO agent created successfully")

        # Load agent's own checkpoint first (for episode data)
        log_progress("CHECKPOINT_LOADING", "Loading agent checkpoint for episode data")
        agent_checkpoint = agent_config["checkpoint"]
        if os.path.exists(agent_checkpoint):
            print(
                f"Agent {agent_id} loading checkpoint for episode data: {agent_checkpoint}"
            )
            agent.load_model(agent_checkpoint)
            log_progress(
                "CHECKPOINT_LOADED", f"Loaded checkpoint from {agent_checkpoint}"
            )

        # Then override with shared model state if provided (preserving episode data)
        if shared_model_state is not None:
            log_progress("SHARED_MODEL_LOADING", "Loading shared model state")
            print(f"Agent {agent_id} loading shared model state (overriding weights)")
            agent.model.actor_critic.load_state_dict(shared_model_state)
            log_progress("SHARED_MODEL_LOADED", "Shared model state loaded")
        if icm_state is not None:
            agent.model.icm.icm.load_state_dict(icm_state)
            log_progress("ICM_LOADED", "ICM state loaded")

        # Only load curriculum checkpoint if we don't have a shared model state
        # (This handles the case where the first iteration failed to load the curriculum checkpoint)
        if (
            shared_model_state is None
            and agent_config.get("load_checkpoint")
            and agent_config["load_checkpoint"] != ""
        ):
            curriculum_checkpoint = agent_config["load_checkpoint"]
            if os.path.exists(curriculum_checkpoint):
                print(
                    f"Agent {agent_id} loading curriculum checkpoint: {curriculum_checkpoint}"
                )
                agent.load_model(curriculum_checkpoint)

        # Run training
        log_progress(
            "TRAINING_START",
            f"Starting training - curriculum: {agent_config['use_curriculum']}",
        )
        if agent_config["use_curriculum"]:
            agent.run_curriculum(
                1, agent_config["N_goals_target"], agent_config["N_goals_increment"]
            )
        else:
            agent.train_agent()

        log_progress("TRAINING_COMPLETE", "Training completed successfully")
        print(f"Agent {agent_id} completed successfully")

        # Save model to checkpoint instead of returning large state dicts
        log_progress("SAVING_CHECKPOINT", f"Saving to {agent_config['checkpoint']}")
        print(f"Agent {agent_id} saving checkpoint to {agent_config['checkpoint']}")
        agent.save_model(agent_config["checkpoint"])
        log_progress("CHECKPOINT_SAVED", "Checkpoint saved successfully")

        # Return only performance metrics and checkpoint path (much smaller data)
        log_progress("PREPARING_RESULTS", "Gathering performance metrics")
        recent_rewards = agent.episode_data.get("episode_rewards", [])[-5:]
        recent_lengths = agent.episode_data.get("episode_lengths", [])[-5:]

        result = {
            "checkpoint_path": agent_config["checkpoint"],
            "episode": agent.episode,
            "performance_metrics": {
                "recent_rewards": recent_rewards,
                "recent_lengths": recent_lengths,
                "avg_reward": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
                "avg_length": float(np.mean(recent_lengths)) if recent_lengths else 0.0,
            },
        }

        log_progress(
            "SUCCESS",
            f"Agent completed - avg reward: {result['performance_metrics']['avg_reward']:.3f}",
        )
        return result

    except Exception as e:
        log_progress("ERROR", f"Exception occurred: {str(e)}")
        print(f"Agent {agent_id} failed with error: {e}")
        error_trace = traceback.format_exc()
        log_progress("TRACEBACK", error_trace)
        traceback.print_exc()
        return None
    finally:
        log_progress("CLEANUP", "Closing environment and finishing")
        env.close()


class PPOParallelRunner:
    """
    Simple parallel runner for PPO agents using multiprocessing.Pool.
    Based on the DQN ParallelAgentRunner pattern.
    """

    def __init__(self, config):
        self.config = config
        self.num_agents = config["ppo_num_agents"]
        self.iterations = config["ppo_iterations"]
        self.base_checkpoint = config["checkpoint"]
        self.total_episodes_run = config["start_episode"]

        # Elite tracking for better multi-agent coordination
        self.elite_performance = float("-inf")
        self.elite_checkpoint = None
        self.performance_history = []

        # Set multiprocessing start method and configure for stability
        if mp.get_start_method(allow_none=True) != "spawn":
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

        # Check memory if psutil is available
        try:
            import psutil

            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            recommended_agents = min(
                self.num_agents, max(1, int(available_memory_gb / 2))
            )  # ~2GB per agent
            if recommended_agents < self.num_agents:
                print(
                    f"⚠️  Memory warning: {available_memory_gb:.1f}GB available, "
                    f"consider reducing agents from {self.num_agents} to {recommended_agents}"
                )
        except ImportError:
            print("💡 Install psutil for memory monitoring: pip install psutil")

        print(
            f"🔧 Multiprocessing configured: {mp.get_start_method()} method, {self.num_agents} agents"
        )

        # Create hang logs directory
        self.hang_summary_file = (
            f"{self.config.get('results_dir', 'stage_1')}/hang_summary.log"
        )

    def _log_agent_hang(self, agent_id, iteration, reason):
        """Log agent hang for analysis"""
        import time

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        hang_details = (
            f"[{timestamp}] Agent {agent_id} - Iteration {iteration} - {reason}\n"
        )

        try:
            with open(self.hang_summary_file, "a") as f:
                f.write(hang_details)
        except Exception as e:
            print(f"Warning: Could not write hang log: {e}")

    def get_shared_model_state(self, iteration):
        """Get the current shared model state if it exists"""
        # For iteration 0, try to load from the curriculum checkpoint first
        if (
            iteration == 0
            and self.config.get("load_checkpoint")
            and self.config["load_checkpoint"] != ""
        ):
            curriculum_checkpoint = self.config["load_checkpoint"]
            if os.path.exists(curriculum_checkpoint):
                try:
                    print(
                        f"Loading initial shared model from curriculum checkpoint: {curriculum_checkpoint}"
                    )
                    device = torch.device("cpu")
                    actor_critic_state = torch.load(
                        f"{curriculum_checkpoint}/actor_critic.pth",
                        map_location=device,
                        weights_only=True,
                    )
                    icm_state = torch.load(
                        f"{curriculum_checkpoint}/icm_icm.pth",
                        map_location=device,
                        weights_only=True,
                    )
                    return actor_critic_state, icm_state
                except Exception as e:
                    print(f"Could not load curriculum checkpoint: {e}")

        # For subsequent iterations or if no curriculum checkpoint, use current stage checkpoint
        if os.path.exists(self.base_checkpoint):
            try:
                print(
                    f"Loading shared model from current stage: {self.base_checkpoint}"
                )
                device = torch.device("cpu")
                actor_critic_state = torch.load(
                    f"{self.base_checkpoint}/actor_critic.pth",
                    map_location=device,
                    weights_only=True,
                )
                icm_state = torch.load(
                    f"{self.base_checkpoint}/icm_icm.pth",
                    map_location=device,
                    weights_only=True,
                )
                return actor_critic_state, icm_state
            except Exception as e:
                print(f"Could not load shared model state: {e}")
                return None, None
        return None, None

    def train(self):
        """Main training loop"""
        print(f"Starting PPO parallel training with {self.num_agents} agents")
        print(f"Running {self.iterations} iterations")

        # Get initial environment shape
        env = Env(self.config)
        try:
            state_shape = env.output_shape()
            num_actions = env.action_space.n
        finally:
            env.close()

        # Training loop
        for iteration in auto_tqdm(range(self.iterations), desc="Iterations"):
            print(f"\n{'=' * 60}")
            print(f"Starting iteration {iteration + 1}/{self.iterations}")
            print(f"{'=' * 60}")

            # Get current shared model state
            shared_model_state, icm_state = self.get_shared_model_state(iteration)

            # Run agents in parallel with timeout and better error handling
            try:
                with mp.Pool(processes=self.num_agents) as pool:
                    # Submit individual jobs for better tracking
                    futures = []
                    for i in range(self.num_agents):
                        future = pool.apply_async(
                            run_single_agent,
                            (
                                i,
                                self.config,
                                shared_model_state,
                                icm_state,
                                iteration,
                                self.base_checkpoint,
                                self.total_episodes_run,
                            ),
                        )
                        futures.append((i, future))

                    # Collect results with individual timeouts
                    timeout_minutes = 8  # Per-agent timeout
                    agent_timeout = timeout_minutes * 60
                    print(
                        f"⏰ Waiting for agents to complete (timeout: {timeout_minutes}min per agent)"
                    )

                    results = []
                    for i, future in futures:
                        try:
                            result = future.get(timeout=agent_timeout)
                            results.append(result)
                            print(f"✅ Agent {i} completed successfully")
                        except mp.TimeoutError:
                            print(
                                f"⏰ Agent {i} timed out after {timeout_minutes} minutes - skipping"
                            )
                            # Log the hang for later analysis
                            self._log_agent_hang(i, iteration, "TIMEOUT")
                            results.append(None)
                        except Exception as e:
                            print(f"❌ Agent {i} failed: {e}")
                            self._log_agent_hang(i, iteration, f"EXCEPTION: {e}")
                            results.append(None)

            except Exception as e:
                print(f"⚠️  Error in multiprocessing pool: {e}")
                try:
                    pool.terminate()
                    pool.join()
                except Exception:
                    pass
                raise

            # Filter out failed agents
            successful_results = [r for r in results if r is not None]

            if not successful_results:
                print("WARNING: All agents failed in this iteration!")
                continue

            print(
                f"\n{len(successful_results)}/{self.num_agents} agents completed successfully"
            )

            # Update episode count
            self.total_episodes_run += self.config["num_episodes"]

            # Intelligent model aggregation and save
            self.smart_model_aggregation(successful_results, state_shape, num_actions)

            print(
                f"\nIteration {iteration + 1} completed. Total episodes: {self.total_episodes_run}"
            )

        print("\nTraining completed!")

    def calculate_composite_score(self, result):
        """Multi-metric performance assessment"""
        metrics = result.get("performance_metrics", {})
        if not metrics:
            return 0.0

        # Recent performance
        avg_reward = metrics.get("avg_reward", 0.0)

        # Efficiency metrics (shorter episodes = more efficient)
        avg_length = metrics.get("avg_length", 0.0)
        if avg_length > 0:
            max_length = float(self.config["episode_length"])
            efficiency = 1.0 - (avg_length / max_length)
        else:
            efficiency = 0.0

        # Stability (lower variance in rewards is better)
        recent_rewards = metrics.get("recent_rewards", [])
        if len(recent_rewards) > 1:
            reward_variance = float(np.var(recent_rewards))
            reward_stability = 1.0 / (1.0 + reward_variance)
        else:
            reward_stability = 0.5

        # Composite score with weights
        score = avg_reward + efficiency * 0.5 + reward_stability * 0.3
        return score

    def performance_weighted_average(self, scored_results, state_shape, num_actions):
        """Weight parameter contributions by agent performance - load from checkpoints"""
        results = [result for _, result in scored_results]
        performances = [
            max(score, 0.1) for score, _ in scored_results
        ]  # Minimum weight

        # Normalize weights
        total_performance = sum(performances)
        if total_performance == 0:
            # Fallback to uniform weights
            weights = [1.0 / len(performances)] * len(performances)
        else:
            weights = [p / total_performance for p in performances]

        print(f"Performance weights: {[f'{w:.3f}' for w in weights]}")

        # Load agents from checkpoints
        agents = []
        for result in results:
            checkpoint_path = result["checkpoint_path"]
            if os.path.exists(checkpoint_path):
                agent = PPOAgent(state_shape, num_actions, self.config)
                agent.load_model(checkpoint_path)
                agents.append(agent)
            else:
                print(f"Warning: Checkpoint {checkpoint_path} not found")
                return None

        if not agents:
            print("Error: No valid checkpoints found for averaging")
            return None

        # Create temporary agent for averaging
        temp_agent = PPOAgent(state_shape, num_actions, self.config)
        device = temp_agent.device

        # Weighted averaging of actor_critic parameters
        for name, param in temp_agent.model.actor_critic.named_parameters():
            weighted_param = torch.zeros_like(param)
            for i, agent in enumerate(agents):
                agent_param = agent.model.actor_critic.get_parameter(name)
                if agent_param.device != device:
                    agent_param = agent_param.to(device)
                weighted_param += weights[i] * agent_param
            param.data.copy_(weighted_param)

        # Weighted averaging of ICM parameters
        for name, param in temp_agent.model.icm.icm.named_parameters():
            weighted_param = torch.zeros_like(param)
            for i, agent in enumerate(agents):
                agent_param = agent.model.icm.icm.get_parameter(name)
                if agent_param.device != device:
                    agent_param = agent_param.to(device)
                weighted_param += weights[i] * agent_param
            param.data.copy_(weighted_param)

        # Use best performing agent's optimizer and scheduler state
        best_agent = agents[0]  # Already sorted by performance
        temp_agent.model.optimizer.load_state_dict(
            best_agent.model.optimizer.state_dict()
        )
        temp_agent.model.scheduler.load_state_dict(
            best_agent.model.scheduler.state_dict()
        )

        # Set episode count
        temp_agent.episode = self.total_episodes_run

        return temp_agent

    def top_k_average(self, scored_results, state_shape, num_actions, k=None):
        """Average only the top-k performing agents"""
        if k is None:
            k = max(2, len(scored_results) // 2)  # Top 50%

        top_results = [result for _, result in scored_results[:k]]
        print(f"Averaging top {len(top_results)}/{len(scored_results)} agents")

        return self.average_models(top_results, state_shape, num_actions)

    def smart_model_aggregation(self, results, state_shape, num_actions):
        """Intelligent model aggregation strategy"""
        import numpy as np

        if not results:
            print("Warning: No successful results to aggregate!")
            return

        # Calculate performance scores
        scored_results = []
        for result in results:
            score = self.calculate_composite_score(result)
            scored_results.append((score, result))

        # Sort by performance (best first)
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Update elite tracking
        best_score, best_result = scored_results[0]
        if best_score > self.elite_performance:
            self.elite_performance = best_score
            self.elite_checkpoint = best_result["checkpoint_path"]
            print(
                f"🏆 New elite agent! Score: {best_score:.3f} (Previous: {self.elite_performance:.3f})"
            )

        # Store performance history
        current_scores = [score for score, _ in scored_results]
        self.performance_history.append(
            {
                "iteration": len(self.performance_history),
                "best_score": best_score,
                "worst_score": current_scores[-1],
                "avg_score": np.mean(current_scores),
                "score_std": np.std(current_scores),
            }
        )

        # Determine aggregation strategy based on performance distribution
        score_std = np.std(current_scores)
        score_range = current_scores[0] - current_scores[-1]

        print(
            f"Performance distribution - Best: {best_score:.3f}, Worst: {current_scores[-1]:.3f}, "
            f"Std: {score_std:.3f}, Range: {score_range:.3f}"
        )

        # Choose aggregation strategy
        if score_std < 0.3 and len(results) > 3:
            # Low variance - use top-k averaging to avoid averaging too many similar agents
            top_k = max(2, len(results) // 3)
            print(f"📊 Low variance detected - using top-{top_k} averaging")
            averaged_agent = self.top_k_average(
                scored_results, state_shape, num_actions, k=top_k
            )

        elif score_range > 1.0:
            # High variance - use performance weighting to heavily favor good agents
            print("📊 High variance detected - using performance-weighted averaging")
            averaged_agent = self.performance_weighted_average(
                scored_results, state_shape, num_actions
            )

        else:
            # Moderate variance - use top half with equal weights
            top_half = max(2, len(results) // 2)
            print(f"📊 Moderate variance - using top-{top_half} equal averaging")
            averaged_agent = self.top_k_average(
                scored_results, state_shape, num_actions, k=top_half
            )

        # Save the aggregated model
        print(f"💾 Saving aggregated model to {self.base_checkpoint}")
        averaged_agent.save_model(self.base_checkpoint)

        # Keep individual agent checkpoints for episode data persistence
        print(
            f"📈 Keeping {len(results)} individual agent checkpoints for episode data"
        )

        # Log entropy information for monitoring
        self.log_entropy_status(averaged_agent)

    def log_entropy_status(self, agent):
        """Log current entropy coefficient for monitoring"""
        current_entropy = agent.model._get_entropy_coef(self.total_episodes_run)
        initial_entropy = agent.model.entropy_coef
        min_entropy = agent.model.entropy_min

        print(
            f"🎲 Entropy Status - Current: {current_entropy:.4f}, "
            f"Initial: {initial_entropy:.4f}, Min: {min_entropy:.4f}, "
            f"Episode: {self.total_episodes_run}"
        )

        # Warn if entropy is getting very low
        if current_entropy <= min_entropy * 1.1:
            print(
                "⚠️  Warning: Entropy approaching minimum - agent may lose exploration capability!"
            )

        # Calculate episodes until minimum entropy
        if current_entropy > min_entropy:
            decay_rate = agent.model.entropy_decay
            episodes_to_min = np.log(min_entropy / initial_entropy) / np.log(decay_rate)
            remaining_episodes = max(0, episodes_to_min - self.total_episodes_run)
            print(
                f"📉 Entropy will reach minimum in ~{remaining_episodes:.0f} episodes"
            )

    def average_models(self, results, state_shape, num_actions):
        """Average model parameters from results and return agent - load from checkpoints"""
        # Load agents from checkpoints
        agents = []
        for result in results:
            checkpoint_path = result["checkpoint_path"]
            if os.path.exists(checkpoint_path):
                agent = PPOAgent(state_shape, num_actions, self.config)
                agent.load_model(checkpoint_path)
                agents.append(agent)
            else:
                print(f"Warning: Checkpoint {checkpoint_path} not found")

        if not agents:
            print("Error: No valid checkpoints found for averaging")
            return None

        # Create a temporary agent to hold averaged model
        temp_agent = PPOAgent(state_shape, num_actions, self.config)
        device = temp_agent.device

        # Average actor_critic parameters
        for name, param in temp_agent.model.actor_critic.named_parameters():
            averaged_param = torch.zeros_like(param)
            for agent in agents:
                agent_param = agent.model.actor_critic.get_parameter(name)
                if agent_param.device != device:
                    agent_param = agent_param.to(device)
                averaged_param += agent_param
            averaged_param /= len(agents)
            param.data.copy_(averaged_param)

        # Average ICM parameters
        for name, param in temp_agent.model.icm.icm.named_parameters():
            averaged_param = torch.zeros_like(param)
            for agent in agents:
                agent_param = agent.model.icm.icm.get_parameter(name)
                if agent_param.device != device:
                    agent_param = agent_param.to(device)
                averaged_param += agent_param
            averaged_param /= len(agents)
            param.data.copy_(averaged_param)

        # Use first agent's optimizer and scheduler state
        if agents:
            temp_agent.model.optimizer.load_state_dict(
                agents[0].model.optimizer.state_dict()
            )
            temp_agent.model.scheduler.load_state_dict(
                agents[0].model.scheduler.state_dict()
            )

        # Set episode count
        temp_agent.episode = self.total_episodes_run

        return temp_agent

    def average_and_save_models(self, results, state_shape, num_actions):
        """Average model parameters from successful agents and save to checkpoint"""
        print("Averaging models from successful agents...")

        # Create a temporary agent to hold averaged model
        temp_agent = PPOAgent(state_shape, num_actions, self.config)
        device = temp_agent.device

        # Average actor_critic parameters
        actor_critic_params = {}
        for name, param in temp_agent.model.actor_critic.named_parameters():
            averaged_param = torch.zeros_like(param)
            for result in results:
                # Move tensor to correct device if needed
                param_tensor = result["actor_critic"][name]
                if param_tensor.device != device:
                    param_tensor = param_tensor.to(device)
                averaged_param += param_tensor
            averaged_param /= len(results)
            actor_critic_params[name] = averaged_param

        # Apply averaged parameters
        for name, param in temp_agent.model.actor_critic.named_parameters():
            param.data.copy_(actor_critic_params[name])

        # Average ICM parameters
        icm_params = {}
        for name, param in temp_agent.model.icm.icm.named_parameters():
            averaged_param = torch.zeros_like(param)
            for result in results:
                # Move tensor to correct device if needed
                param_tensor = result["icm"][name]
                if param_tensor.device != device:
                    param_tensor = param_tensor.to(device)
                averaged_param += param_tensor
            averaged_param /= len(results)
            icm_params[name] = averaged_param

        # Apply averaged ICM parameters
        for name, param in temp_agent.model.icm.icm.named_parameters():
            param.data.copy_(icm_params[name])

        # Average optimizer state (first agent's structure, averaged values)
        if results[0]["optimizer"]["state"]:
            optimizer_state = results[0]["optimizer"].copy()
            for key in optimizer_state["state"]:
                for param_key in optimizer_state["state"][key]:
                    if isinstance(
                        optimizer_state["state"][key][param_key], torch.Tensor
                    ):
                        # Get first tensor to determine shape/device
                        first_tensor = optimizer_state["state"][key][param_key]
                        averaged_value = torch.zeros_like(first_tensor).to(device)
                        for result in results:
                            if key in result["optimizer"]["state"]:
                                param_tensor = result["optimizer"]["state"][key][
                                    param_key
                                ]
                                if param_tensor.device != device:
                                    param_tensor = param_tensor.to(device)
                                averaged_value += param_tensor
                        averaged_value /= len(results)
                        optimizer_state["state"][key][param_key] = averaged_value
            temp_agent.model.optimizer.load_state_dict(optimizer_state)

        # Average scheduler state
        scheduler_state = results[0]["scheduler"].copy()
        for key, value in scheduler_state.items():
            if isinstance(value, (int, float)) and key != "last_epoch":
                averaged_value = sum(r["scheduler"][key] for r in results) / len(
                    results
                )
                scheduler_state[key] = averaged_value
        temp_agent.model.scheduler.load_state_dict(scheduler_state)

        # Set episode count
        temp_agent.episode = self.total_episodes_run

        # Save to checkpoint
        print(f"Saving averaged model to {self.base_checkpoint}")
        temp_agent.save_model(self.base_checkpoint)

        # Keep individual agent checkpoints to preserve episode data across iterations
        print(
            f"Keeping {len(results)} individual agent checkpoints for episode data persistence"
        )
