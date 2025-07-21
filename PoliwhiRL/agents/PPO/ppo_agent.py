# -*- coding: utf-8 -*-
import os
import numpy as np
from collections import deque
from tqdm.auto import tqdm
import torch

from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.utils.visuals import plot_metrics
from PoliwhiRL.replay import PPOMemory
from PoliwhiRL.replay.exploration_memory import ExplorationMemory
from PoliwhiRL.replay.enhanced_exploration_memory import EnhancedExplorationMemory
from PoliwhiRL.models.PPO import PPOModel
from PoliwhiRL.utils.resource_manager import get_resource_pool
from PoliwhiRL.utils.macro_actions import MacroActionLearner


class PPOAgent:
    def __init__(self, input_shape, action_size, config):
        self.config = config
        self.input_shape = input_shape
        self.action_size = action_size
        self.config["input_shape"] = input_shape
        self.config["action_size"] = action_size
        self.device = config["device"]
        self.update_parameters_from_config()
        self.best_reward = float("-inf")
        self.model = PPOModel(input_shape, action_size, config)
        self.memory = PPOMemory(config)
        # Use enhanced exploration memory if enabled
        use_enhanced = config.get("use_enhanced_exploration_memory", True)
        if use_enhanced:
            self.exploration_memory = EnhancedExplorationMemory(
                max_size=100,
                history_length=config.get("ppo_exploration_history_length", 5),
                use_memory=config.get("use_exploration_memory", True),
                action_space_size=action_size,
            )
        else:
            self.exploration_memory = ExplorationMemory(
                max_size=100,
                history_length=config.get("ppo_exploration_history_length", 5),
                use_memory=config.get("use_exploration_memory", True),
            )
        self.reset_tracking()

        # Initialize macro action learning if enabled
        self.use_macro_actions = config.get("use_macro_actions", False)
        if self.use_macro_actions:
            self.macro_learner = MacroActionLearner(
                action_space_size=action_size,
                max_sequence_length=config.get("macro_max_length", 6),
                min_frequency=config.get("macro_min_frequency", 3),
            )
        else:
            self.macro_learner = None

    def update_parameters_from_config(self):
        self.episode = self.config["start_episode"]
        self.record = self.config["record"]
        self.num_episodes = self.config["num_episodes"]
        self.episode_length = self.config["episode_length"]
        self.sequence_length = self.config["sequence_length"]
        self.n_goals = self.config["N_goals_target"]
        self.early_stopping_avg_length = self.config["early_stopping_avg_length"]
        self.record_frequency = self.config["record_frequency"]
        self.results_dir = self.config["results_dir"]
        self.export_state_loc = self.config["export_state_loc"]
        self.extrinsic_reward_weight = self.config["ppo_extrinsic_reward_weight"]
        self.intrinsic_reward_weight = self.config["ppo_intrinsic_reward_weight"]
        self.checkpoint_frequency = self.config["checkpoint_frequency"]
        self.steps = 0
        self.continue_from_state_loc = self.config["continue_from_state_loc"]
        self.continue_from_state = (
            True if os.path.isfile(self.continue_from_state_loc) else False
        )
        self.train_from_memory = self.config["ppo_train_from_memory"]
        self.report_episode = self.config["report_episode"]
        self.update_frequency = self.config["ppo_update_frequency"]
        self.epochs = self.config["ppo_epochs"]

    def reset_tracking(self):
        self.episode_data = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_losses": [],
            "episode_icm_losses": [],
            "moving_avg_reward": deque(maxlen=100),
            "moving_avg_length": deque(maxlen=100),
            "moving_avg_loss": deque(maxlen=100),
            "moving_avg_icm_loss": deque(maxlen=100),
            "buttons_pressed": deque(maxlen=1000),
            "episode_entropies": [],
        }
        self.episode_data["buttons_pressed"].append(0)
        self.exploration_memory.reset()

        # Reset macro learner episode tracking if available
        if hasattr(self, "macro_learner") and self.macro_learner:
            self.macro_learner._reset_episode()

    def run_curriculum(self, start_goal_n, end_goal_n, step_increment):
        initial_episode_length = self.config["episode_length"]
        for n in range(start_goal_n, end_goal_n + 1):
            self.config["N_goals_target"] = n
            self.config["episode_length"] = initial_episode_length + (
                step_increment * (n - 1)
            )
            self.config["early_stopping_avg_length"] = (
                self.config["episode_length"] // 2
            )
            self.memory.reset(config=self.config)
            self.reset_tracking()
            print(f"Starting training for goal {n}")
            print(f"Episode length: {self.config['episode_length']}")
            print(f"Early stopping length: {self.config['early_stopping_avg_length']}")
            self.update_parameters_from_config()
            self.train_agent()

    def train_agent(self):
        if self.train_from_memory:
            print("Training from memory. Loading data from database and training.")
            self.train_from_memories()

        # Check if we're in a subprocess (multiprocessing context)
        import multiprocessing as mp

        is_subprocess = mp.current_process().name != "MainProcess"

        if self.report_episode and not is_subprocess:
            # Get worker-specific tqdm configuration
            position = self.config.get("tqdm_position", 0)
            desc_prefix = self.config.get("tqdm_desc_prefix", "")

            # Create a positioned tqdm progress bar
            pbar = tqdm(
                range(self.num_episodes),
                desc=f"{desc_prefix} Training (Goals: {self.n_goals})",
                position=position + 1,  # +1 to leave room for the main iteration bar
                leave=False,
            )
        else:
            pbar = range(self.num_episodes)
            if is_subprocess:
                print(
                    f"Agent training in subprocess - {self.num_episodes} episodes, {self.n_goals} goals"
                )
        # Check if we should boost entropy at start of training
        self._update_entropy_boost()

        for episode_idx in pbar:
            # Log progress every 5 episodes for hang detection
            if is_subprocess and episode_idx % 5 == 0:
                self._log_training_progress(
                    f"Episode {self.episode}/{self.num_episodes}"
                )

            # Periodically update entropy boost
            if self.episode % 10 == 0:
                self._update_entropy_boost()

            record_loc = (
                f"N_goals_{self.n_goals}/{self.episode}"
                if (self.episode % self.record_frequency == 0 and self.record)
                else None
            )

            if is_subprocess and episode_idx % 5 == 0:
                self._log_training_progress(f"Starting episode {self.episode}")

            self.run_episode(record_loc=record_loc)
            self.episode += 1

            if is_subprocess and episode_idx % 5 == 0:
                self._log_training_progress(f"Completed episode {self.episode - 1}")

            if len(self.memory) > self.sequence_length:
                if is_subprocess and episode_idx % 5 == 0:
                    self._log_training_progress("Updating model")
                self.update_model()

            if self.report_episode and not is_subprocess:
                self._update_progress_bar(pbar)
            if (
                self.episode % 10 == 0 and self.episode > 1
            ) or self.episode == self.num_episodes:
                if is_subprocess:
                    self._log_training_progress("Plotting metrics")
                self._plot_metrics()
            self.model.step_scheduler()

            if (
                self.episode % self.checkpoint_frequency == 0
                and self.config.get("save_checkpoint", True)
                and self.config.get("checkpoint") is not None
            ):
                self.save_model(self.config["checkpoint"])

            # if self._should_stop_early():
            #     break

        if (
            self.config.get("save_checkpoint", True)
            and self.config.get("checkpoint") is not None
        ):
            self.save_model(self.config["checkpoint"])
        # Only save state if explicitly requested
        if self.config.get("save_training_states", False):
            self.run_episode(
                save_path=f"{self.export_state_loc}/N_goals_{self.n_goals}.pkl"
            )
        else:
            self.run_episode(save_path=None)

    def train_from_memories(self):
        memory_ids = self.memory.get_memory_ids(self.config)
        if len(memory_ids) == 0:
            return
        for memory_id in memory_ids:
            data = self.memory.load_from_database(self.config, memory_id)
            self.update_model(data)

    def _should_stop_early(self):
        if (
            np.mean(self.episode_data["moving_avg_length"])
            < self.early_stopping_avg_length
            and self.early_stopping_avg_length > 0
            and self.episode > 110
        ):
            print(
                "Average Steps are below early stopping threshold! Stopping to prevent overfitting."
            )
            return True
        return False

    def run_episode(self, record_loc=None, save_path=None):
        # Use shared temp directory if available in config
        if self.config.get("shared_temp_dir"):
            env = Env(self.config, shared_temp_dir=self.config["shared_temp_dir"])
        else:
            env = Env.create_with_shared_temp(self.config)

        try:
            self.steps = 0
            if self.continue_from_state and (np.random.rand() < 0.9):
                state = env.load_gym_state(
                    self.continue_from_state_loc, self.episode_length, self.n_goals
                )
                self.steps = env.steps
            else:
                state = env.reset()
                self.memory.reset()
                reward_sum = 0
                state_sequence = deque(
                    [state] * self.sequence_length, maxlen=self.sequence_length
                )

                # Update exploration memory with initial screen
                screen = env.get_observation()
                self.exploration_memory.add_screen(screen)
            if record_loc is not None:
                env.enable_record(record_loc, False)

            # Check if we're in a subprocess (multiprocessing context)
            import multiprocessing as mp

            is_subprocess = mp.current_process().name != "MainProcess"

            iter_range = (
                tqdm(
                    range(self.config["episode_length"]),
                    desc=f"{self.config.get('tqdm_desc_prefix', '')} Episode steps",
                    position=self.config.get("tqdm_position", 0)
                    + 2,  # +2 to leave room for iteration and episode bars
                    leave=False,
                )
                if self.report_episode and not is_subprocess
                else range(self.episode_length)
            )

            for _ in iter_range:
                self.steps += 1
                state_seq_arr = np.array(state_sequence)
                # Get exploration memory tensor
                exploration_tensor = self.exploration_memory.get_memory_tensor()
                action = self.model.get_action(state_seq_arr, exploration_tensor)
                self.episode_data["buttons_pressed"].append(action)

                # Log action diversity to file for debugging
                if self.steps % 25 == 0 and self.steps > 0:  # Log every 25 steps
                    self._log_action_diversity_to_file(
                        action, state_seq_arr, exploration_tensor
                    )

                # Track action for macro learning
                if self.macro_learner:
                    state_hash = None
                    if hasattr(self.exploration_memory, "_compute_hash"):
                        state_hash = self.exploration_memory._compute_hash(state)
                    self.macro_learner.add_step(
                        action, 0, state_hash
                    )  # Reward will be updated later

                next_state, extrinsic_reward, done, _ = env.step(action)

                # Update exploration memory with transition information
                if hasattr(self.exploration_memory, "add_transition"):
                    # Enhanced exploration memory with transition tracking
                    coordinates = self._get_coordinates_from_env(env)
                    self.exploration_memory.add_transition(
                        state, action, next_state, extrinsic_reward, coordinates
                    )
                else:
                    # Standard exploration memory
                    self.exploration_memory.add_screen(next_state)

                intrinsic_reward = self.model.compute_intrinsic_reward(
                    state, next_state, action
                )

                # Add exploration bonus if using enhanced memory
                exploration_bonus = 0.0
                if hasattr(self.exploration_memory, "get_exploration_bonus"):
                    state_hash = self.exploration_memory._compute_hash(state)
                    exploration_bonus = (
                        self.exploration_memory.get_exploration_bonus(state_hash) * 0.01
                    )

                total_reward = self._compute_total_reward(
                    extrinsic_reward, intrinsic_reward + exploration_bonus
                )
                reward_sum += extrinsic_reward

                # Update macro learner with reward
                if (
                    self.macro_learner
                    and len(self.macro_learner.current_episode_rewards) > 0
                ):
                    # Update the last step's reward
                    self.macro_learner.current_episode_rewards[-1] = extrinsic_reward

                # Get exploration memory tensor
                exploration_tensor = self.exploration_memory.get_memory_tensor()
                log_prob = self.model.compute_log_prob(
                    state_seq_arr, action, exploration_tensor
                )

                # Get exploration memory tensor
                exploration_tensor = self.exploration_memory.get_memory_tensor()

                self.memory.store_transition(
                    state,
                    next_state,
                    action,
                    total_reward,
                    done,
                    log_prob,
                    exploration_tensor,
                )

                state = next_state
                state_sequence.append(state)

                if done:
                    break

                if (
                    self.steps % self.update_frequency == 0
                    and len(self.memory) > self.sequence_length
                ):
                    self.update_model()

            self._update_episode_stats(reward_sum)

            if save_path is not None:
                env.save_gym_state(save_path)
        finally:
            # Ensure environment is properly closed
            env.close()

    def _compute_total_reward(self, extrinsic_reward, intrinsic_reward):
        return (
            self.extrinsic_reward_weight * extrinsic_reward
            + self.intrinsic_reward_weight * intrinsic_reward
        )

    def _update_episode_stats(self, total_reward):
        self.episode_data["episode_rewards"].append(total_reward)
        self.episode_data["episode_lengths"].append(self.steps)
        self.episode_data["moving_avg_reward"].append(total_reward)
        self.episode_data["moving_avg_length"].append(self.steps)

        # Track current entropy coefficient
        current_entropy = self.model._get_entropy_coef(self.episode)
        self.episode_data["episode_entropies"].append(current_entropy)

        # Process macro actions at end of episode
        if self.macro_learner:
            self.macro_learner.end_episode()

            # Print macro action statistics occasionally
            if self.episode % 50 == 0 and self.episode > 0:
                stats = self.macro_learner.get_macro_statistics()
                if stats.get("num_macros", 0) > 0:
                    print(
                        f"Episode {self.episode}: Discovered {stats['num_macros']} macro actions, "
                        f"avg reward: {stats['avg_macro_reward']:.3f}"
                    )
                    top_macros = self.macro_learner.get_top_macros(3)
                    for i, (sequence, info) in enumerate(top_macros):
                        print(
                            f"  Top {i + 1}: {sequence} (reward: {info['avg_reward']:.3f})"
                        )

    def update_model(self, data=None):
        total_loss = 0
        total_icm_loss = 0
        was_data_none = data is None
        for _ in range(self.epochs):
            data = self.memory.get_all_data() if data is None else data
            if data is None:
                return
            loss, icm_loss = self.model.update(data, self.episode)
            total_loss += loss
            total_icm_loss += icm_loss
        if was_data_none:
            self._update_loss_stats(total_loss, total_icm_loss)
        self.memory.reset()

    def _update_loss_stats(self, total_loss, total_icm_loss):
        steps_since_update = len(self.memory)
        avg_loss = total_loss / (self.epochs * steps_since_update)
        avg_icm_loss = total_icm_loss / (self.epochs * steps_since_update)
        self.episode_data["episode_losses"].append(avg_loss)
        self.episode_data["episode_icm_losses"].append(avg_icm_loss)
        self.episode_data["moving_avg_loss"].append(avg_loss)
        self.episode_data["moving_avg_icm_loss"].append(avg_icm_loss)

    def _update_progress_bar(self, pbar):
        avg_reward = (
            np.mean(self.episode_data["moving_avg_reward"])
            if self.episode_data["moving_avg_reward"]
            else 0
        )
        avg_length = (
            np.mean(self.episode_data["moving_avg_length"])
            if self.episode_data["moving_avg_length"]
            else 0
        )

        current_reward = (
            self.episode_data["episode_rewards"][-1]
            if self.episode_data["episode_rewards"]
            else 0
        )
        current_length = (
            self.episode_data["episode_lengths"][-1]
            if self.episode_data["episode_lengths"]
            else 0
        )

        # Add worker ID to postfix if available
        worker_id = self.config.get("tqdm_worker_id", None)
        postfix = {
            "Avg Reward": f"{avg_reward:.2f}",
            "Avg Length": f"{avg_length:.2f}",
            "Reward": f"{current_reward:.2f}",
            "Length": f"{current_length}",
        }

        if worker_id is not None:
            postfix["Worker"] = worker_id

        pbar.set_postfix(postfix)

    def _plot_metrics(self):
        # Check if this agent has individual agent data (it's part of a multi-agent setup)
        if "individual_agent_data" in self.episode_data:
            # This is the averaged agent with individual agent data
            # First plot the averaged agent's metrics
            plot_metrics(
                self.episode_data["episode_rewards"],
                self.episode_data["episode_losses"],
                self.episode_data["episode_lengths"],
                self.episode_data["buttons_pressed"],
                self.n_goals,
                self.episode,
                save_loc=self.results_dir,
                title_prefix="Averaged Agent",
                entropies=self.episode_data.get("episode_entropies", None),
            )

            # Then plot each individual agent's metrics to their respective results directories
            for i, agent_data in enumerate(self.episode_data["individual_agent_data"]):
                # Construct the agent-specific results directory
                agent_results_dir = (
                    f"{self.config['results_dir'].rstrip('_0123456789')}_{i}"
                )

                plot_metrics(
                    agent_data["episode_rewards"],
                    agent_data["episode_losses"],
                    agent_data["episode_lengths"],
                    agent_data["buttons_pressed"],
                    self.n_goals,
                    self.episode,
                    save_loc=agent_results_dir,
                    title_prefix=f"Agent {i}",
                    entropies=agent_data.get("episode_entropies", None),
                )
        else:
            # Get worker ID if available for title prefix
            worker_id = self.config.get("tqdm_worker_id")
            title_prefix = f"Agent {worker_id}" if worker_id is not None else None

            # Ensure the results directory exists
            os.makedirs(self.results_dir, exist_ok=True)

            # This is a regular agent or an individual agent in a multi-agent setup
            plot_metrics(
                self.episode_data["episode_rewards"],
                self.episode_data["episode_losses"],
                self.episode_data["episode_lengths"],
                self.episode_data["buttons_pressed"],
                self.n_goals,
                self.episode,
                save_loc=self.results_dir,
                title_prefix=title_prefix,
                entropies=self.episode_data.get("episode_entropies", None),
            )

    def save_model(self, path):
        path = f"{path}"
        os.makedirs(path, exist_ok=True)

        resource_pool = get_resource_pool()

        # Use file locking for safe checkpoint writing
        with resource_pool.file_lock(path):
            self.model.save(path)

            # Save additional information
            info = {
                "episode": self.episode,
                "best_reward": (
                    max(self.episode_data["episode_rewards"])
                    if self.episode_data["episode_rewards"]
                    else float("-inf")
                ),
                "episode_data": self.episode_data,
            }
            torch.save(info, f"{path}/info.pth")

            # Save exploration memory if enhanced
            if hasattr(self.exploration_memory, "save_to_file"):
                try:
                    self.exploration_memory.save_to_file(
                        f"{path}/exploration_memory.pkl"
                    )
                except Exception as e:
                    print(f"Warning: Could not save exploration memory: {e}")

    def load_model(self, path):
        try:
            resource_pool = get_resource_pool()

            # Use file locking for safe checkpoint reading
            with resource_pool.file_lock(path):
                self.model.load(f"{path}")
                torch.serialization.add_safe_globals(["numpy", "np"])
                info = torch.load(
                    f"{path}/info.pth", map_location=self.device, weights_only=False
                )
                self.config["start_episode"] = info["episode"]
                self.episode = info["episode"]

                # Debug logging
                worker_id = self.config.get("tqdm_worker_id", "?")
                print(
                    f"Agent {worker_id} loaded checkpoint from {path}, episode {self.episode}"
                )

                # Load episode data if available, but ensure all required keys exist
                loaded_episode_data = info.get("episode_data", {})
                if loaded_episode_data:
                    # Start with fresh episode data to ensure all keys exist
                    fresh_episode_data = {
                        "episode_rewards": [],
                        "episode_lengths": [],
                        "episode_losses": [],
                        "episode_icm_losses": [],
                        "moving_avg_reward": deque(maxlen=100),
                        "moving_avg_length": deque(maxlen=100),
                        "moving_avg_loss": deque(maxlen=100),
                        "moving_avg_icm_loss": deque(maxlen=100),
                        "buttons_pressed": deque(maxlen=1000),
                        "episode_entropies": [],
                    }
                    # Update with loaded data, preserving any existing values
                    for key, value in loaded_episode_data.items():
                        if key in fresh_episode_data:
                            if isinstance(
                                fresh_episode_data[key], deque
                            ) and not isinstance(value, deque):
                                # Convert list to deque if needed
                                fresh_episode_data[key] = deque(value, maxlen=100)
                            else:
                                fresh_episode_data[key] = value
                    self.episode_data = fresh_episode_data
                    # Ensure buttons_pressed has at least one element
                    if len(self.episode_data["buttons_pressed"]) == 0:
                        self.episode_data["buttons_pressed"].append(0)

                # Load exploration memory if enhanced
                if hasattr(self.exploration_memory, "load_from_file"):
                    try:
                        memory_path = f"{path}/exploration_memory.pkl"
                        if os.path.exists(memory_path):
                            self.exploration_memory.load_from_file(memory_path)
                            print(
                                f"Agent {worker_id} loaded exploration memory from {memory_path}"
                            )
                    except Exception as e:
                        print(f"Warning: Could not load exploration memory: {e}")

        except FileNotFoundError:
            print(f"No checkpoint found at {path}, starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch.")

    def _log_action_diversity_to_file(self, action, state_seq_arr, exploration_tensor):
        """Log action diversity metrics to file for debugging agent behavior"""
        try:
            # Create logs directory for this agent
            agent_id = self.config.get("tqdm_worker_id", "main")
            logs_dir = f"{self.results_dir}/action_logs"
            os.makedirs(logs_dir, exist_ok=True)

            log_file = f"{logs_dir}/agent_{agent_id}_actions.log"

            # Get current action probabilities from model
            state_tensor = torch.FloatTensor(state_seq_arr).unsqueeze(0).to(self.device)
            if exploration_tensor is not None:
                exploration_tensor = (
                    torch.FloatTensor(exploration_tensor).unsqueeze(0).to(self.device)
                )

            with torch.no_grad():
                action_probs, _ = self.model.actor_critic(
                    state_tensor, exploration_tensor
                )
                action_probs = action_probs.squeeze().cpu().numpy()

            # Calculate action diversity metrics
            recent_actions = list(self.episode_data["buttons_pressed"])[
                -25:
            ]  # Last 25 actions
            action_counts = {}
            for a in recent_actions:
                action_counts[a] = action_counts.get(a, 0) + 1

            # Calculate entropy of recent actions
            total_recent = len(recent_actions)
            action_entropy = 0.0
            if total_recent > 0:
                for count in action_counts.values():
                    prob = count / total_recent
                    if prob > 0:
                        action_entropy -= prob * np.log2(prob)

            # Calculate entropy of current action probabilities
            prob_entropy = 0.0
            for prob in action_probs:
                if prob > 1e-10:
                    prob_entropy -= prob * np.log2(prob)

            # Get current entropy coefficient
            current_entropy_coef = self.model._get_entropy_coef(self.episode)

            # Map actions to button names for readability
            button_names = ["A", "B", "Right", "Left", "Up", "Down", "Start", "Select"]
            action_name = (
                button_names[action]
                if action < len(button_names)
                else f"Action_{action}"
            )

            # Most frequent recent action
            most_frequent_action = (
                max(action_counts.items(), key=lambda x: x[1])[0]
                if action_counts
                else action
            )
            most_frequent_name = (
                button_names[most_frequent_action]
                if most_frequent_action < len(button_names)
                else f"Action_{most_frequent_action}"
            )
            most_frequent_pct = (
                (action_counts.get(most_frequent_action, 0) / total_recent * 100)
                if total_recent > 0
                else 0
            )

            # Warning flags for critical thresholds
            warning_flags = []
            if action_entropy < 0.5:
                warning_flags.append("LOW_ENTROPY")
            if prob_entropy < 0.5:
                warning_flags.append("OVERCONFIDENT")
            if most_frequent_pct > 85:
                warning_flags.append("ACTION_COLLAPSE")

            warnings_str = (
                f" | WARNINGS: {','.join(warning_flags)}" if warning_flags else ""
            )

            # Create log entry
            log_entry = (
                f"Episode: {self.episode:4d} | Step: {self.steps:4d} | "
                f"Action: {action_name:6s} | "
                f"ActionEntropy: {action_entropy:.3f} | "
                f"ProbEntropy: {prob_entropy:.3f} | "
                f"EntropyCoef: {current_entropy_coef:.4f} | "
                f"MostFreq: {most_frequent_name:6s} ({most_frequent_pct:.1f}%) | "
                f"ActionProbs: [{', '.join([f'{p:.3f}' for p in action_probs])}] | "
                f"RecentActions: {recent_actions[-10:]}{warnings_str}\n"  # Last 10 actions for context
            )

            # Write to file
            with open(log_file, "a") as f:
                f.write(log_entry)

        except Exception as e:
            # Don't let logging errors crash training
            print(f"Warning: Could not log action diversity: {e}")

    def _log_training_progress(self, message):
        """Log training progress for hang detection"""
        try:
            import time

            agent_id = self.config.get("tqdm_worker_id", "main")
            hang_log_dir = f"{self.results_dir}/hang_logs"
            os.makedirs(hang_log_dir, exist_ok=True)
            hang_log_file = f"{hang_log_dir}/agent_{agent_id}_hang_detection.log"

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(hang_log_file, "a") as f:
                f.write(f"[{timestamp}] Agent {agent_id} - PROGRESS: {message}\n")
        except Exception:
            pass  # Don't let logging errors interfere with training

    def _get_coordinates_from_env(self, env):
        """Extract coordinates from environment if available"""
        try:
            if hasattr(env, "ram") and hasattr(env.ram, "get_variables"):
                variables = env.ram.get_variables()
                return (
                    variables.get("X"),
                    variables.get("Y"),
                    variables.get("map_num"),
                )
        except Exception:
            pass
        return None

    def get_episode_data(self):
        return self.episode_data

    def set_episode_data(self, data):
        self.episode_data = data

    def _update_entropy_boost(self):
        """Adaptively adjust entropy based on exploration needs"""
        # Calculate exploration metrics
        if hasattr(self.exploration_memory, "hash_visits"):
            # Get recent exploration statistics
            total_states = len(self.exploration_memory.hash_visits)
            recent_visits = list(self.exploration_memory.hash_visits.values())[-100:]
            avg_visits = np.mean(recent_visits) if recent_visits else 1.0

            # Check if we're stuck (visiting same states repeatedly)
            novelty_rate = 1.0 / avg_visits if avg_visits > 0 else 1.0

            # Boost entropy if:
            # 1. Starting a new curriculum stage (low episode count)
            # 2. Low novelty rate (stuck in local patterns)
            # 3. Haven't reached goal recently

            is_new_stage = self.episode < 20  # First 20 episodes of new stage
            is_low_novelty = novelty_rate < 0.3  # Visiting same states > 3 times avg

            # Check recent goal progress
            recent_lengths = list(self.episode_data["moving_avg_length"])[-10:]
            is_struggling = (
                len(recent_lengths) > 5
                and np.mean(recent_lengths) > self.episode_length * 0.8
            )

            # Calculate entropy boost
            boost = 0.0
            if is_new_stage:
                boost += 0.05  # 5% boost for new curriculum stage
            if is_low_novelty:
                boost += 0.03  # 3% boost for low exploration
            if is_struggling:
                boost += 0.02  # 2% boost if struggling to reach goals

            # Decay boost over time
            self.model.entropy_boost = boost * (0.95 ** (self.episode // 10))

            # Log entropy adjustments
            if boost > 0 and self.episode % 10 == 0:
                print(
                    f"Episode {self.episode}: Entropy boost = {self.model.entropy_boost:.4f} "
                    f"(new_stage={is_new_stage}, low_novelty={is_low_novelty}, struggling={is_struggling}, "
                    f"total_states={total_states})"
                )
        else:
            # No exploration memory, use simple new stage detection
            if self.episode < 20:
                self.model.entropy_boost = 0.05 * (0.95 ** (self.episode // 10))
            else:
                self.model.entropy_boost = 0.0
