# -*- coding: utf-8 -*-
import os
import numpy as np
from collections import deque
from tqdm import tqdm
import torch

from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.utils.visuals import plot_metrics
from PoliwhiRL.replay import PPOMemory
from PoliwhiRL.replay.exploration_memory import ExplorationMemory
from PoliwhiRL.models.PPO import PPOModel


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
        self.exploration_memory = ExplorationMemory(
            max_size=100, 
            history_length=config.get("exploration_history_length", 5), 
            use_memory=config.get("use_exploration_memory", True)
        )
        self.reset_tracking()

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
            "buttons_pressed": deque(maxlen=100),
        }
        self.episode_data["buttons_pressed"].append(0)
        self.exploration_memory.reset()

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
            self.train_from_memories()

        if self.report_episode:
            pbar = tqdm(
                range(self.num_episodes), desc=f"Training (Goals: {self.n_goals})"
            )
        else:
            pbar = range(self.num_episodes)
        for _ in pbar:
            record_loc = (
                f"N_goals_{self.n_goals}/{self.episode}"
                if (self.episode % self.record_frequency == 0 and self.record)
                else None
            )
            self.run_episode(record_loc=record_loc)
            self.episode += 1
            if len(self.memory) > self.sequence_length:
                self.update_model()

            if self.report_episode:
                self._update_progress_bar(pbar)
            if (
                self.episode % 10 == 0 and self.episode > 1
            ) or self.episode == self.num_episodes:
                self._plot_metrics()
            self.model.step_scheduler()

            if self.episode % self.checkpoint_frequency == 0:
                self.save_model(self.config["checkpoint"])

            if self._should_stop_early():
                break

        self.save_model(self.config["checkpoint"])
        self.run_episode(
            save_path=f"{self.export_state_loc}/N_goals_{self.n_goals}.pkl"
        )

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
        env = Env(self.config)
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

        iter_range = (
            tqdm(range(self.config["episode_length"]), desc="Episode steps")
            if self.report_episode
            else range(self.episode_length)
        )

        for _ in iter_range:
            self.steps += 1
            state_seq_arr = np.array(state_sequence)
            # Get exploration memory tensor
            exploration_tensor = self.exploration_memory.get_memory_tensor()
            action = self.model.get_action(state_seq_arr, exploration_tensor)
            self.episode_data["buttons_pressed"].append(action)
            next_state, extrinsic_reward, done, _ = env.step(action)

            # Update exploration memory with new screen
            self.exploration_memory.add_screen(next_state)

            intrinsic_reward = self.model.compute_intrinsic_reward(
                state, next_state, action
            )
            total_reward = self._compute_total_reward(
                extrinsic_reward, intrinsic_reward
            )
            reward_sum += extrinsic_reward

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

        pbar.set_postfix(
            {
                "Avg Reward (100 ep)": f"{avg_reward:.2f}",
                "Avg Length (100 ep)": f"{avg_length:.2f}",
                "Current Reward": f"{current_reward:.2f}",
                "Current Length": f"{current_length}",
            }
        )

    def _plot_metrics(self):
        plot_metrics(
            self.episode_data["episode_rewards"],
            self.episode_data["episode_losses"],
            self.episode_data["episode_lengths"],
            self.episode_data["buttons_pressed"],
            self.n_goals,
            self.episode,
            save_loc=self.results_dir,
        )

    def save_model(self, path):
        path = f"{path}"
        os.makedirs(path, exist_ok=True)
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

    def load_model(self, path):
        try:
            self.model.load(f"{path}")
            torch.serialization.add_safe_globals(["numpy", "np"])
            info = torch.load(
                f"{path}/info.pth", map_location=self.device, weights_only=False
            )
            self.config["start_episode"] = info["episode"]
            self.episode = info["episode"]
            self.episode_data = info.get("episode_data", self.episode_data)

        except FileNotFoundError:
            print(f"No checkpoint found at {path}, starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch.")

    def get_episode_data(self):
        return self.episode_data

    def set_episode_data(self, data):
        self.episode_data = data
