# -*- coding: utf-8 -*-
import os
import numpy as np
from collections import deque
from tqdm.auto import tqdm
import torch

from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.utils.visuals import plot_metrics
from PoliwhiRL.replay import PPOMemory
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
        self.reset_tracking()

    def update_parameters_from_config(self):
        self.episode = self.config["start_episode"]
        self.record = self.config["record"]
        self.num_episodes = self.config["num_episodes"]
        self.episode_length = self.config["episode_length"]
        self.sequence_length = self.config["sequence_length"]
        self.n_goals = self.config["N_goals_target"]
        self.record_frequency = self.config["record_frequency"]
        self.results_dir = self.config["results_dir"]
        self.export_state_loc = self.config["export_state_loc"]
        self.checkpoint_frequency = self.config["checkpoint_frequency"]
        self.steps = 0
        self.report_episode = self.config["report_episode"]
        self.update_frequency = self.config["ppo_update_frequency"]
        self.epochs = self.config["ppo_epochs"]

    def reset_tracking(self):
        self.episode_data = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_losses": [],
            "moving_avg_reward": deque(maxlen=100),
            "moving_avg_length": deque(maxlen=100),
            "moving_avg_loss": deque(maxlen=100),
            "buttons_pressed": deque(maxlen=1000),
            "episode_entropies": [],
        }
        self.episode_data["buttons_pressed"].append(0)

    def train_agent(self):
        if self.report_episode:
            pbar = tqdm(
                range(self.num_episodes),
                desc=f"Training (Goals: {self.n_goals})",
                leave=True,
            )
        else:
            pbar = range(self.num_episodes)

        for episode_idx in pbar:
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

            if (
                self.episode % self.checkpoint_frequency == 0
                and self.config.get("save_checkpoint", True)
                and self.config.get("checkpoint") is not None
            ):
                self.save_model(self.config["checkpoint"])

        if (
            self.config.get("save_checkpoint", True)
            and self.config.get("checkpoint") is not None
        ):
            self.save_model(self.config["checkpoint"])

        if self.config.get("save_training_states", False):
            self.run_episode(
                save_path=f"{self.export_state_loc}/N_goals_{self.n_goals}.pkl"
            )
        else:
            self.run_episode(save_path=None)

    def run_episode(self, record_loc=None, save_path=None):
        env = Env(self.config)

        try:
            self.steps = 0
            state = env.reset()
            self.memory.reset()
            reward_sum = 0
            state_sequence = deque(
                [state] * self.sequence_length, maxlen=self.sequence_length
            )
            # Per-trajectory transformer memory: starts fresh each episode and
            # carries across rollout steps. Each transition stores the memory
            # state used to select its action so update() can reproduce the
            # same context.
            mems = self.model.init_mems(batch_size=1)

            if record_loc is not None:
                env.enable_record(record_loc, False)

            iter_range = (
                tqdm(
                    range(self.config["episode_length"]),
                    desc="Episode steps",
                    leave=False,
                )
                if self.report_episode
                else range(self.episode_length)
            )

            for _ in iter_range:
                self.steps += 1
                state_seq_arr = np.array(state_sequence)
                action, log_prob, new_mems = self.model.get_action(
                    state_seq_arr, mems
                )
                self.episode_data["buttons_pressed"].append(action)

                next_state, reward, done, _ = env.step(action)
                reward_sum += reward

                self.memory.store_transition(
                    state,
                    next_state,
                    action,
                    reward,
                    done,
                    log_prob,
                    mems,
                )

                mems = new_mems
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
            env.close()

    def _update_episode_stats(self, total_reward):
        self.episode_data["episode_rewards"].append(total_reward)
        self.episode_data["episode_lengths"].append(self.steps)
        self.episode_data["moving_avg_reward"].append(total_reward)
        self.episode_data["moving_avg_length"].append(self.steps)

        current_entropy = self.model._get_entropy_coef(self.episode)
        self.episode_data["episode_entropies"].append(current_entropy)

    def update_model(self, data=None):
        total_loss = 0
        for _ in range(self.epochs):
            data = self.memory.get_all_data() if data is None else data
            if data is None:
                return
            loss = self.model.update(data, self.episode)
            total_loss += loss
        self._update_loss_stats(total_loss)
        self.memory.reset()

    def _update_loss_stats(self, total_loss):
        steps_since_update = len(self.memory)
        avg_loss = total_loss / (self.epochs * steps_since_update)
        self.episode_data["episode_losses"].append(avg_loss)
        self.episode_data["moving_avg_loss"].append(avg_loss)

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

        pbar.set_postfix({
            "Avg Reward": f"{avg_reward:.2f}",
            "Avg Length": f"{avg_length:.2f}",
            "Reward": f"{current_reward:.2f}",
            "Length": f"{current_length}",
        })

    def _plot_metrics(self):
        os.makedirs(self.results_dir, exist_ok=True)
        plot_metrics(
            self.episode_data["episode_rewards"],
            self.episode_data["episode_losses"],
            self.episode_data["episode_lengths"],
            self.episode_data["buttons_pressed"],
            self.n_goals,
            self.episode,
            save_loc=self.results_dir,
            entropies=self.episode_data.get("episode_entropies", None),
        )

    def save_model(self, path):
        path = f"{path}"
        os.makedirs(path, exist_ok=True)

        self.model.save(path)

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
            print(f"Loaded checkpoint from {path}, episode {self.episode}")

            loaded_episode_data = info.get("episode_data", {})
            if loaded_episode_data:
                fresh_episode_data = {
                    "episode_rewards": [],
                    "episode_lengths": [],
                    "episode_losses": [],
                    "moving_avg_reward": deque(maxlen=100),
                    "moving_avg_length": deque(maxlen=100),
                    "moving_avg_loss": deque(maxlen=100),
                    "buttons_pressed": deque(maxlen=1000),
                    "episode_entropies": [],
                }
                for key, value in loaded_episode_data.items():
                    if key in fresh_episode_data:
                        if isinstance(
                            fresh_episode_data[key], deque
                        ) and not isinstance(value, deque):
                            fresh_episode_data[key] = deque(value, maxlen=100)
                        else:
                            fresh_episode_data[key] = value
                self.episode_data = fresh_episode_data
                if len(self.episode_data["buttons_pressed"]) == 0:
                    self.episode_data["buttons_pressed"].append(0)

        except FileNotFoundError:
            print(f"No checkpoint found at {path}, starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch.")
