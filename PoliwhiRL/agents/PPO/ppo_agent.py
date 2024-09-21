# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from tqdm import tqdm

from PoliwhiRL.models.PPO.PPOTransformer import PPOTransformer
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.utils.visuals import plot_metrics
from PoliwhiRL.models.ICM import ICMModule
from PoliwhiRL.replay import PPOMemory


class PPOAgent:
    def __init__(self, input_shape, action_size, config):
        self.config = config
        self.input_shape = input_shape
        self.action_size = action_size
        self.config["input_shape"] = self.input_shape
        self.config["action_size"] = self.action_size
        self.device = torch.device(self.config["device"])
        self.update_parameters_from_config()
        self.update_frequency = self.config["update_frequency"]

        self._initialize_networks()
        self._initialize_optimizers()
        self.icm = ICMModule(input_shape, action_size, config)
        self.memory = PPOMemory(config)
        self.reset_tracking()

    def update_parameters_from_config(self):
        self.episode = 0
        self.num_episodes = self.config["num_episodes"]
        self.sequence_length = self.config["sequence_length"]
        self.learning_rate = self.config["learning_rate"]
        self.gamma = self.config["gamma"]
        self.epsilon = self.config["ppo_epsilon"]
        self.epochs = self.config["ppo_epochs"]
        self.batch_size = self.config["batch_size"]
        self.n_goals = self.config["N_goals_target"]
        self.early_stopping_avg_length = self.config["early_stopping_avg_length"]
        self.record_frequency = self.config["record_frequency"]
        self.results_dir = self.config["results_dir"]
        self.export_state_loc = self.config["export_state_loc"]
        self.value_loss_coef = self.config["value_loss_coef"]
        self.entropy_coef = self.config["entropy_coef"]
        self.entropy_decay = self.config["entropy_coef_decay"]
        self.entropy_min = self.config["entropy_coef_min"]
        self.extrinsic_reward_weight = self.config["extrinsic_reward_weight"]
        self.intrinsic_reward_weight = self.config["intrinsic_reward_weight"]
        self.steps = 0

    def _initialize_networks(self):
        self.actor_critic = PPOTransformer(self.input_shape, self.action_size).to(
            self.device
        )

    def _initialize_optimizers(self):
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self.learning_rate
        )
        self._setup_lr_scheduler()

    def reset_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.config["learning_rate"]

    def _setup_lr_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.config["lr_scheduler_factor"],
            patience=self.config["lr_scheduler_patience"],
            threshold=self.config["lr_scheduler_threshold"],
            min_lr=self.config["lr_scheduler_min_lr"],
        )

    def reset_tracking(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.episode_icm_losses = []
        self.moving_avg_reward = deque(maxlen=100)
        self.moving_avg_length = deque(maxlen=100)
        self.moving_avg_loss = deque(maxlen=100)
        self.moving_avg_icm_loss = deque(maxlen=100)
        self.buttons_pressed = deque(maxlen=100)
        self.buttons_pressed.append(0)

    def _initialize_episode_buffers(self):
        self.states = np.zeros(
            (self.max_episode_length,) + self.input_shape, dtype=np.uint8
        )
        self.next_states = np.zeros(
            (self.max_episode_length,) + self.input_shape, dtype=np.uint8
        )
        self.actions = np.zeros(self.max_episode_length, dtype=np.int64)
        self.rewards = np.zeros(self.max_episode_length, dtype=np.float32)
        self.intrinsic_rewards = np.zeros(self.max_episode_length, dtype=np.float32)
        self.dones = np.zeros(self.max_episode_length, dtype=np.bool_)
        self.log_probs = np.zeros(self.max_episode_length, dtype=np.float32)

    def get_action(self, state_sequence):
        state_sequence = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state_sequence)
        action = torch.multinomial(action_probs, 1).item()
        self.buttons_pressed.append(action)
        return action

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
            self.reset_learning_rate()
            self.reset_tracking()
            print(f"Starting training for goal {n}")
            print(f"Episode length: {self.config['episode_length']}")
            print(f"Early stopping length: {self.config['early_stopping_avg_length']}")
            self.update_parameters_from_config()
            self.train_agent()

    def train_agent(self):
        pbar = tqdm(range(self.num_episodes), desc=f"Training (Goals: {self.n_goals})")
        for self.episode in pbar:
            record_loc = (
                f"N_goals_{self.n_goals}/{self.episode}"
                if self.episode % self.record_frequency == 0
                else None
            )
            self.run_episode(record_loc=record_loc)
            if len(self.memory) > self.sequence_length:
                self.update_model()
            self._update_progress_bar(pbar)

            avg_reward = (
                np.mean(self.moving_avg_reward) if self.moving_avg_reward else 0
            )
            self.scheduler.step(avg_reward)

            if self._should_stop_early():
                break

        self.save_model(self.config["checkpoint"])
        self.run_episode(
            save_path=f"{self.export_state_loc}/N_goals_{self.n_goals}.pkl"
        )

    def _should_stop_early(self):
        if (
            np.mean(self.moving_avg_length) < self.early_stopping_avg_length
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
        state = env.reset()
        self.memory.reset()
        reward_sum = 0
        state_sequence = deque(
            [state] * self.sequence_length, maxlen=self.sequence_length
        )
        if record_loc is not None:
            env.enable_record(record_loc, False)

        for step in range(self.config["episode_length"]):
            self.steps += 1
            action = self.get_action(np.array(state_sequence))
            next_state, extrinsic_reward, done, _ = env.step(action)

            intrinsic_reward = self.icm.compute_intrinsic_reward(
                state, next_state, action
            )
            total_reward = self._compute_total_reward(
                extrinsic_reward, intrinsic_reward
            )
            reward_sum += total_reward

            log_prob = self._compute_log_prob(state_sequence, action)

            self.memory.store_transition(
                state, next_state, action, total_reward, done, log_prob
            )

            state = next_state
            state_sequence.append(state)

            if done:
                break
            
            if self.steps % self.update_frequency == 0 and len(self.memory) > self.sequence_length:
                self.update_model()

        self._update_episode_stats(reward_sum)

        if save_path is not None:
            env.save_gym_state(save_path)

        

    def _compute_total_reward(self, extrinsic_reward, intrinsic_reward):
        return (
            self.extrinsic_reward_weight * extrinsic_reward
            + self.intrinsic_reward_weight * intrinsic_reward
        )

    def _compute_log_prob(self, state_sequence, action):
        state_tensor = (
            torch.FloatTensor(np.array(state_sequence)).unsqueeze(0).to(self.device)
        )
        action_probs, _ = self.actor_critic(state_tensor)
        return torch.log(action_probs[0, action]).item()

    def _update_episode_stats(self, total_reward):
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(self.steps)
        self.moving_avg_reward.append(total_reward)
        self.moving_avg_length.append(self.steps)

    def update_model(self):
        total_loss = 0
        total_icm_loss = 0

        for _ in range(self.epochs):
            batch_data = self.memory.get_all_data()
            actor_loss, critic_loss, entropy_loss = self._compute_ppo_losses(
                batch_data
            )
            icm_loss = self.icm.update(
                batch_data["states"][:, -1],
                batch_data["next_states"][:, -1],
                batch_data["actions"],
            )

            loss = actor_loss + critic_loss + entropy_loss
            total_loss += loss.item()
            total_icm_loss += icm_loss

            self._update_networks(loss)

        self._update_loss_stats(total_loss, total_icm_loss)
        self.memory.reset()


    def _get_entropy_coef(self):
        return max(
            self.entropy_coef * self.entropy_decay**self.episode, self.entropy_min
        )

    def _compute_ppo_losses(self, batch_data):
        returns = self._compute_returns(batch_data["rewards"], batch_data["dones"])
        advantages = self._compute_advantages(batch_data["states"], returns)

        new_probs, new_values = self.actor_critic(batch_data["states"])
        new_probs = torch.clamp(new_probs, 1e-10, 1.0)
        new_log_probs = torch.log(
            new_probs.gather(1, batch_data["actions"].unsqueeze(1))
        ).squeeze()

        ratio = torch.exp(new_log_probs - batch_data["old_log_probs"])
        ratio = torch.clamp(ratio, 0.0, 10.0)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        new_values = new_values.squeeze()
        if new_values.dim() == 0:
            new_values = new_values.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)

        critic_loss = self.value_loss_coef * nn.functional.mse_loss(new_values, returns)

        entropy = -(new_probs * torch.log(new_probs + 1e-10)).sum(dim=-1).mean()
        entropy_loss = -self._get_entropy_coef() * entropy

        if (
            torch.isnan(actor_loss)
            or torch.isnan(critic_loss)
            or torch.isnan(entropy_loss)
        ):
            raise ValueError("NaN detected in PPO losses")

        return actor_loss, critic_loss, entropy_loss

    def _update_networks(self, ppo_loss):
        self.optimizer.zero_grad()
        ppo_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
        self.optimizer.step()

    def _update_loss_stats(self, total_loss, total_icm_loss):
        steps_since_update = len(self.memory)
        avg_loss = total_loss / (self.epochs * (steps_since_update / self.batch_size))
        avg_icm_loss = total_icm_loss / (
            self.epochs * (steps_since_update / self.batch_size)
        )
        self.episode_losses.append(avg_loss)
        self.episode_icm_losses.append(avg_icm_loss)
        self.moving_avg_loss.append(avg_loss)
        self.moving_avg_icm_loss.append(avg_icm_loss)

    def _compute_returns(self, rewards, dones):
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (~dones[t])
            returns[t] = running_return
        return returns

    def _compute_advantages(self, states, returns):
        with torch.no_grad():
            _, state_values = self.actor_critic(states)
            advantages = returns - state_values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def _update_progress_bar(self, pbar):
        avg_reward = np.mean(self.moving_avg_reward) if self.moving_avg_reward else 0
        avg_length = np.mean(self.moving_avg_length) if self.moving_avg_length else 0
        avg_loss = np.mean(self.moving_avg_loss) if self.moving_avg_loss else 0
        avg_icm_loss = (
            np.mean(self.moving_avg_icm_loss) if self.moving_avg_icm_loss else 0
        )
        current_reward = self.episode_rewards[-1] if self.episode_rewards else 0
        current_length = self.episode_lengths[-1] if self.episode_lengths else 0
        current_lr = self.optimizer.param_groups[0]["lr"]

        pbar.set_postfix(
            {
                "Avg Reward (100 ep)": f"{avg_reward:.2f}",
                "Avg Length (100 ep)": f"{avg_length:.2f}",
                "Avg Loss (100 ep)": f"{avg_loss:.4f}",
                "Avg ICM Loss (100 ep)": f"{avg_icm_loss:.4f}",
                "Current Reward": f"{current_reward:.2f}",
                "Current Length": f"{current_length}",
                "Learning Rate": f"{current_lr:.2e}",
            }
        )

        if self.episode % 10 == 0 and self.episode > 100:
            self._plot_metrics()

    def _plot_metrics(self):
        plot_metrics(
            self.episode_rewards,
            self.episode_losses,
            self.episode_lengths,
            self.buttons_pressed,
            self.n_goals,
            save_loc=self.results_dir,
        )

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(
            self.actor_critic.state_dict(), f"{path}/actor_critic_{self.n_goals}.pth"
        )
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer_{self.n_goals}.pth")
        torch.save(self.scheduler.state_dict(), f"{path}/scheduler_{self.n_goals}.pth")

        # Save additional information
        info = {
            "episode": self.episode,
            "best_reward": (
                max(self.episode_rewards) if self.episode_rewards else float("-inf")
            ),
        }
        torch.save(info, f"{path}/info_{self.n_goals}.pth")

        self.icm.save(f"{path}/icm_{self.n_goals}")
        print(f"Model saved to {path}")

    def load_model(self, path):
        try:
            actor_critic_state = torch.load(
                f"{path}/actor_critic_{self.n_goals}.pth",
                map_location=self.device,
                weights_only=True,
            )
            self.actor_critic.load_state_dict(actor_critic_state)

            optimizer_state = torch.load(
                f"{path}/optimizer_{self.n_goals}.pth",
                map_location=self.device,
                weights_only=True,
            )
            self.optimizer.load_state_dict(optimizer_state)

            scheduler_state = torch.load(
                f"{path}/scheduler_{self.n_goals}.pth",
                map_location=self.device,
                weights_only=True,
            )
            self.scheduler.load_state_dict(scheduler_state)

            # Load additional information
            info = torch.load(
                f"{path}/info_{self.n_goals}.pth", map_location=self.device
            )
            self.episode = info["episode"]
            best_reward = info["best_reward"]

            self.icm.load(f"{path}/icm_{self.n_goals}")

            print(f"Model loaded from {path}")
            print(f"Loaded model was trained for {self.episode} episodes")
            print(f"Best reward achieved: {best_reward}")
        except FileNotFoundError:
            print(f"No checkpoint found at {path}, starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch.")
