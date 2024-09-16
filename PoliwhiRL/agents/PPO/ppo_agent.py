# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from tqdm import tqdm

from PoliwhiRL.agents.base_agent import BaseAgent
from PoliwhiRL.models.PPO.PPOTransformer import PPOTransformer
from PoliwhiRL.models.ICM import ICM
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.utils.visuals import plot_metrics


class PPOAgent(BaseAgent):
    def __init__(self, input_shape, action_size, config):
        super().__init__()
        self.input_shape = input_shape
        self.action_size = action_size
        self.device = torch.device(config["device"])
        self.config = config
        self.update_parameters_from_config()

        self._initialize_networks()
        self._initialize_optimizers()
        self.reset_tracking()

    def _initialize_networks(self):
        self.actor_critic = PPOTransformer(self.input_shape, self.action_size).to(
            self.device
        )
        self.icm = ICM(self.input_shape, self.action_size).to(self.device)

    def _initialize_optimizers(self):
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self.learning_rate
        )
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=self.learning_rate)
        self._setup_lr_scheduler()

    def _setup_lr_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            verbose=True,
            threshold=self.lr_scheduler_threshold,
            min_lr=self.lr_scheduler_min_lr,
        )

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
        self.curiosity_weight = self.config["curiosity_weight"]
        self.value_loss_coef = self.config["value_loss_coef"]
        self.entropy_coef = self.config["entropy_coef"]
        self.icm_loss_scale = self.config["icm_loss_scale"]
        self.lr_scheduler_patience = self.config["lr_scheduler_patience"]
        self.lr_scheduler_factor = self.config["lr_scheduler_factor"]
        self.lr_scheduler_min_lr = self.config["lr_scheduler_min_lr"]
        self.lr_scheduler_threshold = self.config["lr_scheduler_threshold"]
        self.extrinsic_reward_weight = self.config["extrinsic_reward_weight"]
        self.intrinsic_reward_weight = self.config["intrinsic_reward_weight"]

    def reset_tracking(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.episode_icm_losses = []
        self.moving_avg_reward = deque(maxlen=100)
        self.moving_avg_length = deque(maxlen=100)
        self.moving_avg_loss = deque(maxlen=100)
        self.moving_avg_icm_loss = deque(maxlen=100)
        self.buttons_pressed = deque(maxlen=1000)
        self.buttons_pressed.append(0)

        self.max_episode_length = self.config["episode_length"]
        self._initialize_episode_buffers()

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
        for n in range(start_goal_n, end_goal_n + 1):
            self.config["N_goals_target"] = n
            self.config["episode_length"] = self.config["episode_length"] + (
                step_increment * (n - 1)
            )
            self.config["early_stopping_avg_length"] = (
                self.config["episode_length"] // 2
            )
            self.update_parameters_from_config()
            self.train_agent()
            self.reset_tracking()

    def train_agent(self):
        pbar = tqdm(range(self.num_episodes), desc=f"Training (Goals: {self.n_goals})")
        for self.episode in pbar:
            record_loc = (
                f"N_goals_{self.n_goals}/{self.episode}"
                if self.episode % self.record_frequency == 0
                else None
            )
            episode_length = self.run_episode(record_loc=record_loc)
            self.update_model(episode_length)
            self._update_progress_bar()

            avg_reward = (
                np.mean(self.moving_avg_reward) if self.moving_avg_reward else 0
            )
            self.scheduler.step(avg_reward)

            if self._should_stop_early():
                break

        self.run_episode(
            save_path=f"{self.export_state_loc}/N_goals_{self.n_goals}.pkl"
        )

    def _should_stop_early(self):
        if (
            np.mean(self.moving_avg_length) < self.early_stopping_avg_length
            and self.early_stopping_avg_length > 0
            and self.episode > 50
        ):
            print(
                "Average Steps are below early stopping threshold! Stopping to prevent overfitting."
            )
            return True
        return False

    def run_episode(self, record_loc=None, save_path=None):
        env = Env(self.config)
        state = env.reset()
        episode_length = 0
        state_sequence = deque(
            [state] * self.sequence_length, maxlen=self.sequence_length
        )
        if record_loc is not None:
            env.enable_record(record_loc, False)

        for step in range(self.max_episode_length):
            action = self.get_action(np.array(state_sequence))
            next_state, extrinsic_reward, done, _ = env.step(action)

            intrinsic_reward = self._compute_intrinsic_reward(state, next_state, action)
            total_reward = self._compute_total_reward(
                extrinsic_reward, intrinsic_reward
            )

            log_prob = self._compute_log_prob(state_sequence, action)

            self._store_transition(
                state,
                next_state,
                action,
                total_reward,
                intrinsic_reward,
                done,
                log_prob,
                step,
            )

            state = next_state
            state_sequence.append(state)
            episode_length += 1

            if done:
                break

        self._update_episode_stats(episode_length)

        if save_path is not None:
            env.save_gym_state(save_path)

        return episode_length

    def _compute_intrinsic_reward(self, state, next_state, action):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)

        with torch.no_grad():
            _, pred_next_state_feature, encoded_next_state = self.icm(
                state_tensor, next_state_tensor, action_tensor
            )
            intrinsic_reward = (
                self.curiosity_weight
                * torch.mean(
                    torch.square(pred_next_state_feature - encoded_next_state)
                ).item()
            )

        return intrinsic_reward

    def _compute_total_reward(self, extrinsic_reward, intrinsic_reward):
        weighted_extrinsic_reward = self.extrinsic_reward_weight * extrinsic_reward
        weighted_intrinsic_reward = self.intrinsic_reward_weight * intrinsic_reward
        return weighted_extrinsic_reward + weighted_intrinsic_reward

    def _compute_log_prob(self, state_sequence, action):
        state_tensor = (
            torch.FloatTensor(np.array(state_sequence)).unsqueeze(0).to(self.device)
        )
        action_probs, _ = self.actor_critic(state_tensor)
        return torch.log(action_probs[0, action]).item()

    def _store_transition(
        self,
        state,
        next_state,
        action,
        total_reward,
        intrinsic_reward,
        done,
        log_prob,
        step,
    ):
        self.states[step] = state
        self.next_states[step] = next_state
        self.actions[step] = action
        self.rewards[step] = total_reward
        self.intrinsic_rewards[step] = intrinsic_reward
        self.dones[step] = done
        self.log_probs[step] = log_prob

    def _update_episode_stats(self, episode_length):
        self.episode_rewards.append(sum(self.rewards[:episode_length]))
        self.episode_lengths.append(episode_length)
        self.moving_avg_reward.append(sum(self.rewards[:episode_length]))
        self.moving_avg_length.append(episode_length)

    def update_model(self, episode_length):
        total_loss = 0
        total_icm_loss = 0
        for _ in range(self.epochs):
            for idx in range(
                0, episode_length - self.sequence_length + 1, self.batch_size
            ):
                batch_end = min(
                    idx + self.batch_size, episode_length - self.sequence_length + 1
                )
                batch_indices = np.arange(idx, batch_end)

                batch_data = self._prepare_batch_data(batch_indices)
                actor_loss, critic_loss, entropy_loss = self._compute_ppo_losses(
                    batch_data
                )
                icm_loss = self._compute_icm_loss(batch_data)

                loss = actor_loss + critic_loss + entropy_loss
                total_loss += loss.item()
                total_icm_loss += icm_loss.item()

                self._update_networks(loss, icm_loss)

        self._update_loss_stats(total_loss, total_icm_loss, episode_length)

    def _prepare_batch_data(self, batch_indices):
        return {
            "states": self._get_state_sequences(batch_indices),
            "next_states": self._get_state_sequences(batch_indices, next_state=True),
            "actions": torch.LongTensor(
                self.actions[batch_indices + self.sequence_length - 1]
            ).to(self.device),
            "rewards": torch.FloatTensor(
                self.rewards[batch_indices + self.sequence_length - 1]
            ).to(self.device),
            "dones": torch.BoolTensor(
                self.dones[batch_indices + self.sequence_length - 1]
            ).to(self.device),
            "old_log_probs": torch.FloatTensor(
                self.log_probs[batch_indices + self.sequence_length - 1]
            ).to(self.device),
        }

    def _compute_ppo_losses(self, batch_data):
        returns = self._compute_returns(batch_data["rewards"], batch_data["dones"])
        advantages = self._compute_advantages(batch_data["states"], returns)

        new_probs, new_values = self.actor_critic(batch_data["states"])
        new_log_probs = torch.log(
            new_probs.gather(1, batch_data["actions"].unsqueeze(1))
        ).squeeze()

        ratio = torch.exp(new_log_probs - batch_data["old_log_probs"])
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = self.value_loss_coef * nn.MSELoss()(new_values.squeeze(), returns)

        entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy

        return actor_loss, critic_loss, entropy_loss

    def _compute_icm_loss(self, batch_data):
        pred_actions, pred_next_state_features, encoded_next_states = self.icm(
            batch_data["states"][:, -1],
            batch_data["next_states"][:, -1],
            batch_data["actions"],
        )
        inverse_loss = nn.CrossEntropyLoss()(pred_actions, batch_data["actions"])
        forward_loss = nn.MSELoss()(pred_next_state_features, encoded_next_states)
        return (inverse_loss + forward_loss) * self.icm_loss_scale

    def _update_networks(self, ppo_loss, icm_loss):
        self.optimizer.zero_grad()
        ppo_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

    def _update_loss_stats(self, total_loss, total_icm_loss, episode_length):
        avg_loss = total_loss / (self.epochs * (episode_length / self.batch_size))
        avg_icm_loss = total_icm_loss / (
            self.epochs * (episode_length / self.batch_size)
        )
        self.episode_losses.append(avg_loss)
        self.episode_icm_losses.append(avg_icm_loss)
        self.moving_avg_loss.append(avg_loss)
        self.moving_avg_icm_loss.append(avg_icm_loss)

    def _get_state_sequences(self, indices, next_state=False):
        if next_state:
            sequences = np.array(
                [self.next_states[i : i + self.sequence_length] for i in indices]
            )
        else:
            sequences = np.array(
                [self.states[i : i + self.sequence_length] for i in indices]
            )
        return torch.FloatTensor(sequences).to(self.device)

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

    def _update_progress_bar(self):
        avg_reward = np.mean(self.moving_avg_reward) if self.moving_avg_reward else 0
        avg_length = np.mean(self.moving_avg_length) if self.moving_avg_length else 0
        avg_loss = np.mean(self.moving_avg_loss) if self.moving_avg_loss else 0
        avg_icm_loss = (
            np.mean(self.moving_avg_icm_loss) if self.moving_avg_icm_loss else 0
        )
        current_reward = self.episode_rewards[-1] if self.episode_rewards else 0
        current_length = self.episode_lengths[-1] if self.episode_lengths else 0
        current_lr = self.optimizer.param_groups[0]["lr"]

        tqdm.write(
            f"Episode: {self.episode}, "
            f"Avg Reward (100 ep): {avg_reward:.2f}, "
            f"Avg Length (100 ep): {avg_length:.2f}, "
            f"Avg Loss (100 ep): {avg_loss:.4f}, "
            f"Avg ICM Loss (100 ep): {avg_icm_loss:.4f}, "
            f"Current Reward: {current_reward:.2f}, "
            f"Current Length: {current_length}, "
            f"Learning Rate: {current_lr:.2e}"
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
