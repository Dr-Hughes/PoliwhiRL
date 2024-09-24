# -*- coding: utf-8 -*-
from collections import deque
import math
import numpy as np
import torch
import torch.nn as nn
from PoliwhiRL.models.DQN import TransformerDQN
from PoliwhiRL.replay import SequenceStorage
from PoliwhiRL.utils.visuals import plot_metrics
from tqdm import tqdm
from .multi_agent import ParallelAgentRunner
from .dqn_shared import DQNSharedAgent


class DQNPokemonAgent(DQNSharedAgent):
    def __init__(self, input_shape, action_size, config, load_checkpoint=True):
        self.input_shape = input_shape
        self.action_size = action_size
        self.update_parameters_from_config(config)
        print(f"Using device: {self.device}")
        self.model = TransformerDQN(input_shape, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.9
        )
        if load_checkpoint:
            self.load_model()
        self.target_model = TransformerDQN(input_shape, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.reset_replay_buffer()
        self.episode_rewards = []
        self.episode_losses = []
        self.moving_avg_reward = deque(maxlen=100)
        self.moving_avg_loss = deque(maxlen=100)
        self.episode_steps = []
        self.moving_avg_steps = deque(maxlen=100)
        self.buttons_pressed = deque(maxlen=1000)
        self.buttons_pressed.append(0)

        if self.num_agents > 1:
            self.parallel_runner = ParallelAgentRunner(self.model)
            self.temperatures = np.linspace(
                self.max_temperature,
                self.min_temperature,
                self.num_agents,
            )
        else:
            self.temperatures = [
                self.get_cyclical_temperature(
                    self.temperature_cycle_length,
                    self.min_temperature,
                    self.max_temperature,
                    i,
                )
                for i in range(self.num_episodes)
            ]

    def get_cyclical_temperature(
        self, temperature_cycle_length, min_temperature, max_temperature, i
    ):
        cycle_progress = (i % temperature_cycle_length) / temperature_cycle_length
        return (
            min_temperature
            + (max_temperature - min_temperature)
            * (math.cos(cycle_progress * 2 * math.pi) + 1)
            / 2
        )

    def update_parameters_from_config(self, config):
        self.config = config
        self.episode = 0
        self.num_episodes = self.config["num_episodes"]
        self.sequence_length = self.config["sequence_length"]
        self.gamma = self.config["dqn_gamma"]
        self.num_agents = self.config["dqn_num_agents"]
        self.min_temperature = self.config["dqn_min_temperature"]
        self.max_temperature = self.config["dqn_max_temperature"]
        self.temperature_cycle_length = self.config["dqn_temperature_cycle_length"]
        self.early_stopping_avg_length = self.config["early_stopping_avg_length"]
        self.record_frequency = self.config["record_frequency"]
        self.learning_rate = self.config["dqn_learning_rate"]
        self.target_update_frequency = self.config["dqn_target_update_frequency"]
        self.batch_size = self.config["dqn_batch_size"]
        self.record = self.config["record"]
        self.record_path = self.config["record_path"]
        self.results_dir = self.config["results_dir"]
        self.n_goals = self.config["N_goals_target"]
        self.memory_capacity = self.config["dqn_replay_buffer_capacity"]
        self.epochs = self.config["dqn_epochs"]
        self.db_path = self.config["db_path"]
        self.device = torch.device(config["device"])
        self.checkpoint = self.config["checkpoint"]
        self.continue_from_state_loc = self.config["continue_from_state_loc"]
        self.continue_from_state = True if self.continue_from_state_loc != "" else False
        self.export_state_loc = self.config["export_state_loc"]
        self.num_random_episodes = self.config["dqn_num_random_episodes"]

    def reset_replay_buffer(self):
        self.replay_buffer = SequenceStorage(
            self.memory_capacity,
            self.sequence_length,
            self.model.input_shape,
            self.config["episode_length"],
            self.device,
        )

    def run_ciriculum(self, start_goal_n, end_goal_n, step_increment):
        for n in range(start_goal_n, end_goal_n):
            self.config["N_goals_target"] = n
            self.config["early_stopping_avg_length"] = self.config[
                "early_stopping_avg_length"
            ] + (step_increment * (n - 1))
            self.update_parameters_from_config(self.config)
            self.reset_replay_buffer()
            self.train_agent()

    def train_agent(self):
        pbar = tqdm(range(self.num_episodes), desc="Training")
        for n in pbar:
            self.episode = n
            self._generate_experiences()
            self._update_model()
            self._report_progress(pbar)
            if self.break_condition():
                break

        # Run a final episode to get the export state
        self.run_episode(
            self.model,
            self.config,
            0.1,
            save_path=f"{self.export_state_loc}/N_goals_{self.n_goals}.pkl",
        )
        self.save_model(self.config["checkpoint"])

    def break_condition(self):
        if self.episode < self.num_random_episodes:
            return False
        if (
            np.mean(self.moving_avg_steps) < self.early_stopping_avg_length
            and self.early_stopping_avg_length > 0
            and self.episode > 50
        ):
            print(
                "Average Steps are below early stopping threshold! Stopping to prevent overfitting."
            )
            return True
        return False

    def _calculate_cumulative_rewards(self, rewards, dones, gamma):
        batch_size, seq_len = rewards.shape
        cumulative_rewards = torch.zeros_like(rewards)
        future_reward = torch.zeros(batch_size, device=rewards.device)
        for t in reversed(range(seq_len)):
            future_reward = rewards[:, t] + gamma * future_reward * (~dones[:, t])
            cumulative_rewards[:, t] = future_reward

        return cumulative_rewards

    def _train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0

        total_loss = 0
        num_batches = 0

        # Train on max priority sequences first
        max_priority_generator = (
            self.replay_buffer.get_max_priority_sequences_generator(self.batch_size)
        )
        if max_priority_generator is not None:
            for max_priority_batch in max_priority_generator:
                loss = self._train_on_batch(max_priority_batch, is_max_priority=True)
                total_loss += loss
                num_batches += 1

        # Then proceed with regular training
        regular_batch = self.replay_buffer.sample(self.batch_size)
        if regular_batch is not None:
            loss = self._train_on_batch(regular_batch, is_max_priority=False)
            total_loss += loss
            num_batches += 1

        if num_batches > 0:
            return total_loss / num_batches
        else:
            return 0

    def _train_on_batch(self, batch, is_max_priority=False):
        if is_max_priority:
            states, actions, rewards, next_states, dones, sequence_ids = batch
            weights = torch.ones(self.batch_size, device=self.device)
        else:
            states, actions, rewards, next_states, dones, sequence_ids, weights = batch

        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values_online = self.model(next_states)
            next_actions = next_q_values_online.max(2)[1]
            next_q_values_target = self.target_model(next_states)
            next_q_values = next_q_values_target.gather(
                2, next_actions.unsqueeze(-1)
            ).squeeze(-1)
            cumulative_rewards = self._calculate_cumulative_rewards(
                rewards, dones, self.gamma
            )
            target_q_values = cumulative_rewards + self.gamma * next_q_values * (~dones)

        loss = self.loss_fn(current_q_values, target_q_values)
        if is_max_priority:
            loss = loss.mean()
        else:
            loss = (loss.mean(dim=1) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        if not is_max_priority:
            self.scheduler.step()

        td_errors = (
            torch.abs(current_q_values - target_q_values)
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        self.replay_buffer.update_priorities(sequence_ids, td_errors)

        return loss.item()

    def _add_episode(self, episode, temperature, steps):
        actions = []
        rewards = []
        for experience in episode:
            state, action, reward, next_state, done = experience
            actions.append(action)
            rewards.append(reward)
            self.replay_buffer.add(state, action, reward, next_state, done)
        if temperature == self.min_temperature:
            self._update_monitoring_stats(actions, sum(rewards), steps)

    def _run_multiple_episodes(self, temperatures, record_path=None, load_path=None):
        self.parallel_runner.update_shared_model(self.model)
        episode_experiences = self.parallel_runner.run_agents(
            self.config, temperatures, record_path, load_path
        )
        for episode, temperature, steps in episode_experiences:
            self._add_episode(episode, temperature, steps)

    def _update_monitoring_stats(self, actions, episode_reward, steps):
        for action in actions:
            self.buttons_pressed.append(action)
        self.episode_rewards.append(episode_reward)
        self.moving_avg_reward.append(episode_reward)
        self.episode_steps.append(steps)
        self.moving_avg_steps.append(steps)

    def _update_model(self):
        total_loss = 0
        for _ in range(self.epochs):
            loss = self._train()
            total_loss += loss
        self.episode_losses.append(total_loss)
        self.moving_avg_loss.append(total_loss)
        self._update_target_model()

    def _update_target_model(self):
        if self.episode % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def _generate_experiences(self):
        if (
            self.episode % self.record_frequency == 0
            and self.episode > self.num_random_episodes
        ):
            record_loc = f"N_goals_{self.n_goals}/{self.episode}"
        else:
            record_loc = None

        if self.episode < self.num_random_episodes:
            self.tmp_temperatures = self.temperatures
            self.temperatures = [-1 for _ in self.temperatures]

        if self.num_agents > 1:
            self._run_multiple_episodes(
                self.temperatures,
                record_loc,
                load_path=(
                    self.continue_from_state_loc if self.continue_from_state else None
                ),
            )
        else:
            episode, temperature, steps = self.run_episode(
                self.model,
                self.config,
                self.temperatures[self.episode],
                record_loc,
                (self.continue_from_state_loc if self.continue_from_state else None),
            )
            self._add_episode(episode, temperature, steps)

        self.temperatures = self.tmp_temperatures

    def _report_progress(self, pbar):
        if self.episode < self.num_random_episodes:
            return
        avg_reward = (
            sum(self.moving_avg_reward) / len(self.moving_avg_reward)
            if self.moving_avg_reward
            else 0
        )
        avg_steps = (
            sum(self.moving_avg_steps) / len(self.moving_avg_steps)
            if self.moving_avg_steps
            else 0
        )
        pbar.set_postfix(
            {
                "Avg Reward (100 ep)": f"{avg_reward:.2f}",
                "Avg Steps (100 ep)": f"{avg_steps:.2f}",
            }
        )

        if self.episode % 10 == 0 and self.episode > 100:
            plot_metrics(
                self.episode_rewards,
                self.episode_losses,
                self.episode_steps,
                self.buttons_pressed,
                self.n_goals,
                save_loc=self.results_dir,
            )
