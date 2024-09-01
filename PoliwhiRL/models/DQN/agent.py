# -*- coding: utf-8 -*-
import os
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from PoliwhiRL.models.DQN.DQNModel import TransformerDQN
from PoliwhiRL.replay import SequenceStorage
from PoliwhiRL.utils.visuals import plot_metrics
from tqdm import tqdm


class PokemonAgent:
    def __init__(self, input_shape, action_size, config, env):
        self.input_shape = input_shape
        self.action_size = action_size
        self.sequence_length = config["sequence_length"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon_start"]
        self.episode = 0
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]
        self.target_update_frequency = config["target_update_frequency"]
        self.batch_size = config["batch_size"]
        self.record = config["record"]
        self.record_path = config["record_path"]
        self.n_goals = config["N_goals_target"]
        self.memory_capacity = config["replay_buffer_capacity"]
        self.env = env
        self.epochs = config["epochs"]
        self.db_path = config["db_path"]
        self.device = torch.device(config["device"])
        print(f"Using device: {self.device}")

        self.model = TransformerDQN(input_shape, action_size).to(self.device)
        self.target_model = TransformerDQN(input_shape, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["learning_rate"]
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.9
        )

        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.replay_buffer = SequenceStorage(
            self.db_path, self.memory_capacity, self.sequence_length
        )

        # Metrics tracking
        self.episode_rewards = []
        self.episode_losses = []
        self.epsilons = []
        self.moving_avg_reward = deque(maxlen=100)
        self.moving_avg_loss = deque(maxlen=100)
        self.episode_steps = []
        self.moving_avg_steps = deque(maxlen=100)
        self.buttons_pressed = deque(maxlen=1000)
        self.buttons_pressed.append(0)

    def calculate_cumulative_rewards(self, rewards, dones, gamma):
        """
        Calculate cumulative discounted rewards for each step in the sequence.
        """
        batch_size, seq_len = rewards.shape
        cumulative_rewards = torch.zeros_like(rewards)
        future_reward = torch.zeros(batch_size, device=rewards.device)

        for t in reversed(range(seq_len)):
            future_reward = rewards[:, t] + gamma * future_reward * (~dones[:, t])
            cumulative_rewards[:, t] = future_reward

        return cumulative_rewards

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0

        # Sample a batch of sequences
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return 0

        states, actions, rewards, next_states, dones, sequence_ids, weights = batch

        # Move everything to the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # Compute Q-values for all states in the sequence
        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            # Double DQN: Use online network to select actions
            next_q_values_online = self.model(next_states)
            next_actions = next_q_values_online.max(2)[1]

            # Use target network to evaluate the Q-values of selected actions
            next_q_values_target = self.target_model(next_states)
            next_q_values = next_q_values_target.gather(
                2, next_actions.unsqueeze(-1)
            ).squeeze(-1)

            # Calculate cumulative rewards
            cumulative_rewards = self.calculate_cumulative_rewards(
                rewards, dones, self.gamma
            )

            # Compute target Q-values for all steps in the sequence
            target_q_values = cumulative_rewards + self.gamma * next_q_values * (~dones)

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Apply importance sampling weights
        loss = (loss.mean(dim=1) * weights).mean()

        # Backpropagate and optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities in the replay buffer
        td_errors = (
            torch.abs(current_q_values - target_q_values)
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        self.replay_buffer.update_priorities(sequence_ids, td_errors)

        return loss.item()

    def step(self, state, eval_mode=False):
        action = self.get_action(state, eval_mode)
        next_state, reward, done, _ = self.env.step(action)
        self.buttons_pressed.append(action)
        self.replay_buffer.add(state, action, reward, next_state, done)
        return next_state, reward, done

    def run_episode(self):
        state = self.env.reset()
        episode_reward = 0
        episode_loss = 0
        done = False
        loss = 0
        steps = 0

        while not done:
            state, reward, done = self.step(
                state, eval_mode=True if self.episode % 10 == 0 else False
            )
            episode_reward += reward
            steps += 1
            if self.record and self.episode % 10 == 0:
                self.env.record(f"DQN_{self.n_goals}")

        for _ in range(self.epochs):
            loss += self.train()
            episode_loss += loss
        loss /= self.epochs
        episode_loss /= self.epochs

        self.episode_rewards.append(episode_reward)
        self.moving_avg_reward.append(episode_reward)
        self.episode_steps.append(steps)
        self.moving_avg_steps.append(steps)
        self.episode_losses.append(episode_loss)
        self.moving_avg_loss.append(episode_loss)

        self.epsilons.append(self.epsilon)
        self.decay_epsilon()

    def get_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state)

        # The model outputs Q-values for each action at each time step
        # We're interested in the Q-values for the last time step
        q_values = q_values[0, -1, :]  # Shape: (action_size,)

        if eval_mode:
            # During evaluation, choose the action with the highest Q-value
            return q_values.argmax().item()
        else:
            # During training, use a softer action selection
            temperature = 1.0  # Adjust this value to control exploration
            probs = F.softmax(q_values / temperature, dim=0)
            return torch.multinomial(probs, 1).item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_agent(self, num_episodes):
        pbar = tqdm(range(num_episodes), desc="Training")
        for n in pbar:
            self.episode = n
            self.run_episode()

            if self.episode % 10 == 0:
                self.report_progress()

            if self.episode % self.target_update_frequency == 0:
                self.update_target_model()

            # Update tqdm progress bar with average reward and steps
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

    def report_progress(self):
        plot_metrics(
            self.episode_rewards,
            self.episode_losses,
            self.episode_steps,
            self.buttons_pressed,
            self.n_goals,
            save_loc=self.record_path,
        )

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
