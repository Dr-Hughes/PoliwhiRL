# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
from PoliwhiRL.models.DQN.DQNModel import DeepQNetworkModel
from PoliwhiRL.models.DQN.replay import PrioritizedReplayBuffer
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
        self.n_goals = config["N_goals_target"]
        self.env = env

        self.device = torch.device(config["device"])

        self.model = DeepQNetworkModel(input_shape, action_size).to(self.device)
        self.target_model = DeepQNetworkModel(input_shape, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["learning_rate"]
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.9
        )

        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=20000, sequence_length=self.sequence_length
        )
        self.train_between_episodes = True  # New flag to control training timing
        self.steps_since_train = 0
        self.train_frequency = 4  # Train every 4 steps if training during episodes

        # Metrics tracking
        self.episode_rewards = []
        self.episode_losses = []
        self.moving_avg_reward = deque(maxlen=100)
        self.moving_avg_loss = deque(maxlen=100)
        self.epsilons = []

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return 0

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            initial_lstm_states,
            indices,
            weights,
        ) = self.replay_buffer.sample(batch_size)

        # Move everything to the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        initial_lstm_states = (
            initial_lstm_states[0].to(self.device),
            initial_lstm_states[1].to(self.device),
        )
        weights = weights.to(self.device)

        # Process entire sequences for current Q-values
        current_q_values, _ = self.model(states, initial_lstm_states)
        current_q_values = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            # Double DQN: Use online network to select actions
            next_q_values_online, _ = self.model(next_states, initial_lstm_states)
            next_actions = next_q_values_online.max(2)[1].unsqueeze(-1)

            # Use target network to evaluate the Q-values of selected actions
            next_q_values_target, _ = self.target_model(
                next_states, initial_lstm_states
            )
            next_q_values = next_q_values_target.gather(2, next_actions).squeeze(-1)

        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute TD error for prioritized replay
        td_error = torch.abs(current_q_values - target_q_values).detach()

        # Compute Huber loss
        loss = self.loss_fn(current_q_values, target_q_values)
        loss = (weights * loss).mean()

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()

        # Update priorities
        td_errors = td_error.mean(dim=1).cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        return loss.item()

    def step(self, state, lstm_state):
        action, new_lstm_state = self.get_action(state, lstm_state)
        next_state, reward, done, _ = self.env.step(action)

        self.replay_buffer.add(state, action, reward, next_state, done, lstm_state)

        self.steps_since_train += 1

        if (
            not self.train_between_episodes
            and self.steps_since_train >= self.train_frequency
        ):
            loss = self.train(self.batch_size)
            self.steps_since_train = 0
            self.episode_losses.append(loss)

        return next_state, reward, done, new_lstm_state

    def run_episode(self):
        state = self.env.reset()
        lstm_state = self.model.init_hidden(batch_size=1)
        episode_reward = 0
        episode_loss = 0
        done = False

        while not done:
            state, reward, done, lstm_state = self.step(state, lstm_state)
            episode_reward += reward
            if self.record and self.episode % 10 == 0:
                self.env.record(f"DQN_{self.n_goals}")
        if self.train_between_episodes:
            loss = self.train(self.batch_size)
            episode_loss = loss

        self.episode_rewards.append(episode_reward)
        self.moving_avg_reward.append(episode_reward)
        if episode_loss > 0:
            self.episode_losses.append(episode_loss)
            self.moving_avg_loss.append(episode_loss)

        self.epsilons.append(self.epsilon)
        self.decay_epsilon()

        return episode_reward, episode_loss

    def get_action(self, state, lstm_state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_size), lstm_state

        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        lstm_state = (lstm_state[0].to(self.device), lstm_state[1].to(self.device))

        with torch.no_grad():
            q_values, new_lstm_state = self.model(state, lstm_state)

        action = q_values.argmax(dim=-1).item()
        new_lstm_state = (new_lstm_state[0].cpu(), new_lstm_state[1].cpu())

        return action, new_lstm_state

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_agent(self, num_episodes):
        for episode in tqdm(range(num_episodes)):
            self.episode = episode
            episode_reward, episode_loss = self.run_episode()

            if episode % 10 == 0:
                avg_reward = np.mean(self.moving_avg_reward)
                avg_loss = np.mean(self.moving_avg_loss) if self.moving_avg_loss else 0
                print(f"Episode: {episode}")
                print(f"Episode Reward: {episode_reward}")
                print(f"Average Reward (last 100 episodes): {avg_reward}")
                print(f"Best Reward: {max(self.episode_rewards)}")
                print(f"Episode Loss: {episode_loss}")
                print(f"Average Loss (last 100 episodes): {avg_loss}")
                print(f"Epsilon: {self.epsilon}")
                print("--------------------")

            if episode % self.target_update_frequency == 0:
                self.update_target_model()

        self.save_model("pokemon_model_final.pth")
        return self.episode_rewards, self.episode_losses, self.epsilons

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
