
from collections import deque
import numpy as np
import torch


class PrioritizedReplayBuffer:
    def __init__(self, capacity, sequence_length, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.episode_buffer = []

    def add(self, state, action, reward, next_state, done, lstm_state):
        self.episode_buffer.append((state, action, reward, next_state, done, lstm_state))

        if done:
            for i in range(0, len(self.episode_buffer) - self.sequence_length + 1):
                sequence = self.episode_buffer[i:i+self.sequence_length]
                initial_lstm_state = sequence[0][5]
                self.buffer.append((sequence, initial_lstm_state))
                self.priorities.append(max(self.priorities, default=1))  # New experiences get max priority
            self.episode_buffer = []

    def sample(self, batch_size):
        total_priority = sum(self.priorities)
        probabilities = np.array(self.priorities) / total_priority
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        sampled_sequences = [self.buffer[idx] for idx in indices]
        
        sequences = [seq[0] for seq in sampled_sequences]
        initial_lstm_states = [seq[1] for seq in sampled_sequences]

        states = torch.FloatTensor(np.array([step[0] for seq in sequences for step in seq]))
        actions = torch.LongTensor(np.array([step[1] for seq in sequences for step in seq]))
        rewards = torch.FloatTensor(np.array([step[2] for seq in sequences for step in seq]))
        next_states = torch.FloatTensor(np.array([step[3] for seq in sequences for step in seq]))
        dones = torch.FloatTensor(np.array([step[4] for seq in sequences for step in seq]))

        initial_lstm_states = (torch.cat([s[0] for s in initial_lstm_states], dim=1),
                               torch.cat([s[1] for s in initial_lstm_states], dim=1))

        states = states.view(batch_size, self.sequence_length, *states.shape[1:])
        actions = actions.view(batch_size, self.sequence_length)
        rewards = rewards.view(batch_size, self.sequence_length)
        next_states = next_states.view(batch_size, self.sequence_length, *next_states.shape[1:])
        dones = dones.view(batch_size, self.sequence_length)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** -self.beta
        weights /= weights.max()
        weights = torch.FloatTensor(weights)

        self.beta = min(1.0, self.beta + self.beta_increment)

        return states, actions, rewards, next_states, dones, initial_lstm_states, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + 1e-5) ** self.alpha  # Small constant to avoid zero priority

    def __len__(self):
        return len(self.buffer)
