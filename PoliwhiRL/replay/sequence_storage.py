# -*- coding: utf-8 -*-
import numpy as np
import torch


class SequenceStorage:
    def __init__(
        self,
        capacity,
        sequence_length,
        device,
        state_shape,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001,
    ):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device

        # Pre-allocate memory for experiences
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        self.episode_boundaries = []
        self.current_size = 0
        self.next_index = 0

    def add(self, state, action, reward, done):
        # Add experience to pre-allocated arrays
        self.states[self.next_index] = state
        self.actions[self.next_index] = action
        self.rewards[self.next_index] = reward
        self.dones[self.next_index] = done

        # Update priorities
        self.priorities[self.next_index] = self.max_priority

        # Update episode boundaries if necessary
        if done or not self.episode_boundaries or self.next_index == 0:
            self.episode_boundaries.append(self.next_index)

        # Update indices
        self.next_index = (self.next_index + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample(self, batch_size):
        if self.current_size < self.sequence_length:
            return None

        valid_indices = self.current_size - self.sequence_length + 1
        priorities = self.priorities[:valid_indices] ** self.alpha
        probabilities = priorities / priorities.sum()

        sampled_indices = np.random.choice(
            valid_indices, batch_size, p=probabilities, replace=False
        )

        batch = self._get_sequences(sampled_indices)
        weights = (valid_indices * probabilities[sampled_indices]) ** -self.beta
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return (*batch, sampled_indices, torch.FloatTensor(weights).to(self.device))

    def _get_sequences(self, indices):
        batch_size = len(indices)
        states = np.zeros(
            (batch_size, self.sequence_length, *self.states.shape[1:]), dtype=np.float32
        )
        actions = np.zeros((batch_size, self.sequence_length - 1), dtype=np.int64)
        rewards = np.zeros((batch_size, self.sequence_length - 1), dtype=np.float32)
        next_states = np.zeros(
            (batch_size, self.sequence_length, *self.states.shape[1:]), dtype=np.float32
        )
        dones = np.zeros((batch_size, self.sequence_length), dtype=bool)

        for i, start_idx in enumerate(indices):
            end_idx = start_idx + self.sequence_length
            wrap_idx = end_idx - self.capacity

            if wrap_idx > 0:
                states[i] = np.concatenate(
                    [self.states[start_idx:], self.states[:wrap_idx]], axis=0
                )
                actions[i] = np.concatenate(
                    [self.actions[start_idx:], self.actions[: wrap_idx - 1]], axis=0
                )
                rewards[i] = np.concatenate(
                    [self.rewards[start_idx:], self.rewards[: wrap_idx - 1]], axis=0
                )
                next_states[i] = np.concatenate(
                    [self.states[start_idx + 1 :], self.states[: wrap_idx + 1]], axis=0
                )
                dones[i] = np.concatenate(
                    [self.dones[start_idx:], self.dones[:wrap_idx]], axis=0
                )
            else:
                states[i] = self.states[start_idx:end_idx]
                actions[i] = self.actions[start_idx : end_idx - 1]
                rewards[i] = self.rewards[start_idx : end_idx - 1]
                next_states[i] = self.states[start_idx + 1 : end_idx + 1]
                dones[i] = self.dones[start_idx:end_idx]

        return (
            torch.FloatTensor(states).to(self.device),
            torch.LongTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.BoolTensor(dones).to(self.device),
        )

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + 1e-5) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def __len__(self):
        return self.current_size

    def get_max_priority_sequences(self):
        if self.current_size < self.sequence_length:
            return None

        valid_indices = self.current_size - self.sequence_length + 1
        max_priority = self.priorities[:valid_indices].max()
        max_priority_indices = np.where(
            self.priorities[:valid_indices] == max_priority
        )[0]

        batch = self._get_sequences(max_priority_indices)
        return (*batch, max_priority_indices)
