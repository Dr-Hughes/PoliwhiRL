# -*- coding: utf-8 -*-
import numpy as np
import torch


class SequenceStorage:
    def __init__(
        self,
        capacity,
        sequence_length,
        state_shape,
        max_episode_length,
        device,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001,
    ):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.max_episode_length = max_episode_length
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device

        # Pre-allocate memory for experiences
        self.states = np.zeros(
            (capacity, max_episode_length) + state_shape, dtype=np.uint8
        )
        self.actions = np.zeros((capacity, max_episode_length), dtype=np.int64)
        self.rewards = np.zeros((capacity, max_episode_length), dtype=np.float32)
        self.next_states = np.zeros(
            (capacity, max_episode_length) + state_shape, dtype=np.uint8
        )
        self.dones = np.zeros((capacity, max_episode_length), dtype=np.bool_)

        self.episode_lengths = np.zeros(capacity, dtype=np.int32)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

        self.current_episode_idx = 0
        self.current_step = 0
        self.num_episodes = 0

    def add(self, state, action, reward, next_state, done):
        if self.current_step >= self.max_episode_length:
            print(
                f"Warning: Episode exceeded max length of {self.max_episode_length}. Skipping this experience."
            )
            return

        self.states[self.current_episode_idx, self.current_step] = state
        self.actions[self.current_episode_idx, self.current_step] = action
        self.rewards[self.current_episode_idx, self.current_step] = reward
        self.next_states[self.current_episode_idx, self.current_step] = next_state
        self.dones[self.current_episode_idx, self.current_step] = done

        self.current_step += 1

        if done or self.current_step == self.max_episode_length:
            self.episode_lengths[self.current_episode_idx] = self.current_step
            self.priorities[self.current_episode_idx] = self.max_priority
            self.num_episodes = min(self.num_episodes + 1, self.capacity)
            self.current_episode_idx = (self.current_episode_idx + 1) % self.capacity
            self.current_step = 0

    def sample(self, batch_size):
        if self.num_episodes < batch_size:
            return None

        valid_episodes = np.where(self.episode_lengths >= self.sequence_length)[0]
        if len(valid_episodes) < batch_size:
            return None

        probabilities = self.priorities[valid_episodes] / np.sum(
            self.priorities[valid_episodes]
        )
        sampled_episode_indices = np.random.choice(
            valid_episodes, batch_size, p=probabilities, replace=False
        )

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for episode_idx in sampled_episode_indices:
            episode_length = self.episode_lengths[episode_idx]
            start_idx = np.random.randint(0, episode_length - self.sequence_length + 1)
            end_idx = start_idx + self.sequence_length

            states.append(self.states[episode_idx, start_idx:end_idx])
            actions.append(self.actions[episode_idx, start_idx:end_idx])
            rewards.append(self.rewards[episode_idx, start_idx:end_idx])
            next_states.append(self.next_states[episode_idx, start_idx:end_idx])
            dones.append(self.dones[episode_idx, start_idx:end_idx])

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)

        weights = (
            len(valid_episodes)
            * probabilities[np.isin(valid_episodes, sampled_episode_indices)]
        ) ** -self.beta
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device)

        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            sampled_episode_indices,
            weights,
        )

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + 1e-5) ** self.alpha
        self.max_priority = max(self.max_priority, np.max(self.priorities))

    def __len__(self):
        return sum(self.episode_lengths[: self.num_episodes])

    def get_max_priority_sequences_generator(self, batch_size):
        valid_episodes = np.where(self.episode_lengths >= self.sequence_length)[0]
        if len(valid_episodes) == 0:
            return None

        max_priority = np.max(self.priorities[valid_episodes])
        max_priority_indices = valid_episodes[
            self.priorities[valid_episodes] == max_priority
        ]

        np.random.shuffle(max_priority_indices)

        def generate_batches():
            for i in range(0, len(max_priority_indices), batch_size):
                batch_indices = max_priority_indices[i:i+batch_size]
                yield self._get_sequences_batch(batch_indices)

        return generate_batches()

    def _get_sequences_batch(self, batch_indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for episode_idx in batch_indices:
            episode_length = self.episode_lengths[episode_idx]
            start_idx = np.random.randint(0, episode_length - self.sequence_length + 1)
            end_idx = start_idx + self.sequence_length

            states.append(self.states[episode_idx, start_idx:end_idx])
            actions.append(self.actions[episode_idx, start_idx:end_idx])
            rewards.append(self.rewards[episode_idx, start_idx:end_idx])
            next_states.append(self.next_states[episode_idx, start_idx:end_idx])
            dones.append(self.dones[episode_idx, start_idx:end_idx])

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)

        return states, actions, rewards, next_states, dones, batch_indices

