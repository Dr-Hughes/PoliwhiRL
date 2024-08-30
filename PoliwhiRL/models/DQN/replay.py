# -*- coding: utf-8 -*-
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

    def add(self, state, action, reward, next_state, done):
        self.episode_buffer.append((state, action, reward, next_state, done))
        
        if done or len(self.episode_buffer) >= self.sequence_length:
            while len(self.episode_buffer) >= self.sequence_length:
                sequence = self.episode_buffer[:self.sequence_length]
                self.buffer.append(sequence)
                self.priorities.append(max(self.priorities, default=1))  # New sequences get max priority
                self.episode_buffer = self.episode_buffer[1:]  # Remove the first element
            
            if done:
                self.episode_buffer = []  # Clear episode buffer at the end of an episode

    def sample(self, batch_size):
        total_priority = sum(self.priorities)
        probabilities = np.array(self.priorities) / total_priority
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            sequence = self.buffer[idx]
            seq_states, seq_actions, seq_rewards, seq_next_states, seq_dones = zip(*sequence)
            
            states.append(np.array(seq_states))
            actions.append(np.array(seq_actions))
            rewards.append(np.array(seq_rewards))
            next_states.append(np.array(seq_next_states))
            dones.append(np.array(seq_dones))
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** -self.beta
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + 1e-5) ** self.alpha  # Small constant to avoid zero priority

    def __len__(self):
        return len(self.buffer)