# -*- coding: utf-8 -*-
import torch


class EpisodeMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.hidden_states = []

    def store(self, state, action, log_prob, value, reward, done, hidden_state):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value.unsqueeze(0))
        self.rewards.append(reward)
        self.dones.append(done)
        self.hidden_states.append(hidden_state)

    def clear(self):
        self.__init__()

    def get_batch(self):
        return (
            torch.cat(self.states),
            torch.cat(self.actions),
            torch.cat(self.log_probs),
            torch.cat(self.values),
            self.rewards,
            self.dones,
            self.hidden_states,
        )
