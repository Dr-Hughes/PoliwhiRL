# -*- coding: utf-8 -*-
import numpy as np
import torch


class EpisodeStorage:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.update_frequency = config["update_frequency"]
        self.sequence_length = config["sequence_length"]
        self.input_shape = config["input_shape"]
        self.reset()

    def reset(self, config=None):
        if config is not None:
            self.__init__(config)
        self.states = np.zeros(
            (self.update_frequency,) + self.input_shape, dtype=np.uint8
        )
        self.actions = np.zeros(self.update_frequency, dtype=np.int64)
        self.rewards = np.zeros(self.update_frequency, dtype=np.float32)
        self.dones = np.zeros(self.update_frequency, dtype=np.bool_)
        self.log_probs = np.zeros(self.update_frequency, dtype=np.float32)
        self.last_next_state = None
        self.episode_length = 0

    def store_transition(self, state, next_state, action, reward, done, log_prob):
        idx = self.episode_length
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        self.last_next_state = next_state
        self.episode_length += 1

    def get_all_data(self):
        if self.episode_length < self.sequence_length:
            return None
        num_sequences = self.episode_length - self.sequence_length + 1
        sequences = np.array(
            [self.states[i : i + self.sequence_length] for i in range(num_sequences)]
        )
        next_sequences = np.array(
            [
                self.states[i + 1 : i + self.sequence_length + 1]
                for i in range(num_sequences - 1)
            ]
            + [
                np.concatenate(
                    [self.states[-self.sequence_length + 1 :], [self.last_next_state]]
                )
            ]
        )

        return {
            "states": torch.FloatTensor(sequences).to(self.device),
            "next_states": torch.FloatTensor(next_sequences).to(self.device),
            "actions": torch.LongTensor(
                self.actions[self.sequence_length - 1 : self.episode_length]
            ).to(self.device),
            "rewards": torch.FloatTensor(
                self.rewards[self.sequence_length - 1 : self.episode_length]
            ).to(self.device),
            "dones": torch.BoolTensor(
                self.dones[self.sequence_length - 1 : self.episode_length]
            ).to(self.device),
            "old_log_probs": torch.FloatTensor(
                self.log_probs[self.sequence_length - 1 : self.episode_length]
            ).to(self.device),
        }

    def __len__(self):
        return self.episode_length
