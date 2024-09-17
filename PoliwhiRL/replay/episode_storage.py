import numpy as np
import torch

class EpisodeMemory:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.max_episode_length = config["episode_length"]
        self.sequence_length = config["sequence_length"]
        self.input_shape = config["input_shape"]
        self.reset()

    def reset(self):
        self.states = np.zeros((self.max_episode_length,) + self.input_shape, dtype=np.uint8)
        self.next_states = np.zeros((self.max_episode_length,) + self.input_shape, dtype=np.uint8)
        self.actions = np.zeros(self.max_episode_length, dtype=np.int64)
        self.rewards = np.zeros(self.max_episode_length, dtype=np.float32)
        self.dones = np.zeros(self.max_episode_length, dtype=np.bool_)
        self.log_probs = np.zeros(self.max_episode_length, dtype=np.float32)
        self.episode_length = 0

    def store_transition(self, state, next_state, action, reward, done, log_prob):
        idx = self.episode_length
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        self.episode_length += 1

    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.episode_length - self.sequence_length + 1

        indices = np.random.choice(
            self.episode_length - self.sequence_length + 1, 
            size=batch_size, 
            replace=False
        )

        batch_data = {
            "states": self._get_state_sequences(indices),
            "next_states": self._get_state_sequences(indices, next_state=True),
            "actions": torch.LongTensor(self.actions[indices + self.sequence_length - 1]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[indices + self.sequence_length - 1]).to(self.device),
            "dones": torch.BoolTensor(self.dones[indices + self.sequence_length - 1]).to(self.device),
            "old_log_probs": torch.FloatTensor(self.log_probs[indices + self.sequence_length - 1]).to(self.device),
        }

        return batch_data

    def _get_state_sequences(self, indices, next_state=False):
        state_array = self.next_states if next_state else self.states
        sequences = np.array([state_array[i:i + self.sequence_length] for i in indices])
        return torch.FloatTensor(sequences).to(self.device)

    def get_all_data(self):
        return {
            "states": torch.FloatTensor(self.states[:self.episode_length]).to(self.device),
            "next_states": torch.FloatTensor(self.next_states[:self.episode_length]).to(self.device),
            "actions": torch.LongTensor(self.actions[:self.episode_length]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[:self.episode_length]).to(self.device),
            "dones": torch.BoolTensor(self.dones[:self.episode_length]).to(self.device),
            "old_log_probs": torch.FloatTensor(self.log_probs[:self.episode_length]).to(self.device),
        }

    def __len__(self):
        return self.episode_length