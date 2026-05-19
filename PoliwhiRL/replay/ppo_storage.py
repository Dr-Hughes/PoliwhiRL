# -*- coding: utf-8 -*-
import numpy as np
import torch


class PPOMemory:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.update_frequency = config["ppo_update_frequency"]
        self.sequence_length = config["sequence_length"]
        self.input_shape = config["input_shape"]
        self.ram_obs_dim = int(config["ram_obs_dim"])
        self.reset()

    def reset(self):
        self.states = np.zeros(
            (self.update_frequency,) + self.input_shape, dtype=np.uint8
        )
        self.ram_states = np.zeros(
            (self.update_frequency, self.ram_obs_dim), dtype=np.float32
        )
        self.actions = np.zeros(self.update_frequency, dtype=np.uint8)
        self.rewards = np.zeros(self.update_frequency, dtype=np.float32)
        self.dones = np.zeros(self.update_frequency, dtype=np.bool_)
        self.log_probs = np.zeros(self.update_frequency, dtype=np.float32)
        # Allocated lazily on first store_transition once we know the mems shape.
        self.mems = None
        self.last_next_state = None
        self.last_next_ram = None
        self.episode_length = 0

    def store_transition(
        self,
        state,
        ram,
        next_state,
        next_ram,
        action,
        reward,
        done,
        log_prob,
        mems=None,
    ):
        idx = self.episode_length
        self.states[idx] = state
        self.ram_states[idx] = ram
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        if mems is not None:
            stacked = np.stack(
                [m.detach().squeeze(0).cpu().numpy() for m in mems], axis=0
            )
            if self.mems is None:
                self.mems = np.zeros(
                    (self.update_frequency,) + stacked.shape, dtype=np.float32
                )
            self.mems[idx] = stacked
        self.last_next_state = next_state
        self.last_next_ram = next_ram
        self.episode_length += 1

    def get_all_data(self):
        if self.episode_length < self.sequence_length:
            return None
        num_sequences = self.episode_length - self.sequence_length + 1

        sequences = np.array(
            [self.states[i : i + self.sequence_length] for i in range(num_sequences)]
        )
        ram_sequences = np.array(
            [
                self.ram_states[i : i + self.sequence_length]
                for i in range(num_sequences)
            ]
        )

        tail_start = self.episode_length - self.sequence_length + 1
        next_sequences = np.array(
            [
                self.states[i + 1 : i + self.sequence_length + 1]
                for i in range(num_sequences - 1)
            ]
            + [
                np.concatenate(
                    [
                        self.states[tail_start : self.episode_length],
                        [self.last_next_state],
                    ]
                )
            ]
        )
        next_ram_sequences = np.array(
            [
                self.ram_states[i + 1 : i + self.sequence_length + 1]
                for i in range(num_sequences - 1)
            ]
            + [
                np.concatenate(
                    [
                        self.ram_states[tail_start : self.episode_length],
                        [self.last_next_ram],
                    ]
                )
            ]
        )

        data = {
            "states": torch.FloatTensor(sequences).to(self.device),
            "ram_states": torch.FloatTensor(ram_sequences).to(self.device),
            "next_states": torch.FloatTensor(next_sequences).to(self.device),
            "next_ram_states": torch.FloatTensor(next_ram_sequences).to(self.device),
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

        if self.mems is not None:
            mems_slice = self.mems[self.sequence_length - 1 : self.episode_length]
            num_layers = mems_slice.shape[1]
            data["mems"] = [
                torch.from_numpy(mems_slice[:, layer]).to(self.device)
                for layer in range(num_layers)
            ]

        return data

    def __len__(self):
        return self.episode_length
