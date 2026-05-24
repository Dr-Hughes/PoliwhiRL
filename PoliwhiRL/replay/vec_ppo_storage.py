# -*- coding: utf-8 -*-
"""Rollout buffer for vectorised PPO.

Stores a fixed-length (T, N, ...) rollout, then emits sliding-window
sequences of length `sequence_length` per env. The env axis is preserved
in the output so the agent can compute per-env GAE (advantages don't
cross env boundaries) before flattening for the PPO loss.

Observations are multi-modal: image array + RAM vector are kept as
separate buffers and emitted as parallel sliding windows.
"""
import numpy as np
import torch


class VecPPOMemory:
    def __init__(self, config, num_envs):
        self.config = config
        self.device = torch.device(config["device"])
        self.rollout_length = int(config["ppo_update_frequency"])
        self.sequence_length = int(config["sequence_length"])
        self.num_envs = int(num_envs)
        self.input_shape = tuple(config["input_shape"])
        self.ram_obs_dim = int(config["ram_obs_dim"])
        self.reset()

    def reset(self):
        T, N = self.rollout_length, self.num_envs
        self.states = np.zeros((T, N) + self.input_shape, dtype=np.uint8)
        self.ram_states = np.zeros((T, N, self.ram_obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.bool_)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.mems = None  # lazy: shape depends on model
        self.last_next_obs = None  # (N, *input_shape)
        self.last_next_ram = None  # (N, ram_obs_dim)
        self.t = 0

    def __len__(self):
        return self.t

    def is_full(self):
        return self.t >= self.rollout_length

    def store_step(
        self,
        states,
        ram_states,
        next_states,
        next_ram_states,
        actions,
        rewards,
        dones,
        log_probs,
        mems,
    ):
        """Store one timestep's worth of transitions across all envs."""
        if self.t >= self.rollout_length:
            raise RuntimeError("VecPPOMemory full; call reset() before storing.")
        idx = self.t
        self.states[idx] = np.asarray(states, dtype=np.uint8)
        self.ram_states[idx] = np.asarray(ram_states, dtype=np.float32)
        self.actions[idx] = np.asarray(actions, dtype=np.int64)
        self.rewards[idx] = np.asarray(rewards, dtype=np.float32)
        self.dones[idx] = np.asarray(dones, dtype=np.bool_)
        self.log_probs[idx] = np.asarray(log_probs, dtype=np.float32)

        stacked = np.stack([m.detach().cpu().numpy() for m in mems], axis=1)
        if self.mems is None:
            self.mems = np.zeros(
                (self.rollout_length,) + stacked.shape, dtype=np.float32
            )
        self.mems[idx] = stacked

        self.last_next_obs = np.asarray(next_states, dtype=np.uint8)
        self.last_next_ram = np.asarray(next_ram_states, dtype=np.float32)
        self.t += 1

    def get_data(self):
        """Emit sliding-window data with env axis preserved.

        Returns dict with shapes:
          states:      (W, N, seq_len, *input_shape)
          ram_states:  (W, N, seq_len, ram_obs_dim)
          next_states:     (W, N, seq_len, *input_shape)
          next_ram_states: (W, N, seq_len, ram_obs_dim)
          actions:     (W, N)
          rewards:     (W, N)
          dones:       (W, N)
          old_log_probs:(W, N)
          mems:        list of (W, N, mem_len, d_model)
        plus last_next_obs / last_next_ram for tail bootstrap.
        """
        T = self.t
        seq_len = self.sequence_length
        if T < seq_len + 1:
            return None
        N = self.num_envs
        W = T - seq_len + 1

        # Image sliding windows: (W, seq_len, N, *input_shape) -> (W, N, seq_len, ...)
        states_seq = np.stack([self.states[w : w + seq_len] for w in range(W)], axis=0)
        states_seq = states_seq.transpose(0, 2, 1, *range(3, states_seq.ndim))

        # Same for RAM.
        ram_seq = np.stack([self.ram_states[w : w + seq_len] for w in range(W)], axis=0)
        ram_seq = ram_seq.transpose(0, 2, 1, *range(3, ram_seq.ndim))

        next_states_seq = np.zeros_like(states_seq)
        next_ram_seq = np.zeros_like(ram_seq)
        if W > 1:
            interior_img = np.stack(
                [self.states[w + 1 : w + seq_len + 1] for w in range(W - 1)], axis=0
            )
            interior_img = interior_img.transpose(0, 2, 1, *range(3, interior_img.ndim))
            next_states_seq[: W - 1] = interior_img

            interior_ram = np.stack(
                [self.ram_states[w + 1 : w + seq_len + 1] for w in range(W - 1)], axis=0
            )
            interior_ram = interior_ram.transpose(0, 2, 1, *range(3, interior_ram.ndim))
            next_ram_seq[: W - 1] = interior_ram

        # Tail window: prefix from buffer, appended with last_next_obs / last_next_ram.
        tail_prefix_img = self.states[T - seq_len + 1 : T]
        tail_prefix_img = tail_prefix_img.transpose(
            1, 0, *range(2, tail_prefix_img.ndim)
        )
        tail_img = np.concatenate(
            [tail_prefix_img, self.last_next_obs[:, None]], axis=1
        )
        next_states_seq[-1] = tail_img

        tail_prefix_ram = self.ram_states[T - seq_len + 1 : T]
        tail_prefix_ram = tail_prefix_ram.transpose(1, 0, 2)
        tail_ram = np.concatenate(
            [tail_prefix_ram, self.last_next_ram[:, None]], axis=1
        )
        next_ram_seq[-1] = tail_ram

        end = T
        start = seq_len - 1
        actions = self.actions[start:end]
        rewards = self.rewards[start:end]
        dones = self.dones[start:end]
        old_log_probs = self.log_probs[start:end]
        mems_slice = self.mems[start:end]

        num_layers = mems_slice.shape[2]
        mems_per_layer = [
            torch.from_numpy(mems_slice[:, :, layer]).to(self.device)
            for layer in range(num_layers)
        ]

        return {
            "states": torch.from_numpy(states_seq).float().to(self.device),
            "ram_states": torch.from_numpy(ram_seq).float().to(self.device),
            "next_states": torch.from_numpy(next_states_seq).float().to(self.device),
            "next_ram_states": torch.from_numpy(next_ram_seq).float().to(self.device),
            "actions": torch.from_numpy(actions).long().to(self.device),
            "rewards": torch.from_numpy(rewards).float().to(self.device),
            "dones": torch.from_numpy(dones).to(self.device),
            "old_log_probs": torch.from_numpy(old_log_probs).float().to(self.device),
            "mems": mems_per_layer,
            "last_next_obs": torch.from_numpy(self.last_next_obs)
            .float()
            .to(self.device),
            "last_next_ram": torch.from_numpy(self.last_next_ram)
            .float()
            .to(self.device),
        }
