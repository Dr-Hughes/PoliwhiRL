# -*- coding: utf-8 -*-
"""Rollout buffer for vectorised PPO.

Stores a fixed-length (T, N, ...) rollout, then emits sliding-window
sequences of length `sequence_length` per env. The env axis is preserved
in the output so the agent can compute per-env GAE (advantages don't
cross env boundaries) before flattening for the PPO loss.

This buffer is intentionally minimal: it does not compute returns or
advantages. That logic lives in the agent so it can choose to bootstrap
with V(s_{T+1}) from the value network.
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
        self.reset()

    def reset(self):
        T, N = self.rollout_length, self.num_envs
        self.states = np.zeros((T, N) + self.input_shape, dtype=np.uint8)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.bool_)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.mems = None  # lazy: shape depends on model
        self.last_next_obs = None  # (N, *input_shape) after last step
        self.t = 0

    def __len__(self):
        return self.t

    def is_full(self):
        return self.t >= self.rollout_length

    def store_step(self, states, next_states, actions, rewards, dones, log_probs, mems):
        """Store one timestep's worth of transitions across all envs.

        states: (N, *input_shape) uint8 or castable.
        next_states: (N, *input_shape) — the obs returned by step (post-auto-reset on done).
        actions: (N,) int.
        rewards: (N,) float.
        dones: (N,) bool.
        log_probs: (N,) float.
        mems: list of length num_layers, each tensor of shape (N, mem_len, d_model).
        """
        if self.t >= self.rollout_length:
            raise RuntimeError("VecPPOMemory full; call reset() before storing.")
        idx = self.t
        self.states[idx] = np.asarray(states, dtype=np.uint8)
        self.actions[idx] = np.asarray(actions, dtype=np.int64)
        self.rewards[idx] = np.asarray(rewards, dtype=np.float32)
        self.dones[idx] = np.asarray(dones, dtype=np.bool_)
        self.log_probs[idx] = np.asarray(log_probs, dtype=np.float32)

        # Stack per-layer mems into a single ndarray: (N, num_layers, mem_len, d_model)
        stacked = np.stack(
            [m.detach().cpu().numpy() for m in mems], axis=1
        )
        if self.mems is None:
            self.mems = np.zeros(
                (self.rollout_length,) + stacked.shape, dtype=np.float32
            )
        self.mems[idx] = stacked

        self.last_next_obs = np.asarray(next_states, dtype=np.uint8)
        self.t += 1

    def get_data(self):
        """Emit sliding-window data with env axis preserved.

        Returns a dict with shapes:
          states:      (W, N, seq_len, *input_shape) float
          next_states: (W, N, seq_len, *input_shape) float
          actions:     (W, N) long
          rewards:     (W, N) float
          dones:       (W, N) bool
          old_log_probs:(W, N) float
          mems:        list of length num_layers, each (W, N, mem_len, d_model) float
        where W = T - seq_len + 1.

        Returns None if the buffer is too short.
        """
        T = self.t
        seq_len = self.sequence_length
        if T < seq_len + 1:
            return None
        N = self.num_envs
        W = T - seq_len + 1

        # Build sliding windows along the time axis for each env.
        # states_seq[w, n] = states[w : w+seq_len, n]
        states_seq = np.stack(
            [self.states[w : w + seq_len] for w in range(W)], axis=0
        )  # (W, seq_len, N, *input_shape)
        states_seq = states_seq.transpose(0, 2, 1, *range(3, states_seq.ndim))
        # -> (W, N, seq_len, *input_shape)

        # next_states_seq[w, n] = states[w+1 : w+seq_len, n] + [next_obs_after_w_seq]
        # For w in [0, W-2]: tail is just states[w+seq_len, n] (a real next obs we recorded as that timestep's state).
        # For w = W-1 (rollout tail): tail is last_next_obs[n].
        next_states_seq = np.zeros_like(states_seq)
        if W > 1:
            interior = np.stack(
                [self.states[w + 1 : w + seq_len + 1] for w in range(W - 1)], axis=0
            )  # (W-1, seq_len, N, *input_shape)
            interior = interior.transpose(0, 2, 1, *range(3, interior.ndim))
            next_states_seq[: W - 1] = interior

        tail_prefix = self.states[T - seq_len + 1 : T]  # (seq_len-1, N, *input_shape)
        tail_prefix = tail_prefix.transpose(1, 0, *range(2, tail_prefix.ndim))
        # (N, seq_len-1, *input_shape)
        tail = np.concatenate(
            [tail_prefix, self.last_next_obs[:, None]], axis=1
        )  # (N, seq_len, *input_shape)
        next_states_seq[-1] = tail

        # Action/reward/done/log_prob/mems at position w correspond to the LAST
        # frame of window w, i.e. index w + seq_len - 1 in the raw buffers.
        end = T  # exclusive
        start = seq_len - 1  # inclusive
        actions = self.actions[start:end]      # (W, N)
        rewards = self.rewards[start:end]      # (W, N)
        dones = self.dones[start:end]          # (W, N)
        old_log_probs = self.log_probs[start:end]  # (W, N)
        mems_slice = self.mems[start:end]      # (W, N, num_layers, mem_len, d_model)

        num_layers = mems_slice.shape[2]
        mems_per_layer = [
            torch.from_numpy(mems_slice[:, :, layer]).to(self.device)
            for layer in range(num_layers)
        ]  # each: (W, N, mem_len, d_model)

        return {
            "states": torch.from_numpy(states_seq).float().to(self.device),
            "next_states": torch.from_numpy(next_states_seq).float().to(self.device),
            "actions": torch.from_numpy(actions).long().to(self.device),
            "rewards": torch.from_numpy(rewards).float().to(self.device),
            "dones": torch.from_numpy(dones).to(self.device),
            "old_log_probs": torch.from_numpy(old_log_probs).float().to(self.device),
            "mems": mems_per_layer,
            "last_next_obs": torch.from_numpy(self.last_next_obs).float().to(self.device),
        }
