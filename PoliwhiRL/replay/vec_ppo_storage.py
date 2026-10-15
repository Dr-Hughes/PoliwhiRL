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
        self.action_size = int(config.get("action_size", 9))
        self.reset()

    def reset(self):
        T, N = self.rollout_length, self.num_envs
        self.states = np.zeros((T, N) + self.input_shape, dtype=np.uint8)
        self.ram_states = np.zeros((T, N, self.ram_obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.bool_)
        # Parallel to dones: True only where a done was a budget truncation
        # (vs a natural goal terminal). Drives the GAE bootstrap.
        self.truncated = np.zeros((T, N), dtype=np.bool_)
        # V(actual terminal continuation) for budget-truncated dones. Zero for
        # every other transition; ``truncated`` controls whether it is used.
        self.terminal_bootstrap_values = np.zeros((T, N), dtype=np.float32)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.base_probs = np.zeros((T, N, self.action_size), dtype=np.float32)
        self.behaviour_probs = np.zeros((T, N, self.action_size), dtype=np.float32)
        self.mems = None  # lazy: shape depends on model
        self.outgoing_mems = None  # lazy: memory after each stored observation
        self.next_states = np.zeros((T, N) + self.input_shape, dtype=np.uint8)
        self.next_ram_states = np.zeros((T, N, self.ram_obs_dim), dtype=np.float32)
        self.last_next_obs = None  # compatibility/debug view of final next state
        self.last_next_ram = None
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
        base_probs,
        behaviour_probs,
        mems,
        outgoing_mems,
        truncated=None,
        terminal_bootstrap_values=None,
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
        if truncated is not None:
            self.truncated[idx] = np.asarray(truncated, dtype=np.bool_)
        if terminal_bootstrap_values is not None:
            self.terminal_bootstrap_values[idx] = np.asarray(
                terminal_bootstrap_values, dtype=np.float32
            )
        self.log_probs[idx] = np.asarray(log_probs, dtype=np.float32)
        self.base_probs[idx] = np.asarray(base_probs, dtype=np.float32)
        self.behaviour_probs[idx] = np.asarray(behaviour_probs, dtype=np.float32)

        stacked = np.stack([m.detach().cpu().numpy() for m in mems], axis=1)
        if self.mems is None:
            self.mems = np.zeros(
                (self.rollout_length,) + stacked.shape, dtype=np.float32
            )
        self.mems[idx] = stacked

        outgoing_stacked = np.stack(
            [m.detach().cpu().numpy() for m in outgoing_mems], axis=1
        )
        if self.outgoing_mems is None:
            self.outgoing_mems = np.zeros(
                (self.rollout_length,) + outgoing_stacked.shape, dtype=np.float32
            )
        self.outgoing_mems[idx] = outgoing_stacked

        self.next_states[idx] = np.asarray(next_states, dtype=np.uint8)
        self.next_ram_states[idx] = np.asarray(next_ram_states, dtype=np.float32)
        self.last_next_obs = self.next_states[idx].copy()
        self.last_next_ram = self.next_ram_states[idx].copy()
        self.t += 1

    def get_data(self):
        """Emit NON-OVERLAPPING contiguous segments for BPTT training.

        The T stored timesteps are partitioned per env into ``S = T //
        seq_len`` segments of length ``L = seq_len`` (any remainder tail is
        dropped). Each segment is trained as a unit: the whole L-frame
        sequence is fed to the model with the (detached) memory that
        preceded the segment, so gradient flows across all L positions
        (real BPTT). Segments are contiguous in time within an env, so the
        agent can still run per-env GAE over the full ``T`` timeline before
        slicing into segments.

        Returns dict with shapes (S = num segments, N = num envs, L = seq_len):
          states:      (S, N, L, *input_shape)
          ram_states:  (S, N, L, ram_obs_dim)
          actions/rewards/dones/truncated/terminal_bootstrap_values/old_log_probs:
                       (S, N, L)
          mems:        list of (S, N, mem_len, d_model)  — segment-initial memory
          tail_mems:   list of (N, mem_len, d_model)     — for the V(s_T) bootstrap
          last_next_obs / last_next_ram: (N, ...)        — the post-rollout state
          S, N, L: ints
        """
        T = self.t
        L = self.sequence_length
        if L < 1 or T < L:
            return None
        S = T // L
        if S < 1:
            return None
        Tn = S * L
        N = self.num_envs

        def to_seg(arr):
            # (T, N, *tail) -> (S, N, L, *tail)
            a = arr[:Tn].reshape(S, L, N, *arr.shape[2:])
            return np.ascontiguousarray(np.moveaxis(a, 2, 1))

        states_seg = to_seg(self.states)  # (S,N,L,C,H,W)
        ram_seg = to_seg(self.ram_states)  # (S,N,L,D)
        actions_seg = to_seg(self.actions)  # (S,N,L)
        rewards_seg = to_seg(self.rewards)
        dones_seg = to_seg(self.dones)
        truncated_seg = to_seg(self.truncated)
        terminal_bootstrap_seg = to_seg(self.terminal_bootstrap_values)
        logp_seg = to_seg(self.log_probs)
        base_probs_seg = to_seg(self.base_probs)
        behaviour_probs_seg = to_seg(self.behaviour_probs)

        # Segment-initial memory = the memory that went INTO each segment's
        # first step, mems[s*L]; shape (S, N, layers, mem_len, d_model).
        seg_starts = np.arange(S) * L
        mems_init = self.mems[seg_starts]
        num_layers = mems_init.shape[2]
        mems_per_layer = [
            torch.from_numpy(np.ascontiguousarray(mems_init[:, :, layer])).to(
                self.device
            )
            for layer in range(num_layers)
        ]
        # Tail memory is what came OUT of processing the final included
        # observation, exactly the memory sequential inference uses for s_T.
        tail_mems = self.outgoing_mems[Tn - 1]  # (N, layers, mem_len, d_model)
        tail_mems_per_layer = [
            torch.from_numpy(np.ascontiguousarray(tail_mems[:, layer])).to(self.device)
            for layer in range(num_layers)
        ]

        return {
            "states": torch.from_numpy(states_seg).float().to(self.device),
            "ram_states": torch.from_numpy(ram_seg).float().to(self.device),
            "actions": torch.from_numpy(actions_seg).long().to(self.device),
            "rewards": torch.from_numpy(rewards_seg).float().to(self.device),
            "dones": torch.from_numpy(dones_seg).to(self.device),
            "truncated": torch.from_numpy(truncated_seg).to(self.device),
            "terminal_bootstrap_values": torch.from_numpy(
                terminal_bootstrap_seg
            ).float().to(self.device),
            "old_log_probs": torch.from_numpy(logp_seg).float().to(self.device),
            "old_base_probs": torch.from_numpy(base_probs_seg).float().to(self.device),
            "old_behaviour_probs": torch.from_numpy(behaviour_probs_seg)
            .float()
            .to(self.device),
            "mems": mems_per_layer,
            "tail_mems": tail_mems_per_layer,
            "last_next_obs": torch.from_numpy(self.next_states[Tn - 1])
            .float()
            .to(self.device),
            "last_next_ram": torch.from_numpy(self.next_ram_states[Tn - 1])
            .float()
            .to(self.device),
            "S": S,
            "N": N,
            "L": L,
        }
