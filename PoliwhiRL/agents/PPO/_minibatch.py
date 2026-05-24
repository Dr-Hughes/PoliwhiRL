# -*- coding: utf-8 -*-
"""Shared PPO minibatch update loop.

Both PPOAgent and VecPPOAgent stage a *flat* data dict containing
precomputed `returns`, `advantages`, and `old_values` for the full
rollout. This module shuffles that dict into minibatches, runs the PPO
epochs, and applies KL-based early stopping at epoch granularity.

Precomputation is intentional: GAE/returns must use the actual rollout
ordering (with the V(s_{T+1}) bootstrap on the genuine tail). Shuffling
*before* GAE would scramble that ordering.
"""
import numpy as np
import torch


# Keys consumed by PPOModel._compute_ppo_losses. Anything else in the flat
# data dict (e.g. raw `rewards`, `dones`, `next_states`) is unused at update
# time and not worth slicing per minibatch.
_TENSOR_KEYS = (
    "states",
    "ram_states",
    "actions",
    "old_log_probs",
    "returns",
    "advantages",
    "old_values",
)
_LIST_OF_TENSOR_KEYS = ("mems",)


def _slice_minibatch(data, idx):
    out = {}
    for k in _TENSOR_KEYS:
        if k in data:
            out[k] = data[k][idx]
    for k in _LIST_OF_TENSOR_KEYS:
        if k in data:
            out[k] = [t[idx] for t in data[k]]
    return out


def run_ppo_epochs(model, data, step, epochs, minibatch_size, target_kl):
    """Run `epochs` PPO update passes over `data`, optionally minibatched.

    `step` is the training-progress counter (rollout idx for vec, episode
    idx for single-env). It is forwarded to PPOModel.update so the entropy
    schedule sees the same value the agent records in episode_data.

    Returns (total_summed_loss, epochs_run). `total_summed_loss` is the sum
    of *mean-per-epoch* losses, mirroring the existing single-pass behaviour
    for downstream metric reporting.
    """
    batch_size = data["states"].size(0)
    if minibatch_size is None or minibatch_size <= 0 or minibatch_size >= batch_size:
        mb_size = batch_size
        n_mb = 1
    else:
        mb_size = int(minibatch_size)
        n_mb = (batch_size + mb_size - 1) // mb_size

    device = data["states"].device
    total_loss = 0.0
    epochs_run = 0

    for _ in range(epochs):
        perm = torch.randperm(batch_size, device=device)
        epoch_loss = 0.0
        epoch_kls = []
        for i in range(n_mb):
            idx = perm[i * mb_size : (i + 1) * mb_size]
            mb = _slice_minibatch(data, idx)
            loss, approx_kl = model.update(mb, step)
            epoch_loss += loss
            epoch_kls.append(approx_kl)
        # Average loss over minibatches so the per-update scale matches the
        # pre-minibatching behaviour for metric continuity.
        total_loss += epoch_loss / n_mb
        epochs_run += 1
        if target_kl is not None and float(np.mean(epoch_kls)) > target_kl:
            break

    return total_loss, epochs_run
