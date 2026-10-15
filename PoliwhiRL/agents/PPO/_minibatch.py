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
import copy


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
    "old_base_probs",
    "old_behaviour_probs",
    "dones",
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
    diag_acc = {}
    diag_n = 0
    max_accepted_kl = 0.0
    rejected = False
    base_kl = 0.0
    behaviour_kl = 0.0
    replay_diag = {}
    replay_tolerance = float(model.config.get("ppo_replay_tolerance", 1e-6))
    # Opt-in per stage: legacy/smoke configs did not calibrate an exact-KL
    # ceiling and must retain their previous update behaviour.
    hard_kl = float(model.config.get("ppo_hard_kl", float("inf")))

    if hasattr(model, "replay_policy_diagnostics"):
        replay_diag = model.replay_policy_diagnostics(data)
        policy_replay_values = (
            replay_diag.get("base_replay_kl", float("nan")),
            replay_diag.get("behaviour_replay_kl", float("nan")),
        )
        # Categorical KL measures the complete policy mismatch and is the
        # invariant relevant to PPO. A raw maximum selected-action log delta
        # is unstable for rare actions: negligible probability differences can
        # become large after log without materially changing the distribution.
        # Keep those values as diagnostics, but do not discard an otherwise
        # valid rollout on them.
        if not all(np.isfinite(v) for v in policy_replay_values) or max(
            policy_replay_values
        ) > replay_tolerance:
            print(
                "[PPO] Skipping update: collection/replay mismatch "
                f"{replay_diag} exceeds ppo_replay_tolerance={replay_tolerance}."
            )
            model.last_epoch_diag = {
                **replay_diag,
                "replay_rejected": True,
                "kl_rejected": False,
                "effective_lr": float(model.effective_lr()),
                "accepted_epochs": 0,
                "approx_kl": 0.0,
            }
            return total_loss, epochs_run

    pre_base = pre_behaviour = None
    if hasattr(model, "policy_distributions"):
        with torch.no_grad():
            pre_base, pre_behaviour = model.policy_distributions(data)

    for _ in range(epochs):
        model_state = copy.deepcopy(model.actor_critic.state_dict())
        optimizer_state = copy.deepcopy(model.optimizer.state_dict())
        perm = torch.randperm(batch_size, device=device)
        epoch_loss = 0.0
        epoch_diag_acc = {}
        epoch_diag_n = 0
        for i in range(n_mb):
            idx = perm[i * mb_size : (i + 1) * mb_size]
            mb = _slice_minibatch(data, idx)
            loss, approx_kl = model.update(mb, step)
            epoch_loss += loss
            # Aggregate the per-minibatch optimisation diagnostics the
            # model stashes (clip fraction, loss components, ...). getattr
            # keeps stub models in unit tests working.
            mb_diag = getattr(model, "last_update_diag", None)
            if mb_diag:
                for k, v in mb_diag.items():
                    epoch_diag_acc[k] = epoch_diag_acc.get(k, 0.0) + float(v)
                epoch_diag_n += 1
        if pre_base is None:
            base_kl, behaviour_kl = model.exact_policy_kls(data)
        else:
            with torch.no_grad():
                post_base, post_behaviour = model.policy_distributions(data)
                base_kl = float(model.exact_categorical_kl(pre_base, post_base).item())
                behaviour_kl = float(
                    model.exact_categorical_kl(pre_behaviour, post_behaviour).item()
                )
        finite = np.isfinite(base_kl) and np.isfinite(behaviour_kl)
        if not finite or base_kl > hard_kl or behaviour_kl > hard_kl:
            model.actor_critic.load_state_dict(model_state)
            model.optimizer.load_state_dict(optimizer_state)
            model.halve_lr_multiplier()
            rejected = True
            break
        # Average loss over minibatches so the per-update scale matches the
        # pre-minibatching behaviour for metric continuity.
        total_loss += epoch_loss / n_mb
        epochs_run += 1
        max_accepted_kl = max(max_accepted_kl, base_kl, behaviour_kl)
        for k, v in epoch_diag_acc.items():
            diag_acc[k] = diag_acc.get(k, 0.0) + v
        diag_n += epoch_diag_n
        if target_kl is not None and max(base_kl, behaviour_kl) > target_kl:
            break

    # Rollout-level means, stashed on the model for the agent's metrics
    # (return signature unchanged for existing call sites / tests).
    model.last_epoch_diag = (
        {k: v / diag_n for k, v in diag_acc.items()} if diag_n else {}
    )
    if diag_n:
        model.last_epoch_diag["epochs_run"] = float(epochs_run)
    model.last_epoch_diag.update(
        {
            **replay_diag,
            "base_post_kl": float(base_kl),
            "behaviour_post_kl": float(behaviour_kl),
            "kl_rejected": bool(rejected),
            "replay_rejected": False,
            "effective_lr": float(model.effective_lr()),
            "accepted_epochs": int(epochs_run),
            "approx_kl": float(max_accepted_kl),
        }
    )

    return total_loss, epochs_run
