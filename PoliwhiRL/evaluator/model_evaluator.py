# -*- coding: utf-8 -*-
"""Inference-only model runner.

Loads a trained checkpoint and executes it stochastically (sampling from the
policy distribution, matching training-time action selection) through an
environment, recording PNG screenshots at every step. No PPO update
machinery is involved — it's a pure playthrough recorder.
"""

import os
import random

import numpy as np
import torch

from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.models.PPO import PPOModel


def run_inference(config):
    """Run a trained model in inference mode and record PNG outputs.

    Args:
        config: Merged configuration dict with at minimum:
            - load_checkpoint (str): path to checkpoint directory containing
              actor_critic.pth. Must also have info.pth for shape metadata,
              or the config must supply ram_obs_dim and action_size.
            - episode_length (int): max steps per run.
            - output_base_dir / record_path: where to write PNGs.
    """
    checkpoint = config["load_checkpoint"]
    if not checkpoint:
        raise ValueError(
            "Inference mode requires 'load_checkpoint' pointing to a "
            "checkpoint directory (containing actor_critic.pth)."
        )

    # Probe env for shapes.
    probe = Env(config)
    try:
        state_shape = probe.output_shape()
        ram_obs_dim = probe.ram_observation_shape()[0]
        num_actions = probe.action_space.n
    finally:
        probe.close()

    # Ensure the config has the keys PPOModel needs.
    config.setdefault("ram_obs_dim", ram_obs_dim)

    input_shape = state_shape  # (C, H, W) or (18, 20) tilemap

    # Build model and load weights only.
    model = PPOModel(input_shape, num_actions, config)
    _load_actor_critic_only(model, checkpoint)

    device = torch.device(config["device"])
    model.actor_critic.eval()

    episode_length = config.get("episode_length", 50)
    record_path = config.get("record_path", "Runs")
    num_runs_cfg = int(config.get("num_inference_runs", 1))

    print(
        f"Inference: checkpoint={checkpoint}, episode_length={episode_length}, "
        f"runs={num_runs_cfg}"
    )

    for run_idx in range(num_runs_cfg):
        _run_single(model, config, episode_length, record_path, run_idx, num_runs_cfg)


def _load_actor_critic_only(model, checkpoint):
    """Load only the actor_critic weights (skip optimizer/scheduler)."""
    weight_path = os.path.join(checkpoint, "actor_critic.pth")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"No actor_critic.pth found at {weight_path}. "
            "Ensure load_checkpoint points to a checkpoint directory."
        )
    model.actor_critic.load_state_dict(
        torch.load(weight_path, map_location=model.device, weights_only=True)
    )


def _run_single(model, config, episode_length, record_path, run_idx, total_runs):
    """Execute one full inference run and record PNGs."""
    env = Env(config)
    try:
        env.reset()

        # Replay if configured (walks Rewards forward to curriculum-aligned start).
        replay_paths = config.get("action_replay_paths") or []
        if replay_paths:
            from PoliwhiRL.environment.vec_env import _load_replay_pool

            _, replay_pool = _load_replay_pool(replay_paths)
            if replay_pool:
                traj = replay_pool[random.randrange(len(replay_pool))]
                # Use the full trajectory (no random prefix for inference — we want
                # a deterministic, reproducible starting point).
                if traj:
                    env.replay_actions(traj)

        obs = env.get_observation()
        state, ram = obs["image"], obs["ram"]

        sequence_length = config.get("sequence_length", 16)
        state_seq = [state] * sequence_length
        ram_seq = [ram] * sequence_length

        mems = model.init_mems(batch_size=1)

        # Enable recording for every step.
        folder = f"run_{run_idx}" if total_runs > 1 else "inference"
        env.enable_record(folder, use_episode_number=False)

        print(f"Run {run_idx}: starting ({episode_length} steps max)")

        for step in range(episode_length):
            state_arr = np.array(state_seq)
            ram_arr = np.array(ram_seq)

            action, mems = _sample_action(model, state_arr, ram_arr, mems)

            next_obs, reward, done, _ = env.step(action)
            state, ram = next_obs["image"], next_obs["ram"]

            # Fixed-length sliding window — matches training
            # (vec_ppo_agent.py:367-370). Appending without popping leaves
            # the input tensor growing every step, which the model never
            # saw at training time.
            state_seq.pop(0)
            state_seq.append(state)
            ram_seq.pop(0)
            ram_seq.append(ram)

            if done:
                print(f"Run {run_idx}: terminated at step {step + 1} (done)")
                break
        else:
            print(f"Run {run_idx}: completed all {episode_length} steps")

        # Print summary.
        _print_summary(env, run_idx)

    finally:
        env.close()


def _sample_action(model, state_arr, ram_arr, mems):
    """Forward pass with stochastic action sampling.

    Matches training-time selection (vec_ppo_agent.py:284-285): clamp to
    avoid degenerate zeros, then draw from the categorical distribution.
    Argmax collapses PPO's intentionally-stochastic policy onto a single
    action and frequently locks into ignored-button no-ops at inference.
    """
    state_tensor = torch.FloatTensor(state_arr).unsqueeze(0).to(model.device)
    ram_tensor = torch.FloatTensor(ram_arr).unsqueeze(0).to(model.device)

    with torch.no_grad():
        action_probs, _, new_mems = model.actor_critic(state_tensor, ram_tensor, mems)
        action_probs = torch.clamp(action_probs, 1e-10, 1.0)
        action = torch.multinomial(action_probs[0], 1).item()
    return action, new_mems


def _print_summary(env, run_idx):
    """Print a brief post-run summary."""
    rc = env.reward_calculator
    print(
        f"  Run {run_idx}: N_goals={rc.N_goals}/{rc.N_goals_target}, "
        f"reward={env._fitness:.2f}, steps={env.steps}"
    )
