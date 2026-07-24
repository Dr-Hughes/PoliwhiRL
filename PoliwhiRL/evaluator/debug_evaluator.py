# -*- coding: utf-8 -*-
"""Debug evaluator — surfaces extended RAM state per step.

Three modes selected via ``config["debug_mode"]``:

- ``"menu_probe"`` (default): runs a fixed scripted sequence intended to
  expose menu / text-box RAM addresses. The sequence is

      noop × 3   →   START   →   noop × 3   →   B

  No model is loaded; no action_replay is consumed. Episode begins from
  the configured save state. Each step writes a PNG + JSON sidecar with
  every RAM probe so the user can diff bytes across the sequence and see
  which addresses flip when the menu opens / closes.

- ``"scripted"``: as above but the action sequence comes from
  ``config["debug_action_sequence"]`` — a list of action indices into
  ``env.actions`` (``["", "a", "b", "left", "right", "up", "down",
  "start", "select"]``).

- ``"model"``: runs the trained policy as in normal inference, but uses
  the debug saver so each frame gets the extended RAM dump. Useful for
  inspecting an actual rollout with full RAM visibility.

The debug saver writes through ``env.save_debug_step_img_data`` rather
than ``env.save_step_img_data``; the difference is documented in
``gym_env.py``.

Implementation deliberately does NOT use ``env.step`` for scripted modes:
``env.step`` invokes the reward calculator and may set ``done`` mid-probe
(e.g. on a goal touch), which would truncate the sequence. We bypass via
``env._handle_action`` and write the debug frame directly.
"""

import os
import random

import numpy as np
import torch

from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.models.PPO import PPOModel


# Action index map matches env.actions order in gym_env.py.
_ACTION_NOOP = 0
_ACTION_A = 1
_ACTION_B = 2
_ACTION_START = 7

# Default scripted sequence — what the user asked for: nothing×3, start,
# nothing×3, B. Exposes menu open + menu close transitions.
_MENU_PROBE_SEQUENCE = [
    _ACTION_NOOP,
    _ACTION_NOOP,
    _ACTION_NOOP,
    _ACTION_START,
    _ACTION_NOOP,
    _ACTION_NOOP,
    _ACTION_NOOP,
    _ACTION_B,
]


def run_debug_inference(config):
    """Entry point. Dispatches on ``config["debug_mode"]``."""
    mode = config.get("debug_mode", "menu_probe")
    record_path = config.get("record_path", "Runs")
    print(f"Debug evaluator: mode={mode}, record_path={record_path}")

    if mode == "menu_probe":
        sequence = list(_MENU_PROBE_SEQUENCE)
        _run_scripted(config, sequence, folder="menu_probe")
    elif mode == "scripted":
        sequence = config.get("debug_action_sequence")
        if not sequence:
            raise ValueError(
                "debug_mode='scripted' requires 'debug_action_sequence' "
                "(list of int action indices) in the config."
            )
        sequence = [int(a) for a in sequence]
        _run_scripted(config, sequence, folder="scripted")
    elif mode == "model":
        _run_model_debug(config)
    else:
        raise ValueError(
            f"Unknown debug_mode '{mode}'. "
            "Expected one of: 'menu_probe', 'scripted', 'model'."
        )


# --------------------------------------------------------------------- #
# Scripted modes (menu_probe + scripted)                                 #
# --------------------------------------------------------------------- #


def _run_scripted(config, sequence, folder):
    """Run a fixed sequence of action indices on a fresh env.

    Calls ``env._handle_action`` directly so ``env.step``'s reward /
    done logic doesn't truncate the probe.
    """
    env = Env(config)
    try:
        env.reset()
        # Action replay would warm the env into a non-deterministic position
        # — exactly what we don't want for a probe. Skip it explicitly.
        env.enable_record(folder, use_episode_number=False)
        # enable_record() flips self.record on; the scripted loop will use
        # save_debug_step_img_data() rather than the normal step recorder.

        action_names = env.actions
        print(
            f"Scripted probe: {len(sequence)} actions = "
            + " → ".join(f"{i}:{action_names[a]!r}" for i, a in enumerate(sequence))
        )

        # Record the initial frame BEFORE any action so the user has a
        # baseline "as loaded" snapshot to diff against.
        env.save_debug_step_img_data(
            env.record_folder, outdir=config.get("record_path", "Runs")
        )

        for idx, action in enumerate(sequence):
            env._handle_action(int(action))
            # Recompute fitness so the reward token in the filename reflects
            # what the reward system *would* assign — useful for spotting
            # accidental reward firings during the probe.
            env._calculate_fitness()
            env.save_debug_step_img_data(
                env.record_folder, outdir=config.get("record_path", "Runs")
            )
            print(
                f"  step {env.steps}: action={action_names[action]!r} "
                f"reward={env._fitness:.3f}"
            )

        summary_path = env.finalize_debug_run(
            folder, outdir=config.get("record_path", "Runs")
        )
        print(
            f"Scripted probe complete: {env.steps} steps written to "
            f"{config.get('record_path', 'Runs')}/{folder}/"
            f"\n  run_summary: {summary_path}"
        )
    finally:
        env.close()


# --------------------------------------------------------------------- #
# Model mode — like run_inference but writing debug frames                #
# --------------------------------------------------------------------- #


def _run_model_debug(config):
    checkpoint = config.get("load_checkpoint")
    if not checkpoint:
        raise ValueError(
            "debug_mode='model' requires 'load_checkpoint' pointing to a "
            "checkpoint directory (containing actor_critic.pth)."
        )

    probe = Env(config)
    try:
        state_shape = probe.output_shape()
        ram_obs_dim = probe.ram_observation_shape()[0]
        num_actions = probe.action_space.n
    finally:
        probe.close()
    config.setdefault("ram_obs_dim", ram_obs_dim)

    model = PPOModel(state_shape, num_actions, config)
    _load_actor_critic_only(model, checkpoint)
    model.actor_critic.eval()

    env = Env(config)
    try:
        env.reset()
        replay_paths = config.get("action_replay_paths") or []
        if replay_paths:
            from PoliwhiRL.environment.vec_env import _load_replay_pool

            _, replay_pool = _load_replay_pool(replay_paths)
            if replay_pool:
                traj = replay_pool[random.randrange(len(replay_pool))]
                if traj:
                    env.replay_actions(traj)

        obs = env.get_observation()
        state, ram = obs["image"], obs["ram"]
        sequence_length = config.get("sequence_length", 8)
        state_seq = [state] * sequence_length
        ram_seq = [ram] * sequence_length
        mems = model.init_mems(batch_size=1)

        folder = "model_debug"
        env.enable_record(folder, use_episode_number=False)
        env.save_debug_step_img_data(folder, outdir=config.get("record_path", "Runs"))

        episode_length = config.get("episode_length", 256)
        print(f"Model debug: {episode_length} steps max")
        for step in range(episode_length):
            state_arr = np.array(state_seq)
            ram_arr = np.array(ram_seq)
            action, mems = _sample_action(model, state_arr, ram_arr, mems)

            env._handle_action(int(action))
            env._calculate_fitness()
            env.save_debug_step_img_data(
                folder, outdir=config.get("record_path", "Runs")
            )

            obs = env.get_observation()
            state, ram = obs["image"], obs["ram"]
            state_seq.pop(0)
            state_seq.append(state)
            ram_seq.pop(0)
            ram_seq.append(ram)

            if env.done:
                print(f"  terminated at step {step + 1}")
                break

        rc = env.reward_calculator
        summary_path = env.finalize_debug_run(
            folder, outdir=config.get("record_path", "Runs")
        )
        print(
            f"Model debug done: steps={env.steps}, "
            f"N_goals={rc.N_goals}, flag_fires={rc.flag_goals_completed}, "
            f"reward={env._fitness:.2f}"
            f"\n  run_summary: {summary_path}"
        )
    finally:
        env.close()


def _load_actor_critic_only(model, checkpoint):
    weight_path = os.path.join(checkpoint, "actor_critic.pth")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"No actor_critic.pth found at {weight_path}. "
            "Ensure load_checkpoint points to a checkpoint directory."
        )
    model.actor_critic.load_state_dict(
        torch.load(weight_path, map_location=model.device, weights_only=True)
    )


def _sample_action(model, state_arr, ram_arr, mems):
    state_tensor = torch.FloatTensor(state_arr).unsqueeze(0).to(model.device)
    ram_tensor = torch.FloatTensor(ram_arr).unsqueeze(0).to(model.device)
    with torch.no_grad():
        action_mask = model._action_mask_for(ram_tensor)
        action_probs, _, new_mems = model.actor_critic(
            state_tensor,
            ram_tensor,
            mems,
            action_mask=action_mask,
        )
        action_probs = torch.clamp(action_probs, 1e-10, 1.0)
        action = torch.multinomial(action_probs[0], 1).item()
    return action, new_mems
