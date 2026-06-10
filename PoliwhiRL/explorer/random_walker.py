# -*- coding: utf-8 -*-
"""Random-walk map-discovery tool.

Run a single ``PyBoyEnvironment`` for a fixed number of steps taking random
actions weighted toward movement. Whenever the env enters a new
``(map_bank, map_num)`` tuple, save a PNG frame plus a few short follow-up
frames so the user can inspect what the map looks like and back-infer
which event-flag bits fire there.

No model. No training. No reward signal. The only RAM tap is
``map_bank``/``map_num``/``X``/``Y``/``party_info`` for the manifest.

Designed for back-inferring which maps are reachable from a given save
state + optional action_replay seed, so we can pick meaningful
``flag``-type goals without guessing pokecrystal indices.

Output layout (under ``config["output_base_dir"]``):

    maps/bank{BB}_map{MM}/step_NNNNNN_idx{K}.png   # one PNG per capture
    maps_index.json                                # manifest of all maps

The manifest has the shape::

    {
      "07_03": {"bank": 7, "map_num": 3, "first_step": 142,
                "frames": [{"step": 142, "follow_up_idx": 0,
                            "x": 9, "y": 14, "party_hp": 18,
                            "battle_type": 0, "path": "maps/..."}]},
      ...
    }
"""

import json
import os

import numpy as np
from tqdm import tqdm

from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.environment.rewards import is_ram_state_valid
from PoliwhiRL.environment.vec_env import _load_actions_file


# Action ordering matches env.actions in gym_env.py: noop, A, B, left,
# right, up, down, start, select. Bias toward directional movement and A
# so the walker can push through dialogs (Mr. Pokémon, Elm, etc.) and
# explore. start/select kept at near-zero — the action mask would block
# them anyway in most stages.
_ACTION_PROBS = np.array(
    [
        0.05,   # noop
        0.15,   # A
        0.05,   # B
        0.175,  # left
        0.175,  # right
        0.175,  # up
        0.175,  # down
        0.025,  # start
        0.025,  # select
    ],
    dtype=np.float64,
)
_ACTION_PROBS /= _ACTION_PROBS.sum()

# Follow-up captures per new map: extra frames N steps after the first
# encounter, but only if the walker is still on that map at the offset.
# Spaced to give a few viewpoints if the agent lingers and zero if it
# wandered straight off.
_DEFAULT_FOLLOW_UP_OFFSETS = [5, 20, 50, 100]


def random_walk_map_discovery(config):
    """Walk the env with random actions, save a few frames per new map."""
    walk_steps = int(config.get("walk_steps", 50000))
    follow_up_offsets = list(
        config.get("random_walker_follow_up_offsets", _DEFAULT_FOLLOW_UP_OFFSETS)
    )
    seed = int(config.get("seed", 0))

    output_base = config["output_base_dir"]
    maps_dir = os.path.join(output_base, "maps")
    manifest_path = os.path.join(output_base, "maps_index.json")
    os.makedirs(maps_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    n_actions = len(_ACTION_PROBS)

    env = Env(config)
    try:
        env.reset()
        _apply_replay_seed(env, config)

        first_seen = {}
        scheduled = {}  # step -> list of (bank, map_num, follow_up_idx)
        manifest = {}

        print(
            f"random_walker: steps={walk_steps}, output={output_base}, "
            f"follow_ups_per_map={len(follow_up_offsets)}"
        )

        for step in tqdm(range(walk_steps), desc="random walk"):
            action = int(rng.choice(n_actions, p=_ACTION_PROBS))
            env._handle_action(action)
            env_vars = env.ram.get_variables()

            if not is_ram_state_valid(env_vars):
                continue

            bank = int(env_vars["map_bank"])
            map_num = int(env_vars["map_num"])
            key = (bank, map_num)

            if key not in first_seen:
                first_seen[key] = step
                _save_frame(env, maps_dir, output_base, bank, map_num,
                            step, 0, env_vars, manifest)
                for idx, offset in enumerate(follow_up_offsets, start=1):
                    scheduled.setdefault(step + offset, []).append(
                        (bank, map_num, idx)
                    )

            if step in scheduled:
                due = scheduled.pop(step)
                cur_bank = int(env_vars["map_bank"])
                cur_map = int(env_vars["map_num"])
                for (b, m, idx) in due:
                    # Only capture the follow-up if we're still on the
                    # same map — otherwise the frame would be of some
                    # other location and useless for the manifest.
                    if (cur_bank, cur_map) == (b, m):
                        _save_frame(env, maps_dir, output_base, b, m,
                                    step, idx, env_vars, manifest)

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(
            f"random_walker: discovered {len(first_seen)} unique maps. "
            f"manifest -> {manifest_path}"
        )
    finally:
        env.close()


def _apply_replay_seed(env, config):
    """Optionally walk the env forward through an action_replay seed so
    discovery starts post-replay (e.g. after the stage-3 best trajectory)
    rather than at the title screen."""
    replay_paths = config.get("action_replay_paths") or []
    if not replay_paths:
        return
    # Use the first replay file's first trajectory — single seed is fine
    # for discovery, the user can re-run with a different seed if needed.
    for path in replay_paths:
        trajectories = _load_actions_file(path)
        if trajectories:
            env.replay_actions(trajectories[0])
            print(
                f"random_walker: seeded from {path} "
                f"({len(trajectories[0])} steps)"
            )
            return


def _save_frame(env, maps_dir, output_base, bank, map_num,
                step, follow_up_idx, env_vars, manifest):
    folder = os.path.join(maps_dir, f"bank{bank:02d}_map{map_num:02d}")
    os.makedirs(folder, exist_ok=True)
    fname = f"step_{step:06d}_idx{follow_up_idx}.png"
    fpath = os.path.join(folder, fname)
    env.pyboy.screen.image.save(fpath)
    key = f"{bank:02d}_{map_num:02d}"
    rec = manifest.setdefault(
        key,
        {
            "bank": bank,
            "map_num": map_num,
            "first_step": step,
            "frames": [],
        },
    )
    rec["frames"].append(
        {
            "step": step,
            "follow_up_idx": follow_up_idx,
            "x": int(env_vars["X"]),
            "y": int(env_vars["Y"]),
            "party_hp": int(env_vars["party_info"][2]),
            "battle_type": int(env_vars["battle_type"]),
            "path": os.path.relpath(fpath, start=output_base),
        }
    )
