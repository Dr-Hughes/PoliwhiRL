#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the current policy for N honest (true-start) episodes in parallel and
log a per-step trajectory, for post-hoc "where does it get lost" analysis.

NOT used by training. Training records only one episode in every
``record_frequency``; this produces a batch of comparable honest episodes on
demand from a checkpoint, with the extra per-step fields the stuck-route
analysis needs (story flags, trainer/wild battle, lead HP and damaging PP).

Go-Explore / reverse-curriculum capture is disabled so the run writes no
snapshots into the training output tree.

Usage (from repo root, training stopped):

    uv run python tools/policy_rollouts.py --config configs/stages/open_world.json \
        --episodes 8 --out "Training Outputs/open_world/policy_rollouts"
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

EVENT_FLAGS = 0xDA72
PARTY = 0xDCDF
TRACKED_FLAGS = (30, 31, 65, 66, 1812, 1813)
# Damaging moves the early-game party can know (Gen 2 move ids).
DAMAGING = {1, 10, 16, 33, 40, 44, 52, 55, 64, 75, 84, 98, 99}


def _wram(mem, addr):
    # D000-DFFF is the switchable WRAM bank; game state lives in bank 1.
    return mem[1, addr] if 0xD000 <= addr < 0xE000 else mem[addr]


def _flag(mem, n):
    return (_wram(mem, EVENT_FLAGS + n // 8) >> (n % 8)) & 1


def _lead(mem):
    moves = [_wram(mem, PARTY + 2 + j) for j in range(4)]
    pp = [_wram(mem, PARTY + 23 + j) & 0x3F for j in range(4)]
    hp = (_wram(mem, PARTY + 34) << 8) | _wram(mem, PARTY + 35)
    mhp = (_wram(mem, PARTY + 36) << 8) | _wram(mem, PARTY + 37)
    dmg_pp = sum(p for mv, p in zip(moves, pp) if mv in DAMAGING)
    return hp, mhp, dmg_pp


def _worker(args):
    idx, config, checkpoint, out_dir, max_steps, seed = args
    import random

    import numpy as np
    import torch

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from PoliwhiRL.environment import PyBoyEnvironment as Env
    from PoliwhiRL.evaluator.model_evaluator import _load_actor_critic_only
    from PoliwhiRL.models.PPO import PPOModel

    config = dict(config, device="cpu")
    env = Env(config)
    env.reset()
    config.setdefault("ram_obs_dim", env.ram_observation_shape()[0])
    model = PPOModel(env.output_shape(), env.action_space.n, config)
    _load_actor_critic_only(model, checkpoint)
    model.actor_critic.eval()

    obs = env.get_observation()
    seq_len = config.get("sequence_length", 16)
    state_seq, ram_seq = [obs["image"]] * seq_len, [obs["ram"]] * seq_len
    mems = model.init_mems(batch_size=1)
    mem = env.pyboy.memory

    path = Path(out_dir) / f"ep_{idx:02d}.csv"
    t0 = time.time()
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "bank", "map", "x", "y", "battle", "btn", "reward",
                    *[f"f{n}" for n in TRACKED_FLAGS], "lead_hp", "lead_maxhp",
                    "lead_dmg_pp", "party"])
        step = 0
        for step in range(1, max_steps + 1):
            with torch.no_grad():
                action, _lp, mems = model.get_action(
                    np.array(state_seq), np.array(ram_seq), mems
                )
            obs, reward, done, _ = env.step(action)
            state_seq = state_seq[1:] + [obs["image"]]
            ram_seq = ram_seq[1:] + [obs["ram"]]
            hp, mhp, dmg = _lead(mem)
            w.writerow([step, _wram(mem, 0xDCB5), _wram(mem, 0xDCB6),
                        _wram(mem, 0xDCB8), _wram(mem, 0xDCB7),
                        _wram(mem, 0xD22D), env.button, round(float(reward), 3),
                        *[_flag(mem, n) for n in TRACKED_FLAGS], hp, mhp, dmg,
                        _wram(mem, 0xDCD7)])
            if done:
                break
    env.close()
    return idx, step, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser(description="Honest policy rollouts with trajectory logs.")
    ap.add_argument("--config", default="configs/stages/open_world.json")
    ap.add_argument("--checkpoint", default=None,
                    help="default: <output_base_dir>/Checkpoints")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="default: config episode_length")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.chdir(_REPO)
    import main as entry

    config = entry.merge_configs(entry.load_default_config(),
                                 entry.load_user_config(args.config))
    config.update(goexplore_enabled=False, reverse_curriculum_enabled=False,
                  record=False)
    checkpoint = args.checkpoint or os.path.join(config["output_base_dir"], "Checkpoints")
    max_steps = args.max_steps or int(config["episode_length"])
    Path(args.out).mkdir(parents=True, exist_ok=True)

    jobs = [(i, config, checkpoint, args.out, max_steps, args.seed + i)
            for i in range(args.episodes)]
    workers = args.workers or min(args.episodes, os.cpu_count() or 4)
    with mp.get_context("spawn").Pool(workers) as pool:
        for idx, steps, secs in pool.imap_unordered(_worker, jobs):
            print(f"episode {idx}: {steps} steps in {secs}s", flush=True)


if __name__ == "__main__":
    main()
