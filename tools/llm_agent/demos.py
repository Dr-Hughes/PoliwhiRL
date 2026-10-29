# -*- coding: utf-8 -*-
"""Training demonstrations from LLM runs, rebuilt on the current env.

The demos a run writes while it plays (``demos/*.npz``) freeze the
observation and reward code of that day; any later change to the RAM vector
makes them unloadable. A run's button presses replay exactly (replay.py), so
this rebuilds the demos by replaying them through the current env and keeps
the successful goal attempts:

    uv run python -m tools.llm_agent.demos "<run>" [...] --out "Training Outputs/llm_demos"
    uv run python -m tools.llm_agent.demos --chain "<run>:410" "<run>_resumed" --rc

Each successful attempt becomes ``<out>/<run>_<NN>_<goal>_a<k>.npz`` with
``images`` (T, C, H, W) uint8, ``ram`` (T, D) float32, ``actions`` (T,),
``rewards`` (T,) — the same (observation, action, reward) tuples a training
worker would have produced — plus ``ram_dim`` and ``source="llm_demo"``.
Point ``sil_demo_files`` at them (vec_ppo_agent: demos seed self-imitation).
A run that no longer replays exactly is cut at its first diverging step:
only attempts that finished before it are kept.

With ``--rc`` the replay also runs the env's reverse-curriculum ring (every
``reverse_curriculum_interval`` steps of the training config) with every
story checkpoint as a target, so each checkpoint the LLM reached gets the
same ring of pre-fire states an honest training episode would have left:
``<out>/rc/*.state`` plus ``<out>/rc/<run>.json``, the capture records. Point
``rc_seed_manifests`` at the JSON; a fresh run copies the states into its
own snapshot dir and adds one trajectory per checkpoint to the pool, used
only once honest play shows that checkpoint is weak.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from PoliwhiRL.checkpoints import DERIVED_FLAG_TABLE, is_recordable_checkpoint  # noqa: E402
from PoliwhiRL.environment.gym_env import RAM_OBS_DIM  # noqa: E402
from tools.llm_agent.game import Game, build_config  # noqa: E402
from tools.llm_agent.replay import load_run, replay  # noqa: E402


def _rc_game(run, rc_dir, interval):
    """A replay env whose reverse-curriculum ring (every ``interval`` steps)
    is armed for every story checkpoint (Game's config turns it off)."""
    meta = run["meta"]
    g = Game(meta["config"], state_path=meta.get("state"))
    env = g.env
    env._rc_enabled = True
    env._rc_interval = max(1, int(interval))
    env._goexplore_snapshot_dir = str(rc_dir)
    env.visit_archive.set_rc_targets(
        f for f, _ in DERIVED_FLAG_TABLE if is_recordable_checkpoint(f)
    )
    return g


def _recordable_bits(flags) -> set:
    return {
        f
        for f, _ in DERIVED_FLAG_TABLE
        if is_recordable_checkpoint(f) and (int(flags[f // 8]) >> (f % 8)) & 1
    }


def build(chain, out_dir, rc: bool = False, rc_interval: int = 64) -> list[Path]:
    """``chain``: [(run_dir, until_step or None), ...] replayed as ONE
    episode — a run resumed from another's autosave continues it (the
    resumed run restarts the reward calculator, so its own recording is a
    seeded episode; chained from the original start it is honest)."""
    runs = [(load_run(d), until) for d, until in chain]
    name = runs[-1][0]["dir"].name if len(runs) == 1 else "+".join(r["dir"].name for r, _ in runs)
    ok, order = set(), []
    for run, _ in runs:
        for a in run["attempts"]:
            if a["success"]:
                ok.add((a["goal"], a["attempt"]))
            if a["goal"] not in order:
                order.append(a["goal"])
    parts: dict = {}
    start_idx: dict = {}
    fires: list = []  # (global step index, flag) for story checkpoints
    rc_dir = Path(out_dir) / "rc"
    captures = []
    if rc:
        rc_dir.mkdir(parents=True, exist_ok=True)
    game = _rc_game(runs[0][0], rc_dir, rc_interval) if rc else None
    t = [0]
    seen = [None]

    def on_step(g, row, sample):
        t[0] += 1
        if rc:
            captures.extend(
                dict(c, harness_step=t[0])
                for c in g.env._frontier_captures
                if c.get("kind") == "rc"
            )
            g.env._frontier_captures = []
        bits = _recordable_bits(g.vars()["story_flags"])
        if seen[0] is not None:
            fires.extend((t[0], f) for f in sorted(bits - seen[0]))
        seen[0] = bits
        key = (row["goal"], row["attempt"])
        if key in ok:
            start_idx.setdefault(key, t[0])
            buf = parts.setdefault(
                key, {"images": [], "ram": [], "actions": [], "rewards": [], "last": 0}
            )
            buf["images"].append(sample["image"])
            buf["ram"].append(sample["ram"])
            buf["actions"].append(sample["action"])
            buf["rewards"].append(sample["reward"])
            buf["last"] = t[0]

    g = game
    cut = None
    for k, (run, until) in enumerate(runs):
        if k:
            # The resume went through env.reset(), whose baseline step is a
            # no-op press; repeat it (unrecorded) so the emulator matches.
            g.env.step(0)
        offset = t[0]
        g, diverged = replay(run["dir"], until=until, game=g, on_step=on_step)
        if diverged:
            cut = offset + (diverged[0]["step"] - run["steps"][0]["step"] + 1)
            print(
                f"{run['dir']}: replay diverges at its step {diverged[0]['step']} "
                f"({diverged[0]}); later steps dropped"
            )
            break
    g.close()
    if cut is not None:
        for k in [k for k, b in parts.items() if b["last"] >= cut]:
            del parts[k]
        captures[:] = [c for c in captures if c["harness_step"] < cut]
    if rc:
        manifest = rc_dir / f"{name}.json"
        manifest.write_text(
            json.dumps({"runs": [str(r["dir"]) for r, _ in runs], "captures": captures}, indent=1)
        )
        print(
            f"{manifest}: {len(captures)} ring states for flags "
            f"{sorted({c['flag'] for c in captures})}"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for (goal, attempt), buf in parts.items():
        # The first story checkpoint at or after this leg starts: when honest
        # play is competent at it, the agent retires the leg's demo.
        target = next((f for i, f in fires if i >= start_idx[(goal, attempt)]), -1)
        path = out / f"{name}_{order.index(goal) + 1:02d}_{goal}_a{attempt}.npz"
        np.savez_compressed(
            path,
            images=np.stack(buf["images"]).astype(np.uint8),
            ram=np.stack(buf["ram"]).astype(np.float32),
            actions=np.asarray(buf["actions"], dtype=np.int64),
            rewards=np.asarray(buf["rewards"], dtype=np.float32),
            ram_dim=RAM_OBS_DIM,
            target_flag=int(target),
            source="llm_demo",
        )
        written.append(path)
    return written


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "runs",
        nargs="+",
        help="run dirs; RUN:STEP stops a run at that step. With --chain the runs "
        "are replayed as one episode, else each separately",
    )
    p.add_argument(
        "--chain",
        action="store_true",
        help="replay the runs back to back as one episode (a run resumed from "
        "another's autosave, chained from the original start, is honest)",
    )
    p.add_argument("--out", default="Training Outputs/llm_demos")
    p.add_argument("--rc", action="store_true", help="also write reverse-curriculum ring states")
    p.add_argument(
        "--rc-interval",
        type=int,
        default=64,
        help="steps between ring states (dense: an LLM path is short)",
    )
    args = p.parse_args()

    def parse(spec):
        run, _, until = (
            spec.rpartition(":") if spec.rsplit(":", 1)[-1].isdigit() else (spec, "", "")
        )
        return run, int(until) if until else None

    specs = [parse(r) for r in args.runs]
    chains = [specs] if args.chain else [[s] for s in specs]
    for chain in chains:
        for path in build(chain, args.out, rc=args.rc, rc_interval=args.rc_interval):
            d = np.load(path)
            print(
                f"{path}: {len(d['actions'])} steps, reward {float(d['rewards'].sum()):.1f}, "
                f"leads to flag {int(d['target_flag'])}"
            )


if __name__ == "__main__":
    main()
