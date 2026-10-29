# -*- coding: utf-8 -*-
"""Pre-flight gate for a training run (docs/next-from-scratch-plan.md, Phase 0).

    uv run python tools/preflight.py [--config configs/stages/open_world.json]

Checks, without starting training:
  - the config resolves and the env builds with settle stepping armed (the
    ROM is the build settle.py's hook addresses belong to);
  - the observation size the env produces matches RAM_OBS_DIM;
  - every SIL demo file matches the current RAM vector, and every
    reverse-curriculum seed manifest's states exist;
  - whether the output dir already holds a checkpoint (a fresh run should not);
  - the emulator scenario suite (tools/llm_agent/scenarios.py) passes.
Exits 1 on any failure. The unit tests (pytest) and an LLM ladder run to flag
31 are the rest of the gate.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np  # noqa: E402

from main import load_default_config, load_user_config, merge_configs  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/stages/open_world.json")
    args = p.parse_args()
    failures, notes = [], []

    config = merge_configs(load_default_config(), load_user_config(args.config))
    from PoliwhiRL.environment.gym_env import RAM_OBS_DIM, PyBoyEnvironment

    env = PyBoyEnvironment(dict(config, record=False))
    try:
        obs = env.reset()
        if env._stepper is None:
            failures.append("settle stepping is NOT armed (step_mode or ROM checksum)")
        else:
            notes.append(f"settle stepping armed, cap {env._stepper.cap} frames")
        if obs["ram"].shape[-1] != RAM_OBS_DIM:
            failures.append(f"env RAM vector {obs['ram'].shape} != RAM_OBS_DIM {RAM_OBS_DIM}")
        for _ in range(50):
            env.step(0)
        notes.append(f"env steps fine; RAM_OBS_DIM {RAM_OBS_DIM}, image {obs['image'].shape}")
    finally:
        env.close()

    demos = sorted({f for pat in config.get("sil_demo_files") or [] for f in glob.glob(pat)})
    if config.get("sil_demo_files") and not demos:
        failures.append(f"sil_demo_files matches no files: {config['sil_demo_files']}")
    steps = 0
    for f in demos:
        d = np.load(f)
        if int(d["ram_dim"]) != RAM_OBS_DIM:
            failures.append(
                f"demo {f}: ram_dim {int(d['ram_dim'])} != {RAM_OBS_DIM} "
                "(rebuild with tools/llm_agent/demos.py)"
            )
        steps += len(d["actions"])
    if demos:
        notes.append(f"{len(demos)} SIL demo files, {steps} steps")

    manifests = sorted({f for pat in config.get("rc_seed_manifests") or [] for f in glob.glob(pat)})
    for m in manifests:
        caps = json.loads(Path(m).read_text())["captures"]
        missing = [c["path"] for c in caps if not os.path.exists(c["path"])]
        if missing:
            failures.append(f"{m}: {len(missing)} ring states missing")
        notes.append(f"{m}: {len(caps)} ring states, flags {sorted({c['flag'] for c in caps})}")

    ckpt = Path(config["load_checkpoint"]) / "info.pth"
    if ckpt.exists():
        notes.append(f"WARNING: {ckpt} exists — the run will RESUME, not start fresh")

    from tools.llm_agent.scenarios import run_all

    for name, fails in run_all().items():
        if fails:
            failures += [f"scenario {name}: {f}" for f in fails]
        else:
            notes.append(f"scenario {name}: ok")

    for n in notes:
        print("  ", n)
    if failures:
        print("PRE-FLIGHT FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PRE-FLIGHT OK")


if __name__ == "__main__":
    main()
