# -*- coding: utf-8 -*-
"""Pre-flight gate for a training run (docs/next-from-scratch-plan.md, Phase 0).

    uv run python tools/preflight.py [--config configs/stages/open_world.json]

Checks, without starting training:
  - the config resolves and the env builds with settle stepping armed (the
    ROM is the build settle.py's hook addresses belong to);
  - the observation size the env produces matches RAM_OBS_DIM;
  - every SIL demo file matches the current RAM vector AND the current
    observation resize filter, and every reverse-curriculum seed manifest's
    states exist;
  - the model this config builds: parameter count and encoder identity;
  - whether the run would resume rather than start fresh, either through an
    explicit load_checkpoint or through main.py's auto-resume when it is empty;
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

# The array tools/llm_agent/demos.py stamps each demo with, naming the resize
# filter its images were produced under.
DEMO_STAMP_KEY = "obs_resize_filter"


def env_resize_filter():
    """The observation resize filter the env currently applies, as a string.

    gym_env.OBS_RESIZE_FILTER_NAME is the single source of truth: the env
    applies it and tools/llm_agent/demos.py stamps it, so reading it here
    compares the demo against the pipeline rather than against another copy
    of the same literal. The fallback is the same value so a broken import
    degrades to the right answer instead of passing every demo.
    """
    try:
        from PoliwhiRL.environment.gym_env import OBS_RESIZE_FILTER_NAME

        return str(OBS_RESIZE_FILTER_NAME)
    except Exception:
        return "box"


def demo_failures(d, path, ram_obs_dim, expected_filter):
    """Failure strings for one loaded demo .npz.

    ``d`` is an ``np.load()`` handle (anything with ``.files`` and
    ``__getitem__``), so this stays pure and unit-testable. Both checks are
    hard failures because SIL feeds a demo's stored images straight to the
    network — vec_ppo_agent._load_sil_demos builds ``states`` from
    ``d["images"]``, and _refresh_demo_mems / PPOModel.sil_update pass them
    through actor_critic — so a demo recorded under a different RAM vector
    or a different resize filter trains the policy on observations it can
    never see at rollout time. A MISSING stamp means the file predates the
    stamp and its images came from LANCZOS, which leaked 0.75 tile on every
    side; that is a failure, not a pass.
    """
    fails = []
    if int(d["ram_dim"]) != ram_obs_dim:
        fails.append(
            f"demo {path}: ram_dim {int(d['ram_dim'])} != {ram_obs_dim} "
            "(rebuild with tools/llm_agent/demos.py)"
        )
    if DEMO_STAMP_KEY not in d.files:
        fails.append(
            f"demo {path}: no {DEMO_STAMP_KEY} stamp, so its images predate the stamp and "
            f"were resized with LANCZOS rather than the env's current {expected_filter!r} "
            "(rebuild with tools/llm_agent/demos.py)"
        )
    else:
        raw = d[DEMO_STAMP_KEY]
        # .item() first: a 0-d numpy scalar is not always an instance of the
        # matching Python type (np.bytes_ is not a bytes subclass here), and
        # str() of one would print its repr rather than its value.
        value = raw.item() if hasattr(raw, "item") else raw
        if isinstance(value, bytes):
            value = value.decode()
        stamp = str(value).strip()
        if stamp != expected_filter:
            fails.append(
                f"demo {path}: {DEMO_STAMP_KEY} {stamp!r} != the env's {expected_filter!r}, "
                "so its images are from a different observation pipeline "
                "(rebuild with tools/llm_agent/demos.py)"
            )
    return fails


def resume_note(load_checkpoint, auto_checkpoint_dir):
    """The resume/fresh line, from the two config paths that actually decide it.

    An EMPTY load_checkpoint does not mean "starts fresh": main.py's
    _resolve_auto_resume then points the run at ``checkpoint`` whenever that
    directory already holds an info.pth + actor_critic.pth pair. The old code
    did ``Path(config["load_checkpoint"]) / "info.pth"``, which with the empty
    default and os.chdir(REPO) above tested a repo-root ``info.pth`` — a
    spurious warning if one ever sat there, and a missed auto-resume always.
    """
    ckpt = str(load_checkpoint or "").strip()
    if ckpt:
        info = Path(ckpt) / "info.pth"
        if info.exists():
            return f"WARNING: {info} exists — the run will RESUME, not start fresh"
        return f"load_checkpoint {ckpt!r} holds no info.pth — the run starts fresh"
    auto = str(auto_checkpoint_dir or "").strip()
    if not auto:
        return f"load_checkpoint empty and no checkpoint dir configured — starts fresh"
    if (Path(auto) / "info.pth").exists() and (
        Path(auto) / "actor_critic.pth"
    ).exists():
        return (
            f"WARNING: load_checkpoint is empty but {auto} holds info.pth + "
            f"actor_critic.pth, which main.py auto-resumes from — RESUME, not fresh"
        )
    return f"load_checkpoint empty and {auto} is not resumable — the run starts fresh"


def encoder_identity(strides):
    """Name the image encoder from its conv stride schedule.

    The same rule training_health_report.py's Capacity section applies, and
    for the same reason: conv_strides kept its KEY while changing MEANING
    between the legacy [2, 2] CNN and the tile-grid [4, 1, 1, 1, 2, 1] one, so
    a parameter count alone cannot say which model a run built.
    """
    if not strides:
        return "no Conv2d layers (encoder identity unknown)"
    if strides[0] == 4:
        return "tile-grid encoder"
    if list(strides) == [2, 2]:
        return "legacy stride-2 encoder"
    return f"unrecognised encoder (strides {list(strides)})"


def model_note(config, input_shape, action_size, ram_obs_dim):
    """'model: 808,181 params, tile-grid encoder' for the net this config builds.

    Built on CPU from shapes the env already reported, so the note costs one
    small network and no second emulator. ram_obs_dim is injected the way
    PoliwhiRL/PPO.py's setup_and_train_PPO does before constructing the agent:
    the merged config on its own does not carry it and PPOModel reads it.
    """
    import torch

    from PoliwhiRL.models.PPO import PPOModel

    build_cfg = dict(config, device="cpu", ram_obs_dim=int(ram_obs_dim))
    net = PPOModel(input_shape, action_size, build_cfg).actor_critic
    strides = [
        int(m.stride[0]) for m in net.modules() if isinstance(m, torch.nn.Conv2d)
    ]
    params = sum(int(q.numel()) for q in net.parameters())
    return (
        f"model: {params:,} params, {encoder_identity(strides)} "
        f"(conv strides {strides}, input {tuple(input_shape)}, {action_size} actions)"
    )


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
    shapes = None
    try:
        obs = env.reset()
        if env._stepper is None:
            failures.append("settle stepping is NOT armed (step_mode or ROM checksum)")
        else:
            notes.append(f"settle stepping armed, cap {env._stepper.cap} frames")
        if obs["ram"].shape[-1] != RAM_OBS_DIM:
            failures.append(
                f"env RAM vector {obs['ram'].shape} != RAM_OBS_DIM {RAM_OBS_DIM}"
            )
        for _ in range(50):
            env.step(0)
        notes.append(
            f"env steps fine; RAM_OBS_DIM {RAM_OBS_DIM}, image {obs['image'].shape}"
        )
        shapes = (tuple(env.output_shape()), int(env.action_space.n), RAM_OBS_DIM)
    finally:
        env.close()

    # Capacity, so the operator sees the model they are about to commit hours
    # to before committing them. Informational: a note, never a failure.
    if shapes:
        try:
            notes.append(model_note(config, shapes[0], shapes[1], shapes[2]))
        except Exception as exc:
            notes.append(f"model: not built ({exc.__class__.__name__}: {exc})")

    demos = sorted(
        {f for pat in config.get("sil_demo_files") or [] for f in glob.glob(pat)}
    )
    if config.get("sil_demo_files") and not demos:
        failures.append(f"sil_demo_files matches no files: {config['sil_demo_files']}")
    expected_filter = env_resize_filter()
    steps, n_failed_before = 0, len(failures)
    for f in demos:
        d = np.load(f)
        failures += demo_failures(d, f, RAM_OBS_DIM, expected_filter)
        steps += len(d["actions"])
    if demos:
        # "all stamped" is a claim about these files, so only make it when the
        # loop above accepted every one of them.
        clean = len(failures) == n_failed_before
        stamp = f", all stamped {DEMO_STAMP_KEY}={expected_filter!r}" if clean else ""
        notes.append(f"{len(demos)} SIL demo files, {steps} steps{stamp}")

    manifests = sorted(
        {f for pat in config.get("rc_seed_manifests") or [] for f in glob.glob(pat)}
    )
    for m in manifests:
        caps = json.loads(Path(m).read_text())["captures"]
        missing = [c["path"] for c in caps if not os.path.exists(c["path"])]
        if missing:
            failures.append(f"{m}: {len(missing)} ring states missing")
        notes.append(
            f"{m}: {len(caps)} ring states, flags {sorted({c['flag'] for c in caps})}"
        )

    notes.append(resume_note(config.get("load_checkpoint"), config.get("checkpoint")))

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
