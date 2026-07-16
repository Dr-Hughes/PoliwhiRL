# AGENTS.md

## Goal

PoliwhiRL trains a reinforcement-learning agent to play Pokémon Crystal from
raw screen pixels and RAM state, via the PyBoy emulator. The agent progresses
through a training curriculum — from leaving the starting house through to
open-ended exploration of the wider game world — with the long-term aim of
learning to navigate and make progress using general-purpose reward signals
(story milestones, discovery, exploration) rather than a hand-authored target
for every stage.

This file is a map of the codebase, kept deliberately high-level. It does not
restate mechanism internals, magic numbers, or config defaults — those live in
the code and configs themselves and drift out of sync with prose the moment
they change. If you need the exact behaviour of something, read the source.
If this file and the code ever disagree, the code is right.

## Components

- **`main.py`** — entry point. Loads and merges JSON configs, then dispatches
  to training, inference, or one of the diagnostic/data-collection modes.
- **`configs/`** — JSON configs with inheritance (`extends`). Defaults live
  under `default_configs/`; `curriculum_base.json` holds shared training
  settings; `stages/*.json` are the individual curriculum stages.
- **`PoliwhiRL/environment/`** — the PyBoy-backed Gym-style environment:
  stepping the emulator, building the observation (screen image + a RAM
  feature vector), reward calculation, goal definitions, action masking, and
  the persistent exploration/visit-tracking archive.
- **`PoliwhiRL/models/`** — the policy/value network (CNN + RAM encoder
  feeding a Transformer-XL trunk) and the PPO loss/update implementation.
- **`PoliwhiRL/agents/PPO/`** — the training loop itself: rollout collection
  across vectorised (multiprocess) environments, PPO updates, checkpointing,
  and run metrics.
- **`PoliwhiRL/replay/`** — rollout storage buffers.
- **`PoliwhiRL/evaluator/`**, **`PoliwhiRL/explorer/`**,
  **`PoliwhiRL/reward_evaluation/`** — inference/eval, manual/random data
  collection, and standalone reward-system debugging tools.
- **`RAM_MAPPING.md`** — the RAM address/event-flag reference, verified
  against live emulator reads. Treat this one as ground truth to the extent
  it's marked verified; it documents the game's memory, not this codebase's
  design, so it doesn't drift the way prose about the code does.
- **`tests/`** — unit tests (fast, no emulator) and emulator-backed
  integration tests.
- **`tools/`** — standalone, read-only post-hoc analysis of a checkpoint or
  recorded run; not imported by training. `training_health_report.py` reads
  a checkpoint's `info.pth` (entropy servo, reward mix, archive growth,
  discovery log, probe ladder) to triage a run without hand-authoring a new
  script each time. `world_map.py` reconstructs an empirical per-tile
  walkability/warp/encounter map plus a chronological discovery timeline
  from recorded-episode PNG filenames — see each script's module docstring
  for exact usage and what each check means.

## A note on documentation drift

This project iterates quickly on reward design and training mechanics.
Detailed prose describing *how* something works today is the first thing to
go stale — a previous version of this file described several mechanisms
(replay/seeding schemes, reward-scaling schemes, entropy controllers) that
had since been removed or replaced in code, and it kept describing them
anyway. When you learn something true and specific about how a mechanism
currently works, prefer a comment at the point of implementation over an
addition here.
