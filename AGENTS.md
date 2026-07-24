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
  settings; `stages/open_world.json` is the canonical from-scratch run, and
  the other stage configs form the directed curriculum.
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
- **`tools/`** — primarily standalone, read-only post-hoc analysis of a
  checkpoint or recorded run; the ROM-map parser is the exception and supports
  the runtime collision field. `training_health_report.py` reads a checkpoint's
  `info.pth` (policy entropy, reward mix, archive growth,
  discovery log, honest/seeded checkpoint reach, first-hit replay state,
  frontier-pool size and probe ladder) to triage a run without hand-authoring
  a new script each time. `world_map.py` reconstructs an empirical per-tile
  walkability/warp/encounter map plus a chronological discovery timeline
  from recorded-episode PNG filenames — see each script's module docstring
  for exact usage and what each check means.

## Inspecting a run and detecting problems in the data

Two artefacts hold everything you need to triage a run, both under the run's
`output_base_dir`:

- **`Results/metrics/training_metrics.json`** — the per-episode record. A set
  of parallel arrays indexed by episode (`rewards`, `episode_steps`,
  `unique_maps`, `unique_cells`, `flag_fires`, `archive_size`, `seeded`,
  `reward_sources` — a per-episode dict of the per-source reward totals), plus
  optimiser-step arrays (`lrs`, `policy_entropies`, `entropy_coefs`,
  `approx_kls`, `clip_fractions`), a `summary`, and `per_state_summary`. This
  is the forensic record — use it when you need to see *when* and *how* a run
  turned.
- **`Checkpoints/info.pth`** — the checkpoint-aligned episode data plus the
  canonical visit archive, persisted Go-Explore frontier manifest and
  first-honest checkpoint-recording manifest. `tools/training_health_report.py`
  reads it for repeatable triage (optimisation health, honest/seeded reward
  mix, archive growth, discovery log, time-to-frontier, checkpoint reach and
  probe ladder). Drop to the metrics JSON when you need finer buckets or a cut
  the tool doesn't make.

**The one rule that matters most: split honest vs seeded before judging
anything.** `seeded[i]` is true for episodes started from a Go-Explore
snapshot; those start deep in already-solved territory, so their reward and
coverage are inflated and — critically — they are where farmable reward
channels get exercised hardest. The policy's *real* competence is the honest
(non-seeded) episodes only; `summary.honest_*` fields pre-compute these. The
aggregate `mean_reward` / `max_reward` conflate both and **can climb while the
honest policy collapses** — that is exactly how the 2026-07-20 run hid a
regression from 16 down to 5.6 honest unique_maps behind a rising headline
reward. Always bucket the honest indices over episode order and look at the
*trend*, not just the last-100 mean.

What a healthy run looks like: honest `unique_maps` / `unique_cells` /
`flag_fires` trending **up** across episode buckets; total reward dominated by
the exploration sources (`frontier`, `new_map`, `menu`, `event`); the
milestone/item sources (`checkpoint`, `key_item`, `pokedex`, `level`, `flag`)
small and roughly flat; `policy_entropy` easing down gently (sharpening, not
crashing to ~0); `approx_kl` and `clip_fraction` stable and small.

Failure signatures to check for explicitly:

- **Reward-hacking collapse.** Total or seeded reward rising while honest
  `unique_maps` / `unique_cells` fall. Confirm by looking at per-source maxima
  (`max(ep[src] for ep in reward_sources)` for each `src`): any one source
  whose per-episode value is orders of magnitude above the others is a farm.
  The `frontier`/`new_map` exploration sources should be the earners; a run
  where `key_item`/`pokedex`/`event` dominate has stopped playing the game.
- **Non-monotonic-counter farm.** `key_item` / `pokedex` / `level` are keyed
  on absolute-count thresholds; a large *episode* total in these means a
  counter is wobbling down and back up and re-paying the same threshold (a RAM
  read that isn't monotonic, an item tossed and re-grabbed, a menu reorder, or
  a deliberate loop). These channels are deduped per-episode in
  `Rewards._milestone_refire_scale`; a resurgence means that guard was bypassed
  by a new code path.
- **Dark / mis-wired metric.** A per-episode series pinned at exactly 0 for the
  whole run while its matching reward source is non-zero is a logging bug, not
  a training result. (`flag_fires` used to read the configured-goal count,
  which is 0 in freeform stages, and hid all story progress — it now counts
  derived-table flag fires. When a signal looks "always zero", verify what
  populates it in `vec_env.py`'s terminal-info block before concluding
  anything about the policy.)
- **Plateau vs collapse.** Flat `archive_size` growth in the tail with honest
  coverage *holding* is a plateau (out of ideas). Honest coverage *declining*
  is a collapse (actively unlearning) — a worse state, and usually a sign a
  reward channel is pulling the policy off-task.
- **Training instability.** `approx_kl` spiking or `clip_fraction` climbing
  over the run, `policy_entropy` collapsing toward 0 (premature determinism)
  or ballooning (diverging), or `lrs` decayed to near-zero late in a run that
  then locks in whatever behaviour it had — read these alongside the reward
  trend, since a farm often coincides with an entropy/KL disturbance.

When you finish a forensic pass, prefer to bake the finding into
`training_health_report.py` as a new repeatable check rather than leaving it as
a one-off script, so the next run's triage is one command. Follow
`.opencode/skills/training-run-analysis/SKILL.md` for the complete evidence
order, RAM correlation and report format.

## A note on documentation drift

This project iterates quickly on reward design and training mechanics.
Detailed prose describing *how* something works today is the first thing to
go stale — a previous version of this file described several mechanisms
(replay/seeding schemes, reward-scaling schemes, entropy controllers) that
had since been removed or replaced in code, and it kept describing them
anyway. When you learn something true and specific about how a mechanism
currently works, prefer a comment at the point of implementation over an
addition here.
