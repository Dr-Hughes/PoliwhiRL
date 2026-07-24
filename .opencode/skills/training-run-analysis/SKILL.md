---
name: training-run-analysis
description: Analyse PoliwhiRL training runs, checkpoints, probe ladders, reward health, exploration plateaux, and Pokémon Crystal RAM checkpoint evidence. Use whenever assessing Training Outputs, info.pth, training_metrics.json, checkpoint recordings, story flags, map IDs, Mystery Egg progress, honest-versus-seeded competence, or whether a running policy is learning well.
compatibility: Requires Python, PyTorch, and access to the PoliwhiRL repository and its Training Outputs.
---

# PoliwhiRL training-run analysis

Evaluate a run from persisted evidence rather than headline reward. The central
question is whether the policy starting from the canonical state is learning to
make durable game progress, not whether seeded episodes can collect reward in
already-discovered territory.

## Authoritative sources

Read these before drawing conclusions:

1. `AGENTS.md` for the current health criteria and known failure signatures.
2. `tools/training_health_report.py` for the repeatable first-pass analysis.
3. `RAM_MAPPING.md` for verified `(map_bank, map_num)` and event-flag meanings.
4. `PoliwhiRL/checkpoints.py` for recordable checkpoint flags and display titles.
5. The active stage config—normally `configs/stages/open_world.json`—for horizons,
   rewards, probes, seeding and model settings.

If prose and implementation disagree, use the implementation and flag the
documentation drift.

## Select a coherent snapshot

Use artefacts from one run and state their episode counts:

- `Checkpoints/info.pth` contains the checkpoint-aligned `episode_data`, visit
  archive, frontier pool and first-hit recording manifest.
- `Results/metrics/training_metrics.json` may be slightly newer because metrics
  are written more frequently than checkpoints.
- `Runs/checkpoints/<flag>-<title>/` contains the first honest checkpoint replay,
  its `actions.steps`, metadata header and per-action PNGs.

Small episode-count differences are normal. Never silently combine series from
different episode counts; identify which artefact supports each conclusion.

## First pass

From the repository root, run:

```bash
python tools/training_health_report.py \
  "Training Outputs/<run>/Checkpoints/info.pth"
```

Supply explicit episode lengths only when the report cannot infer them:

```bash
python tools/training_health_report.py <info.pth> \
  --episode-length 8192 \
  --probe-episode-length 8192 \
  --long-probe-episode-length 24576
```

Treat this report as a triage summary, then inspect the persisted arrays when a
result needs explanation.

## Analysis order

### 1. Split honest and seeded episodes

In `info.pth`, use `episode_data["episode_seeded"]`; in the metrics JSON, use
`seeded`. Honest episodes have `False` at the same episode index.

Report honest and seeded results separately. Seeded episodes demonstrate that a
policy can continue from a supplied frontier state; they do not demonstrate an
end-to-end route from the canonical start.

### 2. Measure trend, not only the tail mean

Split honest episode indices into four to eight chronological buckets. Compare:

- reward;
- unique maps and cells;
- derived flag fires;
- episode length;
- archive size.

Rising aggregate reward with falling honest coverage is a collapse. Flat archive
growth with stable honest coverage is a plateau.

### 3. Check PPO stability

Inspect policy entropy, approximate KL, clip fraction and learning rate together.
Healthy optimisation has gradual entropy change and small, stable KL/clip values.
Stable PPO statistics do not prove game progress; they only rule out one failure
class.

### 4. Inspect reward composition

Compare first and last honest quarters using `episode_reward_sources` in
`info.pth`, or `reward_sources` in the metrics JSON. Keep `checkpoint`, `event`,
`frontier`, `new_map`, battle and counter-based channels separate.

Look for:

- a single channel dominating by orders of magnitude;
- checkpoint/event signals shrinking until frontier reward drowns them;
- milestone totals much larger in seeded episodes;
- counter channels repeatedly paying for non-monotonic state.

Do not recommend increasing every reward merely because a checkpoint is rare.
First establish whether the successful trajectory already earns more total reward
than comparable episodes and whether missing frontier seeding or horizon is the
real bottleneck.

### 5. Establish genuine frontier progress

Use all three signals:

- `episode_archive_size` trend;
- `discovery_log` recency and provenance;
- `visit_archive` map/cell counts.

Repeated visits to a map prove reachability, not reliable competence. A new flag
seen only in seeded episodes proves downstream reachability but not a complete
honest route.

### 6. Read probes by checkpoint identity

Use `probe_goal_rates` and `long_probe_goal_rates`, not only aggregate probe
success. Compare the deepest successful short- and long-horizon checkpoint:

- long succeeds, short fails → likely horizon bottleneck;
- both fail → likely policy/reward/exploration bottleneck;
- both succeed repeatedly → capability is probably consolidated.

Account for probe sample size. A two-episode 100% result is weak evidence; report
the numerator and denominator.

## Correlate run evidence with RAM checkpoints

Use `references/ram-correlation.md` for the exact evidence chain. In summary:

1. Resolve locations by the full `(map_bank, map_num)` pair from `RAM_MAPPING.md`.
2. Resolve story progress by event-flag number, not a guessed name or raw byte.
3. Count per-episode fires from `episode_flag_fire_steps`, whose entries are
   `[flag_number, step]` pairs.
4. Join those episode indices to `episode_seeded` before claiming true-start
   competence.
5. Cross-check first honest reaches against `checkpoint_recordings` and its
   replay status.

For the current early-game frontier:

| Evidence | Meaning |
|---|---|
| map `(26, 10)` | Entered Mr. Pokémon's house |
| flag `30` | Received the Mystery Egg |
| flag `31` | Returned the Mystery Egg to Elm; implies surviving the rival sequence |

Use cautious wording:

- **“Ever reached”** may include seeded episodes.
- **“Reached honestly”** means an unseeded episode fired the flag.
- **“Reliably learned”** requires repeated honest probe success.

## Validate first-hit recordings

For each `checkpoint_recordings` entry, check:

- `seeded` is false;
- `status` is `recorded`;
- `checkpoint_observed` is true;
- `actions_path` exists;
- the labelled output directory contains PNG frames;
- the recorded episode and fire step agree with `episode_flag_fire_steps`.

A live first hit with `checkpoint_not_reproduced` is evidence that the policy hit
the checkpoint, but not that the saved action prefix replayed deterministically.
Flag that as a debugging problem rather than discarding the live evidence.

## Report format

Present a compact table with these columns:

| Area | Honest evidence | Seeded evidence | Assessment | Recommended action |
|---|---|---|---|---|

Include separate rows for optimisation, honest coverage trend, reward mix,
archive growth, deepest checkpoint, probe reliability, frontier-pool health and
recording integrity. End with direct answers to:

1. Is optimisation stable?
2. Is honest competence improving?
3. Is the run discovering genuinely new content?
4. What is the deepest honest checkpoint?
5. What is the highest-priority next change?

Do not edit code or training artefacts during an assessment unless the user
explicitly asks for implementation.
