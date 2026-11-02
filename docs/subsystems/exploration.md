# Go-Explore & exploration mechanisms

What this page is for: how this repo actually does exploration — who owns the
canonical visit archive and how workers stay in step with it, how the cell
quantisation and `cell_key` define "novel ground", what triggers a frontier capture
and what caps the capture rate, how the frontier pool is split into spatial and
progress halves and weighted when seeding, when a worker is seeded versus returned to
the honest start, and the probe ladder used to measure unassisted reach. Read this
before touching `goexplore_*` or `probe_*` config, and before believing any coverage
number that has not been split honest/seeded — the reward side of novelty lives in
[rewards.md](rewards.md).

**Sources** — §7 of the earlier reconnaissance notes (scratch notes in the untracked
 working-tree directory `.doc-scratch/`, never committed, so not citable from a fresh
 clone; every claim re-checked against the files
 listed below), backed by
`PoliwhiRL/agents/PPO/vec_ppo_agent.py` (2287 L),
`PoliwhiRL/environment/visit_archive.py` (413 L), `PoliwhiRL/environment/gym_env.py`
and `configs/stages/open_world.json` (`ow:`), on branch
`flag31-breakthrough-and-tuning`. `vec_ppo_agent.py:*` and `ow:*` line numbers are
transcribed from scratch, not re-read in this pass (see Known gaps).

## Ownership: one canonical archive, N replicas

- The **agent owns the canonical** run-wide visit archive. Workers hold read-only
  replicas, and the env's archive is refreshed by broadcast: the terminal-info block
  hands up `visited_cells` / `visited_maps` as the cells/maps *genuinely* visited
  this episode, explicitly described as "the env's archive is a read-only replica
  refreshed by `set_visit_archive` broadcasts"
  (`PoliwhiRL/environment/vec_env.py:188-193`).
- The broadcast back down to workers is a **full-table replace**, which makes it
  self-healing: a worker that missed a merge is corrected wholesale on the next
  broadcast rather than needing a delta protocol
  (`PoliwhiRL/agents/PPO/vec_ppo_agent.py:1106-1115`).
- The same provenance-carrying merge API feeds the reward depletion counters —
  `checkpoint_fire_count(flag_num, seeded=...)`
  (`PoliwhiRL/environment/visit_archive.py:232`),
  `first_trainer_battle_fire_count(seeded=...)` (`:239`),
  `map_count(..., seeded=)` (`:371`), `count(..., seeded=)` (`:393`), milestone
  depletion (`:222`) — which is why the archive, not the env, is the ledger of
  record. Where the merge API has no metadata argument, provenance travels in-band
  (`rewards.py:1289`, `:1358`).
- `archive_size` is reported per episode from the env's replica
  (`vec_env.py:165`), so it is a *replica* reading; treat the agent's archive as
  authoritative when the two disagree mid-step.

## Quantisation: `CELL_SIZE = 2` and `cell_key`

| item | value / behaviour | citation |
|---|---|---|
| `CELL_SIZE` | 2 tiles — coarser than per-tile, finer than per-block | `visit_archive.py:41`, rationale docstring `:29` |
| key builder | the cell-key helper that turns `(map_bank, map_num, x, y)` into a quantised key | `visit_archive.py:44-48` |
| public method | `VisitArchive.cell_key(map_bank, map_num, x, y)` | `visit_archive.py:406` |
| consumer | the frontier bonus keys novelty on exactly this cell, so reward and archive cannot disagree about what "new ground" means | `PoliwhiRL/environment/rewards.py:1342` |

Because quantisation is shared, a 2-tile-wide corridor is one cell: the frontier
reward pays per 2×2 block of world, not per tile. The reward-side caps and the
`revisit_novelty_scale` discount are in
[rewards.md](rewards.md#anti-farming-guards).

## Capture triggers and caps

Captures happen inside the env's `step()` — frontier capture is attempted every step
when Go-Explore is enabled (`gym_env.py:1040-1041`,
`self._maybe_capture_frontier(self._last_env_vars)`), a pre-checkpoint state is
buffered before the action is applied (`gym_env.py:1037`), and all per-episode
capture bookkeeping is cleared at reset: `_frontier_captures`,
`_captured_maps_this_episode`, `_captured_cells_this_episode`,
`_captured_flags_this_episode`, `_captured_party_milestones_this_episode`,
`_pending_flag_capture`, `_frontier_capture_count_this_episode`,
`_progress_capture_count_this_episode`, `_precheckpoint_states`
(`gym_env.py:1162-1170`). Captures reach the agent via
`terminal_info["frontier_captures"]` → `env.get_frontier_captures()`
(`vec_env.py:213-216`).

Capture granularity, triggers and caps (open-world values):

| config key | value | what it gates | citation |
|---|---|---|---|
| `goexplore_capture_granularity` | code default `"map"`; open-world `"cell"` | whether snapshots are keyed by map or by quantised cell | `vec_ppo_agent.py:138`; `ow:57` |
| `goexplore_max_captures_per_episode` | 3 | total captures per episode | `ow:59` |
| `goexplore_capture_cell_count_max` | 3 | cell-type captures per episode | `ow:58` |
| `goexplore_max_progress_captures_per_episode` | 2 | progress-type captures per episode | `ow:60` |
| `goexplore_flag_capture` / `goexplore_flag_count_max` | true / 3 | snapshot on a story-flag fire | `ow:73`, `ow:74` |
| `goexplore_precheckpoint_capture` / `_interval` / `_buffer_size` | true / 16 / 4 | buffering of states ahead of a known checkpoint | `ow:75`, `ow:76`, `ow:77`; buffering call site `gym_env.py:1037` |
| `goexplore_catch_capture` / `goexplore_party_min` | true / 2 | snapshot once the party reaches the minimum size | `ow:61`, `ow:62` |

**The cap enforcement site is not located** — see Known gaps. The counters exist in
the env (`gym_env.py:1168-1169`) and the values are quotable from config, but the
comparison that reads them was not opened, so the mechanism is undocumented.

## Pool split: spatial vs progress, and the seeding weight

Ingested captures are partitioned by `_is_progress_capture`, each pool sorted by
`_live_count` and truncated to its own cap, then concatenated
(`vec_ppo_agent.py:1022-1033`).

| pool | cap key → value | citation |
|---|---|---|
| spatial | `goexplore_spatial_pool_size`, default = `goexplore_pool_size`; `ow:66 = 96` | `vec_ppo_agent.py:150-154`; `ow:66` |
| progress | `goexplore_progress_pool_size`; `ow:67 = 32` | `vec_ppo_agent.py:144-149`; `ow:67` |
| total | `goexplore_pool_size`; `ow:65 = 128` | `ow:65` |

Curation rules that keep the pool honest:

- A durable pre-event `checkpoint` snapshot must **not** be degraded back to a
  `flag` snapshot carrying the same key (`vec_ppo_agent.py:1009-1018`).
- Newest snapshot wins per location (`vec_ppo_agent.py:1019-1021`).
- Files are never deleted mid-run — a worker may be about to load one; the snapshot
  directory is cleared only for a fresh run (`vec_ppo_agent.py:999-1002`).

`_pick_frontier_snapshot` (`vec_ppo_agent.py:1045-1069`) then decides what to seed
from:

1. choose the **progress** pool with probability
   `goexplore_progress_seed_fraction` (code default 0.25
   `vec_ppo_agent.py:156-158`; `ow:68 = 0.25`) if it is non-empty, else the
   spatial pool;
2. sample with weight `1/(1 + live_count)` (`vec_ppo_agent.py:1061-1068`), so
   seeding concentrates on the *rarest* — the true frontier — locations while still
   occasionally revisiting better-learned ground. `live_count` is the number of
   snapshots currently sharing a location key.

## `_goexplore_cycle`: who gets seeded, and who never does

`_goexplore_cycle` (`vec_ppo_agent.py:1071-1104`):

| worker class | behaviour | citation |
|---|---|---|
| probe workers (`env_idx < goexplore_probe_workers`) | left untouched, i.e. **never seeded** | `vec_ppo_agent.py:1072-1073`, `:1079` |
| all other workers | seed with probability `goexplore_seed_fraction` — **code default 0.6** (`vec_ppo_agent.py:142`), **current open-world value 0.4** (`ow:64`) | `vec_ppo_agent.py:1071-1104`, `:142`; `ow:64` |
| otherwise | a previously-seeded worker is returned to `_true_start_path` with `episode_seeded=False`, i.e. back to an honest run | `vec_ppo_agent.py:1095-1103` |
| on seed | `vec_env.set_env_state(env_idx, snap["path"], episode_seeded=True)` | `vec_ppo_agent.py:1089-1091` |
| seed failure | caught and logged; the worker stays honest rather than restarting into a half-applied state | `vec_ppo_agent.py:1093-1094`, `:1102-1103` |
| provenance plumbing | `env_is_seeded_pending` queue, promoted on auto-reset (`:1076-1077`, `:1104`); arrays allocated `:553-554`; per-episode list exposed as metrics `episode_seeded` (`:333`, rationale `:327-332`) | `vec_ppo_agent.py:1076-1104`, `:553-554`, `:333` |

**Why probe workers never seed:** they preserve the honest from-start signal and keep
`goexplore_probe_fraction` of workers training from scratch
(`vec_ppo_agent.py:134-136`). Env 0 is always a probe worker, so recording and probe
behaviour stay from-scratch (`:546-548`); the count is
`max(1, round(num_envs * goexplore_probe_fraction))` (`:550-552`), fraction default
0.25 (`:139-141`), `ow:63 = 0.25` — 2 of 8 workers at `ow:9`.

## The probe ladder and the long-horizon probe

Both probes run single-threaded, so the frequency must stay modest
(`vec_ppo_agent.py:226-227`); scheduling at `:646-653`, pair/pass helpers
`_run_probe_pair` / `_run_probe_pass` at `:1191` and `:1235`.

| key | code default | current open-world value | citation |
|---|---|---|---|
| `probe_enabled` | False (`vec_ppo_agent.py:228`) | **true** | `ow:79` |
| `probe_frequency` | 20 (`:235`) | **400** | `ow:80` |
| `probe_episodes` | 20 → 5 (`:236`) | **5** | `ow:81` |
| `probe_episode_length` | falls back to `episode_length` (`:237-239`) | **8192** | `ow:82` |
| `probe_goals` | — | **12-rung ladder**: flags 1735 / 30 / 31 / 65 / 66 / 1213, maps 24:4, 24:5, 26:10, 10:5, pokédex-owned thresholds 1 and 3 | `ow:83-96` |
| `probe_label` | `"probe"` (`:232`) | **`"ladder_rungs"`** | `ow:97` |
| `long_probe_enabled` | False (`:254`) | **true** | `ow:99` |
| `long_probe_frequency` | 200 (`:257`) | **1000** | `ow:100` |
| `long_probe_episodes` | 3 | **5** | `ow:101` |
| `long_probe_episode_length` | `4 * episode_length` (`:259-261`) | **24576** | `ow:102` |

The probe builds a **copy** of the config with `terminate_on_goal_complete=False`
and `record=False`, so a probe never ends the run and never pollutes the recordings
(`vec_ppo_agent.py:240-247`). Metric series: `probe_success_rate`,
`probe_goal_type_rates`, `probe_fire_steps`, plus the `assisted_*` and
`long_probe_*` counterparts (`vec_ppo_agent.py:381-415`).

The ladder's purpose is time-to-rung, not reward: `goal_fire_steps` /
`flag_fire_steps` are reported per episode to locate the bottleneck rung
(`vec_env.py:194-202`), and `tools/training_health_report.py` reads the frontier-pool
size and probe ladder straight out of `Checkpoints/info.pth` (`AGENTS.md:46-55`,
`:70-76`).

## The honest-coverage reading rule

This is the rule that decides whether any of the above is working
(`AGENTS.md:78-88`):

1. **Split before judging.** `seeded[i]` marks episodes started from a Go-Explore
   snapshot; they begin deep in already-solved territory, so their reward *and*
   coverage are inflated, and they are where farmable channels get exercised hardest.
   The policy's real competence is the honest subset only (`AGENTS.md:78-83`;
   provenance plumbing above at `vec_ppo_agent.py:333`).
2. **Use the pre-computed honest aggregates.** `honest_mean(key)` over unseeded
   episodes (`vec_ppo_agent.py:1833`), published as `h_r` / `h_maps` / `h_cells` /
   `h_flags` (`:1842-1851`); `summary.honest_*` in the metrics JSON
   (`AGENTS.md:83`). `best_discoveries` counts honest checkpoint recordings only
   (`vec_ppo_agent.py:2014-2016`).
3. **Trend, not mean.** Bucket the honest indices over episode order and read the
   *trend*; the aggregate `mean_reward` / `max_reward` conflate both populations and
   can climb while the honest policy collapses — that is exactly how the 2026-07-20
   run hid a fall from 16 to 5.6 honest unique_maps behind a rising headline reward
   (`AGENTS.md:84-88`).
4. **Plateau vs collapse.** Flat `archive_size` growth with honest coverage *holding*
   is a plateau; honest coverage *declining* is a collapse, usually a reward channel
   pulling the policy off-task (`AGENTS.md:119-122`).
5. **Seed pressure is a tunable, and it is currently lowered.** The seeding
   probability is 0.4 in the canonical open-world config against a code default of
   0.6 (`ow:64`, `vec_ppo_agent.py:142`) — if honest coverage is falling while
   `h_cells` stagnates, raising the honest share (probe fraction up, seed fraction
   down) is the first knob, before any reward change
   ([../configuration.md](../configuration.md),
   [../operations/run-triage.md](../operations/run-triage.md)).

## Cross-links

- [rewards.md](rewards.md) — novelty pricing, depletion counters, honest/seeded
  budgets, `DomainRewardBalancer`.
- [environment.md](environment.md) — where captures happen in `step()`, and
  `terminal_info["frontier_captures"]`.
- [training.md](training.md) — the rollout loop that calls `_goexplore_cycle`.
- [models.md](models.md) — the model that consumes the seeded observations.
- [../architecture.md](../architecture.md), [../branch-changes.md](../branch-changes.md)
  — Go-Explore in the system map and on this branch.
- [../configuration.md](../configuration.md), [../operations/run-triage.md](../operations/run-triage.md).

## Known gaps

Carried forward from the §Unverified section of those earlier reconnaissance notes,
plus this pass's limits:

- **Capture-cap enforcement site not located.** For
  `goexplore_max_captures_per_episode`, `goexplore_capture_cell_count_max`,
  `goexplore_max_progress_captures_per_episode` and `goexplore_precheckpoint_interval`
  only the config values are quotable; the capture-generating code (likely the
  `PoliwhiRL/environment/*` env layer) was not opened. This pass confirmed the
  per-episode counters are initialised in `gym_env.py:1162-1170` and that captures
  are attempted at `gym_env.py:1040-1041`, but **not** the comparison that enforces
  the caps.
- Bodies of `_frontier_key`, `_is_progress_capture` and `_live_count` were not read;
  the pool-partition semantics above come from docstrings and call sites
  (`vec_ppo_agent.py:1005`, `:1024-1026`, `:1031-1032`).
- No `vec_ppo_agent.py` or `configs/stages/open_world.json` line was re-read in
  this documentation pass: every `vec_ppo_agent.py:*` and `ow:*` citation is
  transcribed from the scratch pass. `vec_ppo_agent.py` is 2287 L — read ranges.
- Only the env-side capture plumbing was verified this pass
  (`gym_env.py:1037`, `:1040-1041`, `:1162-1170`); the agent-side curation logic
  (`vec_ppo_agent.py:999-1033`) is second-hand here.
- **`goals.py` internals were never opened** (568 L). The claim that only
  `map_fires` still pays reward is docstring-backed (`rewards.py:684-689`), and the
  goal-rung fields that reach the agent (`goal_fire_steps`, `n_map`) are cited in
  [environment.md](environment.md) — but the ladder semantics inside `goals.py` are
  unverified.
- Exact `RAM_OBS_DIM` is **170**, computed from source: 121 base keys
  (`PoliwhiRL/environment/gym_env.py:66-237`) plus 49 derived-flag pairs
  (`PoliwhiRL/checkpoints.py:11-60`), assigned at `gym_env.py:242`. See
  [environment.md](environment.md#known-gaps). It is the shape of every seeded
  observation the probe and cycle manipulate.
- No web access used; every statement is from local source at this branch.
