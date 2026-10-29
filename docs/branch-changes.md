# Branch changes: `flag31-breakthrough-and-tuning` vs `main`

What this page is for: what actually changed on this branch — the six clusters of
work, the pivotal file in each, the behaviour change each cluster introduces, and
the files and config keys that were removed, superseded or introduced. It is the
page to read before reviewing a diff, debugging a "this used to work" regression,
or deciding which branch's docs to trust.

**Sources**: `git diff` against merge-base
`a06e656cd9ead6802a22f3d04e2dd3f0616ebd35`; `main.py`; `PoliwhiRL/PPO.py`;
`PoliwhiRL/environment/*`; `PoliwhiRL/models/PPO/*`;
`PoliwhiRL/agents/PPO/*`; `PoliwhiRL/replay/*`; `configs/**`; `tools/`.

---

## At a glance

```
merge-base  a06e656cd9ead6802a22f3d04e2dd3f0616ebd35
            (verified: `git merge-base main HEAD`)
            123 files changed, 26153 insertions(+), 1975 deletions(-)
            6 files deleted · 3 files renamed · ~93 files added · ~21 modified
```

The shape of the branch is visible in those numbers: this is not a tuning branch
despite its name. The overwhelming majority of new files are new *subsystems*
(`vec_ppo_agent.py`, `_minibatch.py`, `vec_ppo_storage.py`, `visit_archive.py`,
`domain_reward_balancer.py`, `action_mask.py`, `rom_collision.py`, `checkpoints.py`)
plus ~38 new test modules. The tuning is a small slice of cluster 5.

---

## Cluster 1 — Reward system rebuild

| Pivotal files | Status |
|---|---|
| `PoliwhiRL/environment/rewards.py` | modified (1627 L) |
| `PoliwhiRL/environment/domain_reward_balancer.py` | **added** |
| `PoliwhiRL/checkpoints.py` | **added** |
| `PoliwhiRL/environment/event_flags.asm` | added |
| `configs/default_configs/reward_settings.json` | modified |
| `configs/evaluate_reward_system.json` | modified |
| `PoliwhiRL/reward_evaluator/` → `PoliwhiRL/reward_evaluation/` | renamed |

**Behaviour change.** Reward calculation becomes a structured, auditable pipeline
rather than a sum of ad-hoc bonuses: validity gate
(`rewards.py:680`, `is_ram_state_valid` `:107-119`) → per-source raw computation
(`:697-714`) → per-episode breakdown with provenance split (`:716-731`) → raw
per-domain sums (`:741-752`) → one `domain_balancer.scale()` call per domain
(`:753-756`) → sum (`:764-771`) → truncation decisions (`:835-881`) →
round/clip/cast with full bookkeeping (`:885-893`) → return
(`:895`). Four accounting-only adjustment keys
(`rounding_adjustment`, `clip_adjustment`, `delivery_cast_adjustment`,
`delivered_total`) make the delivered reward reconcilable to the raw one
(`:890-893`).

`DomainRewardBalancer` is new: four fixed domains
(`domain_reward_balancer.py:74`), mean-based rather than std-based rescaling
because a near-constant stream has ~zero variance and std-division would blow the
factor up (`:22-36`), a progress-activity multiplier (`:142-147`), and static
priority weights as the durable lever — raising a raw source magnitude is
self-defeating because the running mean absorbs it (`:48-61`). `catching` was
split out of `milestone` so pokédex rewards can be pushed hard without diluting
flag/checkpoint/event's shared factor (`:15-20`, `:84-94`). It is off by default
(`:117`) and enabled per stage (`configs/stages/open_world.json:38`). Scope
limit: per-worker-local, **not** broadcast across workers, unlike
`VisitArchive` (`:63-70`).

Anti-farming guards are the substantive addition: run-wide depletion plus
per-episode threshold dedupe (`_milestone_refire_scale` `rewards.py:927-962`),
the battle episode cap (`:1247-1249`), `(map_key, battle_type)` decay keying so
wild and trainer grinding do not share a counter (`:1194-1195`, `:1201-1202`,
`:1232-1233`), one-time first-trainer-battle semantics that are deliberately
exempt from the cap (`:1208-1227`, `:1174-1177`), `map_lifelong_decay` with no
floor (`:1298-1305`), and discounted revisits after a flag re-open
(`:1346-1367`). Each guard cites the specific observed failure it prevents —
for example `key_items_count` oscillation drove that channel to ~17k/episode and
collapsed honest unique_maps from 16 to 5.6 (`:951-954`).

`PoliwhiRL/checkpoints.py` is new and is, in its own words, "the
observation-vector contract" (`checkpoints.py:4`): an ordered tuple of **49**
`(flag_num, name)` pairs (`:11-61`, counted from the literal at `:12-60`), plus
`NON_RECORDABLE_FLAGS = frozenset({26, 1726})` for observation-only bits
(`:65-66`) and the title/slug helpers (`:68-101`).

## Cluster 2 — Go-Explore

| Pivotal files | Status |
|---|---|
| `PoliwhiRL/environment/visit_archive.py` | **added** (413 L) |
| `PoliwhiRL/agents/PPO/vec_ppo_agent.py` | **added** (seeding cycle lives here) |
| `PoliwhiRL/environment/gym_env.py` | modified (snapshot load/save) |

**Behaviour change.** Frontier memory becomes persistent and shared. The visit
archive quantises the world into cells with `CELL_SIZE = 2` tiles — coarser than
per-tile, finer than per-map (`visit_archive.py:29`, `:41`), via `quantise()`
which floor-divides both coordinates (`:44-48`,
`(bank, num, x // CELL_SIZE, y // CELL_SIZE)`), exposed as
`VisitArchive.cell_key` (`:406`) and used by the frontier bonus
(`rewards.py:1342`).

The agent runs a capture/sew cycle: ingested captures are partitioned by
`_is_progress_capture`, each pool sorted by `_live_count` and truncated to its
cap, then concatenated (`vec_ppo_agent.py:1022-1033`), with separate spatial
(`:150-154`) and progress (`:144-149`) pool sizes under a total pool cap.
`_pick_frontier_snapshot` picks the progress pool with probability
`goexplore_progress_seed_fraction` (default `0.25`, `:156-158`) and then samples
with weight `1/(1 + live_count)` so seeding concentrates on the rarest,
true-frontier locations while occasionally revisiting better-learned ground
(`:1061-1068`). `_goexplore_cycle` never seeds probe workers
(`:1072-1073`, `:1079`), so a fixed fraction of workers always trains from the
true start (`:134-136`; env 0 always a probe, `:546-548`); non-probe workers
seed with probability `goexplore_seed_fraction` and are otherwise returned to
`_true_start_path` with `episode_seeded=False` (`:1095-1103`). Seed failures are
caught and logged, leaving the worker honest (`:1093-1094`, `:1102-1103`).
Archive broadcast back to workers is a full-table replace, self-healing
(`:1106-1115`). Files are never deleted mid-run because a worker may be about to
load one; the snapshot dir is cleared only for a fresh run (`:999-1002`).

Provenance is first-class throughout: honest checkpoints pay
`checkpoint_progress_reward` (floor `honest_checkpoint_reward_floor`) while seeded
pay the lower `seeded_checkpoint_progress_reward` (`rewards.py:978-983`), with
separate archive counters per provenance
(`visit_archive.py:232`, `:239`, `:371`, `:393`). See
[architecture.md](architecture.md#the-single-most-important-reading-rule-split-honest-vs-seeded).

## Cluster 3 — Environment & observation

| Pivotal files | Status |
|---|---|
| `PoliwhiRL/environment/gym_env.py` | modified |
| `PoliwhiRL/environment/RAM.py` | modified |
| `PoliwhiRL/environment/vec_env.py` | **added** |
| `PoliwhiRL/environment/action_mask.py` | **added** |
| `PoliwhiRL/environment/rom_collision.py` | **added** |
| `PoliwhiRL/environment/goals.py` | **added** |
| `tools/rom_maps/*` (4 files), `Scripts/verify_flags_from_states.py`, `Scripts/make_states.py`, `emu_files/states/zero.steps`, `RAM_MAPPING.md` | added / modified |

**Behaviour change.** The observation shrinks in width and grows in information
density. Raw 256-byte story flags were replaced by the curated derived-flag table
(`gym_env.py:33`, keys derived at `:240`, iteration `:299`), and coarser
temporal history is now delivered explicitly as RAM features — recent maps,
steps-since-novel-cell, run-wide cell visit counts — which is what allows the
network to drop absolute positional encoding entirely
(`PPOTransformer.py:162-171`).

The vectorised worker is new and owns the terminal contract: on `done` it
snapshots progress *before* the auto-reset wipes goal state
(`vec_env.py:131-134`), emitting `goal_success` (`:149`), `flag_fires` from the
derived-table fire log rather than the configured-goal count (`:150-159`),
monotonic `unique_cells` (`:160-163`), `unique_maps` (`:164`), `archive_size`
(`:165`), the reward breakdown (`:166`), the GAE `truncated` flag set for
budget truncation only (`:167-176`) and `truncation_cause` (`:177`). The
`flag_fires` fix is itself a documented behaviour change: the metric previously
read the configured-goal count, which is always 0 in freeform stages and made
honest story progress invisible (observed 2026-07-20, `:150-158`).

Action masking is legal-action filtering computed from RAM, enabled by default
(`configs/curriculum_base.json:40`) and applied before softmax
(`PPOTransformer.py:242-250`). `rom_collision.py` plus the `tools/rom_maps/`
extractor support the runtime collision field.

## Cluster 4 — Models & PPO

| Pivotal files | Status |
|---|---|
| `PoliwhiRL/models/PPO/PPOTransformer.py` | modified |
| `PoliwhiRL/models/PPO/ppo_model_implementation.py` | modified |
| `PoliwhiRL/models/CNN/GameBoy.py` | modified |
| `PoliwhiRL/agents/PPO/vec_ppo_agent.py` | **added** (2287 L) |
| `PoliwhiRL/agents/PPO/_minibatch.py` | **added** |
| `PoliwhiRL/replay/vec_ppo_storage.py` | **added** |
| `PoliwhiRL/utils/running_stats.py` | **added** |
| `PoliwhiRL/agents/PPO/ppo_agent.py` | **deleted** |
| `PoliwhiRL/models/transformers/positional_encoding.py` | **deleted** |
| `PoliwhiRL/replay/ppo_storage.py` | modified (now legacy) |
| `PoliwhiRL/agents/__init__.py`, `PoliwhiRL/agents/PPO/__init__.py`, `PoliwhiRL/replay/__init__.py`, `PoliwhiRL/utils/__init__.py` | modified |

**Behaviour change.** One training implementation replaces two. `ppo_agent.py`
was deleted in commit `10766b3` and `VecPPOAgent` is the only agent the entry
point constructs (`PPO.py:56`, `:77`) — "there is ONE training implementation;
`num_envs == 1` just runs a single worker" (`PPO.py:61-64`). Update logic was
extracted into `_minibatch.py`.

Memory semantics changed twice. `positional_encoding.py` was deleted in commit
`889f54a` ("pre-rebuild"), replaced by the learned, zero-initialised `age_emb`
inside `TransformerXLBlock` (`PPOTransformer.py:65-72`, `:89-95`, `:115`) —
deliberately, because with single-frame chunks an absolute encoding is a constant
(`:163-171`). Training now runs real BPTT over contiguous segments with detached
prior memory as context, looping position-by-position so the age embedding is
applied relative to the current frame exactly as at rollout
(`PPOTransformer.py:49-57`, `:97-125`).

Geometry became explicitly config-driven, and the in-code comment records why: the
constructor defaults "silently gave a 2-layer/4-head/128-wide trunk regardless of
the '4-layer' docs" (`ppo_model_implementation.py:189-193`). See
[models.md](subsystems/models.md#geometry-three-tiers-of-default-and-the-one-that-runs)
for the tier table and the `d_ram`/`mem_len` correction.

Sampling changed from a uniform-epsilon mix to temperature reshaping
(`ppo_model_implementation.py:80-92`, `:114-118`), the per-timestep stagnation
entropy boost was removed (`:397-401`), and the adaptive entropy servo plus
plateau boosts were removed (`vec_ppo_agent.py:189-192`) — the latter two are
the branch's most surprising live behaviours, documented as current state in
[training.md](subsystems/training.md#entropy-coefficient-paths-and-two-dead-knobs).

On the stability side: a replay-consistency gate that can skip a whole update
(`_minibatch.py:81-108`), hard-KL rollback with a permanent LR cut
(`:144-150`), a clamped log-ratio to survive MPS overflow
(`ppo_model_implementation.py:370-375`), dropout pinned to `0.0` because the
trainer never calls `eval()` (`:197-204`), and a dropped step when the pre-clip
gradient norm is non-finite (`:442-458`). `ppo_hard_kl` and
`ppo_replay_tolerance` are opt-in per stage (`_minibatch.py:77-79`; values
`configs/stages/open_world.json:21-22`).

## Cluster 5 — Configs & entry point

| Pivotal files | Status |
|---|---|
| `main.py` | modified |
| `PoliwhiRL/PPO.py` | modified |
| `configs/curriculum_base.json` | **added** |
| `configs/stages/{first,second,third,fourth,fifth,sixth}.json`, `configs/stages/freeplay.json`, `configs/stages/open_world.json`, `configs/stages/open_world_probe_temp.json` | **added** |
| `configs/default_configs/{core,episode,outputs,ppo,reward}_settings.json` | modified |
| `configs/debug_eval.json`, `configs/inference.json`, `configs/random_walker.json`, `configs/explore.json` | added / modified |
| `configs/first_steps.json`, `configs/second_steps.json` | **deleted** |
| `pyproject.toml`, `uv.lock`, `requirements.txt`, `.gitignore` | added / modified |

**Behaviour change.** Config inheritance gains an `extends` keyword: user configs
resolve a parent chain with circular-reference detection (`main.py:44-74`,
raised at `:53`, missing parent at `:67-70`), and the merged result overlays the
glob-loaded defaults (`main.py:77-98`, `:111`). The old flat
`configs/first_steps.json` / `configs/second_steps.json` were deleted and
replaced by `configs/curriculum_base.json` plus the numbered
`configs/stages/first.json` … `sixth.json`, every one of which declares
`"extends": "../curriculum_base.json"` (verified in all six, plus
`freeplay.json` and `open_world.json`; `open_world_probe_temp.json` extends
`open_world.json`). **Not part of that verified ladder:**
`configs/stages/freeform_gamearea_event.json` is untracked in git and its declared
parent `freeform_gamearea_goexplore_cell.json` does not exist in `configs/stages/`,
so running it raises `FileNotFoundError` at `main.py:67-70`; see the broken-config
caveat in [`configuration.md`](configuration.md).

Two implicit behaviours became defaults. **Auto-resume**: re-running the same
config picks up the latest checkpoint in its own output folder instead of
silently restarting from scratch (`PPO.py:5-29`). **Auto-gamma**: `"auto"`
derives `min(0.999, 1 - 4/episode_length)` from the episode budget, replacing the
`0.995` cap that pinned the horizon near 200 steps and made distant milestones
worth ~0 (`PPO.py:32-51`, rationale `:37-43`; `"auto"` set at
`configs/curriculum_base.json:5`). Note that `main.py` also globs every
`configs/default_configs/*.json` in one pass (`:21-22`) and remaps
output-base-relative string keys when a stage changes `output_base_dir`
(`:81-96`) — the mechanism by which `core_settings.json` now pins architecture
dimensions for every run.

> **Correction to the brief.** The deleted stage configs are
> `configs/first_steps.json` and `configs/second_steps.json`, **not**
> `configs/stages/first_steps.json` / `configs/stages/second_steps.json`
> (`git diff --diff-filter=D --name-only` against the merge-base).

## Cluster 6 — Tools, eval, tests

| Pivotal files | Status |
|---|---|
| `tools/training_health_report.py`, `tools/world_map.py`, `tools/png_to_video.py`, `tools/llm_game_debug.py`, `tools/rom_maps/` | **added** |
| `PoliwhiRL/evaluator/` (`__init__.py`, `debug_evaluator.py`, `model_evaluator.py`) | **added** |
| `PoliwhiRL/explorer/random_walker.py` | **added** |
| `PoliwhiRL/reward_evaluator/` → `PoliwhiRL/reward_evaluation/` | **renamed** |
| `tests/` — ~38 new modules; `tests/test_ppo_memory.py`, `test_pyboy_ram.py`, `test_pyboyenv.py` modified; `test_PPO.py`, `test_reward_system.py` deleted | added / modified / deleted |
| `AGENTS.md`, `README.md`, `CHANGELOG.txt`, `PLANNING.md`, `PROGRESS.md`, `next_steps.md`, `.opencode/skills/training-run-analysis/` | added / modified |
| **`.github/workflows/run_tests.yml`** | **NOT modified on this branch** |

> **Cluster 6, explicitly:** `.github/workflows/run_tests.yml` exists in the
> working tree and was **not** modified on this branch. `git diff --name-only
> <merge-base> HEAD -- .github/` returns **no paths** — nothing under `.github/`
> appears anywhere in the 123-file diff. CI configuration is untouched; if a test
> failure appears on this branch it is not because CI was edited.

**Behaviour change.** Run triage becomes scripted rather than ad-hoc.
`tools/training_health_report.py` reads a checkpoint's `info.pth` for repeatable
triage — optimisation health, honest/seeded reward mix, archive growth, discovery
log, time-to-frontier, checkpoint reach, probe ladder (`AGENTS.md`, "Inspecting a
run"); `tools/world_map.py` reconstructs an empirical per-tile
walkability/warp/encounter map plus a chronological discovery timeline from
recorded-episode PNG filenames. The repo now carries three new inference/eval
entry modes wired into the dispatcher (`main.py:188-195`) alongside the added
`PoliwhiRL/evaluator/` package and `configs/{debug_eval,inference,random_walk
er}.json`.

The test suite is effectively rewritten around the new stack:
`tests/test_PPO.py` and `tests/test_reward_system.py` were deleted and replaced
by `tests/test_vec_ppo_memory.py` (storage shapes/semantics, e.g. `:91-92`,
`:212-213`), `tests/test_vec_ppo_correctness.py` (mask/KL/stuck-temperature/
`_per_env_gae`; `:44`, `:175-176`, `:209-224`, `:320`, `:501-527`) and
`tests/test_minibatch.py` (`ppo_hard_kl`, `halve_lr_multiplier`, replay
tolerance; `:21`, `:50`, `:176`). `tests/test_ppo_memory.py` still exercises the
legacy `PPOMemory`.

**Authoritative `tools/` listing** (`ls tools/`, working tree today):
`training_health_report.py`, `world_map.py`, `png_to_video.py`,
`llm_game_debug.py`, `ram_flag_deltas.jsonl`, `rom_maps/`, `llm_debug_frames/`,
`llm_debug.db`, `__pycache__/`. The four Python tools named in `AGENTS.md` all
exist; no other scripts were assumed.

---

## Removed / superseded — consolidated

| Path | Status | Evidence | Replaced by |
|---|---|---|---|
| `PoliwhiRL/agents/PPO/ppo_agent.py` | **deleted** | commit `10766b3`; `git diff --diff-filter=D` | `vec_ppo_agent.py` (`VecPPOAgent`) — `PPO.py:56`, `:77` |
| `PoliwhiRL/models/transformers/positional_encoding.py` | **deleted** | commit `889f54a` ("pre-rebuild") | learned `age_emb` — `PPOTransformer.py:65-72`, `:89-95`, `:115` |
| `tests/test_PPO.py` | **deleted** | commit `10766b3` | `test_vec_ppo_memory.py`, `test_vec_ppo_correctness.py`, `test_minibatch.py` |
| `tests/test_reward_system.py` | **deleted** | `git diff --diff-filter=D` | the ~15 cluster-1 reward test modules |
| `configs/first_steps.json` | **deleted** | `git diff --diff-filter=D` | `configs/curriculum_base.json` + `configs/stages/first.json` |
| `configs/second_steps.json` | **deleted** | `git diff --diff-filter=D` | `configs/curriculum_base.json` + `configs/stages/second.json` |
| `PoliwhiRL/reward_evaluator/` | **renamed** | `git diff -M`: `__init__.py` and `moves.py` R100, `evaluator.py` R058 | `PoliwhiRL/reward_evaluation/` (imported at `main.py:5`) |
| `PoliwhiRL/replay/ppo_storage.py` (`PPOMemory`) | **superseded, not deleted** | still exported `PoliwhiRL/replay/__init__.py:2-5`, tested by `tests/test_ppo_memory.py` | `VecPPOMemory` — `vec_ppo_agent.py:116` |
| uniform-epsilon exploration mix | removed (mechanism) | `ppo_model_implementation.py:80-92` | temperature reshaping `:114-118` |
| per-timestep stagnation entropy boost | removed (mechanism) | `ppo_model_implementation.py:397-401` | reward-driven exploration |
| adaptive entropy servo + plateau boosts | removed (mechanism) | `vec_ppo_agent.py:189-192` | constant / time-based schedule |
| raw 256-byte story-flag bytes | removed (representation) | `gym_env.py:33`, `:239` | curated `_DERIVED_FLAG_TABLE` (`checkpoints.py:11-61`) |

---

## Config keys introduced

Keys added to the shared defaults on this branch, i.e. present in the
`configs/default_configs/` diff and absent from `main`. Because
`main.py:20-35` loads every file in that directory, all of these are live on
**every** run unless a stage overrides them.

| Group | Keys added | Where |
|---|---|---|
| Architecture | `d_model` (128), `d_ram` (**128**), `mem_len` (**128**), `n_heads` (4), `num_layers` (2), `ppo_dropout` (0.0) | `configs/default_configs/core_settings.json:16-21` |
| Optimisation | `ppo_adam_eps`, `ppo_clip_value_loss`, `ppo_epochs`, `ppo_gamma`, `ppo_lr_min`, `ppo_max_grad_norm`, `ppo_minibatch_size`, `ppo_target_kl` | `configs/default_configs/ppo_settings.json` |
| Entropy | `ppo_entropy_anneal_enabled` | `configs/default_configs/ppo_settings.json:11` |
| Load semantics | `reset_lr_scheduler_on_load`, `reset_optimizer_on_load` | `configs/default_configs/core_settings.json:14-15` |
| Reward — milestone | `flag_progress_reward`, `checkpoint_progress_reward`, `seeded_checkpoint_progress_reward`, `honest_checkpoint_reward_floor`, `map_goal_reward`, `maps_visited_reward`, `key_item_pickup_reward`, `level_up_reward` | `configs/default_configs/reward_settings.json` |
| Reward — battle | `battle_engagement_reward`, `battle_win_reward`, `battle_decay_coef`, `battle_reward_episode_cap`, `trainer_battle_premium_multiplier`, `first_trainer_battle_reward`, `first_trainer_battle_seeded_reward`, `first_trainer_battle_honest_floor` | ibid. |
| Reward — frontier | `frontier_novelty_bonus`, `frontier_lifelong_decay`, `frontier_lifelong_decay_floor`, `new_map_reward`, `new_bank_reward`, `map_lifelong_decay`, `frontier_sense_radius` | ibid. |
| Reward — misc | `whiteout_penalty`, `step_penalty`, `reward_clip`, `reward_round_dp` | ibid. |
| Outputs / runtime | `export_state_loc`, `snapshot_dir`, `terminate_on_goal_complete` | `configs/default_configs/outputs_settings.json`, `episode_settings.json` |

Stage-only keys introduced in `configs/stages/open_world.json` (not in shared
defaults, so opt-in): `ppo_hard_kl` (`:21` = `0.06`), `ppo_replay_tolerance`
(`:22` = `1e-5`), `ppo_lr_hard_floor` (`:24`), plus the whole
`goexplore_*` / `probe_*` / `domain_reward_balancing_*` surface
(`:33`-`:102`).

Stage keys introduced in `configs/curriculum_base.json`: `ppo_gamma: "auto"`
(`:5`), `entropy_servo_enabled` (`:11`), `entropy_plateau_reset` (`:12`,
**unread by any code**), `ppo_lr_schedule` (`:14`), `reset_reward_scaler_on_load`
(`:18`), `advantage_normalisation` (`:19`), `scaler_min_std` (`:21`),
`stagnation_truncation_steps` / `battle_stagnation_truncation_steps` (`:23-24`,
both `"auto"`), `best_success_window` / `best_success_min_episodes` (`:26-27`),
`critic_warmup_rollouts` (`:29`), `ram_recent_maps_n` (`:30`),
`early_stopping_*` (`:35-38`), `action_mask_enabled` /
`allow_menus_walking` (`:40-41`).

---

## Known gaps

- **Cluster attribution is derived from paths, not from commit messages.** The
  six clusters are a partition of `git diff --name-status` against the merge-base
  by directory and role; the branch's own commit history was not decomposed, so
  individual files could arguably sit in a different cluster (`vec_ppo_agent.py`
  in particular spans clusters 2, 4 and 6).
- **File counts are approximate.** "Added" and "modified" totals were derived by
  subtracting the exact deleted (6) and renamed (3) sets from the authoritative
  123-file diff, not by an independent per-status tally.
- **`d_ram` / `mem_len` effective value.** The correction to 128 follows from
  `core_settings.json:16-17` + `main.py:21-22`; it was not confirmed at runtime
  by printing a built model's dimensions.
- **Capture-cap enforcement** and the bodies of `_frontier_key`,
  `_is_progress_capture`, `_live_count` remain unverified
  (untracked working-tree notes under `.doc-scratch/`, never committed,
  therefore not citable from a fresh clone).
- **`GameBoyBlock` kernel/stride/pooling factors** and the resulting
  `flat_features` width remain unverified
  (untracked working-tree notes under `.doc-scratch/`, never committed,
  therefore not citable from a fresh clone).
- **`goals.py`** (568 L) was not opened; the claim that only `map_fires` still
  pays reward is docstring-backed (`rewards.py:684-689`), not code-verified.
- **No tests, no emulator, no training were run** while producing this page, so
  nothing here is confirmed by execution — only by reading source at this
  branch/commit. `git` facts are the one exception, and those were run.
