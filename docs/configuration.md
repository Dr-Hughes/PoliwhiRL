# Configuration

What this page is for: how a run's configuration is assembled — the default
concatenation, `extends` inheritance, the merge order, the auto-generated CLI
flags, the catalogue of shipped configs and their inheritance chains, the
output layout, and the mode-dispatch table. Read this before changing any
`configs/*.json`.

**Sources**: `main.py`, `configs/default_configs/*.json`,
`configs/curriculum_base.json`, `configs/stages/*.json`,
`configs/{explore,random_walker,inference,debug_eval,evaluate_reward_system}.json`.

## Loading model

`python main.py --use_config <path>` (`main.py:104-110`). The flag name is
`--use_config`; there is no `--config` (`main.py:106`). Four stages, in order:

1. **Base config** — a shallow concatenation of *every* `*.json` in
   `configs/default_configs/`, in `glob()` order (`main.py:20-35`). There is no
   `configs/base.json`; if the directory is empty the loader raises
   `FileNotFoundError` (`main.py:24-25`). For the file whose path contains
   `"outputs"` — `default_configs/outputs_settings.json` — every key except
   `output_base_dir` is prefixed with the base dir (`main.py:29-30`,
   `main.py:38-41`).
2. **User config** — `load_user_config()` reads `--use_config` and resolves
   `extends` recursively (`main.py:44-74`). A missing path yields `{}`, so
   running without `--use_config` is legal and runs on defaults alone
   (`main.py:45-46`). Relative `extends` paths are resolved against the
   **child file's own directory** (`main.py:63-66`); a missing parent raises
   `FileNotFoundError` (`main.py:67-70`); cycles are detected through an
   absolute-path `_seen` set and raise `ValueError` (`main.py:49-54`).
3. **Default ↔ user merge** — `merged_config = default_config.copy()` then
   `.update(user_config)` (`main.py:77-79`). Output-base-relative defaults are
   then re-pointed at the new base dir (`main.py:84-96`).
4. **CLI overrides** — every merged key is registered as its own argparse flag,
   then `final_config.update(cmd_args)` (`main.py:155-158`), with a second
   output-base remap if `--output_base_dir` changed (`main.py:160-168`).

So precedence is: **CLI > user config (and its `extends` chain) > default
configs**.

## ⚠️ All merges are shallow — the one gotcha to internalise

Every merge in this pipeline is a single top-level `dict.update`:

| Merge point | Code |
|---|---|
| default files ↔ each other | `main.py:34` |
| `extends` parent ↔ child | `main.py:73` |
| defaults ↔ user config | `main.py:79` |
| merged config ↔ CLI args | `main.py:158` |

**A child key replaces the parent's value wholesale. Nested dicts and lists are
never deep-merged.** If a parent sets
`"domain_priority_weights": {"catching": 1.75}` and a child sets
`"domain_priority_weights": {"battling": 1.0}`, the parent's `catching` entry
silently disappears (`main.py:73`). The same applies to list-valued keys such as
`goals`, `probe_goals`, `extra_files`, `event_novelty_exclude_flags` and
`ignored_buttons` (`configs/stages/open_world.json:83-96`,
`configs/inference.json:15-20`,
`configs/default_configs/episode_settings.json:7`). **Whenever a child overrides
a dict- or list-valued key, restate the whole value.**

## Auto-generated CLI flags

Every key in the merged config becomes a CLI flag of its own type, with
`argparse.SUPPRESS` as the default so an absent flag leaves the config value
untouched (`main.py:115-128`). Booleans use `StoreBooleanAction`, which accepts
`yes|true|t|1` (case-insensitive) and treats anything else as `False`
(`main.py:15-17`) — `--erase true` works, `--erase` alone does not.

Three extra flags exist that are *not* config keys (`main.py:130-152`):

| Flag | Meaning | Source |
|---|---|---|
| `--add_rollouts N` | Run N rollouts this session, overriding `num_rollouts` | `main.py:140-145` |
| `--add_episodes N` | Same, expressed in episodes; converted via `episode_length / (ppo_update_frequency * num_envs)` | `main.py:146-152` |
| `--use_config PATH` | Path to the user config file | `main.py:130-135` |

`erase` triggers `shutil.rmtree(config["output_base_dir"])` before dispatch
(`main.py:176-178`); it defaults to `false`
(`configs/default_configs/core_settings.json:3`). `verbose: true` prints the
resolved config and waits for Enter (`main.py:180-182`) — the quickest way to
confirm what the merge actually produced.

## Config catalogue

| File | `extends` | Role | Source |
|---|---|---|---|
| `default_configs/core_settings.json` | — | model, device, vision, checkpointing, transformer dims | `core_settings.json:1-22` |
| `default_configs/outputs_settings.json` | — | `output_base_dir` + output sub-dirs | `outputs_settings.json:1-8` |
| `default_configs/rom_settings.json` | — | ROM, state, extra files | `rom_settings.json:1-8` |
| `default_configs/episode_settings.json` | — | horizons, recording, early stopping | `episode_settings.json:1-13` |
| `default_configs/ppo_settings.json` | — | optimiser hyper-parameters | `ppo_settings.json:1-19` |
| `default_configs/reward_settings.json` | — | every reward channel magnitude | `reward_settings.json:1-32` |
| `curriculum_base.json` | — | shared parent for all stages | `curriculum_base.json:1-42` |
| `stages/first.json` … `stages/sixth.json` | `../curriculum_base.json` | hand-authored ladder | `first.json:2`, `second.json:2`, `third.json:2`, `sixth.json:2` |
| `stages/freeplay.json` | `../curriculum_base.json` | post-ladder free play | `freeplay.json:2` |
| `stages/open_world.json` | `../curriculum_base.json` | canonical from-scratch run | `open_world.json:2` |
| `stages/open_world_probe_temp.json` | `open_world.json` | probe-horizon experiment — **deprecated / retired**: its `stuck_action_temperature: 9.0` + `stuck_temperature_onset_exponent: 0.5` pair (`:5-6`) was falsified head-to-head and the experiment is marked retired in [`roadmap.md`](roadmap.md); do not run it as if it were current | `open_world_probe_temp.json:2` |
| `stages/freeform_gamearea_event.json` | `freeform_gamearea_goexplore_cell.json` | event-novelty variant — **untracked, parent missing — would raise `FileNotFoundError` at `main.py:67-70`; do not run** (see caveat) | `freeform_gamearea_event.json:2` |
| `explore.json` | — | `model: explore` memory collection | `explore.json:2` |
| `random_walker.json` | — | `model: random_walk` map discovery | `random_walker.json:2` |
| `inference.json` | — | `model: inference` | `inference.json:2` |
| `debug_eval.json` | — | `model: debug_eval` | `debug_eval.json:2` |
| `evaluate_reward_system.json` | — | `model: reward_eval` | `evaluate_reward_system.json:2` |

### Shared curriculum parent

`curriculum_base.json` carries the values every stage inherits, notably
`sequence_length: 16` (`:2`), `ppo_lr_schedule: "cosine"` (`:14`),
`action_mask_enabled: true` (`:40`) and `checkpoint_frequency: 50` (`:33`).
Note it also sets `num_envs: 8` (`:3`) and `ppo_epochs: 4` (`:7`) — these
*differ* from the default configs (`num_envs: 8`,
`configs/default_configs/episode_settings.json:4`; `ppo_epochs: 8`,
`configs/default_configs/ppo_settings.json:4`), which is exactly why shallow
replacement is visible: a child that wants five epochs must restate the key, not
edit the parent's nested content.

### Ladder chaining

The six stages chain by `load_checkpoint` pointing at the **previous stage's**
`Checkpoints/best`, while all six extend the same parent:

| Stage | Resumes from | Rollouts / episode length | Source |
|---|---|---|---|
| `stages/first.json` | nothing set (fresh) | 300 / 512 | `first.json:4-5` |
| `stages/second.json` | `01_leave_house/Checkpoints/best` | 600 / 1024 | `second.json:4-6` |
| `stages/third.json` | `02_get_starter/Checkpoints/best` | 700 / 1536 | `third.json:4-6` |
| `stages/sixth.json` | `05_mrpokemon/Checkpoints/best` | 1200 / 12288 | `sixth.json:4-6` |
| `stages/freeplay.json` | `06_return_to_elm/Checkpoints/best` | 1000 / 3072 | `freeplay.json:4-6` |
| `stages/open_world.json` | `""` (from scratch) | 163840 / 24576 | `open_world.json:5,13-14` |

`first.json` sets `terminate_on_goal_complete: true` with `n_goals_target: 2`
and two goals (`first.json:7-12`); `open_world.json` explicitly disables that
contract with `n_goals_target: 0` and `"goals": []`
(`open_world.json:26-27`) — a child that wants goals must restate the whole
list, per the shallow-merge rule.

### Resolved chain for `open_world.json`

`default_configs/*.json` concat (`main.py:20-35`) → `curriculum_base.json`
(`open_world.json:2`, merged at `main.py:72-73`) → `open_world.json` → CLI
(`main.py:158`). Verify any resolved value with `--verbose yes`
(`main.py:180-182`).

> **Caveat — broken config.** `configs/stages/freeform_gamearea_event.json:2`
> extends `freeform_gamearea_goexplore_cell.json`, which **does not exist** in
> this checkout. Running it raises `FileNotFoundError` (`main.py:67-70`). The
> file is also untracked (`git status`). Do not reference it in run instructions
> until the parent is committed.

## Output layout

| Key | Default value after load | Source |
|---|---|---|
| `output_base_dir` | `./Training Outputs/` | `outputs_settings.json:2` |
| `checkpoint` | `./Training Outputs/Checkpoints` | `outputs_settings.json:3` + `main.py:38-41` |
| `results_dir` | `./Training Outputs/Results` | `outputs_settings.json:4` |
| `record_path` | `./Training Outputs/Runs` | `outputs_settings.json:5` |
| `export_state_loc` | `./Training Outputs/Training States` | `outputs_settings.json:6` |
| `snapshot_dir` | `./Training Outputs/Snapshots` | `outputs_settings.json:7` |

A stage that overrides `output_base_dir` gets all of the above re-pointed under
its own run directory (`main.py:84-96` at config level,
`main.py:160-168` for a `--output_base_dir` override on the CLI). Two
configs override the base dir per-mode:
`random_walker.json:6`, `debug_eval.json:15`, plus
`explore.json:14-16` which instead points `explore_db_loc` and
`export_state_loc` at `./RandomExplorer/`.

## Mode dispatch

`config["model"]` selects the entry point; unrecognised values raise
`ValueError` (`main.py:184-197`). Default is `"PPO"`
(`configs/default_configs/core_settings.json:2`).

| `model` | Entry point | Source | Minimal CLI |
|---|---|---|---|
| `PPO` | `setup_and_train_PPO` | `main.py:184-185` | `python main.py --use_config configs/stages/open_world.json` |
| `explore` | `memory_collector` | `main.py:186-187` | `python main.py --use_config configs/explore.json` |
| `reward_eval` | `evaluate_reward_system` | `main.py:188-189` | `python main.py --use_config configs/evaluate_reward_system.json` |
| `inference` | `run_inference` | `main.py:190-191` | `python main.py --use_config configs/inference.json` |
| `debug_eval` | `run_debug_inference` | `main.py:192-193` | `python main.py --use_config configs/debug_eval.json` |
| `random_walk` | `random_walk_map_discovery` | `main.py:194-195` | `python main.py --use_config configs/random_walker.json` |
| anything else | `ValueError` | `main.py:196-197` | — |

Import sources for those entry points: `main.py:3-6`.

`debug_eval.json` additionally selects a sub-mode via `debug_mode: "menu_probe"`
and a 64-step episode (`debug_eval.json:3-5`); `explore.json` and
`evaluate_reward_system.json` each declare their own SQLite destination
(`explore.json:14`, `evaluate_reward_system.json:12`).

## Known gaps

- `glob()` ordering of the `default_configs/*.json` concat (`main.py:22`) is
  filesystem-dependent; two default files defining the same key would resolve
  non-deterministically. No such collision was audited here.
- `main.py:155-158` applies CLI overrides *after* the base remap, so a CLI
  `--checkpoint` given alongside `--output_base_dir` is not re-pointed; only
  the second remap loop (`main.py:164-168`) runs, and it uses
  `str.replace` without the base-relative guard of `main.py:89-96`. Untested.
- Whether `ppo_hard_kl`, `ppo_replay_tolerance`, `ppo_lr_hard_floor`
  (`open_world.json:21-24`) and the `goexplore_*` keys (`open_world.json:56-77`)
  are all consumed by the agent was not verified.
- `stages/fourth.json` and `stages/fifth.json` exist but their
  `load_checkpoint`/horizon values were not transcribed above.
