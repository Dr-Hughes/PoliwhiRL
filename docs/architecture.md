# Architecture

What this page is for: the system map of PoliwhiRL — how a frame gets from the
PyBoy emulator to a PPO weight update and back again, which file owns which
responsibility, what the observation contract is, and where a run's artefacts
land on disk. Read this first, then drop into the sibling pages for the
subsystem detail.

**Sources**: `main.py`, `PoliwhiRL/PPO.py`, `PoliwhiRL/environment/gym_env.py`,
`PoliwhiRL/environment/vec_env.py`, `PoliwhiRL/environment/rewards.py`,
`PoliwhiRL/environment/domain_reward_balancer.py`, `PoliwhiRL/checkpoints.py`,
`PoliwhiRL/models/PPO/PPOTransformer.py`,
`PoliwhiRL/models/PPO/ppo_model_implementation.py`,
`PoliwhiRL/agents/PPO/vec_ppo_agent.py`, `PoliwhiRL/agents/PPO/_minibatch.py`,
`PoliwhiRL/replay/vec_ppo_storage.py`, `PoliwhiRL/utils/visuals.py`,
`configs/default_configs/*`, `configs/curriculum_base.json`,
`configs/stages/open_world.json`, `AGENTS.md`.

---

## Data flow

```
PyBoy emulator (Pokémon Crystal ROM)
        │
        ▼
PyBoyEnvironment.step(action)                     gym_env.py
        │   observation = {"image": (C,H,W), "ram": (ram_obs_dim,)}   gym_env.py:1100, :1136
        │   reward + done + terminal_info                              vec_env.py:130, :139-177
        ▼
Vectorised env pool (N worker processes)         vec_env.py:397, :403
        │   stacked obs: {"image": (N,C,H,W), "ram": (N,D)}           vec_env.py:473-474
        ▼
VecPPOAgent.collect rollout                       vec_ppo_agent.py:596, :729
        │  ┌──────────────────────────────────────────────────────────┐
        ├─▶│ GameBoyCNN (image only)      → (B*T, d_model)            │  models.md
        │  │ RAMEncoder  (ram vector)     → (B*T, d_ram)              │
        │  │ cat → fuse Linear(d_model+d_ram, d_model) → (B,L,d_model)│  PPOTransformer.py:265-271
        │  │ Transformer-XL trunk (num_layers blocks, mem_len memory) │
        │  │ fc_actor → action logits (+mask) → softmax                 │
        │  │ fc_critic → V(s)                                          │  PPOTransformer.py:210-211
        │  └──────────────────────────────────────────────────────────┘
        ▼
VecPPOMemory  (T=num_envs×ppo_update_frequency)   vec_ppo_storage.py:20-22, :28-50
        │   partition into S = T // seq_len segments for BPTT        vec_ppo_storage.py:116-137
        ▼
per-env GAE  (budget bootstraps; stagnation does not)  vec_ppo_agent.py:1777-1799
        ▼
run_ppo_epochs  (minibatch, replay gate, two KL regimes)  _minibatch.py:47-182
        ▼
checkpoint  Checkpoints/{actor_critic,optimizer,scheduler,ppo_state,info}.pth  vec_ppo_agent.py:1915-1964
        ▼
eval / triage   tools/training_health_report.py, Checkpoints/info.pth,
                Results/.../training_metrics.json                 visuals.py:1283
```

---

## Components

| Component | Files | Responsibility |
|---|---|---|
| Entry point & config merge | `main.py` | Globs and merges `configs/default_configs/*.json` into the base config (`main.py:20-35`), resolves `extends` chains (`main.py:44-74`), overlays the user config (`main.py:77-98, :111`), dispatches by `config["model"]` (`main.py:184-197`). |
| Training setup | `PoliwhiRL/PPO.py` | Auto-resume (`PPO.py:5-29`), auto-gamma (`PPO.py:32-51`), probes the env for shape/`ram_obs_dim`/action count (`PPO.py:65-71`), constructs the agent (`PPO.py:77`), trains (`PPO.py:82`). |
| Environment | `PoliwhiRL/environment/gym_env.py` | PyBoy stepping, the `{"image","ram"}` observation (`gym_env.py:1100`, returned `:1136`), RAM feature vector built from the curated derived-flag table (`gym_env.py:64`, `:240`, `:299`), Go-Explore snapshot load/save (`gym_env.py:662`, `:731`). |
| Vectorised env pool | `PoliwhiRL/environment/vec_env.py` | Worker processes, `reset()`/`step()` batching (`vec_env.py:397`, `:403`), terminal-info assembly including the GAE bootstrap flag (`vec_env.py:139-177`). |
| Reward engine | `PoliwhiRL/environment/rewards.py` | Per-source raw reward, provenance split, domain grouping, balancing, penalties, truncation decisions. Sum at `rewards.py:764-771`. |
| Domain balancer | `PoliwhiRL/environment/domain_reward_balancer.py` | Per-domain running-mean rescaling and priority weights; four fixed domains (`domain_reward_balancer.py:74`). |
| Checkpoint vocabulary | `PoliwhiRL/checkpoints.py` | The ordered `DERIVED_FLAG_TABLE` of **49** `(flag_num, name)` pairs (`checkpoints.py:11-61`), non-recordable exclusions (`:65-66`), titles and slugs. |
| Visit archive | `PoliwhiRL/environment/visit_archive.py` | Run-wide cell/flag novelty counts, quantised to 2-tile cells (`visit_archive.py:41`, `quantise` `:44-48`). |
| Action mask | `PoliwhiRL/environment/action_mask.py` | Deterministic legal-action mask derived from RAM; consumed at `ppo_model_implementation.py:74`. |
| Network | `PoliwhiRL/models/PPO/PPOTransformer.py`, `PoliwhiRL/models/CNN/GameBoy.py` | Dual encoders → fused Transformer-XL trunk → actor/critic heads (`PPOTransformer.py:153-172`, `:199-211`). |
| PPO losses | `PoliwhiRL/models/PPO/ppo_model_implementation.py` | Geometry from config (`:185-215`), action mask & behaviour policy (`:60-118`), KL (`:146-159`), loss assembly (`:329-430`), optimiser step (`:432-461`). |
| Update loop | `PoliwhiRL/agents/PPO/_minibatch.py` | Shared minibatch/epoch loop, replay-consistency gate, both KL regimes (`_minibatch.py:47-182`). |
| Training loop | `PoliwhiRL/agents/PPO/vec_ppo_agent.py` | Rollout collection, GAE, Go-Explore cycling, checkpointing, metrics. The only agent the entry point constructs (`PPO.py:56`, `:77`). |
| Rollout buffer | `PoliwhiRL/replay/vec_ppo_storage.py` | Fixed-shape rollout tensors and the segment partition (`vec_ppo_storage.py:28-50`, `:116-209`). |
| Running stats | `PoliwhiRL/utils/running_stats.py` | Welford `RunningMeanStd` and `RewardScaler`. |
| Metrics & plots | `PoliwhiRL/utils/visuals.py` | Writes `training_metrics.json` (`visuals.py:1283`) and the metrics PNG (`:954`). |
| Eval / exploration / reward debug | `PoliwhiRL/evaluator/`, `PoliwhiRL/explorer/`, `PoliwhiRL/reward_evaluation/` | Inference, manual/random data collection, standalone reward-system debugging; dispatched from `main.py:186-195`. |
| Post-hoc triage | `tools/` | `training_health_report.py`, `world_map.py`, `png_to_video.py`, `llm_game_debug.py`, `rom_maps/` (verified present in the working tree). |

Dispatch modes accepted by `config["model"]`: `PPO` (`main.py:184-185`),
`explore` (`:186-187`), `reward_eval` (`:188-189`), `inference` (`:190-191`),
`debug_eval` (`:192-193`), `random_walk` (`:194-195`); anything else raises
(`main.py:197`).

---

## The observation contract

The environment hands back a two-key dict and nothing else:

```python
return {"image": image, "ram": ram}      # gym_env.py:1136 (docstring :1100)
```

| Key | Shape (single env) | Shape after vec stacking | Producer |
|---|---|---|---|
| `image` | `(C, H, W)` | `(N, C, H, W)` | `gym_env.py:1136`; stacked `vec_env.py:473` |
| `ram` | `(ram_obs_dim,)` | `(N, D)` | `gym_env.py:1136`; stacked `vec_env.py:474` |

`ram_obs_dim` is discovered once by probing a short-lived env and injected into
the config so the model, buffers and agent never re-query the env
(`PPO.py:68`, `:75`).

Two things the contract deliberately does **not** do:

- **No raw story-flag bytes.** The original 256-byte flag blob was replaced by
  the curated `_DERIVED_FLAG_TABLE` (`gym_env.py:33`; the table itself
  `checkpoints.py:11-61`). Note `gym_env.py:33` describes that as "~70 bits"
  — that is the *observation* bit count, not the table length; the table has
  **49** `(flag_num, name)` pairs (counted from the literal, `checkpoints.py:12-60`).
- **No absolute positional encoding.** Recency comes from learned age
  embeddings; coarser history arrives explicitly through RAM
  (`PPOTransformer.py:162-171`). See [subsystems/models.md](subsystems/models.md).

Terminal information is a separate channel, assembled in the worker before the
auto-reset wipes goal state (`vec_env.py:132-177`). The single most
consequential field is `truncated`, which is set **only** for budget truncation
(`vec_env.py:136-138`, written `:176`) — see the bootstrap rule in
[subsystems/training.md](subsystems/training.md).

---

## Four reward domains at a glance

Grouping happens at the reward call site, not inside the balancer
(`rewards.py:741-752`).

| Domain | Covers | Detail page |
|---|---|---|
| `frontier` | map discovery, per-cell novelty, menu/screen novelty | [subsystems/rewards.md](subsystems/rewards.md) |
| `battle` | engagement and wins, capped per episode | ibid. |
| `milestone` | flags, checkpoints, map goals, map visits, key items, levels, events, first trainer battles (`rewards.py:743-751`) | ibid. |
| `catching` | pokédex sightings and ownership | ibid. |

`whiteout` and `step_penalty` are added **outside** domain scaling as direct
penalties — the final sum is
`frontier + battle + milestone + catching + whiteout + step_penalty`
(`rewards.py:764-771`).

---

## The single most important reading rule: split honest vs seeded

> **Judge nothing on aggregate reward. Split honest (unseeded) from seeded
> episodes first.**

A *seeded* episode is one the agent started from a Go-Explore snapshot, deep in
already-solved territory; an *honest* episode starts from the true game beginning.
The provenance is set by the agent per reset, not by the env
(`vec_ppo_agent.py:1089-1091`, `:1098-1100`), recorded per episode as
`episode_seeded` (`vec_ppo_agent.py:333`), and split at the source in the
reward breakdown — `checkpoint_key = "checkpoint_seeded" if self.episode_seeded
else "checkpoint_honest"` (`rewards.py:717`, keys `:461-462`).

Why it matters, in three facts:

1. Seeded episodes carry inflated reward and coverage, so the headline
   `mean_reward` conflates two different populations (`AGENTS.md`, "the one rule
   that matters most").
2. Honest-only aggregates are pre-computed — `honest_mean(key)`
   (`vec_ppo_agent.py:1833`) published as `h_r` / `h_maps` / `h_cells` /
   `h_flags` (`vec_ppo_agent.py:1842-1851`).
3. Best-model selection in free-play uses **only** first-honest durable
   checkpoint recordings — explicitly never raw reward or unique-map counts
   (`vec_ppo_agent.py:2011-2019`), and `best_discoveries` tracks the honest
   `checkpoint_recordings` count alone (`vec_ppo_agent.py:2014-2016`).

The failure this guards against is documented in the source: the 2026-07-20 run
hid a regression from 16 down to 5.6 honest unique_maps behind a rising headline
reward (`rewards.py:951-954`, echoed `AGENTS.md`). Full treatment on
[subsystems/rewards.md](subsystems/rewards.md).

---

## Run artefact layout

Every path is relative to `output_base_dir`, default `./Training Outputs/`
(`configs/default_configs/outputs_settings.json:2`). The five output keys are
prefixed with that base at load time by `modify_outputs_settings`
(`main.py:29-30`, `:38-41`).

| Path | Config key | Contents |
|---|---|---|
| `<base>Checkpoints/` | `checkpoint` (`outputs_settings.json:3`) | `actor_critic.pth`, `optimizer.pth`, `scheduler.pth`, `ppo_state.pth` (`ppo_model_implementation.py:556-560`), `info.pth` (`vec_ppo_agent.py:1933-1958`) |
| `<base>Checkpoints/best/` | derived | Duplicate copy written when `_should_update_best()` (`vec_ppo_agent.py:1960-1964`) |
| `<base>Results/` | `results_dir` (`outputs_settings.json:4`) | Metrics; `training_metrics.json` written at `visuals.py:1283`, plots at `:954` |
| `<base>Runs/` | `record_path` (`outputs_settings.json:5`) | Recorded episodes (PNG frames; consumed by `tools/world_map.py`, `tools/png_to_video.py`) |
| `<base>Training States/` | `export_state_loc` (`outputs_settings.json:6`) | Emulator save states |
| `<base>Snapshots/` | `snapshot_dir` (`outputs_settings.json:7`) | Go-Explore frontier snapshots; the agent falls back to a `goexplore_snapshot_dir` under this (`vec_ppo_agent.py:497-502`) |

Save cadence: every `checkpoint_frequency` rollouts, gated on `save_checkpoint`
and a non-null `checkpoint` (`vec_ppo_agent.py:662-666`); `curriculum_base.json:33`
sets `50`. Auto-resume picks up this main checkpoint — deliberately the *latest*,
not `Checkpoints/best/` — when `load_checkpoint` is empty and both
`actor_critic.pth` and `info.pth` exist (`PPO.py:20-29`).

For triage, `tools/training_health_report.py` reads `info.pth`; drop to the
metrics JSON when you need finer buckets (`AGENTS.md`, "Inspecting a run").

---

## Where to go next

| Page | Covers |
|---|---|
| [subsystems/models.md](subsystems/models.md) | Network geometry, encoders, Transformer-XL memory, action head, behaviour policy, KL |
| [subsystems/training.md](subsystems/training.md) | Rollout collection, GAE and the bootstrap asymmetry, entropy control, PPO epochs, reward scaling, checkpointing |
| [subsystems/environment.md](subsystems/environment.md) | PyBoy stepping, RAM feature construction, action masking |
| [subsystems/rewards.md](subsystems/rewards.md) | Every reward source, anti-farming guards, truncation causes, domain balancing |
| [subsystems/exploration.md](subsystems/exploration.md) | Visit archive, Go-Explore capture/seeding, probe ladders |
| [configuration.md](configuration.md) | Config inheritance and every key |
| [getting-started.md](getting-started.md) | Install, ROM setup, your first run |
| [operations/](operations/) | Running, monitoring, and recovering a training run |
| [reference/](reference/) | Flag tables, RAM map, metric glossary |
| [roadmap.md](roadmap.md) | Where the project is heading |
| [branch-changes.md](branch-changes.md) | What this branch changed versus `main` |

---

## Known gaps

- **Metrics path (traced, agreed with `AGENTS.md`).** The per-episode record is
  `Results/metrics/training_metrics.json` under the run's `output_base_dir`, exactly
  as `AGENTS.md:68` states. `PoliwhiRL/utils/visuals.py` defaults `save_loc` to
  `"Results"` (`:126`) — the `results_dir` the caller passes (`outputs_settings.json:4`)
  — creates `metrics_dir = os.path.join(save_loc, "metrics")` (`:969`) and writes the
  JSON inside it (`:1282-1283`); the PNGs land one level up (`:954`). Where prose and
  code disagree, the code wins (`AGENTS.md:17`).
- **CNN internals.** Exact kernel sizes, strides, pooling and the
  `_make_norm` choice in `PoliwhiRL/models/CNN/GameBoy.py` were not read, so
  the concrete `flat_features` width for a given `input_shape` is unverified
  (carried from the earlier reconnaissance notes — untracked working-tree notes
  under `.doc-scratch/`, never committed, therefore not citable from a fresh clone).
- **`~70 bits`.** The parenthetical at `gym_env.py:33` was not independently
  counted; only the 49-pair table length was.
- **Terminal-info field list.** Only the fields quoted above
  (`vec_env.py:139-177`) were read; the remainder of that dict and the
  worker↔agent protocol beyond `vec_env.py:397-474` were not enumerated.
- **No runtime execution.** Every shape here comes from source and tests; no
  training run or emulator was launched while writing this page.
