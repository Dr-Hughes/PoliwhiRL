# PoliwhiRL - AGENTS.md

## Overview

PPO training for Pokémon Crystal (Game Boy Color) via the PyBoy emulator. The policy sees both the **screen image** and a **normalised RAM vector** (position, party state, current goal target, exploration summary) and chooses one of nine discrete button presses per step. Rewards come from configurable sequential location goals plus exploration and step penalties. Both single-environment and vectorised multi-process training are supported, with a per-worker save-state pool for curriculum mixing.

**Architecture:**
```
Image (C, H, W) → CNN ──┐
                        ├─ concat → fuse → Transformer-XL → Actor + Critic
RAM   (ram_dim,) → MLP ─┘
```

---

## File Layout

```
main.py                                  # Entry point: parse args, merge configs, dispatch by model type

PoliwhiRL/
├── PPO.py                               # Training entry: env shape probe (image + RAM), dispatch between PPOAgent / VecPPOAgent
├── agents/PPO/
│   ├── __init__.py                      # exports: PPOAgent, VecPPOAgent
│   ├── ppo_agent.py                     # Single-env agent: per-episode rollout, KL early-stop, value clip, best-so-far
│   ├── vec_ppo_agent.py                 # Vec agent: T*N rollout, per-env GAE, save-state cycling, per-state metric tagging
│   └── _minibatch.py                    # Shared mini-batch iterator for both agents
├── environment/
│   ├── __init__.py                      # exports: PyBoyEnvironment, VecPyBoyEnv
│   ├── gym_env.py                       # PyBoy env, dict obs, RAM_FEATURE_KEYS, _build_ram_vector, set_state_path
│   ├── vec_env.py                       # Spawn-context multiprocessing wrapper; state_paths round-robin assignment + cycling
│   ├── rewards.py                       # Reward calculator: goal sequencing, pokedex, step/button penalties, exploration
│   └── RAM.py                           # RAM address book + get_variables()
├── models/
│   ├── CNN/GameBoy.py                   # GameBoyBlock (Conv-GroupNorm-ReLU), GameBoyOptimizedCNN
│   ├── PPO/
│   │   ├── __init__.py                  # exports: PPOModel
│   │   ├── ppo_model_implementation.py  # PPO losses (KL early-stop, value clip), GAE, CosineAnnealing LR, Adam eps
│   │   └── PPOTransformer.py            # GameBoyCNN + RAMEncoder + fusion + TransformerXLBlock × N + actor/critic
│   └── transformers/positional_encoding.py
├── replay/
│   ├── __init__.py                      # exports: PPOMemory, VecPPOMemory
│   ├── ppo_storage.py                   # Single-env rollout buffer (images + RAM)
│   └── vec_ppo_storage.py               # (T, N, ...) rollout buffer with sliding-window emission
├── explorer/                            # Manual/random data collection (separate from PPO)
├── reward_evaluator/                    # Predefined-action evaluation tool
└── utils/
    ├── running_stats.py                 # RunningMeanStd + RewardScaler for return normalisation
    └── visuals.py                       # plot_metrics (with per-state summaries), record_step

configs/
├── curriculum_base.json                 # Shared defaults for stage configs (used via "extends")
├── default_configs/                     # Globally merged defaults — every key here has a default
│   ├── core_settings.json               # model, device, vision, checkpointing
│   ├── episode_settings.json            # episode_length, num_rollouts, num_envs, record_frequency
│   ├── ppo_settings.json                # All PPO hyperparameters
│   ├── reward_settings.json             # Default goal layout and reward magnitudes
│   ├── outputs_settings.json            # Output paths
│   └── rom_settings.json                # ROM and start-state paths
├── stages/{first,second,third}.json     # Curriculum stages, extending curriculum_base.json
├── fourth_steps.json                    # Stage 4: multi-state pool + softer step penalty
├── explore.json                         # Manual/random data collection mode
└── evaluate_reward_system.json          # Reward-evaluation mode

tests/
├── test_PPO.py                          # Model init, losses, GAE/returns, bootstrap, KL
├── test_extends_config.py               # "extends" inheritance resolver
├── test_ppo_memory.py                   # Single-env buffer (image + RAM)
├── test_vec_ppo_memory.py               # Vec buffer shape and alignment
├── test_vec_env.py                      # Live N=2 vec env + one-rollout smoke + state-pool init
├── test_action_replay.py                # Story flags, replay_actions, vec replay pool, .steps loader
├── test_pyboyenv.py                     # Env reset/step/save/load + dict obs shape
├── test_pyboy_ram.py                    # RAM extraction
├── test_reward_system.py                # Reward calc, penalties, goals, target vector
├── test_running_stats.py                # RunningMeanStd + RewardScaler unit tests
└── test_minibatch.py                    # PPO minibatch iterator with stub model
```

---

## Three Modes

Dispatched in `main.py` based on `config["model"]`:

| Mode | Value | Purpose |
|------|-------|---------|
| Train | `"PPO"` | PPO training (single-env or vec, decided by `num_envs`) |
| Explore | `"explore"` | Manual or random data collection to SQLite DB |
| Evaluate | `"evaluate"` | Run predefined action sequences, evaluate reward function |

---

## Training Modes

`setup_and_train_PPO` probes the env for image shape, RAM dim, and action size; then picks the agent:

| `num_envs` | Agent | Iteration unit | Data per update |
|---|---|---|---|
| `1` | `PPOAgent` | One full episode (reset → done) | T transitions, updated every T steps and at episode end |
| `>1` | `VecPPOAgent` | One rollout (T steps × N envs) | (T × N) transitions in one batched update |

Both modes share the same `PPOModel`, same PPO loss path, same minibatch iterator, and same save/load contract. They differ only in how experience is collected and how `num_rollouts` is interpreted.

### How vec mode collects

1. **Setup.** `VecPyBoyEnv(config, num_envs=N)` spawns N subprocesses with `multiprocessing.get_context("spawn")`. Each worker creates its own `PyBoyEnvironment`, which copies the ROM and sidecar files into a fresh per-instance `tempfile.TemporaryDirectory`. Zero file contention between workers. Each worker is assigned a save-state from the `state_paths` pool via round-robin (see "Save-state pool" below).
2. **Rollout.** For `num_rollouts` iterations, the agent steps all N envs `T = ppo_update_frequency` times. Each timestep is one batched forward pass returning `(N, action_size)` action probs and `(N,)` values. Workers auto-reset on `done`; the agent zeros that env's transformer mems and refills its state-sequence/RAM-sequence buffers with the new obs.
3. **Update.** Per env, GAE is computed along the time axis with V(s_{T+1}) bootstrap (advantages **never cross env boundaries**). Returns/advantages/old_values get precomputed once, then the (W × N) flat batch is fed through the minibatch iterator for `ppo_epochs` shuffled passes with KL early-stop and value clipping.
4. **Bookkeeping.** Cosine LR scheduler steps once per rollout. Periodic recording triggers on env 0 every `record_frequency` completed episodes. Each committed episode metric is tagged with the state-pool index that env was running.

---

## Observation

Both modes use a **dict observation** with two keys:

```
obs = {
    "image": ndarray (C, H, W) uint8,    # screen pixels (or 18×20 tilemap when vision=false)
    "ram":   ndarray (ram_dim,) float32, # normalised RAM vector
}
```

### RAM feature vector

The order is defined in `RAM_FEATURE_KEYS` (in `environment/gym_env.py`) and built by `_build_ram_vector`. **Treat the order as a contract** — changing it invalidates trained models. To extend, append new keys at the end. The model reads `ram_dim` from config at startup; nothing else changes.

Current vector width: **`RAM_OBS_DIM = 277`** (21 base + 256 story-flag bytes).

| index | feature | source | scaling |
|---|---|---|---|
| 0–1 | x, y | RAM | / 255 |
| 2–5 | map_num, map_bank, room, warp_number | RAM | / 255 |
| 6–8 | party_level, party_hp, party_exp | RAM | / 100, / 1000, log1p / 20 |
| 9 | money | RAM | / 1e6 |
| 10–11 | pokedex_seen, pokedex_owned | RAM | / 251 |
| 12–15 | collision (down/up/left/right) | RAM | / 255 |
| 16–18 | target_x, target_y, target_map | Rewards.get_current_target_vector() | / 255 |
| 19 | has_active_target | Rewards | 0 or 1 |
| 20 | explored_tile_count | len(Rewards.explored_tiles) | log1p / 6 |
| 21–276 | story_flag_byte_000..255 | RAM 0xDA72–0xDB71 | / 255 |

**Goal-conditioning (indices 16–19)** is the lever that solves narrative backtracking: when the curriculum advances to a new goal whose coordinates differ from the last one, the policy's input changes accordingly, so the same visual state can map to a different action depending on what the agent is currently targeting.

**Story flags (indices 21–276)** are the 256-byte region at `0xDA72–0xDB71` — each byte holds 8 individual story flags as a bitfield (2048 flags total). We expose the bytes raw (normalised /255) rather than expanding to 2048 bits; the policy learns the relevant byte-value patterns. These give the model awareness of game-progress milestones (received starter, beat Falkner, etc.) that the screen image alone doesn't convey.

> **Read-only.** We never write to this region. The 8-byte GameShark-style write quirk is irrelevant to us.

---

## PPO Algorithm Details

### Loss
`loss = actor_loss + critic_loss + entropy_loss`

- **Actor (clipped surrogate):** `-E[min(r·A, clip(r, 1-ε, 1+ε)·A)]`, where `r = exp(new_log_prob - old_log_prob)`.
- **Critic (value-clipped MSE):** `0.5 * E[max((V_new − R)², (V_clip − R)²)]`, where `V_clip = V_old + clamp(V_new − V_old, -ε, +ε)`. Disable with `ppo_clip_value_loss: false`.
- **Entropy bonus:** `-coef * E[H(π)]`, with `coef` decaying per **stage-relative** episode from `ppo_entropy_coef` to `ppo_entropy_coef_min`. The schedule resets on each `load_checkpoint`.

### Returns and Advantages (GAE)
GAE-λ with **V(s_{T+1}) bootstrap** for truncated rollouts. When the rollout boundary isn't a terminal, V(s_{T+1}) is computed via an extra forward pass on `last_next_state` + `last_next_ram` rather than being treated as zero. This was a real bug pre-fix and disproportionately affected long-episode runs.

### Reward normalisation
Both agents own a `RewardScaler` (`utils/running_stats.py`) that tracks the running mean/variance of **discounted returns per env**:

```python
running_return = γ * running_return + reward    # per env
reward_rms.update(running_return)               # streaming Welford
# on done: running_return[env] = 0
```

Before GAE, rewards are scaled by `1 / sqrt(var(returns))`. The critic naturally settles into predicting normalised returns, keeping its target range stable across curriculum stages with very different reward magnitudes. The scaler state is persisted in `info.pth` and restored on resume.

### KL early-stop
After each PPO epoch, the agent computes the **k3 approximate KL** (`E[(r − 1) − log r]`, always non-negative, lower-variance than the naïve `E[old − new]`). If the per-epoch mean exceeds `ppo_target_kl` (default `0.01`), the remaining epochs are skipped. `null` disables the safeguard.

### Mini-batching
The shared iterator in `agents/PPO/_minibatch.py` shuffles the (W × N) flat batch into minibatches of size `ppo_minibatch_size` (default 128), runs `ppo_epochs` passes, and stops a multi-epoch update early when the per-epoch mean KL exceeds the target. Returns/advantages/old_values are **precomputed before** the epoch loop so shuffling doesn't break time-ordered GAE bootstrap.

### Learning-rate schedule
`CosineAnnealingLR(T_max=num_rollouts, eta_min=ppo_lr_min)`, stepped once per outer iteration. Replaced the original `CyclicLR(triangular2)` whose late LR peaks coincided with policy convergence and were implicated in the stage-1 collapse pattern.

### Initialisation
- Orthogonal init on linear and conv layers (gain = √2).
- Actor head: gain = 0.01 → near-uniform initial action probabilities (deliberate exploration at the start).
- Critic head: gain = 1.0 → values start near zero.
- Adam `eps=1e-5` (CleanRL/OpenAI canonical). Stops the optimiser over-amplifying small gradients.

### Best-so-far checkpoint
`save_model` overwrites the latest snapshot every `checkpoint_frequency`, **and** writes a separate `Checkpoints/best/` snapshot whenever the 100-episode rolling-mean reward beats the running best (gated on a full 100-ep window). Point a downstream stage's `load_checkpoint` at `.../Checkpoints/best` to inherit the peak rather than the latest weights.

---

## Model Architecture

### `PPOTransformer`

```
x_image (B, T, C, H, W) ─→ GameBoyCNN ────────→ (B*T, d_model=128)   ┐
                                                                     ├─ concat → Linear(160, 128) → (B*T, 128)
x_ram   (B, T, ram_dim) ─→ RAMEncoder (MLP) ──→ (B*T, d_ram=32)      ┘
                                                                              ↓
                                            PositionalEncoding (sinusoidal, max_len=1000)
                                                                              ↓
                                            N × TransformerXLBlock (MHA d_model=128, heads=8, FFN×4 GELU)
                                                                              ↓
                                            last token (B, d_model)
                                                                              ↓
                                                          ┌─ fc_actor  → softmax → (B, action_size)
                                                          └─ fc_critic → (B, 1)
```

Transformer-XL memory is **caller-managed**: `init_mems(batch_size, device)` returns a per-layer list of `(B, mem_len=16, d_model)` zero tensors. Each forward pass receives mems and returns updated mems (detached). The agent stores the **input** mems at each transition so PPO replay reproduces the exact attention context.

### Normalisation choice
`GameBoyBlock` uses `GroupNorm` (groups = min(8, channels) with auto-fallback for non-divisible channel counts) instead of BatchNorm. BatchNorm is a known PPO pitfall: rollout-time batches (size N) differ from update-time batches (size W×N), and running stats accumulate across mismatched distributions, silently biasing the policy. GroupNorm is batch-size-independent.

### Mems-on-done invariant
When `done=True` for an env, that env's mems get zeroed before the next step. Single-env: a fresh `init_mems(1)` per episode. Vec: in-place `.zero_()` of the env's slice in the shared mems tensor. Same for the per-env `state_sequence` and `ram_sequence` deques/arrays — refilled with the post-reset observation.

---

## Configuration

### Loading

Defaults in `configs/default_configs/*.json` auto-load and merge on startup (`main.py:load_default_config`). A user config (`--use_config path`) overrides those defaults. Individual CLI flags override the user config.

The user config supports an **`"extends": "../path.json"`** key (resolved relative to the file containing it), enabling shared bases. Chained extends are supported. The `extends` key is stripped after merging. See `tests/test_extends_config.py` for the spec.

### Parameter reference

| Parameter | Default | What it does | When to change |
|---|---|---|---|
| `num_envs` | `4` | Parallel envs. `>1` uses `VecPPOAgent`. | Scale with cores; common values 4–32. |
| `num_rollouts` | `12` | **Training budget.** Single-env: outer-loop iterations ≈ episodes run. Vec: number of (T × N) collect+update cycles. | Total env steps = `num_rollouts × ppo_update_frequency × num_envs`. |
| `episode_length` | `50` | Per-episode step cap before forced done. | Tune to goal depth. |
| `sequence_length` | `8` | Transformer input sequence length per forward pass. | Rarely. |
| `record_frequency` | `100` | Save image dumps every N completed episodes (vec: env 0 only). | Bigger = less disk, less visibility. |
| `state_paths` | `[]` | Save-state pool (list of paths). If empty, falls back to single `state_path`. | Add entries to mix in mid-game starts. |
| `state_cycle_strategy` | `"round_robin"` | How envs pick their next save-state on auto-reset. `"round_robin"` / `"random"` / `"none"`. | `"none"` to pin each env to its initial state. |
| `action_replay_paths` | `[]` | Action-replay pool (list of `.steps` paths). Each env replays its assigned sequence after every reset, walking Rewards forward to a curriculum-aligned start. | Set for stage-to-stage continuation. |
| `action_replay_cycle_strategy` | `"round_robin"` | How envs pick their next replay on auto-reset. Same strategies as state cycling. | `"none"` to pin each env to its initial replay. |
| `ppo_update_frequency` | `128` | Transitions per env per PPO update (rollout length T). Larger → bigger batch, fewer mid-episode truncations. | `128`–`256`. With long episodes avoid tiny values. |
| `ppo_epochs` | `3` | PPO update passes per batch. KL early-stop usually bounds this. | `3`–`8`. |
| `ppo_minibatch_size` | `128` | Shuffled minibatch size. `null` or `0` disables minibatching. | Scale with effective batch (W × N). |
| `ppo_learning_rate` | `3e-4` | Peak LR for the cosine schedule. | Rarely. |
| `ppo_lr_min` | `1e-5` | Cosine schedule floor. | Don't go below this. |
| `ppo_target_kl` | `0.01` | KL ceiling per epoch. | `null` to disable. `0.01`–`0.02` typical. |
| `ppo_clip_value_loss` | `true` | Critic-side clip mirroring the actor clip. | `false` for unclipped MSE. |
| `ppo_adam_eps` | `1e-5` | Adam epsilon. | Standard PPO setting. |
| `ppo_epsilon` | `0.2` | PPO clipping range for the policy ratio. | Rarely. |
| `ppo_gamma` | `0.98` | Discount factor. | Higher for long credit. |
| `ppo_gae_lambda` | `0.95` | GAE bias-variance knob. `0` disables GAE. | Rarely. |
| `ppo_value_loss_coef` | `1.0` | Critic loss weight. | Lower if critic dominates. |
| `ppo_entropy_coef` | `0.01` | Initial entropy bonus weight. | Higher (`0.05`) for exploration-heavy stages. |
| `ppo_entropy_coef_decay` | `0.99` | Per-(stage-relative)-episode multiplier. | Slower (`0.999`) for long stages. |
| `ppo_entropy_coef_min` | `0.001` | Entropy bonus floor. | Match to task: too high prevents commitment. |
| `ppo_max_grad_norm` | `0.5` | Gradient clipping. | Standard. |
| `reset_lr_scheduler_on_load` | `true` | Reinit scheduler on `load_checkpoint`. | Keep `true` for stage transitions. |
| `reset_optimizer_on_load` | `false` | Reinit Adam on `load_checkpoint`. | `true` for save-state jumps (stale Adam moments on distribution shift). |
| `goal_reward` | `100` | Per location-goal reward. | |
| `sequence_bonus` | `50` | Added when `require_sequential` and goal hit in order. | |
| `checkpoint_bonus` | `200` | Added at `checkpoint_goals` milestones. | |
| `all_goals_bonus` | `500` | Added on hitting `N_goals_target`. | Set to `0` for early_completion_bonus-only terminals. |
| `early_completion_bonus` | `0` | Added on final goal. | |
| `exploration_reward` | `0.0` | Per first-visit `(x, y, map)` tile. | Small values (`1`–`3`) for sparse-goal stages. |
| `step_penalty` | `-1` | Per-step penalty when `punish_steps`. | Scale to episode_length. Per-episode worst case `step_penalty × episode_length` should stay smaller than a goal reward. |
| `button_penalty` | `-5` (fixed) | Penalty for start/select presses. | Not configurable. |

---

## Reward System

`PoliwhiRL/environment/rewards.py`. Each step's reward is the clipped sum of:

1. **Goal achievement** (sequential by default): `goal_reward + sequence_bonus + (checkpoint_bonus if in checkpoint_goals) + (all_goals_bonus + early_completion_bonus if final)`. Episode terminates if `break_on_goal` and all goals hit.
2. **Pokedex updates** (seen / owned).
3. **Exploration**: `exploration_reward` for each first-visit `(x, y, map_num)` tile this episode.
4. **Step penalty** (if `punish_steps`).
5. **Button penalty** for `start`/`select`.

All summed, clipped to `[-1000, 1000]`, returned as `float32`.

**Goals** are configured as `location_goals: [[ [x, y, map, room], ... ], ... ]` — each outer entry is one ordered goal, each inner list is a set of acceptable coordinates for that goal. Matching uses `[x, y, map_num]`. `Rewards.get_current_target_vector()` exposes the active goal's primary `[x, y, map]` and a `has_active_target` flag — used by the env to populate indices 16–19 of the RAM observation.

---

## Save-state Pool

`VecPyBoyEnv` accepts a list of save-state files via `config["state_paths"]`. Workers are assigned round-robin at init (`worker i` → `state_paths[i % len(pool)]`). Each worker's env boots with its assigned state.

**Per-episode cycling** is driven by the agent (vec mode only). After an env completes an episode:

1. Record the metric tagged with the state that was just used.
2. Pick the next state for that env according to `state_cycle_strategy`: `round_robin` advances cyclically; `random` uniform-samples from the pool; `none` keeps the assignment fixed.
3. Send the new state to the worker via `set_env_state_index`. The worker queues it and applies it on its **next auto-reset** — so the change takes effect one episode later than the cycling decision. The agent tracks this with `env_state_indices` (currently running) and `env_pending_state_indices` (queued).

### Per-state metrics

`episode_data["episode_state_indices"]` is a list parallel to `episode_rewards`, recording the state-pool index each completed episode actually used. The training-metrics JSON output includes this list **and** a `per_state_summary` block with mean / max / min reward and episode count per state — for offline analysis grouped by starting state.

### Goal-list alignment caveat

When you add a save-state whose player position is on an existing goal tile, the agent gets a free reward on step 0 (the reward calculator starts with `current_goal_index=0` every reset). This is the main reason **action replay** (next section) is the preferred mechanism for cross-stage continuation — replay walks the Rewards object forward so the curriculum starts at the right point automatically.

---

## Action Replay (the preferred chaining mechanism)

Save-states snapshot emulator memory but leave the `Rewards` object's curriculum tracking (`current_goal_index`, `N_goals`, `pokedex_*`) reset to 0 on every `env.reset()`. The cleanest fix is to **replay the action sequence** that produced the curriculum endpoint — `Rewards` walks forward through the goals as the actions execute, so the post-replay state is *natively correct* without any goal-list curation.

### File format

A `.steps` file is plain text, one integer action per line. Comments (`#`-prefixed) and blank lines are tolerated:

```
# best_episode_reward = 947.5
4
4
2
1
...
```

Loaded by `_load_actions_file(path)` in `environment/vec_env.py`. Missing files emit a warning and become a no-op (replay disabled for that pool entry).

### Replay invariant

After every `env.reset()` (explicit or auto-reset on done), if a replay sequence is attached the worker calls `env.replay_actions(actions)`:

1. Steps the env through each action, calling `_handle_action` + `_calculate_fitness`. `Rewards.current_goal_index`, `N_goals`, `pokedex_*`, `explored_tiles` all advance naturally.
2. After the sequence, calls `Rewards.start_new_episode()` to clear *per-episode counters* (`steps`, `cumulative_reward`, `done`, `last_action`) while **preserving both curriculum state** (`current_goal_index`, `N_goals`, `pokedex_seen/owned`) **and exploration state** (`explored_tiles`). The exploration set is deliberately kept so re-walking replay-visited tiles doesn't pay a fresh `exploration_reward` — that would defeat the whole point of replay.
3. Resets `env.steps = 0` and `env._fitness = 0` so the training episode starts with a clean step budget.

The training agent never sees the replay transitions — they're not stored in the rollout buffer and don't count toward `ep_returns` / `ep_lengths`. The post-replay observation is treated as step 0 of the training episode. The RAM-vector `explored_tile_count` (feature 20) carries the replay's exploration size into the agent's input, so the policy sees a meaningful "novelty so far" signal from step 0 rather than a fake reset.

### Replay pool

Mirrors the save-state pool exactly:

- `config["action_replay_paths"]`: list of `.steps` paths. Round-robin assigned to workers at init.
- `config["action_replay_cycle_strategy"]`: `"round_robin"` (default), `"random"`, or `"none"`.
- Cycling uses the same lag-by-one pattern as state cycling: `env_replay_indices[i]` tracks the currently-running replay; `env_pending_replay_indices[i]` is queued for the next auto-reset.

State pool and replay pool are independent — a worker uses `state_paths[state_index]` as its base PyBoy load, then walks forward through `action_replay_paths[replay_index]`. Both pools can mix sizes/lengths.

### Best-actions dump

Both agents track `best_episode_reward` (the highest single-episode reward seen across the run) and the action history of every in-flight episode (`current_episode_actions` in single-env, `_env_action_histories[i]` in vec). When a new single-episode reward record is set, the agent writes that episode's action sequence to `<output>/Checkpoints/actions.steps`. The next curriculum stage's `action_replay_paths` references this file:

```json
"load_checkpoint": "./Training Outputs/third_steps/Checkpoints/best",
"action_replay_paths": [
    "./Training Outputs/third_steps/Checkpoints/actions.steps"
]
```

The replay file is overwritten each time a new best is set. If you want to preserve historical best paths, copy them out manually.

### Per-replay metrics

Like per-state tagging: `episode_data["episode_replay_indices"]` is parallel to `episode_rewards`, recording the replay-pool index each committed episode used (`-1` when no replay was active). The training-metrics JSON includes this list. Combined with `episode_state_indices`, you can slice performance by (state, replay) pair for offline analysis.

### Trade-offs

| | save-state pool | action replay |
|---|---|---|
| Rewards state at episode start | reset to 0 (must curate goal lists) | naturally advanced (no curation needed) |
| Setup cost per episode | microseconds | seconds (replays through actions) |
| File size | KB (binary) | bytes (text, diffable) |
| Inspectable / editable | no | yes |
| Risk of stale state mismatch | high | zero |

For curriculum chaining, action replay wins. The save-state pool stays useful for *base* loads (which save-state to start PyBoy from) — replay walks forward from there.

### Cost note

A 300-step replay at `frames_per_action=90` is ~27,000 PyBoy frames per episode start. PyBoy headless is fast (~µs per frame), but multiply by every episode of every env and you pay a real % overhead. For deep curricula (stage 10+) the answer is hybrid: hand-record a save-state to skip the boring early game, then replay only the last N actions to ensure Rewards state is correct.

---

## Curriculum Pattern

`configs/curriculum_base.json` holds shared defaults; stage configs extend it via `"extends": "../curriculum_base.json"` and override only what changes:

```
configs/curriculum_base.json
configs/stages/first.json    →  "extends": "../curriculum_base.json"  (2 goals,  40 steps, single state)
configs/stages/second.json   →  extends + checkpoint load from first
configs/stages/third.json    →  extends + checkpoint load from second
configs/fourth_steps.json    →  extends curriculum_base, multi-state pool, softer step_penalty
```

Each stage sets its own `load_checkpoint` to the previous stage's `Checkpoints/best/` (peak) or `Checkpoints/` (latest). On load:

- Encoder + transformer weights persist.
- Entropy schedule resets to its initial value (`stage_start_episode = self.episode`).
- Optimizer state persists unless `reset_optimizer_on_load: true`.
- LR scheduler reinitialises when `reset_lr_scheduler_on_load: true` (default).
- `RewardScaler` running stats are restored from `info.pth`.

---

## Outputs

```
<output_base_dir>/
  Checkpoints/
    actor_critic.pth                     # latest weights
    optimizer.pth, scheduler.pth, info.pth
    best/                                # snapshot at best 100-ep rolling reward
      (same files)
  Results/
    training_metrics.png                 # cumulative-mean curves
    training_metrics_current.png         # current-stage-only view (only on resume)
    metrics/
      training_metrics.json              # raw arrays + summary + per_state_summary
      training_metrics_current.json      # current-stage variant
  Runs/                                  # per-step PNG dumps when recording is enabled
    N_goals_<n>/ep_<E>/step_<S>_btn_<B>_reward_<R>.png
```

The `info.pth` contains: `episode`, `best_reward`, `episode_data` (including `episode_state_indices`), and `reward_scaler` state.

---

## Running

```bash
# Defaults
python main.py

# Stage configs (using the extends mechanism)
python main.py --use_config configs/stages/first.json
python main.py --use_config configs/stages/second.json
python main.py --use_config configs/stages/third.json

# Stage 4 with multi-state pool
python main.py --use_config configs/fourth_steps.json

# Override anything via CLI
python main.py --num_envs 16 --num_rollouts 1000 --ppo_update_frequency 256

# Data collection / reward sandbox
python main.py --use_config configs/explore.json
python main.py --use_config configs/evaluate_reward_system.json
```

---

## Testing

```bash
pytest tests/ -v
```

105 tests total. Buffer / running-stats / minibatch / extends-config tests are pure numpy/torch (no emulator) and run instantly. The vec env, env, RAM, replay-live tests spin up real PyBoy subprocesses and cap at a handful of steps each.

---

## Implementation Notes for Future You

These are the invariants new code has to maintain:

- **Observation contract:** envs return `{"image": ndarray, "ram": ndarray}`. Vec wrappers stack into a dict-of-batched-arrays. The model takes `(x_image, x_ram, mems)`. Any new modality goes into the dict — don't tunnel it through scalars or string-encoded fields.
- **RAM vector contract:** order is fixed by `RAM_FEATURE_KEYS`. Only ever append. Loaded checkpoints depend on the order.
- **Bootstrap:** any new return computation must use V(s_{T+1}) when the tail isn't terminal. See `_tail_bootstrap_value` and the per-env path in `VecPPOAgent._per_env_gae`. The RAM equivalent is `last_next_ram` — pass it alongside `last_next_state`.
- **Mems-on-done:** zero mems when `done=True`. Refill state_sequence and ram_sequence with the post-reset obs (broadcast to `seq_len`).
- **Metric arrays:** `episode_data["episode_losses"]` is **per-PPO-update** (multiple per episode in single-env, one per rollout in vec). Other arrays in there are **per-completed-episode**. Plot axes reflect this in `visuals.py`.
- **`num_episodes` is gone.** The training budget is `num_rollouts` everywhere. `self.episode` is still a running counter for entropy schedule, recording cadence, and metrics — never a stopping criterion.
- **Vec env workers are torch-free.** Don't import torch inside the worker — it inflates startup and breaks on some platforms.
- **Save-state pool indexing:** the agent's `env_state_indices[i]` always reflects the state currently running in env i. Cycling updates `env_pending_state_indices[i]` first; when the next done occurs, the agent promotes `pending → running` and tags that episode's metric. This lag-by-one is intentional and accurate.
- **Action-replay pool indexing:** mirrors save-state cycling exactly — `env_replay_indices[i]` is the currently-running replay, `env_pending_replay_indices[i]` is queued. Same lag-by-one promotion on done. The agent never trains on replay transitions; they're a pre-episode warm-up only.
- **`.steps` file format:** plain text, one int per line, `#` comments and blanks tolerated. Always overwritten at `<Checkpoints>/actions.steps` when a new single-episode reward record is set — copy out manually if you need history.
- **Story flags are read-only.** We extract 0xDA72–0xDB71 (256 bytes) into the RAM vector. Never write to this region.
- **Reward scaler:** `RewardScaler.observe(rewards, dones)` must be called after every env step in both modes. State is persisted in `info.pth`.
- **Strict loads:** `actor_critic.load_state_dict` and `scheduler.load_state_dict` are strict. If you change layer names or scheduler type, expect existing checkpoints to fail to load — and that's the desired behaviour (the user has not asked for backward-compat tolerance).
