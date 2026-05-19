# PoliwhiRL - AGENTS.md

## Overview

PPO training for Pokémon Crystal (Game Boy Color) via the PyBoy emulator. The agent observes screen images and chooses one of nine discrete button presses per step; rewards come from configurable sequential location goals and a small step penalty. Both single-environment and vectorised multi-process training are supported.

**Architecture:** `Screen Image (C,H,W) -> CNN -> Latent Embedding -> Transformer-XL -> Actor + Critic heads`

---

## File Layout

```
main.py                                  # Entry point: parse args, merge configs, dispatch by model type

PoliwhiRL/
├── __init__.py                          # exports: setup_and_train_PPO
├── PPO.py                               # Training entry: env shape probe, dispatch between PPOAgent and VecPPOAgent
├── agents/
│   └── PPO/
│       ├── __init__.py                  # exports: PPOAgent, VecPPOAgent
│       ├── ppo_agent.py                 # Single-env agent: per-episode rollout, KL early-stop, value clip, best-so-far
│       └── vec_ppo_agent.py             # Vec agent: T*N rollout collection, per-env GAE, periodic recording on env 0
├── environment/
│   ├── __init__.py                      # exports: PyBoyEnvironment, VecPyBoyEnv
│   ├── gym_env.py                       # Single PyBoyEnvironment with per-instance temp dir for ROM/RAM/RTC sidecars
│   ├── vec_env.py                       # Spawn-context multiprocessing wrapper around PyBoyEnvironment (N parallel envs)
│   ├── rewards.py                       # Reward calculator: location goals, pokedex, step/button penalties, exploration
│   └── RAM.py                           # RAM variable extraction
├── models/
│   ├── CNN/GameBoy.py                   # GameBoyBlock (Conv-BN-ReLU)
│   ├── PPO/
│   │   ├── __init__.py                  # exports: PPOModel
│   │   ├── ppo_model_implementation.py  # PPO losses (incl. KL early-stop, value clip), GAE, optimizer, CosineAnnealing LR
│   │   └── PPOTransformer.py            # GameBoyCNN + TransformerXLBlock x N + actor/critic heads
│   └── transformers/positional_encoding.py
├── replay/
│   ├── __init__.py                      # exports: PPOMemory, VecPPOMemory
│   ├── ppo_storage.py                   # Single-env rollout buffer
│   └── vec_ppo_storage.py               # (T, N, ...) rollout buffer with sliding-window emission
├── explorer/                            # Manual/random data collection (separate from PPO training)
├── reward_evaluator/                    # Predefined-action evaluation tool
└── utils/visuals.py                     # Per-step recording, plot_metrics

configs/
├── default_configs/                     # All defaults auto-merged at startup
│   ├── core_settings.json               # model, device, vision, checkpointing
│   ├── episode_settings.json            # episode_length, num_rollouts, num_envs, record_frequency
│   ├── ppo_settings.json                # PPO hyperparameters
│   ├── reward_settings.json             # default goal layout and reward magnitudes
│   ├── outputs_settings.json            # output paths
│   └── rom_settings.json                # ROM and start-state paths
├── first_steps.json                     # Stage 1: 2 goals, short episodes
├── second_steps.json                    # Stage 2: 4 goals, long episodes, loads first_steps checkpoint
├── explore.json                         # Data collection mode
└── evaluate_reward_system.json          # Reward-evaluation mode

tests/
├── test_PPO.py                          # Model init, losses, GAE/returns, bootstrap, KL
├── test_ppo_memory.py                   # Single-env buffer
├── test_vec_ppo_memory.py               # Vec buffer shape and alignment (no emulator)
├── test_vec_env.py                      # Live N=2 vec env + one-rollout end-to-end smoke
├── test_pyboyenv.py                     # Env reset/step/save/load
├── test_pyboy_ram.py                    # RAM extraction
└── test_reward_system.py                # Reward calculation, penalties, goals
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

## Training Modes (PPO)

Inside `setup_and_train_PPO`, `config["num_envs"]` selects the agent:

| `num_envs` | Agent | Iteration unit | Data per update |
|---|---|---|---|
| `1` | `PPOAgent` | One full episode (reset → done) | Buffer of T transitions (single env), updated at every T steps and at episode end |
| `>1` | `VecPPOAgent` | One rollout (T steps × N envs) | (T × N) transitions in one batched update |

Both modes share the same `PPOModel` and PPO loss; they differ only in how experience is collected.

### Why two modes coexist

The single-env agent stayed because it is the simpler implementation, exercises mid-episode updates (a feature, not a bug, with the bootstrap fix), and matches the historical training runs whose checkpoints exist on disk. The vec agent is the recommended default for new runs — larger effective batches, much better gradient signal, and immune to the multi-mid-episode-update fragility that the single-env agent has on long episodes.

### How vec mode actually works

1. **Setup.** `VecPyBoyEnv(config, num_envs=N)` spawns N subprocesses with `multiprocessing.get_context("spawn")`. Each worker creates its own `PyBoyEnvironment`, which in turn copies the ROM and sidecar files into a fresh `tempfile.TemporaryDirectory(prefix="poliwhirl_emu_")`. PyBoy writes back to that copy, so there is zero file contention between envs.
2. **Rollout.** Outer loop runs `num_rollouts` iterations. Each iteration:
   - Step all N envs `T = ppo_update_frequency` times. The model forward is one batched call per timestep returning `(N, action_size)` action probabilities and `(N,)` values.
   - Workers auto-reset on `done`: the next observation returned to the agent is from the new episode, and the agent zeros that env's transformer mems and refills its state-sequence buffer with the new obs.
   - Per-env episode reward sums and step counts are committed to `episode_data` whenever an env hits `done`.
3. **Update.** Per env, GAE is computed along the time axis (advantages **do not** cross env boundaries). The bootstrap value for the last transition uses `V(s_{T+1})` if the env hasn't terminated. After per-env GAE, the agent flattens `(W, N, ...) → (W*N, ...)` and calls `PPOModel.update` for `ppo_epochs` passes with KL early-stop and value clipping.
4. **Bookkeeping.** Cosine LR scheduler steps once per rollout. Periodic recording is triggered on env 0 every `record_frequency` completed episodes.

### Cross-env independence guarantees

Each env's trajectory is its own — they share only the policy parameters, never returns/advantages/log-probs/mems. PPO's importance ratio is per-transition (`new_log_prob / old_log_prob` where `old_log_prob` was stored at the moment that env's action was sampled), so the policy moving between rollouts is handled by the same clip + KL-stop machinery as in single-env.

---

## PPO Algorithm Details

### Loss

`loss = actor_loss + critic_loss + entropy_loss`

- **Actor (clipped surrogate):** `-E[min(r·A, clip(r, 1-ε, 1+ε)·A)]`, where `r = exp(new_log_prob - old_log_prob)`.
- **Critic (value-clipped MSE):** `0.5 * E[max((V_new − R)², (V_clip − R)²)]`, where `V_clip = V_old + clamp(V_new − V_old, -ε, +ε)`. Can be disabled with `ppo_clip_value_loss: false`.
- **Entropy bonus:** `-coef * E[H(π)]`, with `coef` decaying per **stage-relative** episode from `ppo_entropy_coef` to `ppo_entropy_coef_min`. The schedule resets on each `load_checkpoint` resume so a new curriculum stage gets fresh exploration.

### Returns and Advantages (GAE)

GAE-λ with bootstrap for truncated rollouts:

- Standard recursion: `A_t = δ_t + γλ(1−d_t)·A_{t+1}` where `δ_t = r_t + γ(1−d_t)·V(s_{t+1}) − V(s_t)`.
- At the tail of the rollout, when the episode has not actually ended (`done=False`), `V(s_{T+1})` is computed via an extra forward pass on `last_next_state` rather than being treated as zero. This was a real bug pre-fix and disproportionately affected long-episode runs (~14× more truncation events per episode in stage 2 than stage 1).

### KL Early-Stop

After each PPO epoch, the agent computes the **k3 approximate KL** (`E[(r − 1) − log r]`, always non-negative, lower-variance than the naïve `E[old − new]`). If it exceeds `ppo_target_kl` (default `0.015`), the remaining epochs are skipped. `null` disables the safeguard.

This is the single biggest stability gain over the original setup: PPO with high epoch counts can drift the policy outside the trust region in one bad update; KL-stop bounds that drift adaptively.

### Learning-rate schedule

`CosineAnnealingLR(T_max=num_rollouts (or override), eta_min=ppo_lr_min)`, stepped once per outer iteration. Replaced the original `CyclicLR(triangular2)` whose late LR peaks coincided with policy convergence and were implicated in the stage-1 collapse pattern (mean reward window 500–600 was 93.8, window 750–850 was −41.5).

### Best-so-far checkpoint

`save_model` overwrites the latest snapshot every `checkpoint_frequency`, but also writes a separate `Checkpoints/best/` snapshot whenever the 100-episode rolling mean reward beats the running best. Gated on a full window so an early-luck spike doesn't lock in. Useful for stage-to-stage transitions: point `load_checkpoint` at `…/Checkpoints/best` to start the next stage from the peak rather than the latest weights.

---

## Model Architecture

### `PPOTransformer`

```
Input:  (batch, seq_len, C, H, W)
  └─> GameBoyCNN: two GameBoyBlock stages (in_ch -> 16 -> 32), flatten, Linear -> d_model=128, ReLU
  └─> PositionalEncoding (sinusoidal, max_len=1000)
  └─> TransformerXL: 4 × TransformerXLBlock (MHA d_model=128, heads=8, FFN(4×) with GELU, dropout=0.1)
  └─> Last token (batch, d_model)
       ├─> fc_actor -> softmax -> (batch, action_size=9)
       └─> fc_critic -> (batch, 1)
```

Caller-managed transformer memory: `init_mems(batch_size, device)` returns a per-layer list of `(B, mem_len=16, d_model)` zero tensors. Each forward pass receives mems and returns updated mems (detached). The agent stores the **input** mems at each transition so PPO replay can reproduce the exact attention context.

**Mems reset on done.** Per-env, when `done=True`, that env's mems are zeroed before the next step. Single-env: a fresh `init_mems(1)` per episode; vec: in-place `.zero_()` of the env's slice in the shared mems tensor.

---

## Configuration

### Loading

Defaults live in `configs/default_configs/*.json` and are auto-loaded and merged on startup (`main.py:load_default_config`). A user config (`--use_config path/to.json`) overrides defaults. Individual CLI flags override the user config.

### Parameter reference

| Parameter | Default | What it does | When to change |
|---|---|---|---|
| `num_envs` | `1` | Number of parallel envs. `>1` uses `VecPPOAgent`. | `4`–`8` for serious training. Each env is one subprocess with its own emulator. |
| `num_rollouts` | `12` | **Total training budget**. In single-env: outer-loop iterations ≈ episodes run. In vec: number of (T × N) collect+update cycles. | Tune by env-step budget: `total_env_steps = num_rollouts × ppo_update_frequency × num_envs`. |
| `episode_length` | `50` | Per-episode step cap before forced done. | Should reflect the goals' depth. Stage 1: ~40. Stage 2: ~250. |
| `sequence_length` | `8` | Transformer input sequence length per forward pass. | Rarely. Larger = more context per decision, more compute, less data per buffer. |
| `record_frequency` | `100` | Save image dumps every N completed episodes (vec: env 0 only). | Larger for fast iteration, smaller for diagnostic depth. |
| `ppo_update_frequency` | `128` | Transitions per env per PPO update. Larger = bigger batch, fewer mid-episode truncations. | `128`–`256`. With long episodes (`episode_length > 200`), avoid tiny values like `32`. |
| `ppo_epochs` | `4` | PPO update passes per batch. | `3`–`8`. Higher with bigger batches; lower with tiny ones. KL-stop usually bounds this in practice. |
| `ppo_learning_rate` | `3e-4` | Peak LR for the cosine schedule. | Standard PPO LR; rarely tuned. |
| `ppo_lr_min` | `1e-5` | Cosine schedule floor. | Don't go below this — the schedule needs headroom. |
| `ppo_target_kl` | `0.015` | KL ceiling per epoch; aborts remaining epochs above this. | `null` to disable. `0.01`–`0.02` is the standard range. |
| `ppo_clip_value_loss` | `true` | Critic-side clip mirroring the actor clip. | `false` to recover the old unclipped MSE behaviour. |
| `ppo_epsilon` | `0.2` | PPO clipping range for the policy ratio. | Rarely tuned. |
| `ppo_gamma` | `0.98` | Discount factor. | Higher for long-credit tasks; rarely tuned. |
| `ppo_gae_lambda` | `0.95` | GAE bias-variance knob. `0` disables GAE. | Rarely tuned. |
| `ppo_value_loss_coef` | `1.0` | Weight on the critic loss. | Lower (`0.5`) if the critic dominates training. |
| `ppo_entropy_coef` | `0.01` | Initial entropy bonus weight. | Higher (`0.05`) for exploration-heavy curricula. |
| `ppo_entropy_coef_decay` | `0.99` | Per-(stage-relative)-episode multiplier. | Slower (`0.999`) for long stages. |
| `ppo_entropy_coef_min` | `0.001` | Entropy bonus floor. | Match to task: too high prevents commitment to long sequences. Stage 2 lesson: `0.015` works, `0.04` is too high. |
| `ppo_max_grad_norm` | `0.5` | Gradient clipping. | Standard. |
| `reset_lr_scheduler_on_load` | `true` | Reinit scheduler on `load_checkpoint`. | Keep `true` for stage transitions. |
| `reset_optimizer_on_load` | `false` | Reinit Adam on `load_checkpoint`. | `true` if Adam's moments are stale (e.g., long stage with reward-distribution change). |
| `goal_reward` | `100` | Per location-goal reward. | |
| `sequence_bonus` | `50` | Added when `require_sequential` and goal hit in order. | |
| `checkpoint_bonus` | `200` | Added at `checkpoint_goals` milestones. | |
| `all_goals_bonus` | `500` | Added on hitting `N_goals_target`. | Set to `0` to make the early_completion_bonus the primary terminal. |
| `early_completion_bonus` | `0` | Added on final goal. | |
| `exploration_reward` | `0.0` | Per first-visit `(x, y, map)` tile. | Small values (`1`–`3`) help in stages with sparse goals. |
| `step_penalty` | `-1` | Per-step penalty when `punish_steps`. | Scale to episode_length: per-episode worst case `step_penalty × episode_length` should be smaller than a goal's reward. Stage 2 with `-0.5` × 250 = −125 was close to overwhelming. |
| `button_penalty` | `-5` (fixed) | Penalty for start/select presses. | Not configurable. |

---

## Reward System

`PoliwhiRL/environment/rewards.py`. Each `env.step` returns a clipped reward sum of:

1. **Goal achievement** (sequential by default): `goal_reward + sequence_bonus + (checkpoint_bonus if in checkpoint_goals) + (all_goals_bonus + early_completion_bonus if final)`. Episode terminates if `break_on_goal` and all goals hit.
2. **Pokedex updates** (seen / owned).
3. **Exploration**: `exploration_reward` for each first-visit `(x, y, map_num)` tile this episode.
4. **Step penalty** (if `punish_steps`).
5. **Button penalty** for `start`/`select`.

All summed, clipped to `[-1000, 1000]`, returned as `float32`.

Goals are configured as `location_goals: [[ [x, y, map, room], ... ], ... ]` where each outer entry is one ordered goal and each inner list is a set of acceptable coordinates for that goal. The `room` field is currently ignored by the matcher (matching uses `[x, y, map_num]`).

---

## Environment

`PyBoyEnvironment` (`gym_env.py`) wraps PyBoy:

- **Actions:** 9 discrete (`""`, `a`, `b`, `left`, `right`, `up`, `down`, `start`, `select`).
- **Observation:** `(C, H, W)` screen image when `vision=true`, otherwise `(18, 20)` tilemap.
- **Per-action timing:** `button_hold_frames=15`, then ticks the remaining frames to reach `frames_per_action=90` total.
- **Per-instance temp dir:** ROM, RAM, RTC sidecars are copied into a fresh `tempfile.TemporaryDirectory` so PyBoy's writes can never affect the canonical files in `emu_files/`. This is what makes safe parallelism possible.

`VecPyBoyEnv` (`vec_env.py`) spawns N subprocess workers (spawn context), each owning one `PyBoyEnvironment`. Auto-resets on done. Commands: `reset`, `step`, `enable_record`, `close`. Workers report `output_shape` and `action_size` during init; mismatches across workers raise immediately.

---

## Training Loops in More Detail

### Single-env (`PPOAgent.train_agent`)

```
for rollout_idx in range(num_rollouts):
    run_episode():
        for each step until done or episode_length:
            sample action, env.step, store transition
            if steps % update_frequency == 0: update_model()  # mid-episode flush
    if buffer has data: update_model()                         # end-of-episode flush
    step_scheduler()
    every checkpoint_frequency: save_model()
```

`update_model` does the KL-stopped epoch loop over the current buffer (with `old_values` snapshotted before any epoch runs, so the value clip is stable).

### Vec (`VecPPOAgent.train_agent`)

```
vec = VecPyBoyEnv(config, num_envs)
obs = vec.reset()                                              # (N, *obs)
state_seq = tile obs to (N, seq_len, *obs)
mems = init_mems(N)
for rollout_idx in range(num_rollouts):
    memory.reset()
    for t in range(T):
        batched forward: actions, log_probs, new_mems
        next_obs, rewards, dones = vec.step(actions)
        memory.store_step(states_now, next_obs, actions, rewards, dones, log_probs, mems)
        ep_returns[done]  → commit metric, reset state_seq, zero that env's mems
        if dones[0] and self.episode crosses next_record_episode:
            vec.enable_record(...)   # next ep on env 0 records end-to-end
        mems = new_mems
    data = memory.get_data()
    flatten (W, N, ...) → (W*N, ...)
    per-env GAE → returns, advantages
    KL-stopped epoch loop over flat batch
    step_scheduler()
    every checkpoint_frequency: save_model()
```

---

## Curriculum Pattern

`first_steps.json` → train a small skill (`N_goals_target=2`, short episodes). Save its checkpoint. `second_steps.json` sets `load_checkpoint` to that path and extends the goal set and episode length. On load:

- The encoder + transformer weights persist.
- The entropy-coefficient schedule resets to its initial value (`stage_start_episode = self.episode`).
- The optimizer state persists unless `reset_optimizer_on_load: true`.
- The LR scheduler is reinitialised when `reset_lr_scheduler_on_load: true` (default).

If you ever change the LR scheduler type (we recently went `CyclicLR → CosineAnnealingLR`), the load is wrapped in try/except and will reinit fresh on `state_dict` mismatch rather than fail the whole resume.

---

## Outputs

For each run, files go under `output_base_dir`:

```
Checkpoints/
  actor_critic.pth                       # latest weights
  optimizer.pth, scheduler.pth, info.pth
  best/                                  # snapshot at best 100-ep rolling reward
    (same files)
Results/
  training_metrics.png                   # cumulative-mean curves
  training_metrics_current.png           # current-stage-only view (only on resume)
  metrics/training_metrics.json          # raw arrays + summary
Runs/                                    # per-step PNG dumps when recording is enabled
  N_goals_<n>/ep_<E>/step_<S>_btn_<B>_reward_<R>.png
```

---

## Running

```bash
# Defaults
python main.py

# Single-env (legacy behaviour)
python main.py --use_config configs/first_steps.json

# Vec mode by config override
python main.py --use_config configs/first_steps.json --num_envs 4

# Stage transition: pre-trained encoder, fresh entropy + LR schedules
python main.py --use_config configs/second_steps.json

# Override anything via CLI
python main.py --num_rollouts 200 --ppo_update_frequency 256

# Data collection / reward sandbox
python main.py --use_config configs/explore.json
python main.py --use_config configs/evaluate_reward_system.json
```

---

## Testing

```bash
pytest tests/ -v
```

71 tests total. The vec memory tests are pure-numpy (no emulator) and run instantly. The vec env tests spin up real PyBoy subprocesses and are the only place that can hang if the emulator deadlocks — they cap at a handful of steps each.

---

## Implementation Notes for Future You

- **Adding metrics:** `episode_data["episode_losses"]` is per-PPO-update (not per-episode); other arrays in there are per-completed-episode. Plot axes reflect this in `visuals.py`.
- **`num_episodes` is gone.** The training budget is `num_rollouts` everywhere. `self.episode` is still a running counter for entropy schedule and metrics; do not use it as a stopping criterion.
- **Mems and `done`.** Any new code that touches mems must zero them on `done`. The vec agent does this in `_collect_rollout`; if you add a third agent, replicate that.
- **Bootstrap.** Any new return computation must use `V(s_{T+1})` when the tail isn't terminal. See `_tail_bootstrap_value` and the per-env path in `VecPPOAgent._per_env_gae`.
- **Vec env workers** are pure (no torch), spawn-context. Don't import torch inside the worker — it'll inflate startup and break on some platforms.
