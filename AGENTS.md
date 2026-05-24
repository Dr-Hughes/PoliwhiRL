# PoliwhiRL - AGENTS.md

## Overview

PPO training for Pokémon Crystal (Game Boy Color) via the PyBoy emulator. The policy sees both the **screen image** and a **normalised RAM vector** (position, party state, current goal target, exploration summary, curated story-flag bits) and chooses one of nine discrete button presses per step. Rewards come from configurable sequential location goals plus exploration, step penalties, and party progress bonuses. Both single-environment and vectorised multi-process training are supported, with a per-worker save-state pool for curriculum mixing.

**Current status:** Stages 1–2 solved. Stage 3 collapsed mid-training (entropy floor + curriculum jump). Stage 4 previously unreachable (now fixed with multi-fire pokedex goals). Stage 7 collapsed into a "non-fighter" policy: max `goals_total = 12 / 15`, the missing 3 are the `total_level:3` goal slots that require winning a battle. The agent learned to flee for `pokedex_seen` credit instead of engaging combat. Reward system now includes battle engagement, damage-dealt, and per-map first-visit signals (see "Goal & reward semantics" below). See [`model_status.md`](./model_status.md) for full evaluation and recommended fixes.

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
│   ├── ppo_agent.py                     # Single-env agent: per-episode rollout, KL early-stop, value clip, entropy plateau detection
│   ├── vec_ppo_agent.py                 # Vec agent: T×N rollout, per-env GAE, save-state cycling, per-state metric tagging
│   └── _minibatch.py                    # Shared mini-batch iterator for both agents
├── environment/
│   ├── __init__.py                      # exports: PyBoyEnvironment, VecPyBoyEnv, GoalsManager
│   ├── gym_env.py                       # PyBoy env, dict obs, RAM_FEATURE_KEYS, _build_ram_vector, _DERIVED_FLAG_TABLE
│   ├── vec_env.py                       # Spawn-context multiprocessing wrapper; replay pool, multi-trajectory .steps format
│   ├── goals.py                         # GoalsManager: typed goal parser (location/pokedex/level/xp), hard/soft, progress tracking, validation
│   ├── rewards.py                       # Reward calculator: delegates goal tracking to GoalsManager, computes macro+micro rewards
│   └── RAM.py                           # RAM address book + get_variables()
├── models/
│   ├── CNN/GameBoy.py                   # GameBoyBlock (Conv-GroupNorm-ReLU), GameBoyOptimizedCNN
│   ├── PPO/
│   │   ├── __init__.py                  # exports: PPOModel
│   │   ├── ppo_model_implementation.py  # PPO losses (KL early-stop, value clip), GAE, linear entropy annealing, plateau rewind
│   │   └── PPOTransformer.py            # GameBoyCNN + RAMEncoder + fusion + TransformerXLBlock × N + actor/critic
│   └── transformers/positional_encoding.py
├── replay/
│   ├── __init__.py                      # exports: PPOMemory, VecPPOMemory
│   ├── ppo_storage.py                   # Single-env rollout buffer (images + RAM)
│   └── vec_ppo_storage.py               # (T, N, ...) rollout buffer with sliding-window emission
├── explorer/                            # Manual/random data collection (separate from PPO)
├── reward_evaluation/                   # Predefined-action evaluation tool
├── evaluator/                           # Inference-only model runner (greedy playthrough)
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
├── stages/{first,second,third,fourth,fifth}.json  # Curriculum stages, extending curriculum_base.json
├── explore.json                         # Manual/random data collection mode
├── evaluate_reward_system.json          # Reward-evaluation mode (model="reward_eval")
└── inference.json                       # Inference-only model runner (model="inference")

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

## Step Semantics — what "one step" actually means

**One step = one button press = one model decision.** Frame timing is handled inside the env; the agent never sees frames.

| Constant | Value | Meaning |
|---|---|---|
| `frames_per_action` | 90 | Frames advanced per `env.step()` (~1.5s at 60 fps) |
| `button_hold_frames` | 15 | Frames the button is held; remaining 75 frames are no-input so animations/dialogue advance |

Implications for everything else:

- `episode_length=256` means **256 button presses**, not 256 frames. Wall-clock per episode ≈ `256 × 1.5s ≈ 6.5 min` of in-game time.
- Most steps inside the player's house are spent **mashing A through dialogue / cutscenes**, not navigating. Goal-to-goal distance in tiles is ~3–7 for the in-house segment; the 300-step total to "got first pokemon" is dominated by dialogue, not movement.
- Wild encounters and trainer battles are also navigated one button press at a time. The policy must learn to read `battle_type` (RAM index 23) and switch behaviour — battles look nothing like the overworld.
- `start`/`select` cost `button_penalty = -5` per press (fixed in `rewards.py`).

---

## Goal & reward semantics — exact rules

The reward system has three knobs that interact non-obviously. **Read this before touching curriculum configs.**

### Architecture: `GoalsManager` + `Rewards`

Goal management is split into two classes:

- **`GoalsManager`** (`environment/goals.py`): parses typed goal configs, validates them, tracks progress (`N_goals`, `current_goal_index`, pokedex/level/xp counters). Separates *curriculum structure* from *reward magnitudes*.
- **`Rewards`** (`environment/rewards.py`): thin wrapper that delegates all goal tracking to `GoalsManager` and computes macro + micro rewards.

### Goal config format

Goals are specified as a **typed `goals` array** in the stage config. Each entry has a `"type"` field:

```json
"goals": [
    {"type": "location", "positions": [[9, 1, 6, 50]], "hard": true},
    {"type": "location", "positions": [[59, 8, 3, 50]], "hard": false},
    {"type": "pokedex", "kind": "seen", "threshold": 4},
    {"type": "level", "kind": "total_level", "threshold": 3},
    {"type": "xp", "threshold": 5, "xp_per_fire": 10}
]
```

### Goal types

| Type | Config | Behaviour |
|---|---|---|
| `location` | `{"type": "location", "positions": [[x, y, map, room?, bank?], ...], "hard": true\|false}` | Tile match. **`hard`** (default `true`): advances the curriculum index and counts toward `hard_goal_count_target`. **`soft`** (`"hard": false`): pays `soft_waypoint_reward` per hit, never advances curriculum or blocks termination. |
| `pokedex` | `{"type": "pokedex", "kind": "seen"|"owned", "threshold": N}` | **Multi-fires**: a threshold of N contributes N goal slots (one per integer increment). E.g. `threshold: 4` fires at seen=1, 2, 3, 4. |
| `level` | `{"type": "level", "kind": "total_level", "threshold": N}` | **Multi-fires**: fires N times as the party gains N levels from the starting point. Suppressed on party size change. |
| `xp` | `{"type": "xp", "threshold": N, "xp_per_fire": K}` | Fires once per `xp_per_fire` XP gained, capped at N fires total. Suppressed on party size change. |

### Position format

Each position option is `[x, y, map_num, room?, map_bank?]`. The 4th element (room) is ignored for matching. The 5th element (`map_bank`) is used for disambiguation when present — matching requires both `map_num` and `map_bank` to match. Internally normalised to `[x, y, bank, map_num]`.

### `hard_goal_count_target` (formerly `N_goals_target`)

The terminator. When `N_goals >= hard_goal_count_target` and `break_on_goal=true`, the episode ends. If not specified, defaults to the sum of all possible fires across all goal types. **A target lower than hard goal count means later hard goals never train; a target higher than total possible fires is unreachable** and the episode times out on `episode_length` instead.

| Config field | Meaning |
|---|---|
| `hard_goal_count_target` | The terminator count. Defaults to total possible fires if `-1` or omitted. |
| `require_sequential` | If true (default), hard location goals must be hit in order. |
| `checkpoint_goals` | List of `N_goals` values that trigger an extra `checkpoint_bonus`. Default `[3, 6]` in reward_settings.json. Curriculum stages do not use checkpoint bonuses. |
| `break_on_goal` | Terminate the episode on reaching `hard_goal_count_target`. |

### Per-step reward formula

```
r = hard_goal_hit_reward                     # goal_reward + bonuses on hard tile hit
  + soft_waypoint_reward × soft_hits         # per soft goal hit (default 25)
  + pokedex_seen_reward if seen ++           # configurable; default 0 in curriculum_base
  + pokedex_owned_reward if owned ++         # configurable; default 150
  + xp_goal_reward × xp_new_fires            # per xp goal fire
  + all_goals_bonus + early_completion_bonus # when any goal crossing hits target
  + party_level_reward × Δlevel              # configurable; default 10 in curriculum_base
  + party_exp_reward × Δexp                  # configurable; default 0.01 in curriculum_base
  + xp_milestone_reward                      # fires once per threshold of cumulative XP gained
  + exploration_reward                       # if (x, y, map_bank, map_num) novel this episode
  + new_map_reward                           # if (map_bank, map_num) novel this episode
  + distance_shaping                         # potential-based, when getting closer on same map
  + battle_engagement_reward                 # per step when battle_type != 0
  + damage_dealt_reward × Δenemy_hp          # only when enemy HP drops; in-battle only
  + step_penalty                             # if punish_steps
  + button_penalty (-5)                      # if action ∈ {start, select}
clipped to [-1000, 1000]
```

**Party progress rewards** (`_check_party_progress`): tracks the party's total level and EXP between steps. Rewards increases (never decreases — levels/EXP only go up). Seeded from step 0 of each training episode (`start_new_episode()` resets `_prev_party_level` / `_prev_party_exp`) so replay transitions don't produce phantom rewards. When party size changes, the reward is suppressed to avoid compounding with capture/swap bonuses — unless `party_reward_check_battle=True` and the agent is in a battle (`battle_type != 0`). Configurable via `party_level_reward` and `party_exp_reward`; defaults to 0 (disabled) in reward_settings.json, enabled at `10` / `0.01` in curriculum_base.json.

**XP milestone reward** (`_check_party_progress`): fires `xp_milestone_reward` once per `xp_milestone_threshold` of cumulative XP gained. The accumulator pauses on party size change (so a new Pokemon's existing EXP doesn't trigger a false milestone) but is not reset — it resumes accumulation next step. Disabled by default (`xp_milestone_threshold: 0`).

**XP goals** (`_check_xp_goals`): a typed goal type that fires `xp_goal_reward` per fire. Tracks cumulative XP gained since episode start (or last party size change). Each fire consumes `xp_per_fire` XP from the total gained. Capped at the goal's `threshold` fires total. Counts toward `N_goals` and can trigger episode termination.

**Distance shaping** (`_distance_shaping`): potential-based reward when the player gets closer to the active *hard* location goal on the same map. Coefficient `distance_shaping_coef` scales the improvement `Δd`. Reset on goal hit or map change. Only fires when `cur_map == target_map` (and `cur_bank == target_bank` if the goal specifies a bank).

**New-map first-visit bonus** (`_exploration_reward`): in addition to the per-tile `exploration_reward`, a separate `new_map_reward` fires once per never-before-seen `(map_bank, map_num)` pair. The `explored_maps` set is preserved across the replay → training boundary (same as `explored_tiles`) so the replay's discovered maps don't re-fire. Sized smaller than a location-goal hit (default 50 vs `goal_reward + sequence_bonus = 150`) to encourage exploration toward new towns without overwhelming the curriculum path.

**Battle engagement + damage** (`_battle_engagement_reward`): outside battle (`battle_type == 0`) this is a no-op and clears the damage tracker so HP from a finished battle never leaks into the next. Inside a battle pays (a) a flat `battle_engagement_reward` per step (offsets the step penalty so engaging combat isn't net-negative), and (b) `damage_dealt_reward × Δhp` whenever enemy HP drops. The first observed in-battle step seeds `_prev_enemy_hp` without crediting damage. HP increases (healing) are ignored. `_prev_enemy_hp` resets on `start_new_episode()` so replay-time damage cannot be re-credited.

A hard location goal hit yields `goal_reward + sequence_bonus (+ checkpoint_bonus) (+ all_goals_bonus + early_completion_bonus on final)`.
A soft location goal hit yields `soft_waypoint_reward` per hit (goal is removed after matching).

### The current curriculum's goal list (decoded)

These are the location goals used across stages. Goals 6 and 7 are **soft** in stages 5–7 (they pay a waypoint bonus but don't advance the curriculum index):

| # | `[x, y, map, bank]` | What it represents | Hard/Soft | Typical distance from prior goal |
|---|---|---|---|---|
| 1 | `[9, 1, 6, 50]` | Downstairs entry tile in player's house | hard | start (~0) |
| 2 | `[8\|9, 4, 6, 50]` | Talking-to-mother position | hard | ~3 tiles |
| 3 | `[8\|9, 5, 6, 50]` | One tile past mother (commits to leaving) | hard | 1 tile |
| 4 | `[13, 6, 4, 50]` | First step outside on the town map | hard | ~5 tiles + door warp |
| 5 | `[6, 4, 4, 50]` | Professor's lab door | hard | ~7 tiles |
| 6 | `[59, 8\|9, 3, 50]` | Entry to Route 29 (east edge) | **soft** (stages 5–7) | many tiles + lab interior + pokemon cutscene |
| 7 | `[31, 13, 3, 50]` | Tricky gap on Route 29 | **soft** (stages 5–7) | ~30 tiles on the route |

### Multi-fire goal types

**Pokedex goals**: a threshold of N contributes N goal slots. For example, `{"kind": "owned", "threshold": 1}` + `{"kind": "seen", "threshold": 3}` contributes 1 + 3 = 4 goal fires total. Each fire at count k (for k = 1..N) increments `N_goals` by 1. The type is removed from the active pokedex goals once fully consumed. An independent counter `pokedex_goals_completed` tracks the total across types for the RAM-vector progress feature.

**Level goals**: `{"kind": "total_level", "threshold": N}` fires N times as the party gains N levels from the starting point. Suppressed on party size change.

**XP goals**: `{"threshold": N, "xp_per_fire": K}` fires once every K XP gained, up to N fires. Suppressed on party size change.

**When setting `hard_goal_count_target`, count total possible fires as:**
```
total_fires = len(hard_goals) + len(soft_goals)
            + sum(pokedex_goal.threshold for each pokedex goal)
            + sum(level_goal.threshold for each level goal)
            + sum(xp_goal.threshold for each xp goal)
```
For instance, stage 7: `5 hard + 2 soft + 1 owned + 4 seen + 3 level = 15` total fires. If `hard_goal_count_target` exceeds `total_fires`, the episode will timeout. If it is lower than hard goal count, the remaining hard goals will never train. A one-time validation warning fires on config load (see "Config validation" below).

### Config validation

On first instantiation, `GoalsManager` validates the goal configuration and prints warnings if:
- `hard_goal_count_target` exceeds total possible fires (hard + soft + pokedex + level + xp) — the episode will timeout.
- `hard_goal_count_target` is less than hard location goals — some hard goals may never train.

The check uses a module-level `id(config)` set to avoid firing on every env reset.

### Curriculum stages — what each stage actually trains

| Stage | Goals trained | Total possible fires | `hard_goal_count_target` | Notes |
|---|---|---|---|---|
| 1 (`first.json`) | 2 hard location | 2 | 2 | Trains "down stairs → mother". 150 rollouts, 40-step episodes. **SOLVED** (97% success). |
| 2 (`second.json`) | 4 hard location | 4 | 4 | Adds "exit past mother → step outside". Loads stage 1 best. Action replay. 250 rollouts, 256-step episodes. **SOLVED** (100% success). |
| 3 (`third.json`) | 5 hard location + owned:1 | 5 + 1 = 6 | 6 | Adds "walk to lab door, get pokemon (via pokedex)". 1000 rollouts, 1024-step episodes. `distance_shaping_coef=1.5`. **COLLAPSED** at ep ~4050. |
| 4 (`fourth.json`) | 7 hard location + owned:1 | 7 + 1 = 8 | 7 | Adds Route 29 goals (all hard). `distance_shaping_coef=1.5`. Pokedex `owned:1` is dead weight (episode breaks on location goal 7 before pokedex fires). |
| 5 (`fifth.json`) | 5 hard + 2 soft location + owned:1 + seen:3 | 5 + 2 + 1 + 3 = 11 | 10 | Vec mode (num_envs=16), 2048-step episodes. `distance_shaping_coef=1.5`. Goals 6–7 are soft (waypoint bonus only). |
| 6 (`sixth.json`) | 5 hard + 2 soft location + owned:1 + seen:4 | 5 + 2 + 1 + 4 = 12 | 11 | Vec mode, 4096-step episodes. `exploration_reward=2`. Goals 6–7 are soft. |
| 7 (`seventh.json`) | 5 hard + 2 soft location + owned:1 + seen:4 + total_level:3 | 5 + 2 + 1 + 4 + 3 = 15 | 15 | Vec mode, 5120-step episodes. `exploration_reward=2`. Trains battle engagement via level goals. Goals 6–7 are soft. |

`action_replay_paths` chains stages: stage N's training output writes `actions.steps`, stage N+1 replays those before each training episode so the policy only has to learn the *terminal segment* of the cumulative curriculum.

---

## Modes

Dispatched in `main.py` based on `config["model"]`:

| Mode | Value | Purpose |
|------|-------|---------|
| Train | `"PPO"` | PPO training (single-env or vec, decided by `num_envs`) |
| Explore | `"explore"` | Manual or random data collection to SQLite DB |
| Evaluate | `"reward_eval"` | Run predefined action sequences, evaluate reward function |
| Inference | `"inference"` | Greedy playthrough of trained model with PNG recording |

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

Current vector width: **`RAM_OBS_DIM = 76`** (31 base features + 45 derived story-flag bits).

Position normalisation uses `min(val, 32) / 32.0` for x/y coordinates (soft saturation — most relevant tiles are within 0–31). All other features use their respective domain-specific scaling.

| Index | Feature | Source | Scaling |
|---|---|---|---|
| 0 | `x` | RAM `X` | `min(val, 32) / 32.0` |
| 1 | `y` | RAM `Y` | `min(val, 32) / 32.0` |
| 2 | `map_num` | RAM | `/ 255` |
| 3 | `map_bank` | RAM | `/ 255` |
| 4 | `room` | RAM | `/ 255` |
| 5 | `warp_number` | RAM | `/ 255` |
| 6 | `party_level` | RAM | `/ 100` |
| 7 | `party_hp` | RAM | `/ 1000` |
| 8 | `party_exp` | RAM | `log1p / 20` |
| 9 | `money` | RAM | `/ 1e6` |
| 10 | `pokedex_seen` | RAM | `/ 251` |
| 11 | `pokedex_owned` | RAM | `/ 251` |
| 12 | `collision_down` | RAM | `/ 255` |
| 13 | `collision_up` | RAM | `/ 255` |
| 14 | `collision_left` | RAM | `/ 255` |
| 15 | `collision_right` | RAM | `/ 255` |
| 16 | `target_x` | Rewards active goal | `min(val, 32) / 32.0` |
| 17 | `target_y` | Rewards active goal | `min(val, 32) / 32.0` |
| 18 | `target_map` | Rewards active goal | `/ 255` |
| 19 | `target_map_bank` | Rewards active goal | `/ 255` (0.0 when goal has no bank) |
| 20 | `has_active_target` | Rewards | 0 or 1 |
| 21 | `explored_tile_count` | `len(explored_tiles)` | `log1p / 6` |
| 22 | `n_location_goals_completed` | Rewards `current_goal_index` | raw float |
| 23 | `n_pokedex_goals_completed` | Rewards `pokedex_goals_completed` | raw float |
| 24 | `n_level_goals_completed` | Rewards `level_goals_completed` | raw float |
| 25 | `battle_type` | RAM `D22D` (0=none, 1=wild, 2=trainer) | `/ 255` |
| 26 | `johto_badges` | RAM `D857` (8-bit badge bitmask) | `/ 255` |
| 27 | `player_state` | RAM `D95D` (0=walk, 1=bike, 2=skate, 4=surf) | `/ 255` |
| 28 | `key_items_count` | RAM `D8BC` | `/ 25` |
| 29 | `game_hour` | RAM `D4B7` (0–23) | `/ 255` |
| 30 | `bgm_id` | RAM `C2A9` (current BGM track) | `/ 255` |
| 31–75 | derived flags (see table below) | story-flag bit extraction | 0 or 1 |

### Derived flags (indices 31–75)

Individual bits extracted from the story-flag bytes (`0xDA72–0xDB71`) for immediate policy access. Each is 0 (flag clear) or 1 (flag set). Replaces the old 256 raw bytes with 45 curated, semantically meaningful features.

| Index | Feature | Story flag | Meaning |
|---|---|---|---|
| 31 | `has_cut` | 16 | HM01 Cut acquired |
| 32 | `has_surf` | 18 | HM03 Surf acquired |
| 33 | `has_strength` | 19 | HM04 Strength acquired |
| 34 | `has_flash` | 20 | HM05 Flash acquired |
| 35 | `has_rock_smash` | 26 | HM04 Rock Smash acquired |
| 36 | `has_waterfall` | 27 | HM05 Waterfall acquired |
| 37 | `has_fly` | 28 | HM02 Fly acquired |
| 38 | `has_dig` | 33 | HM04 Dig acquired |
| 39 | `has_starter` | 25 | Received Pokemon from Elm (Route 29 opens) |
| 40 | `got_bike` | 100 | Acquired the bicycle |
| 41 | `got_national_dex` | 101 | Received National Pokédex |
| 42 | `met_prof_oak` | 102 | Met Professor Oak |
| 43 | `saw_hooh` | 103 | Encountered Ho-Oh |
| 44 | `saw_lugia` | 104 | Encountered Lugia |
| 45 | `saw_entei` | 105 | Encountered Entei |
| 46 | `sukiyaki_song_heard` | 106 | Heard the sukiyaki song (Radio Tower) |
| 47 | `farfetchd_herded` | 43 | Farfetch'd quest done, Headbutt available |
| 48 | `sudowoodo_defeated` | 44 | Route 36 to Ecruteak open |
| 49 | `slowpoke_well_cleared` | 38 | Team Rocket cleared from Slowpoke Well |
| 50 | `rocket_cleared_radio` | 52 | Radio Tower cleared |
| 51 | `rocket_cleared_hideout` | 53 | Goldenrod Hideout cleared |
| 52 | `falkner_defeated` | 116 | Gym 1 (Violet) beaten |
| 53 | `bugsy_defeated` | 130 | Gym 2 (Azalea) beaten |
| 54 | `whitney_defeated` | 144 | Gym 3 (Goldenrod) beaten |
| 55 | `morty_defeated` | 158 | Gym 4 (Ecruteak) beaten |
| 56 | `jasmine_defeated` | 172 | Gym 5 (Olivine) beaten |
| 57 | `pryce_defeated` | 186 | Gym 6 (Johto) beaten |
| 58 | `clair_defeated` | 200 | Gym 7 (Ice Path) beaten |
| 59 | `elite_four_wiltz` | 160 | Defeated Will (E4 member 1) |
| 60 | `elite_four_koga` | 174 | Defeated Koga (E4 member 2) |
| 61 | `elite_four_sabrina` | 188 | Defeated Sabrina (E4 member 3) |
| 62 | `champion_defeated` | 214 | Defeated the Champion |
| 63 | `bruno_defeated` | 228 | Kanto Gym (Celadon) beaten |
| 64 | `lt_surge_defeated` | 242 | Kanto Gym (Vermilion) beaten |
| 65 | `erika_defeated` | 256 | Kanto Gym (Cerulean) beaten |
| 66 | `blaine_defeated` | 270 | Kanto Gym (Cinnabar) beaten |
| 67 | `goldenrod_civilians_returned` | 1568 | Civilians returned to Goldenrod (post-Rocket) |
| 68 | `radio_tower_stairs_clear` | 1587 | Blackbelt no longer blocks Radio Tower stairs |
| 69 | `ilex_gate_clear` | 1611 | Lass no longer blocks Ilex Forest gate |
| 70 | `route_43_gate_clear` | 1625 | Rocket grunts cleared from Route 43 gate |
| 71 | `mahogany_east_clear` | 1631 | Pokefan no longer blocks Mahogany east exit |
| 72 | `mahogany_gym_clear` | 1632 | Pokefan no longer blocks Mahogany Gym |

Note: `blackthorn_gym_clear` (flag 1646), `dragons_den_clear` (flag 1655), and `snorlax_moved` (flag 1661) are defined in `_DERIVED_FLAG_TABLE` but fall at indices 73–75. If the table grows, `RAM_OBS_DIM` adjusts automatically. Verify the actual count against `len(_DERIVED_FLAG_TABLE)` — currently **42 entries** (indices 31–72).

Wait — recounting: 31 base features (indices 0–30) + 42 derived flags = 73 total (indices 0–72, derived at 31–72 inclusive).

**Goal-conditioning (indices 16–20)** is the lever that solves narrative backtracking: when the curriculum advances to a new goal whose coordinates differ from the last one, the policy's input changes accordingly. The `target_map_bank` feature (index 19) closes the same map-num collision that the player's `map_bank` feature closes — without it, goals at the same map number in different bank groups look identical in the observation.

**Goal-progress counters (indices 22–24)** expose raw integer counts of completed location, pokedex, and level goals. These are carried across the replay boundary so the policy can distinguish "near a goal I've already crossed" from "near a goal I still need to hit."

**Battle type (index 25)** is the single highest-impact raw feature: it tells the policy whether it's exploring (0), in a wild battle (1), or in a trainer battle (2). This prevents the policy from pressing random navigation buttons during battles.

> **Read-only.** We never write to the story-flag region (`0xDA72–0xDB71`). The 8-byte GameShark-style write quirk is irrelevant to us.

---

## PPO Algorithm Details

### Loss
`loss = actor_loss + critic_loss + entropy_loss`

- **Actor (clipped surrogate):** `-E[min(r·A, clip(r, 1-ε, 1+ε)·A)]`, where `r = exp(new_log_prob - old_log_prob)`.
- **Critic (value-clipped MSE):** `0.5 * E[max((V_new − R)², (V_clip − R)²)]`, where `V_clip = V_old + clamp(V_new − V_old, -ε, +ε)`. Disable with `ppo_clip_value_loss: false`.
- **Entropy bonus:** `-coef * E[H(π)]`, with `coef` following a **linear annealing** schedule from `ppo_entropy_coef` to `ppo_entropy_coef_min` over `ppo_entropy_anneal_steps` (defaults to `num_rollouts`). The schedule resets on each `load_checkpoint`.

### Entropy schedule — linear annealing with plateau rewind

The entropy coefficient follows a linear decay:

```
effective_step = max(0, step - reset_offset)
progress = min(effective_step / anneal_steps, 1.0)
coef = entropy_coef * (1 - progress) + entropy_min * progress
```

`step` is a training-progress counter: rollout index in vec mode, episode index in single-env. Previously the schedule used `episode` (actual episode count), which caused the vec agent to hit the entropy floor after only ~6% of training with 16 envs. Now both modes index by `step`, so the schedule denominator matches `num_rollouts` in both modes.

A **plateau-detection mechanism** can rewind the schedule when training stalls. Both agents implement `_check_entropy_plateau()`:

1. After each completed episode, checks a rolling window of recent episodes.
2. Detects a plateau when the goal count is flat across the window **and** still below `N_goals_target`. (The previous reward-CV < 0.15 condition never triggered in practice and was removed.)
3. On detection, sets `_entropy_reset_offset` to rewind the effective step count by `entropy_reset_rewind_fraction` of the window size. The offset is in rollout-space, matching the schedule's step-space.
4. Debounced by `entropy_reset_debounce_fraction` of total budget to prevent rapid cycling.
5. Minimum budget elapsed before activation: `entropy_reset_min_fraction` of total budget.

Config keys (all in `curriculum_base.json`):
- `entropy_plateau_reset`: enable/disable (default `true`)
- `entropy_reset_window_fraction`: rolling window as fraction of budget (default `0.1`)
- `entropy_reset_min_fraction`: minimum fraction of budget before detection activates (default `0.1`)
- `entropy_reset_debounce_fraction`: cooldown between resets as fraction of budget (default `0.125`)
- `entropy_reset_rewind_fraction`: how far back to rewind as fraction of window (default `0.1`)

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
`CosineAnnealingLR(T_max=ppo_scheduler_t_max, eta_min=ppo_lr_min)`, stepped once per outer iteration. In single-env mode, `ppo_scheduler_t_max` defaults to `num_rollouts`. In vec mode, the agent explicitly sets `ppo_scheduler_t_max = num_rollouts`. Replaced the original `CyclicLR(triangular2)` whose late LR peaks coincided with policy convergence and were implicated in the stage-1 collapse pattern.

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
x_image (B, T, C, H, W) → GameBoyCNN ────────→ (B*T, d_model=128)   ┐
                                                                     ├─ concat → Linear(128+64, 128) → (B*T, 128)
x_ram   (B, T, ram_dim)  → RAMEncoder (MLP) ──→ (B*T, d_ram=64)     ┘
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

Transformer-XL memory is **caller-managed**: `init_mems(batch_size, device)` returns a per-layer list of `(B, mem_len=64, d_model)` zero tensors. Each forward pass receives mems and returns updated mems (detached). The agent stores the **input** mems at each transition so PPO replay reproduces the exact attention context.

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

Global defaults (from `default_configs/`):

| Parameter | Default | What it does | When to change |
|---|---|---|---|
| `num_envs` | `8` | Parallel envs. `>1` uses `VecPPOAgent`. | Scale with cores; common values 4–32. |
| `num_rollouts` | `12` | **Training budget.** Single-env: outer-loop iterations ≈ episodes run. Vec: number of (T × N) collect+update cycles. | Total env steps = `num_rollouts × ppo_update_frequency × num_envs`. |
| `episode_length` | `50` | Per-episode step cap before forced done. | Tune to goal depth. |
| `sequence_length` | `16` (core_settings) / `8` (curriculum_base) | Transformer input sequence length per forward pass. | curriculum_base overrides to 8; restore to 16 for stages with long navigation segments. |
| `record_frequency` | `100` | Save image dumps every N completed episodes (vec: env 0 only). | Bigger = less disk, less visibility. |
| `state_paths` | `[]` | Save-state pool (list of paths). If empty, falls back to single `state_path`. | Add entries to mix in mid-game starts. |
| `state_cycle_strategy` | `"random"` | How envs pick their next save-state on auto-reset. `"random"` / `"none"`. | `"none"` to pin each env to its initial state. |
| `action_replay_paths` | `[]` | Action-replay pool (list of `.steps` paths). Workers sample uniformly from the pool on each reset, applying a random prefix. | Set for stage-to-stage continuation. |
| `ppo_update_frequency` | `128` | Transitions per env per PPO update (rollout length T). Larger → bigger batch, fewer mid-episode truncations. | `128`–`256`. With long episodes avoid tiny values. |
| `ppo_epochs` | `3` | PPO update passes per batch. KL early-stop usually bounds this. | `3`–`8`. |
| `ppo_minibatch_size` | `128` | Shuffled minibatch size. `null` or `0` disables minibatching. | Scale with effective batch (W × N). |
| `ppo_learning_rate` | `3e-4` | Peak LR for the cosine schedule. | Rarely. |
| `ppo_lr_min` | `1e-5` | Cosine schedule floor. | Don't go below this. |
| `ppo_scheduler_t_max` | `num_rollouts` | Cosine annealing horizon. Vec agent sets this explicitly. | Override for custom annealing horizon. |
| `ppo_target_kl` | `0.01` | KL ceiling per epoch. | `null` to disable. `0.01`–`0.02` typical. |
| `ppo_clip_value_loss` | `true` | Critic-side clip mirroring the actor clip. | `false` for unclipped MSE. |
| `ppo_adam_eps` | `1e-5` | Adam epsilon. | Standard PPO setting. |
| `ppo_epsilon` | `0.2` | PPO clipping range for the policy ratio. | Rarely. |
| `ppo_gamma` | `0.99` | Discount factor. | Higher for long credit. |
| `ppo_gae_lambda` | `0.95` | GAE bias-variance knob. `0` disables GAE. | Rarely. |
| `ppo_value_loss_coef` | `1.0` | Critic loss weight. | Lower if critic dominates. |
| `ppo_entropy_coef` | `0.01` (global) / `0.05` (curriculum) | Initial entropy bonus weight. | Higher for exploration-heavy stages. |
| `ppo_entropy_coef_decay` | `0.99` (global) / `0.999` (curriculum) | Per-(stage-relative)-episode multiplier (legacy, unused by linear anneal). | Slower (`0.999`) for long stages. |
| `ppo_entropy_coef_min` | `0.005` | Entropy bonus floor. | Keeps non-trivial floor for escape pressure; too high prevents commitment. |
| `ppo_entropy_anneal_steps` | `num_rollouts` | Linear annealing horizon for entropy. | Override for custom annealing. |
| `ppo_max_grad_norm` | `0.5` | Gradient clipping. | Standard. |
| `reset_lr_scheduler_on_load` | `true` | Reinit scheduler on `load_checkpoint`. | Keep `true` for stage transitions. |
| `reset_optimizer_on_load` | `false` | Reinit Adam on `load_checkpoint`. | `true` for save-state jumps (stale Adam moments on distribution shift). |
| `goal_reward` | `100` | Per hard location-goal reward. | |
| `sequence_bonus` | `50` | Added when `require_sequential` and goal hit in order. | |
| `checkpoint_bonus` | `200` | Added at `checkpoint_goals` milestones. | Curriculum stages don't use this. |
| `soft_waypoint_reward` | `25` | Per soft location-goal hit. | Waypoint bonus for non-curriculum goals. |
| `all_goals_bonus` | `500` (global) / `0` (curriculum) | Added on hitting `hard_goal_count_target`. | Set to `0` for early_completion_bonus-only terminals. |
| `early_completion_bonus` | `0` (global) / `150` (curriculum) | Added on final goal. | |
| `exploration_reward` | `0.0` (global) / `1` (curriculum) | Per first-visit `(x, y, map_bank, map_num)` tile. | Small values (`1`–`3`) for sparse-goal stages. |
| `new_map_reward` | `0.0` (global) / `50` (curriculum) | Per first-visit `(map_bank, map_num)` map. Preserved across replay. | Keep below `goal_reward + sequence_bonus` (150) so it doesn't overwrite the golden goal path. |
| `pokedex_seen_reward` | `50` (global) / `0` (curriculum) | Per new species seen. Configurable. | Set to `0` in curriculum to avoid training "see one and flee" behaviour. |
| `pokedex_owned_reward` | `150` | Per new species owned. Configurable. | Rarely changed. |
| `xp_goal_reward` | `100` | Per xp goal fire. | Reward per xp goal threshold crossed. |
| `xp_goal_threshold` | `10` | XP points per xp goal fire. | Each goal fire consumes this much XP from the cumulative total. |
| `battle_engagement_reward` | `0.0` (global) / `1.0` (curriculum) | Per step while `battle_type != 0`. | Set high enough to offset `\|step_penalty\|` so battles aren't net-negative. |
| `damage_dealt_reward` | `0.0` (global) / `0.5` (curriculum) | Per HP point of damage dealt to the enemy this step. | Scale to early-game HP totals (~20–50). A typical attack should be a noticeable but not dominant signal. |
| `step_penalty` | `-1` (global) / `-0.5` (curriculum) | Per-step penalty when `punish_steps`. | Scale to episode_length. Per-episode worst case should stay smaller than a goal reward. |
| `button_penalty` | `-5` (fixed) | Penalty for start/select presses. | Not configurable. |
| `party_level_reward` | `0` (global) / `10` (curriculum) | Per-level-increase reward for the party's total level. | Provides dense signal during battles without creating a navigation gradient. |
| `party_exp_reward` | `0` (global) / `0.01` (curriculum) | Per-EXP-point reward for the party's total EXP. | Seeded from step 0 of each training episode to avoid phantom rewards after replay. |
| `xp_milestone_threshold` | `0` (disabled) | XP points accumulated before a milestone fires. | 0 disables. Accumulator pauses on party size change (capture/swap) but is not reset. |
| `xp_milestone_reward` | `0` | Lump-sum reward per milestone threshold crossed. | Prevents false triggers from new Pokemon's existing EXP. |
| `party_reward_check_battle` | `false` | Allow party progress rewards on size change when in battle. | Disabled by default until timing verified. |
| `distance_shaping_coef` | `0` (global) / `0.5` (curriculum) / `1.5` (stages 3+) | Potential-based distance-to-goal shaping coefficient. | Must exceed `\|step_penalty\|` to produce net-positive signal for correct movement. |
| `entropy_plateau_reset` | `true` (curriculum) | Enable plateau detection and entropy rewind. | |
| `entropy_reset_window_fraction` | `0.1` | Rolling window as fraction of total budget. | |
| `entropy_reset_min_fraction` | `0.1` | Minimum budget fraction before detection activates (faster recovery). | |
| `entropy_reset_debounce_fraction` | `0.125` | Cooldown between resets as fraction of budget. | |
| `entropy_reset_rewind_fraction` | `0.1` | How far back to rewind as fraction of window. | |

---

## Reward System

`PoliwhiRL/environment/rewards.py`. See **Goal & reward semantics** above for the formula, list-vs-target rules, multi-fire pokedex goals, and the actual curriculum goal list.

`Rewards.get_current_target_vector()` exposes the active goal's primary `[x, y, map, map_bank]` and a `has_active_target` flag — used by the env to populate indices 16–20 of the RAM observation, which is how the policy is goal-conditioned.

---

## Save-state Pool

`VecPyBoyEnv` accepts a list of save-state files via `config["state_paths"]`. Workers are assigned round-robin at init (`worker i` → `state_paths[i % len(pool)]`). Each worker's env boots with its assigned state.

**Per-episode cycling** is driven by the agent (vec mode only). After an env completes an episode:

1. Record the metric tagged with the state that was just used.
2. Pick the next state for that env according to `state_cycle_strategy`: `random` uniform-samples from the pool; `none` keeps the assignment fixed.
3. Send the new state to the worker via `set_env_state_index`. The worker queues it and applies it on its **next auto-reset** — so the change takes effect one episode later than the cycling decision. The agent tracks this with `env_state_indices` (currently running) and `env_pending_state_indices` (queued).

### Per-state metrics

`episode_data["episode_state_indices"]` is a list parallel to `episode_rewards`, recording the state-pool index each completed episode actually used. The training-metrics JSON output includes this list **and** a `per_state_summary` block with mean / max / min reward and episode count per state — for offline analysis grouped by starting state.

### Goal-list alignment caveat

When you add a save-state whose player position is on an existing goal tile, the agent gets a free reward on step 0 (the reward calculator starts with `current_goal_index=0` every reset). This is the main reason **action replay** (next section) is the preferred mechanism for cross-stage continuation — replay walks the Rewards object forward so the curriculum starts at the right point automatically.

---

## Action Replay (the preferred chaining mechanism)

Save-states snapshot emulator memory but leave the `Rewards` object's curriculum tracking (`current_goal_index`, `N_goals`, `pokedex_*`) reset to 0 on every `env.reset()`. The cleanest fix is to **replay the action sequence** that produced the curriculum endpoint — `Rewards` walks forward through the goals as the actions execute, so the post-replay state is *natively correct* without any goal-list curation.

### File format

A `.steps` file supports two formats:

1. **Single-trajectory**: plain text, one integer action per line. Comments (`#`-prefixed) and blank lines are tolerated.
2. **Multi-trajectory**: trajectory blocks delimited by `# trajectory N` markers. Each block contributes one trajectory to the pool. Written by checkpoint saves.

```
# trajectory 0
# length=256
4
4
2
1
...
# trajectory 1
# length=312
1
3
...
```

Loaded by `_load_actions_file(path)` in `environment/vec_env.py`. Missing files emit a warning and become a no-op (replay disabled for that pool entry).

### Replay invariant

After every `env.reset()` (explicit or auto-reset on done), if a replay sequence is attached the worker calls `env.replay_actions(actions)`:

1. Steps the env through each action, calling `_handle_action` + `_calculate_fitness`. `Rewards.current_goal_index`, `N_goals`, `pokedex_*`, `explored_tiles`, `_d_prev` all advance naturally.
2. After the sequence, calls `Rewards.start_new_episode()` to clear *per-episode counters* (`steps`, `cumulative_reward`, `done`, `last_action`) while **preserving curriculum state** (`current_goal_index`, `N_goals`, `pokedex_seen/owned`, `pokedex_goals_completed`, `level_goals_completed`) **and exploration state** (`explored_tiles`, `_d_prev`). The exploration set is deliberately kept so re-walking replay-visited tiles doesn't pay a fresh `exploration_reward`.
3. Resets `env.steps = 0` and `env._fitness = 0` so the training episode starts with a clean step budget.

The training agent never sees the replay transitions — they're not stored in the rollout buffer and don't count toward `ep_returns` / `ep_lengths`. The post-replay observation is treated as step 0 of the training episode. The RAM-vector `explored_tile_count` (feature 21) carries the replay's exploration size into the agent's input. The `n_location_goals_completed` (feature 22) and `n_pokedex_goals_completed` (feature 23) carry curriculum progress.

### Replay pool — sampling strategies

**Vec workers**: sample trajectory uniformly, then cutoff with **quadratic bias** toward later indices. P(k) ∝ (k+1), giving roughly 2/3 of samples in the upper half of the trajectory. Per-worker RNGs are seeded per `env_idx` for reproducibility.

**Single-env agent**: samples trajectory uniformly and cutoff **uniformly** in `[0, len(traj)]`. This provides wider distribution of starting states rather than always taking over at the same post-replay endpoint.

There is no per-env cycling state — diversity comes from the random sampling itself.

### Best-actions dump

Both agents track `best_episode_reward` (the highest single-episode reward seen across the run) and the action history of every in-flight episode. When a new single-episode reward record is set, the agent writes that episode's action sequence to `<output>/Checkpoints/actions.steps`. The next curriculum stage's `action_replay_paths` references this file:

```json
"load_checkpoint": "./Training Outputs/third_steps/Checkpoints/best",
"action_replay_paths": [
    "./Training Outputs/third_steps/Checkpoints/actions.steps"
]
```

The replay file is written in multi-trajectory format at each checkpoint save, containing however many trajectories accumulated since the previous checkpoint. It is overwritten each time. If you want to preserve historical best paths, copy them out manually.

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
configs/stages/first.json    →  extends (2 goals, 40 steps, 150 rollouts)
configs/stages/second.json   →  extends + checkpoint load from first + action replay
configs/stages/third.json    →  extends + 5 location goals + owned:1 (stops at pokemon acquisition, target 6)
configs/stages/fourth.json   →  extends + 7 location goals + owned:1 + distance shaping 1.5 (target 7)
configs/stages/fifth.json    →  extends + vec (16 envs) + pokedex seen:3 + distance shaping 1.5
configs/stages/sixth.json    →  extends + vec (16 envs) + pokedex seen:4 + exploration 2
configs/stages/seventh.json  →  extends + vec (16 envs) + pokedex seen:4 + level:3 + exploration 2
```

Each stage sets its own `load_checkpoint` to the previous stage's `Checkpoints/best/` (peak) or `Checkpoints/` (latest). On load:

- Encoder + transformer weights persist.
- Entropy schedule resets to its initial value (`stage_start_episode = self.episode`, `set_entropy_offset(0)`).
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
    actions.steps                        # multi-trajectory replay file (written at each checkpoint)
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

The `info.pth` contains: `episode`, `best_reward`, `episode_data` (including `episode_state_indices`, `episode_goals_total`, `episode_goals_made`, `episode_goals_target`), and `reward_scaler` state.

---

## Running

```bash
# Defaults
python main.py

# Stage configs (using the extends mechanism)
python main.py --use_config configs/stages/first.json
python main.py --use_config configs/stages/second.json
python main.py --use_config configs/stages/third.json
python main.py --use_config configs/stages/fourth.json

# Stage 5 (vec mode, num_envs=16)
python main.py --use_config configs/stages/fifth.json

# Stage 6 (vec mode, num_envs=16, longer episodes)
python main.py --use_config configs/stages/sixth.json

# Stage 7 (vec mode, num_envs=16, level goals)
python main.py --use_config configs/stages/seventh.json

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

## Diagnosing Training Regressions

A practical playbook for "the model used to train fine, now it doesn't" — using the artefacts already on disk under `Training Outputs/`.

### Folder layout

```
Training Outputs/<stage>_steps/
├── Checkpoints/                         # model weights, info.pth, actions.steps
├── Results/
│   ├── training_metrics.png             # full history (all stages combined)
│   ├── training_metrics_current.png     # current stage only
│   └── metrics/
│       ├── training_metrics.json        # numeric arrays for the full plot
│       └── training_metrics_current.json
└── Runs/                                # recorded gameplay videos/frames
```

The `*_current.*` files are the slice for the running stage. Plot PNGs are great for "shape of the problem"; JSON files are where the actual numbers live.

**The single most valuable diagnostic asset is a good baseline to compare against.** Keep a working snapshot (e.g. rename to `Training Outputs_old/` before a risky experiment) — without it you are guessing.

### Diagnostic flow

**1. Establish: "what changed and when"**

```bash
git log --oneline -20
git log --all --oneline -- '*<suspect-keyword>*' | head
git diff <prev>..<sha> -- <files>        # what actually changed
```

Look for default value changes in `configs/`, formula changes (exponential → linear, multiplicative → additive), and index/denominator changes (per-episode vs per-rollout vs per-env).

**2. Compare plots side-by-side**

Open `training_metrics_current.png` from the baseline and the current run. Read panels in order:

| Panel | Healthy | Broken |
|---|---|---|
| Episode Rewards | Steady upward trend, narrowing variance | Flat / noisy / declining; wide variance late |
| Training Loss | Bumpy early, settles low | Spikes mid-run, diverges, or flat near 0 |
| Button Presses | One or two dominant buttons (e.g. `a`) | Heavy skew to navigation (`right`/`up`), almost no `a` |
| Episode Steps | Drops over time (efficient play) | Pinned at `episode_length` (timing out) |
| Entropy Coefficient | Smooth monotonic decay over whole run | Drops to floor early; floor too low; discontinuities |
| Button Diversity | Stable around 6–9 unique buttons | Collapsing to 3–4 |
| Goals at Episode End | Climbs to and saturates at target | Plateaus below target |
| Curriculum Completion Fraction | Reaches ~1.0 quickly | Stuck around 0.5–0.8 |

**3. Get hard numbers from the JSON**

Compare `summary` fields in `training_metrics_current.json`:

```
                           old (good)       current (broken)
last100_mean_reward:       814              76
last100_mean_goals_total:  4.0              2.01
last100_mean_completion:   1.0              0.50
mean_episode_length:       78               257
current_entropy:           0.005            0.001
```

Walk parallel arrays (`rewards[]`, `entropies[]`, `episode_steps[]`, `goals_total[]`) — sample `first 3` and `last 3` of each. The `button_counts` array is the smoking gun for policy collapse.

**4. Confirm causation with a small simulation**

Reproduce the suspect formula in isolation:

```python
# Old vs new entropy over the actual episodes run
ep_count = 752
def old_coef(ep):
    return max(0.05 * (0.999**ep), 0.005)
def new_coef(ep):
    p = min(ep / 250, 1.0)
    return 0.02 * (1-p) + 0.001 * p

import statistics
print(statistics.mean(old_coef(e) for e in range(ep_count)))  # 0.0352
print(statistics.mean(new_coef(e) for e in range(ep_count)))  # 0.0042 = 12% of old
```

If the simulated delta is large enough to explain the regression, you have your culprit. If small, the cause is elsewhere.

**5. Check for compounding axes**

A single regression rarely lives alone. Once you find one cause, check for related ones:

- If a formula changed, did its *denominator* still mean the same thing?
- If a safety net was added, does its trigger actually fire on real data?
- Did any related config defaults change in the same commit?

### Red-flag patterns

- **Button distribution dominated by one navigation key, `a` near zero** → policy collapsed; almost always an exploration problem (entropy too low, schedule too aggressive, plateau detector not firing).
- **Episode Steps pinned at `episode_length`** → agent never finishes — stuck wandering.
- **Entropy hits the floor at <30% through training** → schedule denominator too small for actual episode budget. Common when a per-rollout schedule is applied in vec mode where episodes ≫ rollouts.
- **Reward variance is high but mean is low and flat** → policy found a local optimum and is wiggling. Stagnation detectors using coefficient-of-variation will silently fail here.
- **Loss curve dives near zero almost immediately** → policy too confident too fast. Look for entropy / KL / clip changes.

### Tooling shortcuts

```bash
# Find recent commits that touched a concept
git log --all --oneline -- '*<keyword>*'

# Diff just the agent + model files between two commits
git diff <a>..<b> -- PoliwhiRL/agents/PPO/ PoliwhiRL/models/PPO/ configs/

# Stat the outputs directory to confirm a stage actually ran
ls -la "Training Outputs/<stage>/Results/metrics/"
```

### When you find the bug

1. Re-read the suspect commit diff *in full* — don't fix only what you noticed first. Compounding regressions are the norm.
2. Sanity-check that the baseline run wasn't a special case (different config, stage, or `num_envs`).
3. Write the fix to be robust across configurations, not just the one in front of you. Schedules should index by something that scales with `num_rollouts`, not by raw episode counts.

---

## Future ideas (not yet implemented)

Things considered while diagnosing the stage 7 non-fighter regression but deferred — capture here so they aren't re-derived next time:

- **Mid-battle save-states / battle-targeted action replay.** The current curriculum forces the policy to walk ~30 tiles into Route 29 before any battle signal can fire. Hand-recording a `.steps` file (or save-state) that ends *inside* the battle menu, and mixing it into `state_paths` / `action_replay_paths`, would give the policy direct exposure to the battle UI with a short credit-assignment horizon. Likely needed to make the new `damage_dealt_reward` actually do work — without battle exposure, the gradient never lands.
- **Higher entropy floor for stages loading the stage-6/7 checkpoint.** Stage 7 hit `current_entropy = 0.0059` (floor 0.005). Loading those weights into a new stage with the same floor lets the existing flee-policy stay deterministic. Raising `ppo_entropy_coef_min` to ~0.015–0.02 and extending `ppo_entropy_anneal_steps` would give the policy enough noise to escape the local optimum and try fighting at all.
- **`enemy_hp` as a policy input.** Currently `enemy_hp` is read in `RAM.py` for the reward calc but is *not* appended to `RAM_FEATURE_KEYS`. Adding it would let the policy condition on "the enemy is nearly dead" — but would break checkpoint loading (changes `RAM_OBS_DIM` and the RAM encoder's input width). Worth doing for the next curriculum stage that starts from fresh weights.

---

## Implementation Notes for Future You

These are the invariants new code has to maintain:

- **Observation contract:** envs return `{"image": ndarray, "ram": ndarray}`. Vec wrappers stack into a dict-of-batched-arrays. The model takes `(x_image, x_ram, mems)`. Any new modality goes into the dict — don't tunnel it through scalars or string-encoded fields.
- **RAM vector contract:** order is fixed by `RAM_FEATURE_KEYS`. Only ever append. Loaded checkpoints depend on the order. `RAM_OBS_DIM = len(RAM_FEATURE_KEYS)` adjusts automatically when `_DERIVED_FLAG_TABLE` grows.
- **Bootstrap:** any new return computation must use V(s_{T+1}) when the tail isn't terminal. See `_tail_bootstrap_value` and the per-env path in `VecPPOAgent._per_env_gae`. The RAM equivalent is `last_next_ram` — pass it alongside `last_next_state`.
- **Mems-on-done:** zero mems when `done=True`. Refill state_sequence and ram_sequence with the post-reset obs (broadcast to `seq_len`).
- **Metric arrays:** `episode_data["episode_losses"]` is **per-PPO-update** (multiple per episode in single-env, one per rollout in vec). Other arrays in there are **per-completed-episode**. Plot axes reflect this in `visuals.py`.
- **`num_episodes` is gone.** The training budget is `num_rollouts` everywhere. `self.episode` is still a running counter for entropy schedule, recording cadence, and metrics — never a stopping criterion.
- **Vec env workers are torch-free.** Don't import torch inside the worker — it inflates startup and breaks on some platforms.
- **Save-state pool indexing:** the agent's `env_state_indices[i]` always reflects the state currently running in env i. Cycling updates `env_pending_state_indices[i]` first; when the next done occurs, the agent promotes `pending → running` and tags that episode's metric. This lag-by-one is intentional and accurate.
- **Action-replay pool:** no per-env cycling — each worker samples (trajectory, cutoff) independently on every reset. Vec workers use quadratic-biased cutoff; single-env uses uniform cutoff. The agent never trains on replay transitions; they're a pre-episode warm-up only.
- **Multi-trajectory `.steps` format:** trajectory blocks delimited by `# trajectory N` markers. Legacy single-trajectory format (int-per-line, `#` comments) is still supported by the loader. Written by checkpoint saves in multi-trajectory format.
- **Story flags are read-only.** We extract 0xDA72–0xDB71 (256 bytes) for bit extraction into `_DERIVED_FLAG_TABLE`. Never write to this region.
- **Reward scaler:** `RewardScaler.observe(rewards, dones)` must be called after every env step in both modes. State is persisted in `info.pth`.
- **Strict loads:** `actor_critic.load_state_dict` and `scheduler.load_state_dict` are strict. If you change layer names or scheduler type, expect existing checkpoints to fail to load — and that's the desired behaviour (the user has not asked for backward-compat tolerance).
- **Entropy schedule is linear annealing.** The old multiplicative decay (`coef * decay^episode`) is still in config for compatibility but the actual schedule uses `_get_entropy_coef()` with linear interpolation and an offset for plateau rewind. The schedule indexes by `step` — a training-progress counter meaning rollout index in vec mode, episode index in single-env — so the denominator matches `num_rollouts` in both modes. The plateau detector fires on goals-stagnation alone (flat goal count in the window, below target); the previous reward-CV condition was removed. The `ppo_entropy_anneal_steps` key controls the horizon.
- **Pokedex goals multi-fire.** A threshold of N contributes N goal slots. The `pokedex_goals_completed` counter and `_pokedex_goal_progress` dict track per-type progress independently from the mutable `pokedex_goals` dict (which is consumed as thresholds are reached).
- **Level goals multi-fire.** `{"total_level": N}` fires N times as the party gains N levels from the starting point. The `level_goals_completed` counter tracks total fires. Suppressed on party size change (new Pokemon's existing levels would inflate the total). The `_level_starting_total` baseline is re-seeded when party size changes.
- **Exploration key includes map_bank.** The exploration tile key is `(x, y, map_bank, map_num)` to prevent collisions between maps with the same number in different bank groups.
- **`explored_maps` preserved across replay.** Like `explored_tiles`, the `(map_bank, map_num)` set survives `start_new_episode()` so the new-map bonus is only credited for maps the *training* episode discovered, not maps the replay already visited.
- **Damage-tracker reset is two-way.** `_prev_enemy_hp` clears (a) whenever `battle_type` returns to 0 (so the next battle's first reading isn't compared to a stale HP from the last one) and (b) in `start_new_episode()` (so replay-time damage cannot be re-credited). The very first in-battle step seeds the tracker without paying damage — only subsequent HP drops count.
- **Garbage-RAM guards.** During the few frames at the start of a wild battle the game writes through battle / overworld memory in pieces, so a `ram.get_variables()` call landing in that window can return junk for parts of WRAM (we have observed `battle_type ∈ {122, 255}`, `player_state=50`, `x=y=map=bank=0`). Three layers of defence:
  1. `get_party_info` clamps `num_pokemon` to 6 so the party loop cannot walk past the real party area into unrelated WRAM (each junk slot was contributing up to 16M of fake exp via the 24-bit exp read, blowing `party_exp_reward × Δexp` through the 1000 reward clip in a single step).
  2. `is_ram_state_valid(env_vars)` (module-level helper in `rewards.py`) detects junk snapshots via the two signatures we've observed: `battle_type ∉ {0, 1, 2}` or `player_state ∉ {0, 1, 2, 4}`; and `x=y=map_num=map_bank=0` simultaneously. `Rewards.calculate_reward` gates the macro+micro paths on this — junk frames pay only `step_penalty` (+ `button_penalty` if applicable) and don't touch `explored_tiles`, `explored_maps`, `pokedex_seen/owned`, or the prev trackers. `gym_env.save_step_img_data` uses the same check to skip writing PNGs for junk frames.
  3. `_check_party_progress` and `_battle_engagement_reward` keep their own per-method `battle_type ∉ {0, 1, 2}` short-circuits as defense-in-depth — redundant with the top-level gate today, but they protect those specific paths if reward dispatch is ever refactored.
- **Goal matching includes map_bank.** Location goal entries can specify an optional 5th element (`map_bank`). When present, matching requires both map_num and map_bank to match. The `check_bank` flag on each goal entry is set if any option specifies a bank.
