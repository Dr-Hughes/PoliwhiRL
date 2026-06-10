# AGENTS.md — PoliwhiRL technical reference

Working reference for the PPO agent that plays Pokémon Crystal via PyBoy. This
describes the code as it actually is. If something here disagrees with the
code, the code wins — fix this file.

---

## 1. Entry point & dispatch

`main.py`:

1. `load_default_config()` merges every `configs/default_configs/*.json` into one
   dict (output paths in `outputs_settings.json` are rewritten relative to
   `output_base_dir`).
2. `load_user_config(path)` loads the `--use_config` file, following `"extends"`
   chains (parent merged first, child overrides). Circular chains raise.
3. `merge_configs()` overlays user config on defaults and remaps any default
   value that was `output_base_dir`-relative to the new base.
4. CLI flags (`--key value`) override everything.
5. Dispatch on `config["model"]`:
   - `PPO` → `setup_and_train_PPO` (single-env or vectorised, chosen by `num_envs`)
   - `inference` → `run_inference` (greedy playthrough)
   - `debug_eval` → `run_debug_inference` (per-frame RAM/byte-window dumps)
   - `reward_eval` → `evaluate_reward_system`
   - `explore` → `memory_collector`
   - `random_walk` → `random_walk_map_discovery`

`num_envs > 1` selects `VecPPOAgent` (rollout-based); otherwise the single-env
episode-based `PPOAgent`. The curriculum stages all use `num_envs: 16`.

---

## 2. Environment & step semantics

`environment/gym_env.py :: PyBoyEnvironment`

- One step = `_handle_action` presses a button (held `button_hold_frames=15`)
  then ticks the emulator to `frames_per_action=90` total. So one RL step ≈ 90
  emulator frames.
- `actions = ["", "a", "b", "left", "right", "up", "down", "start", "select"]`
  (9 discrete). `ignored_buttons` (default `["", "start", "select"]`) are not
  physically pressed but still consume frames.
- Observation is a dict: `{"image": (C,H,W) uint8, "ram": (RAM_OBS_DIM,) float32}`.
  Image is the screen (RGB or grayscale, scaled by `scaling_factor`).
- `step()` returns `(obs, reward, done, truncated)`. `done` and `truncated` are
  owned by the `Rewards` object (see §4).
- `reset()` reloads the save-state, rebuilds the `Rewards` object (sharing the
  persistent `VisitArchive`), runs one no-op startup step, and returns the obs.

### Vectorised env (`environment/vec_env.py`)

- `spawn`-based worker processes, one `PyBoyEnvironment` each, auto-reset on
  done. The terminal observation is lost to auto-reset, so the worker snapshots
  terminal progress into `terminal_info` (`n_flag`, `n_pokedex`, `n_map`,
  `flag_fires`, `unique_cells`, `unique_maps`, `archive_size`,
  `reward_breakdown`, `truncated`) for the agent to log.
- **Action replay pool**: trajectories from all `action_replay_paths` files are
  flattened into one pool. On each (auto-)reset a worker samples one trajectory
  and a prefix cutoff `k` (quadratic bias toward later `k`), replays `traj[:k]`
  without storing transitions, and the training episode begins from there.

---

## 3. Observation: the RAM vector

`gym_env.py :: _build_ram_vector` / `RAM_FEATURE_KEYS`. Treat the ordering as a
**contract** — append-only; reordering invalidates trained models. The model
reads its RAM input dim from `RAM_OBS_DIM` at startup.

- Base scalars (all ~[0,1] normalised): position (x, y, map_num, map_bank, room,
  warp), party (level, hp, exp), money, pokédex seen/owned, 4 collision bytes,
  exploration summary (explored-tile count, maps-visited-this-episode), three
  in-episode goal-progress counters, `battle_type` one-hot (none/wild/trainer),
  badges, player_state, key-item count, game hour, bgm, enemy HP (log1p), 4 move
  PP values, and one-hot buckets for the verified script/UI state bytes
  (`0xD438`, `0xCF07`, `0xD43D`).
- Derived story flags: `_DERIVED_FLAG_TABLE` extracts ~50 curated event-flag bits
  (verified against `pret/pokecrystal` `event_flags.asm`, vendored as
  `environment/event_flags.asm` and checked by `tests/test_event_flags.py`).
  **`got_starter` (flag 26) is transient** — the script sets then clears it — so
  it is never used as a goal terminal; use `pokedex_owned` instead.

`N_FLAG_GOALS_RAM_IDX` / `N_POK_GOALS_RAM_IDX` index the in-vector progress
counters; the vec agent reads them to compute per-episode `goals_made`.

---

## 4. Reward system

`environment/rewards.py :: Rewards.calculate_reward`. Per-step reward, clipped to
`±reward_clip` (1000), optionally rounded to `reward_round_dp`. Reward only
accrues when `is_ram_state_valid` (rejects transitional / zeroed RAM snapshots).

Reward is split into two streams — **extrinsic** (directed milestones) and
**intrinsic** (exploration / battle outcome / levelling) — normalised
independently by the agent and recombined with weights (see §4a).

| Term | Stream | Config key | Default | Semantics |
|---|---|---|---|---|
| Story flag | ext | `flag_progress_reward` | 500 | per configured flag 0→1 **this episode** |
| Pokédex owned | ext | `pokedex_owned_reward` | 150 | per species newly owned |
| Pokédex seen | ext | `pokedex_first_sight_reward` | 10 | per fresh species seen |
| Key item | ext | `key_item_pickup_reward` | 5 | per new key item |
| Map goal | ext | `map_goal_reward` | 250 | reaching a configured `map` goal |
| Whiteout | ext | `whiteout_penalty` | −100 | one-shot on party HP > 0 → 0 |
| New-map first discovery | int | `new_map_first_discovery_reward` | 50 | `bonus/(global_entries+1)` on first entry this episode; global ledger NOT written during replay |
| New map (legacy flat) | int | `new_map_reward` | 0 | one-shot per (bank,map) per episode; off by default |
| Frontier novelty | int | `frontier_novelty_bonus` | 3.0 | **per-episode** `bonus/(visits+1)`, once per cell, gated on `script_active` |
| Battle entry | int | `battle_engagement_reward` | 3.0 | first battle **per map** this episode, × per-map decay |
| Battle win | int | `battle_win_reward` | 8.0 | first win **per map** (enemy HP→0, not whiteout/catch), × decay |
| Damage | int | `damage_dealt_reward` | 0.0 | OFF by default (was the farm vector); × Δenemy_hp × decay |
| Level up | int | `level_up_reward` | 10 | per Δparty_total_level (suppressed on party-size change) |

Total per-episode battle reward (entry + win + damage) is clamped to
`battle_reward_episode_cap` (default 30) — a decay-independent anti-farm backstop.

- **Flag/map goals only fire on a fresh transition during the training portion** —
  initial state is snapshotted on first check, so anything the replay already
  satisfied does not pay.
- **Frontier novelty is PER-EPISODE** (`_episode_visit_counts`, reset each
  episode; seeded with replay-visited cells). The intrinsic landscape is
  therefore identical every episode — **stationary**, so it cannot globally drain
  into a collapse, and the critic sees consistent returns for similar states.
  This replaces the old process-lifetime `VisitArchive` per-cell signal (which
  saturated and caused the mid-stage collapse, see §10).
- **`VisitArchive` now backs only the map-level first-discovery ledger**
  (`record_map`/`map_count`), written by genuine training entries but NOT during
  `replay_actions` (guarded by `Rewards._replaying`). The first-discovery bonus is
  the only non-stationary term — kept small/rare on purpose; zero
  `new_map_first_discovery_reward` if value-loss spikes.
- **Battle reward = winning, not damage.** First-entry-per-map engagement bonus +
  per-map win bonus; raw damage off by default; per-map decay
  `1/(1 + battle_decay_coef·n)` still applies; hard per-episode cap is the backstop.

### 4a. Two-stream reward scaling

`agents/PPO/vec_ppo_agent.py` keeps two `RewardScaler`s (`reward_scaler_ext`,
`reward_scaler_int`, `utils/running_stats.py`). Each stream is normalised by its
own running-return std (floored by `scaler_min_std`), then recombined as
`extrinsic_reward_weight·ext + intrinsic_reward_weight·int` (defaults 1.0 / 0.3).
Per-stream normalisation removes *magnitude* dominance (a 500 flag no longer
crushes the dense signal); the intrinsic weight removes *frequency* dominance, so
sparse milestones dominate exploration churn. The per-step split is produced in
`Rewards.calculate_reward` (`_last_extrinsic`/`_last_intrinsic`), surfaced via the
vec worker step, and stored in `VecPPOMemory.reward_split`. Both scalers are
checkpointed; reset per stage by `reset_reward_scaler_on_load` (default true).

---

## 5. Goals & termination

`environment/goals.py :: GoalsManager`. Goals are a per-stage list under
`"goals"`. **`terminate_on` is removed** (raises) — episodes run to
`episode_length` unless `terminate_on_goal_complete: true`, which ends the
episode the step every configured goal is satisfied.

Supported goal types:

- `pokedex` — `{kind: "seen"|"owned", threshold}` — multi-fires per increment.
- `flag` — `{flag_num}` — fires on 0→1 this episode.
- `map` — `{map_bank?, map_num}` — fires on entering that map during training.
- `maps_visited` — `{threshold}` — fires once per unique (bank,map) seen this
  episode, up to `threshold`; completion = `unique maps ≥ threshold`. Folded into
  `n_map_goals_completed()` so it surfaces through the existing metric plumbing.
- `level` — `{threshold}` — fires per total-level gained.
- `xp` — `{threshold, xp_per_fire}` — fires per XP chunk.

`all_goal_thresholds_met()` drives `terminate_on_goal_complete`; returns False
for an empty goal list (free play never "completes"). Every fire bumps `N_goals`
(a generic progress metric for plotting), not a hard checklist.

---

## 6. Curriculum

Five directed stages + one free-play stage, each `extends ../curriculum_base.json`.
Each stage loads the prior stage's `best/` checkpoint and replays its captured
`actions.steps`. Map IDs `(bank,num)` are verified against live RAM
(`RAM_MAPPING.md`); `map` goals fire on entering the map at **any** x/y.

| Stage | File | Goal | terminate | ep_len | rollouts |
|---|---|---|---|---|---|
| 1 leave house | `first.json` | `map (24,4)` New Bark outdoor | yes | 256 | 300 |
| 2 starter | `second.json` | `pokedex_owned ≥ 1` | **no** | 512 | 600 |
| 3 Route 29 | `third.json` | `map (24,3)` | **no** | 768 | 700 |
| 4 Cherrygrove | `fourth.json` | `map (26,3)` | **no** | 1024 | 900 |
| 5 Mr. Pokémon's | `fifth.json` | `map (26,10)` | **no** | 2048 | 1000 |
| 5b free-play | `freeplay.json` | none | no | 3072 | 1000 |

**Milestones are additive, not episode-enders (stages 2–5 do NOT terminate on
goal).** Terminating the instant the goal is reached makes the milestone a
*trade-off* against the rest of the episode's renewable per-episode novelty — and
since terminating forfeits that stream, the agent learns to *abandon* the goal and
wander instead (observed in a stage-2 run: success climbed to 1.0 then collapsed to
0 as episodes ran to full length, see §10a). Not terminating makes the milestone
strictly *additive* (`reach goal → +reward AND keep exploring`), so grabbing it
dominates at every state regardless of reward magnitudes — and it also removes the
sample-imbalance spiral (short success episodes vs long failures biasing the PPO
batch toward wandering). It also matches the project goal of exploring *beyond*
each milestone. `goal_success` (for `best/` selection) is still recorded at
truncation via `all_goal_thresholds_met()`. Stage 1 keeps `terminate=yes`: its
250-reward milestone over a short 256-step episode makes the forfeit negligible, and
its short success demos are cleaner warm-starts for stage 2's replay.

Stages 4–5 lower `battle_reward_episode_cap` to 15 to bias traversal over fighting.
Free-play disables entropy annealing (`ppo_entropy_anneal_enabled:false`) and uses a
constant LR (`ppo_lr_schedule:"constant"`) so the policy keeps exploring instead of
freezing, with `entropy_plateau_signal:"unique_maps"`. **Per-stage state pools must not start
the agent on that stage's target map** (start-on-target counts as "not achieved", so
it would never terminate / never count as success).

### Action replay & checkpointing (vec agent)

- `actions.steps` is captured from up to 2 post-checkpoint **goal-reaching**
  trajectories per env (`_capture_trajectory_post_checkpoint`, gated on
  `goal_success`; optional `replay_capture_max_len`). Degenerate trajectories are
  never laundered back into the pool.
- Hot-swap into the worker replay pool is gated (`replay_hot_swap`, default on) and
  only runs in directed stages (`n_goals_target > 0`), bounded to `replay_pool_max`
  (64) most-recent trajectories.
- `best/` is selected on **goal-success rate** over `best_success_window` (directed
  stages), on intrinsic exploration (unique-maps moving average) in free-play, or on
  mean reward only as a fallback until the first success exists
  (`_should_update_best`). The worker emits `goal_success` in `terminal_info`.

---

## 7. Model

`models/PPO/PPOTransformer.py`

```
image (C,H,W) → GameBoyCNN (2 GameBoyBlocks 16→32, Conv+GroupNorm) → (B*T, d_model=128)
ram   (RAM_OBS_DIM) → RAMEncoder (3-layer GELU MLP)               → (B*T, d_ram=64)
        concat → fuse Linear → (B*T, 128) → reshape (B, T, 128)
        + sinusoidal PositionalEncoding
        → 4 × TransformerXLBlock (MHA, heads=4, mem_len=64, FFN ×4 GELU, dropout 0.1)
        → last token (B,128) → fc_actor (softmax) + fc_critic
```

- **Memory is caller-managed**: `init_mems(B, device)` → per-layer `(B, mem_len, d_model)`
  zeros. Each forward returns updated detached mems; the agent carries them across
  rollout steps and zeroes the per-env slice on episode done.
- Orthogonal init: hidden gain √2, actor head gain 0.01 (near-uniform start),
  critic head gain 1.0.
- **Action mask**: `forward(..., action_mask)` adds `(mask-1)*1e9` to logits before
  softmax so blocked actions get ~0 probability. The same mask is reconstructed
  deterministically from stored `ram_states` at update time, so old/new log-probs
  stay consistent in the PPO ratio.

### Action mask (`environment/action_mask.py`)

Derived from the script/UI one-hots in the RAM vector (last frame):
dialog (script+text_box) → only noop/A/B; menu/walking → noop/A/B/directional;
`start`/`select` blocked while walking unless `allow_menus_walking`.

---

## 8. PPO training

`models/PPO/ppo_model_implementation.py`, `agents/PPO/vec_ppo_agent.py`.

- Loss = clipped surrogate actor + (clipped) value loss + entropy bonus. KL
  early-stop per epoch (`ppo_target_kl`, Schulman k3 estimator). Adam `eps=1e-5`,
  grad-norm clip 0.5, cosine LR schedule (peak → `ppo_lr_min`) over `num_rollouts`.
- **Entropy schedule** (`_get_entropy_coef`): linear `ppo_entropy_coef →
  ppo_entropy_coef_min` over the budget. curriculum_base uses 0.05 → **0.005**
  (decaying — a flat floor prevents the policy from ever committing). Plateau
  detection (`entropy_plateau_reset: true`) rewinds the schedule if goal progress
  stalls; guarded off when no goal has fired yet, and inert in free play
  (goals always 0).
- **Advantage normalisation**: `"rollout"` (default) normalises once across the
  full flattened `W·N` rollout; `"minibatch"` is avoided in sparse regimes.
- **GAE truncation handling**: at an episode boundary `V(s_{T+1})` is bootstrapped
  only when the done was a **truncation** (budget cut-off, `truncated=True`); a
  natural terminal (goal complete) is zeroed. The GAE accumulator resets at every
  boundary regardless. Truncation flags propagate env→worker→agent→storage.
- **Rollout buffer** (`replay/vec_ppo_storage.py`): stores per-step single frames
  in `(T,N,…)` and emits sliding windows of `sequence_length` at update time. ⚠️
  Windows can span an episode-done boundary (act-time sequences/mems reset on
  done, update-time windows don't) — a small bias on ~`seq_len-1` transitions per
  episode.

---

## 9. Key config knobs

Defaults live in `configs/default_configs/`; `curriculum_base.json` holds shared
stage defaults; stage files override per stage; CLI overrides all.

| Key | Where | Note |
|---|---|---|
| `num_envs` | stage | >1 → vectorised agent |
| `episode_length` | stage | step cap / truncation point |
| `num_rollouts` | stage | outer-loop iterations; also LR/entropy schedule length |
| `ppo_update_frequency` | stage | transitions per env per PPO update |
| `terminate_on_goal_complete` | stage | end episode when goals met |
| `goals` | stage | list of goal specs (§5) |
| `action_replay_paths` | stage | warm-start replay pool |
| `load_checkpoint` | stage | prior stage's `best/` |
| `ppo_entropy_coef[_min]` | base/stage | navigation 0.05→0.02; stage 2 0.03→0.005 |
| `ppo_entropy_anneal_enabled` | stage | false → constant entropy (free-play) |
| `ppo_lr_schedule` / `ppo_lr_min` | base/stage | `cosine`\|`constant`; floor 5e-5 (nav) |
| `entropy_plateau_signal` / `entropy_reset_max_count` | base/stage | `goals`\|`unique_maps`\|`archive_size`; cap 3 |
| `frontier_novelty_bonus` / `new_map_first_discovery_reward` | base | intrinsic exploration |
| `battle_win_reward` / `damage_dealt_reward` / `battle_reward_episode_cap` | base/stage | win-based battle reward + cap |
| `extrinsic_reward_weight` / `intrinsic_reward_weight` / `scaler_min_std` | base | two-stream scaler (§4a) |
| `best_success_window` / `replay_hot_swap` / `replay_pool_max` / `replay_capture_max_len` | base | success-based best/ + replay pipeline (§6) |
| `action_mask_enabled` / `allow_menus_walking` | base | action masking |
| `reset_lr_scheduler_on_load` / `reset_optimizer_on_load` / `reset_reward_scaler_on_load` | base | curriculum-transition resets |

---

## 10. Invariants & gotchas

- RAM vector order is a contract (append-only).
- `got_starter` flag is transient — never a terminal; use `pokedex_owned`.
- Flag/map goals don't fire for state already true at episode start (replay).
- Start-on-target is NOT auto-success: a state starting on a `map` goal's target
  never terminates / never counts as success. Per-stage state pools must avoid it.
- `VisitArchive` now backs only the map first-discovery ledger (not per-cell
  frontier); it persists across episodes within a stage and resets between stages.
- Frontier novelty is per-episode and stationary — it does NOT drain across a run.
- `best/` is selected on goal-success rate (directed) / intrinsic exploration
  (free-play), not mean reward (§6).

---

## 10a. Observed failure modes (fixed)

Diagnosed from the pre-rework training metrics and addressed by this rework:

- **Mid-stage collapse to wall-walking.** The process-lifetime per-cell
  `VisitArchive` saturated (and `replay_actions` pumped it every episode), so
  frontier reward decayed from ~240→~20 over a stage; the reward landscape
  collapsed underneath the policy → degenerate single action. **Fix:** per-episode
  (stationary) frontier novelty; replay no longer writes the archive.
- **Goal abandonment / reward farming.** In stages 2–3 the real goal paid ~0 while
  `new_map` (re-paid every episode) + battle damage dominated; best reward was a
  21,985-reward battle-farm episode. **Fix:** decaying first-discovery `new_map`,
  win-based capped battle reward, two-stream scaling so milestones dominate, and
  directed `map`/`pokedex` goals on every stage (stage 3 previously had none).
- **Corruption propagating across stages.** `best/` was the best *farmer* and
  degenerate trajectories were laundered into the replay pool. **Fix:**
  goal-success-based `best/`; capture only goal-reaching trajectories.
- **Premature convergence.** Entropy annealed to 0.005 and LR to 1e-5 long before
  the ~3k episodes/stage finished. **Fix:** higher entropy floor / constant entropy
  + constant LR in free-play; plateau detector that fires on exploration
  stagnation.
- **Terminate-on-goal forfeit (post-rework regression, fixed).** With per-episode
  (renewable) novelty + `terminate_on_goal_complete`, terminating to claim the
  milestone forfeits the rest of the episode's exploration reward, so the agent
  learned the goal (stage-2 success → 1.0) then *un-learned* it, drifting to
  full-length wandering (success → 0, frontier reward up, milestone reward → 0).
  Compounded by sample imbalance (short success episodes vs long failures). A
  magnitude fix (intrinsic cap) doesn't help because the two-stream scaler
  normalises by return-std and is mean-insensitive. **Fix:** directed stages 2–5
  do NOT terminate on goal — the milestone is an additive per-episode bonus
  (`reach goal → +reward AND keep exploring`), which dominates structurally
  regardless of magnitudes and removes the sample imbalance.

---

## 11. Tests

`pytest tests/ -v`. Fast NumPy/PyTorch tests cover model init, losses, GAE,
buffers, config inheritance, reward + goal calculation (`test_phase3_goals.py`,
including `maps_visited`), event-flag numbers (`test_event_flags.py`), whiteout
and rounding. Emulator tests spin up real PyBoy subprocesses.
