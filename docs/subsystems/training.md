# Subsystem: training

What this page is for: the training path end to end — rollout collection and the
rollout buffer's tensor contract, per-env GAE and the deliberate
bootstrap/truncation asymmetry, entropy control (including two knobs that are
currently dead), the PPO epoch loop and its two KL regimes, reward scaling, and
everything checkpoint-related: contents, cadence, best-model selection,
auto-resume and auto-gamma.

**Sources**: `PoliwhiRL/agents/PPO/vec_ppo_agent.py`,
`PoliwhiRL/agents/PPO/_minibatch.py`, `PoliwhiRL/replay/vec_ppo_storage.py`,
`PoliwhiRL/models/PPO/ppo_model_implementation.py`, `PoliwhiRL/PPO.py`,
`PoliwhiRL/environment/vec_env.py`, `PoliwhiRL/environment/rewards.py`,
`PoliwhiRL/utils/running_stats.py`, `PoliwhiRL/utils/visuals.py`,
`main.py`, `configs/default_configs/ppo_settings.json`,
`configs/curriculum_base.json`, `configs/stages/open_world.json`,
`tests/test_minibatch.py`, `tests/test_vec_ppo_memory.py`,
`tests/test_vec_ppo_correctness.py`.

---

## `VecPPOAgent` is the only training path

`PoliwhiRL/agents/PPO/ppo_agent.py` was **deleted** in commit `10766b3`
("Snapshot: pre-fix state of redo-configs branch"), together with
`tests/test_PPO.py` (`git log --diff-filter=D --name-only`). It is superseded by
`VecPPOAgent` (`PoliwhiRL/agents/PPO/vec_ppo_agent.py`), which is the only
agent the entry point constructs (`PPO.py:56` import, `PPO.py:77` construction).

The entry point states this outright: there is **ONE** training implementation,
and `num_envs == 1` just runs a single worker (`PPO.py:61-64`). Do not look
for a single-environment agent; there is only a worker count.

Shared update logic was factored out into `PoliwhiRL/agents/PPO/_minibatch.py`,
whose module docstring still names both agent classes as producers of the flat
data dict (`_minibatch.py:4-7`) — a leftover reference to the deleted class, not
a second implementation.

`PoliwhiRL/replay/ppo_storage.py` (`PPOMemory`, `:6-148`) is **not** deleted: it
is still exported from `PoliwhiRL/replay/__init__.py:2-5` and exercised by
`tests/test_ppo_memory.py`. The live path uses `VecPPOMemory` only
(`vec_ppo_agent.py:116`) — treat the single-env buffer as legacy.

---

## Rollout collection and storage

A rollout is `ppo_update_frequency` steps × `num_envs` envs
(`vec_ppo_agent.py:61` and `:60`, default 1). The buffer mirrors this exactly:
`rollout_length = ppo_update_frequency`, `sequence_length = sequence_length`,
`num_envs` (`vec_ppo_storage.py:20-22`). `is_full()` is `t >= rollout_length`
(`vec_ppo_storage.py:55-56`).

| Quantity | Key | Value | Line |
|---|---|---|---|
| Steps per rollout | `ppo_update_frequency` | `128` | `configs/default_configs/ppo_settings.json:5`; stage `configs/stages/open_world.json:15` |
| Segment length `L` | `sequence_length` | `16` | `configs/curriculum_base.json:2`; stage `open_world.json:16` |
| Envs `N` | `num_envs` | `8` | `configs/curriculum_base.json:3`; stage `open_world.json:9` |
| Epochs | `ppo_epochs` | `8` default, `1` in open-world | `ppo_settings.json:4`, `open_world.json:18` |
| Total rollouts | `num_rollouts` | `163840` (open-world) | `open_world.json:13` |

The main loop iterates `for self.rollout_idx in pbar`
(`vec_ppo_agent.py:596`), incrementing `total_rollouts` per rollout (`:635`).
The update is skipped while `_critic_warmup_remaining > 0` (`:612-615`) — see
[critic warmup](#critic-warmup).

### Buffer tensors

Allocated in `reset()` (`vec_ppo_storage.py:28-50`), `T = rollout_length`,
`N = num_envs`:

| Buffer | Shape | dtype | Line |
|---|---|---|---|
| `states` | `(T, N, *input_shape)` | uint8 | `:30` |
| `ram_states` | `(T, N, ram_obs_dim)` | float32 | `:31` |
| `actions` | `(T, N)` | int64 | `:32` |
| `rewards` | `(T, N)` | float32 | `:33` |
| `dones` | `(T, N)` | bool | `:34` |
| `truncated` | `(T, N)` | bool | `:35-37` |
| `terminal_bootstrap_values` | `(T, N)` | float32 | `:38-40` |
| `log_probs` | `(T, N)` | float32 | `:41` |
| `base_probs` | `(T, N, action_size)` | float32 | `:42` |
| `behaviour_probs` | `(T, N, action_size)` | float32 | `:43` |
| `mems` / `outgoing_mems` | lazy — shape depends on the model | — | `:44-45` |
| `next_states` / `next_ram_states` | `(T, N, ...)` | uint8 / float32 | `:46-47` |
| `last_next_obs` / `last_next_ram` | — | — | `:48-49` |
| `t` | cursor, starts `0` | int | `:50` |

Two fields carry the bootstrap decision, quoted from the source:
`truncated` is "True only where a done was a budget truncation (vs a natural
goal terminal). Drives the GAE bootstrap." (`vec_ppo_storage.py:35-37`), and
`terminal_bootstrap_values` holds `V(actual terminal continuation)` for those
budget dones and zero everywhere else (`:38-40`).

`store_step(...)` writes one timestep across all envs and raises if the buffer is
full (`vec_ppo_storage.py:58-114`, guard at `:76-77`). Per-layer memories are
stacked on axis 1 into `(rollout_length, layers, mem_len, d_model)` for both
incoming (`:94-99`) and outgoing (`:101-108`) mems; `last_next_obs` /
`last_next_ram` snapshot the newest next state (`:112-113`).

Action masks are **not** stored — they are recomputed from stored RAM, which is
deterministic (`ppo_model_implementation.py:65-68`, `:352-356`).

### Truncation plumbing during collection

`vec_ppo_agent.py:729-763`:

- `next_obs, rewards, dones, terminal_infos = vec_env.step(actions)` (`:729`)
- `reward_scaler.observe(rewards, dones)` (`:732`)
- `truncated = [bool(ti and ti.get("truncated")) for ti in terminal_infos]` —
  so `truncated` is only ever `True` where `dones` is `True`, and a missing or
  `None` info means non-truncation (`:734-741`)
- for truncated indices, the *preserved terminal observation*
  (`terminal_infos[i]["terminal_observation"]`, `:746`) is scored to produce
  `terminal_bootstrap_values[truncated_indices]` (`:742-763`)
- all of it is handed to `memory.store_step(..., truncated=...,
  terminal_bootstrap_values=...)` (`:898-912`)
- the per-episode cause is recorded separately as `episode_truncation_cause`
  (`:1601`, `:861`)

The flag originates in the worker, which sets `truncated` from
`rc.truncation_cause == "budget"` only (`vec_env.py:136-138`, written `:176`),
and snapshots the terminal observation before the auto-reset replaces it
(`vec_env.py:131-134`, `:222-224`).

### Segment partition

`get_data()` partitions the `T` steps per env into `S = T // seq_len`
**non-overlapping contiguous** segments of length `L = seq_len`, dropping the
remainder tail (`vec_ppo_storage.py:116-151`). Shapes are documented at
`:128-136` and built at `:183-209`:

| Key | Shape | Line |
|---|---|---|
| `states` | `(S, N, L, C, H, W)` | `:129` |
| `ram_states` | `(S, N, L, D)` | `:130` |
| `actions` / `rewards` / `dones` / `truncated` / `terminal_bootstrap_values` / `old_log_probs` | `(S, N, L)` | `:131-132` |
| `old_base_probs` / `old_behaviour_probs` | `(S, N, L, A)` | `:194`, `:195-197` |
| `mems` | list of `(S, N, mem_len, d_model)` — segment-initial memory, taken at `mems[s*L]` | `:133`, `:164-174` |
| `tail_mems` | list of `(N, mem_len, d_model)` — memory that came OUT of the final included observation, for the `V(s_T)` bootstrap | `:134`, `:175-181` |
| `last_next_obs` / `last_next_ram` | `(N, ...)` | `:200-205` |
| `S`, `N`, `L` | ints | `:206-208` |

Because segments are contiguous in time within an env, the agent can run per-env
GAE over the full `T` timeline before slicing into segments
(`vec_ppo_storage.py:124-126`).

---

## Per-env GAE and the bootstrap asymmetry

`_per_env_gae` — `vec_ppo_agent.py:1742-1815`. `use_gae` requires
`gae_lambda > 0` (`vec_ppo_agent.py:66-67`; `ppo_gae_lambda` `0.95`
`configs/curriculum_base.json:6`).

```
not_done  = ~dones                                            # vec_ppo_agent.py:1780
if truncated is None:  bootstrap = not_done               # :1783-1784
else:                  bootstrap = clamp(not_done + trunc, max=1)   # :1785-1787
next_value = values[t+1]  (tail_values at the tail)       # :1791
next_value = where(truncated[t], terminal_bootstrap_values[t], next_value)  # :1792-1795
delta      = r + gamma * next_value * bootstrap[t] - V(s)  # :1796
gae        = delta + gamma * gae_lambda * not_done[t] * gae # :1797
returns    = advantages + values                            # :1799
```

The non-GAE branch mirrors the terminal-override (`:1804-1808`). Because the
accumulator is multiplied by `not_done[t]`, advantages never leak across episode
boundaries (`:1797`; stated `:1766-1768`). When `truncated` is `None`, every
done is a zero-bootstrap terminal — the conservative classical behaviour
(`:1770-1771`).

### Why `budget` bootstraps and `stagnation` does not

This is a deliberate asymmetry, decided upstream in the reward calculator and
carried through the `truncated` flag.

| Cause | Where decided | Bootstrap | Why |
|---|---|---|---|
| `budget` | `rewards.py:835-843` (`steps > max_steps`); `episode_length` `24576` (`open_world.json:14`) | **YES** | The episode was still progressing, so the value must cover the unobserved continuation. `truncation_cause == "budget"` is what sets the flag (`vec_env.py:136-138`) and `terminal_bootstrap_values` supplies `V` of the preserved terminal observation (`vec_ppo_agent.py:1792-1795`). |
| `stagnation` | `rewards.py:845-863`; `stagnation_truncation_steps: "auto"` (`curriculum_base.json:23`) → `max(256, max_steps // 16)` (`rewards.py:322-324`) | **NO** — zero-bootstrap terminal | Bootstrapping made the critic value a stuck state at ~`V`(post-reset), so a dead pocket looked like a cheap route to a high-value reset with zero advantage in leaving it (the ep_6395 diagnosis) — `rewards.py:849-857`, echoed `vec_ppo_agent.py:1754-1760`. |
| `battle_stagnation` | `rewards.py:865-881`; `battle_stagnation_truncation_steps: "auto"` (`curriculum_base.json:24`) → `max(512, max_steps // 16)` (`rewards.py:355-357`) | **NO** | Same reasoning; backstop for mashing A in one unwinnable battle (observed 36,202 consecutive battle steps) — `rewards.py:871-874`, `:336-337`. |

Both stagnation causes report `truncated = False` even though `env.step`'s own
`truncated` bool is `True` — the worker's comment spells this out verbatim
(`vec_env.py:167-176`). A natural goal terminal is likewise a zero-bootstrap
terminal (`vec_ppo_agent.py:1764-1766`). Setting either stagnation key to `0`
or `None` disables that truncation (`rewards.py:845-863`, `:865-881`).

---

## Entropy coefficient paths — and two dead knobs

Three ordered paths decide the coefficient
(`ppo_model_implementation.py:286-308`, detailed on
[models.md](models.md#entropy-coefficient-model-side)):

1. adaptive servo override (`:300-302`)
2. anneal disabled → constant (`:303-304`)
3. linear decay (`:305-308`)

`step` is the agent's rollout counter — the episode index in single-env mode —
(`ppo_model_implementation.py:287-293`); call sites pass `self.rollout_idx`
(`vec_ppo_agent.py:1736`; logged at `:1341`, `:1627`). The entropy loss is
computed on the **base** policy, `-coef * mean(H(base_probs))`
(`ppo_model_implementation.py:397-406`).

> ### Gotcha — stated plainly, this is current behaviour
>
> **The adaptive entropy servo is inert, and `entropy_plateau_reset` is a dead
> knob.**
>
> - The servo is only ever touched at checkpoint-load time:
>   `if bool(self.config.get("entropy_servo_enabled", True)):
>   self.model.set_entropy_coef(self._entropy_coef) else:
>   self.model.set_entropy_coef(None)` — `vec_ppo_agent.py:2087-2093`.
> - There is **no** adaptive-servo control loop anywhere in the training loop. A
>   repo-wide grep for `_adaptive_entropy_coef` matches only three lines, all on
>   the model side: the field declaration `ppo_model_implementation.py:33`, the
>   read `:300`, and the setter `:327`.
> - `configs/curriculum_base.json:11` sets `entropy_servo_enabled: false`, and a
>   repo-wide grep shows that is the **only** config file which sets the key at
>   all. So loading a checkpoint into a fresh stage clears the override
>   (`set_entropy_coef(None)`) and the time-based schedule runs.
> - `entropy_plateau_reset` appears in exactly one place in the entire repo —
>   `configs/curriculum_base.json:12` — and **no code reads it**.
> - An in-code comment confirms the removal: "the entropy servo and the
>   plateau-triggered boosts were removed" — `vec_ppo_agent.py:189-192`.
>
> Consequence: with `ppo_entropy_anneal_enabled: false`
> (`configs/curriculum_base.json:10`, `ppo_settings.json:11`) the effective
> behaviour today is path 2 — a **constant** `ppo_entropy_coef` of `0.01`
> (`curriculum_base.json:9`, `ppo_settings.json:8`) for the whole run. The decay
> in path 3 and the servo in path 1 are both unreachable in a default run.

`set_entropy_offset(offset)` still rewinds the schedule via
`_entropy_reset_offset` (`ppo_model_implementation.py:310-319`, field `:25-28`),
but with annealing disabled the offset has no effect on the returned coefficient.

---

## `run_ppo_epochs`: minibatching and both KL regimes

`run_ppo_epochs(model, data, step, epochs, minibatch_size, target_kl)` —
`PoliwhiRL/agents/PPO/_minibatch.py:47-182`.

Flat data arrives with `returns`, `advantages` and `old_values` already
precomputed. That ordering is intentional: GAE needs the true rollout ordering
with the genuine `V(s_{T+1})` tail bootstrap, so shuffling *before* GAE would
scramble it (`_minibatch.py:8-11`). They are produced by `_update_from_rollout`
(`vec_ppo_agent.py:1633-1740`).

### Batching

| Step | Behaviour | Line |
|---|---|---|
| Batch size | `data["states"].size(0)` — `S*N` here | `_minibatch.py:58` |
| Full-batch fallback | if `minibatch_size` is `None`, `<= 0` or `>= batch_size` → one minibatch of `batch_size` | `:59-61` |
| Minibatching | otherwise `mb_size = int(p)`, `n_mb = ceil(batch/mb)` | `:62-64` |
| Key | `ppo_minibatch_size` default `128` (`ppo_settings.json:18`); the agent reads it with default `None` (`vec_ppo_agent.py:102`) | — |
| Shuffle | `perm = torch.randperm(batch_size, device)`, sliced into `n_mb` index blocks | `:118`, `:122-124` |
| Slicing | `_slice_minibatch` slices `_TENSOR_KEYS` (`states`, `ram_states`, `actions`, `old_log_probs`, `returns`, `advantages`, `old_values`, `old_base_probs`, `old_behaviour_probs`, `dones`) plus per-tensor lists in `mems` | `:18-33`, `:36-44` |
| Loss scaling | `total_loss += epoch_loss / n_mb`, so per-update scale matches pre-minibatching behaviour | `:151-153` |

### Replay-consistency gate (runs BEFORE any epoch)

`replay_policy_diagnostics` yields `base_replay_kl` / `behaviour_replay_kl`; if
either is non-finite or exceeds `ppo_replay_tolerance` (default `1e-6`) the whole
update is skipped and `accepted_epochs = 0`
(`_minibatch.py:81-108`; tolerance read `:76`; stage value `1e-5` at
`configs/stages/open_world.json:22`). Categorical KL is the invariant; raw
per-action log deltas are kept as diagnostics only, because negligible probability
differences on rare actions become large after log without materially changing the
distribution (`:87-92`).

### Two KL regimes

KL early stopping is at **epoch** granularity, not minibatch: after all
minibatches, post-KL is measured against pre-epoch distributions via
`policy_distributions` + `exact_categorical_kl` (`_minibatch.py:135-143`), or
`exact_policy_kls(data)` when the model lacks `policy_distributions` (`:136`).

| Regime | Key | Default | Stage value | Behaviour | Line |
|---|---|---|---|---|---|
| **Hard KL** | `ppo_hard_kl` | `inf` (opt-in; legacy/smoke configs keep their previous behaviour) | `0.06` (`open_world.json:21`) | If KL is non-finite, or base **or** behaviour KL exceeds it: roll back the model **and** optimizer state, call `model.halve_lr_multiplier()`, mark `rejected = True`, break | read `:79`; check `:144-145`; rollback `:146-150` |
| **Target KL** | `ppo_target_kl` | `None` (`vec_ppo_agent.py:64`); `0.03` (`ppo_settings.json:14`) | `0.02` (`open_world.json:20`) | The epoch is **accepted**, then `break` if `max(base_kl, behaviour_kl) > target_kl` | `:159-160` |

So: **target-KL stops without rollback; hard-KL rolls back and permanently
lowers the learning rate.** Rollback points are a deep copy of the `state_dict`
plus optimizer state taken at the top of each epoch (`:116-117`).
`halve_lr_multiplier` is floored by `ppo_lr_hard_floor`
(`ppo_model_implementation.py:625-634`); open-world sets it to `1e-5`
(`open_world.json:24`). Tested in `tests/test_minibatch.py:21`, `:50`, `:176`.

### Loss assembly

`_compute_ppo_losses` — `ppo_model_implementation.py:329-430`. Raises
`RuntimeError` unless returns/advantages are precomputed (`:342-346`).

| Term | Detail | Line |
|---|---|---|
| Advantage normalisation | `advantage_normalisation` default `"rollout"`; `"minibatch"` normalises inside the loss (`:349-350`); the vec agent normalises **once over the whole rollout** before slicing (`vec_ppo_agent.py:1709-1711`) | `:340`; `configs/curriculum_base.json:19` |
| Ratio | `log_ratio = clamp(new_log_probs - old_log_probs, -20, 20)` to prevent exp overflow on MPS | `:370-375` |
| Policy clip | `min(ratio*A, clamp(ratio, 1±eps)*A)`, negated mean; `eps = ppo_epsilon` `0.2` | `:377-379`; `:20`, `ppo_settings.json:6` |
| Value clip | `ppo_clip_value_loss` default `True`: `v_clipped = old_values + clamp(new-old, ±eps)`, loss `coef * 0.5 * max(unclipped, clipped).mean()`; else plain `coef * MSE` | `:383-393`, `:395`; `:34`, `ppo_settings.json:15` |
| `old_values` provenance | from the same segment forward used for GAE targets, keeping the clip baseline self-consistent | `vec_ppo_agent.py:1670-1680`, `:1730` |
| Value coef | `ppo_value_loss_coef` default `1.0` | `:21` |
| `approx_kl` | Schulman's k3 estimator `(ratio-1) - log_ratio`, plus `clip_fraction` | `:408-411` |
| Diagnostics | stashed on `model.last_update_diag`, aggregated by the loop | `:416-422`; `_minibatch.py:130-134` |

### Optimiser step

`_update_networks` — `ppo_model_implementation.py:432-461`: skips non-finite
loss (`:435-437`); `clip_grad_norm_` with `ppo_max_grad_norm` (default `0.5`,
`ppo_settings.json:13`; open-world `0.2`, `open_world.json:19`) at `:441-451`;
if the returned **pre-clip** norm is non-finite the step is dropped, because the
clip coefficient would be `max_norm/inf = 0` and `0*inf = NaN` (`:442-458`).

---

## `RewardScaler` / `RunningMeanStd`

`PoliwhiRL/utils/running_stats.py`: Welford with parallel-batch merging; `var`
starts at `1.0` and `epsilon` doubles as the initial pseudo-count (`:16-45`);
`std` `:47-49`; `state_dict`/`load_state_dict` `:51-57`.

Scaling uses **var only, with no mean subtraction** — zero reward must stay zero
(`:9-11`). `RewardScaler` tracks per-env discounted returns
(`running_returns = gamma*running_returns + reward`), feeds the RMS, and resets on
done (`:70-85`). `scale_factor = 1 / max(rms.std, min_std, 1e-8)` (`:87-88`);
`min_std` floors the amplification of low-variance streams (`:73-76`). It
serialises `{rms, running_returns}` (`:90-94`) and restores with length clamping
(`:96-101`).

| Key | Value | Line |
|---|---|---|
| `scaler_min_std` | agent default `1e-2`; open-world/curriculum sets **`0.5`** | `vec_ppo_agent.py:118-121`; `configs/curriculum_base.json:21` |
| Scale application | to rewards, **before** GAE | `vec_ppo_agent.py:1641`, `:1646` |
| Observation cadence | every step | `vec_ppo_agent.py:732` |

### Critic warmup

`critic_warmup_rollouts` (default `0`) — after loading a checkpoint with
`reset_reward_scaler_on_load=True`, that many PPO updates are skipped so the
scaler can calibrate first (`vec_ppo_agent.py:2139-2146`; consumed at
`:612-615`; initial field `:189`). `configs/curriculum_base.json:29` sets `2`,
and `:18` sets `reset_reward_scaler_on_load: true`.

---

## Checkpointing

### Contents

`VecPPOAgent.save_model(path)` — `vec_ppo_agent.py:1915-1964`:

| File | Written by | Contents |
|---|---|---|
| `actor_critic.pth` | `PPOModel.save` (`ppo_model_implementation.py:556-560`) | network weights |
| `optimizer.pth` | ibid. | optimizer state |
| `scheduler.pth` | ibid. | LR scheduler state |
| `ppo_state.pth` | ibid. | `{"lr_multiplier"}` |
| `info.pth` | `vec_ppo_agent.py:1933-1958` | `episode`, `best_reward` (max episode reward), `episode_data`, **`reward_scaler` state_dict**, `early_stopped`, `success_rate`, `visit_archive`, `frontier_pool`, `checkpoint_recordings`, `best_success_rate`, `best_discoveries`, `best_selection_reward`, `total_rollouts` |

### Cadence

Saves to `config["checkpoint"]` every `checkpoint_frequency` rollouts, gated on
`save_checkpoint` (`vec_ppo_agent.py:662-666`; key read `:69`;
`configs/curriculum_base.json:32-33` sets `true` / `50`), plus a final save at
`:672-675`. (`configs/default_configs/core_settings.json:12` carries a
`checkpoint_frequency` of `5`, which `curriculum_base.json:33` overrides.)

### Best-model selection

If `_should_update_best()`, a duplicate copy (model + `info.pth`) is written to
`<path>/best/` (`vec_ppo_agent.py:1960-1964`). Priority order
(`vec_ppo_agent.py:1966-2025`):

| # | Condition | Criterion | Line |
|---|---|---|---|
| 0 | precondition | the moving-average window must be full | `:1987-1989` |
| 1 | directed stage (`n_goals > 0`), with ≥ `best_success_min_episodes` and at least one success | **goal success rate**, tie-broken on mean reward | `:1999-2009` |
| 2 | free-play (`n_goals <= 0`) | count of **first-honest durable `checkpoint_recordings`** — explicitly never raw reward or unique-map counts | `:2011-2019` |
| 3 | fallback | mean reward | `:2021-2025` |

Windows are sized by `best_success_window` (`100`) and `best_success_min_episodes`
(`100`) (`vec_ppo_agent.py:281-294`, `:421`;
`configs/curriculum_base.json:26-27`).

### Loading

`PPOModel.load(path)` — `ppo_model_implementation.py:562-591`: loads weights
with `weights_only=True` (`:563-565`); restores `_lr_multiplier` from
`ppo_state.pth`, defaulting to `1.0` if the file predates it (`:566-570`);
`reset_optimizer_on_load` (default `False`) skips optimizer restore, while
`reset_lr_scheduler_on_load` (default `True`) does not — the scheduler state is
otherwise restored (`:572-583`, `:584-591`). Config values:
`configs/curriculum_base.json:16-17`.

`VecPPOAgent.load_model(path)` — `vec_ppo_agent.py:2027-2170+`:

- loads `info.pth` with `weights_only=False` after `add_safe_globals(["numpy","np"])` (`:2030-2033`)
- restores `episode` / `start_episode` / `stage_start_episode` (`:2034-2036`)
- determines `same_run_resume` by `os.path.commonpath` of the configured
  checkpoint dir vs the loaded path (`:2037-2049`) — on a same-run resume it
  restores the `best_*` trackers (with a legacy `success_rate` fallback,
  `:2049-2072`); otherwise it resets them to `-1 / -1 / -inf` so a new
  curriculum stage establishes its own high-water marks (`:2073-2078`)
- restores `total_rollouts` (`:2079`); zeroes the entropy offset and clears the
  early-stop latch and success window (`:2080-2094`, including the servo
  decision quoted under [entropy gotcha](#entropy-coefficient-paths-and-two-dead-knobs))
- reloads the visit archive (`:2097-2104`), `checkpoint_recordings`
  (`:2106-2109`), and frontier snapshots with path repair by basename under
  `goexplore_snapshot_dir` (`:2111-2136`)
- sets `_loaded_checkpoint = True` (`:2137`); critic warmup and scaler restore as
  above (`:2139-2153`); rebuilds `episode_data` from a fresh skeleton with deque
  maxlens (`:2155-2170`)

### Auto-resume

`_resolve_auto_resume(config)` — `PoliwhiRL/PPO.py:5-29`. Fires only when
`load_checkpoint` is empty **and** `config["checkpoint"]` contains **both**
`actor_critic.pth` and `info.pth` (`:20-28`); it then sets `load_checkpoint` to
the checkpoint dir. It resumes the *latest* main checkpoint — which is
overwritten every `checkpoint_frequency` — and explicitly **not**
`Checkpoints/best/` (`:9-13`). `erase: true` wipes the folder first, handled in
`main`, so a fresh run starts clean (`:16-18`). Called before agent construction
(`PPO.py:58`), and the agent is built at `PPO.py:77`, loaded at `:79-80`,
trained at `:82`.

### Auto-gamma

`_resolve_auto_gamma(config)` — `PPO.py:32-51`. When `ppo_gamma` is a string —
`"auto"`, as set at `configs/curriculum_base.json:5` — the discount becomes
`min(0.999, 1 - 4/episode_length)` with `episode_length` floored at `8`
(`:44-47`). Rationale: the previous `0.995` cap pinned the horizon near 200
steps, so milestones hundreds-to-thousands of steps away were worth ~0 at
decision time and the critic learned only to predict the dense exploration return
— described in-source as the single biggest structural barrier to progressing
through the game rather than wandering (2026-07-16 diagnosis) (`:37-43`). At the
open-world `episode_length` of `24576` (`open_world.json:14`) that resolves to
the `0.999` cap.

---

## Known gaps

- **`entropy_plateau_reset` as a dead knob** — confirmed by repo-wide grep
  (`configs/curriculum_base.json:12` is the sole occurrence, no reader), but
  still worth confirming with the maintainer that removal was intentional rather
  than an orphaned key awaiting reimplementation.
- **Whether any stage config can enable the servo today** — the grep covers
  `*.py` and `*.json` in the tree and found only
  `configs/curriculum_base.json:11`; no code path other than `load_model`
  (`vec_ppo_agent.py:2087`) writes `_adaptive_entropy_coef`.
- **`ppo_entropy_coef_decay` and `ppo_lr_min`** exist in
  `configs/default_configs/ppo_settings.json:9`,`:16` and are bound to fields
  (`ppo_model_implementation.py:23`,`:24`), but `_get_entropy_coef`
  (`:286-308`) uses the linear formula and never consults
  `self.entropy_decay`. `ppo_lr_min` vs `ppo_lr_hard_floor` (used at
  `:614`, `:627`) are different keys, and the scheduler's floor semantics
  (`_setup_lr_scheduler`, `:227-245`) were not read.
- **Where `main.py` handles `erase: true`** (referenced by `PPO.py:17-18`) was
  not inspected.
- **Capture-cap enforcement** — `goexplore_max_captures_per_episode`,
  `goexplore_capture_cell_count_max`,
  `goexplore_max_progress_captures_per_episode`,
  `goexplore_precheckpoint_interval` appear in `configs/stages/open_world.json`;
  the code that enforces them was not located here (see
  [exploration.md](exploration.md)).
- **Runtime shapes/dtypes on device** were not executed — no training was
  launched; every shape above comes from the storage code and its tests.
