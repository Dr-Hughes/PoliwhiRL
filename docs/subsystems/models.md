# Subsystem: models

What this page is for: the exact geometry of the policy/value network, how that
geometry is resolved from config (and why the code default is not the number
that actually runs), the two encoder branches, the Transformer-XL memory regime,
the action head and its mask, the behaviour-policy temperature, and the two
stored policies the KL terms are measured against.

**Sources**: `PoliwhiRL/models/PPO/PPOTransformer.py`,
`PoliwhiRL/models/PPO/ppo_model_implementation.py`,
`PoliwhiRL/models/CNN/GameBoy.py`, `PoliwhiRL/replay/vec_ppo_storage.py`,
`PoliwhiRL/agents/PPO/vec_ppo_agent.py`, `PoliwhiRL/agents/PPO/_minibatch.py`,
`PoliwhiRL/environment/action_mask.py`, `main.py`,
`configs/default_configs/core_settings.json`,
`configs/default_configs/ppo_settings.json`, `configs/curriculum_base.json`,
`configs/stages/open_world.json`, `tests/test_vec_ppo_correctness.py`,
`tests/test_vec_ppo_memory.py`.

---

## Trunk container

`PPOTransformer` is the whole network:

```
(Screen Image, RAM vector) -> dual encoders -> fused d_model
                           -> TransformerXL -> Actor/Critic heads
```
— `PPOTransformer.py:153-172` (docstring at `:155`).

Layer layout, in construction order (`PPOTransformer.py:199-211`):

| Module | Definition | Line |
|---|---|---|
| `self.cnn` | `GameBoyCNN(input_shape, d_model)` | `PPOTransformer.py:199` |
| `self.ram_encoder` | `RAMEncoder(ram_dim, d_ram)` | `PPOTransformer.py:200` |
| `self.fuse` | `nn.Linear(d_model + d_ram, d_model)` | `PPOTransformer.py:201` |
| `self.transformer_blocks` | `ModuleList` of `num_layers` × `TransformerXLBlock(d_model, n_heads, mem_len, dropout)` | `PPOTransformer.py:203-208` |
| `self.fc_actor` | `nn.Linear(d_model, action_size)` | `PPOTransformer.py:210` |
| `self.fc_critic` | `nn.Linear(d_model, 1)` | `PPOTransformer.py:211` |

Fusion, executed per forward (`PPOTransformer.py:265-271`): image →
`self.cnn` → `(B*T, d_model)` (`:266`); RAM → `self.ram_encoder` →
`(B*T, d_ram)` (`:268`); `torch.cat` then `self.fuse` → `(B*T, d_model)`
(`:270`); reshaped to `(B, seq_len, d_model)` (`:271`).

---

## Geometry: three tiers of "default", and the one that runs

This is the correction this page exists to make. There are **three** places a
dimension could come from, and only the third one applies at runtime.

1. **The `PPOTransformer` class signature** (`PPOTransformer.py:179-184`):
   `d_model=128, d_ram=128, n_heads=4, num_layers=2, dropout=0.1, mem_len=64`.
   These are **never used** — every one is passed explicitly by the caller
   (`ppo_model_implementation.py:205-215`).
2. **The fallbacks inside `_initialize_networks`**
   (`ppo_model_implementation.py:185-215`) — used only when a key is absent
   from the merged config.
3. **`configs/default_configs/core_settings.json`**, which *does* set all six
   keys (`core_settings.json:16-21`) and is loaded into every run's base config
   because `main.py:20-35` globs the whole `default_configs/` directory and
   `merge_configs` (`main.py:77-79`) overlays the user config on top of it.
   Neither `configs/curriculum_base.json` (all 42 lines inspected) nor
   `configs/stages/open_world.json` overrides any architecture key.

| Dimension | Config key | `.get()` fallback | Merged-config value | **Effective** | Wired at |
|---|---|---|---|---|---|
| RAM input width | `ram_obs_dim` | required, from env | injected at `PPO.py:75` | from env probe `PPO.py:68` | `ppo_model_implementation.py:186` |
| Trunk width | `d_model` | `128` `ppo_model_implementation.py:194` | `128` `core_settings.json:18` | **128** | passed `ppo_model_implementation.py:210`, used `PPOTransformer.py:201`, `PPOTransformer.py:205` |
| RAM branch width | `d_ram` | `64` `ppo_model_implementation.py:187` | `128` `core_settings.json:16` | **128** | passed `ppo_model_implementation.py:209` |
| Attention heads | `n_heads` | `4` `ppo_model_implementation.py:195` | `4` `core_settings.json:19` | **4** | passed `ppo_model_implementation.py:211` |
| Layer count | `num_layers` | `2` `ppo_model_implementation.py:196` | `2` `core_settings.json:20` | **2** | passed `ppo_model_implementation.py:212` |
| Memory length | `mem_len` | `64` `ppo_model_implementation.py:188` | `128` `core_settings.json:17` | **128** | passed `ppo_model_implementation.py:213` |
| Dropout | `ppo_dropout` | `0.0` `ppo_model_implementation.py:204` | `0.0` `core_settings.json:21` | **0.0** | passed `ppo_model_implementation.py:214` |

> **Correction to earlier documentation.** Earlier documentation stated the
> effective geometry is `d_ram=64 / mem_len=64`. That is the `.get()` fallback
> tier, not the effective tier. The superseding values are
> `configs/default_configs/core_settings.json:16-21`; the `64` figures are the
> constructor fallbacks at
> `PoliwhiRL/models/PPO/ppo_model_implementation.py:187-188`, which shadow the
> constructor defaults at `PoliwhiRL/models/PPO/PPOTransformer.py:179-184` — and
> both tiers are bypassed entirely by the explicit call site at
> `PoliwhiRL/models/PPO/ppo_model_implementation.py:205-215`, which passes every
> geometry key. `core_settings.json:16-17` sets `d_ram: 128` and `mem_len: 128`,
> and it is loaded on every run (`main.py:21-22`). Layers, heads, trunk width and
> dropout do agree at 2 / 4 / 128 / 0.0.

The "4-layer" documentation was wrong for a related reason, and the source says
so at the point of implementation: geometry is now explicitly config-driven
"rather than falling through to the `PPOTransformer` constructor defaults (which
silently gave a 2-layer/4-head/128-wide trunk regardless of the '4-layer'
docs)" — `ppo_model_implementation.py:189-193`.

No architecture keys live in
`configs/default_configs/ppo_settings.json` — that file holds only optimisation
and entropy knobs (`ppo_settings.json:2-18`).

### Dropout must stay at zero

Dropout is threaded into both attention and the FFN, but the default is `0.0`
with an explicit warning: the rollout forward (which stores `old_log_probs`)
and the update forward (which recomputes `new_log_probs`) would sample
**different** masks, so PPO's ratio `exp(new − old)` is corrupted by dropout
noise before any policy change — and the trainer never switches the module to
`eval()` during rollout (`ppo_model_implementation.py:197-204`; threading at
`PPOTransformer.py:78`, `:84`, `:86`).

---

## Image branch: `GameBoyCNN` — image only

`PoliwhiRL/models/PPO/PPOTransformer.py:22-42`. Two residual blocks,
`GameBoyBlock(input_shape[0], 16)` then `GameBoyBlock(16, 32)` (`:28-29`); the
flattened width is computed once on a dummy tensor under `torch.no_grad()`
(`:31-34`); the head is `nn.Linear(flat_features, d_model)` followed by ReLU
(`:36`, `:42`). It takes `(C, H, W)` image tensors only (`:27`) and is called
with reshaped frames at `PPOTransformer.py:265-266`. **No RAM enters this
branch.**

The block/norm primitives live in `PoliwhiRL/models/CNN/GameBoy.py`:
`_make_norm` `:7-16`, `GameBoyBlock.__init__` `:20-27`, `forward` `:29-30`.
That file also defines `GameBoyOptimizedCNN` (`:33-48`), a separate backbone
**not** wired into `PPOTransformer`, which uses its own `GameBoyCNN`
(`PPOTransformer.py:199`).

---

## `RAMEncoder`

`PoliwhiRL/models/PPO/PPOTransformer.py:128-150`. An MLP
`Linear(ram_dim, hidden) → GELU → Linear(hidden, hidden) → GELU →
Linear(hidden, output_dim) → GELU`, with `hidden_dim=128` as the default
(`:133-147`). The comment at `:135-139` explains the two hidden layers: the RAM
vector is `RAM_OBS_DIM = 170` wide — 121 base keys (the 67-key literal block
`gym_env.py:66-186`, plus 25 `visited_local_*` `:197-200`, 3 `frontier_dir_*`
`:207-212`, 25 `collision_local_*` `:222-225`, 1 `stagnation_clock` `:236-237`)
plus the 49 derived flags of `DERIVED_FLAG_TABLE` (`checkpoints.py:11-60`), appended
and counted at `gym_env.py:240-242`. Because the fused output
is `d_ram`, the encoder's `hidden_dim=128` no longer automatically equals the
trunk's working width — `d_ram` is resolved separately
(`ppo_model_implementation.py:187`; effective `128` via
`core_settings.json:16`).

---

## Memory: `age_emb`, not positional encoding

There is **no absolute positional encoding**, deliberately: the trainer feeds one
frame per step, so with single-frame chunks an absolute encoding would be a
constant. Recency comes from the blocks' learned age embeddings, and coarser
history — recent maps, steps-since-novel-cell, run-wide cell visit counts —
arrives explicitly through the RAM feature vector
(`PPOTransformer.py:162-171`). Fresh-episode memories are zero tensors, chosen
for fixed buffer shapes (`PPOTransformer.py:169-171`).

`age_emb` is a learned, **zero-initialised** additive embedding of shape
`(mem_len + _AGE_HEADROOM, d_model)` with `_AGE_HEADROOM = 64`, applied to the
attention *input* only — the cache stores un-aged activations
(`PPOTransformer.py:65-72`, applied `:89-95`, `:115`). Overshooting the table
raises `ValueError` (`:110-114`).

Memory is passed **as an argument**, a per-layer list of
`(B, mem_len, d_model)`, rather than stored on the module; callers own the
lifecycle — reset at episode start, carry across rollout steps, snapshot per
transition for replay at update time (`PPOTransformer.py:157-160`).
`init_mems` returns zeros `(B, mem_len, d_model)` per layer
(`PPOTransformer.py:229-233`).

### Segment BPTT vs rollout memory regime

`TransformerXLBlock` runs in two regimes (`PPOTransformer.py:45-125`):

| Regime | Input | Gradient | Line |
|---|---|---|---|
| Rollout | ONE frame per call, carrying memory | none (inference) | `PPOTransformer.py:49-57` |
| Training | a contiguous `seq_len`-frame segment with **detached prior memory** as context | flows across the whole segment — real BPTT | `PPOTransformer.py:49-57` |

The forward loops position-by-position so the age embedding is applied relative
to the *current* frame — the same computation as rollout inference — and
`history` is detached only at the segment boundary (`PPOTransformer.py:97-125`,
`history.detach()` at `:125`). Causality holds by construction: only keys
`k <= q` are ever present in `extended` (`PPOTransformer.py:59-63`,
`:106-116`). `reset_after` `(B, seq_len)` bool zeroes each layer's history
after that position, matching collection-time memory reset on episode end
(`PPOTransformer.py:251-253`, applied `:121-123`).

Block internals: `nn.MultiheadAttention(d_model, n_heads, batch_first=True,
dropout=dropout)` → `norm1` (pre-norm residual on the current query) → FFN
`Linear(d_model, d_model*4) → GELU → Dropout → Linear(d_model*4, d_model) →
Dropout` → `norm2` (`PPOTransformer.py:77-88`, `:116-118`).

Segmentation itself is the buffer's job: `get_data()` emits NON-OVERLAPPING
contiguous segments, feeding each L-frame sequence with the detached memory that
preceded it so gradient flows across all L positions
(`vec_ppo_storage.py:116-137`). See
[training.md](training.md#rollout-collection-and-storage).

---

## Initialisation and heads

`_reset_parameters` (`PPOTransformer.py:215-227`), via `_orthogonal_init`
(`:8-19`):

| Module | Scheme | Gain | Line |
|---|---|---|---|
| everything except the two heads | orthogonal | `sqrt(2)` | `PPOTransformer.py:218-222` |
| `fc_actor` | orthogonal | `0.01` — near-uniform initial action probs, so the policy explores rather than committing | `PPOTransformer.py:223-225` |
| `fc_critic` | orthogonal | `1.0` — values start near zero | `PPOTransformer.py:226-227` |

The value head is a plain `nn.Linear(d_model, 1)` applied to **all** segment
positions → `(B, seq_len, 1)` (`:211`, applied `:288`). The actor head is
`nn.Linear(d_model, action_size)` (`:210`, applied `:280`).

---

## Action mask

`_action_mask_for(ram_sequence)` builds a **per-position**
`(B, seq_len, action_size)` mask by flattening to `(B*seq_len, ram_dim)`,
calling `compute_action_mask(..., allow_menus_walking=...)`, and reshaping
(`ppo_model_implementation.py:60-75`, call at `:74`). It returns `None` when
masking is disabled (`:70-71`); the key is `action_mask_enabled`, read at
`:37` and set `true` in `configs/curriculum_base.json:40`, with
`allow_menus_walking: false` at `:41`.

Masks are **not stored** in the rollout buffer. They are a deterministic function
of each frame's RAM, so they reconstruct identically at rollout and update time,
which keeps PPO's importance ratio self-consistent
(`ppo_model_implementation.py:65-68`, echoed `:352-356`;
`vec_ppo_storage.py` allocates no mask buffer, `:28-50`).

The mask is applied inside the model as an additive shift,
`logits + (action_mask - 1.0) * 1e9`, before softmax — `-1e9`, **not** `-inf`,
so gradients stay finite if everything is masked (`PPOTransformer.py:242-250`,
applied `:281-287`). `_normalise_base_policy` then does `nan_to_num`, zeroes
blocked actions, falls back to uniform-over-valid if the denominator is zero,
and renormalises (`:120-128`).

---

## Behaviour policy: temperature, not epsilon

`_behaviour_policy(action_probs, ram_sequence, action_mask=None)`
(`ppo_model_implementation.py:77-118`) performs **temperature reshaping**, not
an epsilon-uniform mix:

```
shaped_clock = clock ** stuck_temperature_onset_exponent        # :114
temperature  = 1.0 + stuck_action_temperature * shaped_clock     # :115
reshaped     = probs.clamp(min=1e-10) ** (1 / T)                # :116
               then masked (:117) and renormalised (:118)
```

`clock` is the value of `RAM_FEATURE_INDEX["stagnation_clock"]` clamped to
`[0,1]` (`:110`; the index is imported locally from
`PoliwhiRL.environment.gym_env` to avoid import-order coupling, `:97-101`). So
`T` starts at exactly `1.0` — a no-op — at `clock = 0` and climbs to
`1 + stuck_action_temperature` at `clock = 1`.

| Key | Default | Meaning | Line |
|---|---|---|---|
| `stuck_action_temperature` | `0.0` = disabled | max EXTRA temperature at full stagnation | `:39-44` |
| `stuck_temperature_onset_exponent` | `1.0` (linear); floored at `1e-6` | ramp shape — `<1.0` front-loads exploration pressure (0.5 reaches half strength at clock=0.25), `>1.0` back-loads | `:45-54` |

The rationale is in-code: the previous uniform-epsilon mix only engaged past
`clock=0.75` and capped at a 20 % uniform blend, so the first 75 % of the
stagnation budget got no extra exploration
(`ppo_model_implementation.py:80-92`). Open-world sets
`stuck_action_temperature: 5.0` (`configs/stages/open_world.json:48`).

Tests pin the semantics: `tests/test_vec_ppo_correctness.py:44`
(`stuck_action_temperature: 1.0` → `T=2` at clock=1, asserted `:193`), and
`:209-224` (the default exponent is reproduced when the key is absent; sub-linear
vs linear ramp comparison).

---

## Two stored policies, two KLs

The rollout stores **both** distributions side by side:

| Stored tensor | Semantics | Buffer | `get_data()` key |
|---|---|---|---|
| `base_probs` | masked, normalised policy output — the distribution the **loss** is defined on | `vec_ppo_storage.py:42` | `old_base_probs` `:194` |
| `behaviour_probs` | temperature-reshaped distribution actually **sampled** from | `vec_ppo_storage.py:43` | `old_behaviour_probs` `:195-197` |

Both are needed because KL must be measured against each:
`exact_policy_kls` computes base KL and behaviour KL separately
(`ppo_model_implementation.py:154-159`, at `:157-158`), and
`replay_policy_diagnostics` reports `base_replay_kl` / `behaviour_replay_kl`
(`:161-183`, at `:176-177`). Shapes `(S,N,L,A)` are asserted in
`tests/test_vec_ppo_memory.py:91-92`.

`exact_categorical_kl(old_probs, new_probs)` is the mean of
`sum_a p·(log p − log q)` over every batch/sequence position, using
`where(p > 0, ..., 0)` and `clamp(min=1e-10)`
(`ppo_model_implementation.py:146-152`). The update loop consumes both KLs;
see the two KL regimes in [training.md](training.md#two-kl-regimes).

---

## Entropy coefficient (model side)

`_get_entropy_coef(step)` has three ordered paths
(`ppo_model_implementation.py:286-308`):

1. **Adaptive servo override** — `adaptive = getattr(self,
   "_adaptive_entropy_coef", None)`; if not `None`, return it and ignore the
   schedule entirely (`:300-302`). Field declared `:29-33`; setter
   `set_entropy_coef(value)` normalises to `float` or `None` (`:321-327`).
2. **Anneal off** — if `ppo_entropy_anneal_enabled` is false, return the
   constant `self.entropy_coef` (`:303-304`). Both
   `configs/curriculum_base.json:10` and `configs/default_configs/ppo_settings.json:11`
   set it `false`. Rationale: free-play and open-ended stages keep a high
   exploration floor instead of freezing on an early optimum (`:294-296`).
3. **Linear decay** — `total = ppo_entropy_anneal_steps`, falling back to
   `num_rollouts`, then `1` (`:305`); `effective = max(0, step −
   _entropy_reset_offset)` (`:306`); `progress = min(effective/max(total,1),
   1.0)` (`:307`); return `entropy_coef*(1−progress) + entropy_min*progress`
   (`:308`). `entropy_coef`/`entropy_min` come from `ppo_entropy_coef`
   (`0.01`) and `ppo_entropy_coef_min` (`0.001`) (`:22-24`; values
   `ppo_settings.json:8`,`:10`).

`set_entropy_offset(offset)` rewinds the schedule via `_entropy_reset_offset`
(`:310-319`, field `:25-28`).

The entropy **term** in the loss is `-coef * mean(H(base_probs))`, computed on
the **base** (masked) policy, not the behaviour policy
(`ppo_model_implementation.py:397-406`). The old per-timestep stagnation entropy
boost was removed with the rest of the anti-stuck machinery — exploration is
driven by reward (`:397-401`).

> **The servo is inert on this branch.** Path 1 above can only fire if some
> caller writes `_adaptive_entropy_coef`, and the only writer in the repo is
> checkpoint load — see the gotcha in
> [training.md](training.md#entropy-coefficient-paths-and-two-dead-knobs).

`halve_lr_multiplier()` — a persistent LR cut floored by `ppo_lr_hard_floor`
(default `1e-5`), recomputing `_lr_multiplier` against the nominal LR and
writing `target` into every param group (`ppo_model_implementation.py:625-634`);
companion `_apply_lr_multiplier` re-applies it after each scheduler step
(`:613-620`, `step_scheduler` `:606-611`); `effective_lr()` `:622-623`. It is
called on **hard-KL rejection** (`_minibatch.py:148`; test
`tests/test_vec_ppo_correctness.py:320`).

---

## Known gaps

- **`GameBoyBlock` internals** (`PoliwhiRL/models/CNN/GameBoy.py:19-30`) —
  exact kernel sizes, stride, pooling and the `_make_norm` choice (`:7-16`)
  were not read, so the concrete `flat_features` value for a given
  `input_shape` is unverified.
- **Causality is argued, not proven.** `TransformerXLBlock` uses
  `nn.MultiheadAttention` without an explicit `attn_mask`; causality rests on
  the argument that only `k <= q` positions are present in `extended`
  (`PPOTransformer.py:106-118`) — plausible but not independently proven.
- **`ppo_entropy_coef_decay` appears unused.** It is bound to a field
  (`ppo_model_implementation.py:23`) and present in
  `ppo_settings.json:9`, but `_get_entropy_coef` (`:286-308`) uses the
  **linear** formula, not the decay ratio.
- **`ppo_lr_min` vs `ppo_lr_hard_floor` are different keys**, and the
  scheduler's floor semantics (`_setup_lr_scheduler`,
  `ppo_model_implementation.py:227-245`) were not read.
- **Whether the servo can be switched on from a stage config today** — only
  `configs/curriculum_base.json:11` sets `entropy_servo_enabled` among the
  files inspected; the rest of `configs/stages/` was not exhaustively checked
  for that key.
- **Runtime shapes/dtypes on device** were not executed; all shapes come from
  the storage code and its tests.
