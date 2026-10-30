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
`main.py`,
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
— `PPOTransformer.py:148-167` (docstring at `:150`).

Layer layout, in construction order (`PPOTransformer.py:205-219`):

| Module | Definition | Line |
|---|---|---|
| `self.cnn` | `TileGridCNN(input_shape, d_model)` | `PPOTransformer.py:205` |
| `self.ram_encoder` | `RAMEncoder(ram_dim, d_ram, hidden_dim=ram_hidden_dim, input_norm=ram_input_norm)` | `PPOTransformer.py:206-211` |
| `self.fuse` | `nn.Linear(d_model + d_ram, d_model)` | `PPOTransformer.py:212` |
| `self.transformer_blocks` | `ModuleList` of `num_layers` × `TransformerXLBlock(d_model, n_heads, mem_len, dropout)` | `PPOTransformer.py:214-216` |
| `self.fc_actor` | `nn.Linear(d_model, action_size)` | `PPOTransformer.py:218` |
| `self.fc_critic` | `nn.Linear(d_model, 1)` | `PPOTransformer.py:219` |

Fusion, executed per forward (`PPOTransformer.py:262-268`): image →
`self.cnn` → `(B*T, d_model)` (`:263`); RAM → `self.ram_encoder` →
`(B*T, d_ram)` (`:265`); `torch.cat` then `self.fuse` → `(B*T, d_model)`
(`:267`); reshaped to `(B, seq_len, d_model)` (`:268`).

---

## Geometry: three tiers of "default", and the one that runs

This is the correction this page exists to make. There are **three** places a
dimension could come from, and only the third one applies at runtime.

1. **The `PPOTransformer` class signature** (`PPOTransformer.py:169-183`):
   `d_model=128, d_ram=128, n_heads=4, num_layers=2, dropout=0.1, mem_len=64,
   ram_hidden_dim=192, ram_input_norm=True`.
   These are **never used** — every one is passed explicitly by the caller
   (`ppo_model_implementation.py:209-221`).
2. **The fallbacks inside `_initialize_networks`**
   (`ppo_model_implementation.py:184-221`) — used only when a key is absent
   from the merged config.
3. **`configs/default_configs/core_settings.json`**, which *does* set all six
   keys (`core_settings.json:16-21`) and is loaded into every run's base config
   because `main.py:20-35` globs the whole `default_configs/` directory and
   `merge_configs` (`main.py:77-79`) overlays the user config on top of it.
   Neither `configs/curriculum_base.json` (all 42 lines inspected) nor
   `configs/stages/open_world.json` overrides any architecture key. The RAM
   encoder's two keys are the exception to that last clause:
   `configs/stages/open_world_v4.json` sets `ram_hidden_dim` and
   `ram_input_norm` explicitly.

| Dimension | Config key | `.get()` fallback | Merged-config value | **Effective** | Wired at |
|---|---|---|---|---|---|
| RAM input width | `ram_obs_dim` | required, from env | injected at `PPO.py:75` | from env probe `PPO.py:68` | `ppo_model_implementation.py:185` |
| Trunk width | `d_model` | `128` `ppo_model_implementation.py:193` | `128` `core_settings.json:18` | **128** | passed `ppo_model_implementation.py:214`, used `PPOTransformer.py:212`, `PPOTransformer.py:205` |
| RAM branch width | `d_ram` | `64` `ppo_model_implementation.py:186` | `128` `core_settings.json:16` | **128** | passed `ppo_model_implementation.py:213` |
| RAM encoder hidden width | `ram_hidden_dim` | `192` `ppo_model_implementation.py:207` | `192` (`open_world_v4.json`, explicit) | **192** | passed `ppo_model_implementation.py:219` |
| RAM encoder input LayerNorm | `ram_input_norm` | `True` `ppo_model_implementation.py:208` | `true` (`open_world_v4.json`, explicit) | **true** | passed `ppo_model_implementation.py:220` |
| Attention heads | `n_heads` | `4` `ppo_model_implementation.py:194` | `4` `core_settings.json:19` | **4** | passed `ppo_model_implementation.py:215` |
| Layer count | `num_layers` | `2` `ppo_model_implementation.py:195` | `2` `core_settings.json:20` | **2** | passed `ppo_model_implementation.py:216` |
| Memory length | `mem_len` | `64` `ppo_model_implementation.py:187` | `128` `core_settings.json:17` | **128** | passed `ppo_model_implementation.py:217` |
| Dropout | `ppo_dropout` | `0.0` `ppo_model_implementation.py:203` | `0.0` `core_settings.json:21` | **0.0** | passed `ppo_model_implementation.py:218` |

> **Correction to earlier documentation.** Earlier documentation stated the
> effective geometry is `d_ram=64 / mem_len=64`. That is the `.get()` fallback
> tier, not the effective tier. The superseding values are
> `configs/default_configs/core_settings.json:16-21`; the `64` figures are the
> constructor fallbacks at
> `PoliwhiRL/models/PPO/ppo_model_implementation.py:186-187`, which shadow the
> constructor defaults at `PoliwhiRL/models/PPO/PPOTransformer.py:169-183` — and
> both tiers are bypassed entirely by the explicit call site at
> `PoliwhiRL/models/PPO/ppo_model_implementation.py:209-221`, which passes every
> geometry key. `core_settings.json:16-17` sets `d_ram: 128` and `mem_len: 128`,
> and it is loaded on every run (`main.py:21-22`). Layers, heads, trunk width and
> dropout do agree at 2 / 4 / 128 / 0.0.

The "4-layer" documentation was wrong for a related reason, and the source says
so at the point of implementation: geometry is now explicitly config-driven
"rather than falling through to the `PPOTransformer` constructor defaults (which
silently gave a 2-layer/4-head/128-wide trunk regardless of the '4-layer'
docs)" — `ppo_model_implementation.py:187-191`.

No architecture keys live in
`configs/default_configs/ppo_settings.json` — that file holds only optimisation
and entropy knobs (`ppo_settings.json:2-18`).

### Dropout must stay at zero

Dropout is threaded into both attention and the FFN, but the default is `0.0`
with an explicit warning: the rollout forward (which stores `old_log_probs`)
and the update forward (which recomputes `new_log_probs`) would sample
**different** masks, so PPO's ratio `exp(new − old)` is corrupted by dropout
noise before any policy change — and the trainer never switches the module to
`eval()` during rollout (`ppo_model_implementation.py:195-202`; threading at
`PPOTransformer.py:54`, `:59`, `:61`).

---

## Image branch: `TileGridCNN` — image only

`PoliwhiRL/models/CNN/GameBoy.py:41-127`, instantiated at
`PPOTransformer.py:205` and called with reshaped frames at
`PPOTransformer.py:262-263`. It is the **only** image encoder this project
builds: `GameBoyCNN`, `GameBoyBlock` and `GameBoyOptimizedCNN` — the two-block
backbone and the unused alternative this page used to describe, plus the
`cnn_arch` key that selected between them — are deleted, and `run_config` no
longer records a `cnn_arch` for that reason (`vec_ppo_agent.py:3067-3069`).
It takes `(C, H, W)` image tensors only. **No RAM enters this branch.**

Depthwise layout, in execution order (`_features`, `:108-117`):

| Stage | Layer | Line | Output at `(3, 72, 80)` |
|---|---|---|---|
| stem | `Conv2d(3, 32, k=4, s=4, p=0)` — **linear patchify, no norm, no activation** | `:93-95` | `(32, 18, 20)` |
| body | 3 × `TileGridResidualBlock(32)`, **pre-activation**: GroupNorm → ReLU → 3×3 s1 p1 conv on the branch, identity skip, nothing after the add | `:19-38`, built `:96` | `(32, 18, 20)` |
| down | `Conv2d(32, 64, k=2, s=2, p=0)` — **metatile patchify** → GroupNorm → ReLU | `:98-99`, `:116` | `(64, 9, 10)` |
| squeeze | `Conv2d(64, 16, k=1)` → GroupNorm → ReLU | `:100-101`, `:117` | `(16, 9, 10)` |
| flatten | `view(-1, flat_features)`, `16×9×10 = 1440` | `:102-105`, `:126` | `(B, 1440)` |
| head | `nn.Linear(1440, d_model)` → ReLU | `:106`, `:127` | `(B, d_model)` |

Conv stride schedule `[4, 1, 1, 1, 2, 1]`, kernel schedule `[4, 3, 3, 3, 2, 1]`.
GroupNorm throughout, never BatchNorm — `_make_norm` `:7-16` picks the largest
group count ≤ 8 that divides the channel count, because rollout-time and
update-time batch sizes differ and BatchNorm's running stats would silently
bias the policy across that mismatch. Input pixels are divided by 255 once, at
the top of `forward` (`:124`), so every caller is normalised.

**Why a patchify stem.** At `scaling_factor` 0.5 one 8×8 Game Boy tile is 4×4
scaled px, and 160→80 / 144→72 are both exact halvings, so `kernel == stride ==
4` gives exactly one output vector per tile with no overlap — the feature grid
is locked to the game's own tile grid instead of sitting at some arbitrary
phase across it. It is deliberately linear: a norm would mix statistics across
tiles and an activation would discard the signed patch content before the body
sees it. Alignment is measured, not assumed (`:48-52`): patch-to-tile offset
(0,0) on 28/28 settled frames across 10/10 maps, with SCX/SCY 0 mod 16, and the
camera does **not** clamp on maps smaller than the screen — it drives SCX/SCY
negative and holds the player at a fixed screen cell — so the alignment holds on
interiors too.

**Why flatten and not pool.** The screen is player-centred and the camera never
clamps, so a cell's position on the grid means a fixed direction-and-distance
from the player. Global pooling would discard exactly the information the
alignment buys.

**Why a 2×2 down and a pre-activation body.** SCX/SCY are 0 mod 16, so metatile
*m* covers tiles 2*m* and 2*m*+1 exactly. `kernel == stride == 2` makes each
down cell read exactly one metatile — one player step — the same patchify
argument the stem makes for tiles; a 3×3/s2/p1 down would read tiles
2*m*−1..2*m*+1, one tile of a neighbour and lopsided. The body is
pre-activation so the identity skip carries the stem's signed output through
all three blocks; the post-activation form (`relu(x + relu(norm(conv(x))))`)
ReLU'd the stream after block 1 and its branch could only add. Both pinned by
`tests/test_tile_cnn.py` (`TestMetatileAlignment`,
`TestResidualBlockIsPreActivation`). Receptive field: 7 tiles after the body,
8 tiles per metatile cell after the down.

**Why it raises rather than pads.** If a spatial side is not a multiple of
`TILE_PX = 4` (`:78`) the constructor raises `ValueError` (`:85-91`) instead of
silently padding, because padding would break the one-cell-one-tile property
the whole design rests on without any observable symptom. Verified this session
on `input_shape (3, 70, 80)`.

### Parameter budget: the allocation finding, and the part of it that was overstated

The allocation observation this section was written to make stands, and is now
fixed. Measured against
`Training Outputs/open_world_v3/Checkpoints/actor_critic.pth`, the legacy
network was 2,021,193 parameters, of which `cnn.fc.weight` — a single
`Linear(11520, 128)` — was 1,474,688, **73.0%** of the whole model, while both
conv blocks together were 5,184 (0.26%). Today's tile model, instantiated this
session at the effective geometry (`d_model` 128, `num_layers` 2, `mem_len` 128,
`ram_hidden_dim` 192, input `(3, 72, 80)`, `ram_dim` 222), is **808,181**
parameters:

| Module | Params | Share |
|---|---|---|
| `transformer_blocks` | 445,696 | 55.1% |
| `cnn` total | 223,408 | 27.6% |
| — of which `cnn.fc` (`Linear(1440, 128)`) | 184,448 | 22.8% |
| — of which the conv body (everything in `cnn` outside the head) | 38,960 | 4.8% |
| `ram_encoder` | 105,020 | 13.0% |
| `fuse` | 32,896 | 4.1% |
| `fc_actor` + `fc_critic` | 1,161 | 0.1% |

Recompute for any run with `tools/training_health_report.py <info.pth>`'s
Capacity section rather than trusting a table, which moves with `d_model`,
`num_layers`, `ram_hidden_dim` and the input geometry; its
`-- Architecture provenance --` subsection cross-checks `param_count` from
`run_config` against the sibling `actor_critic.pth`.

> **Correction to a claim published in this section.** It also argued that the
> legacy encoder's ~1.75-tile receptive field was a serious deficiency because
> "everything the policy knows about spatial relations beyond roughly one tile
> of context" was assembled by that one dense layer, "so a ledge three tiles to
> the left must be learned separately at every screen position". **That
> overstated the case and is withdrawn.** The receptive-field arithmetic was
> right and the mis-allocation was real, but the CNN was never the policy's
> navigation channel, and the claim ignored what the RAM vector already
> supplies. Read the definitions in `PoliwhiRL/environment/gym_env.py`:
>
> - `collision_local_0..24` — a (2R+1)² = 5×5 **egocentric** grid of the raw
>   ROM collision value for each nearby cell, scaled to [0, 1], at
>   `COLLISION_FIELD_RADIUS = 2` (`gym_env.py:216-228`, packed `:549-557`,
>   built by `RomCollisionField.local_field`, called `:1331-1336`). It is
>   ground truth from the ROM rather than a learned inference from pixels, it is
>   tileset-invariant (a wall reads the same everywhere), and it carries no
>   history: a per-step affordance signal, "what can I walk onto around me".
> - `run_frontier_distance` — `min(d, RUN_FRONTIER_HORIZON) / horizon` path
>   distance to the nearest run-frontier cell, 1.0 when none is reachable or the
>   field is off (`gym_env.py:262-267`, normalised `:569-572`, field from
>   `_run_frontier` `:1453-1481`). A **global path gradient toward fresh
>   ground**, computed over the ROM collision field, not over the screen.
> - `exit_dir_x` / `exit_dir_y` / `exit_distance` — the way-out sense: path
>   direction (descent of the ROM-collision distance field, egocentric map axes)
>   and `min(d, EXIT_HORIZON) / horizon` distance to the nearest door or
>   crossable map edge leading into a map not yet entered this episode; zeros /
>   1.0 when none is reachable or during a script (`gym_env.py:275-283`,
>   `_exit_sense` `:1355-1398`, packed `:579-580`).
>
> So the policy is given an egocentric walkability grid and two path-distance
> gradients through RAM, and the CNN's job is local visual identification, not
> spatial reasoning about ledges. What remains true is narrower and still worth
> stating: 73% of the legacy model sat in one dense layer with no translation
> equivariance, the visual branch was representationally shallow, and
> `RAMEncoder` — which carries the semantically load-bearing signal — got 3%.
> The tile encoder plus the wider RAM branch is the fix, and the shares above
> are the after.

The state_dict records conv *kernel* sizes but not their *strides*, so the
receptive field is only derivable from `run_config` (persisted since the
capacity work) or from the command line. Note that `conv_strides` keeps its key
but its **meaning** changed with the encoder: `[4, 1, 1, 1, 2, 1]` for
`TileGridCNN`, `[2, 2]` for a legacy run (`vec_ppo_agent.py:2980-2992`). A
receptive-field figure is therefore only comparable within one encoder identity.

---

## `RAMEncoder`

`PoliwhiRL/models/PPO/PPOTransformer.py:103-145`. An optional input
`nn.LayerNorm(ram_dim)` (`:118-128`) followed by
`Linear(ram_dim, hidden) → GELU → Linear(hidden, hidden) → GELU →
Linear(hidden, output_dim) → GELU` (`:129-138`), assembled into `self.net`
(`:142`). Both new knobs are config keys resolved in `_initialize_networks`
(`ppo_model_implementation.py:207-208`) and passed explicitly (`:219-220`):

| Config key | `.get()` fallback | Meaning |
|---|---|---|
| `ram_hidden_dim` | `192` | the encoder's hidden width, both hidden layers |
| `ram_input_norm` | `True` | whether the input `LayerNorm` is present at all |

`PPOTransformer` mirrors them as attributes (`:194-195`) so triage tooling reads
the shape off the live module instead of re-deriving it from config, and
`run_config` persists both (`vec_ppo_agent.py:3070-3071`). Two consequences to
know before reading a state_dict: with the norm **on** every `ram_encoder.net.N`
index shifts by one (`:139-141`), and because the fused output is `d_ram` —
resolved separately and effectively 128 (`ppo_model_implementation.py:186`,
`core_settings.json`) — widening `ram_hidden_dim` leaves `fuse` at
`Linear(256, 128)` and the trunk geometry untouched.

The LayerNorm is a **known, accepted risk, gated behind `ram_input_norm` so it
can be A/B'd without a code change** (`:119-127`): the RAM vector is largely
sparse binary — battle-type one-hots, derived event flags, menu state —
alongside [0,1]-normalised fields, and LayerNorm normalises *across* the feature
dimension per sample, so when most entries are 0 the per-sample std is set by
the handful of nonzero flags and the norm can amplify those flags rather than
stabilise the branch.

The comment at `:112-116` explains the two hidden layers: the RAM
vector is `RAM_OBS_DIM = 222` wide — 173 base keys plus the 49 derived flags of
`DERIVED_FLAG_TABLE`, concatenated into `RAM_FEATURE_KEYS` and counted at
`gym_env.py:304-305` (all three figures re-verified this session). Read the
base-key breakdown off `_BASE_RAM_FEATURE_KEYS`
and its `_EXIT_KEYS` / `_MENU_KEYS` / `_BATTLE_ID_KEYS` extensions rather than
off prose: it has grown several times (170, then 171, then 222) and a
hand-listed subgroup count is the first thing to go stale. Verify the live
value with `python -c "from PoliwhiRL.environment.gym_env import RAM_OBS_DIM;
print(RAM_OBS_DIM)"`, or read `ram_dim` out of any run's `info.pth`.

---

## Memory: `age_emb`, not positional encoding

There is **no absolute positional encoding**, deliberately: the trainer feeds one
frame per step, so with single-frame chunks an absolute encoding would be a
constant. Recency comes from the blocks' learned age embeddings, and coarser
history — recent maps, steps-since-novel-cell, run-wide cell visit counts —
arrives explicitly through the RAM feature vector
(`PPOTransformer.py:157-163`). Fresh-episode memories are zero tensors, chosen
for fixed buffer shapes (`PPOTransformer.py:164-166`).

`age_emb` is a learned, **zero-initialised** additive embedding of shape
`(mem_len + _AGE_HEADROOM, d_model)` with `_AGE_HEADROOM = 64`, applied to the
attention *input* only — the cache stores un-aged activations
(`PPOTransformer.py:47-49`, `:64-70`, applied `:90`). Overshooting the table
raises `ValueError` (`:85-89`).

Memory is passed **as an argument**, a per-layer list of
`(B, mem_len, d_model)`, rather than stored on the module; callers own the
lifecycle — reset at episode start, carry across rollout steps, snapshot per
transition for replay at update time (`PPOTransformer.py:152-155`).
`init_mems` returns zeros `(B, mem_len, d_model)` per layer
(`PPOTransformer.py:237-241`).

### Segment BPTT vs rollout memory regime

`TransformerXLBlock` runs in two regimes (`PPOTransformer.py:22-100`, docstring
`:26-34`):

| Regime | Input | Gradient | Line |
|---|---|---|---|
| Rollout | ONE frame per call, carrying memory | none (inference) | `PPOTransformer.py:27-28` |
| Training | a contiguous `seq_len`-frame segment with **detached prior memory** as context | flows across the whole segment — real BPTT | `PPOTransformer.py:29-34` |

The forward loops position-by-position so the age embedding is applied relative
to the *current* frame — the same computation as rollout inference — and
`history` is detached only at the segment boundary (`PPOTransformer.py:73-100`,
`history.detach()` at `:100`). Causality holds by construction: only keys
`k <= q` are ever present in `extended` (`PPOTransformer.py:83`, `:91`;
docstring `:36-40`). `reset_after` `(B, seq_len)` bool zeroes each layer's history
after that position, matching collection-time memory reset on episode end
(`PPOTransformer.py:248`, applied `:96-98`, `:272`).

Block internals: `nn.MultiheadAttention(d_model, n_heads, batch_first=True,
dropout=dropout)` → `norm1` (pre-norm residual on the current query) → FFN
`Linear(d_model, d_model*4) → GELU → Dropout → Linear(d_model*4, d_model) →
Dropout` → `norm2` (`PPOTransformer.py:54-63`, applied `:91-93`).

Segmentation itself is the buffer's job: `get_data()` emits NON-OVERLAPPING
contiguous segments, feeding each L-frame sequence with the detached memory that
preceded it so gradient flows across all L positions
(`vec_ppo_storage.py:116-137`). See
[training.md](training.md#rollout-collection-and-storage).

---

## Initialisation and heads

`_reset_parameters` (`PPOTransformer.py:223-235`), via `_orthogonal_init`
(`:8-19`):

| Module | Scheme | Gain | Line |
|---|---|---|---|
| everything except the two heads | orthogonal | `sqrt(2)` | `PPOTransformer.py:226-230` |
| `fc_actor` | orthogonal | `0.01` — near-uniform initial action probs, so the policy explores rather than committing | `PPOTransformer.py:233` |
| `fc_critic` | orthogonal | `1.0` — values start near zero | `PPOTransformer.py:235` |

The value head is a plain `nn.Linear(d_model, 1)` applied to **all** segment
positions → `(B, seq_len, 1)` (`:219`, applied `:279`). The actor head is
`nn.Linear(d_model, action_size)` (`:218`, applied `:277`).

---

## Policy normalisation

There is no action mask (removed 2026-10-28; see
[environment.md](environment.md#actions)). `_normalise_base_policy` does
`nan_to_num`, falls back to uniform if the denominator is zero, and renormalises.

---

## Behaviour policy: temperature, not epsilon

`_behaviour_policy(action_probs, ram_sequence)`
(`ppo_model_implementation.py:77-118`) performs **temperature reshaping**, not
an epsilon-uniform mix:

```
shaped_clock = clock ** stuck_temperature_onset_exponent        # :114
temperature  = 1.0 + stuck_action_temperature * shaped_clock     # :115
reshaped     = probs.clamp(min=1e-10) ** (1 / T)                # :116
               then renormalised
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
| `base_probs` | normalised policy output — the distribution the **loss** is defined on | `vec_ppo_storage.py:42` | `old_base_probs` `:194` |
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
the **base** policy, not the behaviour policy
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

- **`TileGridCNN` alignment invariants are asserted on the grid, not proven on
  the emulator.** `_features` (`PoliwhiRL/models/CNN/GameBoy.py:96-105`) is split
  out of `forward` so tile isolation of the stem and translation equivariance of
  stem+body can be asserted directly; the (0,0) patch-to-tile offset figure
  quoted above is a measurement over settled frames, not a property the code
  enforces at runtime.
- **Causality is argued, not proven.** `TransformerXLBlock` uses
  `nn.MultiheadAttention` without an explicit `attn_mask`; causality rests on
  the argument that only `k <= q` positions are present in `extended`
  (`PPOTransformer.py:83-91`) — plausible but not independently proven.
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
