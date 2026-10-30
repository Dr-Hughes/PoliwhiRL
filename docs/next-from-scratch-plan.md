# Plan for the next from-scratch run (open_world_v3)

Written 2026-10-28, after the LLM-harness audit, the switch to settle
stepping, an independent plan review, and live testing of healing and the
Route 30 battles. It records what the run starts with and why, how to launch
it, and how to judge it. Current run status is in [`status.md`](status.md);
the backlog is in [`roadmap.md`](roadmap.md). Where this page and the code
disagree, the code is right.

## Why the run starts from scratch

The environment contract changed; a checkpoint cannot carry across. Old
weights do not even load: `info.pth` records `ram_dim`, and a mismatch now
stops the load with an error instead of silently half-loading.

| Change | Where |
|---|---|
| Settle stepping: a step lasts until the game next waits for input (~20 frames to walk a tile, one step per text box, ~11 per wild battle) | `PoliwhiRL/environment/settle.py` |
| `wOptions` forced to 0xD1: instant text, battle scene off, SET battle style | `game_options_byte` |
| Action set is 8: the mask is removed and SELECT is dropped | `gym_env.ACTIONS` |
| RAM vector grows 178 → 222: `last_step_frames`, exit sense, menu state, battle identity; `rival_cherrygrove` held at 0 | `gym_env.py`, `RAM.py` |
| Ledges modelled as the game does them (affects run-frontier, path sense, exit sense) | `rom_collision._edges` |
| `new_map` paid after entry scenes; event flags 0-7 excluded; stagnation counts scripted steps | `rewards.py` |

## What the LLM runs and live tests showed

1. **The story path is short in decisions.** The LLM went from the bedroom
   to flag 31 in 1,060 honest steps. That includes 268 steps stuck in
   Route 29's ledge strip before it was given walking routes.
2. **Navigation beyond the local view is the shared failure.** The LLM made
   every story decision correctly, but needed shortest-path routes to leave
   the ledge strip. The policy fails the same way at the Route 30 gap. Live
   testing found a framework bug that made this worse: the collision graph
   modelled ledges as hop-from-the-cell-before, which closed the real way
   round Youngster Mikey at the gap. The exit sense and run-frontier field
   now see that route.
3. **The reward pays correctly along the story path.** The verifier checked
   every event on the LLM's path: no unexplained payouts, and no real misses.
4. **Game traps for an A-mashing policy are real.** One was the "Will KRIS
   change POKéMON?" party-screen loop, in every multi-mon trainer battle; SET
   battle style removes it. Another was the Dude's self-playing demo; it is
   now one step.

## How to launch

```bash
uv run python -m pytest tests -q          # includes the emulator scenario suite
uv run python tools/preflight.py          # config, settle armed, obs size, demos, rings, scenarios
uv run python main.py --use_config configs/stages/open_world.json
```

`open_world.json` writes to `Training Outputs/open_world_v3/`. Pre-flight
warns if that directory already holds a checkpoint, in which case the run
resumes.

**Comparison arm (recommended if the hardware allows two runs):** the same
config without demonstrations.

```bash
uv run python main.py --use_config configs/stages/open_world.json \
    --output_base_dir "./Training Outputs/open_world_v3_nodemo/" \
    --load_checkpoint "./Training Outputs/open_world_v3_nodemo/Checkpoints" \
    --sil_demo_fraction 0.0
```

## What the run uses from the LLM

All of it is offline: the LLM is never in the training loop.

- **SIL demonstrations.** `Training Outputs/llm_demos/*.npz` is built by
  `tools.llm_agent.demos --chain`, which replays `settle_run_1` and its
  resumed continuation as one honest episode through the current env, so
  observations and rewards match the current code. Each leg is tagged with
  the next story checkpoint it leads to.
  - Each SIL batch draws at most `sil_demo_fraction` (25%) from demos.
    Batches are never demo-only.
  - Demo memories are burnt in by running the current policy from the leg's
    start, refreshed every 10 rollouts.
  - A leg is retired once honest play reaches its checkpoint at
    `rc_exit_rate`. This retirement is the guard against permanent imitation
    of the LLM's detours: the review showed that (R − V)+ alone does not
    switch demos off, because demo-only states get little value training.
- **Reverse-curriculum seed rings.** The same replay writes pre-fire rings
  every 64 steps for flags 1735, 27, 30, 31, 65 and 66 (43 states). A fresh
  run copies them into its own snapshot directory. They are used only once
  honest play shows a flag is weak, and honest rings evict them first.
- **Regression checks.**
  - `tools.llm_agent.scenarios` replays two recorded trajectories through the
    verifier with invariants and probes: bedroom → Route 29, and bedroom →
    flag 31 through the rival battle.
  - `tools.llm_agent.replay` checks any run still replays exactly.
- **Held-out leg.** No demo covers Route 30 gap → Violet. Honest progress
  there is the test of whether imitation generalises.

## Step budgets

These stay as they were and are now pinned explicitly, not `auto`.
Walking, which dominates episodes, still costs one step per tile under
settle stepping.

| Setting | Value |
|---|---|
| `episode_length` | 49152 |
| `stagnation_truncation_steps` | 3072 |
| `battle_stagnation_truncation_steps` | 3072 |
| `ppo_gamma` | 0.999 |

A pilot to re-baseline step budgets was dropped. A from-scratch pilot reaches
flag 66 too rarely to measure from (v2: 3 of 270 honest episodes by episode
377), and `auto` would have scaled the stagnation limit with any longer
episode.

## How to judge it

**Baseline: open_world_v2 at episode 377**, from scratch, with the old
stepping, after about 13 hours. Honest rates at that point:

| Flag | Honest rate |
|---|---|
| 30 | 63% |
| 39 | 54% |
| 31 | 8% |
| 66 | 1.5% |

Compare per environment step and per wall-clock hour, not per episode: steps
per episode differ under settle stepping.

Check with `uv run python tools/training_health_report.py "Training Outputs/open_world_v3/Checkpoints/info.pth"`.
Its sections:

- **Stepping:** frames per step and capped steps. Capped steps should stay
  about 0.
- **Competence / reverse curriculum / self-imitation:** honest
  per-checkpoint rates and seeded rings.
- The usual optimisation, reward-mix and coverage sections
  ([`operations/run-triage.md`](operations/run-triage.md)).

Success looks like this:
- honest flag 31 well ahead of v2 per step;
- then honest Route 31 / Violet entries, a leg with no demos;
- honest coverage and the exploration sources still leading the reward mix.

Warning signs specific to this run:
- **Imitation without generalisation:** progress only on demo legs.
- **A demo leg never retires:** its checkpoint stays below `rc_exit_rate`
  while SIL keeps imitating it.
- **Door-touring:** the exit sense targets every unentered map, including
  dead-end houses.
- **Menu dwelling:** a high START share in `button_counts` with stagnation
  truncations rising.
- **The seeded/honest gap widening.**

## Known risks and open items

- **Seeded episodes start with empty visit history,** so the exit sense and
  coverage features in a seeded episode treat already-done maps as new.
  Inheriting the visit history at capture is the higher-value general fix not
  yet built ([`status.md`](status.md) lists it as "not tried").
- **Only one honest LLM playthrough exists.** A second, recorded on the
  final env, would give demos for the catch leg and, if the LLM gets
  there, Violet.
- **Deferred: a "walk until something happens" macro action.** Revisit only
  if walking steps dominate after the run starts.

---

# The run after this one: open_world_v4

The plan above was executed as `open_world_v3`. This section is the v4 run,
which changes the **encoder** and the **observation filter** and nothing else.

## Why it starts from scratch again

Both changes alter every observation and every weight, so nothing carries over.
`info.pth` records `ram_dim` and a mismatch stops the load, and the RAM branch
gained an input LayerNorm that shifts every `ram_encoder.net.N` index.

| Change | Where |
|---|---|
| `TileGridCNN` replaces `GameBoyCNN` / `GameBoyBlock` / `GameBoyOptimizedCNN` (all deleted). Linear `Conv2d(3,32,k=4,s=4,p=0)` patchify stem — one 4×4 scaled-px patch is exactly one Game Boy tile — then 3 residual blocks at 32ch, a stride-2 down to `(64,9,10)` where one cell is one player step, a 1×1 squeeze to 16ch, and a flatten (not a pool) into `Linear(1440,128)`. Raises `ValueError` rather than padding when a side is not a multiple of 4 | `PoliwhiRL/models/CNN/GameBoy.py` |
| `RAMEncoder` gains an input `LayerNorm(222)` and a configurable hidden width — `ram_hidden_dim` 192 (was 128), `ram_input_norm` true. `d_ram` stays 128, so `fuse` is unchanged | `PoliwhiRL/models/PPO/PPOTransformer.py` |
| Observation resize filter `Image.LANCZOS` → `Image.BOX`, hardcoded and exposed as `OBS_RESIZE_FILTER` / `OBS_RESIZE_FILTER_NAME` | `gym_env.py`, `get_screen_image` |
| The `vision` config key and the `get_game_area()` tile-ID observation are deleted; the rendered screen is always the observation | `gym_env.py`, all configs |
| 2,021,193 → **818,421** parameters. `cnn.fc` falls from 1,474,688 (73.0% of the net) to 184,448 (22.5%); the conv body rises from 5,184 (0.26%) to 49,200 (6.0%); the transformer is untouched at 445,696 | measured, pinned by `tests/test_tile_cnn.py` |

**Why BOX.** Measured, not assumed: at `scaling_factor` 0.5 a tile is 4×4
scaled px and 160→80 / 144→72 are exact halvings, so BOX is an exact 2×2 area
average with zero cross-tile leakage. LANCZOS leaked **0.75 tile on all four
sides at 5.490% of peak**, so a tile's pixels reached into its neighbours'
patch vectors and the stem's tile isolation was false for every training frame.
Separately measured on the live emulator across 28 settled frames and 10 maps:
the patch-to-tile offset is (0,0) and SCX/SCY are 0 mod 16, and the camera does
**not** clamp on maps smaller than the screen — it drives SCX/SCY negative and
holds the player at a fixed screen cell — so the alignment holds on the interior
maps the curriculum starts in.

**Flatten, not pool.** The screen is player-centred and never clamps, so a
cell's grid position means a fixed direction-and-distance from the player.
Global pooling would discard exactly that.

## What had to be regenerated

`sil_demo_files` are consumed as stored pixels: `_load_sil_demos` reads the
`images` array and the SIL update feeds it straight through the network, so the
eight LANCZOS-era `.npz` files were training the new encoder on a distribution
the env can no longer produce. They were regenerated with
`tools.llm_agent.demos --chain` and now carry an `obs_resize_filter` stamp;
only `images` changed (`ram` / `actions` / `rewards` / `target_flag` are
bit-identical, 1060 steps unchanged). `tools/preflight.py` fails on a missing
or mismatched stamp. The `rc/*.state` reverse-curriculum rings are PyBoy save
states replayed live, so they were never stale and were left alone.

## How to launch

```bash
uv run python -m pytest tests -q
uv run python tools/preflight.py --config configs/stages/open_world_v4.json
uv run python main.py --use_config configs/stages/open_world_v4.json
```

`configs/stages/open_world_v4.json` is byte-identical to `open_world.json`
except its `output_base_dir` / `load_checkpoint` (both `open_world_v4/`) and an
explicit `ram_hidden_dim` / `ram_input_norm`. Every reward, step budget, SIL,
reverse-curriculum, Go-Explore, LR, KL and grad-clip value is held identical to
v3, so the encoder and the filter are the only variables. Pre-flight also prints
the model note (`model: 818,421 params, tile-grid encoder (conv strides
[4, 1, 1, 1, 2, 1], input (3, 72, 80), 8 actions)`) so a wrong geometry is
visible before committing hours to the run.

## How to judge it

**Judge it primarily against the success criteria already written above**, not
against v3: honest flag 31 well ahead of v2 per environment step, then honest
Route 31 / Violet entries on the leg that has no demos, and the exploration
sources still leading the reward mix. Read them on **honest** episodes bucketed
over episode order, with the seeded fraction per bucket alongside.

v3 is the only legacy baseline, and comparing against it **confounds the encoder
change with the BOX filter change**. Treat it as secondary evidence. The filter
is a minor perturbation for the legacy encoder (its first conv is 3×3 stride 2
and does not care about tile alignment), so the comparison is mostly about the
CNN — but a null result cannot be attributed cleanly.

**Attribution caveat.** v4 changes three things at once: the encoder,
`ram_hidden_dim` 192, and the RAM input LayerNorm. A win or a loss cannot be
attributed to any one of them without further arms. If the result is ambiguous,
the cheapest disambiguation is a third arm at `ram_hidden_dim: 128` with
`ram_input_norm: false`. Note the LayerNorm is the one change carrying an
identified risk: the RAM vector is 222 wide and largely sparse binary flags, and
`LayerNorm` normalises across the feature dimension per sample, so it can
amplify the few nonzero entries. It is behind a config key for exactly that
reason.

New sections in `tools/training_health_report.py` for this run — see
[`operations/run-triage.md`](operations/run-triage.md):

- `=== Stall location ===` — where honest episodes actually end, `stagnation`
  and `battle_stagnation` bucketed separately, with the deepest flag fired
  before stalling. This is what decides whether a plateau is one gate or diffuse.
- `=== Trust region ===` — how many PPO epochs actually complete, with replay-
  and KL-rejections kept out of the early-stop count.
- `-- Architecture provenance --` — `param_count`, `obs_resize_filter`,
  `ram_hidden_dim`, `ram_input_norm`, `git_rev`, `config_sha256` and the encoder
  identity, so a future triage never has to guess what produced the run.
