# Next steps — mid-run notes on 00_freeform_scratch_new (2026-07-11)

## Two-stage game-area run is set up and validated — see bottom of doc

The next run to actually execute is the two-stage curriculum described in
"New: two-stage curriculum + game_area (tile-ID) input" near the end of
this file. Read that section for the exact run commands. Everything
between here and there is the analysis that led to it.

## Status: a training run is IN PROGRESS as this is written

`configs/stages/freeform_scratch.json` currently points at
`./Training Outputs/00_freeform_scratch_new/` with `episode_length: 2048`,
`num_rollouts: 2048`. Everything below is analysis of that run's
in-progress checkpoint (312 episodes / rollout 1249 in at time of
writing) plus code changes made in parallel. **The config/code changes
below do NOT affect the currently-running process** (it already has its
config and module bytecode loaded) — they apply the next time training is
(re)started. Do not erase `00_freeform_scratch_new` or edit its config
while it's still running.

## Headline result: the starter sequence fired on-policy

For the first time across three runs, training itself (not just recorded
episodes) reached flags 26/27/28 (`got_starter`/`got_cyndaquil`/
`got_totodile`) and pokédex threshold 1, six times across episodes
123–236 — e.g. episode 123: flag 28 (totodile) at step 1384, flag 26
(got_starter) at step 1581. Previous runs (512-step episodes) never got
past the mom flag (1735) even after 768 episodes. Contributing changes,
in likely order of impact:

1. **`episode_length` 512 → 2048.** The deepest milestone this run has
   ever reached lands at step 2040 of 2048 (99.6% of budget) — episode
   length was very plausibly the binding constraint in every prior run,
   exactly as the trigger in the previous version of this doc predicted.
2. **Directional frontier-novelty features** (`rewards.py`
   `directional_frontier_potential` / RAM vector
   `frontier_potential_{up,down,left,right}`): the policy can now
   perceive, in each of 4 directions, the exact frontier payout it would
   get for stepping that way — collapsing a long, noisy
   walk-then-correlate-with-delayed-reward credit-assignment problem into
   a direct per-step signal. Town + all 3 interiors (Elm's Lab, neighbour's
   house, Elm's house) were discovered by episode 63 this run, vs. spread
   across the full 256–768 episodes in prior runs.
3. **Periodic archive decay enabled** (`curriculum_base.json`:
   `archive_decay_rate: 0.9`, `archive_decay_frequency: 50`, ~330-rollout
   half-life). Directly implements "novelty should be bounded, not a
   permanent cross-episode debt" — known ground's frontier payout now
   recovers over time instead of decaying to a fixed residual forever.
4. Milestone re-fire depletion (previous session) is doing its job:
   reward-mix trend, first quarter vs last quarter of episodes — flag
   87.6 → 37.9, frontier 100.3 → 86.5. Frontier has overtaken milestone
   reward as the dominant signal; no farming equilibrium detected.

## Bug found: the regular probe's horizon can't see this progress

`probe_episode_length` is still 512 while training runs at 2048. The
probe's `pokedex: 0%` across every event in this run is NOT a competence
signal — the starter fires at step ~1400–1900 on-policy, which the
512-step probe structurally cannot reach. **Don't read the regular probe's
pokédex/deep rungs as "the policy can't do this" until probe_episode_length
is fixed** — it's measuring the wrong window, not the wrong policy.

Fix for the next run (not applied to the live one):
`"probe_episode_length": 2048` (match training), and turn on the new
long-horizon probe as the direct test of "would this policy progress
further given more runway":

    "long_probe_enabled": true,
    "long_probe_frequency": 256,
    "long_probe_episodes": 3,
    "long_probe_episode_length": 4096

## New tool: `tools/training_health_report.py`

Standalone, offline, reads only `info.pth` (no emulator, no weights).
Answers "how do we monitor for collapse / plateaus" without hand-rolling
a new analysis script every session:

    python tools/training_health_report.py "Training Outputs/00_freeform_scratch_new/Checkpoints/info.pth" --episode-length 2048 --probe-episode-length 512

Reports: entropy-band adherence + coefficient clamp-pinning, reward-mix
trend (flags milestone-farming signatures), archive-growth plateau,
discovery-log staleness, time-to-frontier (median AND max deepest fire
vs episode_length — use max, median is dominated by shallow recurring
milestones), and probe-vs-long-probe rung comparison (flags when a rung
is probe-horizon-limited rather than competence-limited, exactly the bug
above). 12 unit tests on synthetic data (`tests/test_training_health_report.py`).

Current read on the live run (rollout 1249): entropy in-band only 29% of
the time, coefficient pinned at the floor 60% of rollouts, archive flat
for the whole final quarter (108 cells since ~episode 100), no new
discovery in 174 episodes. This reads as a genuine exploration plateau
within the explored region (town + 3 interiors) — expected, since that
region is now fully mapped and the next frontier (Route 29 / leaving
town) may be gated by game logic on having a starter, which the run has
now started to satisfy only in the last third of episodes. Re-run the
health report once training finishes to see whether archive growth
resumes.

## Open question: is the CNN appropriately sized for GBC input?

Looked at `PPOTransformer.py`'s `GameBoyCNN` (via `GameBoyBlock` in
`models/CNN/GameBoy.py`): two stride-2 conv+GroupNorm+ReLU blocks (16
then 32 channels), NO residual/skip connections despite the "ResNet-style"
docstring — it's a plain shallow encoder, not a deep one. Parameter count
is dominated by the flatten→FC bridge (11520→128 at the default
`scaling_factor: 0.5` RGB input), not by conv depth. Verdict: the encoder
itself is not overcomplicated.

The more consequential question is the INPUT REPRESENTATION: `vision:
true` feeds scaled RGB screen pixels; pyboy also exposes `game_area()` —
an 18×20 grid of background tile IDs (`vision: false` path, already
wired in `gym_env.py`, currently unused by any active config). For a
tile-grid, low-color native game, tile IDs are a lower-dimensional and
exactly-discriminating representation (no risk of two different rooms'
door frames looking pixel-similar) vs. RGB pixels the CNN has to learn
room-identity from. Caveat before switching: this project has no
Crystal-specific pyboy game wrapper (see `tools/world_map.py`'s
docstring — it reconstructs the map from movement deltas instead of tile
decoding for exactly this reason), so `game_area()` almost certainly
returns background tiles only, NOT composited sprites — the player marker,
NPCs (including mom), and battle sprites would likely vanish from the
image channel entirely. Much of that is already covered by RAM features
(position, facing, battle state, enemy HP), but it's unverified whether
enough survives for e.g. NPC-avoidance/interaction. Recommend a cheap,
reversible ablation — a short run with `"vision": false` — checking the
resulting `obs["image"]` visually (or via a quick script) for sprite
presence before trusting it, rather than switching by default.

## What to check once the current run finishes

1. Re-run `training_health_report.py` — does archive growth resume once
   the starter sequence starts firing more reliably? Does Route 29 (or
   another new map) get discovered?
2. Reward mix — does `flag` keep decaying toward pokédex/level taking
   over as the "new" milestone source, the way `flag` did from
   `frontier`?
3. Probe — once `probe_episode_length` is fixed for the NEXT run,
   confirm the regular probe's pokédex rung actually lifts off (it was
   structurally blind before).
4. Long-horizon probe (once enabled) — does the pokédex rung show up
   there before it shows up on the regular probe? That's the direct
   "would longer episodes help" test the model owner asked for.

## Deferred (with triggers)

- **Save-state / backward-expanding starts**: trigger (median/max deepest
  fire vs budget) has now fired at 2048 length too (99.6%). Worth
  revisiting once this run's final numbers are in — if the starter
  sequence consistently eats nearly the whole budget, that's the
  strongest case yet for snapshot seeding.
- **World-map / discovery-graph as training input**: still offline-only.

## New: two-stage curriculum + game_area (tile-ID) input (2026-07-11, later session)

Two follow-up requests, both now implemented and validated (274 tests
green, plus live end-to-end smokes — details below):

### 1. Stage 1 (as just run) → Stage 2 (same weights, 10x episode length, fewer rollouts)

The curriculum system already supports this cleanly: the visit archive
persists across stage loads (exploration knowledge carries forward), the
reward scaler resets per stage by default (new episode length = new
reward-magnitude distribution), and the entropy servo/plateau counters
reset per stage. Two real gaps were found and fixed while checking this
was actually safe (not just "should work"):

1. **`best_success_window`'s deque was hardcoded to `maxlen=100`**,
   ignoring the config value entirely, and — a separate, worse bug — even
   the config-driven version wouldn't have survived a checkpoint reload:
   `load_model`'s schema-migration code kept a RELOADED deque's OLD
   maxlen instead of rebuilding it at the new stage's window (`isinstance
   (value, deque)` was already true post-unpickle, so the rewrap branch
   never ran). A stage that completes far fewer episodes than 100 (any
   10x-longer-episode stage, by construction) would never write a `best/`
   checkpoint at all. Fixed: the deque always rebuilds at
   `best_success_window` on load now
   (`PoliwhiRL/agents/PPO/vec_ppo_agent.py`, `reset_tracking` +
   `load_model`).
2. **The entropy-plateau window had a hardcoded `max(50, ...)` floor** —
   a stage whose total episode budget never reaches 50 completed episodes
   could never satisfy `len(series) >= window_size`, so the plateau/boost
   mechanism (the exact thing that broke the RGB run out of its
   "talked_to_mom" camping loop around episode 190) would be silently
   inert for the whole stage. Fixed: floor is now
   `entropy_reset_window_floor` (config, default 50 — unchanged
   behaviour unless a stage overrides it).

Both fixes are covered by dedicated tests (`test_best_and_capture.py::
TestLoadModelWindowRebuild`, `test_anti_convergence.py::
test_entropy_reset_window_floor_enables_short_stage_resets`), and a full
tiny-scale stage1→stage2 chain (real `PyBoyEnvironment` +
`setup_and_train_PPO`, 2 rollouts each, `episode_length` 20→200 to prove
the ratio) was run end-to-end: stage 1 trained, wrote `best/`, stage 2
loaded it ("Visit archive carried over: N cells") and kept training at
the new episode length without error.

`mem_len` (TXL memory depth, 128) is deliberately left unchanged for the
long stage — exploration/progress state is carried through explicit RAM
features (steps_since_novel_cell, recent_maps, directional frontier
potential, n_*_goals_completed), not raw attention over the whole
episode, so a fixed-depth memory window was already designed to not need
to span the full episode.

### 2. `game_area` (tile-ID) input, config-swappable

Investigated whether the CNN was overbuilt for GBC's low-res/low-colour
input (see "Open question" above). Finding: the conv encoder itself is
fine (2 shallow blocks, no real depth); the actual lever is input
representation. `vision: false` already existed as the config switch to
pyboy's `game_area()` tile-ID grid, but was untested and — confirmed by a
live smoke — **broken**: `output_shape()` returned a bare `(18, 20)`
2-tuple with no channel dimension, which `GameBoyCNN` misreads as 18
channels of a channel-less image and fails at construction with a Conv2d
shape error. Fixed in `PyBoyEnvironment.get_game_area()` (`gym_env.py`):
now returns channel-first `(1, 18, 20)` uint8, matching
`get_screen_image()`'s `(C, H, W)` convention. No other change was
needed — `GameBoyCNN` already derives its shape from `input_shape` at
construction time, so the SAME model class now handles both input modes;
there is no second "replica" model. `vision` (already a config key) is
the swap.

Verified live: raw `game_area()` values are tile IDs 0–92 (52 unique),
change correctly as the player moves between rooms; `flat_features`
drops from 11520 (RGB) to 800 (tile grid) — a real, substantial compute
win, as hoped. One caveat, unresolved and worth checking empirically
before fully trusting this mode: this project has no Crystal-specific
pyboy game wrapper, so `game_area()` is almost certainly background tiles
ONLY — sprites (player marker, NPCs, battle sprites) likely don't appear
in the image channel. Much of that is already covered by RAM features,
and this does NOT affect the recorded debug PNGs (`save_step_img_data`
reads `pyboy.screen.image` directly, completely independent of `vision`
— visual episode review is unaffected either way). Regression tests:
`test_pyboyenv.py::test_no_vision` (updated to the corrected shape) +
`test_game_area_input_shape_builds_a_valid_cnn` (pins the exact crash)
+ `test_vec_env.py::TestVecPPOAgentSmokeGameArea` (full multiprocess
training-loop smoke, real PyBoy, both input modes).

### The two configs to run

- `configs/stages/freeform_gamearea.json` — mirrors the just-completed
  RGB run's scale (2048 rollouts, 2048-step episodes) but with `"vision":
  false`. Also fixes the probe-horizon bug found in the RGB run:
  `probe_episode_length` now matches training length (2048, was
  incorrectly 512), and `long_probe_enabled: true` at 4096 steps — the
  direct "would more runway help" test.
- `configs/stages/freeform_gamearea_long.json` — loads stage 1's `best/`,
  `episode_length: 20480` (10x), `num_rollouts: 1024` (half of stage 1 —
  wall-clock per rollout is ~constant regardless of episode_length, since
  a rollout is a fixed `ppo_update_frequency`-step slice, so this really
  is roughly half the wall-clock for a run that reaches 10x deeper).
  `entropy_reset_window_floor: 10` and `best_success_window: 10` sized to
  this stage's own ~25-completed-episode budget (`num_rollouts × num_envs
  × ppo_update_frequency / episode_length`) — both would otherwise be
  structurally unreachable (see fixes above). `probe_episode_length:
  20480` (matches training), `long_probe_episode_length: 40960` (2x —
  tests extrapolation beyond even the new training length).

Run:

    python main.py --use_config configs/stages/freeform_gamearea.json
    python main.py --use_config configs/stages/freeform_gamearea_long.json

Run stage 2 only after stage 1 finishes and has written
`Training Outputs/00_freeform_gamearea/Checkpoints/best/` (stage 2's
`load_checkpoint` points there already). Both are independent of, and
safe to run alongside or after, the currently-running RGB
`00_freeform_scratch_new` — different output directories, and a running
process is unaffected by any of these file changes regardless.

## Post-mortem: gamearea_long run + episodic floor / stagnation truncation (2026-07-12)

Both stages above ran to completion. Stage 1 (2048-step) was healthy
throughout. Stage 2 (20480-step) surfaced a NEW failure class and, in its
last four episodes, also the strongest results of the whole project.
Full causal analysis below drove two reward-system changes (implemented,
288 tests green) and two explicit rejections.

### What happened in stage 2

- **Mid-run detachment trough (~eps 60-120):** recorded ep_104 spent
  18,408 of 20,480 steps pressing `down` into a wall at one cell of a
  fully-decayed room. Not an outlier: 15 of 40 episodes in that stretch
  had negative total reward, no run-first discovery for 66 episodes, and
  the archive NET-SHRANK (decay evictions outpacing discovery).
- **Late recovery (eps 121-124):** best episodes of the project — rewards
  810/1149/1108, 126-155 unique cells, all 7 maps, five run-firsts
  including a level-8 first at step 20,061. Probe hit flag=100% map=100%.
  What broke the detachment: archive decay slowly FORGAVE the explored
  region (novelty re-paid) + the ep-122 entropy boost (which fired with
  measured entropy ~1.17 < band top, i.e. in the regime where it has
  authority). So the system self-corrects — with a catastrophic latency
  (~60 episodes x 20,480 steps of near-waste).

### Root cause (generalises across every run to date)

The cross-run novelty decay is one-sided: it correctly REDIRECTS the
gradient at the frontier, but in fully-decayed regions it EXTINGUISHES
the local gradient entirely — every action pays the same (nothing), so
whatever degenerate attractor the policy happens to have (wall-bump,
two-cell pacing) is stable. Compounding: `directional_frontier_potential`
is 1-cell lookahead (goes dark exactly where flatness lives), and gamma
caps the credit horizon at ~200 steps (`_resolve_auto_gamma` clamps at
0.995) so distant frontier payouts can't propagate value back through the
desert either.

### Changes made (both in rewards.py, config in curriculum_base.json)

1. **Episodic novelty floor** (`frontier_novelty_floor`, default 0.2):
   frontier payout is now `bonus * max(1/sqrt(1+visits), floor)`, same
   floor on new_map, and `directional_frontier_potential` mirrors it so
   the observation stays truthful. NGU-style split: the EPISODIC signal
   (first visit this episode) never dies; the LIFETIME term only
   amplifies the frontier (max preference ratio 1/floor = 5x). "Cover
   ground you haven't covered yet this episode" is now self-sustaining
   income at any game depth — the general curiosity behaviour itself
   stays rewarded, per the model owner's stated goal (thorough explorer
   over golden-path speedrunner). Milestones deliberately do NOT get the
   floor — at 500-scale a floored milestone is the farming annuity again.
2. **Stagnation truncation** (`stagnation_truncation_steps`, default
   "auto" = max(256, episode_length // 16)): an episode that goes a full
   threshold of FREE-WALKING steps (scripted/battle frames freeze the
   counter) without claiming one first-this-episode cell is truncated —
   time-limit semantics, bootstrapped, NOT a terminal. Shape-agnostic
   damage cap: any absorbing loop, including ones we haven't met yet,
   costs at most the threshold instead of the episode budget. ep_104's
   18k wasted steps would have been ~17 fresh attempts.

### Explicitly rejected (so the reasoning isn't re-litigated later)

- **Collision-byte action masking:** the wall was incidental —
  absorption is the disease, and two-cell pacing is unmaskable. Collision
  bytes encode tile TYPES (ledges must be pressable, NPC blocking is
  dynamic, late-game water/surf) — masking on them trades an early-game
  symptom for late-game traversal bugs.
- **Entropy-servo "sign fix":** investigated as a bug, exonerated as
  design. Cut-during-boost when measured > band top is deliberate and
  pinned (`test_boost_never_pushes_past_band_top`); the run's own
  recovery used the boost exactly as designed. The servo's real
  limitation stands (aggregate mean nats can't see one absorbed env) —
  addressed structurally by the two changes above, not by re-tuning the
  servo. Expect the plateau detector to fire RARELY post-floor; if it
  still saturates, that's the signal to delete it, not re-tune it.

### Still deferred, with updated triggers

- **Frontier snapshot seeding (Go-Explore starts):** the time-to-frontier
  trigger has now fired at BOTH 2048 (100%) and 20480 (86%) — but the
  episodic floor changes the calculus (the desert pays again, so the
  corridor is learnable without teleporting) and the model owner prefers
  learned traversal over seeded starts. Revisit ONLY if a post-floor run
  still shows the detachment signature (negative-reward episodes +
  net-shrinking archive) at scale.
- **BFS distance-to-frontier observation feature:** the archive is
  already a graph (world_map.py). Revisit if post-floor runs plateau with
  healthy per-episode coverage but no frontier progress.

### Next run

Same two configs. Point `output_base_dir` at fresh directories (or clear
the old ones) — stage 1's `load_checkpoint` is empty so it trains from
scratch either way, but stale `Checkpoints/` in the same dir would be
overwritten confusingly. Watch for: (1) truncation shortening the average
episode length well below budget early on (expected, healthy — it means
absorbed episodes are being recycled), (2) whether the reward-mix
`frontier` share stays dominant, (3) whether the plateau detector fires
at all.

### What to check once stage 1 (gamearea) finishes

1. Sprite-visibility caveat above — worth a quick visual check (dump a
   `game_area()` frame around a moment with mom or an NPC on screen; the
   recorded PNGs from the SAME run show the true screen for comparison).
2. `training_health_report.py` against stage 1's `info.pth` — does the
   milestone-farming pattern/frontier-takeover trend look like the RGB
   run's, given the much cheaper input representation?
3. Once stage 2 runs: does the long-horizon probe's pokédex/map rungs
   exceed the regular probe's, and do both improve over stage 1's probe
   numbers? That's the direct answer to "does scaling episode length
   keep progressing" — the model owner's stated bar for calling this
   approach successful.

## Post-mortem: probe archive-sync bug + wall-attraction root cause (2026-07-14)

Two real, previously-undiscovered bugs found this session, both fixed.

### Bug 1: probes evaluate against a permanently empty visit archive

`vec_ppo_agent.py`'s `_flush_visit_archive` broadcasts the run-wide
`VisitArchive` to `vec_env` (the real training workers) at rollout start,
but `_probe_env`/`_long_probe_env` never received that broadcast anywhere
in the file — every probe episode, in every run, was scored against a
fresh, empty archive, an observation distribution the policy never
trains on. This is very likely why the pokédex-owned probe rung read 0%
for the whole project despite `got_starter` (flag 26) firing in 82% of
the last 50 real training episodes at the time of writing — probes were
never measuring in-distribution behaviour. Fixed: `_run_probe_pass` now
syncs `env.visit_archive.load_state(self.visit_archive.to_state())` at
the top of every call, mirroring the existing `vec_env` broadcast. This
only affects diagnostics, not training — it will not retroactively
correct any already-running process's in-memory probe numbers (it applies
to any freshly-launched run), and doesn't need a restart to justify
itself.

### Bug 2 (the important one): `directional_frontier_potential` has no
reachability check, and a wall reads as the single best move forever

This is the root cause of the "picks a wall and walks into it repeatedly"
pattern that both the episodic novelty floor and stagnation truncation
(see above) only ever bounded the damage from, never fixed. The compass
forecasts each neighbour cell's payout via raw coordinate arithmetic
against `VisitArchive.count()`. A wall/obstacle tile is a cell the
player's (X, Y) can never equal, so its visit count is — permanently,
not just early in training — 0. That plugs into the exact same formula
as the single freshest cell in the game: `max(1/sqrt(1+0), floor) == 1.0`,
the ceiling. Unlike over-explored-but-reachable ground (which the floor
correctly keeps learnable), a wall's count never grows, so it never even
enters the regime the floor was designed for — it sits at the raw,
undecayed maximum forever. This is a static function of map geometry, not
of training progress, so no amount of additional training changes it —
consistent with the model owner's report that the behaviour hasn't faded
despite very substantial training since the floor/truncation fixes
landed.

An independent subagent analysis (read-only, no code access) converged on
the same root cause from a cold read of the code, and its proposed fix
was adopted over an initial collision-byte-based approach that had been
prototyped and empirically verified (bedroom/Cherrygrove/Route
30/46/Pokémon Lab manual states, walked — not tapped — repeatedly:
`collision_byte=0` always moves, `7`/`255` always block with zero
exceptions, `24`/`112`/`113` are legitimate walkable terrain (ledges,
warps, stairs) that move successfully every time). That approach was
rejected as the shipped mechanism specifically because it would require
trusting an undocumented byte encoding that was only sampled across a
handful of early-game rooms — a whitelist risking silent false negatives
in tilesets never sampled (caves, gyms, water edges, later regions).

**Fix shipped** (`rewards.py`): ground-truth position tracking, not
collision-byte interpretation. `Rewards` now tracks `_prev_pos` and
`_blocked_direction` — whichever of up/down/left/right was just pressed
and left `(map_bank, map_num, X, Y)` unchanged, gated on the same
free-walking condition already used for the stagnation counter (script/
battle frames never register a false block — dialogue legitimately locks
movement for reasons unrelated to collision). `directional_frontier_potential`
forces that direction's forecast to `0.0` instead of the archive-derived
value; a non-directional action leaves the known block as-is; any
successful move clears it. This self-corrects every step and needs zero
assumptions about tile semantics — a ledge, NPC, cut-tree, or un-surfed
water tile all read "blocked now" and clear the same way (the first
successful step through). One bump per direction is still possible (the
mechanism only knows a direction is blocked after trying it once); the
*repeated* bumping the model owner reported is what this eliminates.
7 new tests in `tests/test_wall_dead_end_compass.py`; full suite (295)
green.

### Restart vs. continue

Recommended: **continue from the existing checkpoint chain, do not
restart from scratch.** This is a forecast/observation correction, not a
reward-scale or milestone-structure change — the realized reward path
(`_frontier_novelty_bonus`) was never wrong (a wall-bump always correctly
paid ~0), so the value function's calibration is not invalidated. The
existing policy's substantial learned competence (got_starter at 82% of
recent episodes, 9 maps, level 11, reliable multi-map exploration) is
orthogonal to this bug and not worth discarding. Expect a short
re-convergence period as the policy stops needing to override a
now-corrected observation feature, not a retraining-from-zero event.

## Verifying the wall-dead-end-compass fix against the live run (2026-07-14)

`00_freeform_gamearea_xlong3` (PID started 00:34:08) was launched 5 minutes
*after* `rewards.py`'s mtime (00:29:47) and the config's — so the
in-progress process has the fix from the section above loaded; it is not
running stale bytecode. Confirmed by checking every modified tracked
file's mtime against the process start time, not just rewards.py.

### Bug found in the analysis tooling itself: `world_map.py` never inferred
walkability from real data

While using `tools/world_map.py` to empirically check for repeated
wall-bumping in recorded episodes, found its walkability inference was
dead code: `WorldMap.update()` gated on `player_state == "walking"`, but
recorded PNGs use `PLAYER_STATE_LABELS` values (`"walk"`/`"bike"`/
`"skate"`/`"surf"` — see `gym_env.py`) and never the literal `"walking"`.
The gate always failed, so `walkable`/`known_wall` were never populated
against any real run — silent since `test_world_map.py`'s fixtures
defaulted to the same wrong literal, so the tests validated a scenario
that never occurs in real data. Fixed: gate on `battle_type` instead
(mirrors the actual free-walking condition `rewards.py` uses for its own
stagnation/blocked-direction accounting), fixture default corrected to
`"walk"`, added a regression test that exercises all four real
`player_state` labels. Full suite green (296, up from 295 — one net new
test after replacing the now-nonsensical "non_walking_state" test).

### Wall-bump streak comparison, pre-fix vs post-fix (7 recorded episodes each)

Using the corrected tool, measured the longest run of consecutive
identical-direction, no-position-change presses (a direct proxy for "picks
a wall and walks into it repeatedly") in the 7 recorded episodes each
retains for `00_freeform_gamearea_xlong2` (entirely pre-fix) and
`00_freeform_gamearea_xlong3` (entirely post-fix, since the whole run
started after the fix landed):

| | pre-fix (xlong2) | post-fix (xlong3) |
|---|---|---|
| repeated-bump events (streak >= 2) | 4808 | 3059 |
| total steps spent mid-bump-streak | 19647 | 8934 |
| longest single streak | 81 | 21 |
| streaks >= 5 | 1180 | 343 |
| streaks >= 10 | 286 | 27 |

**Verdict: substantially improved, not eliminated.** The pathological
50-80-step dead attractors are gone (max streak 81 -> 21, streaks >= 10
down ~10x), consistent with the fix removing the permanent "this wall is
the best move forever" false signal. But it is NOT reading as zero, and
per the fix's own documented scope this is expected, not a sign the fix
is incomplete: `_blocked_direction` only zeroes a direction's forecast
*after* one failed attempt on it — it is an observation correction, not
an action mask — so a stochastic policy (mean rollout entropy ~1.06 nats,
see `training_health_report.py` output below) can still re-select a
just-blocked direction a handful of times before the zeroed forecast (plus
the policy's own on-policy update) pushes it away. If streak lengths don't
keep trending down over the next few thousand episodes, that would be the
signal to revisit (e.g. whether the compass value should propagate to
neighbouring cells/directions instead of only the exact one tried).

### Current run snapshot (2891 episodes in, via `training_health_report.py`)

Archive growth and discovery log are both flagged flat for the last
quarter/1000 episodes respectively — on its own this would read as a
plateau, but the reward-mix and probe numbers say otherwise: frontier
reward has overtaken milestone reward (167.8 vs 22.0 last-quarter avg, the
"OK" case, not the farming pattern), and the probe ladder is still at
flag=100%/map ~67-100% with the long-horizon probe's map rung matching or
exceeding the regular probe's — i.e. the policy keeps demonstrating the
capability, it's just not banking new run-wide firsts in the archive's 9
known maps right now. Time-to-frontier is still triggered (deepest fire at
65% of the 40960-step budget), reaffirming episode length as the live
constraint, unchanged from the assessment above. Entropy coefficient
pinned at floor 39% of rollouts — worth a re-check in a later session if
the servo warning persists once the wall-fix re-convergence settles.

## Episode-length follow-up: stagnation truncation is firing constantly, and why (2026-07-14)

Model owner noticed mean episode length (7942 steps, `episode_lengths` in
`info.pth`) is far below the 40960-step budget and asked whether this meant
the process was crashing/restarting. It isn't — same OS PID has run
continuously for 15+ hours; episode count climbs steadily. It's
`stagnation_truncation_steps` (`"auto"` -> `max(256, episode_length // 16)`
= 2560 steps) firing as designed: only 80/2891 episodes (2.8%) ever reach
the full budget, essentially all the rest end via this mechanism (not
whiteout — that reward source averages 0.0 all run). This is intentional,
documented above as the fix for absorbing loops burning the entire budget
on one bad attempt — truncation is a time-limit (value bootstraps), not a
punished terminal.

Model owner's response: rather than only capping the damage from a stuck
episode, prefer teaching the policy to escape a stall in-episode instead
of relying on truncation as a "rescue." Investigated whether this is
learnable rather than requiring a new heuristic override:

- The entropy servo (`_update_entropy_servo`) is global and rollout-level —
  one coefficient for all 16 envs from their aggregate mean entropy. It
  cannot single out one stuck env (already noted above as a known
  limitation) — confirmed, not a bug, just the wrong granularity.
- Found the SAME collapse one level deeper, in the loss itself:
  `ppo_model_implementation.py`'s entropy term computed correct per-timestep
  entropy from `new_probs` but immediately `.mean()`-ed it across the whole
  PPO minibatch *before* multiplying by `entropy_coef` — so even a
  perfectly-tuned servo coefficient could never give one stalled timestep
  more exploration pressure than a fine one in the same batch.
- Good news: the policy doesn't need a new "I'm stuck" signal — the RAM
  vector already carries `steps_since_novel_cell` (log1p-scaled steps since
  the current env last found a not-yet-visited-this-episode cell; see
  `Rewards.steps_since_novel_cell`, `gym_env.py` RAM_FEATURE_KEYS index 68).
  The gap was never perception, it was that nothing in training rewarded
  *using* that perception to try something different.

**Fix shipped**: `ppo_entropy_stagnation_boost` (config key, default `0.0`
= off, exact old behaviour). When > 0, `_compute_ppo_losses` weights each
timestep's entropy contribution by `1 + boost * steps_since_novel_cell`
before averaging, instead of a flat mean — concentrating exploration
pressure on transitions that are actually stalled, in proportion to how
stalled they are, without diluting into envs/timesteps that are already
progressing. General/shape-agnostic (works for wall-bumps, 2-cell pacing,
or any future absorbing pattern — same principle as the truncation cap),
and checkpoint-compatible (no observation or architecture change, so it
does not force a retrain from scratch). 3 new tests in
`test_anti_convergence.py::TestEntropyStagnationWeighting`; full suite
(299) green.

**Not yet enabled anywhere** — no config sets `ppo_entropy_stagnation_boost`,
so this is currently a no-op for every stage including the live
`00_freeform_gamearea_xlong3` run (which also wouldn't pick it up without
a restart regardless, per the mtime-vs-PID-start check used earlier this
session). Next step, when there's an appetite to spend a restart on it: pick
a stage/config to trial a boost value (e.g. 1.0, so a fully-stalled
timestep at the 2560-step stagnation limit — `steps_since_novel_cell`
feature value ~1.3 — gets its entropy weighted ~2.3x) and watch
`training_health_report.py`'s entropy-servo and archive-growth sections,
plus the wall-bump-streak measurement above, for whether stagnation streak
lengths/frequency actually drop.

## x3 wrap-up: stuck-in-battle failure mode, a tool bug it exposed, and x4 (2026-07-15)

`00_freeform_gamearea_xlong3` finished its full configured budget (8192
rollouts, 3825 episodes, `early_stopped: False` — not a crash). Two things
came out of reviewing it, plus a follow-up config.

### Finding: episodes were getting stuck in single, unwinnable-looking battles for tens of thousands of steps

Model owner's hypothesis: the agent runs out of move PP mid-battle and
doesn't back out/heal/switch, inflating episode length. Investigated by
going back to first principles rather than trusting the aggregate
`episode_lengths` numbers alone:

1. Noticed only 80/2891 episodes reached the full 40960-step budget at the
   session's earlier check, and asked why stagnation truncation (which
   should catch a stuck episode within ~2560 steps) wasn't firing on those.
   Re-read `rewards.py`'s stagnation-accounting gate (`not script_active and
   battle_type == 0`, see next_steps.md above) and realised: it's
   deliberately frozen during battle (so a genuine multi-turn trainer fight
   isn't truncated mid-fight) — but that means a battle that's stuck, not
   progressing, has NO ceiling short of the full episode budget.
2. To get direct evidence rather than reasoning from the mechanism alone,
   parsed the recorded-episode PNG filenames in
   `Training Outputs/00_freeform_gamearea_xlong3/Runs/N_goals_0/` (same
   `tools/world_map.py` filename parser, reused ad hoc rather than adding a
   new tool for a one-off check) and measured the longest continuous run of
   `battle_type != "none"` per recorded episode. Two of the ~15 sampled
   episodes had hit the full budget: `ep_2302` (19,988 consecutive
   battle-steps, 49% of the episode) and `ep_3404` (**36,202 consecutive
   battle-steps — 88% of the entire episode**, steps 962-40960, one
   uninterrupted wild encounter).
3. Confirmed it wasn't a productive long grind, not just a long fight: for
   `ep_3404`, the per-step `reward` encoded in each filename is flat at
   exactly the step penalty (-0.02) for the last ~35,000 steps — zero
   engagement/win reward after step 4958. Button-press histogram for the
   battle stretch: `a` 35599 times (98%), `b` 356, directionals under 100
   combined — heavily consistent with mashing the same (likely PP-exhausted
   or otherwise blocked) move-select option rather than backing out.
4. Checked whether the entropy-stagnation-boost fix (above) would even see
   this failure mode, since it reads `steps_since_novel_cell`, not the
   (battle-frozen) stagnation-truncation counter: confirmed
   `_frontier_novelty_bonus` (which maintains `steps_since_novel_cell`)
   runs unconditionally every valid step regardless of `battle_type`, so it
   keeps climbing through a stuck battle — the boost should apply rising
   exploration pressure there too, for free. Whether that's actually enough
   to break an 88%-of-episode loop, versus needing a battle-specific
   stagnation cap (e.g. no-HP-change-based, distinct from the free-walking
   one), is unverified and deliberately deferred rather than guessed at —
   revisit once x4 has run a while and can be checked the same way.

### Bug found in `training_health_report.py`: "last quarter" was really "the last episode"

While re-checking whether x3's progress was still trending up before it
stopped, `_quartile_slices(series, n_parts=4)` built slices by stepping
`range(0, n, size)` with `size = n // n_parts`. For `n` not an exact
multiple of `n_parts` (e.g. the real run's 3825 episodes, size 956), that
range yields FIVE stops (0, 956, 1912, 2868, 3824), not four — the fifth a
1-element remainder slice. Every caller (`report_reward_mix`,
`report_archive_growth`) reads `quarters[-1]` as "the last quarter," which
had silently become "the single last episode" instead. Caught by
recomputing the same figure two ways and getting wildly different answers
(a claimed step_penalty last-quarter average of -815/episode, versus -150
to -205 when the same window was sliced into deciles by hand) — the
discrepancy, not a code read, is what surfaced it. Confirmed directly:
`slices[-1]` held exactly one episode, whose own step_penalty was -815.46
(a ~40773-step episode — almost certainly one of the stuck-battle
episodes above). **Fixed**: last slice now absorbs the remainder, always
exactly `n_parts` slices. Regression test added
(`test_non_divisible_length_still_returns_four_slices`, pinned against the
exact n=13 case that used to spill a 5th sliver); full suite (300) green.

### So, was x3 still improving? (re-answered with the corrected tool + finer-grained checks)

Quartile/decile means of `episode_rewards`, `episode_unique_cells`,
`episode_unique_maps`, and `episode_reward_sources` (all already in
`info.pth`, no new instrumentation needed), plus the full history of
`probe_goal_type_rates`/`long_probe_goal_type_rates` (not just the last 3
events `training_health_report.py` prints) and the checkpoint's own
10-episode `moving_avg_reward` deque:

- **Exploration depth plateaued around episode ~1900 (roughly halfway)
  and stayed flat for the rest of the run.** Corrected last-quarter
  archive growth: -2 cells (flat/decay-noise). Last genuine discovery-log
  entry: episode 1891 (1934 episodes before the run ended). Deepest
  milestone ever reached (26585 steps) hadn't moved since a mid-run check
  at episode 2891. `episode_unique_cells` per episode rose sharply early
  (decile means ~34 -> ~89) then flattened (~108 -> ~107) — a real
  per-episode ceiling, not noise.
- **Per-episode competency kept improving.** Median `episode_rewards` rose
  every quarter (84 -> 89 -> 119 -> 184). `episode_unique_maps` per episode
  kept climbing too, though with diminishing returns (3.4 -> 5.1 -> 5.6 ->
  5.75).
- **Mean reward looked flat/noisy, and the stuck-battle failure mode is
  why**: the checkpoint's own `moving_avg_reward` (last 10 episodes at
  save time) had 4 of 10 values in the -463 to -567 range — i.e. as of the
  last thing this run did, roughly 40% of its most recent episodes were
  total-loss stuck-battle failures, not a capability regression.
- **The pokédex probe rung never fired, not once.** Checked every
  `probe_goal_type_rates`/`long_probe_goal_type_rates` entry across the
  whole run (66 + 25 events, not just the tool's last-3-events default) —
  pokedex-owned success rate is exactly 0.0 in all of them. A standing
  capability gap distinct from the battle issue, present from the start,
  not a late regression.
- **`best_reward` (1459.8, episode 42) is a red herring**, not evidence of
  early-training being better: it's an artifact of the intentional
  milestone-refire decay (an early, undecayed first-hit pays more than the
  same milestone re-fired later), not a sign later episodes played worse.

### x4 config

`configs/stages/freeform_gamearea_xlong4.json` added: same
episode-length/rollout budget as x3 (following the established
stage-to-stage pattern of only bumping `load_checkpoint`/
`output_base_dir` between stages), `load_checkpoint` pointed at x3's
finished `Checkpoints/`, plus `"ppo_entropy_stagnation_boost": 1.0` — the
trial value discussed above, now actually turned on for the first time.
Nothing else changed; the battle-specific stagnation question above is
deliberately left for a future config once there's evidence on whether
the entropy boost alone is enough.

## Battle-stuck fixes + in-episode stuck-recovery + enemy-HP bug (2026-07-15, later session)

Acting on the x3 stuck-battle post-mortem above. Four changes shipped
(320 tests green, up from 300; full offline unit coverage + a live
2-rollout vec smoke with every new key enabled + emulator RAM re-probes).
All are checkpoint-compatible (no architecture or observation-dim change),
so they apply whether the next run continues from x3 or starts fresh.

### 1. Enemy-HP byte-order bug (RAM.py) — VERIFIED against a save state

`get_enemy_hp`/`get_enemy_max_hp` read the wEnemyMon HP bytes in the wrong
order. Empirically (loaded `Manual Investigation States/in_wild_battle_
route_13.state`, a level-2 Sentret with 14 HP): the true value lives at
`0xD216=0` (high), `0xD217=14` (low) — big-endian — and `0xD217` counts
down 14→8→3→0 as it takes damage. The old code returned `(0xD217<<8)|0xD216`
= **3584** (256× inflated) for any enemy with HP < 256, i.e. every
early-game encounter, and went non-monotonic once the high byte was in
use (gym leaders). Impact was bounded, not fatal — `enemy_hp_ratio`
(cur/max) cancels the error since both were swapped identically, and KO
detection at exactly 0 still worked, so battle-win rewards always fired —
but the raw `enemy_hp` feature (RAM idx 37, `log1p/6`-scaled) the policy
sees was saturated garbage. Fixed to big-endian; `get_enemy_hp()` now
returns 14 on that state. Regression tests in `test_battle_watchdog.py::
TestEnemyHPByteOrder` (fake-pyboy, no emulator needed).

### 2. Battle-progress watchdog (rewards.py, config `battle_stagnation_truncation_steps`)

The free-walking stagnation truncation deliberately FREEZES during battle
(so a real multi-turn fight isn't cut), which left a stuck battle with no
ceiling short of the whole episode budget — the direct cause of x3's
ep_3404 (36,202 consecutive battle steps, 88% of the episode). New
battle-specific counter: consecutive battle steps with NO change in enemy
HP or party HP; ANY resolved turn (dealing or taking damage) resets it, so
a progressing fight is never truncated; a genuinely-stuck one is truncated
(time-limit/bootstrap, NOT a punished terminal — mirrors the free-walk
cut-off). Truncation ends the EPISODE (env resets, value bootstraps), NOT
the training run. This is NOT the primary escape route — it is a LOOSE
last-resort backstop so one pathological env can't pollute a rollout with
tens of thousands of identical dead transitions. `"auto"` =
`max(512, episode_length // 16)` (2560 at 40960), set well above where the
stuck-temperature ramp saturates (~1k steps) so in-episode random recovery
gets a long window to find AND repeat the escape (learned from the buffer;
truncation only fires when escape did NOT happen, so it never erases a
successful-escape transition) before the budget is reallocated.

### 3. Behaviour-time stuck-temperature exploration (vec_ppo_agent.py, config `stuck_action_temperature`) — THE HEADLINE

Directly implements the model owner's request: "introduce more randomness
in actions when clearly stuck so it can observe correcting states and
LEARN, rather than just avoid them / pollute the pool with an overabundance
of bad states." The existing `ppo_entropy_stagnation_boost` only reweights
the LOSS — it can't make the behaviour policy try anything different DURING
collection, so a deterministic stuck policy generates the same garbage
transition 36k times. The new mechanism raises a per-env action-sampling
TEMPERATURE keyed on `steps_since_novel_cell` (RAM idx 68 — climbs through
wall-bumps, 2-cell pacing, AND stuck battles alike, since position is
frozen in a fight; one general signal, no per-area tuning). Temperature
ramps 1.0→`max` (default 3.0) linearly above a raw-step `threshold`
(default 256). Implemented as `normalize(probs ** (1/T))` (softmax
temperature scaling), with the action mask re-applied afterward so
tempering can never resurrect a masked action, and the tempered
(behaviour) log-prob stored as `old_log_prob` so PPO's importance ratio
correctly discounts the injected off-policy exploration. The servo's
entropy diagnostic is taken from the UN-tempered policy so injected noise
doesn't fool it into cutting the coefficient. Off by default (temperature
0 => identity). Tests: `test_stuck_temperature.py` (off/below-threshold
identity, deep-stall entropy rise, mask preservation, cap).

### 4. Wild-battle flee reward (rewards.py, config `battle_flee_reward`)

Small positive (default off) on a WILD battle (`battle_type==1`) ending
with the enemy still alive and the party not whited out — a genuine
escape. Catching is out of scope for this curriculum and wild battles are
pure traversal obstacles, so "get out and keep exploring" is the aligned
outcome; it shortens the credit path for the escape action the
stuck-temperature mechanism discovers. Cannot apply to trainer battles
(you can't flee them), so it can never let the agent skip the mandatory
rival fight. Folded into the same `battle_reward_episode_cap` as
engagement/win. Tests in `test_battle_watchdog.py::TestFleeReward`.

### PP information — considered, deliberately NOT expanded

The policy ALREADY sees `player_pp1..4` (RAM idx 39-42, /64). Inspecting
`in_wild_battle_route_13.state` shows the ambiguity that limits their
value: a 2-move mon reads `pp=[35,30,0,0]` — a `0` means "empty slot" OR
"depleted", indistinguishable. Adding a disambiguated PP/"all-moves-out"
feature would change `RAM_OBS_DIM` and break checkpoint compatibility
(forcing a from-scratch retrain) for marginal gain — the stuck signal the
agent actually needs (`steps_since_novel_cell`) already climbs through a
PP-exhausted battle, and the stuck-temperature mechanism acts on it. Left
as-is on purpose.

### Config: enhanced `freeform_gamearea_xlong4.json` (continue from x3)

x4's dir is empty (never run) and no training process is live, so x4 is
the clean next run. Added: `battle_flee_reward: 1.0`,
`battle_stagnation_truncation_steps: "auto"`, `stuck_action_temperature: 1.0`
(+ max 4.0, threshold 256), alongside the pre-existing
`ppo_entropy_stagnation_boost: 1.0`, and `num_envs` 16 -> 24 (M3 Ultra: 28
CPU cores / 96 GB / 60 GPU cores — CPU-core-bound since each env is a PyBoy
process; 24 leaves headroom for the learner/OS and grows the PPO batch to
24x128=3072). Ordering by design: temperature ramps from step 256, saturates
(~temp 4.0) around step 1024, and the watchdog only truncates at 2560 — so
random recovery gets a ~1500-step window at high randomness to break AND
repeat a stall before the backstop reallocates the budget. The watchdog is
subordinate to the learning mechanism, not the front-line fix.

### CRITICAL follow-up: the action mask was blocking battle escape (2026-07-15, x4 live-run diagnosis)

Model owner watched x4 (~65 episodes in, ep 3890) still sit in a battle for
thousands of steps pressing only `a`/`b`/noop — no directionals. Root cause
found by running x4's ACTUAL policy on the wild-battle save state with a deep
stall injected: the policy was NOT peaked (entropy 1.03) and the
stuck-temperature was firing — but `compute_action_mask` was zeroing all four
directionals. A stuck battle is dialog-heavy (move-select + attack/`no PP`
text = `script_active & text_box`), and the mask's dialog rule blocks
directionals — but the battle menu (RUN, move-switch) is D-pad driven, so the
agent was PHYSICALLY UNABLE to select escape, and no amount of temperature/
entropy can raise a masked action above 0. The mask (meant to stop walking
during overworld cutscenes) was fighting the battle escape the other three
changes were trying to teach.

**Fix** (`action_mask.py`): battle overrides the dialog block — when the
battle one-hot (`battle_wild`/`battle_trainer`) is set, directionals stay
allowed regardless of the text-box byte. Verified: with the fix, the same
policy on the same state puts **96.6% on directionals** (up 0.53, right 0.28,
A only 0.03) — i.e. it was already TRYING to navigate the menu and the mask
was suppressing it. This is the actual front-line fix for battle escape; the
temperature/watchdog/flee are support. 4 tests in
`test_action_mask.py::TestBattleMask`; start/select stay blocked in battle.
NOTE: the running process must be restarted to pick this up (bytecode was
loaded pre-fix); x4's ~65 pre-fix episodes should be discarded.

### Scratch vs continue — recommendation: CONTINUE from x3 (run x4)

The changes are checkpoint-compatible, and the plateau hypothesis is that
the battle sink is what stopped x3 crossing the wild-battle-dense routes
(29/30) — so continuing directly tests whether removing the sink unblocks
the frontier, while preserving x3's expensive navigation competence
(town+interiors, got_starter, 9 maps, level 11) that a from-scratch run
would spend thousands of episodes just to re-earn before the battle fixes
could even be exercised on the routes. The entrenched "mash-A" prior is
exactly what the stuck-temperature mechanism reshapes (it injects
exploration the moment that loop stalls), so continuation pressures the
bad behaviour rather than inheriting it passively.

WATCH, first few hundred episodes of x4: (1) mean episode length should
DROP (battle watchdog recycling stuck fights); (2) `battle` reward-source
should stay within its 15.0 cap with flee contributing; (3) does the
archive finally grow past 9 maps / does the probe reach Cherrygrove
(map 24/4-5)? If after ~500 episodes the archive still won't grow past 9,
THAT is the trigger to restart from scratch as the clean-slate test (copy
`freeform_gamearea.json`, add the four new keys, empty `load_checkpoint`).
