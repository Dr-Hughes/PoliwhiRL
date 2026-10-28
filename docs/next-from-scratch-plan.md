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
