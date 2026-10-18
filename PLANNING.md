# PoliwhiRL — status and plan (2026-10-18)

Branch: `flag31-breakthrough-and-tuning` (on top of `redo-configs`).
See `CHANGELOG.txt` for the detailed per-commit breakdown of the work
this covers.

## Current state

The live production run (`open_world`, background task `b91wc1n91`,
`Training Outputs/open_world/`) is still running on the **unmodified**
`episode_length: 16384` config — the `episode_length -> 24576` change
in this branch is committed to the repo but **has not been applied to
the running process**. The user explicitly asked to stop before
restarting it, so no process-level action should be taken on this
until asked.

As of episode 2703 (2026-10-18 ~08:23):

- 9 honest flag-31 fires total (episodes 2086, 2157, 2161, 2241, 2246,
  2348, 2372, 2382, 2470) — the egg-delivery blocker that held for
  ~2000 episodes before the restart described in `CHANGELOG.txt`.
- Honest progress has reached flag 66 (Learned to Catch) once (episode
  2157), but `honest_last100_mean_pokedex_owned` is still pinned at
  ~1.0 — no second Pokémon has been honestly caught yet despite
  reaching the catching tutorial.
- Archive size: 662 (flat since ~episode 2520).
- **No honest flag-31 fire since episode 2470** — a ~230-episode gap,
  the longest since the breakthrough started recurring.
- `current_policy_entropy` has drifted down to ~1.30 (still above the
  1.0 intervention floor, but the lowest reading since the restart —
  worth watching, not yet acting on).
- `last_approx_kl` ~0.0005, `last_clip_fraction` 0.0 — stable.
- Disk: ~91-92GB free, no pressure.

## Diagnosis: why flag-31 recurrence stalled

Traced the last 150 honest episodes' `truncation_causes`:
**budget-truncation share is 42.7%**, up from ~14-32% at the original
10-hour diagnosis (back when `stagnation` truncation at ~82% was
clearly the dominant failure mode and budget was ruled out as a lever).

Cross-referencing the raw flag-fire logs for episodes after 2470: many
honest episodes don't pick up the Mystery Egg (flag 30) until step
10,000-13,500 of the 16,384-step episode. Episode 2086's successful
run needed ~8,400 steps between egg pickup (flag 30) and delivery
(flag 31). An episode that doesn't have the egg by roughly step 8,000
cannot mathematically also complete the backtrack before truncating.

This is a genuinely different bottleneck than the one diagnosed before
the breakthrough — the critical path got *longer* once flag 31 became
reachable (starter -> egg -> deliver -> catching tutorial -> actual
catch -> Violet City -> Gym 1), and `episode_length` wasn't re-sized
for it.

## Plan going forward

1. **Apply the `episode_length: 24576` change to the live run** (stop
   `b91wc1n91`, relaunch `uv run python main.py --use_config
   configs/stages/open_world.json` — auto-resume picks up the
   checkpoint, archive, and frontier pool with zero progress lost).
   This is evidence-based and low-risk (the stagnation-truncation limit
   scales automatically via `"auto"`, and 24576 is already exercised by
   `long_probe_episode_length`), but **requires the user's go-ahead**
   before touching the running process — do not restart unilaterally.
2. **Re-run the same truncation-cause trace after a few hundred more
   episodes** at the new length to confirm budget share drops back down
   and honest flag-31 recurrence resumes. If it does, episode_length
   was the correct lever and no further reward changes are needed for
   this specific gap.
3. **If flag-31 recurrence still doesn't resume** despite the longer
   budget, re-open the question of whether the early-game portion
   (leaving the house -> starter -> Mr. Pokémon's house) itself needs
   attention — the variance in when flag 30 fires (step ~250 to
   step ~13,500+) suggests the early legs of the episode are themselves
   inconsistent, which is a separate problem from the egg->Elm
   backtrack and would need its own trace-based diagnosis rather than
   another length bump.
4. **Catching progress (pokedex_owned stuck at 1.0)** is the next real
   gap once flag-31 recurrence is healthy again: honest episodes reach
   "Learned to Catch" but aren't yet completing an actual second catch.
   Worth a dedicated trace (does the episode run out of budget again
   before a wild encounter after the tutorial, or does the agent fail
   to execute the catch action itself) before considering any reward
   change here — same evidence-first discipline as the trainer-battle
   work.
5. **Do not re-introduce the stuck-temperature retune**
   (`stuck_action_temperature: 9.0` / `onset_exponent: 0.5`) — it was
   tested head-to-head against the unchanged baseline and made
   stagnation measurably worse (92.3% vs 67.9% honest stagnation-
   truncation, 1 vs 6 honest flag-31 fires over the comparison window).
   The `stuck_temperature_onset_exponent` knob itself is worth keeping
   (general, backward-compatible, zero cost at its default), but no
   evidence currently supports moving it off 1.0.
6. **General principle for whatever comes next**: prefer reward/config
   mechanisms that generalize across milestones (domain-weighted
   rewards, checkpoint-style decay, "auto"-scaling limits) over
   hand-authored, area-specific fixes — consistent with this session's
   approach and the project's own stated preference in `AGENTS.md`.
   Validate every change with a direct trace of `trajectory.csv` /
   `truncation_causes` / `flag_fire_steps` before acting, not by
   assumption.

## What NOT to do

- Don't restart or modify the live run without explicit confirmation —
  it's the long-haul production run and the user has asked to pause
  here.
- Don't assume the `open_world_probe_temp` config/output directory is
  still relevant — it's a retired, one-shot experiment (see
  `CHANGELOG.txt`), not a second production path.
- Don't re-litigate the stuck-temperature question without new evidence
  — it was tested properly and falsified for this specific problem.
