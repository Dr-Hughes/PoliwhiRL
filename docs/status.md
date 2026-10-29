# Status

## open_world_v3 — first run to beat the gym (2026-10-29)

From scratch, launched 2026-10-28 21:22 from commit `789cdba`; training code
is unchanged since. By episode ~300 (~2,390 PPO updates), honest
(non-seeded) episodes have beaten **Falkner (flag 1213) 5 times** and got
**Flash (flag 20) 8 times**, with recordings under
`Training Outputs/open_world_v3/Runs/checkpoints/`. Over the 192-episode
honest window: flag 30 79%, 39 64%, 31 46%, 66 38%; 55 new-species
catches, and 66% of episodes win at least one trainer battle. Probes at
updates 800 / 1600 show flag 66 at 0.9 / 0.3. Improvement stopped around
episode 200, with policy entropy drifting 1.3 → ~1.65. The coefficient was
halved 0.02 → 0.01 at update 2400 to compensate (commit `8d567af`);
`configs/stages/open_world.json:93`.

## What we want

Honest episodes from the true start that walk past Gym 1 — Sprout Tower and
Falkner are reached, now **reliably and repeatedly**, rather than five times
out of ~300 episodes. Beyond that: further into the game, on general-purpose
rewards rather than hand-authored per-milestone targets.

## The bottleneck

**Reached Falkner, but not yet reliably.** Five honest Falkner wins in
~300 episodes, eight Flash gets, and the probe ladder still reports flag 66
falling (0.9 → 0.3 between updates 800 and 1600). Improvement stopped around
episode 200 with entropy still drifting upward, so the bottleneck is now
retention and generalisation past the point already reached, not the
navigation that got there.

The previous bottleneck — the Route 30 gap at (26:1, x 5, y 24–25) — is
resolved. Route 29 → Cherrygrove → Route 30 → Route 31 → gate 26:11 → Violet
was a ~250-tile walk with no story reward between the last milestone and
Violet, and γ = 0.999 discounted that gap to almost nothing. The v3 run
walks it.

## What would show it is working

- Honest Falkner and Flash rates climbing per 25-episode bucket and staying
  there, rather than peaking and fading.
- The probe ladder's "Beat Falkner (Gym 1)" rung > 0 on the base (not just
  assisted) probe, at every probe rather than at one.
- Flags 31/65/66 and honest coverage holding while this happens.

Warning signs: the starter choice still swinging (flags 27/29 in the
health report's checkpoint table), honest coverage or Mr. Pokémon's House rate falling (an early
leg regressing), entropy running away, map-rung reward appearing in honest
episodes while honest coverage falls (a farm).

## How to test

```bash
uv run python tools/training_health_report.py "Training Outputs/open_world_v3/Checkpoints/info.pth"
```

Read the "Competence / reverse curriculum" section: per-flag honest rates,
the RC rung lines, and "honest map-rung entries per 25 honest episodes" for
`map 26:2` (Route 31) and `map 10:5` (Violet). The live progress bar's
`rc=` field shows the same rates (`m26:2@k…h12%`). Unit tests:
`uv run python -m pytest tests -q`.

## Previous status

The warm-started `open_world` run (2026-10-25 → 27) is archived in
[`archive/status-pre-v3-2026-10.md`](archive/status-pre-v3-2026-10.md).
