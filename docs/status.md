# Status

Last updated 2026-10-27 20:00. Run: `Training Outputs/open_world/` (config
`configs/stages/open_world.json`), resumed at episode 1277, log
`training_run_20261027_140127.log`. Numbers below are honest (non-seeded)
episodes only; re-derive them before acting — the run is live.

## What we want

Honest episodes from the true start that walk on to **Violet City (map 10:5)**
after "Learned To Catch" (flag 66), reliably — then Sprout Tower and Falkner.

## Where it is (episode ~1504)

- Starter choice settled (Cyndaquil 60/60 recent honest episodes).
- Flags 30 / 31 (200-ep window): 72% / 64%, up from 49% / 45.5% at the
  starter fix; latest 20-ep bucket 18/20 for flags 30–66, the best yet.
- Coverage ~16–19 maps / ~470–520 cells per honest episode.
- Route 31 (26:2): ~1 per 20 honest episodes; gate 2 and Violet 2 in the last 60.
- Entropy drifting up ~1.2 → ~1.5; KL ~0.015. Archive ~1500.

## The bottleneck

Nothing in the game blocks the trip after flag 66 (no item or flag needed).
Route 29 → Cherrygrove → north up Route 30 through a 1-tile gap at
(26:1, x 5, y 24–25) → west along Route 31 → gate 26:11 → Violet, ~250 tiles.
Everything before the gap is already walked that episode, so the coverage
reward is weak there. Time is not the limit: honest episodes that reach flag
66 (median step ~8k) keep going a median ~14.7k more steps, then 69/71 end
on the no-new-cell stagnation timeout without finding the gap. Violet is
4.5k–13k steps after the last story reward, and γ = 0.999 discounts a reward
that far ahead almost to nothing. Honest
episodes found the route (eps 670–712: Route 31 7/16, Violet 3 times) and
then lost it again.

## What we've tried (2026-10-25 → 27)

| Change | Result |
|---|---|
| Reward rebuild: balancer off, competence-gated checkpoints, species catch reward, self-imitation (SIL), reverse-curriculum starts | Flags 31/65/66 up to ~55%; SIL was inert until the MPS copy bug was fixed |
| Run-wide frontier decay | Reverted — zeroed the walk back to Elm and collapsed flag 31 to 0% |
| Practice starts: hysteresis (enter <60%, exit 80%) + map first-entries as targets | Route 31 and Violet became targets |
| Long-range "which way is unwalked" sense from ROM collision BFS | Points at the Route 30 gap on a real save state; no clear behaviour change yet |
| Route 31 / Violet entry reward (40, checkpoint-priced); practice starts only from the approach side; longer SIL lookback | Brief recovery (0 → ~12% Route 31), then lost to the flip below |
| Starter flags 27/28/29 priced and targeted as one checkpoint (`canonical_checkpoint`) | Worked: choice settled within ~60 eps, flags 30–66 recovered to ~65–90% per bucket. Priced per flag, the rarer starter paid full while the habitual one sat at the floor, so the choice flipped Cyndaquil ↔ Chikorita every ~500 eps and each flip cut flags 30–66 |

Next: find where the ~15k post-flag-66 steps go (recorded-episode frames
via `tools/world_map.py`) — circling Cherrygrove, walking back to Elm, or
reaching Route 30 and missing the gap.

Not tried: seeded episodes inheriting the visit history at capture,
honest/seeded split of event-novelty counts, feeding probe successes into
practice starts.

## What would show it is working

- Honest Route 31 entries climbing past ~25% per 25-episode bucket and
  staying there, rather than peaking and fading.
- Honest Violet entries in most buckets, then the probe ladder's
  "Reached Violet City" > 0 on the base (not just assisted) probe.
- Flags 31/65/66 and honest coverage holding while this happens.

Warning signs: the starter choice still swinging (flags 27/29 in the
health report's checkpoint table), honest coverage or Mr. Pokémon's House rate falling (an early
leg regressing), entropy running away, map-rung reward appearing in honest
episodes while honest coverage falls (a farm).

## How to test

```bash
uv run python tools/training_health_report.py "Training Outputs/open_world/Checkpoints/info.pth"
```

Read the "Competence / reverse curriculum" section: per-flag honest rates,
the RC rung lines, and "honest map-rung entries per 25 honest episodes" for
`map 26:2` (Route 31) and `map 10:5` (Violet). The live progress bar's
`rc=` field shows the same rates (`m26:2@k…h12%`). Unit tests:
`uv run python -m pytest tests -q`.
