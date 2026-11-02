# v6 run verification (2026-11-01)

Independent verification of the `open_world_v6` diagnosis
(`Training Outputs/_analysis/v6_20261101/` verdict docs +
`REVIEW_final.md`) by read-only recompute over the final checkpoint plus live
PyBoy re-simulations. Full report and evidence:
`Training Outputs/_analysis/v6_verify_20261101/`
(`VERIFICATION_FINAL.md`, `ws1_results.json`, `ws2_results.json`,
`arms/ws3_arm_results.json`; the run data and this analysis are gitignored,
`.gitignore:168`).

## Conclusion

The v6 diagnosis is directionally right but wrong in its specifics: the
honest-coverage plateau, the shaping refund, the frozen Go-Explore pool, the
chronic KL early-stopping and the debug-evaluator memory bug are all real and
now measured against the final checkpoint; the post-Falkner "progression" was
never trained (0.0 on all 11 post-Falkner probe goals, and the reverse
curriculum never seeded a single episode due to a manifest-format gap), the
aux head is broken on `battle_started` (0.0251) behind a healthy aggregate,
the egg-handover result does not survive live replay (2/4, not 4/4), and the
diagnosis's per-eighth tables are stale. No perception defect was found. The
Option-B A/B confirms `ppo_target_kl 0.03` removes the chronic KL-cut with no
in-window coverage regression.

## Verdict table (compact)

| # | v6 claim | Verdict | Key numbers |
|---|---|---|---|
| 1 | Coverage plateau | CONFIRMED (6 map banks, not 29) | 1,333 cells; door cell (10,1,9,4) the only missing cell (32.3% of 331 door rows show warp); gate `rewards.py:2288` |
| 2 | Shaping refund drain | CONFIRMED | last-96 ledger −1,046.17 honest / −1,066.32 mixed; refund intercept 25.866; per-step slopes not persisted |
| 3 | Pool exhaustion | CONFIRMED (symptom) | pool frozen: 109 snapshots, min live count 25 vs capture rule `count<=3` |
| 4 | Egg handover / PP-lock | CONFIRMED-raw / REFUTED-behavioural | live 2/4 handover (f31 at steps 37, 1,313); PP-lock not reproduced (0 in 6 replays) |
| 5 | KL early-stop truncates budget | CONFIRMED | 3.4553/4 accepted epochs at halt; 93.43% of 1,339 cuts < 0.03; CNN grad share 49.4% at halt (not 83.7%) |
| 6 | Aux head healthy | PARTIALLY REFUTED | aggregate 0.9822 masks battle_started 0.0251 (n=358) |
| 7 | Post-Falkner progression / RC seeding | REFUTED | 0.0 on all 11 post-Falkner goals at probes 800/1600/2400; 0 rc-seeded episodes (`rc_seed_manifests: []`, no manifest JSONs on disk) |
| T4c | debug_evaluator mems bug | CONFIRMED + fixed | old-vs-fixed KL 0.090 nats mean / 0.50 p95 over 4,096 steps; fix + test landed |

## Option-B A/B (200 rollouts each, from the rollout-3000 checkpoint, LR
cosine restarted at 3e-4 in both arms)

| metric | v6 baseline | A (0.02) | B (0.03) |
|---|---|---|---|
| mean accepted epochs | 3.4553 | 3.1364 | 3.8283 |
| full-epoch share | 0.5537 | 0.2828 | 0.8535 |
| KL-cut < 0.03 share | 0.9343 | 0.9648 | 0.0 |
| hard rejects | 0 | 0 | 1 |
| final entropy | 0.8393 | 0.8250 | 0.6332 |
| honest unique_maps H1/H2 | 25.29/29.93 | 35.20/31.90 | 25.85/28.65 |

B confirms the mechanism (85% full-epoch, no sub-0.03 cuts); A drifts worse
than baseline; no coverage regression below the v6 e6 dip in either arm.
B's entropy sharpening (0.95 → 0.63) is the monitoring item.

## Next run (ranked)

1. **Train the post-Falkner segment** — create companion manifest JSONs
   `{"captures":[{flag, rc_rank, path}]}` for the two existing raw states
   (`emu_files/states/post_falkner.state`,
   `emu_files/states/ruins_inner_chamber.state`) and add `rc_seed_manifests`
   config entries; the pool deletes files of evicted trajectories (135
   observed in the arms) and preflight passes vacuously today — the smoke must
   show non-zero rc-seeded episodes.
2. **`ppo_target_kl` 0.03** (A/B confirmed; monitor entropy — B at 0.63 after
   200 rollouts).
3. **Fix or re-weight the broken `battle_started` aux class** — apply the
   per-class logging patch first (`Training Outputs/_analysis/v6_verify_20261101/patches/per_class_aux_logging.patch`);
   the series land in `info.pth`/`episode_data`, not the metrics JSON.
4. **Unfreeze the Go-Explore pool** — raise `count_max` from 3 toward the live
   distribution (~10) or a relative rule; min live count 25 makes capture
   impossible today.
5. **Cap or make monotonic the frontier reward** — it pays re-sweeps of the
   1,333-cell archive (break-even 710 true-new cells vs ~0.72/ep supply).
6. **Decide the door-cell merge** — (10,1,9,4) never merges via the
   `script_active` gate, so Route 32's door cannot be a frontier target.
7. **Decide `reset_lr_scheduler_on_load` semantics** — every resume restarts
   cosine at 3e-4 and both arms showed fast entropy falls under it.
8. **Copy, don't reference, run snapshot dirs** — rc-pool pruning deletes
   referenced files in place; a run's own `goexplore_snapshots` is not a safe
   read-only source.

## Corrections to the diagnosis

Stale per-eighth tables (doc snapshot 2026-11-01 11:45 at rollout 2950/395
vs final 3,000/402); handover 4/4 → live 2/4; PP-lock not reproduced; CNN
grad share 83.7% → 49.4% at halt; archive 29 → 6 banks; pool min-live 5 → 25;
quarter stagnation 0.69→0.92 → 0.77/0.86/0.95/0.92; final entropy ~1.4 →
0.8393; "41 zero-merge transitions" and the per-step slopes dropped (no
provenance / not persisted); tile grid 192 → 180 dims; held-Φ 23.277 → 23.281.

Full detail, probe ladders, provenance and the per-eighth tables:
`Training Outputs/_analysis/v6_verify_20261101/VERIFICATION_FINAL.md`.
