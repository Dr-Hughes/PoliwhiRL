# Run triage

What this page is for: how to judge a run from persisted artefacts rather than
headline reward — the evidence chain, an array-by-array field guide, the
honest-versus-seeded rule that governs every claim, the failure signatures with
the exact fields that expose them, and the prescribed analysis order condensed.

**Sources**: `tools/training_health_report.py`,
`.opencode/skills/training-run-analysis/SKILL.md`, `tools/world_map.py`,
`configs/default_configs/outputs_settings.json`.

## The rule that outranks everything else

`episode_seeded[idx] is False` means the episode is an **honest true-start**
from the canonical state; `True` means the episode began from a Go-Explore
frontier snapshot (`tools/training_health_report.py:288-291`,
`tools/training_health_report.py:468`). **Seeded episodes prove downstream
reachability only — never an end-to-end route from the canonical start**
(`.opencode/skills/training-run-analysis/SKILL.md:70-72`). Any statement about
what the policy "can do" must be backed by honest episodes; report honest and
seeded numbers separately
(`.opencode/skills/training-run-analysis/SKILL.md:65-72`).

Wording discipline (`.opencode/skills/training-run-analysis/SKILL.md:157-161`):
"ever reached" may include seeded; "reached honestly" means an unseeded episode
fired the flag; "reliably learned" requires repeated honest probe success.

## Evidence chain

| Artefact | What it holds | Notes |
|---|---|---|
| `<run>/Checkpoints/info.pth` | checkpoint-aligned `episode_data`, `visit_archive`, persisted frontier pool, first-hit recording manifest (`:32-33`) | loaded with `weights_only=False` at `tools/training_health_report.py:531`; source: `.opencode/skills/training-run-analysis/SKILL.md:32-40` |
| `<run>/Results/metrics/training_metrics.json` | the same fields, written more often | may be slightly newer than the checkpoint — `.opencode/skills/training-run-analysis/SKILL.md:34-35` |
| `<run>/Runs/checkpoints/<flag>-<title>/` | the first honest replay: `actions.steps`, metadata header, per-action PNGs | `.opencode/skills/training-run-analysis/SKILL.md:36-37` |

Small episode-count differences between these are normal; never silently combine
series from different episode counts, and name which artefact supports each
conclusion (`.opencode/skills/training-run-analysis/SKILL.md:39-40`). If prose
and implementation disagree, the implementation wins and the prose is the bug
(`.opencode/skills/training-run-analysis/SKILL.md:25-26`).

## Field guide

All names below are keys of `info.pth["episode_data"]` unless stated.

| Array | Shape / meaning | Source |
|---|---|---|
| `rollout_policy_entropy` | base-policy entropy in nats; the entropy the report bands | `training_health_report.py:86` |
| `rollout_behaviour_entropy` | assisted (seeded/replay) behaviour entropy | `training_health_report.py:87` |
| `rollout_entropy_coef` | live entropy coefficient; pinned-at-floor/ceiling fractions | `training_health_report.py:88,103-105` |
| `rollout_approx_kl` | approximate KL, mean + max | `training_health_report.py:90,122-126` |
| `rollout_clip_fraction` | PPO clip fraction | `training_health_report.py:97,175-179` |
| `rollout_lr` | learning rate; only first and latest are printed | `training_health_report.py:98,180-184` |
| `rollout_base_post_kl`, `rollout_behaviour_post_kl`, `rollout_base_replay_kl`, `rollout_behaviour_replay_kl` | exact post-update and collection/replay KL splits | `training_health_report.py:91-94` |
| `rollout_replay_rejected` | per-update replay-rejection booleans, by quarter | `training_health_report.py:95,149-169` |
| `rollout_kl_rejected` | PPO epochs that breached the hard KL limit and were rolled back → **CRITICAL** | `training_health_report.py:96,170-174` |
| `episode_reward_sources` | per-episode per-channel reward dict | `training_health_report.py:287` |
| `episode_seeded` | honest/seeded flag, index-aligned to the above | `training_health_report.py:288` |
| `episode_archive_size` | Go-Explore archive size per episode | `training_health_report.py:300` |
| `discovery_log` | run-wide first-ever milestones, each with `episode`, `type`, `seeded` | `training_health_report.py:319-325` |
| `episode_flag_fire_steps` | per-episode list of `[flag_number, step]` pairs | `training_health_report.py:352`; `.opencode/skills/training-run-analysis/SKILL.md:142-143` |
| `probe_goal_rates` / `long_probe_goal_rates` | per-event rate dicts for the short and long-horizon probe ladders | `training_health_report.py:390,415` |
| `assisted_probe_goal_rates` / `assisted_long_probe_goal_rates` | same, for assisted probes | `training_health_report.py:404-406,445-447` |
| `checkpoint_recordings` | first-honest replay manifest: `status`, `output_folder` | `training_health_report.py:495-503` |
| `frontier_pool` | persisted Go-Explore frontier snapshots (count only) | `training_health_report.py:504-506` |
| `best_reward` | top-line reward next to the episode count | `training_health_report.py:538` |
| `visit_archive.cells` / `.maps` | archive cell and map counts | `training_health_report.py:539-542` |

In `training_metrics.json` the equivalents are `seeded` and `reward_sources`
(`.opencode/skills/training-run-analysis/SKILL.md:67-68,96-68`).

## Failure signatures

| Signature | Exact fields | Where it is computed |
|---|---|---|
| Reward hacking / milestone re-fire farming | the `milestone` bucket **this tool** accumulates — `flag`, `checkpoint`, `checkpoint_honest`, `checkpoint_seeded`, `pokedex`, `level` (`training_health_report.py:208-215`) — that are *not* decaying while `frontier` is flat → `WARNING: milestone reward isn't decaying relative to frontier`. That bucket is `tools/training_health_report.py`'s own grouping, **not** the canonical taxonomy: `pokedex` is priced under the `catching` domain by the balancer (`rewards.py:752`; `domain_reward_balancer.py:102-103`) and `rewards.py` never writes a bare `checkpoint` key — only `checkpoint_honest` / `checkpoint_seeded` (`rewards.py:461-462`, `:717-718`). See [`../subsystems/rewards.md`](../subsystems/rewards.md) for the canonical channel/domain table. | `training_health_report.py:208-222` (channel sets), `:256-265` (warning) |
| Non-monotonic-counter farm | `delivered_total` that does not reconcile with the raw channel sum plus `rounding_adjustment`, `clip_adjustment`, `delivery_cast_adjustment` → `conservation error` > 1e-4 | `training_health_report.py:194-199`, `:234-255` |
| Collapse | aggregate reward rising while honest coverage falls (honest maps/cells, honest flag fires) | `.opencode/skills/training-run-analysis/SKILL.md:84-85`; arrays at `training_health_report.py:287-291,300` |
| Exploration plateau | `episode_archive_size` growth ≤ 0 across the final quarter **and** `discovery_log` with no honest entry within `max(20, n_episodes // 4)` | `training_health_report.py:304-314`, `:343-347` |
| Instability | entropy outside the band (default `[0.6, 1.2]`) for most rollouts → `:118-119`; `rollout_entropy_coef` pinned at the floor for >30% → `:120-121`; rising replay rejection (`final-quarter rate ≥ 5%` and above the first quarter) → `:164-169`; any `rollout_kl_rejected` truthy → **CRITICAL epoch rollback** `:170-174` | `training_health_report.py:118-121,164-174`; band/floor defaults `:523-526` |
| Horizon bottleneck (not a failure) | a probe rung whose long-horizon rate exceeds the short rate by >0.15 → `NOTE: … the short probe's horizon is hiding it` | `training_health_report.py:428-438` |
| Episode budget binding | deepest milestone fire past ~60% of `episode_length` → `TRIGGER` | `training_health_report.py:373-383` |
| Recording integrity | a live first hit whose replay status is `checkpoint_not_reproduced`: the checkpoint was reached, but the saved action prefix did not replay deterministically | `.opencode/skills/training-run-analysis/SKILL.md:174-176` |

Note the intentional false-positive guard: milestone and frontier can both fall
legitimately, because frontier pays only for genuinely fresh cells — the farming
signature is milestone staying flat/high, not frontier failing to strictly
overtake (`training_health_report.py:271-282`).

## Analysis order (condensed)

1. Implementation beats prose; if they disagree, flag the doc drift
   (`.opencode/skills/training-run-analysis/SKILL.md:25-26`).
2. Pick one coherent snapshot — `info.pth`, `training_metrics.json` or a
   `Runs/checkpoints/<flag>-<title>/` replay — and state its episode counts
   (`SKILL.md:28-40`).
3. Run the health report first, from the repo root
   (`SKILL.md:42-49`; flags `SKILL.md:51-58`; section order
   `training_health_report.py:544-552`).
4. Split honest from seeded before interpreting anything
   (`SKILL.md:65-72`).
5. Trend over four-to-eight chronological buckets — reward, unique maps and
   cells, flag fires, episode length, archive size — not the tail mean
   (`SKILL.md:74-83`).
6. Read entropy, KL, clip fraction and LR together; stable optimisation
   statistics only rule out one failure class
   (`SKILL.md:87-92`).
7. Decompose reward by channel, first quarter vs last quarter, keeping
   checkpoint/event/frontier/new_map/battle/counter separate
   (`SKILL.md:94-105`; implementation `training_health_report.py:187-222`).
8. Establish frontier progress from all three of `episode_archive_size`,
   `discovery_log` recency and `visit_archive` counts
   (`SKILL.md:112-122`).
9. Read probes by checkpoint identity with numerator **and** denominator; a
   two-episode 100% is weak evidence (`SKILL.md:124-134`).
10. Correlate with RAM evidence by full `(map_bank, map_num)` pair plus event
    flag number, joining `episode_flag_fire_steps` indices to
    `episode_seeded` (`SKILL.md:136-148`).
11. Validate first-hit recordings: `seeded` false, `status` recorded,
    `checkpoint_observed` true, `actions_path` exists, PNGs present, fire step
    agrees with `episode_flag_fire_steps` (`SKILL.md:163-172`).
12. Report as the fixed table — Area / Honest evidence / Seeded evidence /
    Assessment / Recommended action — with rows for optimisation, honest
    coverage, reward mix, archive growth, deepest checkpoint, probe
    reliability, frontier-pool health and recording integrity, ending with the
    five direct answers (`SKILL.md:178-193`).

Do not recommend raising every reward because a checkpoint is rare; first
establish whether the successful trajectory already out-earns comparable
episodes and whether seeding or horizon is the real bottleneck
(`.opencode/skills/training-run-analysis/SKILL.md:107-110`).

## Known gaps

- [`docs/reference/ram-mapping.md`](../reference/ram-mapping.md) (formerly
  `RAM_MAPPING.md`) and the skill's own `references/ram-correlation.md` are cited by
  the skill as authoritative (`SKILL.md:20,138`). Both pointers have since been
  repointed into `docs/reference/` (`SKILL.md:20` and `:140` name
  `docs/reference/ram-mapping.md`; `references/ram-correlation.md:8` links back to
  it), so there is no stale-path gap left to close here — the only remaining
  documentation hygiene item is the in-code comment pointers tracked in
  [`../roadmap.md`](../roadmap.md) under **Open work → Docs**.
- `checkpoint_title()` and `is_recordable_checkpoint()` gate the story-checkpoint
  table (`training_health_report.py:51-54,471,490`); the flag→title mapping in
  `PoliwhiRL/checkpoints.py` was not transcribed here.
- Whether `episode_lengths` is always present for the auto-inference of
  `--episode-length` (`training_health_report.py:534-535`) is unverified.
- The `_quartile_slices` remainder rule (`training_health_report.py:57-76`)
  makes "last quarter" larger than a strict quarter for short runs; thresholds
  quoted above inherit that behaviour.
