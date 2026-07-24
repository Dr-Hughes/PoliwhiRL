#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline health-check report over a checkpoint's ``info.pth``.

NOT used by training — a standalone diagnostic that turns the ad-hoc
"pull info.pth, eyeball the entropy trace, eyeball the reward mix"
analysis this project does after every run into a repeatable command,
so a stalled run can be triaged without hand-authoring a new script each
time. Every check reads only data already logged in ``episode_data`` /
``visit_archive`` (see ``VecPPOAgent.reset_tracking`` /
``VisitArchive.to_state``) — no emulator, no model weights.

Checks:

1. Optimisation health — measured rollout entropy and coefficient, approximate
   KL, clip fraction and learning-rate behaviour where available.
2. Reward-mix trend — milestone (flag/pokedex/level) vs frontier vs
   new_map share, first quarter of the run vs last quarter. A milestone
   share that hasn't decayed relative to frontier is the exact farming
   signature from the 2026-07-11 "talked_to_mom" equilibrium (see
   Rewards._milestone_refire_scale).
3. Archive growth — flags a flat ``episode_archive_size`` tail (no new
   cells recorded for the last portion of the run) as a plateau.
4. Discovery log — how many genuine run-wide firsts, and how long ago
   (in episodes) the most recent one landed.
5. Time-to-frontier — median deepest milestone-fire step vs episode length
   (episode length may be binding once this crosses roughly 60%).
6. Story checkpoints — honest/seeded reach, recency, first-honest replay status
   and persisted Go-Explore frontier-pool size.
7. Probe ladder — latest per-checkpoint rates, plus (if a long-horizon
   probe ran) a side-by-side comparison. A rung that's flat on the short
   probe but climbs on the long-horizon probe means the capability is
   already there and probe/episode length is the bottleneck, not
   competence — the exact mismatch a probe with too short a horizon
   would otherwise hide (see VecPPOAgent._run_long_horizon_probe).

Usage (from repo root):

    python tools/training_health_report.py "Training Outputs/open_world/Checkpoints/info.pth"
    python tools/training_health_report.py path/to/info.pth --episode-length 2048 --probe-episode-length 512
"""
import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PoliwhiRL.checkpoints import (  # noqa: E402 - repo root inserted above
    checkpoint_title,
    is_recordable_checkpoint,
)


def _quartile_slices(series, n_parts=4):
    """Split a series into exactly n_parts contiguous chunks, the last of
    which absorbs any remainder.

    Stepping range(0, n, size) instead produces a stray extra slice
    whenever n isn't an exact multiple of size (e.g. n=3825, n_parts=4 ->
    size=956 -> a 5-element range yielding a final 1-episode sliver). Every
    caller here indexes quarters[-1] as "the last quarter" for a trend
    comparison; against the old implementation that silently became "the
    single last episode" instead, which is not a trend and is exactly the
    kind of episode most likely to be a pathological outlier (a stuck
    battle running the full episode budget) rather than representative.
    """
    n = len(series)
    if n == 0:
        return []
    size = max(1, n // n_parts)
    slices = [series[i * size : (i + 1) * size] for i in range(n_parts - 1)]
    slices.append(series[(n_parts - 1) * size :])
    return slices


def _mean(xs):
    xs = list(xs)
    return statistics.fmean(xs) if xs else 0.0


def report_entropy(ed, band_low, band_high, coef_floor, coef_ceiling):
    print("\n=== Optimisation ===")
    ent = ed.get("rollout_policy_entropy", [])
    coef = ed.get("rollout_entropy_coef", [])
    boost = ed.get("rollout_entropy_boost", [])
    approx_kl = ed.get("rollout_approx_kl", [])
    clip_fraction = ed.get("rollout_clip_fraction", [])
    learning_rates = ed.get("rollout_lr", [])
    if not ent:
        print("  no rollout_policy_entropy data")
    else:
        in_band = sum(1 for e in ent if band_low <= e <= band_high) / len(ent)
        at_floor = sum(1 for c in coef if c <= coef_floor * 1.05) / max(1, len(coef))
        at_ceiling = sum(1 for c in coef if c >= coef_ceiling * 0.95) / max(
            1, len(coef)
        )
        boost_frac = sum(1 for b in boost if b) / max(1, len(boost))
        print(f"  mean entropy: {_mean(ent):.3f} nats (band [{band_low}, {band_high}])")
        print(f"  fraction of rollouts inside band: {in_band:.0%}")
        print(f"  fraction of rollouts with coef pinned at floor:   {at_floor:.0%}")
        print(f"  fraction of rollouts with coef pinned at ceiling: {at_ceiling:.0%}")
        print(f"  fraction of rollouts under a plateau boost: {boost_frac:.0%}")
        if in_band < 0.4:
            print("  WARNING: entropy spends most of its time outside the target band.")
        if at_floor > 0.3:
            print("  WARNING: coefficient frequently pinned at the floor.")
    if approx_kl:
        print(
            f"  approximate KL: mean={_mean(approx_kl):.4f}, max={max(approx_kl):.4f}"
        )
    if clip_fraction:
        print(
            f"  clip fraction: mean={_mean(clip_fraction):.3f}, "
            f"max={max(clip_fraction):.3f}"
        )
    if learning_rates:
        print(
            f"  learning rate: first={learning_rates[0]:.3g}, "
            f"latest={learning_rates[-1]:.3g}"
        )


def _report_reward_mix_group(sources, label):
    print(f"\n  -- {label} --")
    if not sources:
        print("  no episode_reward_sources data")
        return
    quarters = _quartile_slices(sources, 4)
    first, last = quarters[0], quarters[-1]
    keys = sorted({k for ep in sources for k in ep})
    print(f"  {'source':<14}{'first-¼ avg':>14}{'last-¼ avg':>14}")
    milestone_first = milestone_last = 0.0
    frontier_first = frontier_last = 0.0
    for k in keys:
        f_avg = _mean(ep.get(k, 0.0) for ep in first)
        l_avg = _mean(ep.get(k, 0.0) for ep in last)
        print(f"  {k:<14}{f_avg:>14.2f}{l_avg:>14.2f}")
        if k in ("flag", "checkpoint", "pokedex", "level"):
            milestone_first += f_avg
            milestone_last += l_avg
        if k in ("frontier",):
            frontier_first += f_avg
            frontier_last += l_avg
    print(f"  milestone total: {milestone_first:.2f} -> {milestone_last:.2f}")
    print(f"  frontier total:  {frontier_first:.2f} -> {frontier_last:.2f}")
    milestone_depleting = milestone_last > 0 and milestone_last < milestone_first * 0.8
    if (
        milestone_last > 0
        and not milestone_depleting
        and frontier_last <= frontier_first
    ):
        print(
            "  WARNING: milestone reward isn't decaying relative to frontier — "
            "possible re-fire farming equilibrium (check merge_milestones wiring)."
        )
    elif frontier_last > milestone_last:
        print(
            "  OK: frontier has overtaken milestone reward — exploration is the "
            "dominant driver, not a farmed milestone annuity."
        )
    elif milestone_depleting:
        # Both channels can legitimately fall together: frontier reward is
        # NOT expected to grow monotonically — it pays only for genuinely
        # fresh cells (or, with archive_decay enabled, cells whose count
        # has decayed back down), so it's naturally spiky/episodic rather
        # than a smooth trend. The actual farming signature is milestone
        # staying flat/high, not frontier failing to strictly overtake it.
        print(
            "  OK: milestone reward is depleting as designed (re-fires cost "
            "less each time) — not the farming pattern, even though "
            "frontier hasn't strictly overtaken it this window."
        )


def report_reward_mix(ed):
    print("\n=== Reward mix (first quarter vs last quarter of episodes) ===")
    sources = ed.get("episode_reward_sources", [])
    seeded = ed.get("episode_seeded", [])
    if seeded and len(seeded) == len(sources):
        honest_sources = [ep for ep, is_seeded in zip(sources, seeded) if not is_seeded]
        seeded_sources = [ep for ep, is_seeded in zip(sources, seeded) if is_seeded]
        _report_reward_mix_group(honest_sources, "honest true-start episodes")
        _report_reward_mix_group(seeded_sources, "Go-Explore seeded episodes")
    else:
        _report_reward_mix_group(sources, "all episodes (seed split unavailable)")


def report_archive_growth(ed):
    print("\n=== Archive growth ===")
    arch = ed.get("episode_archive_size", [])
    if not arch:
        print("  no episode_archive_size data")
        return
    quarters = _quartile_slices(arch, 4)
    last_q = quarters[-1]
    growth_last_q = last_q[-1] - last_q[0] if len(last_q) > 1 else 0
    print(f"  archive size: {arch[0]} -> {arch[-1]} over {len(arch)} episodes")
    print(f"  growth in final quarter: {growth_last_q} cells")
    if growth_last_q <= 0:
        print(
            "  WARNING: archive has been flat for the last quarter of the run — "
            "exploration plateau (compare honest coverage, probes and the persisted "
            "frontier pool)."
        )


def report_discovery_log(ed):
    print("\n=== Discovery log ===")
    log = ed.get("discovery_log", [])
    n_episodes = len(ed.get("episode_rewards", []))
    if not log:
        print("  no discoveries logged")
        return
    honest_log = [entry for entry in log if not bool(entry.get("seeded", False))]
    seeded_log = [entry for entry in log if bool(entry.get("seeded", False))]
    print(
        f"  {len(log)} all-episode archive firsts logged "
        f"({len(honest_log)} honest, {len(seeded_log)} seeded)"
    )
    latest = honest_log[-1] if honest_log else None
    if latest is not None:
        last_ep = int(latest["episode"])
        print(
            f"  most recent honest: {latest} " f"({n_episodes - last_ep} episodes ago)"
        )
    else:
        last_ep = 0
        print("  no honest discoveries logged")
    by_type = {}
    for e in log:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    print(f"  by type: {by_type}")
    if not honest_log or n_episodes - last_ep > max(20, n_episodes // 4):
        print(
            "  WARNING: no new discovery in a while relative to run length — "
            "possible plateau (cross-check with archive growth above)."
        )


def report_time_to_frontier(ed, episode_length):
    print("\n=== Time-to-frontier (snapshot-seeding trigger) ===")
    fire_series = ed.get("episode_flag_fire_steps", [])
    deepest = [max(s for _f, s in ep) for ep in fire_series if ep]
    if not deepest:
        print("  no flag fires recorded yet")
        return
    median_step = statistics.median(deepest)
    max_step = max(deepest)
    p90_step = (
        statistics.quantiles(deepest, n=10)[-1] if len(deepest) >= 10 else max_step
    )
    print(f"  episodes with >=1 flag fire: {len(deepest)} / {len(fire_series)}")
    print(
        f"  median deepest fire step: {median_step:.0f}  "
        f"(typical episode — dominated by the shallowest recurring milestone, "
        f"NOT the run's frontier)"
    )
    print(f"  P90 deepest fire step: {p90_step:.0f}")
    print(
        f"  MAX deepest fire step ever reached: {max_step:.0f} "
        f"(the run's actual frontier — the number that matters for this check)"
    )
    if episode_length:
        frac = max_step / episode_length
        print(f"  max, as a fraction of episode_length ({episode_length}): {frac:.0%}")
        if frac > 0.6:
            print(
                "  TRIGGER: the deepest milestone this run has ever reached lands "
                "past ~60% of the episode budget — episode length itself may be "
                "the binding constraint. Compare short and long true-start probes "
                "before extending training episodes; also verify Go-Explore seeding "
                "is active."
            )
    else:
        print("  (pass --episode-length to compare against the training budget)")


def report_probes(ed, probe_episode_length, long_probe_episode_length):
    print("\n=== Probe ladder ===")
    rates = ed.get("probe_goal_rates", []) or ed.get("probe_goal_type_rates", [])
    if rates:
        print(
            f"  regular probe (episode_length={probe_episode_length or '?'}), "
            f"last 3 events:"
        )
        idxs = ed.get("probe_rollout_idx", [])
        for i, tr in list(zip(idxs, rates))[-3:]:
            print(
                f"    rollout {i}: {{{', '.join(f'{k}={v:.0%}' for k, v in tr.items())}}}"
            )
    else:
        print("  no regular probe data")

    long_rates = ed.get("long_probe_goal_rates", []) or ed.get(
        "long_probe_goal_type_rates", []
    )
    if long_rates:
        long_idxs = ed.get("long_probe_rollout_idx", [])
        print(
            f"  long-horizon probe (episode_length="
            f"{long_probe_episode_length or '?'}), last 3 events:"
        )
        for i, tr in list(zip(long_idxs, long_rates))[-3:]:
            print(
                f"    rollout {i}: {{{', '.join(f'{k}={v:.0%}' for k, v in tr.items())}}}"
            )
        if rates:
            latest_short, latest_long = rates[-1], long_rates[-1]
            for k in sorted(set(latest_short) | set(latest_long)):
                short_rate = latest_short.get(k, 0.0)
                long_rate = latest_long.get(k, 0.0)
                if long_rate > short_rate + 0.15:
                    print(
                        f"  NOTE: rung '{k}' is higher on the long-horizon probe "
                        f"({long_rate:.0%} vs {short_rate:.0%}) — the policy already has this "
                        f"capability, the short probe's horizon is hiding it."
                    )
    else:
        print(
            "  no long-horizon probe data (long_probe_enabled=False, or none "
            "have fired yet)"
        )


def report_checkpoints(ed, checkpoint_data=None):
    """Report true-start versus seeded checkpoint reach and replay status."""
    print("\n=== Story checkpoints ===")
    fire_series = ed.get("episode_flag_fire_steps", [])
    seeded = ed.get("episode_seeded", [])
    if not fire_series:
        print("  no checkpoint fire data")
        return

    rows = {}
    for episode_idx, episode_fires in enumerate(fire_series):
        is_seeded = bool(seeded[episode_idx]) if episode_idx < len(seeded) else False
        for flag_num, step in episode_fires:
            flag_num = int(flag_num)
            if not is_recordable_checkpoint(flag_num):
                continue
            row = rows.setdefault(flag_num, {"honest": [], "seeded": []})
            row["seeded" if is_seeded else "honest"].append(
                (episode_idx + 1, int(step))
            )

    if not rows:
        print("  no durable checkpoints reached")
        return
    print(f"  {'checkpoint':<42}{'honest':>8}{'seeded':>9}{'last honest':>14}")
    for flag_num, row in sorted(
        rows.items(),
        key=lambda item: max(
            [ep for ep, _step in item[1]["honest"] + item[1]["seeded"]]
        ),
    ):
        latest = row["honest"][-1][0] if row["honest"] else None
        print(
            f"  {checkpoint_title(flag_num):<42}"
            f"{len(row['honest']):>8}{len(row['seeded']):>9}"
            f"{(latest if latest is not None else 'never'):>14}"
        )

    recordings = (checkpoint_data or {}).get("checkpoint_recordings", {})
    if recordings:
        print("  first-honest replay recordings:")
        for raw_flag, entry in recordings.items():
            print(
                f"    {checkpoint_title(int(raw_flag))}: "
                f"{entry.get('status', 'unknown')} -> "
                f"{entry.get('output_folder', '?')}"
            )
    pool = (checkpoint_data or {}).get("frontier_pool", [])
    if checkpoint_data is not None:
        print(f"  persisted Go-Explore frontier pool: {len(pool)} snapshots")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("info_path", help="Path to a checkpoint's info.pth")
    p.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Training episode_length, for the time-to-frontier check "
        "(inferred from episode_lengths if omitted)",
    )
    p.add_argument("--probe-episode-length", type=int, default=None)
    p.add_argument("--long-probe-episode-length", type=int, default=None)
    p.add_argument("--band-low", type=float, default=0.6)
    p.add_argument("--band-high", type=float, default=1.2)
    p.add_argument("--coef-floor", type=float, default=1e-3)
    p.add_argument("--coef-ceiling", type=float, default=0.1)
    args = p.parse_args()

    import torch  # deferred: keep --help fast and dependency-light

    d = torch.load(args.info_path, map_location="cpu", weights_only=False)
    ed = d["episode_data"]
    episode_length = args.episode_length
    if episode_length is None and ed.get("episode_lengths"):
        episode_length = max(ed["episode_lengths"])

    n_episodes = len(ed.get("episode_rewards", []))
    print(f"Episodes: {n_episodes}   Best reward: {d.get('best_reward', 0):.1f}")
    va = d.get("visit_archive", {})
    print(
        f"Archive: {len(va.get('cells', {}))} cells, " f"{len(va.get('maps', {}))} maps"
    )

    report_entropy(
        ed, args.band_low, args.band_high, args.coef_floor, args.coef_ceiling
    )
    report_reward_mix(ed)
    report_archive_growth(ed)
    report_discovery_log(ed)
    report_time_to_frontier(ed, episode_length)
    report_checkpoints(ed, d)
    report_probes(ed, args.probe_episode_length, args.long_probe_episode_length)


if __name__ == "__main__":
    sys.exit(main())
