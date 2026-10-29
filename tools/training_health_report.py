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
3. Trainer battles & catching — honest-only win rate and catch counts
   (see ``Rewards.get_episode_event_counts``), plus the honest
   ``pokedex_owned`` ceiling. Flags an all-duplicate catching stretch (zero
   new-species catches while duplicates accrue) as a sign the catching
   reward may be reinforcing repetition rather than pokédex growth.
4. Archive growth — flags a flat ``episode_archive_size`` tail (no new
   cells recorded for the last portion of the run) as a plateau.
5. Discovery log — how many genuine run-wide firsts, and how long ago
   (in episodes) the most recent one landed.
6. Time-to-frontier — median deepest milestone-fire step vs episode length
   (episode length may be binding once this crosses roughly 60%).
7. Story checkpoints — honest/seeded reach, recency, first-honest replay status
   and persisted Go-Explore frontier-pool size.
8. Probe ladder — latest per-checkpoint rates, plus (if a long-horizon
   probe ran) a side-by-side comparison. A rung that's flat on the short
   probe but climbs on the long-horizon probe means the capability is
   already there and probe/episode length is the bottleneck, not
   competence — the exact mismatch a probe with too short a horizon
   would otherwise hide (see VecPPOAgent._run_long_horizon_probe).
9. Model capacity — parameter distribution across the architecture, the visual
   receptive field where it is derivable, parameters per fresh transition,
   critic explained variance, gradient allocation, whether the trust region
   rather than the network bounds progress, the termination mix and the honest
   coverage trend. Quantities the artefacts do not carry are named as
   unavailable rather than estimated.

Usage (from repo root):

    python tools/training_health_report.py "Training Outputs/open_world/Checkpoints/info.pth"
    python tools/training_health_report.py path/to/info.pth --episode-length 2048 --probe-episode-length 512
    python tools/training_health_report.py path/to/info.pth --conv-strides 2,2 --scaling-factor 0.5
"""
import argparse
import math
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


def _is_num(v):
    """True for a real, finite number.

    NaN in the rollout series means "degenerate minibatch" or "this rollout's
    update was skipped". It is excluded from means and counts, never read as 0.
    """
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _finite(xs):
    """The finite entries of a series, in order, NaN/inf dropped."""
    return [float(v) for v in (xs or []) if _is_num(v)]


def report_entropy(ed, band_low, band_high, coef_floor, coef_ceiling):
    print("\n=== Optimisation ===")
    ent = ed.get("rollout_policy_entropy", [])
    behaviour_ent = ed.get("rollout_behaviour_entropy", [])
    coef = ed.get("rollout_entropy_coef", [])
    boost = ed.get("rollout_entropy_boost", [])
    approx_kl = ed.get("rollout_approx_kl", [])
    base_post_kl = ed.get("rollout_base_post_kl", [])
    behaviour_post_kl = ed.get("rollout_behaviour_post_kl", [])
    base_replay_kl = ed.get("rollout_base_replay_kl", [])
    behaviour_replay_kl = ed.get("rollout_behaviour_replay_kl", [])
    replay_rejected = ed.get("rollout_replay_rejected", [])
    kl_rejected = ed.get("rollout_kl_rejected", [])
    clip_fraction = ed.get("rollout_clip_fraction", [])
    learning_rates = ed.get("rollout_lr", [])
    if not ent:
        print("  no rollout_policy_entropy data")
    else:
        in_band = sum(1 for e in ent if band_low <= e <= band_high) / len(ent)
        at_floor = sum(1 for c in coef if c <= coef_floor * 1.05) / max(1, len(coef))
        at_ceiling = sum(1 for c in coef if c >= coef_ceiling * 0.95) / max(1, len(coef))
        boost_frac = sum(1 for b in boost if b) / max(1, len(boost))
        print(
            f"  mean base-policy entropy: {_mean(ent):.3f} nats "
            f"(band [{band_low}, {band_high}])"
        )
        if behaviour_ent:
            print(f"  mean assisted behaviour entropy: {_mean(behaviour_ent):.3f} nats")
        print(f"  fraction of rollouts inside band: {in_band:.0%}")
        print(f"  fraction of rollouts with coef pinned at floor:   {at_floor:.0%}")
        print(f"  fraction of rollouts with coef pinned at ceiling: {at_ceiling:.0%}")
        print(f"  fraction of rollouts under a plateau boost: {boost_frac:.0%}")
        if in_band < 0.4:
            # Deliberately not gated on the coefficient: a flat coefficient with
            # out-of-band entropy is exactly the unattended case that needs the
            # warning. The context line only says whether a controller was in the
            # loop at all, so an out-of-band run is not mistaken for a tuned one.
            print("  WARNING: entropy spends most of its time outside the target band.")
            coef_finite = _finite(coef)
            if not coef_finite:
                print(
                    "  context: no finite rollout_entropy_coef values to read a "
                    "controller from."
                )
            elif max(coef_finite) - min(coef_finite) <= 1e-12:
                print(
                    f"  context: coefficient fixed at {coef_finite[0]:.3g}, "
                    "no controller in the loop."
                )
            else:
                print(
                    f"  context: coefficient moved {coef_finite[0]:.3g} -> "
                    f"{coef_finite[-1]:.3g}."
                )
        if at_floor > 0.3:
            print("  WARNING: coefficient frequently pinned at the floor.")
    if approx_kl:
        print(f"  approximate KL: mean={_mean(approx_kl):.3e}, " f"max={max(approx_kl):.3e}")
    if base_post_kl:
        print(
            f"  exact base post-update KL: mean={_mean(base_post_kl):.3e}, "
            f"max={max(base_post_kl):.3e}"
        )
    if behaviour_post_kl:
        print(
            "  exact assisted post-update KL: "
            f"mean={_mean(behaviour_post_kl):.3e}, "
            f"max={max(behaviour_post_kl):.3e}"
        )
    if base_replay_kl:
        print(
            f"  base collection/replay KL: mean={_mean(base_replay_kl):.3e}, "
            f"max={max(base_replay_kl):.3e}"
        )
    if behaviour_replay_kl:
        print(
            "  assisted collection/replay KL: "
            f"mean={_mean(behaviour_replay_kl):.3e}, "
            f"max={max(behaviour_replay_kl):.3e}"
        )
    if replay_rejected:
        rejected = sum(bool(v) for v in replay_rejected)
        recent = replay_rejected[-100:]
        recent_rejected = sum(bool(v) for v in recent)
        quarters = _quartile_slices(replay_rejected, 4)
        quarter_rates = [sum(bool(v) for v in q) / len(q) for q in quarters if q]
        print(
            f"  replay-rejected updates: {rejected}/{len(replay_rejected)} "
            f"({rejected / len(replay_rejected):.1%}); final 100: "
            f"{recent_rejected}/{len(recent)} ({recent_rejected / len(recent):.1%})"
        )
        print(
            "  replay rejection by quarter: " + ", ".join(f"{rate:.1%}" for rate in quarter_rates)
        )
        if (
            len(quarter_rates) > 1
            and quarter_rates[-1] >= 0.05
            and quarter_rates[-1] > quarter_rates[0]
        ):
            print("  WARNING: replay rejection rate increased over the run.")
    if any(kl_rejected):
        print(
            f"  CRITICAL: {sum(bool(v) for v in kl_rejected)} PPO epoch(s) "
            "breached the hard KL limit and were rolled back."
        )
    if clip_fraction:
        print(f"  clip fraction: mean={_mean(clip_fraction):.3f}, " f"max={max(clip_fraction):.3f}")
    if learning_rates:
        print(
            f"  learning rate: first={learning_rates[0]:.3g}, " f"latest={learning_rates[-1]:.3g}"
        )


def _load_state_dict(path):
    """Return ``(state_dict, None)`` on success, ``(None, reason)`` on failure.

    Deliberately exception-safe: an unreadable model checkpoint degrades the
    Capacity section to a single "unavailable" line rather than aborting a
    report that is otherwise reading good data.
    """
    import torch  # deferred: keep --help fast and dependency-light

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}"
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            inner = obj.get(key)
            if isinstance(inner, dict):
                obj = inner
                break
    if not isinstance(obj, dict) or not any(hasattr(v, "numel") for v in obj.values()):
        return None, "holds no recognisable state_dict"
    return obj, None


def _param_groups(sd):
    """Bucket a state_dict into architectural groups by key prefix.

    Keyed on the prefixes this project's ActorCritic uses. An unrecognised
    top-level module still gets its own row rather than being dropped, so the
    table always sums to the printed total.
    """
    groups = {}
    for key, tensor in sd.items():
        numel = int(tensor.numel()) if hasattr(tensor, "numel") else 0
        if numel <= 0:
            continue
        parts = key.split(".")
        if parts[0] == "transformer_blocks" and len(parts) > 1:
            name = f"transformer_blocks[{parts[1]}]"
        elif key.startswith("cnn.fc"):
            name = "cnn.fc"
        elif key.startswith("cnn."):
            name = "cnn convs"
        else:
            name = parts[0]
        groups[name] = groups.get(name, 0) + numel
    return groups


def _parse_conv_strides(raw):
    """``'2,2'`` -> ``[2, 2]``; ``None``/empty -> ``None``. Never guessed."""
    if raw is None:
        return None
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        raise SystemExit(f"--conv-strides wants comma-separated ints, got {raw!r}")


def report_capacity(ed, d, args, info_path):
    """Is the network under-sized, over-sized or mis-shaped for this run?

    Reads only what the run persisted, plus — optionally — the sibling
    ``actor_critic.pth`` state_dict for parameter counts. Quantities the
    artefacts do not carry are reported as unavailable rather than estimated:
    a state_dict records conv KERNEL sizes but no STRIDES and no
    scaling_factor — runs saved since ``run_config`` gained ``conv_strides``
    carry both there and derive the field with no flags — and the reward
    scaler's variance is not the variance of the GAE returns that reach the
    value loss, so no explained-variance proxy can be reconstructed from
    critic_loss.

    Does not repeat the Optimisation section's statistics. The only KL use
    here is the question that section leaves open: whether the trust region,
    rather than the network, is what bounds progress.
    """
    print("\n=== Capacity ===")
    run_config = (d or {}).get("run_config") or {}
    missing = []

    def rc(key, cli_default):
        """The run's own config wins over the CLI default whenever it has it.

        A key the run persisted as None collapses to cli_default here, which is
        right for a threshold the run simply never set. The two KL keys where a
        persisted None IS the effective setting go through rc_nullable instead.
        """
        value = run_config.get(key)
        return cli_default if value is None else value

    def rc_nullable(key, cli_default):
        """(value, source) for a key where a persisted None is itself a setting.

        source is 'run_config' whenever the run persisted the key at all —
        including when it persisted None — and 'CLI' only when the key is
        absent, i.e. a checkpoint written before run_config existed. That is the
        distinction rc() deliberately does not make: vec_ppo_agent.save_model
        keeps ``ppo_target_kl: None`` on purpose, because None there means the
        run configured NO KL early-stop, so substituting --target-kl's 0.02
        default would judge the run against a ceiling it never had.
        """
        if key not in run_config:
            return cli_default, "CLI"
        return run_config[key], "run_config"

    # ---- 1. architecture -------------------------------------------------
    print("\n  -- Architecture --")
    model_path = args.model_path
    if not model_path:
        sibling = Path(info_path).resolve().parent / "actor_critic.pth"
        model_path = str(sibling)
    sd, load_error = _load_state_dict(model_path)
    total_params = None
    if sd is None:
        print(f"  parameter counts: unavailable ({model_path}: {load_error})")
        missing.append("parameter counts: no loadable model state_dict next to info.pth")
    else:
        tensors = [(k, v) for k, v in sd.items() if hasattr(v, "numel")]
        total_params = sum(int(v.numel()) for _, v in tensors)
        groups = _param_groups(sd)
        n_blocks = sum(1 for g in groups if g.startswith("transformer_blocks["))
        rows = sorted(groups.items(), key=lambda kv: -kv[1])
        if n_blocks:
            combined = sum(c for g, c in groups.items() if g.startswith("transformer_blocks["))
            rows.append((f"transformer_blocks (all {n_blocks})", combined))
            rows.sort(key=lambda kv: -kv[1])
        print(f"  source: {model_path}")
        print(f"  total parameters: {total_params:,}")
        print(f"  {'group':<28}{'params':>10}{'share':>8}")
        for label, count in rows:
            print(f"  {label:<28}{count:>10,}{count / total_params:>8.1%}")
        biggest = max(tensors, key=lambda kv: int(kv[1].numel()))
        biggest_n = int(biggest[1].numel())
        shape = "x".join(str(int(x)) for x in biggest[1].shape)
        print(f"  largest tensor: {biggest[0]} ({shape}) = {biggest_n:,} params "
              f"({biggest_n / total_params:.1%})")
        dense = [(k, v) for k, v in tensors if hasattr(v, "dim") and v.dim() == 2]
        if dense:
            name, tensor = max(dense, key=lambda kv: int(kv[1].numel()))
            share = int(tensor.numel()) / total_params
            if share >= 0.5:
                print(f"  WARNING: one dense layer, {name}, holds {share:.1%} of all "
                      f"parameters — capacity is concentrated in a single matrix, so "
                      f"'is the network big enough?' is really 'is that matrix the "
                      f"right shape?'. That is a shape finding, and it holds however "
                      f"healthy the run above looks.")

    # ---- 2. receptive field ----------------------------------------------
    print("\n  -- Receptive field --")
    conv_layers = [
        (k, int(v.shape[-2]), int(v.shape[-1]))
        for k, v in (sd or {}).items()
        if hasattr(v, "dim") and v.dim() == 4
    ]
    scaling = rc("scaling_factor", args.scaling_factor)
    # Stride schedule: the run's own walk off the live CNN wins over --conv-strides,
    # the same preference every other threshold in this section gives run_config. A
    # missing key, an explicit None (what a Conv2d-free net persists) or an empty
    # list all mean "this run recorded no strides", so fall through to the CLI
    # rather than trusting a half-written value.
    cfg_strides = run_config.get("conv_strides")
    strides, strides_src = args.conv_strides, "CLI"
    if isinstance(cfg_strides, (list, tuple)) and cfg_strides and all(
        isinstance(s, int) and not isinstance(s, bool) for s in cfg_strides
    ):
        strides, strides_src = [int(s) for s in cfg_strides], "run_config"
    if not conv_layers:
        reason = (
            "the model state_dict did not load"
            if sd is None
            else "no 4-D conv weights in the state_dict"
        )
        print(f"  receptive field: unavailable ({reason})")
        missing.append(f"receptive field: conv kernel sizes unreadable — {reason}")
    elif scaling is None or strides is None:
        kernels = ", ".join(f"{k}={kh}x{kw}" for k, kh, kw in conv_layers)
        print("  receptive field: not derivable from a state_dict — it records conv "
              "kernel sizes but no strides, and scaling_factor is not in it either")
        print(f"    kernels in definition order: {kernels}")
        need = []
        if strides is None:
            need.append(f"--conv-strides with {len(conv_layers)} comma-separated ints")
        if scaling is None:
            need.append("--scaling-factor")
        print("    would become derivable with: a run_config carrying both conv_strides "
              "and scaling_factor, which newer runs persist so no flags are needed; "
              f"or by hand with {' and '.join(need)}")
        missing.append("receptive field: conv strides and/or scaling_factor not persisted")
    elif len(strides) != len(conv_layers):
        print(f"  receptive field: {strides_src} supplied {len(strides)} strides for "
              f"{len(conv_layers)} conv layers — not guessing which layer takes which")
    else:
        # RF = 1 + sum over layers of (kernel - 1) * cumulative stride before it.
        # One stride per layer, applied to both axes (the flag cannot express a
        # non-square stride, so a non-square kernel is the only way H and W differ).
        rf_h = rf_w = 1
        cum = 1
        for (_key, kh, kw), stride in zip(conv_layers, strides):
            rf_h += (kh - 1) * cum
            rf_w += (kw - 1) * cum
            cum *= stride
        orig_h, orig_w = rf_h / scaling, rf_w / scaling
        print(f"  strides used, both axes (from {strides_src}): {strides}")
        print(f"  receptive field: {rf_h}x{rf_w} px of the scaled CNN input")
        print(f"  receptive field: {orig_h:.1f}x{orig_w:.1f} px of the original screen")
        print(f"  receptive field: {orig_h / 8:.2f}x{orig_w / 8:.2f} Game Boy tiles "
              f"(an 8x8 original-px tile)")
        if min(orig_h, orig_w) / 8 < 2:
            print("  WARNING: the visual stack sees under 2 tiles — one layer of "
                  "context around a single sprite, so no wall, doorway or NPC can be "
                  "perceived in relation to the agent's own tile.")

    # ---- 3. data rate ----------------------------------------------------
    print("\n  -- Data rate --")
    update_freq = rc("ppo_update_frequency", args.update_frequency)
    freq_src = "run_config" if run_config.get("ppo_update_frequency") is not None else "CLI"
    num_envs = rc("num_envs", None)
    envs_src = "run_config"
    if num_envs is None:
        returns = ((d or {}).get("reward_scaler") or {}).get("running_returns")
        if returns is not None and len(returns):
            num_envs, envs_src = len(returns), "len(reward_scaler.running_returns)"
    if num_envs is None or update_freq is None:
        print("  transitions per rollout: unavailable (num_envs and/or "
              "ppo_update_frequency unknown — not guessing either)")
        missing.append("data rate: num_envs / ppo_update_frequency not persisted")
    else:
        per_rollout = int(update_freq) * int(num_envs)
        print(f"  fresh transitions per rollout: {per_rollout:,} = "
              f"ppo_update_frequency {update_freq} ({freq_src}) x num_envs {num_envs} "
              f"({envs_src})")
        if total_params:
            print(f"  total parameters: {total_params:,}")
            print(f"  parameters per fresh transition: {total_params / per_rollout:,.1f}")
            print("    that ratio is the over-parameterisation pressure the dense "
                  "visual layer sees on every update")
        else:
            print("  parameters per fresh transition: unavailable (parameter count unknown)")

    # ---- 4. value fit ----------------------------------------------------
    print("\n  -- Value fit --")
    ev_series = ed.get("rollout_explained_variance") or []
    ev = _finite(ev_series)
    if not ev_series:
        print("  explained variance: unavailable for this run (not logged before this "
              "change)")
        print("    not reconstructed from critic_loss: the reward scaler's variance is "
              "not the variance of the GAE returns that reach the value loss, so any "
              "such ratio here would be fabricated")
        missing.append("value fit: rollout_explained_variance not logged by this run")
    elif not ev:
        print(f"  explained variance: all {len(ev_series)} logged rollouts are NaN "
              f"(degenerate minibatches or skipped updates)")
        missing.append("value fit: every rollout_explained_variance entry is NaN")
    else:
        mean_ev = _mean(ev)
        print(f"  explained variance: mean {mean_ev:.3f} over {len(ev)}/{len(ev_series)} "
              f"finite rollouts")
        rmse = ed.get("rollout_value_error_rmse") or []
        rstd = ed.get("rollout_return_std") or []
        mask = [i for i, v in enumerate(ev_series) if _is_num(v)]
        if len(rmse) == len(ev_series) and len(rstd) == len(ev_series):
            pairs = [
                (float(rmse[i]) ** 2, float(rstd[i]) ** 2)
                for i in mask
                if _is_num(rmse[i]) and _is_num(rstd[i])
            ]
            denom = _mean([v for _e, v in pairs])
            if pairs and denom > 0:
                cross = 1.0 - _mean([e for e, _v in pairs]) / denom
                print(f"  cross-check 1 - mean(value_error^2)/mean(return_std^2) over the "
                      f"same {len(pairs)} rollouts: {cross:.3f}")
            else:
                print("  cross-check: unavailable (mean squared return_std is 0)")
        else:
            print("  cross-check: unavailable (rollout_value_error_rmse / "
                  "rollout_return_std missing or not aligned with explained variance)")
        if mean_ev >= 0.9:
            print("  OK: critic fit is not capacity-limited")
        elif mean_ev >= 0.5:
            print("  critic only partially fits returns; value capacity or return scale "
                  "may be limiting")
        else:
            print("  WARNING: critic underfits returns")
    critic_loss = _finite(ed.get("rollout_critic_loss"))
    if critic_loss:
        print(f"  raw critic loss, mean {(_mean(critic_loss)):.4f} — quoted with no "
              f"interpretation: it mixes epochs whose reward scaling differed")

    # ---- 5. gradient allocation ------------------------------------------
    print("\n  -- Gradient allocation --")
    pv_series = ed.get("rollout_pv_grad_norm") or []
    pv = _finite(pv_series)
    if not pv_series or not pv:
        print("  gradient allocation: unavailable for this run")
        missing.append("gradient allocation: rollout_pv_grad_norm not logged by this run")
    else:
        print(f"  policy-value grad norm: mean {_mean(pv):.3e}, max {max(pv):.3e} "
              f"({len(pv)}/{len(pv_series)} finite rollouts)")
        ent_series = ed.get("rollout_ent_grad_norm") or []
        ent = _finite(ent_series)
        if ent:
            print(f"  entropy grad norm:      mean {_mean(ent):.3e}, max {max(ent):.3e} "
                  f"({len(ent)}/{len(ent_series)} finite rollouts)")
        else:
            print("  entropy grad norm:      unavailable for this run")
        max_grad = rc("ppo_max_grad_norm", args.max_grad_norm)
        if max_grad is None:
            print("  fraction above ppo_max_grad_norm: threshold unknown (no run_config "
                  "and no --max-grad-norm) — not guessing it")
        else:
            over = sum(1 for v in pv if v > float(max_grad))
            print(f"  rollouts with grad norm above ppo_max_grad_norm ({max_grad:g}): "
                  f"{over}/{len(pv)} ({over / len(pv):.1%})")
        group_keys = (
            ("cnn", "rollout_grad_norm_cnn"),
            ("ram", "rollout_grad_norm_ram"),
            ("trunk", "rollout_grad_norm_trunk"),
            ("heads", "rollout_grad_norm_heads"),
        )
        means = {}
        for label, key in group_keys:
            vals = _finite(ed.get(key))
            if vals:
                means[label] = _mean(vals)
        total = sum(means.values())
        if means and total > 0:
            print("  mean share of the summed group norms — these are PRE-clip, so the "
                  "shares are meaningful and the absolutes are not comparable to "
                  "ppo_max_grad_norm once clipping engages:")
            for label, _key in group_keys:
                if label in means:
                    print(f"    {label:<6}{means[label]:>12.3e}{means[label] / total:>9.1%}")
        else:
            print("  per-group grad norms: unavailable for this run")
            missing.append("gradient allocation: per-group rollout_grad_norm_* not logged")

    # ---- 6. optimisation pressure ----------------------------------------
    print("\n  -- Optimisation pressure --")
    approx_kl = _finite(ed.get("rollout_approx_kl"))
    # rc_nullable, not rc: for these two keys a persisted None is the SETTING
    # (no KL early-stop / no hard ceiling); only an absent key — an older
    # checkpoint — may fall back to the CLI default.
    target_kl, target_src = rc_nullable("ppo_target_kl", args.target_kl)
    hard_kl, hard_src = rc_nullable("ppo_hard_kl", args.hard_kl)
    target_txt = "none configured" if target_kl is None else f"{target_kl:g}"
    hard_txt = "none configured" if hard_kl is None else f"{hard_kl:g}"
    print(f"  KL ceilings: ppo_target_kl {target_txt} ({target_src}), "
          f"ppo_hard_kl {hard_txt} ({hard_src})")
    if not approx_kl:
        print("  KL pressure: unavailable (no finite rollout_approx_kl)")
        missing.append("optimisation pressure: rollout_approx_kl not logged")
    else:
        # Same refusal as the ppo_max_grad_norm branch above: name the gap
        # rather than measure against a threshold the run never configured.
        over_target = None
        if target_kl is None:
            print("  fraction above ppo_target_kl: undefined — this run configured "
                  "no ppo_target_kl, so there is no KL early-stop ceiling (not "
                  "falling back to --target-kl)")
            missing.append(
                "optimisation pressure: run persisted ppo_target_kl None, i.e. no "
                "KL early-stop ceiling, so the over-target fraction is undefined"
            )
        else:
            over_target = sum(1 for v in approx_kl if v > float(target_kl))
            print(f"  rollouts above ppo_target_kl ({target_kl:g}): {over_target}/"
                  f"{len(approx_kl)} ({over_target / len(approx_kl):.1%})")
        if hard_kl is None:
            print("  fraction above ppo_hard_kl: undefined — this run configured no "
                  "ppo_hard_kl, so there is no hard KL ceiling to count against "
                  "(not falling back to --hard-kl)")
            missing.append(
                "optimisation pressure: run persisted ppo_hard_kl None, i.e. no hard "
                "KL ceiling, so the over-ceiling fraction is undefined"
            )
        else:
            over_hard = sum(1 for v in approx_kl if v > float(hard_kl))
            print(f"  rollouts above ppo_hard_kl ({hard_kl:g}): {over_hard}/"
                  f"{len(approx_kl)} ({over_hard / len(approx_kl):.1%})")
        lrs = _finite(ed.get("rollout_lr"))
        lr_fixed = None
        if len(lrs) >= 2:
            lr_fixed = lrs[0] == lrs[-1]
            print(f"  learning rate, first vs last finite value: "
                  f"{'did not change' if lr_fixed else 'changed'}")
        else:
            print("  learning rate, first vs last finite value: unavailable")
        # All three verdicts judge the over-target fraction against
        # ppo_target_kl, so with None persisted they would describe --target-kl's
        # default rather than this run: skip them instead of inventing a ceiling.
        frac = None if over_target is None else over_target / len(approx_kl)
        if frac is None:
            print("  trust-region verdict: skipped — this run configured no "
                  "ppo_target_kl, so there is no ceiling to judge against")
        elif frac >= 0.4 and lr_fixed:
            print(f"  OK: trust-region-limited — the policy presses the KL ceiling on "
                  f"{frac:.0%} of updates at a constant LR, so update size, not network "
                  f"capacity, bounds progress")
        elif frac < 0.1:
            print("  WARNING: updates are far inside the trust region; the model may "
                  "lack the capacity or the signal to move")
        else:
            print("  updates sit near the trust-region target")

    # ---- 7. termination mix ----------------------------------------------
    print("\n  -- Termination mix --")
    causes = [str(c) for c in (ed.get("episode_truncation_cause") or [])]
    if not causes:
        print("  truncation causes: unavailable for this run")
        missing.append("termination mix: episode_truncation_cause not logged")
    else:
        counts = {}
        for cause in causes:
            counts[cause] = counts.get(cause, 0) + 1
        for cause, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {cause:<22}{count:>6}{count / len(causes):>9.1%}")
        stagnant = sorted(c for c in counts if "stagnation" in c)
        stag = sum(counts[c] for c in stagnant)
        if stagnant and stag / len(causes) >= 0.6:
            print(f"  WARNING: {stag / len(causes):.0%} of episodes ended on stagnation "
                  f"({', '.join(stagnant)}) rather than on the step budget — this run is "
                  f"idea-limited, not step-limited, so extra network capacity buys "
                  f"nothing until the reward or curriculum supplies new ideas")

    # ---- 8. honest coverage trend ----------------------------------------
    print("\n  -- Honest coverage trend --")
    seeded = ed.get("episode_seeded") or []

    def honest(series):
        return [v for i, v in enumerate(series) if i >= len(seeded) or not seeded[i]]

    h_maps = honest(ed.get("episode_unique_maps") or [])
    h_cells = honest(ed.get("episode_unique_cells") or [])
    if not h_maps and not h_cells:
        print("  honest coverage: unavailable for this run")
        missing.append("honest coverage: episode_unique_maps / episode_unique_cells absent")
    else:
        print(f"  {max(len(h_maps), len(h_cells))} honest episodes, oldest first:")
        trends = {}
        for label, series in (("unique_maps", h_maps), ("unique_cells", h_cells)):
            if not series:
                print(f"    {label:<13} unavailable for this run")
                continue
            # Drop empty buckets rather than averaging them to 0.0: with fewer
            # honest episodes than buckets a padded 0.0 tail would invent a
            # collapse. Fewer than two real buckets means no trend is claimable.
            buckets = [round(_mean(b), 1) for b in _quartile_slices(series, 8) if b]
            print(f"    {label:<13} {len(buckets)} buckets: {buckets}")
            trends[label] = buckets[-1] > buckets[0] if len(buckets) >= 2 else None
        decided = {k: v for k, v in trends.items() if v is not None}
        if len(decided) < 2:
            print("  not enough honest episodes for a first-to-last trend in both series")
        else:
            rose = [k for k, v in decided.items() if v]
            fell = [k for k, v in decided.items() if not v]
            if not fell:
                print("  OK: honest coverage still rising first-to-last in both series; "
                      "no underfitting signature")
            elif not rose:
                print("  WARNING: honest coverage declining first-to-last in both series")
            else:
                print(f"  mixed: {', '.join(rose)} rose first-to-last while "
                      f"{', '.join(fell)} fell")

    # ---- 9. missing diagnostics ------------------------------------------
    print("\n  -- Missing diagnostics --")
    if not run_config:
        missing.insert(
            0,
            "run_config: absent — geometry, PPO thresholds and num_envs were not "
            "persisted with this checkpoint",
        )
    if not missing:
        print("  none — every capacity signal this section can read was present")
    else:
        print("  what this run's artefacts cannot answer:")
        for line in missing:
            print(f"    {line}")
    if not run_config:
        print("  every threshold above came from CLI defaults (--update-frequency, "
              "--target-kl, --hard-kl, --max-grad-norm, --scaling-factor), not from "
              "the run's own config")


def _report_reward_mix_group(sources, label):
    print(f"\n  -- {label} --")
    if not sources:
        print("  no episode_reward_sources data")
        return
    quarters = _quartile_slices(sources, 4)
    first, last = quarters[0], quarters[-1]
    accounting_keys = {
        "delivered_total",
        "rounding_adjustment",
        "clip_adjustment",
        "delivery_cast_adjustment",
    }
    keys = sorted({k for ep in sources for k in ep} - accounting_keys)
    print(f"  {'source':<14}{'first-¼ avg':>14}{'last-¼ avg':>14}")
    milestone_first = milestone_last = 0.0
    frontier_first = frontier_last = 0.0
    for k in keys:
        f_avg = _mean(ep.get(k, 0.0) for ep in first)
        l_avg = _mean(ep.get(k, 0.0) for ep in last)
        print(f"  {k:<14}{f_avg:>14.2f}{l_avg:>14.2f}")
        if k in (
            "flag",
            "checkpoint",
            "checkpoint_honest",
            "checkpoint_seeded",
            "pokedex",
            "level",
        ):
            milestone_first += f_avg
            milestone_last += l_avg
        if k in ("frontier",):
            frontier_first += f_avg
            frontier_last += l_avg
    print(f"  milestone total: {milestone_first:.2f} -> {milestone_last:.2f}")
    print(f"  frontier total:  {frontier_first:.2f} -> {frontier_last:.2f}")
    if any(key in ep for ep in sources for key in accounting_keys):
        print("  delivery accounting:")
        for key in (
            "delivered_total",
            "rounding_adjustment",
            "clip_adjustment",
            "delivery_cast_adjustment",
        ):
            f_avg = _mean(ep.get(key, 0.0) for ep in first)
            l_avg = _mean(ep.get(key, 0.0) for ep in last)
            print(f"    {key:<20}{f_avg:>10.4f} -> {l_avg:>10.4f}")
        raw_first = sum(_mean(ep.get(k, 0.0) for ep in first) for k in keys)
        raw_last = sum(_mean(ep.get(k, 0.0) for ep in last) for k in keys)
        delivered_first = _mean(ep.get("delivered_total", 0.0) for ep in first)
        delivered_last = _mean(ep.get("delivered_total", 0.0) for ep in last)
        expected_first = raw_first + _mean(
            ep.get("rounding_adjustment", 0.0)
            + ep.get("clip_adjustment", 0.0)
            + ep.get("delivery_cast_adjustment", 0.0)
            for ep in first
        )
        expected_last = raw_last + _mean(
            ep.get("rounding_adjustment", 0.0)
            + ep.get("clip_adjustment", 0.0)
            + ep.get("delivery_cast_adjustment", 0.0)
            for ep in last
        )
        error_first = delivered_first - expected_first
        error_last = delivered_last - expected_last
        print(f"    {'raw source total':<20}{raw_first:>10.4f} -> {raw_last:>10.4f}")
        print(f"    {'conservation error':<20}{error_first:>10.6f} -> {error_last:>10.6f}")
        if abs(error_first) > 1e-4 or abs(error_last) > 1e-4:
            print("    WARNING: delivered reward does not reconcile with source accounting.")
    milestone_depleting = milestone_last > 0 and milestone_last < milestone_first * 0.8
    if milestone_last > 0 and not milestone_depleting and frontier_last <= frontier_first:
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


def report_stepping(ed):
    """Settle stepping (PoliwhiRL/environment/settle.py): frames per step and
    steps that hit the frame cap. Capped steps should stay ~0; a rising count
    means some screen never reports itself ready for input."""
    print("\n=== Stepping (settle) ===")
    frames = ed.get("episode_step_frames", [])
    steps = ed.get("episode_lengths", [])
    if not frames:
        print("  no episode_step_frames data (run predates settle-stepping metrics)")
        return
    capped = ed.get("episode_capped_steps", [])
    auto = ed.get("episode_auto_frames", [])
    n = min(len(frames), len(steps)) or len(frames)
    per_step = [f / max(1, s) for f, s in zip(frames[-n:], steps[-n:])] if steps else []
    if per_step:
        print(f"  frames per step: mean {_mean(per_step):.1f} (last {len(per_step)} episodes)")
    total_capped = int(sum(capped))
    print(
        f"  capped steps: {total_capped} over {len(capped)} episodes; "
        f"auto-input frames: {int(sum(auto))}"
    )
    if total_capped > len(capped):
        print(
            "  WARNING: more than one capped step per episode — a screen is not "
            "being recognised as waiting for input (see settle.py)."
        )


def report_battle_and_catching(ed):
    """Trainer-battle-win and catching counters (see
    Rewards.get_episode_event_counts), honest episodes only. These arrays
    only start accumulating from the episode this instrumentation was
    deployed, so they're the TAIL of a longer run, not a prefix of it — same
    alignment convention as visuals.py's _render_event_counts."""
    print("\n=== Trainer battles & catching (honest only) ===")
    trainer_wins = ed.get("episode_trainer_wins", [])
    catches = ed.get("episode_catches", [])
    new_species = ed.get("episode_new_species_catches", [])
    duplicate = ed.get("episode_duplicate_catches", [])
    seeded = ed.get("episode_seeded", [])
    pokedex_owned = ed.get("episode_pokedex_owned", [])

    if pokedex_owned and seeded and len(pokedex_owned) == len(seeded):
        honest_owned = [v for v, s in zip(pokedex_owned, seeded) if not s]
        if honest_owned:
            print(f"  honest pokedex_owned: max ever reached = {max(honest_owned)}")

    if not trainer_wins:
        print("  no trainer-win/catch counters recorded (run predates this instrumentation)")
        return

    n = len(trainer_wins)
    seeded_tail = (seeded or [])[-n:]
    is_honest = [not s for s in seeded_tail] if len(seeded_tail) == n else [True] * n

    def _honest(series):
        return [v for v, h in zip(series, is_honest) if h]

    h_wins, h_catches, h_new, h_dup = (
        _honest(trainer_wins),
        _honest(catches),
        _honest(new_species),
        _honest(duplicate),
    )
    n_honest = len(h_wins)
    print(f"  {n} episodes recorded since deploy ({n_honest} honest)")
    if n_honest == 0:
        print("  no honest episodes recorded yet")
        return

    wins_with = sum(1 for w in h_wins if w > 0)
    print(
        f"  honest trainer-battle-win rate: {wins_with / n_honest:.1%} "
        f"({wins_with}/{n_honest} episodes with >=1 win)"
    )
    total_catches, total_new, total_dup = sum(h_catches), sum(h_new), sum(h_dup)
    print(
        f"  honest catches since deploy: total={total_catches}  "
        f"new-species={total_new}  duplicate={total_dup}"
    )
    if total_catches > 0 and total_new == 0:
        print(
            "  WARNING: every honest catch since deploy has been a duplicate species — "
            "zero new-species catches. The catching-domain reward push may be "
            "reinforcing re-catching known species rather than expanding pokedex "
            "coverage; before raising catching priority further, check whether this "
            "is an encounter-table reachability limit or a policy choice."
        )


def report_party_and_navigation(ed):
    """Whiteouts, non-whiteout heals, wall bumps and who gains the levels
    (Rewards._track_episode_diagnostics), honest episodes by quarter. Like
    the catching counters these start at their deploy, so they are the TAIL
    of the run. Read whiteouts against heals: healthy is heals rising while
    whiteouts fall, not whiteouts traded for avoiding battles (check trainer
    wins above)."""
    print("\n=== Party health & navigation (honest only) ===")
    whiteouts = ed.get("episode_whiteouts", [])
    if not whiteouts:
        print("  no party/navigation counters recorded (run predates this instrumentation)")
        return
    n = len(whiteouts)
    seeded_tail = (ed.get("episode_seeded") or [])[-n:]
    honest = [i for i in range(n) if len(seeded_tail) != n or not seeded_tail[i]]
    if not honest:
        print(f"  {n} episodes recorded since deploy, none honest yet")
        return
    print(f"  {n} episodes recorded since deploy ({len(honest)} honest)")

    def series(key):
        """Tail-aligned ``(values, covered)`` pair, or ``(None, 0)`` if absent.

        A series shorter than the deploy window used to return None and the row
        was dropped silently, so episode_wall_bumps covering 85 of 473 episodes
        printed nothing at all. It now comes back whole with its true coverage,
        tail-aligned to the same window, and the caller labels the row
        "[85 of 473 episodes]" instead of hiding it.
        """
        vals = ed.get(f"episode_{key}", [])
        if not vals:
            return None, 0
        if len(vals) >= n:
            return vals[-n:], n
        return vals, len(vals)

    def covered_idx(covered, quarter):
        """The quarter's indices that a tail-aligned series actually covers."""
        offset = n - covered
        return [i - offset for i in quarter if i - offset >= 0]

    def note(covered):
        """Coverage label, empty for a series that reaches the whole window."""
        return "" if covered >= n else f"   [{covered} of {n} episodes]"

    rows = (
        ("whiteouts", "whiteouts / episode"),
        ("heals", "non-whiteout heals / episode"),
        ("wall_bumps", "wall bumps / episode"),
        ("lead_level_gains", "lead level gains / episode"),
        ("nonlead_level_gains", "non-lead level gains / episode"),
    )
    quarters = [q for q in _split(honest, 4) if q]
    print(f"  {'':34s}" + "".join(f"  Q{k + 1:<6d}" for k in range(len(quarters))))
    for key, label in rows:
        vals, covered = series(key)
        if vals is None:
            continue
        cells = ""
        for q in quarters:
            idx = covered_idx(covered, q)
            cells += f"  {_mean([vals[i] for i in idx]):<7.2f}" if idx else f"  {'-':<7}"
        print(f"  {label:34s}{cells}{note(covered)}")
    (bumps, bumps_cov), (presses, presses_cov) = series("wall_bumps"), series("walk_presses")
    if bumps and presses:
        covered = min(bumps_cov, presses_cov)
        cells = ""
        for q in quarters:
            idx = covered_idx(covered, q)
            share = sum(bumps[i] for i in idx) / max(1, sum(presses[i] for i in idx))
            cells += f"  {share:<7.1%}" if idx else f"  {'-':<7}"
        print(f"  {'wall-bump share of walk presses':34s}{cells}{note(covered)}")
    top3, top3_cov = series("top3_levels")
    if top3:
        for k, name in enumerate(("1st", "2nd", "3rd")):
            cells = ""
            for q in quarters:
                idx = covered_idx(top3_cov, q)
                levels = [(top3[i] + [0, 0, 0])[k] for i in idx]
                cells += f"  {_mean(levels):<7.1f}" if idx else f"  {'-':<7}"
            print(f"  {name + ' highest level at end':34s}{cells}{note(top3_cov)}")
    nonlead, nonlead_cov = series("nonlead_level_gains")
    lead, lead_cov = series("lead_level_gains")
    if nonlead and lead:
        tail = covered_idx(min(nonlead_cov, lead_cov), quarters[-1])
        total = sum(lead[i] + nonlead[i] for i in tail)
        if total and sum(nonlead[i] for i in tail) / total < 0.1:
            print(
                "  NOTE: under 10% of last-quarter level gains go to non-lead "
                "Pokémon — the party is still one fighter (docs/next-stage-training.md, "
                "point 4)."
            )


def _split(seq, k):
    """Split seq into k near-equal chronological chunks."""
    size, rem = divmod(len(seq), k)
    out, start = [], 0
    for j in range(k):
        end = start + size + (1 if j < rem else 0)
        out.append(seq[start:end])
        start = end
    return out


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
        print(f"  most recent honest: {latest} " f"({n_episodes - last_ep} episodes ago)")
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
    p90_step = statistics.quantiles(deepest, n=10)[-1] if len(deepest) >= 10 else max_step
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
        print(f"  regular probe (episode_length={probe_episode_length or '?'}), " f"last 3 events:")
        idxs = ed.get("probe_rollout_idx", [])
        for i, tr in list(zip(idxs, rates))[-3:]:
            print(f"    rollout {i}: {{{', '.join(f'{k}={v:.0%}' for k, v in tr.items())}}}")
    else:
        print("  no regular probe data")

    assisted_rates = ed.get("assisted_probe_goal_rates", []) or ed.get(
        "assisted_probe_goal_type_rates", []
    )
    if assisted_rates:
        print("  assisted regular probe, last 3 events:")
        assisted_idxs = ed.get("assisted_probe_rollout_idx", [])
        for i, tr in list(zip(assisted_idxs, assisted_rates))[-3:]:
            print(f"    rollout {i}: {{{', '.join(f'{k}={v:.0%}' for k, v in tr.items())}}}")

    long_rates = ed.get("long_probe_goal_rates", []) or ed.get("long_probe_goal_type_rates", [])
    if long_rates:
        long_idxs = ed.get("long_probe_rollout_idx", [])
        print(
            f"  long-horizon probe (episode_length="
            f"{long_probe_episode_length or '?'}), last 3 events:"
        )
        for i, tr in list(zip(long_idxs, long_rates))[-3:]:
            print(f"    rollout {i}: {{{', '.join(f'{k}={v:.0%}' for k, v in tr.items())}}}")
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
        print("  no long-horizon probe data (long_probe_enabled=False, or none " "have fired yet)")

    cont_rates = ed.get("continuation_probe_goal_rates", [])
    if cont_rates:
        print(
            "  continuation probe (from continuation_probe_state_path, e.g. post-Falkner), "
            "last 3 events:"
        )
        cont_idxs = ed.get("continuation_probe_rollout_idx", [])
        for i, tr in list(zip(cont_idxs, cont_rates))[-3:]:
            print(f"    rollout {i}: {{{', '.join(f'{k}={v:.0%}' for k, v in tr.items())}}}")
    else:
        print("  no continuation probe data (continuation_probe_enabled=False, or none yet)")

    assisted_long_rates = ed.get("assisted_long_probe_goal_rates", []) or ed.get(
        "assisted_long_probe_goal_type_rates", []
    )
    if assisted_long_rates:
        print("  assisted long-horizon probe, last 3 events:")
        assisted_long_idxs = ed.get("assisted_long_probe_rollout_idx", [])
        for i, tr in list(zip(assisted_long_idxs, assisted_long_rates))[-3:]:
            print(f"    rollout {i}: {{{', '.join(f'{k}={v:.0%}' for k, v in tr.items())}}}")


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
            row["seeded" if is_seeded else "honest"].append((episode_idx + 1, int(step)))

    if not rows:
        print("  no durable checkpoints reached")
        return
    print(f"  {'checkpoint':<42}{'honest':>8}{'seeded':>9}{'last honest':>14}")
    for flag_num, row in sorted(
        rows.items(),
        key=lambda item: max([ep for ep, _step in item[1]["honest"] + item[1]["seeded"]]),
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
            # Threshold checkpoints record under a string key
            # (e.g. "new_species_catches_1"), not a flag number.
            try:
                title = checkpoint_title(int(raw_flag))
            except (TypeError, ValueError):
                title = str(raw_flag)
            print(
                f"    {title}: "
                f"{entry.get('status', 'unknown')} -> "
                f"{entry.get('output_folder', '?')}"
            )
    pool = (checkpoint_data or {}).get("frontier_pool", [])
    if checkpoint_data is not None:
        print(f"  persisted Go-Explore frontier pool: {len(pool)} snapshots")


# Delivered-reward domains. With per-fire domain balancing ON, the balancer's
# rescaling is booked as one lump (domain_scale_adjustment) that cannot be
# attributed per source here; the share table then reports raw shares and
# says so. The 2026-10-25 diagnosis computed it with the learned multipliers:
# coverage was 93.5% of delivered honest reward, story 1.4%, catching 0.6%.
_DOMAIN_SOURCES = {
    "coverage": ("frontier", "new_map", "menu"),
    "battle": ("battle", "first_trainer_battle", "first_trainer_battle_win"),
    "story": (
        "flag",
        "checkpoint_honest",
        "checkpoint_seeded",
        "map_rung",
        "key_item",
        "event",
    ),
    "catching": ("pokedex", "catch"),
    "level": ("level",),
    "talk": ("talk",),
}


def report_domain_shares(ed):
    """Share of honest reward by domain, last quarter. A run where coverage
    is most of the reward is optimising the coverage annuity, not the game."""
    print("\n=== Reward share by domain (honest, last quarter) ===")
    sources = ed.get("episode_reward_sources") or []
    seeded = ed.get("episode_seeded") or []
    honest = [i for i in range(len(sources)) if i >= len(seeded) or not seeded[i]]
    if not honest:
        print("  no honest episodes")
        return
    tail = honest[-max(1, len(honest) // 4) :]
    totals = {
        dom: _mean([sum(sources[i].get(s, 0.0) for s in srcs) for i in tail])
        for dom, srcs in _DOMAIN_SOURCES.items()
    }
    balancer = _mean([sources[i].get("domain_scale_adjustment", 0.0) for i in tail])
    grand = sum(max(0.0, v) for v in totals.values())
    for dom, v in totals.items():
        share = (v / grand) if grand > 0 else 0.0
        print(f"  {dom:<10} {v:9.2f}  ({share:5.1%})")
    if abs(balancer) > 1e-6:
        print(
            f"  NOTE: domain balancer adjusted delivered reward by {balancer:+.1f}/episode; "
            "shares above are RAW (pre-balancer) and understate whichever domain "
            "the balancer up-weights."
        )
    if grand > 0 and totals["coverage"] / grand > 0.75:
        print(
            "  WARNING: coverage is >75% of honest reward — the policy is paid mostly "
            "for re-walking ground; story/catching/battle cannot win by size."
        )


def _rung_label(rung):
    # Mirrors visit_archive.rung_label (kept import-free: this tool runs on a
    # bare info.pth). Map rungs are first entries into a map, ids >= 1 << 20.
    rung = int(rung)
    if rung >= 1 << 20:
        rung -= 1 << 20
        return f"map {rung // 256}:{rung % 256}"
    return f"flag {rung}"


def report_curriculum(ed, d):
    """Honest checkpoint competence, reverse-curriculum targets/levels and
    the self-imitation buffer (2026-10-25 mechanisms)."""
    log = ed.get("curriculum_log") or []
    print("\n=== Competence / reverse curriculum / self-imitation ===")
    if not log:
        print("  no curriculum_log entries (mechanisms off, or run predates them)")
        return
    last = log[-1]
    print(
        f"  @ rollout {last['rollout']} (episode {last['episode']}), honest window "
        f"{last['honest_window']} episodes"
    )
    comp = last.get("competence") or {}
    for flag, rate in sorted(comp.items(), key=lambda kv: -kv[1]):
        marker = "  <- RC target" if flag in (last.get("rc_targets") or []) else ""
        print(f"    flag {int(flag):5d}: honest rate {rate:6.1%}{marker}")
    for flag in last.get("rc_targets") or []:
        stats = (last.get("rc_stats") or {}).get(flag, {})
        n = int(stats.get("episodes", 0))
        k = int(stats.get("successes", 0))
        print(
            f"  RC rung {_rung_label(flag)}: start rank {last['rc_levels'].get(flag, 1)} of "
            f"{last['rc_max_rank'].get(flag, 0)}, pool "
            f"{last['rc_pool'].get(flag, 0)} trajectories, seeded success "
            f"{k}/{n}"
        )
    if len(log) >= 2:
        first = log[0].get("competence") or {}
        for flag in sorted(set(first) | set(comp)):
            a, b = first.get(flag, 0.0), comp.get(flag, 0.0)
            if abs(b - a) >= 0.05:
                print(f"  trend flag {flag}: {a:.0%} -> {b:.0%} since rollout {log[0]['rollout']}")
    sil_buf = ed.get("rollout_sil_buffer") or []
    sil_loss = [x for x in (ed.get("rollout_sil_loss") or []) if x]
    if sil_buf:
        print(
            f"  SIL buffer {sil_buf[-1]} segments; mean SIL loss (non-zero rollouts) "
            f"{_mean(sil_loss):.4f} over {len(sil_loss)} rollouts"
        )
    rc_jobs = [j for j in (ed.get("episode_rc_job") or []) if j]
    if rc_jobs:
        print(f"  RC-seeded episodes: {len(rc_jobs)} ({sum(j[2] for j in rc_jobs)} succeeded)")
    # Honest first entries into map rungs, oldest bucket first. A rung that
    # rises then falls back to 0 is a behaviour found and then lost (eps
    # 670-712 of the 2026-10-25 run reached Route 31 7/16, then 0).
    history = ((d or {}).get("curriculum_state") or {}).get("honest_map_history") or []
    rungs = [r for r in (last.get("rc_targets") or []) if int(r) >= 1 << 20]
    if history and rungs:
        bucket = 25
        print(f"  honest map-rung entries per {bucket} honest episodes (oldest first):")
        for r in rungs:
            counts = [
                sum(int(r) in set(m) for m in history[i : i + bucket])
                for i in range(0, len(history), bucket)
            ]
            print(f"    {_rung_label(r)}: {counts}")


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
    p.add_argument(
        "--model-path",
        default=None,
        help="Model checkpoint the Capacity section reads parameter counts from; "
        "defaults to the sibling actor_critic.pth next to info_path. run_config "
        "carries no equivalent key, so this one is never overridden.",
    )
    p.add_argument(
        "--scaling-factor",
        type=float,
        default=None,
        help="Screen scaling_factor applied before the CNN, for the Capacity "
        "receptive-field check; run_config's scaling_factor overrides it when present.",
    )
    p.add_argument(
        "--conv-strides",
        default=None,
        help="Comma-separated conv strides in layer order, e.g. '2,2', for the Capacity "
        "receptive-field check. A state_dict records kernel sizes but no strides, so "
        "run_config's conv_strides overrides this when the run persisted it; the "
        "schedule is never guessed.",
    )
    p.add_argument(
        "--update-frequency",
        type=int,
        default=256,
        help="ppo_update_frequency (transitions per env per rollout), for the Capacity "
        "data-rate check; run_config overrides it when present.",
    )
    p.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="ppo_target_kl, for the Capacity optimisation-pressure check; run_config "
        "overrides it when present.",
    )
    p.add_argument(
        "--hard-kl",
        type=float,
        default=0.06,
        help="ppo_hard_kl, for the Capacity optimisation-pressure check; run_config "
        "overrides it when present.",
    )
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="ppo_max_grad_norm, for the Capacity gradient-allocation check; run_config "
        "overrides it when present. With neither, the over-threshold fraction is "
        "reported as unknown rather than guessed.",
    )
    args = p.parse_args()
    args.conv_strides = _parse_conv_strides(args.conv_strides)

    import torch  # deferred: keep --help fast and dependency-light

    d = torch.load(args.info_path, map_location="cpu", weights_only=False)
    ed = d["episode_data"]
    episode_length = args.episode_length
    if episode_length is None and ed.get("episode_lengths"):
        episode_length = max(ed["episode_lengths"])

    n_episodes = len(ed.get("episode_rewards", []))
    print(f"Episodes: {n_episodes}   Best reward: {d.get('best_reward', 0):.1f}")
    va = d.get("visit_archive", {})
    print(f"Archive: {len(va.get('cells', {}))} cells, " f"{len(va.get('maps', {}))} maps")

    report_entropy(ed, args.band_low, args.band_high, args.coef_floor, args.coef_ceiling)
    report_capacity(ed, d, args, args.info_path)
    report_reward_mix(ed)
    report_domain_shares(ed)
    report_curriculum(ed, d)
    report_battle_and_catching(ed)
    report_party_and_navigation(ed)
    report_stepping(ed)
    report_archive_growth(ed)
    report_discovery_log(ed)
    report_time_to_frontier(ed, episode_length)
    report_checkpoints(ed, d)
    report_probes(ed, args.probe_episode_length, args.long_probe_episode_length)


if __name__ == "__main__":
    sys.exit(main())
