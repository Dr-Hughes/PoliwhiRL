# -*- coding: utf-8 -*-
import json
import numpy as np
import os
import matplotlib.pyplot as plt


def record_step(
    episode_id,
    step_id,
    img,
    button_press,
    reward,
    phase,
    out_dir,
    location=None,
):
    """Save one step's screen image with metadata in the filename.

    `location`, if provided, is a dict of name -> value (e.g. {"x": 9,
    "y": 1, "map": 6, "bank": 24, "state": "start"}). The canonical
    keys (x, y, map, bank, room) are coerced to int; any extra keys are
    rendered as-is (caller is responsible for filename safety). Each
    key/value is appended to the filename so the user can visually
    verify goal-match conditions without re-running the env.
    """
    if out_dir is None or out_dir == "":
        out_dir = "Results"

    save_dir = os.path.join(out_dir, phase)
    if episode_id != -1:
        save_dir = os.path.join(save_dir, str(episode_id))

    os.makedirs(save_dir, exist_ok=True)

    loc_chunk = ""
    if location:
        # Render in a stable order so files sort consistently and the
        # filename is grep-able for any one coordinate.
        for key in ("x", "y", "map", "bank", "room"):
            if key in location:
                loc_chunk += f"_{key}_{int(location[key])}"
        # Include any extra keys the caller passed in, in dict order.
        for key, val in location.items():
            if key not in ("x", "y", "map", "bank", "room"):
                loc_chunk += f"_{key}_{val}"

    # round() on a Python float, NOT np.around: rounding a float32 scalar
    # happens in float32 space where -0.02 has no exact representation, so
    # the filename would render at full precision (-0.019999999552965164).
    filename = (
        f"step_{step_id}{loc_chunk}_btn_{button_press}"
        f"_reward_{round(float(reward), 4)}.png"
    )

    img.save(os.path.join(save_dir, filename))


def plot_metrics(
    rewards,
    losses,
    episode_steps,
    button_presses,
    n,
    episode,
    save_loc="Results",
    title_prefix=None,
    entropies=None,
    stage_data_offsets=None,
    state_indices=None,
    goals_total=None,
    goals_made=None,
    goals_target=None,
    flag_fires=None,
    unique_cells=None,
    unique_maps=None,
    archive_size=None,
    reward_sources=None,
    is_probe=None,
    goal_success=None,
    goals_at_start=None,
    policy_entropies=None,
    entropy_coefs=None,
    lrs=None,
    goal_fire_steps=None,
    approx_kls=None,
    clip_fractions=None,
    scaler_ext=None,
    scaler_int=None,
    durations=None,
):
    """Render training-metric plots and a JSON summary dump.

    Headline addition: the PROBE success rate — the fraction of
    probe-env episodes (true-origin starts, never snapshot-seeded) that
    completed the full goal ladder. This is the honest from-scratch
    competence signal; seeded-episode success is plotted alongside for
    contrast. ``policy_entropies`` (rollout-indexed) shows how
    deterministic the behaviour policy actually is, independent of the
    entropy *coefficient*.

    Other signals:

    - ``goals_total`` / ``goals_made`` — total progress fires per episode
      (flag + pokedex + level + xp). ``goals_made`` excludes goals already
      complete at episode start (snapshot-seeded progress).
    - ``flag_fires`` — story-flag transitions per episode. The cleanest
      "real game progress" signal.
    - ``unique_cells`` — quantised cells visited this episode (matches the
      novelty-bonus accounting). Roaming-vs-stuck proxy.
    - ``unique_maps`` — distinct ``(map_bank, map_num)`` visited this
      episode.
    - ``archive_size`` — cumulative cells seen across the whole run. The
      single clearest "is the policy expanding the frontier?" signal —
      monotonically increases and should slope upward across stages.

    Pass any subset of these as parallel-to-rewards lists. Missing arrays
    just leave the corresponding panel blank.
    """
    os.makedirs(save_loc, exist_ok=True)

    shared = dict(
        button_presses=button_presses,
        n=n,
        episode=episode,
        save_loc=save_loc,
        title_prefix=title_prefix,
    )

    _render_metrics(
        rewards=rewards,
        losses=losses,
        episode_steps=episode_steps,
        entropies=entropies,
        filename_suffix="",
        title_suffix="",
        state_indices=state_indices,
        goals_total=goals_total,
        goals_made=goals_made,
        goals_target=goals_target,
        flag_fires=flag_fires,
        unique_cells=unique_cells,
        unique_maps=unique_maps,
        archive_size=archive_size,
        reward_sources=reward_sources,
        is_probe=is_probe,
        goal_success=goal_success,
        goals_at_start=goals_at_start,
        policy_entropies=policy_entropies,
        entropy_coefs=entropy_coefs,
        lrs=lrs,
        goal_fire_steps=goal_fire_steps,
        approx_kls=approx_kls,
        clip_fractions=clip_fractions,
        scaler_ext=scaler_ext,
        scaler_int=scaler_int,
        durations=durations,
        **shared,
    )

    # When resumed from a checkpoint, also render a plot of just this
    # stage's data. The offsets dict was set by the agent at load_model.
    if stage_data_offsets:
        offs = {
            "rewards": stage_data_offsets.get("rewards", 0),
            "losses": stage_data_offsets.get("losses", 0),
            "steps": stage_data_offsets.get("steps", 0),
            "entropies": stage_data_offsets.get("entropies", 0),
            "state_indices": stage_data_offsets.get("state_indices", 0),
            "goals_total": stage_data_offsets.get("goals_total", 0),
            "goals_made": stage_data_offsets.get("goals_made", 0),
            "goals_target": stage_data_offsets.get("goals_target", 0),
            "flag_fires": stage_data_offsets.get("flag_fires", 0),
            "unique_cells": stage_data_offsets.get("unique_cells", 0),
            "unique_maps": stage_data_offsets.get("unique_maps", 0),
            "archive_size": stage_data_offsets.get("archive_size", 0),
        }
        # Episode-indexed additions share the rewards offset; rollout-indexed
        # series get their own.
        ep_off = stage_data_offsets.get("rewards", 0)
        ro_off = stage_data_offsets.get("rollouts", 0)
        if (offs["rewards"] < len(rewards)
                or offs["losses"] < len(losses)
                or offs["steps"] < len(episode_steps)):
            rs_offset = stage_data_offsets.get("reward_sources", 0) if stage_data_offsets else 0
            _render_metrics(
                rewards=rewards[offs["rewards"]:],
                losses=losses[offs["losses"]:],
                episode_steps=episode_steps[offs["steps"]:],
                entropies=(entropies[offs["entropies"]:] if entropies is not None else None),
                filename_suffix="_current",
                title_suffix=" (current stage)",
                state_indices=(state_indices[offs["state_indices"]:] if state_indices is not None else None),
                goals_total=(goals_total[offs["goals_total"]:] if goals_total is not None else None),
                goals_made=(goals_made[offs["goals_made"]:] if goals_made is not None else None),
                goals_target=(goals_target[offs["goals_target"]:] if goals_target is not None else None),
                flag_fires=(flag_fires[offs["flag_fires"]:] if flag_fires is not None else None),
                unique_cells=(unique_cells[offs["unique_cells"]:] if unique_cells is not None else None),
                unique_maps=(unique_maps[offs["unique_maps"]:] if unique_maps is not None else None),
                archive_size=(archive_size[offs["archive_size"]:] if archive_size is not None else None),
                reward_sources=(reward_sources[rs_offset:] if reward_sources is not None else None),
                is_probe=(is_probe[ep_off:] if is_probe is not None else None),
                goal_success=(goal_success[ep_off:] if goal_success is not None else None),
                goals_at_start=(goals_at_start[ep_off:] if goals_at_start is not None else None),
                policy_entropies=(policy_entropies[ro_off:] if policy_entropies is not None else None),
                entropy_coefs=(entropy_coefs[ro_off:] if entropy_coefs is not None else None),
                lrs=(lrs[ro_off:] if lrs is not None else None),
                goal_fire_steps=(goal_fire_steps[ep_off:] if goal_fire_steps is not None else None),
                approx_kls=(approx_kls[ro_off:] if approx_kls is not None else None),
                clip_fractions=(clip_fractions[ro_off:] if clip_fractions is not None else None),
                scaler_ext=(scaler_ext[ro_off:] if scaler_ext is not None else None),
                scaler_int=(scaler_int[ro_off:] if scaler_int is not None else None),
                durations=(durations[ro_off:] if durations is not None else None),
                **shared,
            )


def _probe_rung_survival(goals_total, is_probe, n_target):
    """Per-rung survival series over PROBE episodes: for each k in
    1..n_target, a 0/1 list of whether each probe episode reached at least
    k goals. Returns {} when inputs are missing/misaligned or no target."""
    if (
        goals_total is None
        or is_probe is None
        or not n_target
        or len(goals_total) == 0
        or len(goals_total) != len(is_probe)
    ):
        return {}
    probe_totals = [g for g, p in zip(goals_total, is_probe) if p]
    if not probe_totals:
        return {}
    return {
        k: [1.0 if g >= k else 0.0 for g in probe_totals]
        for k in range(1, int(n_target) + 1)
    }


def _probe_fire_step_series(goal_fire_steps, is_probe):
    """Per-rung fire-step series over PROBE episodes: rung k -> list of the
    episode step at which the k-th goal fired, in episode order. Returns {}
    when inputs are missing/misaligned."""
    if (
        goal_fire_steps is None
        or is_probe is None
        or len(goal_fire_steps) == 0
        or len(goal_fire_steps) != len(is_probe)
    ):
        return {}
    series = {}
    for steps, probe in zip(goal_fire_steps, is_probe):
        if not probe or not steps:
            continue
        for idx, s in enumerate(steps):
            series.setdefault(idx + 1, []).append(float(s))
    return series


def _split_success_series(goal_success, is_probe):
    """Split per-episode success flags into (probe, seeded) series.
    Returns (None, None) when the inputs are missing or misaligned."""
    if (
        goal_success is None
        or is_probe is None
        or len(goal_success) == 0
        or len(goal_success) != len(is_probe)
    ):
        return None, None
    probe = [1.0 if s else 0.0 for s, p in zip(goal_success, is_probe) if p]
    seeded = [1.0 if s else 0.0 for s, p in zip(goal_success, is_probe) if not p]
    return probe, seeded


def _moving_average(arr, window):
    """Right-aligned moving average. Window saturates at array length so
    early entries reflect a partial window (not NaN), keeping the curve
    interpretable from episode 0."""
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return a
    csum = np.cumsum(a)
    counts = np.minimum(np.arange(1, a.size + 1), window)
    # Subtract the value `window` positions back when the window is full
    head = csum.copy()
    if a.size > window:
        head[window:] = csum[window:] - csum[:-window]
    return head / counts


def _render_reward_breakdown(
    reward_sources, save_loc, filename_prefix, filename_suffix, title_prefix
):
    """Render a per-episode per-source reward breakdown figure.

    `reward_sources` is a list parallel to episodes; each entry is a dict
    mapping source name (e.g. ``"flag"``, ``"frontier"``) to that source's
    total contribution that episode. Sources may be missing from individual
    dicts — treated as 0. Two panels: rolling-mean stacked area over time
    (so the user can see signal density evolving) and a bar chart of the
    last-100-episode mean (so they can see which sources actually dominate
    the latest training distribution).
    """
    series_per_source = {}
    for entry in reward_sources:
        if not isinstance(entry, dict):
            continue
        for key in entry:
            series_per_source.setdefault(key, [])
    n_episodes = len(reward_sources)
    for entry, idx in zip(reward_sources, range(n_episodes)):
        for key in series_per_source:
            val = float(entry.get(key, 0.0)) if isinstance(entry, dict) else 0.0
            series_per_source[key].append(val)
    if not series_per_source:
        return
    sorted_keys = sorted(series_per_source.keys())

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    ax_stack, ax_bar = axes

    # Stacked rolling means — gives the "which signals are firing per
    # episode" view that the main reward plot doesn't expose.
    window = min(100, max(1, n_episodes // 5 or 1))
    stack_values = []
    for key in sorted_keys:
        stack_values.append(_moving_average(series_per_source[key], window))
    x = np.arange(n_episodes)
    ax_stack.stackplot(
        x, stack_values, labels=sorted_keys, alpha=0.8,
    )
    ax_stack.set_title(
        f"{title_prefix + ' - ' if title_prefix else ''}Reward Sources per Episode "
        f"({window}-ep rolling mean, stacked)"
    )
    ax_stack.set_xlabel("Completed Episode")
    ax_stack.set_ylabel("Mean reward contribution")
    ax_stack.legend(loc="upper left", fontsize=8)
    ax_stack.grid(True, alpha=0.3)

    # Last-100 bar — answers "which sources actually matter right now?"
    last100_mean = []
    for key in sorted_keys:
        last100 = series_per_source[key][-100:]
        last100_mean.append(float(np.mean(last100)) if last100 else 0.0)
    bars = ax_bar.bar(sorted_keys, last100_mean)
    ax_bar.set_title(
        f"{title_prefix + ' - ' if title_prefix else ''}Reward Sources (last 100 episodes mean)"
    )
    ax_bar.set_xlabel("Source")
    ax_bar.set_ylabel("Mean per-episode contribution")
    ax_bar.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, last100_mean):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(f"{save_loc}/{filename_prefix}reward_sources{filename_suffix}.png")
    plt.close()


def _line_panel(axis, series, title, ylabel, color, overlay=None):
    """Render a per-episode panel with raw + 100-ep rolling mean.

    ``overlay`` is an optional ``(series, label, linestyle, color)`` tuple
    plotted on top — used by the progress-total panel to show the
    ``n_goals_target`` threshold alongside the per-episode total.
    """
    if series is None or len(series) == 0:
        axis.axis("off")
        return
    arr = np.asarray(series, dtype=float)
    axis.plot(arr, alpha=0.2, color=color, label="raw")
    axis.plot(_moving_average(arr, 100), color=color, label="100-ep rolling mean")
    if overlay is not None:
        ov_series, ov_label, ov_style, ov_color = overlay
        if ov_series is not None and len(ov_series) == len(series):
            axis.plot(
                np.asarray(ov_series, dtype=float),
                linestyle=ov_style, color=ov_color, label=ov_label,
            )
    axis.set_title(title)
    axis.set_xlabel("Completed Episode")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")


def _render_metrics(
    rewards,
    losses,
    episode_steps,
    button_presses,
    n,
    episode,
    save_loc,
    title_prefix,
    entropies,
    filename_suffix,
    title_suffix,
    state_indices=None,
    goals_total=None,
    goals_made=None,
    goals_target=None,
    flag_fires=None,
    unique_cells=None,
    unique_maps=None,
    archive_size=None,
    reward_sources=None,
    is_probe=None,
    goal_success=None,
    goals_at_start=None,
    policy_entropies=None,
    entropy_coefs=None,
    lrs=None,
    goal_fire_steps=None,
    approx_kls=None,
    clip_fractions=None,
    scaler_ext=None,
    scaler_int=None,
    durations=None,
):
    actions = ["", "a", "b", "left", "right", "up", "down", "start", "select"]
    rewards_arr = np.asarray(rewards, dtype=float)
    losses_arr = np.asarray(losses, dtype=float)
    steps_arr = np.asarray(episode_steps, dtype=float)
    button_presses = np.array(button_presses, dtype=int)
    button_counts = np.bincount(button_presses, minlength=len(actions))

    # 6x3 layout: reward / loss / buttons / entropy (operational signals),
    # progress-fires / archive growth / unique cells & maps / flag fires
    # (frontier signals), the from-scratch success row (probe vs seeded
    # success, actual policy entropy, learning rate + scalers), and the
    # ladder/optimisation row (probe rung survival, goal fire steps,
    # KL + clip fraction).
    fig, axes = plt.subplots(6, 3, figsize=(30, 42))
    ax = axes.flatten()
    prefix = (
        f"{title_prefix}{title_suffix} - "
        if title_prefix
        else (f"{title_suffix.strip()} - " if title_suffix.strip() else "")
    )

    # --- Row 1: training health ---
    if rewards_arr.size:
        ax[0].plot(np.cumsum(rewards_arr) / np.arange(1, len(rewards_arr) + 1),
                   label="cumulative mean", color="C0")
        ax[0].plot(_moving_average(rewards_arr, 100), color="C3", alpha=0.7,
                   label="100-ep rolling mean")
        ax[0].legend(loc="best")
    ax[0].set_title(f"{prefix}Episode Rewards")
    ax[0].set_xlabel("Completed Episode")
    ax[0].set_ylabel("Reward")
    ax[0].grid(True, alpha=0.3)

    if losses_arr.size:
        ax[1].plot(np.cumsum(losses_arr) / np.arange(1, len(losses_arr) + 1))
    ax[1].set_title(f"{prefix}Training Loss (cumulative mean)")
    ax[1].set_xlabel("PPO Update")
    ax[1].set_ylabel("Loss")
    ax[1].grid(True, alpha=0.3)

    ax[2].bar(actions, button_counts)
    ax[2].set_title(f"{prefix}Button Distribution (last {len(button_presses)} actions)")
    ax[2].set_xlabel("Button")
    ax[2].set_ylabel("Count")
    for i, count in enumerate(button_counts):
        if count > 0:
            ax[2].text(i, count, str(count), ha="center", va="bottom")

    # --- Row 2: policy / exploration health ---
    if entropies is not None and len(entropies) > 0:
        ax[3].plot(entropies)
        ax[3].set_title(f"{prefix}Entropy Coefficient")
        ax[3].set_xlabel("Completed Episode")
        ax[3].set_ylabel("Entropy coefficient")
        ax[3].grid(True, alpha=0.3)
    else:
        ax[3].axis("off")

    if steps_arr.size:
        ax[4].plot(np.cumsum(steps_arr) / np.arange(1, len(steps_arr) + 1))
    ax[4].set_title(f"{prefix}Episode Length (cumulative mean)")
    ax[4].set_xlabel("Completed Episode")
    ax[4].set_ylabel("Steps")
    ax[4].grid(True, alpha=0.3)

    if len(button_presses) > 100:
        window_size = 100
        num_windows = len(button_presses) // window_size
        diversity = [
            len(np.unique(button_presses[i*window_size:(i+1)*window_size]))
            for i in range(num_windows)
        ]
        ax[5].plot(diversity)
        ax[5].set_title(f"{prefix}Button Diversity (unique per 100 steps)")
        ax[5].set_xlabel("Window")
        ax[5].set_ylabel("Unique buttons")
        ax[5].grid(True, alpha=0.3)
    else:
        ax[5].axis("off")

    # --- Row 3: Phase-4 progress signals (the headline metrics) ---
    _line_panel(
        ax[6], flag_fires, title=f"{prefix}Flag Fires per Episode",
        ylabel="Story-flag 0→1 transitions", color="C2",
    )
    _line_panel(
        ax[7], goals_total, title=f"{prefix}Progress Fires per Episode (total)",
        ylabel="Flag + pokedex + level + xp fires", color="C0",
        overlay=(goals_target, "target", "--", "C2"),
    )
    _line_panel(
        ax[8], goals_made,
        title=f"{prefix}Fires Contributed by Training (excludes replay prefix)",
        ylabel="Fires made this episode", color="C3",
    )

    # --- Row 4: Phase-4 frontier signals ---
    _line_panel(
        ax[9], unique_maps, title=f"{prefix}Unique Maps per Episode",
        ylabel="(map_bank, map_num) seen this episode", color="C4",
    )
    _line_panel(
        ax[10], unique_cells, title=f"{prefix}Unique Cells per Episode",
        ylabel="Quantised (map, x/4, y/4) seen this episode", color="C5",
    )
    # Archive growth — the single best "is the frontier expanding" signal.
    if archive_size is not None and len(archive_size) > 0:
        ax[11].plot(archive_size, color="C1")
        ax[11].set_title(f"{prefix}Archive Growth (cumulative cells ever seen)")
        ax[11].set_xlabel("Completed Episode")
        ax[11].set_ylabel("|VisitArchive|")
        ax[11].grid(True, alpha=0.3)
    else:
        ax[11].axis("off")

    # --- Row 5: from-scratch competence + optimisation diagnostics ---
    probe_succ, seeded_succ = _split_success_series(goal_success, is_probe)
    if probe_succ is not None:
        if len(probe_succ):
            ax[12].plot(
                _moving_average(probe_succ, 100), color="C2",
                label=f"probe (from scratch, n={len(probe_succ)})",
            )
        if len(seeded_succ):
            ax[12].plot(
                _moving_average(seeded_succ, 100), color="C1", alpha=0.6,
                label=f"seeded (snapshot start, n={len(seeded_succ)})",
            )
        ax[12].set_ylim(-0.05, 1.05)
        ax[12].set_title(f"{prefix}Goal-Success Rate (100-ep rolling)")
        ax[12].set_xlabel("Episode (within group)")
        ax[12].set_ylabel("Success rate")
        ax[12].grid(True, alpha=0.3)
        ax[12].legend(loc="best")
    else:
        ax[12].axis("off")

    if policy_entropies is not None and len(policy_entropies) > 0:
        ax[13].plot(policy_entropies, color="C0", label="policy entropy (nats)")
        if entropy_coefs is not None and len(entropy_coefs) > 0:
            twin = ax[13].twinx()
            twin.plot(entropy_coefs, color="C3", alpha=0.6, label="entropy coef")
            twin.set_ylabel("Entropy coefficient", color="C3")
        ax[13].set_title(f"{prefix}Policy Entropy (behaviour) vs Coefficient")
        ax[13].set_xlabel("Rollout")
        ax[13].set_ylabel("Entropy (nats)", color="C0")
        ax[13].grid(True, alpha=0.3)
    else:
        ax[13].axis("off")

    if lrs is not None and len(lrs) > 0:
        ax[14].plot(lrs, color="C4", label="LR")
        if scaler_ext is not None and len(scaler_ext) > 0:
            twin = ax[14].twinx()
            twin.plot(scaler_ext, color="C0", alpha=0.6, label="scaler ext")
            if scaler_int is not None and len(scaler_int) > 0:
                twin.plot(scaler_int, color="C1", alpha=0.6, label="scaler int")
            twin.set_ylabel("Reward scale factor (1/std)")
            twin.legend(loc="upper right", fontsize=8)
        ax[14].set_title(f"{prefix}Learning Rate + Reward Scaler Factors")
        ax[14].set_xlabel("Rollout")
        ax[14].set_ylabel("LR", color="C4")
        ax[14].grid(True, alpha=0.3)
    else:
        ax[14].axis("off")

    # --- Row 6: ladder + optimisation health ---
    # Probe rung survival: rolling fraction of PROBE episodes reaching at
    # least k goals — the single best "where does the ladder break" view.
    rung_survival = _probe_rung_survival(goals_total, is_probe, n)
    if rung_survival:
        for k, series in rung_survival.items():
            ma = _moving_average(series, 100)
            ax[15].plot(ma, label=f"≥{k} goals (now: {ma[-1]:.0%})")
        ax[15].set_ylim(-0.05, 1.05)
        ax[15].set_title(f"{prefix}Probe Rung Survival (100-ep rolling)")
        ax[15].set_xlabel("Probe episode")
        ax[15].set_ylabel("Fraction reaching ≥ k goals")
        ax[15].grid(True, alpha=0.3)
        ax[15].legend(loc="best", fontsize=8)
    else:
        ax[15].axis("off")

    # Step-of-fire per rung (probe episodes): how deep into the episode
    # budget each rung lands — exposes the bottleneck rung and whether the
    # episode budget is the binding constraint.
    fire_series = _probe_fire_step_series(goal_fire_steps, is_probe)
    if fire_series:
        for k, series in fire_series.items():
            ax[16].plot(
                _moving_average(series, 50), label=f"rung {k} (n={len(series)})",
            )
        ax[16].set_title(f"{prefix}Step of Goal Fire per Rung (probe, 50-fire rolling)")
        ax[16].set_xlabel("Occurrence")
        ax[16].set_ylabel("Episode step at fire")
        ax[16].grid(True, alpha=0.3)
        ax[16].legend(loc="best", fontsize=8)
    else:
        ax[16].axis("off")

    if approx_kls is not None and len(approx_kls) > 0:
        ax[17].plot(approx_kls, color="C0", label="approx KL")
        if clip_fractions is not None and len(clip_fractions) > 0:
            twin = ax[17].twinx()
            twin.plot(clip_fractions, color="C3", alpha=0.6, label="clip fraction")
            twin.set_ylabel("Clip fraction", color="C3")
            twin.set_ylim(0, 1)
        ax[17].set_title(f"{prefix}PPO Update Health (KL + clip fraction)")
        ax[17].set_xlabel("Rollout")
        ax[17].set_ylabel("Approx KL", color="C0")
        ax[17].grid(True, alpha=0.3)
    else:
        ax[17].axis("off")

    fig.tight_layout()

    filename_prefix = f"{title_prefix.replace(' ', '_')}_" if title_prefix else ""
    fig.savefig(f"{save_loc}/{filename_prefix}training_metrics{filename_suffix}.png")
    plt.close()

    # Per-source reward breakdown — separate figure so the main 4x3
    # layout stays stable. Two panels: rolling-mean stacked area of each
    # source over time, plus last-100-episode mean as a bar chart.
    if reward_sources is not None and len(reward_sources) > 0:
        _render_reward_breakdown(
            reward_sources, save_loc, filename_prefix, filename_suffix,
            f"{prefix}".strip(" -"),
        )

    metrics_dir = os.path.join(save_loc, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    last100 = rewards_arr[-100:] if rewards_arr.size else rewards_arr

    summary = {
        "total_episodes": int(rewards_arr.size),
        "mean_reward": float(rewards_arr.mean()) if rewards_arr.size else None,
        "last100_mean_reward": float(last100.mean()) if last100.size else None,
        "max_reward": float(rewards_arr.max()) if rewards_arr.size else None,
        "min_reward": float(rewards_arr.min()) if rewards_arr.size else None,
        "mean_loss": float(losses_arr.mean()) if losses_arr.size else None,
        "mean_episode_length": float(steps_arr.mean()) if steps_arr.size else None,
        "current_entropy": (
            float(entropies[-1])
            if entropies is not None and len(entropies) > 0
            else None
        ),
        "current_policy_entropy": (
            float(policy_entropies[-1])
            if policy_entropies is not None and len(policy_entropies) > 0
            else None
        ),
        "current_lr": (
            float(lrs[-1]) if lrs is not None and len(lrs) > 0 else None
        ),
    }

    # From-scratch competence — the headline number.
    probe_s, seeded_s = _split_success_series(goal_success, is_probe)
    if probe_s is not None:
        summary["probe_episodes"] = len(probe_s)
        summary["probe_success_rate_last100"] = (
            float(np.mean(probe_s[-100:])) if probe_s else None
        )
        summary["seeded_episodes"] = len(seeded_s)
        summary["seeded_success_rate_last100"] = (
            float(np.mean(seeded_s[-100:])) if seeded_s else None
        )

    # Ladder health: where does the run break, and how deep into the
    # episode budget does each rung land?
    rung_survival = _probe_rung_survival(goals_total, is_probe, n)
    if rung_survival:
        summary["probe_rung_survival_last100"] = {
            str(k): float(np.mean(series[-100:]))
            for k, series in rung_survival.items()
        }
    fire_series = _probe_fire_step_series(goal_fire_steps, is_probe)
    if fire_series:
        summary["probe_median_fire_step_last100"] = {
            str(k): float(np.median(series[-100:]))
            for k, series in fire_series.items()
        }

    # Optimisation health snapshot.
    for name, series in (
        ("approx_kl", approx_kls),
        ("clip_fraction", clip_fractions),
        ("scaler_ext", scaler_ext),
        ("scaler_int", scaler_int),
        ("rollout_duration_s", durations),
    ):
        if series is not None and len(series) > 0:
            summary[f"last_{name}"] = float(series[-1])

    # Progress / frontier summary stats — headline metrics under Phase 4.
    def _series_stats(name, series):
        if series is None or len(series) == 0:
            return
        arr = np.asarray(series, dtype=float)
        last100 = arr[-100:]
        summary[f"mean_{name}"] = float(arr.mean())
        summary[f"last100_mean_{name}"] = float(last100.mean())
        summary[f"max_{name}"] = float(arr.max())

    _series_stats("goals_total", goals_total)
    _series_stats("goals_made", goals_made)
    _series_stats("flag_fires", flag_fires)
    _series_stats("unique_cells", unique_cells)
    _series_stats("unique_maps", unique_maps)
    if archive_size is not None and len(archive_size) > 0:
        # Archive is a monotonic counter — the final value is the
        # interesting one, not the mean.
        summary["final_archive_size"] = int(archive_size[-1])
        summary["archive_size_growth_last100"] = (
            int(archive_size[-1]) - int(archive_size[-100])
            if len(archive_size) >= 100 else int(archive_size[-1])
        )

    # Per-source reward breakdown — surface the last-100 mean for each
    # source in the summary so a glance at the JSON answers "which
    # signals are firing right now?" without needing the plot.
    if reward_sources is not None and len(reward_sources) > 0:
        last100_sources = reward_sources[-100:]
        keys = set()
        for entry in last100_sources:
            if isinstance(entry, dict):
                keys.update(entry.keys())
        per_source_mean = {}
        for key in sorted(keys):
            vals = [float(e.get(key, 0.0)) for e in last100_sources if isinstance(e, dict)]
            if vals:
                per_source_mean[key] = float(sum(vals) / len(vals))
        if per_source_mean:
            summary["last100_mean_reward_sources"] = per_source_mean

    stats = {
        "episode": int(episode),
        "n_goals": int(n),
        "title_prefix": title_prefix,
        "scope": "current_stage" if filename_suffix == "_current" else "all",
        "summary": summary,
        "button_counts": {
            action: int(count) for action, count in zip(actions, button_counts)
        },
        "rewards": rewards_arr.tolist(),
        "losses": losses_arr.tolist(),
        "episode_steps": steps_arr.tolist(),
        "entropies": ([float(e) for e in entropies] if entropies is not None else []),
        "state_indices": (
            [int(s) for s in state_indices] if state_indices is not None else []
        ),
        "goals_total": ([int(v) for v in goals_total] if goals_total is not None else []),
        "goals_made": ([int(v) for v in goals_made] if goals_made is not None else []),
        "goals_target": ([int(v) for v in goals_target] if goals_target is not None else []),
        "flag_fires": ([int(v) for v in flag_fires] if flag_fires is not None else []),
        "unique_cells": ([int(v) for v in unique_cells] if unique_cells is not None else []),
        "unique_maps": ([int(v) for v in unique_maps] if unique_maps is not None else []),
        "archive_size": ([int(v) for v in archive_size] if archive_size is not None else []),
        "reward_sources": (
            [
                {k: float(v) for k, v in (entry or {}).items()}
                for entry in reward_sources
            ]
            if reward_sources is not None else []
        ),
        "is_probe": ([bool(v) for v in is_probe] if is_probe is not None else []),
        "goal_success": (
            [bool(v) for v in goal_success] if goal_success is not None else []
        ),
        "goals_at_start": (
            [int(v) for v in goals_at_start] if goals_at_start is not None else []
        ),
        "policy_entropies": (
            [float(v) for v in policy_entropies]
            if policy_entropies is not None else []
        ),
        "entropy_coefs": (
            [float(v) for v in entropy_coefs] if entropy_coefs is not None else []
        ),
        "lrs": ([float(v) for v in lrs] if lrs is not None else []),
        "goal_fire_steps": (
            [[int(s) for s in (steps or [])] for steps in goal_fire_steps]
            if goal_fire_steps is not None else []
        ),
        "approx_kls": (
            [float(v) for v in approx_kls] if approx_kls is not None else []
        ),
        "clip_fractions": (
            [float(v) for v in clip_fractions]
            if clip_fractions is not None else []
        ),
        "scaler_ext": (
            [float(v) for v in scaler_ext] if scaler_ext is not None else []
        ),
        "scaler_int": (
            [float(v) for v in scaler_int] if scaler_int is not None else []
        ),
        "rollout_durations_s": (
            [float(v) for v in durations] if durations is not None else []
        ),
    }

    # Per-state summary so downstream analysis can compare starting
    # states quickly. Reports the Phase-4 fields too where available.
    if (
        state_indices is not None
        and len(state_indices) == rewards_arr.size
        and rewards_arr.size
    ):
        per_state = {}
        state_arr = np.asarray(state_indices, dtype=int)
        def _arr_or_none(series):
            return np.asarray(series, dtype=float) if series is not None and len(series) == rewards_arr.size else None
        gt_arr = _arr_or_none(goals_total)
        gm_arr = _arr_or_none(goals_made)
        ff_arr = _arr_or_none(flag_fires)
        uc_arr = _arr_or_none(unique_cells)
        um_arr = _arr_or_none(unique_maps)
        for idx in sorted(set(state_indices)):
            mask = state_arr == idx
            r = rewards_arr[mask]
            s = steps_arr[mask]
            last = r[-100:] if r.size else r
            entry = {
                "episodes": int(r.size),
                "mean_reward": float(r.mean()) if r.size else None,
                "last100_mean_reward": float(last.mean()) if last.size else None,
                "max_reward": float(r.max()) if r.size else None,
                "min_reward": float(r.min()) if r.size else None,
                "mean_episode_length": float(s.mean()) if s.size else None,
            }
            for key, src in (("goals_total", gt_arr), ("goals_made", gm_arr),
                             ("flag_fires", ff_arr), ("unique_cells", uc_arr),
                             ("unique_maps", um_arr)):
                if src is not None:
                    masked = src[mask]
                    if masked.size:
                        entry[f"mean_{key}"] = float(masked.mean())
                        entry[f"max_{key}"] = float(masked.max())
            per_state[str(int(idx))] = entry
        stats["per_state_summary"] = per_state

    with open(
        os.path.join(
            metrics_dir, f"{filename_prefix}training_metrics{filename_suffix}.json"
        ),
        "w",
    ) as f:
        json.dump(stats, f, indent=2)
