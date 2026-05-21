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

    filename = (
        f"step_{step_id}{loc_chunk}_btn_{button_press}"
        f"_reward_{np.around(reward, 4)}.png"
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
):
    """Render training-metric plots and a JSON dump with summary stats.

    Reward is no longer the headline progress signal — with random
    starting positions (uniform-cutoff replay) it carries noise from
    "how far into the curriculum did this episode start." `goals_total`,
    `goals_made`, and `goals_target` (per-completed-episode arrays) are
    plotted alongside reward so we can see real curriculum progress.

    `goals_total`: number of goals reached by end of episode (cumulative
        across replay + training portion).
    `goals_made`: number of goals reached in the *training* portion only
        (goals_total - goals_at_replay_endpoint). Most direct measure of
        what the policy is contributing.
    `goals_target`: N_goals_target at the time of that episode. Used to
        compute completion fraction.
    `state_indices`: optional list parallel to `rewards`, recording which
        save-state pool index each completed episode used. When provided,
        the rendered JSON includes per-state summary stats so the user
        can analyse performance by starting state offline.
    """
    os.makedirs(save_loc, exist_ok=True)

    _render_metrics(
        rewards=rewards,
        losses=losses,
        episode_steps=episode_steps,
        button_presses=button_presses,
        n=n,
        episode=episode,
        save_loc=save_loc,
        title_prefix=title_prefix,
        entropies=entropies,
        filename_suffix="",
        title_suffix="",
        state_indices=state_indices,
        goals_total=goals_total,
        goals_made=goals_made,
        goals_target=goals_target,
    )

    # When resumed from a checkpoint, also render a plot of just this stage's data.
    if stage_data_offsets:
        r_off = stage_data_offsets.get("rewards", 0)
        l_off = stage_data_offsets.get("losses", 0)
        s_off = stage_data_offsets.get("steps", 0)
        e_off = stage_data_offsets.get("entropies", 0)
        st_off = stage_data_offsets.get("state_indices", 0)
        gt_off = stage_data_offsets.get("goals_total", 0)
        gm_off = stage_data_offsets.get("goals_made", 0)
        gtar_off = stage_data_offsets.get("goals_target", 0)
        if r_off < len(rewards) or l_off < len(losses) or s_off < len(episode_steps):
            _render_metrics(
                rewards=rewards[r_off:],
                losses=losses[l_off:],
                episode_steps=episode_steps[s_off:],
                button_presses=button_presses,
                n=n,
                episode=episode,
                save_loc=save_loc,
                title_prefix=title_prefix,
                entropies=(entropies[e_off:] if entropies is not None else None),
                filename_suffix="_current",
                title_suffix=" (current stage)",
                state_indices=(
                    state_indices[st_off:] if state_indices is not None else None
                ),
                goals_total=(goals_total[gt_off:] if goals_total is not None else None),
                goals_made=(goals_made[gm_off:] if goals_made is not None else None),
                goals_target=(
                    goals_target[gtar_off:] if goals_target is not None else None
                ),
            )


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
):
    actions = ["", "a", "b", "left", "right", "up", "down", "start", "select"]

    has_entropy = entropies is not None and len(entropies) > 0
    has_goals = (
        goals_total is not None
        and goals_made is not None
        and len(goals_total) == len(rewards)
        and len(rewards) > 0
    )

    # Layout: 3x3 if we have goals (which is the new default). Falls back
    # to the older 2x3 / 2x2 layouts when running against checkpoints
    # saved before the goal-metric arrays existed.
    if has_goals:
        fig, axes = plt.subplots(3, 3, figsize=(30, 22))
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axes.flatten()
    elif has_entropy:
        fig, axes = plt.subplots(2, 3, figsize=(30, 15))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        ax7 = ax8 = ax9 = None
    else:
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        ax1, ax2, ax3, ax4 = axes.flatten()
        ax5 = ax6 = ax7 = ax8 = ax9 = None

    prefix = (
        f"{title_prefix}{title_suffix} - "
        if title_prefix
        else (f"{title_suffix.strip()} - " if title_suffix.strip() else "")
    )

    # Reward kept but de-emphasised — plot both cumulative-mean (smooth
    # long-run signal) and a 100-episode rolling mean (responsive view).
    rewards_arr = np.asarray(rewards, dtype=float)
    if rewards_arr.size:
        cumulative_mean_rewards = np.cumsum(rewards_arr) / np.arange(
            1, len(rewards_arr) + 1
        )
        ax1.plot(cumulative_mean_rewards, label="cumulative mean", color="C0")
        ax1.plot(
            _moving_average(rewards_arr, 100),
            label="100-ep rolling mean",
            color="C3",
            alpha=0.7,
        )
        ax1.legend(loc="best")
    ax1.set_title(f"{prefix}Episode Rewards (noisy with random start positions)")
    ax1.set_xlabel("Completed Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)

    losses_arr = np.asarray(losses, dtype=float)
    if losses_arr.size:
        cumulative_mean_losses = np.cumsum(losses_arr) / np.arange(
            1, len(losses_arr) + 1
        )
        ax2.plot(cumulative_mean_losses)
    ax2.set_title(f"{prefix}Training Loss per PPO Update (Cumulative Mean)")
    ax2.set_xlabel("PPO Update")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)

    button_presses = np.array(button_presses, dtype=int)
    num_actions = len(actions)
    button_counts = np.bincount(button_presses, minlength=num_actions)
    ax3.bar(actions, button_counts)
    ax3.set_title(f"{prefix}Button Presses (Last {len(button_presses)} actions)")
    ax3.set_xlabel("Button")
    ax3.set_ylabel("Count")
    for i, count in enumerate(button_counts):
        if count > 0:
            ax3.text(i, count, str(count), ha="center", va="bottom")

    steps_arr = np.asarray(episode_steps, dtype=float)
    if steps_arr.size:
        cumulative_mean_episode_steps = np.cumsum(steps_arr) / np.arange(
            1, len(steps_arr) + 1
        )
        ax4.plot(cumulative_mean_episode_steps)
    ax4.set_title(f"{prefix}Episode Steps (Cumulative Mean)")
    ax4.set_xlabel("Completed Episode")
    ax4.set_ylabel("Steps")
    ax4.grid(True, alpha=0.3)

    if ax5 is not None and has_entropy:
        ax5.plot(entropies)
        ax5.set_title(f"{prefix}Entropy Coefficient")
        ax5.set_xlabel("Completed Episode")
        ax5.set_ylabel("Entropy Coefficient")
        ax5.grid(True, alpha=0.3)

    if ax6 is not None and len(button_presses) > 100:
        window_size = 100
        num_windows = len(button_presses) // window_size
        button_diversity = []
        for i in range(num_windows):
            window = button_presses[i * window_size : (i + 1) * window_size]
            unique_buttons = len(np.unique(window))
            button_diversity.append(unique_buttons)
        ax6.plot(button_diversity)
        ax6.set_title(f"{prefix}Button Diversity (unique buttons per 100 steps)")
        ax6.set_xlabel("Window")
        ax6.set_ylabel("Unique Buttons")
        ax6.grid(True, alpha=0.3)
    elif ax6 is not None:
        ax6.text(
            0.5,
            0.5,
            "Not enough data for diversity plot",
            ha="center",
            va="center",
            transform=ax6.transAxes,
        )
        ax6.set_title(f"{prefix}Button Diversity")

    # New: curriculum-progress panels. These are the headline metrics in
    # the random-start-position regime.
    if has_goals and ax7 is not None:
        goals_total_arr = np.asarray(goals_total, dtype=float)
        goals_made_arr = np.asarray(goals_made, dtype=float)
        target_arr = (
            np.asarray(goals_target, dtype=float)
            if goals_target is not None and len(goals_target) == len(goals_total)
            else None
        )

        # Panel 7: goals reached at end of episode, raw + rolling.
        ax7.plot(goals_total_arr, alpha=0.2, color="C0", label="raw")
        ax7.plot(
            _moving_average(goals_total_arr, 100),
            color="C0",
            label="100-ep rolling mean",
        )
        if target_arr is not None:
            ax7.plot(target_arr, linestyle="--", color="C2", label="target")
        ax7.set_title(f"{prefix}Goals at Episode End (total)")
        ax7.set_xlabel("Completed Episode")
        ax7.set_ylabel("Goals reached")
        ax7.grid(True, alpha=0.3)
        ax7.legend(loc="best")

        # Panel 8: goals contributed by the training portion (excludes
        # whatever the replay walked through). This is the cleanest
        # signal for "is the policy actually learning beyond the demo."
        ax8.plot(goals_made_arr, alpha=0.2, color="C3", label="raw")
        ax8.plot(
            _moving_average(goals_made_arr, 100),
            color="C3",
            label="100-ep rolling mean",
        )
        ax8.set_title(f"{prefix}Goals Made by Training (excludes replay prefix)")
        ax8.set_xlabel("Completed Episode")
        ax8.set_ylabel("Goals made")
        ax8.grid(True, alpha=0.3)
        ax8.legend(loc="best")

        # Panel 9: completion fraction = goals_total / N_goals_target.
        if target_arr is not None:
            safe_target = np.where(target_arr > 0, target_arr, 1.0)
            fraction = np.clip(goals_total_arr / safe_target, 0.0, 1.0)
            ax9.plot(fraction, alpha=0.2, color="C1", label="raw")
            ax9.plot(
                _moving_average(fraction, 100),
                color="C1",
                label="100-ep rolling mean",
            )
            ax9.set_title(f"{prefix}Curriculum Completion Fraction")
            ax9.set_xlabel("Completed Episode")
            ax9.set_ylabel("goals_total / N_goals_target")
            ax9.set_ylim(0.0, 1.05)
            ax9.grid(True, alpha=0.3)
            ax9.legend(loc="best")
        else:
            ax9.axis("off")

    fig.tight_layout()

    filename_prefix = f"{title_prefix.replace(' ', '_')}_" if title_prefix else ""
    fig.savefig(f"{save_loc}/{filename_prefix}training_metrics{filename_suffix}.png")
    plt.close()

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
    }

    # Goal/progress summary stats — headline metrics under the new regime.
    if has_goals:
        goals_total_arr = np.asarray(goals_total, dtype=float)
        goals_made_arr = np.asarray(goals_made, dtype=float)
        last100_total = goals_total_arr[-100:]
        last100_made = goals_made_arr[-100:]
        summary.update(
            {
                "mean_goals_total": float(goals_total_arr.mean()),
                "last100_mean_goals_total": float(last100_total.mean()),
                "max_goals_total": int(goals_total_arr.max()),
                "mean_goals_made": float(goals_made_arr.mean()),
                "last100_mean_goals_made": float(last100_made.mean()),
                "max_goals_made": int(goals_made_arr.max()),
            }
        )
        if goals_target is not None and len(goals_target) == len(goals_total):
            target_arr = np.asarray(goals_target, dtype=float)
            safe_target = np.where(target_arr > 0, target_arr, 1.0)
            fraction = np.clip(goals_total_arr / safe_target, 0.0, 1.0)
            last100_frac = fraction[-100:]
            summary.update(
                {
                    "mean_completion_fraction": float(fraction.mean()),
                    "last100_mean_completion_fraction": float(last100_frac.mean()),
                    "max_completion_fraction": float(fraction.max()),
                }
            )

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
        "goals_total": (
            [int(v) for v in goals_total] if goals_total is not None else []
        ),
        "goals_made": ([int(v) for v in goals_made] if goals_made is not None else []),
        "goals_target": (
            [int(v) for v in goals_target] if goals_target is not None else []
        ),
    }

    # Per-state summary so downstream analysis can compare states quickly.
    if (
        state_indices is not None
        and len(state_indices) == rewards_arr.size
        and rewards_arr.size
    ):
        per_state = {}
        state_arr = np.asarray(state_indices, dtype=int)
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
            if has_goals:
                gt = np.asarray(goals_total, dtype=float)[mask]
                gm = np.asarray(goals_made, dtype=float)[mask]
                entry.update(
                    {
                        "mean_goals_total": float(gt.mean()) if gt.size else None,
                        "mean_goals_made": float(gm.mean()) if gm.size else None,
                        "max_goals_total": int(gt.max()) if gt.size else None,
                    }
                )
            per_state[str(int(idx))] = entry
        stats["per_state_summary"] = per_state

    with open(
        os.path.join(
            metrics_dir, f"{filename_prefix}training_metrics{filename_suffix}.json"
        ),
        "w",
    ) as f:
        json.dump(stats, f, indent=2)
