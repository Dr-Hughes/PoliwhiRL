# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt


def record_step(episode_id, step_id, img, button_press, reward, phase, out_dir):
    # Ensure out_dir exists and is properly formatted
    if out_dir is None or out_dir == "":
        out_dir = "Results"

    # Construct the full save directory path
    save_dir = os.path.join(out_dir, phase)
    if episode_id != -1:
        save_dir = os.path.join(save_dir, str(episode_id))

    # Create all necessary directories at once
    os.makedirs(save_dir, exist_ok=True)

    # Construct the filename
    filename = f"step_{step_id}_btn_{button_press}_reward_{np.around(reward, 4)}.png"

    # Save the image
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
):
    os.makedirs(save_loc, exist_ok=True)
    actions = ["", "a", "b", "left", "right", "up", "down", "start", "select"]

    # Create 2x3 grid if entropy data is provided, otherwise 2x2
    if entropies is not None and len(entropies) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(30, 15))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        ax1, ax2, ax3, ax4 = axes.flatten()

    # Add title prefix if provided
    prefix = f"{title_prefix} - " if title_prefix else ""

    # Plot cumulative mean of rewards
    cumulative_mean_rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    ax1.plot(cumulative_mean_rewards)
    ax1.set_title(f"{prefix}Episode Rewards (Cumulative Mean)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    # Plot cumulative mean of losses
    cumulative_mean_losses = np.cumsum(losses) / np.arange(1, len(losses) + 1)
    ax2.plot(cumulative_mean_losses)
    ax2.set_title(f"{prefix}Training Loss (Cumulative Mean)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")

    # Plot button presses as bar chart (now tracking up to 1000)
    button_presses = np.array(button_presses, dtype=int)
    num_actions = len(actions)
    button_counts = np.bincount(button_presses, minlength=num_actions)
    ax3.bar(actions, button_counts)
    ax3.set_title(f"{prefix}Button Presses (Last {len(button_presses)} actions)")
    ax3.set_xlabel("Button")
    ax3.set_ylabel("Count")
    # Add count labels on bars
    for i, count in enumerate(button_counts):
        if count > 0:
            ax3.text(i, count, str(count), ha="center", va="bottom")

    # Plot cumulative mean of episode steps
    cumulative_mean_episode_steps = np.cumsum(episode_steps) / np.arange(
        1, len(episode_steps) + 1
    )
    ax4.plot(cumulative_mean_episode_steps)
    ax4.set_title(f"{prefix}Episode Steps (Cumulative Mean)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Steps")

    # Plot entropy if provided
    if entropies is not None and len(entropies) > 0:
        ax5.plot(entropies)
        ax5.set_title(f"{prefix}Entropy Coefficient")
        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Entropy")
        ax5.grid(True, alpha=0.3)

        # Plot distribution of button presses over time
        if len(button_presses) > 100:
            # Show button distribution in windows
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
        else:
            ax6.text(
                0.5,
                0.5,
                "Not enough data for diversity plot",
                ha="center",
                va="center",
                transform=ax6.transAxes,
            )
            ax6.set_title(f"{prefix}Button Diversity")

    fig.tight_layout()

    # Include the title prefix in the filename if provided
    filename_prefix = f"{title_prefix.replace(' ', '_')}_" if title_prefix else ""
    fig.savefig(
        f"{save_loc}/{filename_prefix}training_metrics_episode_{episode}_goals_{n}.png"
    )
    plt.close()
