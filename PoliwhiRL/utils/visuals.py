# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt


def record_step(episode_id, step_id, img, button_press, reward, phase, out_dir):
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


def plot_metrics(rewards, losses, episode_steps, button_presses, n, save_loc="Results"):
    os.makedirs(save_loc, exist_ok=True)
    actions = ["", "a", "b", "left", "right", "up", "down", "start", "select"]
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Plot rolling mean of rewards for the last 100 episodes
    rolling_mean_rewards = np.convolve(rewards, np.ones(100), mode="valid") / 100
    ax1.plot(rolling_mean_rewards)
    ax1.set_title("Episode Rewards (Rolling Mean)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    # Plot rolling mean of losses for the last 100 episodes
    rolling_mean_losses = np.convolve(losses, np.ones(100), mode="valid") / 100
    ax2.plot(rolling_mean_losses)
    ax2.set_title("Training Loss (Rolling Mean)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")

    # Plot button presses as bar chart
    button_presses = np.array(button_presses, dtype=int)
    num_actions = len(actions)
    button_presses = np.bincount(button_presses, minlength=num_actions)
    ax3.bar(actions, button_presses)
    ax3.set_title("Button Presses")
    ax3.set_xlabel("Button")

    # Plot rolling mean of episode steps for the last 100 episodes
    rolling_mean_episode_steps = (
        np.convolve(episode_steps, np.ones(100), mode="valid") / 100
    )
    ax4.plot(rolling_mean_episode_steps)
    ax4.set_title("Episode Steps (Rolling Mean)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Steps")

    fig.tight_layout()
    fig.savefig(f"{save_loc}/training_metrics_{n}.png")
    plt.close()
