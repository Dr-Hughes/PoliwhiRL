# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


def document(episode_id, step_id, img, button_press, reward, phase):
    try:
        if not os.path.isdir("./runs"):
            os.mkdir("./runs")
    except Exception as e:
        print(e)
    fldr = f"./runs/{phase}/"
    # Ensure all directories exist
    os.makedirs(fldr, exist_ok=True)
    save_dir = f"{fldr}/{episode_id}"
    os.makedirs(save_dir, exist_ok=True)
    # Determine if the image is grayscale or RGB and handle accordingly
    if img.ndim == 2:  # Grayscale
        img = Image.fromarray(img, mode="L")  # 'L' mode for grayscale
    elif img.ndim == 3 and img.shape[2] == 3:  # RGB
        img = Image.fromarray(img, mode="RGB")
    elif img.ndim == 3 and img.shape[2] == 1:  # Also grayscale but with shape (H, W, 1)
        img = Image.fromarray(img[:, :, 0], mode="L")
    else:
        raise ValueError("Unsupported image format")
    # Construct filename with relevant information
    filename = f"step_{step_id}_btn_{button_press}_reward_{np.around(reward, 4)}.png"
    # Save image
    img.save(os.path.join(save_dir, filename))


def save_results(results_path, episodes, results):
    # Save results
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    results_path = results_path + f"results_{episodes}.txt"
    with open(results_path, "w") as f:
        f.write(str(results))


def plot_metrics(rewards, losses, epsilons, n=1):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    # Plot losses
    ax2.plot(losses)
    ax2.set_title("Training Loss")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")

    # Plot epsilon decay
    ax3.plot(epsilons)
    ax3.set_title("Epsilon Decay")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Epsilon")

    plt.tight_layout()
    plt.savefig(f"training_metrics_{n}.png")
    plt.close()
