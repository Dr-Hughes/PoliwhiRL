# -*- coding: utf-8 -*-
from PIL import Image
import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools


def epsilon_by_frame(frame_idx, epsilon_start, epsilon_final, epsilon_decay):
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1.0 * frame_idx / epsilon_decay
    )


def image_to_tensor(image, device):
    image = torch.from_numpy(np.transpose(image, (2, 0, 1))) / 255.0
    image = image.to(device)
    return image


def images_to_tensors(images, device):
    tensors = []
    for image in images:
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))) / 255.0
        tensor = tensor.to(device)
        tensors.append(tensor)
    return tensors


def select_action(state, epsilon, device, movements, model):
    if random.random() > epsilon:
        with torch.no_grad():
            return model(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(len(movements))]],
            dtype=torch.long,
            device=device,
        )


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


def plot_best_attempts(results_path, episodes, phase, results):
    # Ensure the results directory exists
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    results_path = os.path.join(results_path, f"best_attempts_{episodes}_{phase}.png")

    # Calculate cumulative mean
    cum_mean = np.cumsum(results) / np.arange(1, len(results) + 1)

    # Create plot
    fig, ax = plt.subplots(1, figsize=(10, 6), dpi=100)
    ax.plot(cum_mean, label="Cumulative Mean", color="blue", linewidth=2)

    ax.set_title("Performance Over Episodes")
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Average Reward")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    fig.tight_layout()

    fig.savefig(results_path)
    np.savetxt(results_path.replace(".png", ".csv"), results, delimiter=",")
    plt.close(fig)


def plot_losses(results_path, episodes, losses):
    # Ensure the results directory exists
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    results_path = os.path.join(results_path, f"losses_{episodes}.png")

    # Calculate cumulative mean
    cum_mean = np.cumsum(np.abs(losses)) / np.arange(1, len(losses) + 1)

    # Create plot
    fig, ax = plt.subplots(1, figsize=(10, 6), dpi=100)
    ax.plot(cum_mean, label="Cumulative Mean", color="blue", linewidth=2)

    ax.set_title("Loss Over Episodes")
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Cumulative Mean Loss")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    fig.tight_layout()

    fig.savefig(results_path)
    np.savetxt(results_path.replace(".png", ".csv"), cum_mean, delimiter=",")

    plt.close(fig)


def log_rewards(batch_rewards):
    return f"Average reward for last batch: {np.mean(batch_rewards)} | Best reward: {np.max(batch_rewards)}"


def chunked_iterable(iterable, size):
    it = iter(iterable)
    for _ in range(0, len(iterable), size):
        yield tuple(itertools.islice(it, size))


def weighted_random_indices(rewards, size=1):
    min_reward = min(rewards)
    if min_reward < 0:
        adjusted_rewards = [reward + abs(min_reward) for reward in rewards]
    else:
        adjusted_rewards = rewards

    # Calculate probabilities
    probabilities = np.array(adjusted_rewards) / np.sum(adjusted_rewards)
    return np.random.choice(len(rewards), size=size, p=probabilities).tolist()


def plot_multiple_metrics(results_path, episodes, metrics_dict):
    # Ensure the results directory exists
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    results_path = os.path.join(results_path, f"training_metrics_{episodes}.png")

    # Create plot
    fig, ax = plt.subplots(1, figsize=(10, 6), dpi=100)

    for label, data in metrics_dict.items():
        # Calculate cumulative mean
        cum_mean = np.cumsum(data) / np.arange(1, len(data) + 1)
        ax.plot(cum_mean, label=label, linewidth=2)

    ax.set_title("Training Metrics Over Episodes")
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Cumulative Mean Value")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    fig.tight_layout()

    fig.savefig(results_path)

    # Save data to CSV
    csv_path = results_path.replace(".png", ".csv")
    with open(csv_path, "w") as f:
        f.write(",".join(metrics_dict.keys()) + "\n")
        for values in zip(*metrics_dict.values()):
            f.write(",".join(map(str, values)) + "\n")

    plt.close(fig)