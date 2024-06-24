# -*- coding: utf-8 -*-
import random
import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F
import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from .PPO import PPOModel
from PoliwhiRL.utils.utils import image_to_tensor, plot_best_attempts, plot_losses
from PoliwhiRL.environment import PyBoyEnvironment as Env
from .training_memory import EpisodeMemory


def compute_returns(next_value, rewards, masks, gamma=0.99):
    returns = []
    R = next_value
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return torch.tensor(returns)


def make_new_env(config):
    return Env(config)


def setup_environment_and_model(config):
    env = Env(config)
    if config["vision"]:
        height, width, channels = env.get_screen_size()
        input_dim = (channels, height, width)
    else:
        input_dim = np.prod(env.get_game_area().shape)
    output_dim = env.action_space.n
    config["num_actions"] = output_dim
    model = PPOModel(input_dim, output_dim, config["vision"]).to(config["device"])
    start_episode = load_latest_checkpoint(model, config["checkpoint"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
    return env, model, optimizer, scheduler, start_episode


def train(model, env, optimizer, scheduler, config, start_episode):
    losses = []
    train_rewards = []
    env.episode = -1
    for episode in tqdm(
        range(start_episode, start_episode + config["num_episodes"]), desc="Training"
    ):
        episode_rewards, episode_loss = run_episode(model, env, optimizer, config)

        scheduler.step()
        train_rewards.append(np.sum(episode_rewards))
        losses.extend(episode_loss)

        post_episode_jobs(model, config, episode, train_rewards, losses)

    return train_rewards, losses


def run_episode(model, env, optimizer, config):
    state, _ = env.reset()
    episode = env.episode
    episode_rewards = []
    episode_losses = []
    memory = EpisodeMemory()
    done = False
    steps_since_update = 0
    sequence = []
    hidden_state = None  # Initialize hidden state

    # Initialize epsilon for this episode
    epsilon = max(
        config["final_epsilon"],
        config["initial_epsilon"] * (config["epsilon_decay_rate"] ** episode),
    )

    if episode % config["record_frequency"] == 0:
        env.enable_render()

    while not done:
        state_tensor = prepare_state_tensor(state, config)
        sequence.append(state_tensor)

        if len(sequence) == config["sequence_length"]:
            sequence_tensor = torch.cat(sequence, dim=1)
            action, log_prob, value, hidden_state = select_action(
                model, sequence_tensor, hidden_state, epsilon, config
            )
            sequence = sequence[1:]  # Remove the oldest state
        else:
            action = torch.tensor(env.action_space.sample(), device=config["device"])
            log_prob = torch.tensor(0.0, device=config["device"])
            value = torch.tensor(0.0, device=config["device"])

        next_state, reward, done, _, _ = env.step(action.item())

        episode_rewards.append(reward)
        memory.store(
            state_tensor,
            action.unsqueeze(0),
            log_prob.unsqueeze(0),
            value,
            reward,
            done,
            hidden_state,  # Store hidden state in memory
        )

        state = next_state
        steps_since_update += 1

        if episode % config["record_frequency"] == 0:
            env.record(
                f"PPO_training_{config['episode_length']}_N_goals_{config['N_goals_target']}"
            )

        if steps_since_update == config["update_timestep"] or done:
            loss = update_model_from_memory(model, optimizer, memory, config)
            episode_losses.append(loss)
            memory.clear()
            steps_since_update = 0
            hidden_state = None  # Reset hidden state after update

    return episode_rewards, episode_losses


def select_action(model, state, hidden_state, epsilon, config):
    if random.random() < epsilon:
        # Choose a random action
        rnd_number = random.randint(0, config["num_actions"] - 1)
        action = torch.tensor(rnd_number, device=config["device"])
        action_probs, value_estimates, new_hidden_state = model(state, hidden_state)
        last_step_probs = action_probs[0, -1, :]
        last_step_value = value_estimates[0, -1, 0]
        log_prob = torch.log(last_step_probs[action])
    else:
        # Choose action based on the model
        action_probs, value_estimates, new_hidden_state = model(state, hidden_state)
        last_step_probs = action_probs[0, -1, :]  # shape: [num_actions]
        last_step_value = value_estimates[0, -1, 0]  # shape: scalar
        dist = torch.distributions.Categorical(last_step_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

    return action, log_prob, last_step_value, new_hidden_state


def prepare_state_tensor(state, config):
    if config["vision"]:
        return image_to_tensor(state, config["device"]).unsqueeze(0).unsqueeze(0)
    else:
        return (
            torch.tensor(
                state.flatten().astype(np.float32),
                dtype=torch.float32,
                device=config["device"],
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )


def update_model_from_memory(model, optimizer, memory, config):
    states, actions, old_log_probs, old_values, rewards, masks, _ = memory.get_batch()

    # Compute returns
    next_value = old_values[-1]
    returns = compute_returns(next_value, rewards, masks, config["gamma"]).to(
        config["device"]
    )

    # Create batches
    batch_size = config["batch_size"]
    num_sequences = states.size(0)
    losses = []
    for start_idx in range(0, num_sequences, batch_size):
        end_idx = min(start_idx + batch_size, num_sequences)
        batch_states = states[start_idx:end_idx].permute(
            (1, 0, 2, 3, 4) if config["vision"] else (1, 0, 2)
        )
        batch_actions = actions[start_idx:end_idx]
        batch_old_log_probs = old_log_probs[start_idx:end_idx]
        batch_old_values = old_values[start_idx:end_idx]
        batch_returns = returns[start_idx:end_idx]

        loss = update_model(
            model,
            optimizer,
            batch_old_log_probs,
            batch_old_values,
            batch_returns,
            batch_states,
            batch_actions,
            config["epsilon"],
            config["ppo_epochs"],
        )
        losses.append(loss)

    return np.mean(loss)


def update_model(
    model,
    optimizer,
    saved_log_probs,
    saved_values,
    returns,
    sequences,
    actions,
    epsilon=0.2,
    epochs=5,
):

    advantages = returns - saved_values.squeeze(-1)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Detach necessary tensors
    old_log_probs = saved_log_probs.detach()
    advantages = advantages.detach()
    returns = returns.detach()
    hidden_state = None
    for _ in range(epochs):
        # Recalculate probabilities and values
        action_probs, new_values, _ = model(sequences, hidden_state)

        # Adjust shapes
        action_probs = action_probs.squeeze(0)  # Remove batch dimension if present
        new_values = new_values.squeeze(0).squeeze(
            -1
        )  # Remove batch and last dimensions if present

        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        ratio = (new_log_probs - old_log_probs).exp()

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
        action_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(new_values, returns)

        # Entropy bonus
        entropy = dist.entropy().mean()

        # Total loss
        loss = action_loss + 0.5 * value_loss - 0.01 * entropy

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return loss.item()


def post_episode_jobs(model, config, episode, train_rewards, losses):
    if episode % config.get("plot_every", 10) == 0:
        plot_losses("./results/", f"latest_{config['N_goals_target']}_N_goals", losses)
        plot_best_attempts(
            "./results/",
            "latest",
            f"PPO_training_{config['N_goals_target']}_N_goals",
            train_rewards,
        )
    if episode % config.get("checkpoint_interval", 100) == 0:
        save_checkpoint(model, config["checkpoint"], episode)


def load_latest_checkpoint(model, checkpoint_dir):

    if not os.path.isdir(checkpoint_dir):
        print(
            f"No checkpoint directory found at '{checkpoint_dir}'. Starting from scratch."
        )
        return 0
    checkpoints = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("ppo_model_ep") and f.endswith(".pth")
    ]
    if not checkpoints:
        print("No checkpoints found. Starting from scratch.")
        return 0

    episodes = [int(f.split("ep")[1].split(".")[0]) for f in checkpoints]
    latest_episode = max(episodes)
    latest_checkpoint = os.path.join(
        checkpoint_dir, f"ppo_model_ep{latest_episode}.pth"
    )

    model.load_state_dict(
        torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
    )
    print(f"Loaded checkpoint from episode {latest_episode}")
    return latest_episode


def save_checkpoint(model, checkpoint_dir, episode):
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Checkpoint directory {checkpoint_dir} created.")

    checkpoint_path = os.path.join(checkpoint_dir, f"ppo_model_ep{episode}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at '{checkpoint_path}' for episode {episode}.")
