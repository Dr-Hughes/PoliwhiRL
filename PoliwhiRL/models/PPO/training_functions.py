# -*- coding: utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm
from .PPO import PPO
from PoliwhiRL.utils.utils import plot_best_attempts, plot_losses, plot_multiple_metrics
from PoliwhiRL.environment import PyBoyEnvironment as Env

def make_new_env(config):
    return Env(config)

def setup_environment_and_model(config):
    env = Env(config)
    input_dim = env.get_game_area().shape
    output_dim = env.action_space.n
    config["num_actions"] = output_dim
    device = config["device"]
    model = PPO(input_dim, output_dim, config["learning_rate"], device, batch_size=config["batch_size"])
    start_episode = 0
    return env, model, start_episode

def train(model, env, config, start_episode):
    num_episodes = config["num_episodes"]
    max_steps = config["episode_length"]
    update_interval = config["update_interval"]
    save_interval = config["checkpoint_interval"]
    all_rewards = []
    all_lengths = []
    all_losses = []
    all_policy_losses = []
    all_value_losses = []
    all_entropies = []
    for episode in tqdm(range(start_episode, num_episodes)):
        state, _ = env.reset()
        if episode % config["record_frequency"] == 0:
            env.enable_render()
        episode_reward = 0
        episode_length = 0
        done = False
        hidden = model.actor_critic.init_hidden(1)

        for step in range(max_steps):
            action, log_prob, value, hidden = model.choose_action(state, hidden)
            next_state, reward, done, _, _ = env.step(action.item())
            model.remember(state, action, log_prob, value, reward, done, hidden)
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if episode % config["record_frequency"] == 0:
                env.record(
                    f"PPO_training_{config['episode_length']}_N_goals_{config['N_goals_target']}"
                )

            if done:
                break

        if (episode + 1) % update_interval == 0:
            loss, policy_loss, value_loss, entropy = model.learn()
            all_losses.append(loss)
            all_policy_losses.append(policy_loss)
            all_value_losses.append(value_loss)
            all_entropies.append(entropy)

        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)

        # Save the model if it's time
        if (episode + 1) % save_interval == 0:
            torch.save(
                {
                    "episode": episode,
                    "model_state_dict": model.actor_critic.state_dict(),
                    "reward": episode_reward,
                },
                f"checkpoint_episode_{episode+1}.pth",
            )

        # Print episode stats
        print(
            f"Episode {episode+1}, Reward: {episode_reward}, Length: {episode_length}"
        )

        post_episode_jobs(config, episode, all_rewards, all_losses, all_policy_losses, all_value_losses, all_entropies)

    return model

def post_episode_jobs(config, episode, train_rewards, losses, policy_losses, value_losses, entropies):
    if episode % config.get("plot_every", 10) == 0:
        plot_losses("./results/", f"latest_{config['N_goals_target']}_N_goals", losses)
        plot_best_attempts(
            "./results/",
            "latest",
            f"PPO_training_{config['N_goals_target']}_N_goals",
            train_rewards,
        )
    plot_multiple_metrics(
        "./results/",
        f"{config['N_goals_target']}_N_goals",
        {
            "Policy Loss": policy_losses,
            "Value Loss": value_losses,
            "Entropy": entropies
        }
    )
