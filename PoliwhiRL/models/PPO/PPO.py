# -*- coding: utf-8 -*-
import os
import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
from .actorcritic import ActorCritic
from .PPO_memory import PPOMemory
import multiprocessing as mp
from PoliwhiRL.utils.utils import plot_losses, plot_best_attempts, plot_multiple_metrics

def worker(config, env_fn, max_steps, model_state_dict, device, worker_id, episode):
    env = env_fn(config)
    state, _ = env.reset()
    env.episode = episode # because creating a new one will reset the episode count
    if config.get("vision", False):
        height, width, channels = env.get_screen_size()
        input_dim = (height, width, channels)
    else:
        input_dim = env.get_game_area().shape
    output_dim = env.action_space.n
    config["num_actions"] = output_dim
    model = ActorCritic(input_dim, output_dim).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    hidden = model.init_hidden(1)
    episode_reward = 0
    episode_length = 0
    done = False
    if worker_id == 0:
        env.enable_render()

    transitions = []

    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        hidden = (hidden[0].to(device), hidden[1].to(device))
        with torch.no_grad():
            action_probs, value, new_hidden = model(state_tensor, hidden)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, done, _, _ = env.step(action.item())

        episode_reward += reward
        episode_length += 1

        transitions.append(
            (
                state,
                action.cpu(),
                log_prob.cpu(),
                value.cpu(),
                reward,
                done,
                (hidden[0].cpu(), hidden[1].cpu()),
            )
        )

        state = next_state
        hidden = (new_hidden[0].cpu(), new_hidden[1].cpu())

        if worker_id == 0:
            env.record(
                f"PPO_training_{config['episode_length']}_N_goals_{config['N_goals_target']}"
                    )

        if done:
            break

    env.close()
    return transitions, episode_reward, episode_length


class ParallelPPO:
    def __init__(
        self,
        input_dims,
        n_actions,
        config
    ):
        self.device = config.get('device', 'cpu')
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.policy_clip = config.get('policy_clip', 0.2)
        self.n_epochs = config.get('n_epochs', 10)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.num_updates = 0

        self.checkpoint_file = config.get('checkpoint', 'PPO_checkpoint.pth')
        # create folder for checkpoint file if doesnt exist
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)

        self.actor_critic = ActorCritic(input_dims, n_actions).to(self.device)
        
        lr = config.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        lr_decay_step = config.get('lr_decay_step', 1000)
        lr_decay_gamma = config.get('lr_decay_gamma', 0.9)
        self.scheduler = StepLR(
            self.optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma
        )

        batch_size = config.get('batch_size', 64)
        self.memory = PPOMemory(batch_size)

        initial_entropy_coef = config.get('initial_entropy_coef', 0.01)
        final_entropy_coef = config.get('final_entropy_coef', 0.001)
        entropy_decay_steps = config.get('entropy_decay_steps', 1000)
        
        self.entropy_coef = initial_entropy_coef
        self.final_entropy_coef = final_entropy_coef
        self.entropy_decay_steps = entropy_decay_steps
        self.entropy_decay_rate = (
            initial_entropy_coef - final_entropy_coef
        ) / entropy_decay_steps

    def remember(self, state, action, probs, vals, reward, done, hidden):
        self.memory.store_memory(state, action, probs, vals, reward, done, hidden)

    def choose_action(self, state, hidden):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        hidden = (hidden[0].to(self.device), hidden[1].to(self.device))
        action_probs, value, new_hidden = self.actor_critic(state, hidden)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
            (new_hidden[0].cpu().numpy(), new_hidden[1].cpu().numpy()),
        )

    def decay_entropy_coef(self):
        self.entropy_coef = max(
            self.final_entropy_coef, self.entropy_coef - self.entropy_decay_rate
        )

    def clear_memory(self):
        self.memory.clear_memory()

    def learn(self):
        self.num_updates += 1
        print(f"Training update {self.num_updates}")
        
        device = torch.device("mps") if torch.backends.mps.is_available() else self.device
        self.actor_critic.to(device)

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, done_arr, hidden_arr, batches = self.memory.generate_batches()

            values = torch.tensor(vals_arr, device=device)
            advantage = self.compute_gae(reward_arr, values.cpu().numpy(), done_arr)
            advantage = torch.tensor(advantage, device=device)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float, device=device)
                old_probs = torch.tensor(old_prob_arr[batch], device=device)
                actions = torch.tensor(action_arr[batch], device=device)
                
                hidden = self.process_hidden_states(hidden_arr, batch, device)
                action_probs, critic_value, _ = self.actor_critic(states, hidden)
                
                critic_value = critic_value.squeeze()
                dist = Categorical(action_probs)
                new_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = ((returns - critic_value) ** 2).mean()

                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()

        self.memory.clear_memory()
        if device.type == "mps":
            self.actor_critic.to("cpu")

        return total_loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()

    def compute_gae(self, rewards, values, dones):
        gae = 0
        advantages = []
        last_value = values[-1]
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[step + 1]
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return np.array(advantages)

    def process_hidden_states(self, hidden_arr, batch, device):
        hidden_batch = [hidden_arr[i] for i in batch]
        hidden_tensor = (
            torch.stack([torch.tensor(h[0]) for h in hidden_batch]).transpose(0, 1),
            torch.stack([torch.tensor(h[1]) for h in hidden_batch]).transpose(0, 1)
        )
        return (hidden_tensor[0].to(device), hidden_tensor[1].to(device))
    def save_models(self):
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "num_updates": self.num_updates,
                "entropy_coef": self.entropy_coef,
            },
            self.checkpoint_file,
            _use_new_zipfile_serialization=True,
        )

    def load_models(self):
        try:
            checkpoint = torch.load(self.checkpoint_file, map_location=self.device, weights_only=True)
            self.actor_critic.load_state_dict(checkpoint["actor_critic"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.num_updates = checkpoint.get("num_updates", 0)
            self.entropy_coef = checkpoint.get("entropy_coef", self.entropy_coef)
            print("Checkpoint loaded successfully.")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch.")

    def get_state_dict(self):
        return {
            "actor_critic": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "num_updates": self.num_updates,
            "entropy_coef": self.entropy_coef,
        }

    def load_state_dict(self, state_dict):
        self.actor_critic.load_state_dict(state_dict["actor_critic"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.num_updates = state_dict.get("num_updates", 0)
        self.entropy_coef = state_dict.get("entropy_coef", self.entropy_coef)

    def train_parallel(self, env_fn, config):
        num_episodes = config["num_episodes"]
        max_steps = config["episode_length"]
        update_interval = config["update_interval"]
        save_interval = config["checkpoint_interval"]

        num_processes = mp.cpu_count()
        pool = mp.Pool(processes=num_processes)

        all_rewards = []
        all_lengths = []
        all_losses = []
        all_policy_losses = []
        all_value_losses = []
        all_entropies = []

        pbar = tqdm(total=num_episodes, desc="Training Progress")
        episode = 0
        while episode < num_episodes:
            # Clear the memory before collecting new experiences
            self.clear_memory()

            # Collect experiences from parallel environments
            model_state_dict = self.actor_critic.state_dict()
            results = pool.starmap(
                worker,
                [
                    (config, env_fn, max_steps, model_state_dict, self.device, i, episode)
                    for i in range(num_processes)
                ],
            )

            # Process the results and store in memory
            for transitions, episode_reward, episode_length in results:
                for state, action, log_prob, value, reward, done, hidden in transitions:
                    self.remember(state, action, log_prob, value, reward, done, hidden)

                all_rewards.append(episode_reward)
                all_lengths.append(episode_length)

                pbar.set_postfix(
                    {"Reward": f"{episode_reward:.2f}", "Length": episode_length}
                )
                pbar.update(1)
                episode += 1

                if episode >= num_episodes:
                    break

            # Perform a single learning update
            loss, policy_loss, value_loss, entropy = self.learn()
            all_losses.append(loss)
            all_policy_losses.append(policy_loss)
            all_value_losses.append(value_loss)
            all_entropies.append(entropy)

            # Decay entropy coefficient
            self.decay_entropy_coef()

            # Save the model at specified intervals
            if episode % save_interval < num_processes:
                self.save_models()

            # Perform post-episode jobs
            self._post_episode_jobs(
                config,
                episode,
                all_rewards,
                all_losses,
                all_policy_losses,
                all_value_losses,
                all_entropies,
            )

        pool.close()
        pool.join()
        pbar.close()

        return self

    def _post_episode_jobs(
        self,
        config,
        episode,
        train_rewards,
        losses,
        policy_losses,
        value_losses,
        entropies,
    ):
        if episode % config.get("plot_every", 10) == 0:
            plot_losses(
                "./results/", f"latest_{config['N_goals_target']}_N_goals", losses
            )
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
                "Entropy": entropies,
            },
        )
