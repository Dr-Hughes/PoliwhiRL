# -*- coding: utf-8 -*-
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


def worker(config, env_fn, max_steps, model_state_dict, device):
    env = env_fn(config)
    state, _ = env.reset()
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

        if done:
            break

    env.close()
    return transitions, episode_reward, episode_length


class ParallelPPO:
    def __init__(
        self,
        input_dims,
        n_actions,
        lr,
        device,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        lr_decay_step=1000,
        lr_decay_gamma=0.9,
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_updates = 0

        self.actor_critic = ActorCritic(input_dims, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.scheduler = StepLR(
            self.optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma
        )
        self.memory = PPOMemory(batch_size)

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

    def clear_memory(self):
        self.memory.clear_memory()

    def learn(self):
        self.num_updates += 1

        # Check if MPS is available
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            self.actor_critic.to(mps_device)
        else:
            mps_device = None

        for _ in range(self.n_epochs):
            (
                state_arr,
                action_arr,
                old_prob_arr,
                vals_arr,
                reward_arr,
                done_arr,
                hidden_arr,
                batches,
            ) = self.memory.generate_batches()

            values = vals_arr

            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (
                        reward_arr[k]
                        + self.gamma * values[k + 1] * (1 - int(done_arr[k]))
                        - values[k]
                    )
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = torch.tensor(advantage)
            values = torch.tensor(values)

            if mps_device:
                advantage = advantage.to(mps_device)
                values = values.to(mps_device)
            else:
                advantage = advantage.to(self.device)
                values = values.to(self.device)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float)
                old_probs = torch.tensor(old_prob_arr[batch])
                actions = torch.tensor(action_arr[batch])

                if mps_device:
                    states = states.to(mps_device)
                    old_probs = old_probs.to(mps_device)
                    actions = actions.to(mps_device)
                else:
                    states = states.to(self.device)
                    old_probs = old_probs.to(self.device)
                    actions = actions.to(self.device)

                # Process each sample in the batch independently
                batch_size = states.shape[0]
                action_probs_list = []
                critic_value_list = []

                for i in range(batch_size):
                    hidden = (
                        torch.tensor(hidden_arr[batch[i]][0]).unsqueeze(0),
                        torch.tensor(hidden_arr[batch[i]][1]).unsqueeze(0),
                    )
                    if mps_device:
                        hidden = (hidden[0].to(mps_device), hidden[1].to(mps_device))
                    else:
                        hidden = (hidden[0].to(self.device), hidden[1].to(self.device))

                    action_probs, critic_value, _ = self.actor_critic(
                        states[i].unsqueeze(0), hidden
                    )
                    action_probs_list.append(action_probs)
                    critic_value_list.append(critic_value)

                action_probs = torch.cat(action_probs_list, dim=0)
                critic_value = torch.cat(critic_value_list, dim=0).squeeze()

                dist = Categorical(action_probs)
                new_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantage[batch]
                )
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

            # Step the scheduler after each epoch
            self.scheduler.step()

        self.memory.clear_memory()

        # Move model back to CPU if it was on MPS
        if mps_device:
            self.actor_critic.to("cpu")

        return total_loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()

    def save_models(self):
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "num_updates": self.num_updates,
            },
            "ppo_checkpoint.pth",
        )

    def load_models(self):
        try:
            checkpoint = torch.load("ppo_checkpoint.pth", map_location=self.device)
            self.actor_critic.load_state_dict(checkpoint["actor_critic"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            try:
                self.num_updates = checkpoint["num_updates"]
            except KeyError:
                self.num_updates = 0
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch.")

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
                    (config, env_fn, max_steps, model_state_dict, self.device)
                    for _ in range(num_processes)
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
