# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_dim=256):
        super(ActorCritic, self).__init__()

        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out_size(input_dims)

        self.lstm = nn.LSTM(conv_out_size, hidden_dim, batch_first=True)

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def _get_conv_out_size(self, input_dims):
        return self.conv(torch.zeros(1, 1, *input_dims)).shape[1]

    def forward(self, state, hidden):
        conv_out = self.conv(state.unsqueeze(1))  # Add channel dimension
        lstm_out, hidden = self.lstm(conv_out.unsqueeze(1), hidden)
        action_probs = self.actor(lstm_out.squeeze(1))
        value = self.critic(lstm_out.squeeze(1))
        return action_probs, value, hidden

    def act(self, state, hidden):
        action_probs, value, hidden = self.forward(state, hidden)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, hidden

    def init_hidden(self, batch_size):
        return (
            torch.zeros(1, batch_size, self.hidden_dim),
            torch.zeros(1, batch_size, self.hidden_dim),
        )


class PPOMemory:
    def __init__(self, batch_size, reward_bias=0.7):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.hiddens = []
        self.batch_size = batch_size
        self.reward_bias = reward_bias  # Controls the strength of the bias towards high rewards

    def generate_batches(self):
        n_states = len(self.states)
        
        # Calculate sampling probabilities based on rewards
        rewards = np.array(self.rewards)
        reward_probs = self._calculate_reward_probabilities(rewards)
        
        # Create biased indices
        biased_indices = np.random.choice(
            n_states, 
            size=n_states, 
            p=reward_probs, 
            replace=True
        )
        
        # Create batches using biased indices
        batch_start = np.arange(0, n_states, self.batch_size)
        batches = [biased_indices[i : i + self.batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array([p.detach().cpu().numpy() for p in self.probs]),
            np.array([v.detach().cpu().numpy() for v in self.vals]),
            np.array(self.rewards),
            np.array(self.dones),
            self.hiddens,
            batches,
        )

    def _calculate_reward_probabilities(self, rewards):
        # Normalize rewards to be non-negative
        min_reward = np.min(rewards)
        normalized_rewards = rewards - min_reward + 1e-8  # Add small epsilon to avoid division by zero
        
        # Calculate probabilities
        probs = normalized_rewards ** self.reward_bias
        return probs / np.sum(probs)

    def store_memory(self, state, action, probs, vals, reward, done, hidden):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        hidden_to_store = (
            hidden[0].squeeze(1).detach().cpu().numpy(),
            hidden[1].squeeze(1).detach().cpu().numpy(),
        )
        self.hiddens.append(hidden_to_store)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.hiddens = []
class PPO:
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

        self.actor_critic = ActorCritic(input_dims, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
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
            action.cpu(),
            log_prob.cpu(),
            value.cpu(),
            (new_hidden[0].cpu(), new_hidden[1].cpu()),
        )

    def learn(self):
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
            advantage = torch.tensor(advantage).to(self.device)

            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(
                    self.device
                )
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)

                hidden = (
                    torch.tensor(np.stack([h[0] for h in hidden_arr])[batch])
                    .transpose(0, 1)
                    .contiguous()
                    .to(self.device),
                    torch.tensor(np.stack([h[1] for h in hidden_arr])[batch])
                    .transpose(0, 1)
                    .contiguous()
                    .to(self.device),
                )

                action_probs, critic_value, _ = self.actor_critic(states, hidden)
                critic_value = torch.squeeze(critic_value)

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
        return total_loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()

    def save_models(self):
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, "ppo_checkpoint.pth")

    def load_models(self):
        checkpoint = torch.load("ppo_checkpoint.pth", map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])