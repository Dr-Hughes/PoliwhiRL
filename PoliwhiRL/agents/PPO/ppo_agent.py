from PoliwhiRL.agents.base_agent import BaseAgent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from PoliwhiRL.models.PPO.PPOTransformer import PPOTransformer
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.utils.visuals import plot_metrics
from tqdm import tqdm

class PPOAgent(BaseAgent):
    def __init__(self, input_shape, action_size, config):
        super().__init__()
        self.input_shape = input_shape
        self.action_size = action_size
        self.device = torch.device(config['device'])
        self.config = config
        self.update_parameters_from_config()
        
        self.actor_critic = PPOTransformer(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        

        self.reset_tracking()

    def reset_tracking(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.moving_avg_reward = deque(maxlen=100)
        self.moving_avg_length = deque(maxlen=100)
        self.moving_avg_loss = deque(maxlen=100)
        self.buttons_pressed = deque(maxlen=1000)
        self.buttons_pressed.append(0)

        self.max_episode_length = self.config['episode_length']
        self.states = np.zeros((self.max_episode_length,) + self.input_shape, dtype=np.uint8)
        self.actions = np.zeros(self.max_episode_length, dtype=np.int64)
        self.rewards = np.zeros(self.max_episode_length, dtype=np.float32)
        self.dones = np.zeros(self.max_episode_length, dtype=np.bool_)
        self.log_probs = np.zeros(self.max_episode_length, dtype=np.float32)

    def update_parameters_from_config(self):
        self.episode = 0
        self.num_episodes =self.config['num_episodes']
        self.sequence_length =self.config['sequence_length']
        self.learning_rate =self.config['learning_rate']
        self.gamma =self.config['gamma']
        self.epsilon =self.config['ppo_epsilon']
        self.epochs =self.config['ppo_epochs']
        self.batch_size =self.config['batch_size']
        self.n_goals =self.config["N_goals_target"]
        self.early_stopping_avg_length =self.config["early_stopping_avg_length"]
        self.record_frequency =self.config["record_frequency"]
        self.results_dir =self.config["results_dir"]
        self.export_state_loc =self.config["export_state_loc"]

    def get_action(self, state_sequence):
        state_sequence = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state_sequence)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def run_curriculum(self, start_goal_n, end_goal_n, step_increment):
        for n in range(start_goal_n, end_goal_n + 1):
            self.config["N_goals_target"] = n
            self.config['episode_length'] = self.config['episode_length'] + (step_increment * (n-1))
            self.config['early_stopping_avg_length'] = self.config['episode_length'] // 2
            self.update_parameters_from_config()
            self.train_agent()
            self.reset_tracking()

    def train_agent(self):
        pbar = tqdm(range(self.num_episodes), desc=f"Training (Goals: {self.n_goals})")
        for episode in pbar:
            self.episode += 1
            record_loc = f"N_goals_{self.n_goals}/{self.episode}" if self.episode % self.record_frequency == 0 else None
            episode_length = self.run_episode(record_loc=record_loc)
            self.update_model(episode_length)
            self._update_progress_bar(pbar)
            if self.break_condition():
                break

        # Run a final episode to get the export state
        self.run_episode(save_path=f"{self.export_state_loc}/N_goals_{self.n_goals}.pkl")

    def break_condition(self):
        if (
            np.mean(self.moving_avg_length) < self.early_stopping_avg_length
            and self.early_stopping_avg_length > 0
            and self.episode > 50
        ):
            print(
                "Average Steps are below early stopping threshold! Stopping to prevent overfitting."
            )
            return True
        return False

    def run_episode(self, record_loc=None, save_path=None):
        env = Env(self.config)
        state = env.reset()
        episode_length = 0
        state_sequence = deque([state] * self.sequence_length, maxlen=self.sequence_length)
        if record_loc is not None:
            env.enable_record(record_loc, False)

        for step in range(self.max_episode_length):
            action = self.get_action(np.array(state_sequence))
            next_state, reward, done, _ = env.step(action)
            
            state_tensor = torch.FloatTensor(np.array(state_sequence)).unsqueeze(0).to(self.device)
            action_probs, _ = self.actor_critic(state_tensor)
            log_prob = torch.log(action_probs[0, action]).item()

            self.states[step] = state
            self.actions[step] = action
            self.rewards[step] = reward
            self.dones[step] = done
            self.log_probs[step] = log_prob

            state = next_state
            state_sequence.append(state)
            episode_length += 1

            if done:
                break

        self.episode_rewards.append(sum(self.rewards[:episode_length]))
        self.episode_lengths.append(episode_length)
        self.moving_avg_reward.append(sum(self.rewards[:episode_length]))
        self.moving_avg_length.append(episode_length)

        if save_path is not None:
            env.save_gym_state(save_path)

        return episode_length

    def update_model(self, episode_length):
        total_loss = 0
        for _ in range(self.epochs):
            for idx in range(0, episode_length - self.sequence_length + 1, self.batch_size):
                batch_end = min(idx + self.batch_size, episode_length - self.sequence_length + 1)
                batch_indices = np.arange(idx, batch_end)
                
                batch_states = self._get_state_sequences(batch_indices)
                batch_actions = torch.LongTensor(self.actions[batch_indices + self.sequence_length - 1]).to(self.device)
                batch_rewards = torch.FloatTensor(self.rewards[batch_indices + self.sequence_length - 1]).to(self.device)
                batch_dones = torch.BoolTensor(self.dones[batch_indices + self.sequence_length - 1]).to(self.device)
                batch_old_log_probs = torch.FloatTensor(self.log_probs[batch_indices + self.sequence_length - 1]).to(self.device)

                # Compute returns and advantages
                returns = self._compute_returns(batch_rewards, batch_dones)
                with torch.no_grad():
                    _, state_values = self.actor_critic(batch_states)
                    advantages = returns - state_values.squeeze()

                new_probs, new_values = self.actor_critic(batch_states)
                new_log_probs = torch.log(new_probs.gather(1, batch_actions.unsqueeze(1))).squeeze()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(new_values.squeeze(), returns)

                loss = actor_loss + 0.5 * critic_loss
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()

        avg_loss = total_loss / (self.epochs * (episode_length / self.batch_size))
        self.episode_losses.append(avg_loss)
        self.moving_avg_loss.append(avg_loss)

    def _get_state_sequences(self, indices):
        sequences = np.array([self.states[i:i+self.sequence_length] for i in indices])
        return torch.FloatTensor(sequences).to(self.device)

    def _compute_returns(self, rewards, dones):
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (~dones[t])
            returns[t] = running_return
        return returns

    def _update_progress_bar(self, pbar):
        avg_reward = sum(self.moving_avg_reward) / len(self.moving_avg_reward) if self.moving_avg_reward else 0
        avg_length = sum(self.moving_avg_length) / len(self.moving_avg_length) if self.moving_avg_length else 0
        avg_loss = sum(self.moving_avg_loss) / len(self.moving_avg_loss) if self.moving_avg_loss else 0
        current_reward = self.episode_rewards[-1] if self.episode_rewards else 0
        current_length = self.episode_lengths[-1] if self.episode_lengths else 0
        
        pbar.set_postfix({
            'Episode': self.episode,
            'Avg Reward (100 ep)': f'{avg_reward:.2f}',
            'Avg Length (100 ep)': f'{avg_length:.2f}',
            'Avg Loss (100 ep)': f'{avg_loss:.4f}',
            'Current Reward': f'{current_reward:.2f}',
            'Current Length': current_length
        })
        pbar.update(1)

        if self.episode % 10 == 0 and self.episode > 100:
            plot_metrics(
                self.episode_rewards,
                self.episode_losses,
                self.episode_lengths,
                self.buttons_pressed,
                self.n_goals,
                save_loc=self.results_dir,
            )