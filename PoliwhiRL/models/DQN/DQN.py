import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from tqdm import tqdm
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.utils.utils import plot_best_attempts
import matplotlib.pyplot as plt
from torch.nn import functional as F

class PokemonRLModel(nn.Module):
    def __init__(self, input_shape, action_size, lstm_size=32, fc_size=64):
        super(PokemonRLModel, self).__init__()
        self.input_shape = input_shape
        self.action_size = action_size
        self.lstm_size = lstm_size

        # Flatten the input
        self.flatten = nn.Flatten()
        flat_size = input_shape[0] * input_shape[1]

        # Fully connected layer before LSTM
        self.fc_pre = nn.Linear(flat_size, fc_size)

        # LSTM layer
        self.lstm = nn.LSTM(fc_size, lstm_size, batch_first=True)

        # Fully connected layers after LSTM
        self.fc_post = nn.Sequential(
            nn.Linear(lstm_size, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, action_size)
        )

    def forward(self, x, hidden_state):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Reshape input to (batch_size * seq_len, *input_shape)
        x = x.view(batch_size * seq_len, *self.input_shape)
        
        x = self.flatten(x)
        x = torch.relu(self.fc_pre(x))
        
        # Reshape to (batch_size, seq_len, fc_size)
        x = x.view(batch_size, seq_len, -1)
        
        x, hidden_state = self.lstm(x, hidden_state)
        
        # Apply fc_post to each timestep
        x = self.fc_post(x.contiguous().view(batch_size * seq_len, -1))
        x = x.view(batch_size, seq_len, self.action_size)
        
        return x, hidden_state
    
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))

class ReplayBuffer:
    def __init__(self, capacity, sequence_length):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)
        self.episode_buffer = []

    def add(self, state, action, reward, next_state, done, lstm_state):
        self.episode_buffer.append((state, action, reward, next_state, done, lstm_state))
        
        if done:
            # Add sequences and their corresponding LSTM states to the main buffer
            for i in range(0, len(self.episode_buffer) - self.sequence_length + 1):
                sequence = self.episode_buffer[i:i+self.sequence_length]
                initial_lstm_state = sequence[0][5]  # LSTM state at the start of the sequence
                self.buffer.append((sequence, initial_lstm_state))
            self.episode_buffer = []

    def sample(self, batch_size):
        sampled_sequences = random.sample(self.buffer, batch_size)
        
        sequences = [seq[0] for seq in sampled_sequences]
        initial_lstm_states = [seq[1] for seq in sampled_sequences]
        
        states = torch.FloatTensor(np.array([step[0] for seq in sequences for step in seq]))
        actions = torch.LongTensor(np.array([step[1] for seq in sequences for step in seq]))
        rewards = torch.FloatTensor(np.array([step[2] for seq in sequences for step in seq]))
        next_states = torch.FloatTensor(np.array([step[3] for seq in sequences for step in seq]))
        dones = torch.FloatTensor(np.array([step[4] for seq in sequences for step in seq]))
        
        # Correctly format LSTM states
        initial_lstm_states = (torch.cat([s[0] for s in initial_lstm_states], dim=1),
                               torch.cat([s[1] for s in initial_lstm_states], dim=1))
        
        # Reshape tensors to (batch_size, sequence_length, *)
        states = states.view(batch_size, self.sequence_length, *states.shape[1:])
        actions = actions.view(batch_size, self.sequence_length)
        rewards = rewards.view(batch_size, self.sequence_length)
        next_states = next_states.view(batch_size, self.sequence_length, *next_states.shape[1:])
        dones = dones.view(batch_size, self.sequence_length)
        
        return states, actions, rewards, next_states, dones, initial_lstm_states

    def __len__(self):
        return len(self.buffer)


class PokemonAgent:
    def __init__(self, state_shape, action_size, sequence_length=3, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_shape = state_shape
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("mps")

        self.model = PokemonRLModel(state_shape, action_size).to(self.device)
        self.target_model = PokemonRLModel(state_shape, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Changed to Huber Loss

        self.replay_buffer = ReplayBuffer(capacity=10000, sequence_length=sequence_length)

    def get_action(self, state, lstm_state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_size), lstm_state

        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        lstm_state = (lstm_state[0].to(self.device), lstm_state[1].to(self.device))
        with torch.no_grad():
            q_values, new_lstm_state = self.model(state, lstm_state)
        return q_values.argmax().item(), (new_lstm_state[0].cpu(), new_lstm_state[1].cpu())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train(self, batch_size):
            if len(self.replay_buffer) < batch_size:
                return

            states, actions, rewards, next_states, dones, lstm_states = self.replay_buffer.sample(batch_size)
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            lstm_states = (lstm_states[0].to(self.device), lstm_states[1].to(self.device))

            # Initialize loss
            total_loss = 0

            # Process the sequence step by step
            for t in range(self.sequence_length):
                # Get current Q values
                current_q_values, lstm_states = self.model(states[:, t:t+1], lstm_states)
                current_q_values = current_q_values.squeeze(1).gather(1, actions[:, t:t+1])

                # Double DQN: get actions from current model
                with torch.no_grad():
                    next_actions = self.model(next_states[:, t:t+1], lstm_states)[0].squeeze(1).argmax(1, keepdim=True)
                    next_q_values = self.target_model(next_states[:, t:t+1], lstm_states)[0].squeeze(1)
                    next_q_values = next_q_values.gather(1, next_actions)

                # Compute target Q values
                target_q_values = rewards[:, t:t+1] + (1 - dones[:, t:t+1]) * self.gamma * next_q_values

                # Compute Huber loss for this step
                loss = self.loss_fn(current_q_values, target_q_values)
                total_loss += loss

            # Average loss over sequence
            average_loss = total_loss / self.sequence_length

            # Backpropagate
            self.optimizer.zero_grad()
            average_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Added gradient clipping
            self.optimizer.step()

            return average_loss.item()
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())


def create_env(config):
    return Env(config)


def setup_and_train(config):
    env = create_env(config)
    state_shape = env.get_game_area().shape
    num_actions = env.action_space.n
    agent = PokemonAgent(state_shape, num_actions)

    try:
        agent.load_model('pokemon_model_final.pth')
    except:
        print("No model found, training from scratch.")

    num_episodes = config['num_episodes']
    batch_size = 512
    target_update_frequency = config['target_update_frequency']

    # Metrics tracking
    episode_rewards = []
    moving_avg_rewards = deque(maxlen=100)
    losses = []
    epsilons = []

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        lstm_state = agent.model.init_hidden(batch_size=1)
        episode_reward = 0
        episode_loss = 0
        step = 0

        while True:
            action, lstm_state = agent.get_action(state, lstm_state)
            next_state, reward, done, _ = env.step(action)
            
            agent.replay_buffer.add(state, action, reward, next_state, done, lstm_state)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                loss = agent.train(batch_size)
                episode_loss += loss



            state = next_state
            step += 1

            if done:
                break

        # Update metrics
        episode_rewards.append(episode_reward)
        moving_avg_rewards.append(episode_reward)
        losses.append(episode_loss / step if step > 0 else 0)
        epsilons.append(agent.epsilon)
        agent.decay_epsilon()
        if episode % target_update_frequency == 0:
            agent.update_target_model()

        # Reporting
        if episode % 10 == 0:
            avg_reward = np.mean(moving_avg_rewards)
            print(f"Episode: {episode}")
            print(f"Episode Reward: {episode_reward}")
            print(f"Average Reward (last 100 episodes): {avg_reward}")
            print(f"Episode Loss: {episode_loss / step if step > 0 else 0}")
            print(f"Epsilon: {agent.epsilon}")
            print("--------------------")

        # Save model periodically
        if episode % 100 == 0:
            agent.save_model(f"pokemon_model_episode_{episode}.pth")

    # Final save
    agent.save_model("pokemon_model_final.pth")

    # Plot metrics
    plot_metrics(episode_rewards, losses, epsilons, config['N_goals_target'])

    return agent, episode_rewards, losses, epsilons

def plot_metrics(rewards, losses, epsilons, n=1):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')

    # Plot losses
    ax2.plot(losses)
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')

    # Plot epsilon decay
    ax3.plot(epsilons)
    ax3.set_title('Epsilon Decay')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')

    plt.tight_layout()
    plt.savefig(f'training_metrics_{n}.png')
    plt.close()