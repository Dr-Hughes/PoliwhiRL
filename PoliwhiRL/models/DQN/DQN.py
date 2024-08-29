import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from tqdm import tqdm
from PoliwhiRL.environment import PyBoyEnvironment as Env
import matplotlib.pyplot as plt
from torch.nn import functional as F

class PokemonRLModel(nn.Module):
    def __init__(self, input_shape, action_size, lstm_size=32, fc_size=64):
        super(PokemonRLModel, self).__init__()
        self.input_shape = input_shape
        self.action_size = action_size
        self.lstm_size = lstm_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of flattened features after conv layers
        conv_out_size = self._get_conv_out_size(input_shape)

        # Fully connected layer before LSTM
        self.fc_pre = nn.Linear(conv_out_size, fc_size)

        # LSTM layer
        self.lstm = nn.LSTM(fc_size, lstm_size, batch_first=True)

        # Fully connected layers after LSTM
        self.fc_post = nn.Sequential(
            nn.Linear(lstm_size, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, action_size)
        )

    def _get_conv_out_size(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x, hidden_state):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Reshape and process with conv layers
        x = x.view(batch_size * seq_len, *self.input_shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size * seq_len, -1)

        x = F.relu(self.fc_pre(x))
        x = x.view(batch_size, seq_len, -1)

        x, hidden_state = self.lstm(x, hidden_state)

        x = self.fc_post(x.contiguous().view(batch_size * seq_len, -1))
        x = x.view(batch_size, seq_len, self.action_size)

        return x, hidden_state

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))
    

class PrioritizedReplayBuffer:
    def __init__(self, capacity, sequence_length, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.episode_buffer = []

    def add(self, state, action, reward, next_state, done, lstm_state):
        self.episode_buffer.append((state, action, reward, next_state, done, lstm_state))

        if done:
            for i in range(0, len(self.episode_buffer) - self.sequence_length + 1):
                sequence = self.episode_buffer[i:i+self.sequence_length]
                initial_lstm_state = sequence[0][5]
                self.buffer.append((sequence, initial_lstm_state))
                self.priorities.append(max(self.priorities, default=1))  # New experiences get max priority
            self.episode_buffer = []

    def sample(self, batch_size):
        total_priority = sum(self.priorities)
        probabilities = np.array(self.priorities) / total_priority
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        sampled_sequences = [self.buffer[idx] for idx in indices]
        
        sequences = [seq[0] for seq in sampled_sequences]
        initial_lstm_states = [seq[1] for seq in sampled_sequences]

        states = torch.FloatTensor(np.array([step[0] for seq in sequences for step in seq]))
        actions = torch.LongTensor(np.array([step[1] for seq in sequences for step in seq]))
        rewards = torch.FloatTensor(np.array([step[2] for seq in sequences for step in seq]))
        next_states = torch.FloatTensor(np.array([step[3] for seq in sequences for step in seq]))
        dones = torch.FloatTensor(np.array([step[4] for seq in sequences for step in seq]))

        initial_lstm_states = (torch.cat([s[0] for s in initial_lstm_states], dim=1),
                               torch.cat([s[1] for s in initial_lstm_states], dim=1))

        states = states.view(batch_size, self.sequence_length, *states.shape[1:])
        actions = actions.view(batch_size, self.sequence_length)
        rewards = rewards.view(batch_size, self.sequence_length)
        next_states = next_states.view(batch_size, self.sequence_length, *next_states.shape[1:])
        dones = dones.view(batch_size, self.sequence_length)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** -self.beta
        weights /= weights.max()
        weights = torch.FloatTensor(weights)

        self.beta = min(1.0, self.beta + self.beta_increment)

        return states, actions, rewards, next_states, dones, initial_lstm_states, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + 1e-5) ** self.alpha  # Small constant to avoid zero priority

    def __len__(self):
        return len(self.buffer)


class PokemonAgent:
    def __init__(self, input_shape, action_size, config, env):
        self.input_shape = input_shape
        self.action_size = action_size
        self.sequence_length = config['sequence_length']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.target_update_frequency = config['target_update_frequency']
        self.batch_size = config['batch_size']
        self.env = env

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.model = PokemonRLModel(input_shape, action_size).to(self.device)
        self.target_model = PokemonRLModel(input_shape, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = PrioritizedReplayBuffer(capacity=20000, sequence_length=self.sequence_length)
        self.train_between_episodes = True  # New flag to control training timing
        self.steps_since_train = 0
        self.train_frequency = 4  # Train every 4 steps if training during episodes

        # Metrics tracking
        self.episode_rewards = []
        self.episode_losses = []
        self.moving_avg_reward = deque(maxlen=100)
        self.moving_avg_loss = deque(maxlen=100)
        self.epsilons = []

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return 0  # Return 0 loss if not enough samples

        states, actions, rewards, next_states, dones, initial_lstm_states, indices, weights = self.replay_buffer.sample(batch_size)

        # Move everything to the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        initial_lstm_states = (initial_lstm_states[0].to(self.device), initial_lstm_states[1].to(self.device))
        weights = weights.to(self.device)

        # Initialize loss
        total_loss = 0
        td_errors = []

        # Initialize LSTM states for both current and target networks
        lstm_state = initial_lstm_states
        target_lstm_state = initial_lstm_states

        # Process the sequence step by step
        for t in range(self.sequence_length):
            # Get current Q values
            current_q_values, lstm_state = self.model(states[:, t:t+1], lstm_state)
            current_q_values = current_q_values.squeeze(1).gather(1, actions[:, t:t+1])

            # Double DQN: get actions from current model
            with torch.no_grad():
                next_q_values, target_lstm_state = self.model(next_states[:, t:t+1], target_lstm_state)
                next_actions = next_q_values.squeeze(1).argmax(1, keepdim=True)
                
                target_next_q_values, _ = self.target_model(next_states[:, t:t+1], target_lstm_state)
                next_q_values = target_next_q_values.squeeze(1).gather(1, next_actions)

            # Compute target Q values
            target_q_values = rewards[:, t:t+1] + (1 - dones[:, t:t+1]) * self.gamma * next_q_values

            # Compute TD error for prioritized replay
            td_error = torch.abs(current_q_values - target_q_values).detach()
            td_errors.append(td_error)

            # Compute Huber loss for this step
            loss = self.loss_fn(current_q_values, target_q_values)
            # Compute importance-sampling weighted loss
            loss = (weights * loss).mean()
            total_loss += loss

        # Average loss over sequence
        average_loss = total_loss / self.sequence_length

        # Backpropagate
        self.optimizer.zero_grad()
        average_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()

        # Update priorities
        td_errors = torch.cat(td_errors, dim=1).mean(dim=1).cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        return average_loss.item()
    def step(self, state, lstm_state):
        action, new_lstm_state = self.get_action(state, lstm_state)
        next_state, reward, done, _ = self.env.step(action)

        self.replay_buffer.add(state, action, reward, next_state, done, lstm_state)
        
        self.steps_since_train += 1

        if not self.train_between_episodes and self.steps_since_train >= self.train_frequency:
            loss = self.train(self.batch_size)
            self.steps_since_train = 0
            self.episode_losses.append(loss)
        
        return next_state, reward, done, new_lstm_state

    def run_episode(self):
        state = self.env.reset()
        lstm_state = self.model.init_hidden(batch_size=1)
        episode_reward = 0
        episode_loss = 0
        done = False

        while not done:
            state, reward, done, lstm_state = self.step(state, lstm_state)
            episode_reward += reward

            self.env.record("DQN")

        if self.train_between_episodes:
            loss = self.train(self.batch_size)
            episode_loss = loss

        self.episode_rewards.append(episode_reward)
        self.moving_avg_reward.append(episode_reward)
        if episode_loss > 0:
            self.episode_losses.append(episode_loss)
            self.moving_avg_loss.append(episode_loss)

        self.epsilons.append(self.epsilon)
        self.decay_epsilon()

        return episode_reward, episode_loss
    
    def get_action(self, state, lstm_state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_size), lstm_state

        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        lstm_state = (lstm_state[0].to(self.device), lstm_state[1].to(self.device))
        with torch.no_grad():
            q_values, new_lstm_state = self.model(state, lstm_state)
        return q_values.argmax().item(), (new_lstm_state[0].cpu(), new_lstm_state[1].cpu())

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)



    def train_agent(self, num_episodes):
        for episode in tqdm(range(num_episodes)):
            episode_reward, episode_loss = self.run_episode()

            if episode % 10 == 0:
                avg_reward = np.mean(self.moving_avg_reward)
                avg_loss = np.mean(self.moving_avg_loss) if self.moving_avg_loss else 0
                print(f"Episode: {episode}")
                print(f"Episode Reward: {episode_reward}")
                print(f"Average Reward (last 100 episodes): {avg_reward}")
                print(f"Episode Loss: {episode_loss}")
                print(f"Average Loss (last 100 episodes): {avg_loss}")
                print(f"Epsilon: {self.epsilon}")
                print("--------------------")

            if episode % self.target_update_frequency == 0:
                self.update_target_model()

        self.save_model("pokemon_model_final.pth")
        return self.episode_rewards, self.episode_losses, self.epsilons

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

def create_env(config):
    return Env(config)


def setup_and_train(config):
    env = create_env(config)

    if config['vision']:
        state_shape = env.get_screen_size()
        # rearrange so it is channels first 
        state_shape = (state_shape[2], state_shape[0], state_shape[1])
    else:
        state_shape = env.get_game_area().shape

    num_actions = env.action_space.n
    agent = PokemonAgent(state_shape, num_actions, config, env)
    try:
        agent.load_model('pokemon_model_final.pth')
    except:
        print("No model found, training from scratch.")

    num_episodes = config['num_episodes']

    episode_rewards, losses, epsilons = agent.train_agent(num_episodes)

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