import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from .model import PokemonTransformer
from PoliwhiRL.environment import PyBoyEnvironment as Env
from torch.nn import functional as F


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size, sequence_length):
        batch = []
        for _ in range(batch_size):
            start_index = max(0, len(self.buffer) - sequence_length)
            end_index = len(self.buffer)
            sequence = list(self.buffer)[start_index:end_index]
            
            # Pad the sequence if it's shorter than sequence_length
            if len(sequence) < sequence_length:
                pad_length = sequence_length - len(sequence)
                padding = [(torch.zeros_like(sequence[0][0]), 
                            torch.zeros_like(sequence[0][1]), 
                            torch.zeros_like(sequence[0][2]), 
                            torch.zeros_like(sequence[0][3]), 
                            torch.zeros_like(sequence[0][4]))] * pad_length
                sequence = padding + sequence
            
            batch.append(sequence)
        return batch
    
    def __len__(self):
        return len(self.buffer)

class PokemonAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 32)
        self.sequence_length = config.get("sequence_length", 10)
        
        # Model parameters
        d_model = config.get("d_model", 256)
        nhead = config.get("nhead", 8)
        num_layers = config.get("num_layers", 6)
        
        # Initialize networks
        self.policy_net = PokemonTransformer(state_dim, action_dim, d_model, nhead, num_layers).to(self.device)
        self.target_net = PokemonTransformer(state_dim, action_dim, d_model, nhead, num_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(config.get("memory_size", 10000))
        
        self.current_episode = []
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if random.random() > self.epsilon:
            with torch.no_grad():
                # Use only the most recent sequence_length states
                recent_states = [s[0] for s, _, _, _, _ in self.current_episode[-self.sequence_length+1:]]
                
                # Pad with copies of the first state if needed
                pad_length = max(0, self.sequence_length - len(recent_states) - 1)
                first_state = recent_states[0] if recent_states else state_tensor
                padded_states = [first_state] * pad_length + recent_states + [state_tensor]
                
                states = torch.cat(padded_states).view(self.sequence_length, 1, -1).to(self.device)
                
                # Create an attention mask (False for real data, True for padding)
                attention_mask = torch.zeros(1, self.sequence_length, dtype=torch.bool).to(self.device)
                attention_mask[0, :pad_length] = True
                
                q_values = self.policy_net(states, attention_mask)
                return q_values[-1].max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size, self.sequence_length)
        
        state_batch = torch.stack([torch.cat([s for s, _, _, _, _ in episode]) for episode in batch]).transpose(0, 1).to(self.device)
        action_batch = torch.stack([torch.cat([a for _, a, _, _, _ in episode]) for episode in batch]).transpose(0, 1).to(self.device)
        reward_batch = torch.stack([torch.cat([r for _, _, r, _, _ in episode]) for episode in batch]).transpose(0, 1).to(self.device)
        next_state_batch = torch.stack([torch.cat([ns for _, _, _, ns, _ in episode]) for episode in batch]).transpose(0, 1).to(self.device)
        done_batch = torch.stack([torch.cat([d for _, _, _, _, d in episode]) for episode in batch]).transpose(0, 1).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch)#
        
        # Compute V(s_{t+1}) for all next states.
        next_state_values = self.target_net(next_state_batch).max(2)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma * (1 - done_batch)) + reward_batch
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train(env, agent, config):
    num_episodes = config.get("num_episodes", 1000)
    target_update = config.get("target_update", 10)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action.item())
            
            total_reward += reward
            
            # Flatten the states if they are 2D matrices
            flat_state = state.flatten() if isinstance(state, np.ndarray) else state
            flat_next_state = next_state.flatten() if isinstance(next_state, np.ndarray) else next_state
            
            agent.memory.push(
                torch.FloatTensor(flat_state).unsqueeze(0),
                action,
                torch.FloatTensor([reward]),
                torch.FloatTensor(flat_next_state).unsqueeze(0),
                torch.FloatTensor([done])
            )
            
            state = next_state
            
            agent.optimize_model()
        
        agent.update_epsilon()
        
        if episode % target_update == 0:
            agent.update_target_network()
        
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
def run(config):
    # Example usage
    env = Env(config)
    input_dim = np.prod(env.get_game_area().shape)
    print(input_dim)
    output_dim = env.action_space.n
    config["num_actions"] = output_dim

    agent = PokemonAgent(input_dim, output_dim, config)
    train(env, agent, config)