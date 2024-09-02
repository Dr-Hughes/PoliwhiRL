import torch
import torch.multiprocessing as mp
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.models.DQN.DQNModel import TransformerDQN
from PoliwhiRL.replay import SequenceStorage

class MultiAgentDQN:
    def __init__(self, config, num_agents=4):
        self.config = config
        self.num_agents = num_agents
        self.device = torch.device(config["device"])
        
        # Shared model
        state_shape = config["state_shape"]
        action_size = config["action_size"]
        self.shared_model = TransformerDQN(state_shape, action_size).to(self.device)
        self.shared_model.share_memory()  # Required for multiprocessing
        
        # Shared replay buffer
        self.shared_replay_buffer = SequenceStorage(
            config["db_path"],
            config["replay_buffer_capacity"],
            config["sequence_length"]
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.shared_model.parameters(),
            lr=config["learning_rate"]
        )
        
    def train(self, num_episodes):
        processes = []
        for i in range(self.num_agents):
            p = mp.Process(target=self.agent_process, args=(i, num_episodes))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    
    def agent_process(self, agent_id, num_episodes):
        env = Env(self.config)
        local_model = TransformerDQN(self.config["state_shape"], self.config["action_size"]).to(self.device)
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Sync local model with shared model
                local_model.load_state_dict(self.shared_model.state_dict())
                
                # Select action
                action = self.get_action(local_model, state)
                
                # Take action in environment
                next_state, reward, done, _ = env.step(action)
                
                # Add experience to shared replay buffer
                self.shared_replay_buffer.add(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
                # Train on a batch from the shared replay buffer
                self.train_step()
            
            print(f"Agent {agent_id}, Episode {episode}, Reward: {episode_reward}")
    
    def get_action(self, model, state):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = model(state)
        return q_values[0, -1, :].argmax().item()
    
    def train_step(self):
        if len(self.shared_replay_buffer) < self.config["batch_size"]:
            return
        
        batch = self.shared_replay_buffer.sample(self.config["batch_size"])
        if batch is None:
            return
        
        states, actions, rewards, next_states, dones, sequence_ids, weights = batch
        
        # Move everything to the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Compute Q-values
        current_q_values = self.shared_model(states).gather(2, actions.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            next_q_values = self.shared_model(next_states).max(2)[0]
            target_q_values = rewards + self.config["gamma"] * next_q_values * (~dones)
        
        # Compute loss
        loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        loss = (loss.mean(dim=1) * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.shared_model.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update priorities in the replay buffer
        td_errors = torch.abs(current_q_values - target_q_values).mean(dim=1).detach().cpu().numpy()
        self.shared_replay_buffer.update_priorities(sequence_ids, td_errors)

def setup_and_train_multi_agent(config, num_agents=4):
    multi_agent_dqn = MultiAgentDQN(config, num_agents)
    multi_agent_dqn.train(config["num_episodes"])