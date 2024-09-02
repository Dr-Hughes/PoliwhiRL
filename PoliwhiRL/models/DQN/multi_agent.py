import torch
import torch.multiprocessing as mp
from copy import deepcopy
from .agent import PokemonAgent

from tqdm import tqdm

class MultiAgentPokemon(PokemonAgent):
    def __init__(self, input_shape, action_size, config, env, num_agents=4):
        super().__init__(input_shape, action_size, config, env)
        self.num_agents = num_agents
        self.agents = []
        self.main_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.inference_device = torch.device("cpu")
        
        # Create multiple agents
        for _ in range(num_agents):
            agent = deepcopy(self)
            agent.model.to(self.inference_device)
            agent.model.eval()
            self.agents.append(agent)
        
        # Move main model to MPS if available
        self.model.to(self.main_device)
    
    def collect_experiences(self, agent_id):
        agent = self.agents[agent_id]
        state = agent.env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            action = agent.get_action(state, eval_mode=True)
            next_state, reward, done, _ = agent.env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            episode_steps += 1
        
        return episode_reward, episode_steps
    
    def run_parallel_episodes(self):
        with mp.Pool(processes=self.num_agents) as pool:
            results = pool.map(self.collect_experiences, range(self.num_agents))
        
        total_reward = sum(reward for reward, _ in results)
        total_steps = sum(steps for _, steps in results)
        return total_reward / self.num_agents, total_steps / self.num_agents
    
    def train_agent(self, num_episodes):
        pbar = tqdm(range(num_episodes), desc="Training")
        for n in pbar:
            self.episode = n
            
            # Collect experiences using multiple agents
            avg_reward, avg_steps = self.run_parallel_episodes()
            
            # Train the main model
            episode_loss = 0
            for _ in range(self.epochs):
                loss = self.train()
                episode_loss += loss
            episode_loss /= self.epochs
            
            # Update metrics
            self.episode_rewards.append(avg_reward)
            self.moving_avg_reward.append(avg_reward)
            self.episode_steps.append(avg_steps)
            self.moving_avg_steps.append(avg_steps)
            self.episode_losses.append(episode_loss)
            self.moving_avg_loss.append(episode_loss)
            
            if self.episode % 10 == 0:
                self.report_progress()
            
            if self.episode % self.target_update_frequency == 0:
                self.update_target_model()
                # Update all agent models
                for agent in self.agents:
                    agent.model.load_state_dict(self.model.state_dict())
            
            # Update tqdm progress bar
            pbar.set_postfix({
                "Avg Reward (100 ep)": f"{sum(self.moving_avg_reward) / len(self.moving_avg_reward):.2f}",
                "Avg Steps (100 ep)": f"{sum(self.moving_avg_steps) / len(self.moving_avg_steps):.2f}",
            })
    
    def get_action(self, state, eval_mode=False):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.inference_device)
        with torch.no_grad():
            q_values = self.model(state)
        q_values = q_values[0, -1, :]
        if eval_mode:
            return q_values.argmax().item()
        else:
            temperature = self.get_cyclical_temperature()
            probs = F.softmax(q_values / temperature, dim=0)
            return torch.multinomial(probs, 1).item()
    
    def train(self):
        # Move batch to main device (MPS) for training
        # todo if this is useful then we refactor the main to check the current
        # device as a class variable 
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return 0
        
        states, actions, rewards, next_states, dones, sequence_ids, weights = batch
        states = states.to(self.main_device)
        actions = actions.to(self.main_device)
        rewards = rewards.to(self.main_device)
        next_states = next_states.to(self.main_device)
        dones = dones.to(self.main_device)
        weights = weights.to(self.main_device)
        
        # Compute Q-values for all states in the sequence
        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            # Double DQN: Use online network to select actions
            next_q_values_online = self.model(next_states)
            next_actions = next_q_values_online.max(2)[1]

            # Use target network to evaluate the Q-values of selected actions
            next_q_values_target = self.target_model(next_states)
            next_q_values = next_q_values_target.gather(
                2, next_actions.unsqueeze(-1)
            ).squeeze(-1)

            # Calculate cumulative rewards
            cumulative_rewards = self.calculate_cumulative_rewards(
                rewards, dones, self.gamma
            )

            # Compute target Q-values for all steps in the sequence
            target_q_values = cumulative_rewards + self.gamma * next_q_values * (~dones)

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Apply importance sampling weights
        loss = (loss.mean(dim=1) * weights).mean()

        # Backpropagate and optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities in the replay buffer
        td_errors = (
            torch.abs(current_q_values - target_q_values)
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        self.replay_buffer.update_priorities(sequence_ids, td_errors)

        return loss.item()