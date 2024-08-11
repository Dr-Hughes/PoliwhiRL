import numpy as np


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
        
        if n_states < self.batch_size:
            return (
                np.array(self.states),
                np.array(self.actions),
                np.array([p.detach().cpu().numpy() for p in self.probs]),
                np.array([v.detach().cpu().numpy() for v in self.vals]),
                np.array(self.rewards),
                np.array(self.dones),
                np.array(self.hiddens),
                [biased_indices]
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
            np.array(self.hiddens),
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


