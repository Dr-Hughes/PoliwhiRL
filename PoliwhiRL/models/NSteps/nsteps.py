import torch

class NStepReturns:
    def __init__(self, gamma, n_steps):
        self.gamma = gamma
        self.n_steps = n_steps
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, reward, value, done):
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self):
        if len(self.rewards) < self.n_steps:
            return None
        
        n_step_return = 0
        for i in range(self.n_steps):
            if self.dones[i]:
                return n_step_return
            n_step_return += (self.gamma ** i) * self.rewards[i]

        if not self.dones[self.n_steps - 1]:
            n_step_return += (self.gamma ** self.n_steps) * self.values[self.n_steps - 1]

        return n_step_return

    def clear(self):
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
