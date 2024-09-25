import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

from .PPOTransformer import PPOTransformer
from PoliwhiRL.models.ICM import ICMModule

class PPOModel:
    def __init__(self, input_shape, action_size, config):
        self.config = config
        self.input_shape = input_shape
        self.action_size = action_size
        self.device = torch.device(self.config["device"])
        
        self.learning_rate = self.config["ppo_learning_rate"]
        self.gamma = self.config["ppo_gamma"]
        self.epsilon = self.config["ppo_epsilon"]
        self.value_loss_coef = self.config["ppo_value_loss_coef"]
        self.entropy_coef = self.config["ppo_entropy_coef"]
        self.entropy_decay = self.config["ppo_entropy_coef_decay"]
        self.entropy_min = self.config["ppo_entropy_coef_min"]
        
        self._initialize_networks()
        self._initialize_optimizers()

    def _initialize_networks(self):
        self.actor_critic = PPOTransformer(self.input_shape, self.action_size).to(self.device)
        self.icm = ICMModule(self.input_shape, self.action_size, self.config)

    def _initialize_optimizers(self):
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self._setup_lr_scheduler()

    def _setup_lr_scheduler(self):
        self.scheduler = CyclicLR(
            self.optimizer,
            base_lr=1e-5,
            max_lr=self.learning_rate,
            step_size_up=100,
            mode="triangular2",
        )

    def get_action(self, state_sequence):
        state_sequence = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state_sequence)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def get_action_half_precision(self, state_sequence):
        with torch.no_grad():
            state_sequence = torch.HalfTensor(state_sequence).unsqueeze(0).to(self.device)
            action_probs, _ = self.actor_critic.half()(state_sequence)
        action = torch.multinomial(action_probs.float(), 1).item()
        return action

    def compute_log_prob(self, state_sequence, action):
        state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        action_probs, _ = self.actor_critic(state_tensor)
        return torch.log(action_probs[0, action]).item()

    def compute_intrinsic_reward(self, state, next_state, action):
        return self.icm.compute_intrinsic_reward(state, next_state, action)

    def update(self, data, episode):
        actor_loss, critic_loss, entropy_loss = self._compute_ppo_losses(data, episode)
        icm_loss = self.icm.update(data["states"][:, -1], data["next_states"][:, -1], data["actions"])

        loss = actor_loss + critic_loss + entropy_loss
        self._update_networks(loss)

        return loss.item(), icm_loss

    def _get_entropy_coef(self, episode):
        return max(self.entropy_coef * self.entropy_decay ** episode, self.entropy_min)

    def _compute_ppo_losses(self, data, episode):
        returns = self._compute_returns(data["rewards"], data["dones"])
        advantages = self._compute_advantages(data["states"], returns)

        new_probs, new_values = self.actor_critic(data["states"])
        new_probs = torch.clamp(new_probs, 1e-10, 1.0)
        new_log_probs = torch.log(new_probs.gather(1, data["actions"].unsqueeze(1))).squeeze()

        ratio = torch.exp(new_log_probs - data["old_log_probs"])
        ratio = torch.clamp(ratio, 0.0, 10.0)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        new_values = new_values.squeeze()
        if new_values.dim() == 0:
            new_values = new_values.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)

        critic_loss = self.value_loss_coef * nn.functional.mse_loss(new_values, returns)

        entropy = -(new_probs * torch.log(new_probs + 1e-10)).sum(dim=-1).mean()
        entropy_loss = -self._get_entropy_coef(episode) * entropy

        if torch.isnan(actor_loss) or torch.isnan(critic_loss) or torch.isnan(entropy_loss):
            raise ValueError("NaN detected in PPO losses")

        return actor_loss, critic_loss, entropy_loss

    def _update_networks(self, ppo_loss):
        self.optimizer.zero_grad()
        ppo_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
        self.optimizer.step()

    def _compute_returns(self, rewards, dones):
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (~dones[t])
            returns[t] = running_return
        return returns

    def _compute_advantages(self, states, returns):
        with torch.no_grad():
            _, state_values = self.actor_critic(states)
            advantages = returns - state_values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), f"{path}/actor_critic.pth")
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pth")
        torch.save(self.scheduler.state_dict(), f"{path}/scheduler.pth")
        self.icm.save(f"{path}/icm")

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(f"{path}/actor_critic.pth", map_location=self.device, weights_only=True))
        self.optimizer.load_state_dict(torch.load(f"{path}/optimizer.pth", map_location=self.device, weights_only=True))
        self.scheduler.load_state_dict(torch.load(f"{path}/scheduler.pth", map_location=self.device, weights_only=True))
        self.icm.load(f"{path}/icm")

    def step_scheduler(self):
        self.scheduler.step()