# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from PoliwhiRL.models.PPO.PPOTransformer import PPOTransformer


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
        # Offset subtracted from the episode counter before decay so that
        # plateau-triggered resets can "rewind" the schedule and boost
        # exploration without permanently changing the base coefficient.
        self._entropy_reset_offset = 0
        self.clip_value_loss = self.config.get("ppo_clip_value_loss", True)

        self._initialize_networks()
        self._initialize_optimizers()

    def _initialize_networks(self):
        ram_dim = int(self.config["ram_obs_dim"])
        d_ram = int(self.config.get("d_ram", 64))
        mem_len = int(self.config.get("mem_len", 64))
        self.actor_critic = PPOTransformer(
            self.input_shape,
            self.action_size,
            ram_dim=ram_dim,
            d_ram=d_ram,
            mem_len=mem_len,
        ).to(self.device)

    def _initialize_optimizers(self):
        # Adam eps=1e-5 is the standard PPO setting (CleanRL/OpenAI). The
        # PyTorch default 1e-8 over-amplifies the bias correction when
        # gradients spike (e.g. early stages of a curriculum transition).
        adam_eps = float(self.config.get("ppo_adam_eps", 1e-5))
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self.learning_rate, eps=adam_eps
        )
        self._setup_lr_scheduler()

    def _setup_lr_scheduler(self):
        # Cosine anneal from peak LR to lr_min over the planned stage. Replaces
        # the previous CyclicLR(triangular2), whose late peaks coincided with
        # policy convergence and were implicated in mid-run collapses. T_max
        # counts scheduler.step() calls — once per outer iteration in single-env
        # mode, once per rollout in vec mode (which sets ppo_scheduler_t_max).
        t_max = int(
            self.config.get("ppo_scheduler_t_max", self.config.get("num_rollouts", 1))
        )
        t_max = max(1, t_max)
        eta_min = float(self.config.get("ppo_lr_min", 1e-5))
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=t_max, eta_min=eta_min
        )

    def init_mems(self, batch_size=1):
        return self.actor_critic.init_mems(batch_size, self.device)

    def get_action(self, state_sequence, ram_sequence, mems=None):
        state_sequence = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        ram_sequence = torch.FloatTensor(ram_sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _, new_mems = self.actor_critic(
                state_sequence, ram_sequence, mems
            )
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs[0, action] + 1e-10).item()
        return action, log_prob, new_mems

    def compute_log_prob(self, state_sequence, ram_sequence, action, mems=None):
        state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        ram_tensor = torch.FloatTensor(ram_sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _, _ = self.actor_critic(state_tensor, ram_tensor, mems)
        return torch.log(action_probs[0, action] + 1e-10).item()

    def update(self, data, episode):
        actor_loss, critic_loss, entropy_loss, approx_kl = self._compute_ppo_losses(
            data, episode
        )
        loss = actor_loss + critic_loss + entropy_loss
        self._update_networks(loss)
        return loss.item(), approx_kl

    def _get_entropy_coef(self, episode):
        # Linear decay from initial to min over the planned budget, with an
        # offset that lets plateau detection "rewind" part of the schedule.
        total = self.config.get("ppo_entropy_anneal_steps",
                                self.config.get("num_rollouts", 1))
        effective_ep = max(0, episode - self._entropy_reset_offset)
        progress = min(effective_ep / max(total, 1), 1.0)
        return self.entropy_coef * (1 - progress) + self.entropy_min * progress

    def set_entropy_offset(self, offset):
        """Rewind the entropy schedule by setting an episode offset.

        The effective episode used for decay becomes (episode - offset),
        so a larger offset means the schedule is further back and entropy
        is higher. Call this when plateau detection fires to inject a
        temporary exploration boost.
        """
        self._entropy_reset_offset = offset

    def _compute_ppo_losses(self, data, episode):
        use_gae = self.config.get("ppo_gae_lambda", 0) > 0
        mems = data.get("mems", None)

        # Vec agent precomputes per-env GAE before flattening across envs;
        # accept those directly so we don't mistakenly recompute advantages
        # across env boundaries.
        if "returns" in data and "advantages" in data:
            returns = data["returns"]
            advantages = data["advantages"]
            if advantages.shape[0] > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
        else:
            # Bootstrap V(s_{T+1}) for the tail of a truncated rollout. Mid-episode
            # buffer flushes leave the last transition non-terminal; without this
            # the return computation treats it as if the episode ended there.
            last_value = self._tail_bootstrap_value(data, mems)

            if use_gae:
                with torch.no_grad():
                    _, values, _ = self.actor_critic(
                        data["states"], data["ram_states"], mems
                    )
                    values = values.squeeze()

                returns, advantages = self._compute_gae(
                    data["rewards"], values, data["dones"], last_value=last_value
                )
                if advantages.shape[0] > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
            else:
                returns = self._compute_returns(
                    data["rewards"], data["dones"], last_value=last_value
                )
                advantages = self._compute_advantages(
                    data["states"], data["ram_states"], returns, mems
                )

        new_probs, new_values, _ = self.actor_critic(
            data["states"], data["ram_states"], mems
        )
        new_probs = torch.clamp(new_probs, 1e-10, 1.0)
        new_log_probs = torch.log(
            new_probs.gather(1, data["actions"].unsqueeze(1)) + 1e-10
        ).squeeze()

        log_ratio = new_log_probs - data["old_log_probs"]
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        new_values = new_values.squeeze()
        if new_values.dim() == 0:
            new_values = new_values.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)

        old_values = data.get("old_values", None)
        if self.clip_value_loss and old_values is not None:
            # Mirror the actor clip on the critic to limit per-update value drift.
            v_clipped = old_values + torch.clamp(
                new_values - old_values, -self.epsilon, self.epsilon
            )
            v_loss_unclipped = (new_values - returns).pow(2)
            v_loss_clipped = (v_clipped - returns).pow(2)
            critic_loss = (
                self.value_loss_coef
                * 0.5
                * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            )
        else:
            critic_loss = self.value_loss_coef * nn.functional.mse_loss(
                new_values, returns
            )

        entropy = -(new_probs * torch.log(new_probs + 1e-10)).sum(dim=-1).mean()
        entropy_loss = -self._get_entropy_coef(episode) * entropy

        # Schulman's k3 estimator: always non-negative, lower-variance than (old-new).
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()

        if (
            torch.isnan(actor_loss)
            or torch.isnan(critic_loss)
            or torch.isnan(entropy_loss)
        ):
            print(
                f"New probs range: ({new_probs.min().item()}, {new_probs.max().item()})"
            )
            print(f"Ratio range: ({ratio.min().item()}, {ratio.max().item()})")
            print(
                f"Advantages range: ({advantages.min().item()}, {advantages.max().item()})"
            )
            print(f"Returns range: ({returns.min().item()}, {returns.max().item()})")

        return actor_loss, critic_loss, entropy_loss, approx_kl

    def _update_networks(self, ppo_loss):
        self.optimizer.zero_grad()
        ppo_loss.backward()
        max_grad_norm = self.config.get("ppo_max_grad_norm", 0.5)
        torch.nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), max_norm=max_grad_norm
        )
        self.optimizer.step()

    def _tail_bootstrap_value(self, data, mems):
        # Returns V(s_{T+1}) for the last transition in the rollout, or None if
        # the rollout ended at a true terminal (in which case the bootstrap is
        # 0 and the multiplication by (~done) zeros it anyway).
        next_states = data.get("next_states", None)
        next_ram_states = data.get("next_ram_states", None)
        dones = data["dones"]
        if (
            next_states is None
            or next_ram_states is None
            or len(dones) == 0
            or bool(dones[-1].item())
        ):
            return None
        tail_input = next_states[-1:].detach()
        tail_ram = next_ram_states[-1:].detach()
        tail_mems = None
        if mems is not None:
            tail_mems = [m[-1:].detach() for m in mems]
        with torch.no_grad():
            _, tail_v, _ = self.actor_critic(tail_input, tail_ram, tail_mems)
        return tail_v.squeeze().detach()

    def _compute_returns(self, rewards, dones, last_value=None):
        returns = torch.zeros_like(rewards)
        running_return = 0.0 if last_value is None else float(last_value)
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (~dones[t])
            returns[t] = running_return
        return returns

    def _compute_gae(self, rewards, values, dones, last_value=None):
        gae_lambda = self.config.get("ppo_gae_lambda", 0.95)
        advantages = torch.zeros_like(rewards)
        gae = 0
        tail_value = 0.0 if last_value is None else float(last_value)

        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(rewards) else tail_value
            delta = rewards[t] + self.gamma * next_value * (~dones[t]) - values[t]
            gae = delta + self.gamma * gae_lambda * (~dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return returns, advantages

    def _compute_advantages(self, states, ram_states, returns, mems=None):
        with torch.no_grad():
            _, state_values, _ = self.actor_critic(states, ram_states, mems)
            advantages = returns - state_values.squeeze()

            if advantages.shape[0] > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
            else:
                advantages = advantages - advantages.mean()

            if torch.isnan(advantages).any():
                advantages = torch.nan_to_num(advantages, nan=0.0)

        return advantages

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), f"{path}/actor_critic.pth")
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pth")
        torch.save(self.scheduler.state_dict(), f"{path}/scheduler.pth")

    def load(self, path):
        self.actor_critic.load_state_dict(
            torch.load(
                f"{path}/actor_critic.pth", map_location=self.device, weights_only=True
            )
        )

        reset_optim = self.config.get("reset_optimizer_on_load", False)
        reset_sched = self.config.get("reset_lr_scheduler_on_load", True)

        if not reset_optim:
            self.optimizer.load_state_dict(
                torch.load(
                    f"{path}/optimizer.pth", map_location=self.device, weights_only=True
                )
            )

        if reset_sched or reset_optim:
            # Re-init scheduler so it starts fresh using the (possibly
            # freshly-initialised) optimizer.
            self._setup_lr_scheduler()
        else:
            self.scheduler.load_state_dict(
                torch.load(
                    f"{path}/scheduler.pth",
                    map_location=self.device,
                    weights_only=True,
                )
            )

    def step_scheduler(self):
        self.scheduler.step()
