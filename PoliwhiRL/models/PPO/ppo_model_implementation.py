# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from PoliwhiRL.models.PPO.PPOTransformer import PPOTransformer
from PoliwhiRL.environment.action_mask import compute_action_mask


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
        # When set (by the agent's adaptive-entropy controller), this scalar
        # overrides the time-based schedule entirely: entropy becomes a closed
        # loop on training progress rather than a hand-tuned curve. ``None``
        # keeps the legacy schedule behaviour.
        self._adaptive_entropy_coef = None
        self.clip_value_loss = self.config.get("ppo_clip_value_loss", True)
        # Phase-1 action mask. Default on. Per-stage opt-in to also allow
        # start/select while walking (for stages where menus matter).
        self.action_mask_enabled = bool(self.config.get("action_mask_enabled", True))
        self.allow_menus_walking = bool(self.config.get("allow_menus_walking", False))

        self._initialize_networks()
        self._initialize_optimizers()

    def _action_mask_for(self, ram_sequence):
        """Build the (B, action_size) mask for the *last* frame of each
        sequence in the batch. Returns None when masking is disabled, in
        which case the model's forward stays mask-free.
        """
        if not self.action_mask_enabled:
            return None
        # ram_sequence: (B, seq_len, ram_dim). The mask is per-frame and
        # we only sample / evaluate the most recent frame.
        return compute_action_mask(
            ram_sequence[:, -1, :], allow_menus_walking=self.allow_menus_walking
        )

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
        # ``constant`` keeps LR flat — for free-play stages where a cosine
        # decay to lr_min would freeze the policy long before the budget ends.
        # ``cosine``/``cosine_floor`` both anneal to ``ppo_lr_min`` (set a
        # higher floor via ppo_lr_min for navigation stages so late updates
        # still move).
        schedule = self.config.get("ppo_lr_schedule", "cosine")
        if schedule == "constant":
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=t_max, eta_min=eta_min
            )

    def init_mems(self, batch_size=1):
        return self.actor_critic.init_mems(batch_size, self.device)

    def get_action(self, state_sequence, ram_sequence, mems=None):
        state_sequence = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        ram_sequence = torch.FloatTensor(ram_sequence).unsqueeze(0).to(self.device)
        action_mask = self._action_mask_for(ram_sequence)
        with torch.no_grad():
            action_probs, _, new_mems = self.actor_critic(
                state_sequence, ram_sequence, mems, action_mask=action_mask
            )
        action_probs = torch.clamp(
            torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0),
            1e-10, 1.0,
        )
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs[0, action] + 1e-10).item()
        return action, log_prob, new_mems

    def compute_log_prob(self, state_sequence, ram_sequence, action, mems=None):
        state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        ram_tensor = torch.FloatTensor(ram_sequence).unsqueeze(0).to(self.device)
        action_mask = self._action_mask_for(ram_tensor)
        with torch.no_grad():
            action_probs, _, _ = self.actor_critic(
                state_tensor, ram_tensor, mems, action_mask=action_mask
            )
        return torch.log(action_probs[0, action] + 1e-10).item()

    def update(self, data, step):
        actor_loss, critic_loss, entropy_loss, approx_kl = self._compute_ppo_losses(
            data, step
        )
        loss = actor_loss + critic_loss + entropy_loss
        self._update_networks(loss)
        return loss.item(), approx_kl

    def _get_entropy_coef(self, step):
        # Linear decay from initial to min over the planned budget. `step` is
        # a training-progress counter owned by the agent — rollout index in
        # vec mode, episode index in single-env mode (where one outer-loop
        # iteration runs one episode and matches one PPO update). Indexing by
        # rollouts (not raw episodes) keeps the schedule's effective length
        # aligned with `num_rollouts` regardless of how many envs the agent
        # is running.
        # Free-play / open-ended stages can disable annealing entirely so the
        # policy keeps a high exploration floor across thousands of episodes
        # instead of freezing onto an early local optimum.
        # Adaptive controller (closed loop on progress) takes precedence over
        # the time-based schedule when the agent has set it. getattr keeps
        # this robust to stubs / checkpoints predating the controller.
        adaptive = getattr(self, "_adaptive_entropy_coef", None)
        if adaptive is not None:
            return adaptive
        if not self.config.get("ppo_entropy_anneal_enabled", True):
            return self.entropy_coef
        total = self.config.get("ppo_entropy_anneal_steps",
                                self.config.get("num_rollouts", 1))
        effective = max(0, step - self._entropy_reset_offset)
        progress = min(effective / max(total, 1), 1.0)
        return self.entropy_coef * (1 - progress) + self.entropy_min * progress

    def set_entropy_offset(self, offset):
        """Rewind the entropy schedule by setting a step offset.

        The effective step used for decay becomes (step - offset), so a
        larger offset means the schedule is further back and entropy is
        higher. Plateau detection fires this in the same units the caller
        will use for `_get_entropy_coef` (rollout idx for vec, episode idx
        for single-env).
        """
        self._entropy_reset_offset = offset

    def set_entropy_coef(self, value):
        """Directly set the entropy coefficient (adaptive controller path).

        Overrides the time-based schedule in ``_get_entropy_coef``. ``None``
        restores schedule behaviour.
        """
        self._adaptive_entropy_coef = None if value is None else float(value)

    def _compute_ppo_losses(self, data, step):
        use_gae = self.config.get("ppo_gae_lambda", 0) > 0
        mems = data.get("mems", None)

        # Per-minibatch advantage normalisation is opt-in. In sparse-reward
        # regimes (Phase 4 navigation stages), normalising per minibatch
        # makes the rare positive-advantage transitions get pushed down
        # toward the bulk of zero-reward steps. The default "rollout" mode
        # normalises once over the full rollout in the agent layer and
        # skips renormalisation here.
        norm_mode = self.config.get("advantage_normalisation", "rollout")

        # Vec agent precomputes per-env GAE before flattening across envs;
        # accept those directly so we don't mistakenly recompute advantages
        # across env boundaries.
        if "returns" in data and "advantages" in data:
            returns = data["returns"]
            advantages = data["advantages"]
            if norm_mode == "minibatch" and advantages.shape[0] > 1:
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
                    # Value-only call — mask is irrelevant to the critic
                    # head but we pass it for consistency with the actor
                    # branch and to keep behaviour identical across calls.
                    _, values, _ = self.actor_critic(
                        data["states"], data["ram_states"], mems,
                        action_mask=self._action_mask_for(data["ram_states"]),
                    )
                    values = values.squeeze()

                returns, advantages = self._compute_gae(
                    data["rewards"], values, data["dones"],
                    last_value=last_value, truncated=data.get("truncated"),
                )
                if norm_mode == "minibatch" and advantages.shape[0] > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
            else:
                returns = self._compute_returns(
                    data["rewards"], data["dones"],
                    last_value=last_value, truncated=data.get("truncated"),
                )
                advantages = self._compute_advantages(
                    data["states"], data["ram_states"], returns, mems
                )

        # Critical: the mask used here MUST match the one used at action
        # sampling time, otherwise new_log_probs will diverge from
        # old_log_probs in PPO's ratio test and the gradient estimator
        # breaks. The mask is a deterministic function of the stored
        # ram_states, so reconstructing it here gives an identical result.
        update_mask = self._action_mask_for(data["ram_states"])
        new_probs, new_values, _ = self.actor_critic(
            data["states"], data["ram_states"], mems, action_mask=update_mask,
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
        entropy_loss = -self._get_entropy_coef(step) * entropy

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
        # Skip the entire step if the loss is already non-finite — backprop
        # would produce NaN/inf gradients and poison every weight.
        if not torch.isfinite(ppo_loss):
            print("[PPOModel] Skipping update: non-finite loss.")
            return False

        self.optimizer.zero_grad()
        ppo_loss.backward()
        max_grad_norm = self.config.get("ppo_max_grad_norm", 0.5)
        # clip_grad_norm_ returns the *pre-clip* total norm. It does NOT guard
        # against inf/nan grads — when a grad is inf the clip coefficient is
        # max_norm/inf=0 and 0*inf=NaN, silently converting an inf gradient
        # into NaN weights on the next step. So we check the returned norm and
        # skip stepping when it is non-finite: one diverging update is dropped
        # instead of permanently poisoning the network (which then crashes
        # multinomial with "probability tensor contains nan").
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), max_norm=max_grad_norm
        )
        if not torch.isfinite(total_norm):
            print(
                f"[PPOModel] Skipping optimizer step: non-finite grad norm "
                f"({total_norm.item()})."
            )
            self.optimizer.zero_grad(set_to_none=True)
            return False

        self.optimizer.step()
        return True

    def _tail_bootstrap_value(self, data, mems):
        # Returns V(s_{T+1}) for the last transition in the rollout, or None
        # when it ended at a true terminal (bootstrap is 0 there). A done
        # that was a *truncation* (budget cut-off) still bootstraps, so a
        # terminal is only ``done and not truncated``.
        next_states = data.get("next_states", None)
        next_ram_states = data.get("next_ram_states", None)
        dones = data["dones"]
        truncated = data.get("truncated")
        if next_states is None or next_ram_states is None or len(dones) == 0:
            return None
        last_done = bool(dones[-1].item())
        last_trunc = (
            truncated is not None
            and len(truncated) > 0
            and bool(truncated[-1].item())
        )
        if last_done and not last_trunc:
            return None
        tail_input = next_states[-1:].detach()
        tail_ram = next_ram_states[-1:].detach()
        tail_mems = None
        if mems is not None:
            tail_mems = [m[-1:].detach() for m in mems]
        with torch.no_grad():
            _, tail_v, _ = self.actor_critic(
                tail_input, tail_ram, tail_mems,
                action_mask=self._action_mask_for(tail_ram),
            )
        return tail_v.squeeze().detach()

    def _compute_returns(self, rewards, dones, last_value=None, truncated=None):
        returns = torch.zeros_like(rewards)
        running_return = 0.0 if last_value is None else float(last_value)
        for t in reversed(range(len(rewards))):
            if bool(dones[t].item()):
                # Boundary: bootstrap only on truncation, else zero the
                # future. (Single-env rollouts only ever carry a done at
                # the final step, whose bootstrap is folded into last_value;
                # this branch keeps the general case correct.)
                is_trunc = truncated is not None and bool(truncated[t].item())
                running_return = float(last_value) if (is_trunc and last_value is not None) else 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        return returns

    def _compute_gae(self, rewards, values, dones, last_value=None, truncated=None):
        """GAE with truncation-aware bootstrap.

        At an episode boundary V(s_{T+1}) is bootstrapped only when the
        episode was *truncated* (budget cut-off); a natural terminal (goal
        complete) has no continuation and is zeroed. The GAE recurrence
        resets at every boundary via ``(~dones[t]) * gae`` regardless. When
        ``truncated`` is None every done is treated as a terminal. See the
        matching note in ``vec_ppo_agent._per_env_gae`` for the bias caveat.
        """
        gae_lambda = self.config.get("ppo_gae_lambda", 0.95)
        advantages = torch.zeros_like(rewards)
        gae = 0
        tail_value = 0.0 if last_value is None else float(last_value)
        not_done = (~dones).to(rewards.dtype)
        if truncated is None:
            bootstrap = not_done
        else:
            bootstrap = torch.clamp(not_done + truncated.to(rewards.dtype), max=1.0)

        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(rewards) else tail_value
            delta = rewards[t] + self.gamma * next_value * bootstrap[t] - values[t]
            gae = delta + self.gamma * gae_lambda * not_done[t] * gae
            advantages[t] = gae

        returns = advantages + values
        return returns, advantages

    def _compute_advantages(self, states, ram_states, returns, mems=None):
        with torch.no_grad():
            _, state_values, _ = self.actor_critic(
                states, ram_states, mems,
                action_mask=self._action_mask_for(ram_states),
            )
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
