# -*- coding: utf-8 -*-
"""Vectorised PPO agent.

Rollout-based training loop (in contrast to the episode-based single-env
PPOAgent). Each iteration:
  1. Collect T env steps across N envs in parallel.
  2. Compute per-env returns and advantages with V(s_{T+1}) bootstrap.
  3. Flatten (W, N, ...) -> (W*N, ...) and run PPO update with KL early-stop.

Episode-level metrics (reward sum, length) are tracked per env and committed
to the same data dicts as the single-env agent the moment an env finishes
an episode, so plotting/checkpoint code is shared.
"""
import os
from collections import deque
import numpy as np
import torch
from tqdm.auto import tqdm

from PoliwhiRL.environment import VecPyBoyEnv
from PoliwhiRL.replay import VecPPOMemory
from PoliwhiRL.models.PPO import PPOModel
from PoliwhiRL.utils.visuals import plot_metrics


class VecPPOAgent:
    def __init__(self, input_shape, action_size, config):
        self.config = config
        self.input_shape = tuple(input_shape)
        self.action_size = int(action_size)
        self.config["input_shape"] = self.input_shape
        self.config["action_size"] = self.action_size
        self.device = torch.device(config["device"])

        self.num_envs = int(config.get("num_envs", 1))
        self.rollout_length = int(config["ppo_update_frequency"])
        self.sequence_length = int(config["sequence_length"])
        self.epochs = int(config["ppo_epochs"])
        self.target_kl = config.get("ppo_target_kl", None)
        self.gamma = float(config["ppo_gamma"])
        self.gae_lambda = float(config.get("ppo_gae_lambda", 0.95))
        self.use_gae = self.gae_lambda > 0
        self.results_dir = config["results_dir"]
        self.checkpoint_frequency = int(config["checkpoint_frequency"])
        self.report_episode = config["report_episode"]
        self.n_goals = config["N_goals_target"]
        self.num_rollouts = int(config["num_rollouts"])
        # Recording: every record_frequency completed episodes (across all envs),
        # capture env 0's *next* episode end-to-end. Mirrors the single-env
        # behaviour but only records one env to keep disk usage sane.
        self.record_enabled = bool(config.get("record", False))
        self.record_frequency = int(config.get("record_frequency", 100))
        self._next_record_episode = max(1, self.record_frequency)
        # Cosine scheduler over rollouts (one scheduler.step per rollout).
        config["ppo_scheduler_t_max"] = self.num_rollouts

        self.model = PPOModel(self.input_shape, self.action_size, config)
        self.memory = VecPPOMemory(config, self.num_envs)

        self.best_reward = float("-inf")
        self.episode = int(config["start_episode"])  # completed-episode counter
        self.stage_start_episode = self.episode
        self.stage_data_offsets = None
        self.rollout_idx = 0

        self.reset_tracking()

    def _stage_episode(self):
        return self.episode - self.stage_start_episode

    def reset_tracking(self):
        self.episode_data = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_losses": [],
            "moving_avg_reward": deque(maxlen=100),
            "moving_avg_length": deque(maxlen=100),
            "moving_avg_loss": deque(maxlen=100),
            "buttons_pressed": deque(maxlen=1000),
            "episode_entropies": [],
        }
        self.episode_data["buttons_pressed"].append(0)

    # ---------- training loop ----------

    def train_agent(self):
        vec_env = VecPyBoyEnv(self.config, self.num_envs)
        try:
            self._train_loop(vec_env)
        finally:
            vec_env.close()

    def _train_loop(self, vec_env):
        obs = vec_env.reset()  # (N, *input_shape)
        # Per-env state history for transformer input: (N, seq_len, *input_shape)
        state_seq = np.broadcast_to(
            obs[:, None], (self.num_envs, self.sequence_length) + self.input_shape
        ).copy()
        # Per-env mems: list of (N, mem_len, d_model)
        mems = self.model.init_mems(batch_size=self.num_envs)

        # Per-env running episode stats (sum reward, step count) — committed to
        # episode_data only when an env finishes an episode.
        ep_returns = np.zeros(self.num_envs, dtype=np.float32)
        ep_lengths = np.zeros(self.num_envs, dtype=np.int64)

        pbar = (
            tqdm(range(self.num_rollouts), desc=f"VecPPO N={self.num_envs}")
            if self.report_episode
            else range(self.num_rollouts)
        )

        for self.rollout_idx in pbar:
            self.memory.reset()
            self._collect_rollout(vec_env, state_seq, mems, ep_returns, ep_lengths)
            # After collect, state_seq/mems/ep_* have been mutated in-place
            # to reflect the post-rollout state.

            data = self.memory.get_data()
            if data is not None:
                loss_val, epochs_run = self._update_from_rollout(data)
                self._record_loss(loss_val, epochs_run)

            self.model.step_scheduler()

            if self.report_episode and hasattr(pbar, "set_postfix"):
                self._update_progress_bar(pbar)

            if (self.rollout_idx + 1) % 10 == 0:
                self._plot_metrics()

            if (
                (self.rollout_idx + 1) % self.checkpoint_frequency == 0
                and self.config.get("save_checkpoint", True)
                and self.config.get("checkpoint") is not None
            ):
                self.save_model(self.config["checkpoint"])

        if (
            self.config.get("save_checkpoint", True)
            and self.config.get("checkpoint") is not None
        ):
            self.save_model(self.config["checkpoint"])

    def _collect_rollout(self, vec_env, state_seq, mems, ep_returns, ep_lengths):
        # state_seq, mems, ep_returns, ep_lengths are mutated in place.
        for _ in range(self.rollout_length):
            # Forward over all envs at once.
            state_tensor = torch.from_numpy(state_seq).float().to(self.device)
            with torch.no_grad():
                action_probs, _, new_mems = self.model.actor_critic(state_tensor, mems)
                action_probs = torch.clamp(action_probs, 1e-10, 1.0)
                actions_t = torch.multinomial(action_probs, 1).squeeze(1)
                log_probs_t = torch.log(
                    action_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1) + 1e-10
                )

            actions = actions_t.cpu().numpy().astype(np.int64)
            log_probs_np = log_probs_t.cpu().numpy().astype(np.float32)

            for a in actions:
                self.episode_data["buttons_pressed"].append(int(a))

            next_obs, rewards, dones = vec_env.step(actions)

            # Last frame of each env's current state_seq is "s_t" (the state
            # used to choose this action).
            states_now = state_seq[:, -1]
            self.memory.store_step(
                states=states_now,
                next_states=next_obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                log_probs=log_probs_np,
                mems=mems,
            )

            ep_returns += rewards
            ep_lengths += 1

            # For envs that just ended an episode: commit metrics, refill the
            # state sequence with the new (post-reset) obs, and zero mems.
            for i in range(self.num_envs):
                if dones[i]:
                    self._commit_episode(float(ep_returns[i]), int(ep_lengths[i]))
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0
                    state_seq[i] = np.broadcast_to(
                        next_obs[i], (self.sequence_length,) + self.input_shape
                    ).copy()
                    for layer in range(len(new_mems)):
                        new_mems[layer][i].zero_()
                else:
                    state_seq[i, :-1] = state_seq[i, 1:]
                    state_seq[i, -1] = next_obs[i]

            # If env 0 just auto-reset and we're due, enable recording on the
            # fresh episode it has just started. The worker's reset cleared
            # env.record, so a re-enable here applies cleanly to the new run.
            if dones[0]:
                self._maybe_enable_recording(vec_env)

            mems = new_mems

    def _maybe_enable_recording(self, vec_env):
        if not self.record_enabled or self.record_frequency <= 0:
            return
        if self.episode < self._next_record_episode:
            return
        folder = f"N_goals_{self.n_goals}/ep_{self.episode}"
        try:
            vec_env.enable_record(folder, use_episode_number=False, env_idx=0)
        except Exception as e:  # don't kill training over a recording hiccup
            print(f"[VecPPOAgent] Failed to enable recording: {e}")
            return
        self._next_record_episode = self.episode + self.record_frequency

    def _commit_episode(self, reward_sum, length):
        self.episode += 1
        self.episode_data["episode_rewards"].append(reward_sum)
        self.episode_data["episode_lengths"].append(length)
        self.episode_data["moving_avg_reward"].append(reward_sum)
        self.episode_data["moving_avg_length"].append(length)
        self.episode_data["episode_entropies"].append(
            self.model._get_entropy_coef(self._stage_episode())
        )

    # ---------- update ----------

    def _update_from_rollout(self, data):
        # Per-env GAE/returns: reshape so the time axis is contiguous within
        # an env, then fold the env axis into the batch dim for the PPO loss.
        rewards = data["rewards"]      # (W, N)
        dones = data["dones"]          # (W, N)
        states = data["states"]        # (W, N, seq_len, *input_shape)
        next_states = data["next_states"]  # (W, N, seq_len, *input_shape)
        actions = data["actions"]      # (W, N)
        old_log_probs = data["old_log_probs"]  # (W, N)
        mems = data["mems"]            # list of (W, N, mem_len, d_model)

        W, N = rewards.shape

        # Flatten (W*N, ...) for batched forward passes.
        flat_states = states.reshape(W * N, *states.shape[2:])
        flat_next_states = next_states.reshape(W * N, *next_states.shape[2:])
        flat_mems = [m.reshape(W * N, *m.shape[2:]) for m in mems]

        with torch.no_grad():
            _, values_flat, _ = self.model.actor_critic(flat_states, flat_mems)
            values_flat = values_flat.squeeze(-1)  # (W*N,)
            values = values_flat.reshape(W, N)

            # Bootstrap V(s_{T+1}) for each env using the last next_state's
            # sequence (uses the most recent mems snapshot per env).
            tail_states = next_states[-1]            # (N, seq_len, *input_shape)
            tail_mems = [m[-1] for m in mems]        # list of (N, mem_len, d_model)
            _, tail_v, _ = self.model.actor_critic(tail_states, tail_mems)
            tail_values = tail_v.squeeze(-1)         # (N,)

        returns, advantages = self._per_env_gae(
            rewards, values, dones, tail_values
        )

        # Flatten remaining tensors so the PPO loss sees a flat batch.
        flat_actions = actions.reshape(W * N)
        flat_log_probs = old_log_probs.reshape(W * N)
        flat_returns = returns.reshape(W * N)
        flat_advantages = advantages.reshape(W * N)
        flat_old_values = values.reshape(W * N).detach()

        flat_data = {
            "states": flat_states,
            "next_states": flat_next_states,
            "actions": flat_actions,
            "rewards": rewards.reshape(W * N),
            "dones": dones.reshape(W * N),
            "old_log_probs": flat_log_probs,
            "mems": flat_mems,
            "returns": flat_returns,
            "advantages": flat_advantages,
            "old_values": flat_old_values,
        }

        total_loss = 0.0
        epochs_run = 0
        for _ in range(self.epochs):
            loss, approx_kl = self.model.update(flat_data, self._stage_episode())
            total_loss += loss
            epochs_run += 1
            if self.target_kl is not None and approx_kl > self.target_kl:
                break
        return total_loss, epochs_run

    def _per_env_gae(self, rewards, values, dones, tail_values):
        """GAE along the time axis, independently per env.

        rewards, values, dones: (W, N) tensors. tail_values: (N,).
        Returns returns, advantages of shape (W, N).
        """
        W, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
        not_done = (~dones).to(rewards.dtype)

        if self.use_gae:
            for t in reversed(range(W)):
                next_value = values[t + 1] if t + 1 < W else tail_values
                delta = rewards[t] + self.gamma * next_value * not_done[t] - values[t]
                gae = delta + self.gamma * self.gae_lambda * not_done[t] * gae
                advantages[t] = gae
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards)
            running = tail_values.clone()
            for t in reversed(range(W)):
                running = rewards[t] + self.gamma * running * not_done[t]
                returns[t] = running
            advantages = returns - values
        return returns, advantages

    # ---------- metrics / IO ----------

    def _record_loss(self, total_loss, epochs_run):
        avg_loss = total_loss / max(1, epochs_run)
        self.episode_data["episode_losses"].append(avg_loss)
        self.episode_data["moving_avg_loss"].append(avg_loss)

    def _update_progress_bar(self, pbar):
        ma_r = self.episode_data["moving_avg_reward"]
        ma_l = self.episode_data["moving_avg_length"]
        pbar.set_postfix(
            {
                "ep": self.episode,
                "avg_r": f"{float(np.mean(ma_r)):.2f}" if ma_r else "n/a",
                "avg_len": f"{float(np.mean(ma_l)):.1f}" if ma_l else "n/a",
            }
        )

    def _plot_metrics(self):
        os.makedirs(self.results_dir, exist_ok=True)
        plot_metrics(
            self.episode_data["episode_rewards"],
            self.episode_data["episode_losses"],
            self.episode_data["episode_lengths"],
            self.episode_data["buttons_pressed"],
            self.n_goals,
            self.episode,
            save_loc=self.results_dir,
            entropies=self.episode_data.get("episode_entropies", None),
            stage_data_offsets=self.stage_data_offsets,
        )

    def save_model(self, path):
        path = f"{path}"
        os.makedirs(path, exist_ok=True)
        self.model.save(path)
        info = {
            "episode": self.episode,
            "best_reward": (
                max(self.episode_data["episode_rewards"])
                if self.episode_data["episode_rewards"]
                else float("-inf")
            ),
            "episode_data": self.episode_data,
        }
        torch.save(info, f"{path}/info.pth")

        ma_buf = self.episode_data["moving_avg_reward"]
        if len(ma_buf) >= ma_buf.maxlen:
            current_ma = float(np.mean(ma_buf))
            if current_ma > self.best_reward:
                self.best_reward = current_ma
                best_path = os.path.join(path, "best")
                os.makedirs(best_path, exist_ok=True)
                self.model.save(best_path)
                torch.save(info, f"{best_path}/info.pth")

    def load_model(self, path):
        try:
            self.model.load(f"{path}")
            torch.serialization.add_safe_globals(["numpy", "np"])
            info = torch.load(
                f"{path}/info.pth", map_location=self.device, weights_only=False
            )
            self.config["start_episode"] = info["episode"]
            self.episode = info["episode"]
            self.stage_start_episode = self.episode
            print(f"Loaded checkpoint from {path}, episode {self.episode}")

            loaded_episode_data = info.get("episode_data", {})
            if loaded_episode_data:
                fresh = {
                    "episode_rewards": [],
                    "episode_lengths": [],
                    "episode_losses": [],
                    "moving_avg_reward": deque(maxlen=100),
                    "moving_avg_length": deque(maxlen=100),
                    "moving_avg_loss": deque(maxlen=100),
                    "buttons_pressed": deque(maxlen=1000),
                    "episode_entropies": [],
                }
                for key, value in loaded_episode_data.items():
                    if key in fresh:
                        if isinstance(fresh[key], deque) and not isinstance(value, deque):
                            fresh[key] = deque(value, maxlen=100)
                        else:
                            fresh[key] = value
                self.episode_data = fresh
                if len(self.episode_data["buttons_pressed"]) == 0:
                    self.episode_data["buttons_pressed"].append(0)

            self.stage_data_offsets = {
                "rewards": len(self.episode_data["episode_rewards"]),
                "losses": len(self.episode_data["episode_losses"]),
                "steps": len(self.episode_data["episode_lengths"]),
                "entropies": len(self.episode_data["episode_entropies"]),
            }
        except FileNotFoundError:
            print(f"No checkpoint found at {path}, starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch.")
