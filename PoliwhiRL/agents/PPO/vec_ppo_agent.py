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
from PoliwhiRL.utils import plot_metrics, RewardScaler
from PoliwhiRL.agents.PPO._minibatch import run_ppo_epochs


class VecPPOAgent:
    def __init__(self, input_shape, action_size, config):
        self.config = config
        self.input_shape = tuple(input_shape)
        self.ram_obs_dim = int(config["ram_obs_dim"])
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
        self.minibatch_size = config.get("ppo_minibatch_size", None)
        # Recording: every record_frequency completed episodes (across all envs),
        # capture env 0's *next* episode end-to-end. Mirrors the single-env
        # behaviour but only records one env to keep disk usage sane.
        self.record_enabled = bool(config.get("record", False))
        self.record_frequency = int(config.get("record_frequency", 100))
        self._next_record_episode = max(1, self.record_frequency)
        # Multi-state pool config. Resolved against the vec env after the env
        # is constructed (the vec env owns the canonical state_paths list).
        self.state_cycle_strategy = config.get("state_cycle_strategy", "round_robin")
        # Action-replay pool. Same cycling pattern as state pool. -1 means
        # "no replay assigned to this env."
        self.action_replay_cycle_strategy = config.get(
            "action_replay_cycle_strategy", "round_robin"
        )
        # Cosine scheduler over rollouts (one scheduler.step per rollout).
        config["ppo_scheduler_t_max"] = self.num_rollouts

        self.model = PPOModel(self.input_shape, self.action_size, config)
        self.memory = VecPPOMemory(config, self.num_envs)
        self.reward_scaler = RewardScaler(gamma=self.gamma, num_envs=self.num_envs)

        # Populated when train_agent() builds the vec env.
        self.state_paths = None
        self.env_state_indices = None
        self.env_pending_state_indices = None
        self.action_replay_paths = None
        self.env_replay_indices = None
        self.env_pending_replay_indices = None
        # Track the highest single-episode reward seen across all envs and
        # the action sequence that produced it. Saved to Checkpoints/actions.steps
        # for the next curriculum stage's action_replay_paths.
        self.best_episode_reward = float("-inf")

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
            # Parallel to episode_rewards: state-pool index for each
            # completed episode. Allows downstream analysis to group
            # performance by starting save-state.
            "episode_state_indices": [],
            # Parallel to episode_rewards: action-replay pool index for
            # each completed episode (-1 when no replay was assigned).
            "episode_replay_indices": [],
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
        # Snapshot the env's canonical state-pool view so the agent can tag
        # per-episode metrics and drive cycling. `running_state_idx` is the
        # state actually in effect for env i's current episode; `pending`
        # is the state that will take effect on env i's NEXT auto-reset
        # (lag-by-one because the worker auto-resets before the agent can
        # process the done signal).
        self.state_paths = list(vec_env.state_paths)
        self.env_state_indices = list(vec_env.state_indices)
        self.env_pending_state_indices = list(vec_env.state_indices)
        self.action_replay_paths = list(vec_env.action_replay_paths)
        self.env_replay_indices = list(vec_env.replay_indices)
        self.env_pending_replay_indices = list(vec_env.replay_indices)
        # Per-env action history for "best path" capture. Reset on done.
        self._env_action_histories = [[] for _ in range(self.num_envs)]

        obs = vec_env.reset()  # {"image": (N, C, H, W), "ram": (N, D)}
        # Per-env state histories for the transformer input.
        state_seq = np.broadcast_to(
            obs["image"][:, None],
            (self.num_envs, self.sequence_length) + self.input_shape,
        ).copy()
        ram_seq = np.broadcast_to(
            obs["ram"][:, None],
            (self.num_envs, self.sequence_length, self.ram_obs_dim),
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
            self._collect_rollout(
                vec_env, state_seq, ram_seq, mems, ep_returns, ep_lengths
            )

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

    def _collect_rollout(
        self, vec_env, state_seq, ram_seq, mems, ep_returns, ep_lengths
    ):
        # state_seq, ram_seq, mems, ep_returns, ep_lengths mutate in place.
        for _ in range(self.rollout_length):
            state_tensor = torch.from_numpy(state_seq).float().to(self.device)
            ram_tensor = torch.from_numpy(ram_seq).float().to(self.device)
            with torch.no_grad():
                action_probs, _, new_mems = self.model.actor_critic(
                    state_tensor, ram_tensor, mems
                )
                action_probs = torch.clamp(action_probs, 1e-10, 1.0)
                actions_t = torch.multinomial(action_probs, 1).squeeze(1)
                log_probs_t = torch.log(
                    action_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1) + 1e-10
                )

            actions = actions_t.cpu().numpy().astype(np.int64)
            log_probs_np = log_probs_t.cpu().numpy().astype(np.float32)

            for i, a in enumerate(actions):
                self.episode_data["buttons_pressed"].append(int(a))
                self._env_action_histories[i].append(int(a))

            next_obs, rewards, dones = vec_env.step(actions)
            next_image, next_ram = next_obs["image"], next_obs["ram"]
            self.reward_scaler.observe(rewards, dones)

            states_now = state_seq[:, -1]
            ram_now = ram_seq[:, -1]
            self.memory.store_step(
                states=states_now,
                ram_states=ram_now,
                next_states=next_image,
                next_ram_states=next_ram,
                actions=actions,
                rewards=rewards,
                dones=dones,
                log_probs=log_probs_np,
                mems=mems,
            )

            ep_returns += rewards
            ep_lengths += 1

            for i in range(self.num_envs):
                if dones[i]:
                    # Capture the just-finished episode's action sequence
                    # before resetting the per-env history.
                    self._maybe_dump_best_actions(
                        float(ep_returns[i]), self._env_action_histories[i]
                    )
                    self._commit_episode(
                        env_idx=i,
                        reward_sum=float(ep_returns[i]),
                        length=int(ep_lengths[i]),
                    )
                    self._env_action_histories[i] = []
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0
                    # Refill both sequences with the post-reset obs.
                    state_seq[i] = np.broadcast_to(
                        next_image[i], (self.sequence_length,) + self.input_shape
                    ).copy()
                    ram_seq[i] = np.broadcast_to(
                        next_ram[i], (self.sequence_length, self.ram_obs_dim)
                    ).copy()
                    for layer in range(len(new_mems)):
                        new_mems[layer][i].zero_()
                    # The auto-reset that just happened in the worker used
                    # the state + replay that were *pending* before this done.
                    # Promote both to "running," then queue the next ones.
                    self.env_state_indices[i] = self.env_pending_state_indices[i]
                    self.env_replay_indices[i] = self.env_pending_replay_indices[i]
                    self._cycle_env_state(vec_env, i)
                    self._cycle_env_replay(vec_env, i)
                else:
                    state_seq[i, :-1] = state_seq[i, 1:]
                    state_seq[i, -1] = next_image[i]
                    ram_seq[i, :-1] = ram_seq[i, 1:]
                    ram_seq[i, -1] = next_ram[i]

            # Recording fires on env-0 dones, after the cycling above so the
            # folder naming reflects the just-finished episode.
            if dones[0]:
                self._maybe_enable_recording(vec_env)

            mems = new_mems

    def _cycle_env_state(self, vec_env, env_idx):
        """Pick the next state for env_idx according to state_cycle_strategy
        and send it to the worker. The worker's next auto-reset will use it.
        The choice is queued in env_pending_state_indices and promoted to
        env_state_indices when that auto-reset actually fires.
        """
        if len(self.state_paths) <= 1 or self.state_cycle_strategy == "none":
            return  # nothing to cycle
        if self.state_cycle_strategy == "round_robin":
            next_idx = (
                self.env_state_indices[env_idx] + 1
            ) % len(self.state_paths)
        elif self.state_cycle_strategy == "random":
            next_idx = int(np.random.randint(0, len(self.state_paths)))
        else:
            return
        try:
            vec_env.set_env_state_index(env_idx, next_idx)
        except Exception as e:
            print(f"[VecPPOAgent] Failed to cycle env {env_idx} to state {next_idx}: {e}")
            return
        self.env_pending_state_indices[env_idx] = next_idx

    def _cycle_env_replay(self, vec_env, env_idx):
        """Same lag-by-one pattern as state cycling, for the action-replay pool."""
        if (
            not self.action_replay_paths
            or len(self.action_replay_paths) <= 1
            or self.action_replay_cycle_strategy == "none"
        ):
            return
        cur = self.env_replay_indices[env_idx]
        if self.action_replay_cycle_strategy == "round_robin":
            next_idx = (cur + 1) % len(self.action_replay_paths)
        elif self.action_replay_cycle_strategy == "random":
            next_idx = int(np.random.randint(0, len(self.action_replay_paths)))
        else:
            return
        try:
            vec_env.set_env_replay_index(env_idx, next_idx)
        except Exception as e:
            print(
                f"[VecPPOAgent] Failed to cycle env {env_idx} to replay {next_idx}: {e}"
            )
            return
        self.env_pending_replay_indices[env_idx] = next_idx

    def _maybe_dump_best_actions(self, reward_sum, actions):
        if reward_sum <= self.best_episode_reward or not actions:
            return
        self.best_episode_reward = float(reward_sum)
        ckpt = self.config.get("checkpoint")
        if not ckpt:
            return
        try:
            os.makedirs(ckpt, exist_ok=True)
            path = os.path.join(ckpt, "actions.steps")
            with open(path, "w") as f:
                f.write(f"# best_episode_reward = {reward_sum}\n")
                for a in actions:
                    f.write(f"{int(a)}\n")
        except Exception as e:
            print(f"[VecPPOAgent] Failed to write best actions file: {e}")

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

    def _commit_episode(self, env_idx, reward_sum, length):
        self.episode += 1
        self.episode_data["episode_rewards"].append(reward_sum)
        self.episode_data["episode_lengths"].append(length)
        self.episode_data["episode_state_indices"].append(
            int(self.env_state_indices[env_idx])
        )
        self.episode_data["episode_replay_indices"].append(
            int(self.env_replay_indices[env_idx])
        )
        self.episode_data["moving_avg_reward"].append(reward_sum)
        self.episode_data["moving_avg_length"].append(length)
        self.episode_data["episode_entropies"].append(
            self.model._get_entropy_coef(self._stage_episode())
        )

    # ---------- update ----------

    def _update_from_rollout(self, data):
        # Per-env GAE/returns: reshape so the time axis is contiguous within
        # an env, then fold the env axis into the batch dim for the PPO loss.
        # Rewards are normalised by the running std of discounted returns;
        # critic values are already in the same normalised space because the
        # critic learns to predict normalised returns.
        rewards = data["rewards"] * float(self.reward_scaler.scale_factor())
        dones = data["dones"]                  # (W, N)
        states = data["states"]                # (W, N, seq_len, *input_shape)
        ram_states = data["ram_states"]        # (W, N, seq_len, ram_obs_dim)
        next_states = data["next_states"]
        next_ram_states = data["next_ram_states"]
        actions = data["actions"]              # (W, N)
        old_log_probs = data["old_log_probs"]  # (W, N)
        mems = data["mems"]                    # list of (W, N, mem_len, d_model)

        W, N = rewards.shape

        # Flatten (W*N, ...) for batched forward passes.
        flat_states = states.reshape(W * N, *states.shape[2:])
        flat_ram_states = ram_states.reshape(W * N, *ram_states.shape[2:])
        flat_next_states = next_states.reshape(W * N, *next_states.shape[2:])
        flat_next_ram_states = next_ram_states.reshape(
            W * N, *next_ram_states.shape[2:]
        )
        flat_mems = [m.reshape(W * N, *m.shape[2:]) for m in mems]

        with torch.no_grad():
            _, values_flat, _ = self.model.actor_critic(
                flat_states, flat_ram_states, flat_mems
            )
            values_flat = values_flat.squeeze(-1)  # (W*N,)
            values = values_flat.reshape(W, N)

            # Bootstrap V(s_{T+1}) per env from the last next_state sequence
            # (uses the most recent mems snapshot per env).
            tail_states = next_states[-1]                 # (N, seq_len, *input_shape)
            tail_ram = next_ram_states[-1]                # (N, seq_len, ram_obs_dim)
            tail_mems = [m[-1] for m in mems]             # list of (N, mem_len, d_model)
            _, tail_v, _ = self.model.actor_critic(tail_states, tail_ram, tail_mems)
            tail_values = tail_v.squeeze(-1)              # (N,)

        returns, advantages = self._per_env_gae(
            rewards, values, dones, tail_values
        )

        flat_actions = actions.reshape(W * N)
        flat_log_probs = old_log_probs.reshape(W * N)
        flat_returns = returns.reshape(W * N)
        flat_advantages = advantages.reshape(W * N)
        flat_old_values = values.reshape(W * N).detach()

        flat_data = {
            "states": flat_states,
            "ram_states": flat_ram_states,
            "next_states": flat_next_states,
            "next_ram_states": flat_next_ram_states,
            "actions": flat_actions,
            "rewards": rewards.reshape(W * N),
            "dones": dones.reshape(W * N),
            "old_log_probs": flat_log_probs,
            "mems": flat_mems,
            "returns": flat_returns,
            "advantages": flat_advantages,
            "old_values": flat_old_values,
        }

        return run_ppo_epochs(
            model=self.model,
            data=flat_data,
            episode=self._stage_episode(),
            epochs=self.epochs,
            minibatch_size=self.minibatch_size,
            target_kl=self.target_kl,
        )

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
            state_indices=self.episode_data.get("episode_state_indices", None),
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
            "reward_scaler": self.reward_scaler.state_dict(),
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

            scaler_state = info.get("reward_scaler")
            if scaler_state is not None:
                self.reward_scaler.load_state_dict(scaler_state)

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
                    "episode_state_indices": [],
                    "episode_replay_indices": [],
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
                "state_indices": len(
                    self.episode_data.get("episode_state_indices", [])
                ),
                "replay_indices": len(
                    self.episode_data.get("episode_replay_indices", [])
                ),
            }
        except FileNotFoundError:
            print(f"No checkpoint found at {path}, starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch.")
