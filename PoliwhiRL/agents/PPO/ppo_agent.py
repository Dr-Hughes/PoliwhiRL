# -*- coding: utf-8 -*-
import os
import random
import shutil
from collections import deque
import numpy as np
from tqdm.auto import tqdm
import torch

from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.environment.vec_env import _load_replay_pool, write_actions_file
from PoliwhiRL.utils import plot_metrics, RewardScaler
from PoliwhiRL.replay import PPOMemory
from PoliwhiRL.models.PPO import PPOModel
from PoliwhiRL.agents.PPO._minibatch import run_ppo_epochs


class PPOAgent:
    def __init__(self, input_shape, action_size, config):
        self.config = config
        self.input_shape = input_shape
        self.action_size = action_size
        self.config["input_shape"] = input_shape
        self.config["action_size"] = action_size
        self.device = config["device"]
        self.update_parameters_from_config()
        self.best_reward = float("-inf")
        self.model = PPOModel(input_shape, action_size, config)
        self.memory = PPOMemory(config)
        # Running variance of discounted returns; rewards are divided by
        # sqrt(var) before GAE so the critic regresses onto a normalised
        # target whose scale is roughly stable across stages.
        self.reward_scaler = RewardScaler(
            gamma=float(self.config["ppo_gamma"]), num_envs=1
        )
        # Action-replay pool: flat list of trajectories pooled across every
        # configured .steps file. On each reset we sample one trajectory
        # uniformly and a cutoff uniformly in [0, len(trajectory)].
        _, self._replay_pool = _load_replay_pool(
            self.config.get("action_replay_paths") or []
        )
        # Buffer of completed-episode trajectories captured since the last
        # checkpoint. At checkpoint time these are written to actions.steps
        # so the next stage / next training run can replay them.
        self._checkpoint_trajectories = []
        self.reset_tracking()
        # Stage-relative episode counter for entropy schedule. Updated by
        # load_model when resuming so each curriculum stage decays from fresh.
        self.stage_start_episode = self.episode
        # Per-list lengths at the start of the current stage. None on a fresh
        # run; populated by load_model so plot_metrics can render a current-
        # stage-only view alongside the all-data view.
        self.stage_data_offsets = None

    def _stage_episode(self):
        return self.episode - self.stage_start_episode

    def update_parameters_from_config(self):
        self.episode = self.config["start_episode"]
        self.record = self.config["record"]
        # In single-env mode one outer-loop iteration runs one episode, so
        # num_rollouts here is functionally "number of episodes to run."
        self.num_rollouts = self.config["num_rollouts"]
        self.episode_length = self.config["episode_length"]
        self.sequence_length = self.config["sequence_length"]
        self.n_goals = self.config["N_goals_target"]
        self.record_frequency = self.config["record_frequency"]
        self.results_dir = self.config["results_dir"]
        self.export_state_loc = self.config["export_state_loc"]
        self.checkpoint_frequency = self.config["checkpoint_frequency"]
        self.steps = 0
        self.report_episode = self.config["report_episode"]
        self.update_frequency = self.config["ppo_update_frequency"]
        self.epochs = self.config["ppo_epochs"]
        # None disables adaptive early-stop; numeric value is the per-update
        # approx-KL ceiling above which we abort remaining epochs.
        self.target_kl = self.config.get("ppo_target_kl", None)
        self.minibatch_size = self.config.get("ppo_minibatch_size", None)

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
            # Curriculum-progress metrics. With uniform-cutoff replay the
            # starting N_goals varies per episode, so we record both the
            # absolute total at episode end and the delta the training
            # portion contributed.
            "episode_goals_total": [],
            "episode_goals_made": [],
            "episode_goals_target": [],
        }
        self.episode_data["buttons_pressed"].append(0)
        # Tracks the last episode at which entropy was reset via plateau
        # detection, so we can debounce subsequent resets.
        self._entropy_last_reset_ep = 0

    def _check_entropy_plateau(self):
        """Detect training plateaus and reset the entropy schedule.

        If a rolling window of recent episodes shows flat rewards and no
        new goals achieved (while still below target), rewind the entropy
        decay offset to inject exploration pressure. Uses percentage-based
        thresholds so it scales with total training budget.
        """
        if not self.config.get("entropy_plateau_reset", True):
            return

        total_budget = self.num_rollouts
        window_size = max(
            50,
            int(total_budget * self.config.get("entropy_reset_window_fraction", 0.1)),
        )
        min_eps = int(
            total_budget * self.config.get("entropy_reset_min_fraction", 0.25)
        )
        debounce_eps = int(
            total_budget * self.config.get("entropy_reset_debounce_fraction", 0.125)
        )

        rewards = self.episode_data["episode_rewards"]
        goals_total = self.episode_data["episode_goals_total"]

        # Need enough data: minimum budget elapsed + full window available
        if self.episode < min_eps or len(rewards) < window_size:
            return

        # Debounce: don't reset more often than debounce_interval
        if self.episode - self._entropy_last_reset_ep < debounce_eps:
            return

        recent_rewards = rewards[-window_size:]
        recent_goals = goals_total[-window_size:]

        # Flat reward: coefficient of variation below 0.15
        mean_r = float(np.mean(recent_rewards))
        std_r = float(np.std(recent_rewards))
        if abs(mean_r) > 1.0:
            reward_flat = (std_r / abs(mean_r)) < 0.15
        else:
            reward_flat = True  # near-zero rewards are always "flat"

        cv = (std_r / abs(mean_r)) if abs(mean_r) > 1.0 else float("inf")

        # No new goals in window
        goals_flat = max(recent_goals) == min(recent_goals)

        # Only reset if still below target (avoid wasting budget on solved stages)
        max_goals_in_window = max(recent_goals)
        if max_goals_in_window >= self.n_goals:
            return

        if reward_flat and goals_flat:
            rewind_by = int(
                window_size * self.config.get("entropy_reset_rewind_fraction", 0.1)
            )
            rewind_by = max(50, rewind_by)
            new_offset = max(0, self.episode - rewind_by)
            self.model.set_entropy_offset(new_offset)
            self._entropy_last_reset_ep = self.episode
            print(
                f"[PPOAgent] Entropy plateau reset at ep {self.episode}: "
                f"rewound offset to {new_offset} "
                f"(reward CV={cv:.3f}, goals stuck at {max_goals_in_window}/{self.n_goals})"
            )

    def train_agent(self):
        if self.report_episode:
            pbar = tqdm(
                range(self.num_rollouts),
                desc=f"Training (Goals: {self.n_goals})",
                leave=True,
            )
        else:
            pbar = range(self.num_rollouts)

        for episode_idx in pbar:
            record_loc = (
                f"N_goals_{self.n_goals}/{self.episode}"
                if (self.episode % self.record_frequency == 0 and self.record)
                else None
            )

            self.run_episode(record_loc=record_loc)
            self.episode += 1

            if len(self.memory) > self.sequence_length:
                self.update_model()

            if self.report_episode:
                self._update_progress_bar(pbar)

            if (
                self.episode % 10 == 0 and self.episode > 1
            ) or self.episode == self.num_rollouts:
                self._plot_metrics()

            self.model.step_scheduler()

            if (
                self.episode % self.checkpoint_frequency == 0
                and self.config.get("save_checkpoint", True)
                and self.config.get("checkpoint") is not None
            ):
                self.save_model(self.config["checkpoint"])

        if (
            self.config.get("save_checkpoint", True)
            and self.config.get("checkpoint") is not None
        ):
            self.save_model(self.config["checkpoint"])

        if self.config.get("save_training_states", False):
            self.run_episode(
                save_path=f"{self.export_state_loc}/N_goals_{self.n_goals}.pkl"
            )
        else:
            self.run_episode(save_path=None)

    def run_episode(self, record_loc=None, save_path=None):
        env = Env(self.config)

        try:
            self.steps = 0
            env.reset()
            # Replay walks the env (and Rewards) forward to a curriculum-
            # aligned start. The replay transitions are not stored in memory
            # or counted toward the episode reward — only the training
            # episode that follows is. Trajectory + cutoff are uniformly
            # sampled per episode so the agent sees a wide distribution of
            # starting states rather than always taking over at the same
            # post-replay endpoint.
            if self._replay_pool:
                traj = self._replay_pool[random.randrange(len(self._replay_pool))]
                k = random.randint(0, len(traj))
                if k > 0:
                    obs = env.replay_actions(traj[:k])
                else:
                    obs = env.get_observation()
            else:
                obs = env.get_observation()
            state, ram = obs["image"], obs["ram"]
            self.memory.reset()
            reward_sum = 0
            # Curriculum-progress snapshot at the start of the training
            # portion (after replay walked Rewards forward).
            goals_at_start = int(env.reward_calculator.N_goals)
            n_goals_target = int(env.reward_calculator.N_goals_target)
            current_episode_actions = []
            # Parallel sliding windows for the dual-input model.
            state_sequence = deque(
                [state] * self.sequence_length, maxlen=self.sequence_length
            )
            ram_sequence = deque(
                [ram] * self.sequence_length, maxlen=self.sequence_length
            )
            # Per-trajectory transformer memory: starts fresh each episode and
            # carries across rollout steps. Each transition stores the memory
            # state used to select its action so update() can reproduce the
            # same context.
            mems = self.model.init_mems(batch_size=1)

            if record_loc is not None:
                env.enable_record(record_loc, False)

            iter_range = (
                tqdm(
                    range(self.config["episode_length"]),
                    desc="Episode steps",
                    leave=False,
                )
                if self.report_episode
                else range(self.episode_length)
            )

            for _ in iter_range:
                self.steps += 1
                state_seq_arr = np.array(state_sequence)
                ram_seq_arr = np.array(ram_sequence)
                action, log_prob, new_mems = self.model.get_action(
                    state_seq_arr, ram_seq_arr, mems
                )
                self.episode_data["buttons_pressed"].append(action)
                current_episode_actions.append(int(action))

                next_obs, reward, done, _ = env.step(action)
                next_state, next_ram = next_obs["image"], next_obs["ram"]
                reward_sum += reward
                self.reward_scaler.observe(reward, done)

                self.memory.store_transition(
                    state,
                    ram,
                    next_state,
                    next_ram,
                    action,
                    reward,
                    done,
                    log_prob,
                    mems,
                )

                mems = new_mems
                state = next_state
                ram = next_ram
                state_sequence.append(state)
                ram_sequence.append(ram)

                if done:
                    break

                if (
                    self.steps % self.update_frequency == 0
                    and len(self.memory) > self.sequence_length
                ):
                    self.update_model()

            goals_total = int(env.reward_calculator.N_goals)
            goals_made = goals_total - goals_at_start
            self._update_episode_stats(
                reward_sum, goals_total, goals_made, n_goals_target
            )
            self._check_entropy_plateau()
            self._record_completed_trajectory(current_episode_actions)

            if save_path is not None:
                env.save_gym_state(save_path)
        finally:
            env.close()

    def _record_completed_trajectory(self, actions):
        """Buffer the just-finished training trajectory for the next
        checkpoint write. Single-env: one trajectory per completed
        episode; the checkpoint dumps however many accumulated since
        the previous checkpoint and clears the buffer.
        """
        if actions:
            self._checkpoint_trajectories.append(list(actions))

    def _write_checkpoint_actions(self, ckpt_dir):
        """Dump the per-checkpoint trajectory buffer to actions.steps.

        Uses the multi-trajectory `.steps` format (one block per
        trajectory). If no trajectories accumulated since the previous
        write, leaves any existing file untouched.
        """
        if not ckpt_dir:
            return None
        trajectories = self._checkpoint_trajectories
        if not trajectories:
            return None
        path = os.path.join(ckpt_dir, "actions.steps")
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            metadata = [{"length": len(t)} for t in trajectories]
            write_actions_file(path, trajectories, metadata=metadata)
            self._checkpoint_trajectories = []
            return path
        except Exception as e:
            print(f"[PPOAgent] Failed to write actions.steps: {e}")
            return None

    def _update_episode_stats(
        self, total_reward, goals_total, goals_made, n_goals_target
    ):
        self.episode_data["episode_rewards"].append(total_reward)
        self.episode_data["episode_lengths"].append(self.steps)
        self.episode_data["moving_avg_reward"].append(total_reward)
        self.episode_data["moving_avg_length"].append(self.steps)
        self.episode_data["episode_goals_total"].append(int(goals_total))
        self.episode_data["episode_goals_made"].append(int(goals_made))
        self.episode_data["episode_goals_target"].append(int(n_goals_target))

        current_entropy = self.model._get_entropy_coef(self._stage_episode())
        self.episode_data["episode_entropies"].append(current_entropy)

    def update_model(self, data=None):
        if data is None:
            data = self.memory.get_all_data()
        if data is None:
            return

        # Reward normalisation: scale rewards by the inverse running std of
        # discounted returns. The critic naturally settles into predicting
        # values in this normalised space; V(s_{T+1}) bootstrap stays
        # consistent because it comes from the same network.
        data["rewards"] = data["rewards"] * float(self.reward_scaler.scale_factor())

        self._precompute_targets(data)

        total_loss, epochs_run = run_ppo_epochs(
            model=self.model,
            data=data,
            episode=self._stage_episode(),
            epochs=self.epochs,
            minibatch_size=self.minibatch_size,
            target_kl=self.target_kl,
        )

        self._update_loss_stats(total_loss, epochs_run)
        self.memory.reset()

    def _precompute_targets(self, data):
        """Run V(s_t), V(s_{T+1}) tail bootstrap, and GAE once for the whole
        rollout so subsequent minibatched updates can shuffle freely without
        breaking the time-ordered return computation."""
        mems = data.get("mems", None)
        with torch.no_grad():
            _, values, _ = self.model.actor_critic(
                data["states"], data["ram_states"], mems
            )
            values = values.squeeze()
            if values.dim() == 0:
                values = values.unsqueeze(0)

            # Only bootstrap when the final transition is genuinely truncated
            # (not terminal). For a terminal tail, V(s_{T+1}) = 0 regardless.
            last_value = None
            dones = data["dones"]
            if dones.numel() > 0 and not bool(dones[-1].item()):
                tail_input = data["next_states"][-1:].detach()
                tail_ram = data["next_ram_states"][-1:].detach()
                tail_mems = (
                    [m[-1:].detach() for m in mems] if mems is not None else None
                )
                _, tail_v, _ = self.model.actor_critic(tail_input, tail_ram, tail_mems)
                last_value = tail_v.squeeze().detach()

        use_gae = self.config.get("ppo_gae_lambda", 0) > 0
        if use_gae:
            returns, advantages = self.model._compute_gae(
                data["rewards"], values, data["dones"], last_value=last_value
            )
        else:
            returns = self.model._compute_returns(
                data["rewards"], data["dones"], last_value=last_value
            )
            advantages = returns - values

        data["returns"] = returns
        data["advantages"] = advantages
        data["old_values"] = values.detach()

    def _update_loss_stats(self, total_loss, epochs_run):
        steps_since_update = max(1, len(self.memory))
        denom = max(1, epochs_run) * steps_since_update
        avg_loss = total_loss / denom
        self.episode_data["episode_losses"].append(avg_loss)
        self.episode_data["moving_avg_loss"].append(avg_loss)

    def _update_progress_bar(self, pbar):
        avg_reward = (
            np.mean(self.episode_data["moving_avg_reward"])
            if self.episode_data["moving_avg_reward"]
            else 0
        )
        avg_length = (
            np.mean(self.episode_data["moving_avg_length"])
            if self.episode_data["moving_avg_length"]
            else 0
        )

        current_reward = (
            self.episode_data["episode_rewards"][-1]
            if self.episode_data["episode_rewards"]
            else 0
        )
        current_length = (
            self.episode_data["episode_lengths"][-1]
            if self.episode_data["episode_lengths"]
            else 0
        )

        pbar.set_postfix(
            {
                "Avg Reward": f"{avg_reward:.2f}",
                "Avg Length": f"{avg_length:.2f}",
                "Reward": f"{current_reward:.2f}",
                "Length": f"{current_length}",
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
            goals_total=self.episode_data.get("episode_goals_total", None),
            goals_made=self.episode_data.get("episode_goals_made", None),
            goals_target=self.episode_data.get("episode_goals_target", None),
        )

    def save_model(self, path):
        path = f"{path}"
        os.makedirs(path, exist_ok=True)

        self.model.save(path)

        # Flush the per-checkpoint trajectory buffer to actions.steps.
        self._write_checkpoint_actions(path)

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

        # Snapshot a separate "best" checkpoint when the 100-episode rolling
        # mean reward improves. Gated on a full window so we don't lock in
        # an early-luck spike. Latest checkpoint above is still overwritten
        # every cycle for vanilla resume; this is purely additive.
        ma_buf = self.episode_data["moving_avg_reward"]
        if len(ma_buf) >= ma_buf.maxlen:
            current_ma = float(np.mean(ma_buf))
            if current_ma > self.best_reward:
                self.best_reward = current_ma
                best_path = os.path.join(path, "best")
                os.makedirs(best_path, exist_ok=True)
                self.model.save(best_path)
                torch.save(info, f"{best_path}/info.pth")
                # Copy actions.steps so the next curriculum stage can load
                # weights and replay from the same run.
                src = os.path.join(path, "actions.steps")
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(best_path, "actions.steps"))

    def load_model(self, path):
        try:
            self.model.load(f"{path}")
            torch.serialization.add_safe_globals(["numpy", "np"])
            info = torch.load(
                f"{path}/info.pth", map_location=self.device, weights_only=False
            )
            self.config["start_episode"] = info["episode"]
            self.episode = info["episode"]
            # Reset stage-relative counter so entropy schedule starts fresh
            # for this curriculum stage.
            self.stage_start_episode = self.episode
            # Clear plateau-detection state so the new stage starts with a
            # clean entropy schedule (no stale offset from the previous run).
            self._entropy_last_reset_ep = self.episode
            self.model.set_entropy_offset(0)
            print(f"Loaded checkpoint from {path}, episode {self.episode}")

            scaler_state = info.get("reward_scaler")
            if scaler_state is not None:
                self.reward_scaler.load_state_dict(scaler_state)

            loaded_episode_data = info.get("episode_data", {})
            if loaded_episode_data:
                fresh_episode_data = {
                    "episode_rewards": [],
                    "episode_lengths": [],
                    "episode_losses": [],
                    "moving_avg_reward": deque(maxlen=100),
                    "moving_avg_length": deque(maxlen=100),
                    "moving_avg_loss": deque(maxlen=100),
                    "buttons_pressed": deque(maxlen=1000),
                    "episode_entropies": [],
                    "episode_goals_total": [],
                    "episode_goals_made": [],
                    "episode_goals_target": [],
                }
                for key, value in loaded_episode_data.items():
                    if key in fresh_episode_data:
                        if isinstance(
                            fresh_episode_data[key], deque
                        ) and not isinstance(value, deque):
                            fresh_episode_data[key] = deque(value, maxlen=100)
                        else:
                            fresh_episode_data[key] = value
                self.episode_data = fresh_episode_data
                if len(self.episode_data["buttons_pressed"]) == 0:
                    self.episode_data["buttons_pressed"].append(0)

            # Snapshot list lengths so plot_metrics can slice out current-stage
            # data and render a second "current stage only" plot.
            self.stage_data_offsets = {
                "rewards": len(self.episode_data["episode_rewards"]),
                "losses": len(self.episode_data["episode_losses"]),
                "steps": len(self.episode_data["episode_lengths"]),
                "entropies": len(self.episode_data["episode_entropies"]),
                "goals_total": len(self.episode_data.get("episode_goals_total", [])),
                "goals_made": len(self.episode_data.get("episode_goals_made", [])),
                "goals_target": len(self.episode_data.get("episode_goals_target", [])),
            }

        except FileNotFoundError:
            print(f"No checkpoint found at {path}, starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch.")
