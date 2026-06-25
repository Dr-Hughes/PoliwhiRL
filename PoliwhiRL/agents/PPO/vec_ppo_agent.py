# -*- coding: utf-8 -*-
"""Vectorised PPO agent — the ONLY training implementation.

Rollout-based training loop; num_envs == 1 simply runs a single worker.
Each iteration:
  1. Collect T env steps across N envs in parallel.
  2. Compute per-env returns and advantages with V(s_{T+1}) bootstrap.
  3. Flatten (W, N, ...) -> (W*N, ...) and run PPO update with KL early-stop.

The agent owns the run-level shared state: the canonical visit archive
(broadcast to workers once per rollout, persisted via checkpoints).
All workers start from scratch — no snapshot seeding. Reward is a single
stream.
"""
import os
import time
from collections import deque
import numpy as np
import torch
from tqdm.auto import tqdm

from PoliwhiRL.environment import VecPyBoyEnv
from PoliwhiRL.environment.visit_archive import VisitArchive
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
        # ``n_goals_target`` is purely a logging / early-stop threshold in
        # the new design — there is no hard "checklist size". Defaults to 0
        # if a stage doesn't care about the metric.
        self.n_goals = config.get("n_goals_target", 0)
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
        self.state_cycle_strategy = config.get("state_cycle_strategy", "random")
        # Cosine scheduler over rollouts (one scheduler.step per rollout).
        config["ppo_scheduler_t_max"] = self.num_rollouts

        self.model = PPOModel(self.input_shape, self.action_size, config)
        self.memory = VecPPOMemory(config, self.num_envs)
        # Single-stream reward scaling.
        scaler_min_std = float(config.get("scaler_min_std", 1e-2))
        self.reward_scaler = RewardScaler(
            gamma=self.gamma, num_envs=self.num_envs, min_std=scaler_min_std)

        # Populated when train_agent() builds the vec env.
        self.state_paths = None
        self.env_state_indices = None
        self.env_pending_state_indices = None
        self._vec_env = None

        # CANONICAL visit archive — the single source of truth for the
        # depleting novelty landscape, global across all workers and
        # persistent across curriculum stages (checkpointed in info.pth).
        # Workers report each episode's genuinely-visited cells/maps in
        # terminal_info; the agent merges them here (+1 per episode) and
        # broadcasts the table back once per rollout. Worker archives are
        # read-only replicas.
        self.visit_archive = VisitArchive()
        self._archive_dirty = False
        self._critic_warmup_remaining = 0

        # Live entropy coefficient (for tracking/logging).
        self._entropy_coef = float(config.get("ppo_entropy_coef", 0.02))

        self.best_reward = float("-inf")
        # best/ is selected on goal-success rate (directed stages) or intrinsic
        # exploration (free-play), not mean reward — so the next stage inherits
        # a competent policy, not the best reward-farmer.
        self.best_success_rate = -1.0
        self.best_intrinsic = float("-inf")
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
            # Curriculum-progress metrics, parallel to episode_rewards.
            # Replay cutoff randomises the per-episode starting N_goals so
            # episode_rewards alone is a noisy progress signal — these
            # arrays let us watch real progress instead.
            "episode_goals_total": [],
            "episode_goals_made": [],
            "episode_goals_target": [],
            # Phase-4 progress signals — see ppo_agent.py for the same set.
            "episode_flag_fires": [],
            "episode_unique_cells": [],
            "episode_unique_maps": [],
            "episode_archive_size": [],
            # Per-source episode reward breakdown — see ppo_agent.py.
            "episode_reward_sources": [],
            # Per-episode "hit the stage milestone" boolean (from the
            # worker's terminal_info goal_success). Drives best/ selection
            # on goal-success rate instead of mean reward.
            "episode_goal_success": [],
            # Step at which each goal rung fired (list per episode) —
            # bottleneck-rung / time-budget analysis.
            "episode_goal_fire_steps": [],
            # Rollout-indexed diagnostics (one entry per rollout).
            "rollout_policy_entropy": [],
            "rollout_entropy_coef": [],
            "rollout_lr": [],
            # PPO optimisation health (means over the rollout's minibatches;
            # 0.0 on rollouts with no update).
            "rollout_approx_kl": [],
            "rollout_clip_fraction": [],
            "rollout_actor_loss": [],
            "rollout_critic_loss": [],
            # Wall-clock per rollout (collection + update) — throughput
            # degradation is an early memory-pressure signal.
            "rollout_duration_s": [],
        }
        self.episode_data["buttons_pressed"].append(0)
        self._entropy_last_reset_ep = 0
        self._entropy_reset_count = 0
        self._early_stopped = False
        # Per-stage latch: set once this stage's goal-success rate clears
        # ``entropy_reset_solved_success_rate``. Once solved, a later drop in
        # success is reward-hacking / regression, not under-exploration, so the
        # entropy plateau reset must NOT re-inject exploration. Reset per stage
        # in load_model (and starts fresh each process).
        self._stage_solved = False
        # Rolling window of recent goal_success flags for best/ selection.
        self._goal_success_window = deque(
            maxlen=int(self.config.get("best_success_window", 100))
        )

    def _real_episode_budget(self):
        """Estimate of how many real episodes this stage will complete.

        Window/min/debounce sizing must be in *completed-episode* units, not
        ``num_rollouts × num_envs``: each episode of ``episode_length`` steps
        spans ``episode_length / ppo_update_frequency`` rollouts, so the naive
        product over-counts episodes by that factor (≈8× for a 1024-step
        episode at update_frequency 128) — which silently made plateau windows
        far too wide to ever fire. Falls back to the old product when the
        timing config isn't available (keeps stub-based unit tests stable).
        """
        rollouts = int(getattr(self, "num_rollouts", self.config.get("num_rollouts", 1)))
        envs = int(getattr(self, "num_envs", 1))
        upd = int(self.config.get("ppo_update_frequency", 0))
        ep_len = int(self.config.get("episode_length", 0))
        if upd > 0 and ep_len > 0:
            per_env = max(1.0, rollouts * upd / ep_len)
            return int(per_env * envs)
        return rollouts * envs

    def _goal_ceiling(self):
        """Stage-local achievement ceiling and how long it has been stuck.

        Returns ``(best, stuck_eps)``: the highest ``episode_goals_total``
        reached in THIS stage, and the number of completed stage episodes
        since that ceiling was FIRST reached. First occurrence, not last:
        re-hitting an old best is consolidation, not advancement — the
        mean-progress version of this signal is exactly what let stage 4
        sit at 3/4 goals for 2800+ episodes while "progress" (better
        consistency on goals 1–3, archive churn in the reachable region)
        kept standing exploration down. ``best == 0`` counts as stuck since
        stage start. The series is sliced to the current stage because
        checkpoints carry ``episode_data`` across stages.
        """
        goals = self.episode_data.get("episode_goals_total", [])
        stage_eps = self.episode - getattr(self, "stage_start_episode", 0)
        n = min(len(goals), max(0, stage_eps))
        if n <= 0:
            return 0, 0
        stage_goals = goals[-n:]
        best = max(stage_goals)
        if best <= 0:
            return 0, n
        return best, n - 1 - stage_goals.index(best)

    def _stage_sliced(self, key):
        """This stage's slice of a per-episode series (checkpoints carry
        episode_data across stages, so plain indexing would mix stages)."""
        series = self.episode_data.get(key, [])
        n = min(len(series), max(0, self._stage_episode()))
        return series[-n:] if n > 0 else []

    def _check_entropy_plateau(self):
        """Detect training plateaus and rewind the entropy schedule.

        The stagnation SIGNAL is configurable (``entropy_plateau_signal``):
          - ``goals`` (default for directed stages): episode_goals_total.
            Stagnant = the goal CEILING did not advance — no episode in the
            recent window reached a new stage-best goal count. (The old
            strict flat test, max==min, was unsatisfiable once per-episode
            goals bounce between rungs of the ladder, so it never fired on
            the exact failure it existed for.) Keeps the bootstrap guard
            (don't rewind before the first goal is ever hit — that would
            pin the policy near-random and prevent finding goal 1).
          - ``unique_maps`` / ``archive_size`` (for free-play, where goals
            are legitimately ~0): exploration counts are noisy, so flatness
            is a TREND test — the recent half of the window did not improve
            on the earlier half. This makes the detector actually fire in
            free-play, where the old goals-only gate never did.

        Window / debounce / rewind are fractions of the per-stage budget.
        Resets are capped per stage (``entropy_reset_max_count``).
        """
        if not self.config.get("entropy_plateau_reset", True):
            return
        max_resets = int(self.config.get("entropy_reset_max_count", 3))
        if max_resets > 0 and self._entropy_reset_count >= max_resets:
            return
        # If this stage was ever solved, a subsequent plateau/drop is
        # regression or reward-hacking — re-injecting exploration would just
        # restart the farming spiral that broke the old curriculum. Don't.
        if getattr(self, "_stage_solved", False):
            return

        # Episode-space budget for window/debounce sizing, in *completed
        # episodes* (not num_rollouts × num_envs — see _real_episode_budget;
        # the naive product over-counts by episode_length/update_frequency and
        # made the window too wide to ever fire).
        total_budget_eps = self._real_episode_budget()
        window_size = max(50, int(total_budget_eps * self.config.get(
            "entropy_reset_window_fraction", 0.1)))
        min_eps = int(total_budget_eps * self.config.get(
            "entropy_reset_min_fraction", 0.1))
        debounce_eps = int(total_budget_eps * self.config.get(
            "entropy_reset_debounce_fraction", 0.125))

        signal_name = self.config.get("entropy_plateau_signal", "goals")
        series_key = {
            "goals": "episode_goals_total",
            "unique_maps": "episode_unique_maps",
            "archive_size": "episode_archive_size",
        }.get(signal_name, "episode_goals_total")
        series = self.episode_data.get(series_key, [])
        # Slice to the current stage — checkpoints carry episode_data across
        # stages, and a previous stage's history must not poison the window
        # or the ceiling test.
        stage_eps = self.episode - getattr(self, "stage_start_episode", 0)
        if stage_eps > 0:
            series = series[-min(len(series), stage_eps):]

        if stage_eps < min_eps or len(series) < window_size:
            return
        if self.episode - self._entropy_last_reset_ep < debounce_eps:
            return

        recent = series[-window_size:]

        if signal_name == "goals":
            # Bootstrap guard: never rewind before the first goal is hit.
            if max(series) == 0:
                return
            # Already solved? Rate-based on all episodes.
            if self.n_goals > 0:
                solved_rate = float(self.config.get(
                    "entropy_reset_solved_success_rate", 0.5))
                win = getattr(self, "_goal_success_window", deque())
                if len(win) >= max(10, window_size // 2):
                    if float(np.mean(list(win)[-window_size:])) >= solved_rate:
                        return
            # Ceiling test: still climbing if the recent window set a NEW
            # stage-best goal count; stuck if it merely re-hit (or fell
            # short of) a ceiling established before the window.
            prior = series[:-window_size]
            prior_best = max(prior) if prior else 0
            if max(recent) > prior_best:
                return
            detail = f"goal ceiling stuck at {max(series)}/{self.n_goals}"
        else:
            # Trend test for noisy exploration counts: stagnant if the recent
            # half didn't improve on the earlier half.
            half = window_size // 2
            if half < 1:
                return
            early = float(np.mean(recent[:half]))
            late = float(np.mean(recent[half:]))
            eps = 1e-6 * (abs(early) + 1.0)
            if late > early + eps:
                return  # still improving
            detail = f"{signal_name} flat (early={early:.2f} late={late:.2f})"

        # Rewind is applied in the same units as the entropy schedule
        # (rollout-indexed).
        rewind_rollouts = max(10, int(self.num_rollouts * self.config.get(
            "entropy_reset_rewind_fraction", 0.1)))
        new_offset = max(0, self.rollout_idx - rewind_rollouts)
        self.model.set_entropy_offset(new_offset)
        self._entropy_last_reset_ep = self.episode
        self._entropy_reset_count += 1
        print(
            f"[VecPPOAgent] Entropy plateau reset {self._entropy_reset_count}"
            f"/{max_resets} at ep {self.episode} (rollout {self.rollout_idx}): "
            f"rewound offset to {new_offset} ({detail})"
        )

    def _check_early_stopping(self):
        """Check if training should stop early due to sufficient goal completion."""
        if self._early_stopped:
            return True
        if not self.config.get("early_stopping_enabled", False):
            return False
        # Guard: ``g >= 0`` is trivially true for every episode, so an
        # ``n_goals_target`` of 0 would cause an immediate spurious early
        # stop the moment we accumulate ``min_episodes``. Treat <=0 as
        # "no meaningful target configured" and skip.
        if self.n_goals <= 0:
            return False

        window = int(self.config.get("early_stopping_window", 100))
        threshold = float(self.config.get("early_stopping_threshold", 0.3))
        min_eps = int(self.config.get("early_stopping_min_episodes", 50))

        succ = list(self._goal_success_window)
        if len(succ) < min_eps or len(succ) < window:
            return False

        recent = succ[-window:]
        solved = int(sum(recent))
        fraction = solved / window

        if fraction >= threshold:
            self._early_stopped = True
            print(
                f"[VecPPOAgent] Early stop at ep {self.episode} "
                f"(rollout {self.rollout_idx}): "
                f"{solved}/{window} ({fraction:.0%}) recent episodes "
                f"completed all {self.n_goals} goals "
                f"(threshold {threshold:.0%})"
            )
            return True
        return False

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
        # Hold a reference for archive broadcasts.
        self._vec_env = vec_env
        # Push the checkpoint-loaded canonical archive to the fresh workers
        # BEFORE the first reset, so a later stage starts with the previous
        # stages' depleted novelty landscape instead of re-paying the
        # corridor once per stage.
        if self.visit_archive.n_cells_seen() or self.visit_archive.to_state()["maps"]:
            print(
                f"[VecPPOAgent] Visit archive carried over: "
                f"{self.visit_archive.n_cells_seen()} cells"
            )
            self._archive_dirty = True
            self._flush_visit_archive(vec_env)

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
            rollout_t0 = time.monotonic()
            self.memory.reset()
            self._rollout_entropy_sum = 0.0
            self._rollout_entropy_n = 0
            self.model.last_epoch_diag = {}
            self._collect_rollout(
                vec_env, state_seq, ram_seq, mems, ep_returns, ep_lengths
            )
            # Broadcast the merged visit-archive so every worker's novelty
            # landscape reflects ALL workers' discoveries.
            self._flush_visit_archive(vec_env)

            data = self.memory.get_data()
            if data is not None:
                if self._critic_warmup_remaining > 0:
                    self._critic_warmup_remaining -= 1
                else:
                    loss_val, epochs_run = self._update_from_rollout(data)
                    self._record_loss(loss_val, epochs_run)

            # Periodic archive decay so saturated cells gradually become
            # attractive again without aggressively resetting the landscape.
            _decay_rate = float(self.config.get("archive_decay_rate", 0.0))
            _decay_freq = int(self.config.get("archive_decay_frequency", 100))
            if _decay_rate > 0 and self.rollout_idx > 0 and self.rollout_idx % _decay_freq == 0:
                self.visit_archive.decay(_decay_rate)
                self._archive_dirty = True

            # After the update so the optimisation diagnostics (KL, clip
            # fraction, loss components) belong to THIS rollout; before
            # step_scheduler so the logged LR is the one actually used.
            self._record_rollout_diagnostics(time.monotonic() - rollout_t0)

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

            if self._check_early_stopping():
                break

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
                action_mask = self.model._action_mask_for(ram_tensor)
                action_probs, _, new_mems = self.model.actor_critic(
                    state_tensor, ram_tensor, mems, action_mask=action_mask,
                )
                # clamp() does NOT remove NaN (clamping NaN returns NaN), so
                # sanitise non-finite entries first, then floor at 1e-10. With
                # the non-finite-gradient guard in _update_networks this should
                # never fire, but it keeps a transient bad row from killing the
                # whole run at multinomial.
                action_probs = torch.nan_to_num(
                    action_probs, nan=0.0, posinf=0.0, neginf=0.0
                )
                action_probs = torch.clamp(action_probs, 1e-10, 1.0)
                actions_t = torch.multinomial(action_probs, 1).squeeze(1)
                log_probs_t = torch.log(
                    action_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1) + 1e-10
                )
                # Diagnostic: ACTUAL policy entropy at behaviour time (the
                # logged coefficient alone said nothing about how
                # deterministic the policy had become).
                step_entropy = (
                    -(action_probs * torch.log(action_probs + 1e-10))
                    .sum(dim=-1)
                    .mean()
                )
                self._rollout_entropy_sum += float(step_entropy)
                self._rollout_entropy_n += 1

            actions = actions_t.cpu().numpy().astype(np.int64)
            log_probs_np = log_probs_t.cpu().numpy().astype(np.float32)

            for a in actions:
                self.episode_data["buttons_pressed"].append(int(a))

            next_obs, rewards, dones, terminal_infos = vec_env.step(actions)
            next_image, next_ram = next_obs["image"], next_obs["ram"]
            # Observe the reward stream's running variance.
            self.reward_scaler.observe(rewards, dones)

            # Reconstruct the per-step truncation flag from terminal_infos.
            # truncated is only ever True where dones is True; a missing or
            # None info (no episode end) is a non-truncation by definition.
            truncated = np.array(
                [bool(ti and ti.get("truncated")) for ti in terminal_infos],
                dtype=np.bool_,
            )

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
                truncated=truncated,
            )

            ep_returns += rewards
            ep_lengths += 1

            for i in range(self.num_envs):
                if dones[i]:
                    # Terminal goal counts come from the worker (the
                    # post-reset obs has the *new* episode's counts).
                    info = terminal_infos[i] or {}
                    # Merge the episode's genuine visits into the CANONICAL
                    # archive (+1 per cell/map per episode). Broadcast back
                    # to all workers at rollout end.
                    visited_cells = info.get("visited_cells") or []
                    visited_maps = info.get("visited_maps") or []
                    if visited_cells or visited_maps:
                        self.visit_archive.merge_visits(visited_cells, visited_maps)
                        self._archive_dirty = True
                    goals_total = (
                        int(info.get("n_flag", 0))
                        + int(info.get("n_pokedex", 0))
                        + int(info.get("n_map", 0))
                    )
                    n_target = int(info.get("n_target", self.n_goals))
                    self._commit_episode(
                        env_idx=i,
                        reward_sum=float(ep_returns[i]),
                        length=int(ep_lengths[i]),
                        goals_total=goals_total,
                        n_goals_target=int(n_target),
                        flag_fires=int(info.get("flag_fires", 0)),
                        unique_cells=int(info.get("unique_cells", 0)),
                        unique_maps=int(info.get("unique_maps", 0)),
                        # Canonical (global, cross-stage) archive size — the
                        # worker-local value undercounts by ~num_envs.
                        archive_size=int(self.visit_archive.n_cells_seen()),
                        reward_breakdown=info.get("reward_breakdown"),
                        goal_success=bool(info.get("goal_success", False)),
                        goal_fire_steps=info.get("goal_fire_steps"),
                    )
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
                    # the state that was pending before this done. Promote
                    # to "running," then queue the next one.
                    self.env_state_indices[i] = self.env_pending_state_indices[i]
                    self._cycle_env_state(vec_env, i)
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
        if self.state_cycle_strategy == "random":
            next_idx = int(np.random.randint(0, len(self.state_paths)))
        else:
            return
        try:
            vec_env.set_env_state_index(env_idx, next_idx)
        except Exception as e:
            print(f"[VecPPOAgent] Failed to cycle env {env_idx} to state {next_idx}: {e}")
            return
        self.env_pending_state_indices[env_idx] = next_idx

    def _flush_visit_archive(self, vec_env):
        """Broadcast the canonical visit-archive to the workers' read-only
        replicas (full-table replace — self-healing, no drift)."""
        if not self._archive_dirty:
            return
        try:
            vec_env.set_visit_archive(self.visit_archive.to_state())
        except Exception as e:
            print(f"[VecPPOAgent] Failed to broadcast visit archive: {e}")
        self._archive_dirty = False

    def _record_rollout_diagnostics(self, duration_s=0.0):
        ent_n = max(1, getattr(self, "_rollout_entropy_n", 0))
        self.episode_data["rollout_policy_entropy"].append(
            getattr(self, "_rollout_entropy_sum", 0.0) / ent_n
        )
        self.episode_data["rollout_entropy_coef"].append(
            float(self.model._get_entropy_coef(self.rollout_idx))
        )
        lr = None
        optimizer = getattr(self.model, "optimizer", None)
        if optimizer is not None and optimizer.param_groups:
            lr = float(optimizer.param_groups[0].get("lr", 0.0))
        self.episode_data["rollout_lr"].append(lr if lr is not None else 0.0)
        # Optimisation health, aggregated over this rollout's minibatches
        # by run_ppo_epochs (empty dict on rollouts with no update).
        diag = getattr(self.model, "last_epoch_diag", None) or {}
        self.episode_data["rollout_approx_kl"].append(
            float(diag.get("approx_kl", 0.0)))
        self.episode_data["rollout_clip_fraction"].append(
            float(diag.get("clip_fraction", 0.0)))
        self.episode_data["rollout_actor_loss"].append(
            float(diag.get("actor_loss", 0.0)))
        self.episode_data["rollout_critic_loss"].append(
            float(diag.get("critic_loss", 0.0)))
        self.episode_data["rollout_duration_s"].append(float(duration_s))

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

    def _commit_episode(
        self,
        env_idx,
        reward_sum,
        length,
        goals_total,
        n_goals_target,
        flag_fires=0,
        unique_cells=0,
        unique_maps=0,
        archive_size=0,
        reward_breakdown=None,
        goal_success=False,
        goal_fire_steps=None,
    ):
        self.episode += 1
        self.episode_data["episode_rewards"].append(reward_sum)
        self.episode_data["episode_lengths"].append(length)
        self.episode_data["episode_state_indices"].append(
            int(self.env_state_indices[env_idx])
        )
        self.episode_data["episode_goals_total"].append(int(goals_total))
        self.episode_data["episode_goals_made"].append(int(goals_total))
        self.episode_data["episode_goals_target"].append(int(n_goals_target))
        self.episode_data["episode_flag_fires"].append(int(flag_fires))
        self.episode_data["episode_unique_cells"].append(int(unique_cells))
        self.episode_data["episode_unique_maps"].append(int(unique_maps))
        self.episode_data["episode_archive_size"].append(int(archive_size))
        self.episode_data["episode_reward_sources"].append(
            dict(reward_breakdown) if reward_breakdown else {}
        )
        self.episode_data["episode_goal_success"].append(bool(goal_success))
        self.episode_data["episode_goal_fire_steps"].append(
            [int(s) for s in (goal_fire_steps or [])]
        )
        # All episodes are honest (no snapshot seeding) — feed success window.
        self._goal_success_window.append(1.0 if goal_success else 0.0)
        # Latch "this stage was solved" once the rolling success rate clears
        # the threshold — gates the entropy plateau reset (see _check_entropy_plateau).
        if not self._stage_solved and len(self._goal_success_window) >= min(
            int(self.config.get("best_success_min_episodes",
                                self.config.get("best_success_window", 100))),
            self._goal_success_window.maxlen,
        ):
            solved_rate = float(self.config.get(
                "entropy_reset_solved_success_rate", 0.5))
            if float(np.mean(self._goal_success_window)) >= solved_rate:
                self._stage_solved = True
        self.episode_data["moving_avg_reward"].append(reward_sum)
        self.episode_data["moving_avg_length"].append(length)
        self._check_entropy_plateau()
        self.episode_data["episode_entropies"].append(
            self.model._get_entropy_coef(self.rollout_idx)
        )
        self._check_early_stopping()

    # ---------- update ----------

    def _update_from_rollout(self, data):
        # Per-env GAE/returns: reshape so the time axis is contiguous within
        # an env, then fold the env axis into the batch dim for the PPO loss.
        # Single-stream reward normalised by running-return std.
        rewards = data["rewards"] * float(self.reward_scaler.scale_factor())
        dones = data["dones"]                  # (W, N)
        truncated = data.get("truncated")      # (W, N) or None
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
            rewards, values, dones, tail_values, truncated=truncated
        )

        flat_actions = actions.reshape(W * N)
        flat_log_probs = old_log_probs.reshape(W * N)
        flat_returns = returns.reshape(W * N)
        flat_advantages = advantages.reshape(W * N)
        flat_old_values = values.reshape(W * N).detach()

        # Per-rollout advantage normalisation (default). Done once across
        # the full flattened W*N tensor so subsequent minibatches don't
        # renormalise across small slices. See
        # ppo_model_implementation._compute_ppo_losses for the rationale.
        norm_mode = self.config.get("advantage_normalisation", "rollout")
        if norm_mode == "rollout" and flat_advantages.numel() > 1:
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (
                flat_advantages.std() + 1e-8
            )

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
            step=self.rollout_idx,
            epochs=self.epochs,
            minibatch_size=self.minibatch_size,
            target_kl=self.target_kl,
        )

    def _per_env_gae(self, rewards, values, dones, tail_values, truncated=None):
        """GAE along the time axis, independently per env.

        rewards, values, dones: (W, N) tensors. tail_values: (N,).
        truncated: (W, N) bool tensor, or None. ``True`` marks a done that
        was a budget cut-off rather than a natural terminal.
        Returns returns, advantages of shape (W, N).

        Truncation vs terminal: at a boundary we bootstrap V(s_{T+1}) only
        when the episode was *truncated* (cut by the step budget) — a
        natural terminal (goal complete) has no continuation, so its value
        is zeroed. Both cases reset the GAE accumulator via ``not_done``,
        so advantages never leak across episode boundaries.

        When ``truncated`` is None we fall back to treating every done as a
        terminal (zero bootstrap) — the conservative classical behaviour.

        Caveat: the truncation bootstrap uses ``values[t+1]`` which is the
        *post-reset* state's value, not the unobserved continuation of the
        truncated episode. This slightly over-estimates the return at the
        boundary timestep (post-reset states have full novelty available).
        The fully-correct fix requires storing the terminal_obs and a
        separate forward pass — deferred; the single-step bias is small.
        """
        W, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
        not_done = (~dones).to(rewards.dtype)
        # bootstrap[t] = 1 where V(next) should flow into the target:
        # non-terminal steps (~done) and truncated dones; 0 at true terminals.
        if truncated is None:
            bootstrap = not_done
        else:
            trunc = truncated.to(rewards.dtype)
            bootstrap = torch.clamp(not_done + trunc, max=1.0)

        if self.use_gae:
            for t in reversed(range(W)):
                next_value = values[t + 1] if t + 1 < W else tail_values
                delta = rewards[t] + self.gamma * next_value * bootstrap[t] - values[t]
                gae = delta + self.gamma * self.gae_lambda * not_done[t] * gae
                advantages[t] = gae
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards)
            running = tail_values.clone()
            for t in reversed(range(W)):
                next_value = values[t + 1] if t + 1 < W else tail_values
                # At a boundary, carry the bootstrap value (V(next) if
                # truncated, else 0); otherwise carry the running return.
                carry = torch.where(
                    dones[t], bootstrap[t] * next_value, running
                )
                running = rewards[t] + self.gamma * carry
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
        win = self._goal_success_window
        ent = self.episode_data.get("rollout_policy_entropy") or []
        pbar.set_postfix({
            "ep": self.episode,
            "avg_r": f"{float(np.mean(ma_r)):.2f}" if ma_r else "n/a",
            "avg_len": f"{float(np.mean(ma_l)):.1f}" if ma_l else "n/a",
            "success_sr": f"{float(np.mean(win)):.0%}" if win else "n/a",
            "pol_ent": f"{ent[-1]:.3f}" if ent else "n/a",
        })

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
            goals_total=self.episode_data.get("episode_goals_total", None),
            goals_made=self.episode_data.get("episode_goals_made", None),
            goals_target=self.episode_data.get("episode_goals_target", None),
            flag_fires=self.episode_data.get("episode_flag_fires", None),
            unique_cells=self.episode_data.get("episode_unique_cells", None),
            unique_maps=self.episode_data.get("episode_unique_maps", None),
            archive_size=self.episode_data.get("episode_archive_size", None),
            reward_sources=self.episode_data.get("episode_reward_sources", None),
            goal_success=self.episode_data.get("episode_goal_success", None),
            policy_entropies=self.episode_data.get("rollout_policy_entropy", None),
            entropy_coefs=self.episode_data.get("rollout_entropy_coef", None),
            lrs=self.episode_data.get("rollout_lr", None),
            goal_fire_steps=self.episode_data.get("episode_goal_fire_steps", None),
            approx_kls=self.episode_data.get("rollout_approx_kl", None),
            clip_fractions=self.episode_data.get("rollout_clip_fraction", None),
            durations=self.episode_data.get("rollout_duration_s", None),
        )

    def save_model(self, path):
        path = f"{path}"
        os.makedirs(path, exist_ok=True)
        self.model.save(path)

        success_sr = (
            float(np.mean(self._goal_success_window))
            if len(self._goal_success_window) > 0
            else None
        )
        print(
            f"[VecPPOAgent] checkpoint @ rollout {self.rollout_idx + 1}: "
            f"success rate "
            f"{'n/a' if success_sr is None else f'{success_sr:.0%}'} "
            f"(last {len(self._goal_success_window)} eps)"
        )

        info = {
            "episode": self.episode,
            "best_reward": (
                max(self.episode_data["episode_rewards"])
                if self.episode_data["episode_rewards"]
                else float("-inf")
            ),
            "episode_data": self.episode_data,
            "reward_scaler": self.reward_scaler.state_dict(),
            "early_stopped": getattr(self, "_early_stopped", False),
            "success_rate": success_sr,
            # Canonical visit archive — persists the depleting novelty
            # landscape across restarts AND curriculum stages (the next
            # stage loads best/info.pth).
            "visit_archive": self.visit_archive.to_state(),
        }
        torch.save(info, f"{path}/info.pth")

        if self._should_update_best():
            best_path = os.path.join(path, "best")
            os.makedirs(best_path, exist_ok=True)
            self.model.save(best_path)
            torch.save(info, f"{best_path}/info.pth")

    def _should_update_best(self):
        """Decide whether the current policy beats the stored best, updating
        the relevant best-tracker(s) as a side effect when True.

        Selection metric, in priority order:
          1. Directed stage (n_goals>0) with ≥1 success in the window →
             goal-SUCCESS RATE (tie-break on mean reward). This is the point
             of the rework: promote the best task-doer, not the best farmer.
          2. Free-play (n_goals<=0) → intrinsic exploration (moving-average
             unique maps), never raw reward (which battle-farms).
          3. Fallback (directed stage, no success yet) → mean reward, so a
             best/ checkpoint always exists for the next stage to load.
        """
        ma_buf = self.episode_data["moving_avg_reward"]
        if len(ma_buf) < ma_buf.maxlen:
            return False
        current_ma = float(np.mean(ma_buf))

        win = self._goal_success_window
        min_eps = int(self.config.get(
            "best_success_min_episodes",
            self.config.get("best_success_window", 100)))

        # 1. Directed stage with real success history.
        if self.n_goals > 0 and len(win) >= min(min_eps, win.maxlen) and sum(win) > 0:
            sr = float(np.mean(win))
            if sr > self.best_success_rate + 1e-9 or (
                abs(sr - self.best_success_rate) <= 1e-9
                and current_ma > self.best_reward
            ):
                self.best_success_rate = sr
                self.best_reward = current_ma
                return True
            return False

        # 2. Free-play: intrinsic exploration, not reward.
        if self.n_goals <= 0:
            um = self.episode_data.get("episode_unique_maps", [])
            if len(um) >= ma_buf.maxlen:
                cur = float(np.mean(um[-ma_buf.maxlen:]))
                if cur > self.best_intrinsic + 1e-9:
                    self.best_intrinsic = cur
                    self.best_reward = current_ma
                    return True
            return False

        # 3. Fallback: reward-based until a success exists.
        if current_ma > self.best_reward:
            self.best_reward = current_ma
            return True
        return False

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
            # Clear ALL per-stage state for the new curriculum stage so a
            # runner that chains stages in one process behaves like separate
            # processes: plateau debounce/offset, the per-stage reset budget,
            # the early-stop latch, the solved latch, and the success window.
            self._entropy_last_reset_ep = self.episode
            self.model.set_entropy_offset(0)
            self._entropy_reset_count = 0
            self._early_stopped = False
            self._stage_solved = False
            self._entropy_coef = float(self.config.get("ppo_entropy_coef", 0.02))
            self.model.set_entropy_coef(None)
            self._goal_success_window.clear()
            print(f"Loaded checkpoint from {path}, episode {self.episode}")

            # Restore the canonical visit archive so the novelty landscape
            # stays depleted across stages/restarts (the corridor must not
            # re-pay every stage). Broadcast to workers happens at
            # _train_loop start.
            archive_state = info.get("visit_archive")
            if archive_state:
                self.visit_archive.load_state(archive_state)
                self._archive_dirty = True

            # Critic warmup: skip PPO updates for the first N rollouts after
            # loading so the reward scaler can calibrate before the critic
            # sees returns. Only active when reset_reward_scaler_on_load=True.
            reset_scaler_check = self.config.get("reset_reward_scaler_on_load", True)
            if reset_scaler_check:
                self._critic_warmup_remaining = int(
                    self.config.get("critic_warmup_rollouts", 0)
                )

            # Scaler restore (default: reset per stage, since each
            # curriculum stage introduces different reward magnitudes).
            if not reset_scaler_check:
                scaler_state = info.get("reward_scaler")
                if scaler_state is not None:
                    self.reward_scaler.load_state_dict(scaler_state)

            loaded_episode_data = info.get("episode_data", {})
            if loaded_episode_data:
                # Start with a complete fresh skeleton so any per-episode
                # field added since the checkpoint was saved (Phase-4
                # additions: episode_flag_fires, episode_unique_cells,
                # episode_unique_maps, episode_archive_size) is present
                # with an empty list. Then overlay whatever the checkpoint
                # actually carried.
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
                    "episode_goals_total": [],
                    "episode_goals_made": [],
                    "episode_goals_target": [],
                    "episode_flag_fires": [],
                    "episode_unique_cells": [],
                    "episode_unique_maps": [],
                    "episode_archive_size": [],
                    "episode_reward_sources": [],
                    "episode_goal_success": [],
                    "episode_goal_fire_steps": [],
                    "rollout_policy_entropy": [],
                    "rollout_entropy_coef": [],
                    "rollout_lr": [],
                    "rollout_approx_kl": [],
                    "rollout_clip_fraction": [],
                    "rollout_actor_loss": [],
                    "rollout_critic_loss": [],
                    "rollout_duration_s": [],
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
                "state_indices": len(self.episode_data.get("episode_state_indices", [])),
                "goals_total": len(self.episode_data.get("episode_goals_total", [])),
                "goals_made": len(self.episode_data.get("episode_goals_made", [])),
                "goals_target": len(self.episode_data.get("episode_goals_target", [])),
                "flag_fires": len(self.episode_data.get("episode_flag_fires", [])),
                "unique_cells": len(self.episode_data.get("episode_unique_cells", [])),
                "unique_maps": len(self.episode_data.get("episode_unique_maps", [])),
                "archive_size": len(self.episode_data.get("episode_archive_size", [])),
                "reward_sources": len(self.episode_data.get("episode_reward_sources", [])),
                "rollouts": len(self.episode_data.get("rollout_policy_entropy", [])),
            }
        except FileNotFoundError:
            print(f"No checkpoint found at {path}, starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch.")
