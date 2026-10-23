# -*- coding: utf-8 -*-
"""Vectorised PPO agent — the ONLY training implementation.

Rollout-based training loop; num_envs == 1 simply runs a single worker.
Each iteration:
  1. Collect T env steps across N envs in parallel.
  2. Compute per-env returns and advantages with V(s_{T+1}) bootstrap.
  3. Flatten (W, N, ...) -> (W*N, ...) and run PPO update with KL early-stop.

The agent owns the run-level shared state: the canonical visit archive and
Go-Explore frontier pool, both persisted via checkpoints and broadcast to
workers. A configurable honest-worker subset always starts from the canonical
state while the remaining workers may start from frontier snapshots.
"""
import math
import os
import shutil
import time
from collections import deque
import numpy as np
import torch
from tqdm.auto import tqdm

from PoliwhiRL.environment import PyBoyEnvironment, VecPyBoyEnv
from PoliwhiRL.checkpoints import (
    checkpoint_slug,
    checkpoint_title,
    is_recordable_checkpoint,
    threshold_checkpoint_slug,
)

# Maps a threshold-checkpoint "metric" name (config-facing) to the
# per-episode milestone_state key that reports its episode max (see
# Rewards.get_milestone_state). Extend this dict, not the claiming logic
# below, to support a new metric -- keeps "record a run that reaches X" a
# config-only lever rather than per-metric code.
_THRESHOLD_METRIC_STATE_KEYS = {
    "pokedex_owned": "pokedex_owned_max",
    "pokedex_seen": "pokedex_seen_max",
}

# "new_species_catches" lives in terminal_info's "episode_event_counts"
# (Rewards.get_episode_event_counts), not "milestone_state" -- it isn't a
# run-wide-mergeable maximum, just this episode's count, and it's deliberately
# NOT a pokedex_owned_max alias: pokedex_owned_max also rises on evolution (no
# party_size change), which isn't a catch -- see Rewards._catch_progress.
_THRESHOLD_EVENT_COUNT_KEYS = {
    "new_species_catches": "new_species_catches",
}
from PoliwhiRL.environment.vec_env import _load_actions_file, write_actions_file
from PoliwhiRL.environment.visit_archive import VisitArchive
from PoliwhiRL.environment.domain_reward_balancer import DomainRewardBalancer
from PoliwhiRL.replay import VecPPOMemory
from PoliwhiRL.models.PPO import PPOModel
from PoliwhiRL.utils import plot_metrics, RewardScaler
from PoliwhiRL.agents.PPO._minibatch import run_ppo_epochs

# Hard clamp on the servo-controlled entropy coefficient. Mechanism
# bounds, not tuning knobs: the floor keeps the gradient defined, the
# ceiling stops a runaway servo from dissolving the policy outright.
# The floor matters for RECOVERY, not regularisation: the servo climbs
# multiplicatively, so how fast it can respond to an entropy collapse is
# set by how far below useful values the coefficient was allowed to sink.
# At 1e-4 the 2026-07-11 run needed ~30 rollouts to climb back to an
# effective value — the policy had fully collapsed by then. 1e-3 is still
# ~an order of magnitude below where entropy pressure visibly bites.
_ENTROPY_COEF_MIN = 1e-3
_ENTROPY_COEF_MAX = 0.1


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
        # This SESSION's training length. With auto-resume, each invocation
        # continues from the latest checkpoint and runs this many more
        # rollouts (additive), so --add_rollouts / --add_episodes let you
        # append training rather than treating num_rollouts as an absolute
        # stop point. --add_rollouts is the exact unit; --add_episodes is
        # converted to a rollout count via the per-episode rollout rate
        # (episode_length / (ppo_update_frequency * num_envs)).
        self.num_rollouts = int(config["num_rollouts"])
        _add_rollouts = config.get("add_rollouts")
        _add_episodes = config.get("add_episodes")
        if _add_rollouts is not None:
            self.num_rollouts = max(1, int(_add_rollouts))
            print(
                f"[VecPPOAgent] --add_rollouts: {self.num_rollouts} rollouts this session"
            )
        elif _add_episodes is not None:
            _upd = int(config.get("ppo_update_frequency", 1))
            _n = int(config.get("num_envs", 1))
            _el = int(config.get("episode_length", 1))
            _roll_per_ep = max(1.0, _el / max(1, _upd * _n))
            self.num_rollouts = max(
                1, int(math.ceil(int(_add_episodes) * _roll_per_ep))
            )
            print(
                f"[VecPPOAgent] --add_episodes {int(_add_episodes)}: ~"
                f"{self.num_rollouts} rollouts this session"
            )
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
            gamma=self.gamma, num_envs=self.num_envs, min_std=scaler_min_std
        )

        # Populated when train_agent() builds the vec env.
        self.state_paths = None
        self.env_state_indices = None
        self.env_pending_state_indices = None
        self._vec_env = None

        # Go-Explore frontier seeding. When enabled, workers capture save-states
        # at rarely-seen ("frontier") maps (see gym_env._maybe_capture_frontier);
        # the agent curates them into a pool (rarest per map region) and RESTARTS
        # a fraction of episodes from those frontier states so the region beyond
        # the choke point actually gets training time. The first
        # goexplore_probe_fraction of workers (incl. env 0) are never seeded —
        # they train from-scratch, keeping the honest from-start signal and
        # letting the learned exploration skill robustify back onto the full run.
        self.goexplore_enabled = bool(config.get("goexplore_enabled", False))
        self.goexplore_granularity = config.get("goexplore_capture_granularity", "map")
        self.goexplore_probe_fraction = float(
            config.get("goexplore_probe_fraction", 0.25)
        )
        self.goexplore_seed_fraction = float(config.get("goexplore_seed_fraction", 0.6))
        self.goexplore_pool_size = int(config.get("goexplore_pool_size", 48))
        self.goexplore_progress_pool_size = int(
            config.get(
                "goexplore_progress_pool_size",
                max(1, self.goexplore_pool_size // 4),
            )
        )
        self.goexplore_spatial_pool_size = int(
            config.get(
                "goexplore_spatial_pool_size",
                max(0, self.goexplore_pool_size - self.goexplore_progress_pool_size),
            )
        )
        self.goexplore_progress_seed_fraction = float(
            config.get("goexplore_progress_seed_fraction", 0.25)
        )
        self._frontier_index = {}
        self.spatial_frontier_pool = []
        self.progress_frontier_pool = []
        # Combined persisted/reporting view. Sampling is stratified below.
        self.frontier_pool = []
        self.goexplore_probe_workers = 0
        self.env_is_seeded = None
        self.env_is_seeded_pending = None
        self._true_start_path = None
        self._loaded_checkpoint = False

        # First honest reach of each durable story checkpoint. Actions are
        # retained per worker across rollout boundaries, then replayed from the
        # canonical start through the ordinary PNG recorder exactly once.
        self.checkpoint_recording_enabled = bool(
            config.get("checkpoint_recording_enabled", True)
        )
        self.checkpoint_recordings = {}
        # Non-flag "record the first honest episode that reaches metric >=
        # threshold" checkpoints (e.g. a full honest catching run), config-
        # driven so a new one needs no code change -- see
        # _THRESHOLD_METRIC_STATE_KEYS and _claim_checkpoint_recordings.
        # Each item: {"metric": "pokedex_owned", "threshold": 6, "title": ...}.
        self._threshold_checkpoints = config.get("threshold_checkpoints", [])
        # Flag numbers to force a fresh first-honest recording of on this
        # run's next honest fire, even though a (now-stale) recording from
        # an earlier, weaker policy already exists -- see load_model, which
        # drops these from the restored checkpoint_recordings dedup set.
        # One-shot: remove the flag from this list once a fresh recording
        # lands, or it keeps re-claiming on every restart.
        self._recapture_checkpoints = set(config.get("recapture_checkpoints", []))
        self._episode_action_traces = [[] for _ in range(self.num_envs)]
        self._frontier_restore_missing = False

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

        # Merged snapshot of every worker's DomainRewardBalancer state at
        # the last checkpoint save (see DomainRewardBalancer.merge_states).
        # Unlike visit_archive this is NOT kept canonical/live during
        # training — the balancer stays deliberately per-worker-local — it
        # is only used to seed fresh workers once at process start so a
        # restart doesn't reset reward scaling to its under-informed
        # initial state.
        self._domain_balancer_state = None

        # Populated by load_model() from a checkpoint; consumed once by
        # _train_loop() right after the vec env is constructed.
        self._pending_domain_balancer_state = None

        # Fixed entropy coefficient (2026-07-16 rebuild). The closed-loop
        # entropy servo and the plateau-triggered boosts were removed: they
        # existed to pump entropy when exploration stalled, but pumping
        # entropy produces undirected dithering (a near-uniform policy is a
        # diffusive random walk that crosses corridors WORSE), and the stall
        # was really a reward-geometry problem, not an under-exploration one.
        # A small fixed coefficient is what the game-beating Pokémon-RL runs
        # used; set ppo_entropy_anneal_enabled: false in config so the model
        # holds it flat.
        self._entropy_coef = float(config.get("ppo_entropy_coef", 0.01))

        self.best_reward = float("-inf")
        # best/ is selected on goal-success rate (directed stages) or recent
        # run-first discoveries (free-play), not mean reward — so the next
        # stage inherits a competent policy, not the best reward-farmer.
        self.best_success_rate = -1.0
        # Free-play best/ metric: high-water mark of first-honest durable
        # checkpoint recordings. Starts at
        # -1 so the first full-window evaluation always writes a best/
        # (guarantees the next stage has something to load), after which
        # only a policy that actually pushed the run frontier can beat it.
        self.best_discoveries = -1
        self.episode = int(config["start_episode"])  # completed-episode counter
        self.stage_start_episode = self.episode
        self.stage_data_offsets = None
        self.rollout_idx = 0
        self.total_rollouts = 0

        # Regression probe: periodically measure success on a FIXED earlier
        # skill (e.g. stage 2's "get the starter") from the true starting
        # save-state, using the CURRENT weights, independent of whatever
        # this stage's own training distribution looks like. This is the
        # only way to catch catastrophic forgetting during an undirected
        # ("freeform") stage — the stage's own metrics only measure whether
        # ITS OWN goal is progressing, not whether earlier skills eroded.
        # Runs single-threaded (no vectorisation) so keep probe_frequency
        # and probe_episodes modest — this adds real wall-clock overhead.
        self.probe_enabled = bool(config.get("probe_enabled", False))
        # Shared by both probes (regular + long-horizon) as a print-label
        # prefix, so read unconditionally — either probe may be enabled
        # independently of the other.
        self.probe_label = config.get("probe_label", "probe")
        self._probe_env = None
        if self.probe_enabled:
            self.probe_frequency = int(config.get("probe_frequency", 20))
            self.probe_episodes = int(config.get("probe_episodes", 5))
            self.probe_episode_length = int(
                config.get("probe_episode_length", config["episode_length"])
            )
            probe_config = dict(config)
            probe_config["episode_length"] = self.probe_episode_length
            probe_config["goals"] = config.get("probe_goals", [])
            probe_config["terminate_on_goal_complete"] = False
            probe_config["record"] = False
            if config.get("probe_state_path"):
                probe_config["state_path"] = config["probe_state_path"]
            self._probe_config = probe_config

        # Long-horizon capability probe (see _run_long_horizon_probe):
        # same ladder, same policy, much longer episode_length and a much
        # lower frequency (real wall-clock cost, single-threaded, and
        # proportional to episode_length). Independent on/off switch —
        # opt in per stage.
        self.long_probe_enabled = bool(config.get("long_probe_enabled", False))
        self._long_probe_env = None
        if self.long_probe_enabled:
            self.long_probe_frequency = int(config.get("long_probe_frequency", 200))
            self.long_probe_episodes = int(config.get("long_probe_episodes", 3))
            self.long_probe_episode_length = int(
                config.get("long_probe_episode_length", 4 * config["episode_length"])
            )
            long_probe_config = dict(config)
            long_probe_config["episode_length"] = self.long_probe_episode_length
            long_probe_config["goals"] = config.get("probe_goals", [])
            long_probe_config["terminate_on_goal_complete"] = False
            long_probe_config["record"] = False
            if config.get("probe_state_path"):
                long_probe_config["state_path"] = config["probe_state_path"]
            self._long_probe_config = long_probe_config

        self.reset_tracking()

    def _stage_episode(self):
        return self.episode - self.stage_start_episode

    def reset_tracking(self):
        self.episode_data = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_losses": [],
            # maxlen tracks best_success_window (NOT hardcoded): _should_
            # update_best gates entirely on this buffer reaching maxlen
            # (`len(ma_buf) < ma_buf.maxlen: return False`), so a stage
            # with a short episode budget (e.g. a long-episode stage that
            # only completes a few dozen episodes) needs a matching window
            # or best/ never gets written at all.
            "moving_avg_reward": deque(
                maxlen=int(self.config.get("best_success_window", 100))
            ),
            "moving_avg_length": deque(
                maxlen=int(self.config.get("best_success_window", 100))
            ),
            "moving_avg_loss": deque(
                maxlen=int(self.config.get("best_success_window", 100))
            ),
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
            # Why each episode ended ("budget" / "stagnation" /
            # "battle_stagnation" / None for a genuine terminal) and how
            # close to the stagnation limit it got (0-1), even when that
            # wasn't the cause. Tracks whether the agent is learning to
            # avoid getting stuck at all, not just escaping once stuck.
            "episode_truncation_cause": [],
            "episode_final_stagnation_clock": [],
            # Pokedex progress ceiling reached within the episode (see
            # Rewards.get_milestone_state) — lets "push catching hard" be
            # judged against real trend data instead of only the sparse
            # discovery log (which only fires on a run-wide first).
            "episode_pokedex_seen": [],
            "episode_pokedex_owned": [],
            # Discrete rare-event counts per episode (see
            # Rewards.get_episode_event_counts) — trainer-battle wins and
            # catches, split new-species vs duplicate. Parallel to
            # episode_rewards from the episode this instrumentation was
            # added onward; shorter than episode_rewards for any run that
            # predates it (see the load-time skeleton below).
            "episode_trainer_wins": [],
            "episode_catches": [],
            "episode_new_species_catches": [],
            "episode_duplicate_catches": [],
            # Go-Explore: was this episode started from a seeded frontier
            # snapshot (True) or from the true start (False)? Parallel to
            # episode_rewards. Lets metrics separate HONEST base-policy
            # progress from teleported (seeded) reach — without it, ~45% of
            # the stream is teleported and "reached bank X" / mean_unique_maps
            # blend the two and can't answer "can the base policy get there?".
            "episode_seeded": [],
            # Per-source episode reward breakdown — see ppo_agent.py.
            "episode_reward_sources": [],
            # Per-episode "hit the stage milestone" boolean (from the
            # worker's terminal_info goal_success). Drives best/ selection
            # on goal-success rate instead of mean reward.
            "episode_goal_success": [],
            # Step at which each goal rung fired (list per episode) —
            # bottleneck-rung / time-budget analysis.
            "episode_goal_fire_steps": [],
            # [flag_num, step] pairs per episode for EVERY derived-table
            # flag fire, unconditioned on the stage's goal list. The
            # time-to-rung series for goal-less stages, and the input to
            # the snapshot-seeding trigger (median deepest-fire step vs
            # episode budget).
            "episode_flag_fire_steps": [],
            # Diagnostic-only, run-wide, flat log of every genuine first-
            # ever milestone fire (flag/map/pokedex/level/key_item), each
            # stamped with episode + rollout_idx + in-episode step — see
            # Rewards._discoveries_this_episode / _commit_episode. Lets a
            # discovery-order graph be reconstructed after a run instead of
            # hand-authored.
            "discovery_log": [],
            # Rollout-indexed diagnostics (one entry per rollout).
            "rollout_policy_entropy": [],
            "rollout_behaviour_entropy": [],
            "rollout_entropy_coef": [],
            "rollout_lr": [],
            # PPO optimisation health (means over the rollout's minibatches;
            # 0.0 on rollouts with no update).
            "rollout_approx_kl": [],
            "rollout_base_post_kl": [],
            "rollout_behaviour_post_kl": [],
            "rollout_base_replay_kl": [],
            "rollout_behaviour_replay_kl": [],
            "rollout_action_log_prob_abs_max": [],
            "rollout_action_log_prob_abs_mean": [],
            "rollout_replay_rejected": [],
            "rollout_kl_rejected": [],
            "rollout_clip_fraction": [],
            "rollout_actor_loss": [],
            "rollout_critic_loss": [],
            # Wall-clock per rollout (collection + update) — throughput
            # degradation is an early memory-pressure signal.
            "rollout_duration_s": [],
            # Regression probe (see __init__) — sparse series, one entry
            # per probe event, not per rollout. probe_rollout_idx records
            # which rollout each probe_success_rate entry corresponds to.
            "probe_success_rate": [],
            "probe_rollout_idx": [],
            # Per-goal-TYPE completion rate per probe event (dict, e.g.
            # {"flag": 0.9, "map": 0.8, "pokedex": 0.2}) — rung survival
            # when probe_goals is a multi-rung ladder, where the
            # all-or-nothing probe_success_rate only measures the deepest
            # rung.
            "probe_goal_type_rates": [],
            # Per configured goal identity (not just broad type), e.g.
            # "Got Mystery Egg From Mr Pokemon" -> 0.35.
            "probe_goal_rates": [],
            # Per probe event: list (one entry per probe episode) of that
            # episode's goal_fire_steps — on-policy time-to-rung from the
            # true start.
            "probe_fire_steps": [],
            "assisted_probe_success_rate": [],
            "assisted_probe_rollout_idx": [],
            "assisted_probe_goal_type_rates": [],
            "assisted_probe_goal_rates": [],
            "assisted_probe_fire_steps": [],
            # Long-horizon capability probe (see _run_long_horizon_probe) —
            # same shape as the probe_* series above, at a much longer
            # episode_length. Answers "would this policy progress further
            # in the story given more runway", independent of the
            # training episode-length constraint.
            "long_probe_success_rate": [],
            "long_probe_rollout_idx": [],
            "long_probe_goal_type_rates": [],
            "long_probe_goal_rates": [],
            "long_probe_fire_steps": [],
            "assisted_long_probe_success_rate": [],
            "assisted_long_probe_rollout_idx": [],
            "assisted_long_probe_goal_type_rates": [],
            "assisted_long_probe_goal_rates": [],
            "assisted_long_probe_fire_steps": [],
        }
        self.episode_data["buttons_pressed"].append(0)
        self._early_stopped = False
        # Rolling window of recent goal_success flags for best/ selection.
        self._goal_success_window = deque(
            maxlen=int(self.config.get("best_success_window", 100))
        )
        self._discovery_keys_seen = set()

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
        rollouts = int(
            getattr(self, "num_rollouts", self.config.get("num_rollouts", 1))
        )
        envs = int(getattr(self, "num_envs", 1))
        upd = int(self.config.get("ppo_update_frequency", 0))
        ep_len = int(self.config.get("episode_length", 0))
        if upd > 0 and ep_len > 0:
            per_env = max(1.0, rollouts * upd / ep_len)
            return int(per_env * envs)
        return rollouts * envs

    def _stage_sliced(self, key):
        """This stage's slice of a per-episode series (checkpoints carry
        episode_data across stages, so plain indexing would mix stages)."""
        series = self.episode_data.get(key, [])
        n = min(len(series), max(0, self._stage_episode()))
        return series[-n:] if n > 0 else []

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
        # Go-Explore: resolve the frontier-snapshot dir (default under the run's
        # output). A fresh run clears stale files; an auto-resumed run preserves
        # the files referenced by the checkpointed frontier-pool manifest.
        if self.goexplore_enabled and not self.config.get("goexplore_snapshot_dir"):
            self.config["goexplore_snapshot_dir"] = os.path.join(
                self.config.get("output_base_dir", "."), "goexplore_snapshots"
            )
        if self.goexplore_enabled:
            snap_dir = self.config["goexplore_snapshot_dir"]
            if not self._loaded_checkpoint and os.path.isdir(snap_dir):
                shutil.rmtree(snap_dir, ignore_errors=True)
            os.makedirs(snap_dir, exist_ok=True)
            if self._loaded_checkpoint and (
                not self.frontier_pool or self._frontier_restore_missing
            ):
                # Legacy checkpoint recovery: historical rarity counts can
                # otherwise prevent the now-empty pool from ever repopulating.
                self.config["goexplore_rearm_checkpoint_capture"] = True
                self.config["goexplore_flag_capture"] = True
                print(
                    "[GoExplore] resumed with an empty/incomplete frontier pool; "
                    "re-arming one-time durable-checkpoint captures",
                    flush=True,
                )

        vec_env = VecPyBoyEnv(self.config, self.num_envs)
        try:
            self._train_loop(vec_env)
        finally:
            vec_env.close()
            if self._probe_env is not None:
                self._probe_env.close()
            if self._long_probe_env is not None:
                self._long_probe_env.close()

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
        self._true_start_path = (
            self.state_paths[0] if self.state_paths else self.config["state_path"]
        )
        self._episode_action_traces = [[] for _ in range(self.num_envs)]

        # Go-Explore: resolve the true-start save-state (what un-seeded workers
        # always reset from) and the never-seeded probe-worker count. env 0 is
        # always a probe worker so recording/probe behaviour stays from-scratch.
        if self.goexplore_enabled:
            self.goexplore_probe_workers = max(
                1, int(round(self.num_envs * self.goexplore_probe_fraction))
            )
            self.env_is_seeded = [False] * self.num_envs
            self.env_is_seeded_pending = [False] * self.num_envs
            print(
                f"[GoExplore] enabled: {self.goexplore_probe_workers}/"
                f"{self.num_envs} from-scratch (probe) workers, "
                f"seed_fraction={self.goexplore_seed_fraction}, "
                f"capture map_count<={self.config.get('goexplore_capture_map_count_max', 100)}"
            )
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

        # One-time (not per-rollout) seed of every fresh worker's domain
        # reward balancer from the last checkpoint's merged estimate. After
        # this the balancer is per-worker-local again, as designed.
        if self._pending_domain_balancer_state:
            vec_env.set_domain_balancer_state(self._pending_domain_balancer_state)
            self._domain_balancer_state = self._pending_domain_balancer_state
            self._pending_domain_balancer_state = None
            print(
                "[VecPPOAgent] Domain reward balancer state carried over "
                "from checkpoint",
                flush=True,
            )

        obs = vec_env.reset()  # {"image": (N, C, H, W), "ram": (N, D)}
        self._retry_checkpoint_recordings()
        # ROLLOUT is single-frame Transformer-XL inference: one frame per
        # step with the memory carried across steps. (The trainable segment
        # length `sequence_length` only governs how the STORED steps are
        # batched for the BPTT update — see VecPPOMemory.get_data.) So the
        # rollout "window" is length 1.
        state_seq = obs["image"][:, None].copy()  # (N, 1, *input_shape)
        ram_seq = obs["ram"][:, None].copy()  # (N, 1, ram_obs_dim)
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
            self._rollout_behaviour_entropy_sum = 0.0
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

            # MPS's caching allocator does not return freed memory to the OS
            # on its own; over hundreds of rollouts of forward/backward
            # passes with varying batch shapes this accumulates until the
            # system swaps itself to a crawl (observed: ~5s/rollout growing
            # to 30s+/rollout over ~175 rollouts). Cheap relative to a
            # multi-second rollout, so just do it every rollout.
            if self.device.type == "mps":
                torch.mps.empty_cache()

            # (Periodic archive decay REMOVED — it re-inflated the novelty of
            # already-swept ground so re-farming the known region paid again;
            # exploration is now a flat per-episode signal, see rewards.py.)

            # After the update so the optimisation diagnostics (KL, clip
            # fraction, loss components) belong to THIS rollout; before
            # step_scheduler so the logged LR is the one actually used.
            self._record_rollout_diagnostics(time.monotonic() - rollout_t0)
            self.total_rollouts += 1

            if bool((self.model.last_epoch_diag or {}).get("kl_rejected", False)):
                print(
                    "[VecPPOAgent] Stopping after rejected PPO epoch: exact KL "
                    "exceeded the hard safety limit; weights and optimiser were restored."
                )
                break

            self.model.step_scheduler()

            if self.probe_enabled and self.total_rollouts % self.probe_frequency == 0:
                self._run_probe()

            if (
                self.long_probe_enabled
                and self.total_rollouts % self.long_probe_frequency == 0
            ):
                self._run_long_horizon_probe()

            if self.report_episode and hasattr(pbar, "set_postfix"):
                self._update_progress_bar(pbar)

            if (self.rollout_idx + 1) % 10 == 0:
                self._plot_metrics()

            if (
                (self.rollout_idx + 1) % self.checkpoint_frequency == 0
                and self.config.get("save_checkpoint", True)
                and self.config.get("checkpoint") is not None
            ):
                self.save_model(self.config["checkpoint"], vec_env=vec_env)

            if self._check_early_stopping():
                break

        if (
            self.config.get("save_checkpoint", True)
            and self.config.get("checkpoint") is not None
        ):
            self.save_model(self.config["checkpoint"], vec_env=vec_env)

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
                    state_tensor,
                    ram_tensor,
                    mems,
                    action_mask=action_mask,
                )
                # forward returns (N, 1, A) for the single rollout frame;
                # act on that frame.
                action_probs = action_probs[:, -1]
                # clamp() does NOT remove NaN (clamping NaN returns NaN), so
                # sanitise non-finite entries first, then floor at 1e-10. With
                # the non-finite-gradient guard in _update_networks this should
                # never fire, but it keeps a transient bad row from killing the
                # whole run at multinomial.
                step_mask = action_mask[:, -1] if action_mask is not None else None
                base_probs = self.model._normalise_base_policy(action_probs, step_mask)
                behaviour_probs = self.model._behaviour_policy(
                    base_probs, ram_tensor, action_mask
                )
                base_entropy = -(
                    base_probs * torch.log(base_probs.clamp(min=1e-10))
                ).sum(dim=-1).mean()
                behaviour_entropy = -(
                    behaviour_probs * torch.log(behaviour_probs.clamp(min=1e-10))
                ).sum(dim=-1).mean()
                self._rollout_entropy_sum += float(base_entropy)
                self._rollout_behaviour_entropy_sum += float(behaviour_entropy)
                self._rollout_entropy_n += 1
                actions_t = torch.multinomial(behaviour_probs, 1).squeeze(1)
                log_probs_t = torch.log(
                    behaviour_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1) + 1e-10
                )

            actions = actions_t.cpu().numpy().astype(np.int64)
            log_probs_np = log_probs_t.cpu().numpy().astype(np.float32)
            base_probs_np = base_probs.cpu().numpy().astype(np.float32)
            behaviour_probs_np = behaviour_probs.cpu().numpy().astype(np.float32)

            for a in actions:
                self.episode_data["buttons_pressed"].append(int(a))
            for i, action in enumerate(actions):
                self._episode_action_traces[i].append(int(action))

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

            terminal_bootstrap_values = np.zeros(self.num_envs, dtype=np.float32)
            truncated_indices = np.flatnonzero(truncated)
            if truncated_indices.size:
                terminal_obs = [
                    terminal_infos[i]["terminal_observation"]
                    for i in truncated_indices
                ]
                terminal_images = torch.from_numpy(
                    np.stack([o["image"] for o in terminal_obs])
                ).float().to(self.device).unsqueeze(1)
                terminal_ram = torch.from_numpy(
                    np.stack([o["ram"] for o in terminal_obs])
                ).float().to(self.device).unsqueeze(1)
                terminal_idx = torch.as_tensor(
                    truncated_indices, device=self.device, dtype=torch.long
                )
                terminal_mems = [m.index_select(0, terminal_idx) for m in new_mems]
                with torch.no_grad():
                    _, terminal_v, _ = self.model.actor_critic(
                        terminal_images, terminal_ram, terminal_mems
                    )
                terminal_bootstrap_values[truncated_indices] = (
                    terminal_v[:, -1, 0].cpu().numpy().astype(np.float32)
                )

            # Take ownership before the done path overwrites state_seq with
            # the worker's post-reset observation.  A view here paired the
            # pre-step policy/memory with reset observations in stored BPTT.
            states_now = state_seq[:, -1].copy()
            ram_now = ram_seq[:, -1].copy()

            ep_returns += rewards
            ep_lengths += 1

            checkpoint_jobs = []
            for i in range(self.num_envs):
                if dones[i]:
                    # Terminal goal counts come from the worker (the
                    # post-reset obs has the *new* episode's counts).
                    info = terminal_infos[i] or {}
                    running_seeded = bool(
                        self.goexplore_enabled
                        and self.env_is_seeded is not None
                        and self.env_is_seeded[i]
                    )
                    # Merge the episode's genuine visits into the CANONICAL
                    # archive (+1 per cell/map per episode). Broadcast back
                    # to all workers at rollout end.
                    visited_cells = info.get("visited_cells") or []
                    visited_maps = info.get("visited_maps") or []
                    if visited_cells or visited_maps:
                        self.visit_archive.merge_visits(visited_cells, visited_maps)
                        self._archive_dirty = True
                    # Go-Explore: fold this episode's frontier save-states into
                    # the curated pool we seed workers from.
                    if self.goexplore_enabled:
                        caps = info.get("frontier_captures") or []
                        if caps:
                            self._ingest_frontier_captures(caps)
                    # Milestone ledger (see get_milestone_state) — backs the
                    # discovery log AND the milestone re-fire depletion the
                    # reward path reads. Merged the same way, before _commit_episode logs
                    # discoveries against it. Always merge (even all-zero)
                    # since level/pokedex maxima are meaningful at 0. A
                    # changed ledger must mark the archive dirty in its own
                    # right: once the cell archive saturates nothing else
                    # does, and without the broadcast the workers' dedup
                    # replicas go permanently stale.
                    milestone_state = info.get(
                        "milestone_state",
                        {
                            "flags_fired": [],
                            "pokedex_seen_max": 0,
                            "pokedex_owned_max": 0,
                            "level_max": 0,
                            "key_items_max": 0,
                            "milestone_fires": [],
                        },
                    )
                    event_counts = info.get("episode_event_counts", {})
                    if self.visit_archive.merge_milestones(
                        checkpoint_seeded=running_seeded, **milestone_state
                    ):
                        self._archive_dirty = True
                    goals_total = (
                        int(info.get("n_flag", 0))
                        + int(info.get("n_pokedex", 0))
                        + int(info.get("n_map", 0))
                    )
                    n_target = int(info.get("n_target", self.n_goals))
                    checkpoint_jobs.extend(
                        self._claim_checkpoint_recordings(
                            env_idx=i,
                            info=info,
                            actions=self._episode_action_traces[i],
                            seeded=running_seeded,
                            state_index=int(self.env_state_indices[i]),
                        )
                    )
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
                        discoveries=info.get("discoveries"),
                        flag_fire_steps=info.get("flag_fire_steps"),
                        # env_is_seeded[i] here is the RUNNING flag for the
                        # episode that just ended (promotion to the next
                        # episode's flag happens below, after this commit).
                        seeded=running_seeded,
                        truncation_cause=info.get("truncation_cause"),
                        final_stagnation_clock=float(
                            info.get("final_stagnation_clock", 0.0)
                        ),
                        pokedex_seen=int(milestone_state.get("pokedex_seen_max", 0)),
                        pokedex_owned=int(milestone_state.get("pokedex_owned_max", 0)),
                        trainer_wins=int(event_counts.get("trainer_wins", 0)),
                        catches=int(event_counts.get("catches", 0)),
                        new_species_catches=int(event_counts.get("new_species_catches", 0)),
                        duplicate_catches=int(event_counts.get("duplicate_catches", 0)),
                    )
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0
                    self._episode_action_traces[i] = []
                    # Single-frame rollout window: load the post-reset obs and
                    # zero this env's carried memory (fresh episode).
                    state_seq[i, 0] = next_image[i]
                    ram_seq[i, 0] = next_ram[i]
                    for layer in range(len(new_mems)):
                        new_mems[layer][i].zero_()
                    # The auto-reset that just happened in the worker used
                    # the state that was pending before this done. Promote
                    # to "running," then queue the next one.
                    self.env_state_indices[i] = self.env_pending_state_indices[i]
                    if self.goexplore_enabled:
                        self.env_is_seeded[i] = self.env_is_seeded_pending[i]
                    self._cycle_env_state(vec_env, i)
                else:
                    state_seq[i, 0] = next_image[i]
                    ram_seq[i, 0] = next_ram[i]

            # Claim every same-step first before replaying any of them, so two
            # workers cannot race to record the same checkpoint.
            for job in checkpoint_jobs:
                self._record_checkpoint_replay(job)

            # Recording fires on env-0 dones, after the cycling above so the
            # folder naming reflects the just-finished episode.
            if dones[0]:
                self._maybe_enable_recording(vec_env)

            self.memory.store_step(
                states=states_now,
                ram_states=ram_now,
                next_states=next_image,
                next_ram_states=next_ram,
                actions=actions,
                rewards=rewards,
                dones=dones,
                log_probs=log_probs_np,
                base_probs=base_probs_np,
                behaviour_probs=behaviour_probs_np,
                mems=mems,
                outgoing_mems=new_mems,
                truncated=truncated,
                terminal_bootstrap_values=terminal_bootstrap_values,
            )
            # Preserve carried inference history across rollout boundaries.
            # Done rows in new_mems were zeroed above for their reset episode.
            mems[:] = new_mems

    def _cycle_env_state(self, vec_env, env_idx):
        """Pick the next state for env_idx according to state_cycle_strategy
        and send it to the worker. The worker's next auto-reset will use it.
        The choice is queued in env_pending_state_indices and promoted to
        env_state_indices when that auto-reset actually fires.
        """
        if self.goexplore_enabled:
            self._goexplore_cycle(vec_env, env_idx)
            return
        if len(self.state_paths) <= 1 or self.state_cycle_strategy == "none":
            return  # nothing to cycle
        if self.state_cycle_strategy == "random":
            next_idx = int(np.random.randint(0, len(self.state_paths)))
        else:
            return
        try:
            vec_env.set_env_state_index(env_idx, next_idx)
        except Exception as e:
            print(
                f"[VecPPOAgent] Failed to cycle env {env_idx} to state {next_idx}: {e}"
            )
            return
        self.env_pending_state_indices[env_idx] = next_idx

    def _frontier_key(self, cap):
        """Pool key for a snapshot: the event-flag bit for a flag-state
        capture, the quantised cell in "cell" mode, else the whole map region.
        Determines what counts as "the same frontier location" for
        dedup/eviction. Flag and cell/map captures coexist in one pool."""
        kind = cap.get("kind", self.goexplore_granularity)
        if kind in ("flag", "checkpoint"):
            return ("flag", int(cap["flag"]))
        if kind == "party":
            return ("party", int(cap["party_size"]))
        if kind == "cell":
            return self.visit_archive.cell_key(
                int(cap["bank"]), int(cap["num"]), int(cap["x"]), int(cap["y"])
            )
        return (int(cap["bank"]), int(cap["num"]))

    @staticmethod
    def _is_progress_capture(cap):
        """Whether a capture belongs to the protected durable-progress pool."""
        kind = cap.get("kind")
        if kind in ("checkpoint", "party"):
            return True
        # Existing checkpoints persist kind="flag"; classify them correctly
        # when restoring an older manifest.
        return kind == "flag" and is_recordable_checkpoint(cap.get("flag", -1))

    def _live_count(self, cap):
        """Current run-wide rarity of a snapshot's frontier, judged LIVE (not
        by the stale capture-time count): run-wide fire count of the event flag
        for a flag capture, cell count in "cell" mode, else map count. Live
        ranking lets early start-region snapshots sink and be evicted while the
        genuine frontier (rarely-visited cell / rarely-fired flag) stays on
        top and is seeded most (weight 1/(1+count))."""
        kind = cap.get("kind", self.goexplore_granularity)
        if kind in ("flag", "checkpoint"):
            return int(self.visit_archive.event_flag_fire_count(int(cap["flag"])))
        if kind == "party":
            # Run-wide rarity of reaching this party size (registered as a
            # "party_size" milestone in the reward path). Rarer (deeper /
            # bigger team) -> lower count -> seeded more (weight 1/(1+count)).
            return int(
                self.visit_archive.milestone_fire_count(
                    "party_size", int(cap["party_size"])
                )
            )
        if kind == "cell":
            return int(
                self.visit_archive.count(
                    int(cap["bank"]), int(cap["num"]), int(cap["x"]), int(cap["y"])
                )
            )
        return int(self.visit_archive.map_count(int(cap["bank"]), int(cap["num"])))

    def _ingest_frontier_captures(self, caps):
        """Fold an episode's frontier save-states into the curated pool: one
        snapshot per frontier location (see _frontier_key), pool ranked by LIVE
        rarity and capped to goexplore_pool_size rarest locations. Files are
        never deleted mid-run (a worker may be about to load one) — the snapshot
            dir is cleared only for a fresh run and preserved across resume; disk is
            bounded because a location stops being captured once its count passes the
            capture threshold."""
        grew = False
        for c in caps:
            key = self._frontier_key(c)
            if key not in self._frontier_index:
                grew = True
            existing = self._frontier_index.get(key)
            if (
                existing is not None
                and existing.get("kind") == "checkpoint"
                and c.get("kind") == "flag"
            ):
                # A durable pre-event snapshot must not be degraded back to a
                # post-event state by a later generic flag capture with the
                # same flag key. A future pre-event checkpoint capture may
                # still replace it with a fresher equivalent.
                continue
            # Keep the newest snapshot for the location (all are frontier frames,
            # so effectively equivalent; newest keeps the freshest file path).
            self._frontier_index[key] = c
        captures = list(self._frontier_index.values())
        spatial = [
            c for c in captures if not VecPPOAgent._is_progress_capture(c)
        ]
        progress = [c for c in captures if VecPPOAgent._is_progress_capture(c)]
        spatial_cap = int(
            getattr(self, "goexplore_spatial_pool_size", self.goexplore_pool_size)
        )
        progress_cap = int(getattr(self, "goexplore_progress_pool_size", 0))
        self.spatial_frontier_pool = sorted(spatial, key=self._live_count)[:spatial_cap]
        self.progress_frontier_pool = sorted(progress, key=self._live_count)[:progress_cap]
        self.frontier_pool = self.spatial_frontier_pool + self.progress_frontier_pool
        if grew:
            print(
                f"[GoExplore] frontier locations={len(self._frontier_index)} "
                f"(spatial={len(self.spatial_frontier_pool)}, "
                f"progress={len(self.progress_frontier_pool)}, "
                f"mode={self.goexplore_granularity}); "
                f"pool live-counts: "
                f"{sorted(self._live_count(c) for c in self.frontier_pool)[:12]}",
                flush=True,
            )

    def _pick_frontier_snapshot(self):
        """Sample a frontier snapshot, biased toward rarer locations by LIVE
        count (weight 1/(1+live_count)), so seeding concentrates on the true
        frontier while still occasionally revisiting nearer, better-learned
        ground."""
        spatial = getattr(self, "spatial_frontier_pool", self.frontier_pool)
        progress = getattr(self, "progress_frontier_pool", [])
        if not spatial and not progress:
            return None
        progress_fraction = float(
            getattr(self, "goexplore_progress_seed_fraction", 0.0)
        )
        if progress and (not spatial or np.random.rand() < progress_fraction):
            pool = progress
        else:
            pool = spatial or progress
        weights = np.array(
            [1.0 / (1.0 + self._live_count(c)) for c in pool],
            dtype=np.float64,
        )
        total = weights.sum()
        if total <= 0:
            return pool[0]
        idx = int(np.random.choice(len(pool), p=weights / total))
        return pool[idx]

    def _goexplore_cycle(self, vec_env, env_idx):
        """Choose env_idx's NEXT start state (Go-Explore). Probe workers
        (env 0 .. probe_workers-1) always train from-scratch and are left
        untouched. Other workers restart from a frontier snapshot with
        probability seed_fraction, else return to the true start. Takes effect
        on the worker's next auto-reset; the seeded flag is queued in
        env_is_seeded_pending and promoted alongside env_state_indices."""
        seeded = False
        if env_idx >= self.goexplore_probe_workers:
            seeded = bool(self.env_is_seeded[env_idx])
            choose_seed = bool(
                self.frontier_pool
                and np.random.rand() < self.goexplore_seed_fraction
            )
            if choose_seed:
                snap = self._pick_frontier_snapshot()
                if snap is not None:
                    try:
                        vec_env.set_env_state(
                            env_idx, snap["path"], episode_seeded=True
                        )
                        seeded = True
                    except Exception as e:
                        print(f"[GoExplore] seed failed env {env_idx}: {e}")
            elif self.env_is_seeded[env_idx]:
                # Was running from a seed; hand it back to the true start.
                try:
                    vec_env.set_env_state(
                        env_idx, self._true_start_path, episode_seeded=False
                    )
                    seeded = False
                except Exception as e:
                    print(f"[GoExplore] return-to-start failed env {env_idx}: {e}")
        self.env_is_seeded_pending[env_idx] = seeded

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

    def _run_probe(self):
        """Regression probe: run `probe_episodes` short episodes from the
        TRUE starting save-state with the CURRENT policy weights, checking
        success against a fixed earlier-skill goal (`probe_goals`) —
        independent of this stage's own (possibly goal-less) training
        distribution.

        This is the only mechanism that can catch catastrophic forgetting
        during an undirected ("freeform") stage: the stage's own success
        metric only measures whether ITS OWN objective is progressing, and
        says nothing about whether a previously-learned skill eroded.
        """
        self._probe_env = self._run_probe_pair(
            env=self._probe_env,
            probe_config=self._probe_config,
            episode_length=self.probe_episode_length,
            n_episodes=self.probe_episodes,
            base_keys=(
                "probe_success_rate",
                "probe_rollout_idx",
                "probe_goal_type_rates",
                "probe_goal_rates",
                "probe_fire_steps",
            ),
            assisted_keys=(
                "assisted_probe_success_rate",
                "assisted_probe_rollout_idx",
                "assisted_probe_goal_type_rates",
                "assisted_probe_goal_rates",
                "assisted_probe_fire_steps",
            ),
            label=self.probe_label,
        )

    def _run_long_horizon_probe(self):
        """Same mechanism as ``_run_probe``, at a MUCH longer episode
        length and lower frequency (real wall-clock cost, single-threaded).

        Purpose: the regular probe (and training itself) only ever runs
        the CURRENT policy for `episode_length` steps, so nothing in
        training answers "would this policy actually progress further in
        the story if simply given more time" — the exact question that
        motivates within-episode-bounded exploration rewards in the first
        place. This probe is that direct measurement: same ladder, same
        policy, no training happening on it, just more runway. If the
        rung rates here consistently exceed the short probe's (e.g. the
        pokédex rung lifts off here but not on the regular probe), the
        policy already has the underlying capability and the constraint is
        episode length / consolidation reps, not competence — evidence FOR
        moving to a longer-episode curriculum stage. If they track the
        short probe closely, extra runway isn't the bottleneck.
        """
        self._long_probe_env = self._run_probe_pair(
            env=self._long_probe_env,
            probe_config=self._long_probe_config,
            episode_length=self.long_probe_episode_length,
            n_episodes=self.long_probe_episodes,
            base_keys=(
                "long_probe_success_rate",
                "long_probe_rollout_idx",
                "long_probe_goal_type_rates",
                "long_probe_goal_rates",
                "long_probe_fire_steps",
            ),
            assisted_keys=(
                "assisted_long_probe_success_rate",
                "assisted_long_probe_rollout_idx",
                "assisted_long_probe_goal_type_rates",
                "assisted_long_probe_goal_rates",
                "assisted_long_probe_fire_steps",
            ),
            label=self.probe_label + "_long",
        )

    def _run_probe_pair(
        self,
        env,
        probe_config,
        episode_length,
        n_episodes,
        base_keys,
        assisted_keys,
        label,
    ):
        """Run base/assisted probes without perturbing subsequent training RNG."""
        cpu_rng = torch.random.get_rng_state()
        device_rng = None
        if self.device.type == "cuda":
            device_rng = torch.cuda.get_rng_state(self.device)
        elif self.device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
            device_rng = torch.mps.get_rng_state()
        try:
            env = self._run_probe_pass(
                env,
                probe_config,
                episode_length,
                n_episodes,
                base_keys,
                label + "_base",
                assisted=False,
            )
            env = self._run_probe_pass(
                env,
                probe_config,
                episode_length,
                n_episodes,
                assisted_keys,
                label + "_assisted",
                assisted=True,
            )
        finally:
            torch.random.set_rng_state(cpu_rng)
            if self.device.type == "cuda" and device_rng is not None:
                torch.cuda.set_rng_state(device_rng, self.device)
            elif self.device.type == "mps" and device_rng is not None:
                torch.mps.set_rng_state(device_rng)
        return env

    def _run_probe_pass(
        self, env, probe_config, episode_length, n_episodes, keys, label, assisted
    ):
        """Shared body of ``_run_probe`` / ``_run_long_horizon_probe``:
        run `n_episodes` from the true start with the current policy
        (no gradient), score against `probe_config`'s goal ladder, and
        record under the given episode_data `keys` (success_rate,
        rollout_idx, goal_type_rates, fire_steps). Returns the (possibly
        newly-constructed) probe env for the caller to cache."""
        if env is None:
            from PoliwhiRL.environment.gym_env import PyBoyEnvironment

            env = PyBoyEnvironment(probe_config)

        # Give the paired base/assisted passes the same action-sampling seed.
        # Their trajectories may diverge because their distributions differ,
        # but differences are not caused by unrelated RNG initialisation.
        torch.manual_seed(int(self.total_rollouts))

        # Sync the run-wide novelty landscape into the probe env every call
        # (not just on construction) — vec_env workers get this same
        # broadcast at the start of every rollout (_flush_visit_archive), so
        # without it the probe policy sees a permanently fresh, empty
        # archive: a wildly out-of-distribution observation (directional
        # frontier features etc.) relative to what it actually trained
        # against. Left unsynced, probe results reflect the policy's
        # behaviour on an observation distribution it has never seen, not
        # its true in-distribution competence.
        env.visit_archive.load_state(self.visit_archive.to_state())

        success_key, rollout_key, rates_key, goal_rates_key, fire_steps_key = keys
        successes = 0
        type_hits = {}  # goal type -> completed goals, summed over episodes
        type_totals = {}  # goal type -> configured goals × episodes
        goal_hits = {}  # stable goal label -> successful probe episodes
        goal_keys = None
        fire_steps = []
        for _ in range(n_episodes):
            obs = env.reset()
            state, ram = obs["image"], obs["ram"]
            mems = self.model.init_mems(batch_size=1)

            for _step in range(episode_length):
                # Single-frame inference with carried memory, matching the
                # training rollout (get_action adds the batch axis).
                action, _log_prob, mems = self.model.get_action(
                    state[None], ram[None], mems, assisted=assisted
                )
                next_obs, _reward, done, _truncated = env.step(action)
                state, ram = next_obs["image"], next_obs["ram"]
                if done:
                    break

            rc = env.reward_calculator
            if rc.goals.all_goal_thresholds_met():
                successes += 1
            last_vars = env._last_env_vars or {}
            statuses = rc.goals.per_goal_status(
                pokedex_seen=int(last_vars.get("pokedex_seen", 0)),
                pokedex_owned=int(last_vars.get("pokedex_owned", 0)),
            )
            if goal_keys is None:
                labels = [label for label, _met in statuses]
                label_counts = {label: labels.count(label) for label in set(labels)}
                goal_keys = [
                    label if label_counts[label] == 1 else f"{label} [{idx + 1}]"
                    for idx, label in enumerate(labels)
                ]
            for idx, (goal, (_goal_label, met)) in enumerate(
                zip(rc.goals._goals_raw, statuses)
            ):
                goal_key = goal_keys[idx]
                goal_hits[goal_key] = goal_hits.get(goal_key, 0) + int(met)
                gtype = str(goal.get("type", "unknown"))
                type_totals[gtype] = type_totals.get(gtype, 0) + 1
                type_hits[gtype] = type_hits.get(gtype, 0) + int(met)
            # On-policy time-to-rung from the true start.
            fire_steps.append([int(s) for s in rc.goal_fire_steps])

        rate = successes / max(1, n_episodes)
        type_rates = {
            k: type_hits.get(k, 0) / v for k, v in sorted(type_totals.items())
        }
        goal_rates = {k: goal_hits.get(k, 0) / max(1, n_episodes) for k in goal_hits}
        self.episode_data[success_key].append(rate)
        self.episode_data[rollout_key].append(self.total_rollouts)
        self.episode_data[rates_key].append(type_rates)
        self.episode_data[goal_rates_key].append(goal_rates)
        self.episode_data[fire_steps_key].append(fire_steps)
        rung_desc = " | ".join(f"{k}={v:.0%}" for k, v in goal_rates.items())
        print(
            f"[VecPPOAgent] Probe ({label}) @ rollout "
            f"{self.total_rollouts}: {successes}/{n_episodes} "
            f"({rate:.0%}){' | ' + rung_desc if rung_desc else ''}"
        )
        return env

    def _record_rollout_diagnostics(self, duration_s=0.0):
        ent_n = max(1, getattr(self, "_rollout_entropy_n", 0))
        self.episode_data["rollout_policy_entropy"].append(
            getattr(self, "_rollout_entropy_sum", 0.0) / ent_n
        )
        self.episode_data["rollout_behaviour_entropy"].append(
            getattr(self, "_rollout_behaviour_entropy_sum", 0.0) / ent_n
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
        self.episode_data["rollout_approx_kl"].append(float(diag.get("approx_kl", 0.0)))
        self.episode_data["rollout_base_post_kl"].append(
            float(diag.get("base_post_kl", 0.0))
        )
        self.episode_data["rollout_behaviour_post_kl"].append(
            float(diag.get("behaviour_post_kl", 0.0))
        )
        self.episode_data["rollout_base_replay_kl"].append(
            float(diag.get("base_replay_kl", 0.0))
        )
        self.episode_data["rollout_behaviour_replay_kl"].append(
            float(diag.get("behaviour_replay_kl", 0.0))
        )
        self.episode_data["rollout_action_log_prob_abs_max"].append(
            float(diag.get("action_log_prob_abs_max", 0.0))
        )
        self.episode_data["rollout_action_log_prob_abs_mean"].append(
            float(diag.get("action_log_prob_abs_mean", 0.0))
        )
        self.episode_data["rollout_replay_rejected"].append(
            bool(diag.get("replay_rejected", False))
        )
        self.episode_data["rollout_kl_rejected"].append(
            bool(diag.get("kl_rejected", False))
        )
        self.episode_data["rollout_clip_fraction"].append(
            float(diag.get("clip_fraction", 0.0))
        )
        self.episode_data["rollout_actor_loss"].append(
            float(diag.get("actor_loss", 0.0))
        )
        self.episode_data["rollout_critic_loss"].append(
            float(diag.get("critic_loss", 0.0))
        )
        self.episode_data["rollout_duration_s"].append(float(duration_s))

    def _maybe_enable_recording(self, vec_env):
        if not self.record_enabled or self.record_frequency <= 0:
            return
        if self.episode < self._next_record_episode:
            return
        folder = f"N_goals_{self.n_goals}/ep_{self.episode}"
        try:
            # Periodic trajectory sampling only — "log" (lightweight CSV),
            # not "images". See open_trajectory_log's docstring: nothing
            # downstream reads these PNGs' pixels, only the position/state
            # fields that used to live in each filename. Checkpoint replays
            # (_record_checkpoint_replay) keep the PNG default since those
            # ARE meant for visual confirmation.
            vec_env.enable_record(
                folder, use_episode_number=False, env_idx=0, record_format="log"
            )
        except Exception as e:  # don't kill training over a recording hiccup
            print(f"[VecPPOAgent] Failed to enable recording: {e}")
            return
        self._next_record_episode = self.episode + self.record_frequency

    def _claim_checkpoint_recordings(self, env_idx, info, actions, seeded, state_index):
        """Claim first honest checkpoint hits and persist their action prefixes."""
        if seeded or state_index != 0 or not actions:
            return []

        jobs = []
        for raw_flag, raw_step in info.get("flag_fire_steps") or []:
            flag_num = int(raw_flag)
            if (
                not is_recordable_checkpoint(flag_num)
                or flag_num in self.checkpoint_recordings
            ):
                continue
            # Rewards counts the reset-time baseline as step 1; the parent
            # trace contains only agent-selected actions.
            fire_step = int(raw_step)
            action_count = max(0, min(fire_step - 1, len(actions)))
            if action_count == 0:
                continue
            action_prefix = list(actions[:action_count])
            title = checkpoint_title(flag_num)
            phase = os.path.join("checkpoints", checkpoint_slug(flag_num))
            output_folder = os.path.join(self.config["record_path"], phase)
            actions_path = os.path.join(output_folder, "actions.steps")
            metadata = {
                "flag": flag_num,
                "title": title,
                "episode": self.episode + 1,
                "rollout": self.total_rollouts,
                "env": env_idx,
                "fire_step": fire_step,
                "action_count": action_count,
                "seeded": False,
                "state_index": state_index,
            }
            entry = {
                **metadata,
                "phase": phase,
                "actions_path": actions_path,
                "output_folder": output_folder,
                "status": (
                    "pending" if self.checkpoint_recording_enabled else "disabled"
                ),
            }
            # Central parent-owned claim is the deduplication lock for workers
            # finishing the same checkpoint on the same vector step.
            self.checkpoint_recordings[flag_num] = entry
            if self.checkpoint_recording_enabled:
                write_actions_file(actions_path, [action_prefix], metadata=[metadata])
                jobs.append({"entry": entry, "actions": action_prefix})

        # Non-flag "reached metric >= threshold" checkpoints (e.g. a full
        # honest catching run) -- same honest/state_index==0 gate above, but
        # the whole episode is replayed (no single firing step to truncate
        # to) since the achievement is cumulative, not instantaneous.
        milestone_state = info.get("milestone_state") or {}
        event_counts = info.get("episode_event_counts") or {}
        for spec in self._threshold_checkpoints:
            metric = spec.get("metric")
            state_key = _THRESHOLD_METRIC_STATE_KEYS.get(metric)
            event_key = _THRESHOLD_EVENT_COUNT_KEYS.get(metric)
            if state_key is None and event_key is None:
                continue
            threshold = int(spec.get("threshold", 0))
            key = f"{metric}_{threshold}"
            if key in self.checkpoint_recordings:
                continue
            source = event_counts if event_key is not None else milestone_state
            lookup_key = event_key if event_key is not None else state_key
            if int(source.get(lookup_key, 0)) < threshold:
                continue
            title = spec.get("title") or f"{metric} >= {threshold}"
            phase = os.path.join("checkpoints", threshold_checkpoint_slug(title))
            output_folder = os.path.join(self.config["record_path"], phase)
            actions_path = os.path.join(output_folder, "actions.steps")
            full_actions = list(actions)
            metadata = {
                "metric": metric,
                "threshold": threshold,
                "title": title,
                "episode": self.episode + 1,
                "rollout": self.total_rollouts,
                "env": env_idx,
                "action_count": len(full_actions),
                "seeded": False,
                "state_index": state_index,
            }
            entry = {
                **metadata,
                "kind": "threshold",
                "phase": phase,
                "actions_path": actions_path,
                "output_folder": output_folder,
                "status": (
                    "pending" if self.checkpoint_recording_enabled else "disabled"
                ),
            }
            self.checkpoint_recordings[key] = entry
            if self.checkpoint_recording_enabled:
                write_actions_file(actions_path, [full_actions], metadata=[metadata])
                jobs.append({"entry": entry, "actions": full_actions})
        return jobs

    def _record_checkpoint_replay(self, job):
        """Replay one claimed true-start trajectory through the PNG recorder."""
        entry = job["entry"]
        actions = list(job["actions"])
        env = None
        try:
            replay_config = dict(self.config)
            replay_config["state_path"] = self._true_start_path
            replay_config.pop("state_paths", None)
            replay_config["goexplore_enabled"] = False
            replay_config["action_replay_paths"] = []
            replay_config["record"] = False
            replay_config["episode_length"] = max(
                int(replay_config.get("episode_length", 1)), len(actions) + 1
            )
            env = PyBoyEnvironment(replay_config)
            env.visit_archive.load_state(self.visit_archive.to_state())
            env.reset()
            env.enable_record(entry["phase"], use_episode_number=False)
            kind = entry.get("kind", "flag")
            completed = 0
            observed = False
            for action in actions:
                _obs, _reward, done, _truncated = env.step(int(action))
                completed += 1
                if kind == "threshold":
                    rc = env.reward_calculator
                    current_value = (
                        rc._episode_max_pokedex_owned
                        if entry["metric"] == "pokedex_owned"
                        else rc._episode_max_pokedex_seen
                    )
                    observed = int(current_value) >= int(entry["threshold"])
                else:
                    observed = any(
                        int(flag) == int(entry["flag"])
                        for flag, _step in env.reward_calculator.flag_fire_step_log
                    )
                if done:
                    break
            entry["replayed_steps"] = completed
            entry["checkpoint_observed"] = observed
            entry["status"] = (
                "recorded"
                if completed == len(actions) and observed
                else "checkpoint_not_reproduced"
            )
            print(
                f"[Checkpoint] {entry['status']} first honest '{entry['title']}' "
                f"({completed} steps) -> {entry['output_folder']}",
                flush=True,
            )
            if entry["status"] == "recorded":
                self._finalize_recording_to_video(entry)
        except Exception as exc:  # recording must never kill training
            entry["status"] = "error"
            entry["error"] = str(exc)
            print(
                f"[Checkpoint] replay failed for '{entry['title']}': {exc}",
                flush=True,
            )
        finally:
            if env is not None:
                env.close()

    def _finalize_recording_to_video(self, entry):
        """Stitch a successful replay's PNG sequence into an MP4 and delete
        the raw frames. The video is the useful artefact; thousands of PNGs
        per checkpoint add up fast and nothing downstream reads them once
        the MP4 exists. Never raises -- a failed video must not affect the
        recording's "recorded" status or kill training."""
        try:
            from pathlib import Path

            from tools.png_to_video import add_overlays, make_mp4, sorted_pngs

            folder = entry["output_folder"]
            pngs = sorted_pngs(Path(folder))
            if not pngs:
                return
            out = os.path.join(folder, f"{os.path.basename(folder)}.mp4")
            overlayed, tmp_dir = add_overlays(pngs)
            try:
                make_mp4(overlayed, Path(out), fps=10)
            finally:
                tmp_dir.cleanup()
            for p in pngs:
                os.remove(p)
            entry["video_path"] = out
            print(f"[Checkpoint] video -> {out}", flush=True)
        except Exception as exc:
            print(
                f"[Checkpoint] video generation failed for '{entry.get('title')}': {exc}",
                flush=True,
            )

    def _retry_checkpoint_recordings(self):
        """Retry interrupted first-hit replays after loading a checkpoint."""
        for entry in self.checkpoint_recordings.values():
            if entry.get("status") not in (
                "pending",
                "error",
                "checkpoint_not_reproduced",
            ):
                continue
            actions_path = entry.get("actions_path", "")
            if not os.path.isfile(actions_path):
                candidate = os.path.join(
                    self.config["record_path"],
                    entry.get(
                        "phase",
                        os.path.join("checkpoints", checkpoint_slug(entry["flag"])),
                    ),
                    "actions.steps",
                )
                if os.path.isfile(candidate):
                    actions_path = candidate
                    entry["actions_path"] = candidate
                    entry["output_folder"] = os.path.dirname(candidate)
            trajectories = _load_actions_file(actions_path)
            if trajectories:
                self._record_checkpoint_replay(
                    {
                        "entry": entry,
                        "actions": trajectories[0],
                    }
                )

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
        discoveries=None,
        flag_fire_steps=None,
        seeded=False,
        truncation_cause=None,
        final_stagnation_clock=0.0,
        pokedex_seen=0,
        pokedex_owned=0,
        trainer_wins=0,
        catches=0,
        new_species_catches=0,
        duplicate_catches=0,
    ):
        self.episode += 1
        # Diagnostic-only: stamp each of this episode's genuine run-wide
        # first-ever milestone fires with the (global, monotonic) episode
        # index and current rollout, so a discovery-order graph can be
        # reconstructed after the run. Never read during training.
        for d in discoveries or []:
            raw_key = d.get("key")
            stable_key = tuple(raw_key) if isinstance(raw_key, list) else raw_key
            discovery_key = (str(d.get("type")), stable_key)
            if discovery_key in self._discovery_keys_seen:
                continue
            self._discovery_keys_seen.add(discovery_key)
            self.episode_data["discovery_log"].append(
                {
                    "episode": int(self.episode),
                    "rollout_idx": int(self.total_rollouts),
                    "type": d.get("type"),
                    "key": d.get("key"),
                    "step": int(d.get("step", 0)),
                    "seeded": bool(seeded),
                }
            )
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
        self.episode_data["episode_truncation_cause"].append(truncation_cause)
        self.episode_data["episode_final_stagnation_clock"].append(
            float(final_stagnation_clock)
        )
        self.episode_data["episode_pokedex_seen"].append(int(pokedex_seen))
        self.episode_data["episode_pokedex_owned"].append(int(pokedex_owned))
        self.episode_data["episode_trainer_wins"].append(int(trainer_wins))
        self.episode_data["episode_catches"].append(int(catches))
        self.episode_data["episode_new_species_catches"].append(int(new_species_catches))
        self.episode_data["episode_duplicate_catches"].append(int(duplicate_catches))
        self.episode_data["episode_reward_sources"].append(
            dict(reward_breakdown) if reward_breakdown else {}
        )
        self.episode_data["episode_goal_success"].append(bool(goal_success))
        self.episode_data["episode_goal_fire_steps"].append(
            [int(s) for s in (goal_fire_steps or [])]
        )
        self.episode_data["episode_flag_fire_steps"].append(
            [[int(f), int(s)] for f, s in (flag_fire_steps or [])]
        )
        self.episode_data["episode_seeded"].append(bool(seeded))
        # Feed the success window (best/ selection, early-stop) ONLY with
        # HONEST episodes — a seeded episode starts partway through, so its
        # goal_success is inflated and must not drive best/checkpoint or
        # early-stop decisions.
        if not seeded:
            self._goal_success_window.append(1.0 if goal_success else 0.0)
        self.episode_data["moving_avg_reward"].append(reward_sum)
        self.episode_data["moving_avg_length"].append(length)
        self.episode_data["episode_entropies"].append(
            self.model._get_entropy_coef(self.rollout_idx)
        )
        self._check_early_stopping()

    # ---------- update ----------

    def _update_from_rollout(self, data):
        """Segment-BPTT update. Data arrives as non-overlapping contiguous
        segments (S, N, L, ...). GAE is computed once over the FULL per-env
        timeline (T = S*L steps), then sliced back into segments for the PPO
        loss so the model trains on whole L-step sequences with real BPTT.
        """
        S, N, L = int(data["S"]), int(data["N"]), int(data["L"])
        T = S * L
        scale = float(self.reward_scaler.scale_factor())

        states = data["states"]  # (S, N, L, *input_shape)
        ram_states = data["ram_states"]  # (S, N, L, ram_obs_dim)
        actions = data["actions"]  # (S, N, L)
        rewards_seg = data["rewards"] * scale  # (S, N, L)
        dones_seg = data["dones"]  # (S, N, L)
        trunc_seg = data.get("truncated")  # (S, N, L) or None
        terminal_bootstrap_seg = data.get("terminal_bootstrap_values")
        old_log_probs = data["old_log_probs"]  # (S, N, L)
        old_base_probs = data["old_base_probs"]  # (S, N, L, A)
        old_behaviour_probs = data["old_behaviour_probs"]  # (S, N, L, A)
        mems = data["mems"]  # list of (S, N, mem_len, d_model)
        tail_mems = data["tail_mems"]  # list of (N, mem_len, d_model)

        # (S, N, L) time-scalar -> (T, N) contiguous per-env timeline.
        def to_time(x):
            return x.permute(0, 2, 1).reshape(T, N)

        # (T, N) -> (S, N, L) segments.
        def to_seg(x):
            return x.reshape(S, L, N).permute(0, 2, 1).contiguous()

        # Flatten segments to (S*N, L, ...) for batched forward passes.
        flat_states = states.reshape(S * N, *states.shape[2:])
        flat_ram_states = ram_states.reshape(S * N, *ram_states.shape[2:])
        flat_mems = [m.reshape(S * N, *m.shape[2:]) for m in mems]
        flat_dones = dones_seg.reshape(S * N, L)

        with torch.no_grad():
            # Per-position values from the SAME segment forward the critic
            # loss trains against (keeps GAE targets and value-clip baselines
            # self-consistent with the training view).
            _, values_flat, _ = self.model.actor_critic(
                flat_states,
                flat_ram_states,
                flat_mems,
                reset_after=flat_dones,
            )
            values_seg = values_flat.squeeze(-1).reshape(S, N, L)  # (S, N, L)
            values_time = to_time(values_seg)  # (T, N)

            # Bootstrap V(s_T) per env from the post-rollout state using the
            # memory produced by the final stored observation.
            tail_obs = data["last_next_obs"].unsqueeze(1)  # (N, 1, *input_shape)
            tail_ram = data["last_next_ram"].unsqueeze(1)  # (N, 1, ram_obs_dim)
            _, tail_v, _ = self.model.actor_critic(tail_obs, tail_ram, tail_mems)
            tail_values = tail_v[:, -1].squeeze(-1)  # (N,)

        rewards_time = to_time(rewards_seg)
        dones_time = to_time(dones_seg)
        trunc_time = to_time(trunc_seg) if trunc_seg is not None else None
        terminal_bootstrap_time = (
            to_time(terminal_bootstrap_seg)
            if terminal_bootstrap_seg is not None
            else None
        )

        returns_time, adv_time = self._per_env_gae(
            rewards_time,
            values_time,
            dones_time,
            tail_values,
            truncated=trunc_time,
            terminal_bootstrap_values=terminal_bootstrap_time,
        )

        # Advantage normalisation once over the whole rollout (default).
        norm_mode = self.config.get("advantage_normalisation", "rollout")
        if norm_mode == "rollout" and adv_time.numel() > 1:
            adv_time = (adv_time - adv_time.mean()) / (adv_time.std() + 1e-8)

        # Back to segments, then flatten to (S*N, L) for the minibatch loop.
        returns_seg = to_seg(returns_time)
        adv_seg = to_seg(adv_time)

        flat_data = {
            "states": flat_states,
            "ram_states": flat_ram_states,
            "actions": actions.reshape(S * N, L),
            "old_log_probs": old_log_probs.reshape(S * N, L),
            "old_base_probs": old_base_probs.reshape(S * N, L, self.action_size),
            "old_behaviour_probs": old_behaviour_probs.reshape(
                S * N, L, self.action_size
            ),
            "dones": flat_dones,
            "mems": flat_mems,
            "returns": returns_seg.reshape(S * N, L),
            "advantages": adv_seg.reshape(S * N, L),
            "old_values": values_seg.reshape(S * N, L).detach(),
        }

        return run_ppo_epochs(
            model=self.model,
            data=flat_data,
            step=self.rollout_idx,
            epochs=self.epochs,
            minibatch_size=self.minibatch_size,
            target_kl=self.target_kl,
        )

    def _per_env_gae(
        self,
        rewards,
        values,
        dones,
        tail_values,
        truncated=None,
        terminal_bootstrap_values=None,
    ):
        """GAE along the time axis, independently per env.

        rewards, values, dones: (W, N) tensors. tail_values: (N,).
        truncated: (W, N) bool tensor, or None. ``True`` marks a done that
        should bootstrap. Upstream (vec_env) this is set ONLY for a budget
        cut-off — a still-progressing episode that ran out of clock. A
        stagnation / battle-stagnation cut-off is deliberately reported as
        NOT truncated here, so it is treated as a zero-bootstrap terminal
        (see the ep_6395 diagnosis: bootstrapping a stuck state to the
        novelty-rich post-reset value made being stuck look valuable).
        Returns returns, advantages of shape (W, N).

        Truncation vs terminal: at a boundary we bootstrap V(s_{T+1}) only
        when the episode was *truncated* (budget cut-off, bootstrap flag) —
        a natural terminal (goal complete) and a stuck cut-off both have no
        value-worthy continuation, so their value is zeroed. All cases reset
        the GAE accumulator via ``not_done``, so advantages never leak across
        episode boundaries.

        When ``truncated`` is None we fall back to treating every done as a
        terminal (zero bootstrap) — the conservative classical behaviour.

        terminal_bootstrap_values: (W, N) values of the preserved terminal
        observations. Required for correct budget-truncation bootstrapping;
        post-reset values are never used at those boundaries.
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
                if terminal_bootstrap_values is not None:
                    next_value = torch.where(
                        truncated[t], terminal_bootstrap_values[t], next_value
                    )
                delta = rewards[t] + self.gamma * next_value * bootstrap[t] - values[t]
                gae = delta + self.gamma * self.gae_lambda * not_done[t] * gae
                advantages[t] = gae
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards)
            running = tail_values.clone()
            for t in reversed(range(W)):
                next_value = values[t + 1] if t + 1 < W else tail_values
                if terminal_bootstrap_values is not None:
                    next_value = torch.where(
                        truncated[t], terminal_bootstrap_values[t], next_value
                    )
                # At a boundary, carry the bootstrap value (V(next) if
                # truncated, else 0); otherwise carry the running return.
                carry = torch.where(dones[t], bootstrap[t] * next_value, running)
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
        rewards = self.episode_data["episode_rewards"]
        seeded = self.episode_data.get("episode_seeded") or []
        honest = [
            i
            for i in range(len(rewards))
            if i >= len(rewards) - 500 and (i >= len(seeded) or not bool(seeded[i]))
        ][-100:]

        def honest_mean(key):
            values = self.episode_data.get(key) or []
            selected = [float(values[i]) for i in honest if i < len(values)]
            return float(np.mean(selected)) if selected else None

        ent = self.episode_data.get("rollout_policy_entropy") or []
        kl = self.episode_data.get("rollout_approx_kl") or []
        clip = self.episode_data.get("rollout_clip_fraction") or []
        recent_seeded = seeded[-100:]
        honest_reward = honest_mean("episode_rewards")
        honest_maps = honest_mean("episode_unique_maps")
        honest_cells = honest_mean("episode_unique_cells")
        honest_flags = honest_mean("episode_flag_fires")
        postfix = {
            "ep": self.episode,
            "h_r": f"{honest_reward:.1f}" if honest_reward is not None else "n/a",
            "h_maps": f"{honest_maps:.1f}" if honest_maps is not None else "n/a",
            "h_cells": f"{honest_cells:.0f}" if honest_cells is not None else "n/a",
            "h_flags": f"{honest_flags:.1f}" if honest_flags is not None else "n/a",
            "archive": self.visit_archive.n_cells_seen(),
            "seed": f"{float(np.mean(recent_seeded)):.0%}" if recent_seeded else "0%",
            "pool": len(self.frontier_pool),
            "ent": f"{ent[-1]:.3f}" if ent else "n/a",
            "kl": f"{kl[-1]:.4f}" if kl else "n/a",
            "clip": f"{clip[-1]:.1%}" if clip else "n/a",
        }
        if self.probe_enabled:
            probe_sr = self.episode_data.get("probe_success_rate") or []
            postfix["probe_all"] = f"{probe_sr[-1]:.0%}" if probe_sr else "n/a"
            goal_rates = self.episode_data.get("probe_goal_rates") or []
            if goal_rates and goal_rates[-1]:
                deepest_label, deepest_rate = list(goal_rates[-1].items())[-1]
                postfix["probe_last"] = f"{deepest_label[:12]}:{deepest_rate:.0%}"
        pbar.set_postfix(postfix)

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
            flag_fire_steps=self.episode_data.get("episode_flag_fire_steps", None),
            probe_rollout_idx=self.episode_data.get("probe_rollout_idx", None),
            probe_goal_rates=self.episode_data.get("probe_goal_rates", None),
            long_probe_rollout_idx=self.episode_data.get(
                "long_probe_rollout_idx", None
            ),
            long_probe_goal_rates=self.episode_data.get("long_probe_goal_rates", None),
            checkpoint_recordings=self.checkpoint_recordings,
            frontier_pool_size=len(self.frontier_pool),
            approx_kls=self.episode_data.get("rollout_approx_kl", None),
            clip_fractions=self.episode_data.get("rollout_clip_fraction", None),
            durations=self.episode_data.get("rollout_duration_s", None),
            seeded=self.episode_data.get("episode_seeded", None),
            truncation_causes=self.episode_data.get("episode_truncation_cause", None),
            final_stagnation_clocks=self.episode_data.get(
                "episode_final_stagnation_clock", None
            ),
            pokedex_seen=self.episode_data.get("episode_pokedex_seen", None),
            pokedex_owned=self.episode_data.get("episode_pokedex_owned", None),
            event_counts={
                "trainer_wins": self.episode_data.get("episode_trainer_wins", []),
                "catches": self.episode_data.get("episode_catches", []),
                "new_species_catches": self.episode_data.get(
                    "episode_new_species_catches", []
                ),
                "duplicate_catches": self.episode_data.get(
                    "episode_duplicate_catches", []
                ),
            },
        )

    def save_model(self, path, vec_env=None):
        path = f"{path}"
        os.makedirs(path, exist_ok=True)
        self.model.save(path)

        # Pull each live worker's current domain-reward-balancer estimate
        # and merge them, so the checkpoint carries a non-degenerate
        # reward-scale state forward across the next restart (see
        # DomainRewardBalancer.merge_states and the one-time broadcast in
        # _train_loop). Best-effort: a query failure must not block saving
        # the actual model/checkpoint.
        if vec_env is not None:
            try:
                worker_states = vec_env.get_domain_balancer_states()
                self._domain_balancer_state = DomainRewardBalancer.merge_states(
                    worker_states
                )
            except Exception as e:
                print(f"[VecPPOAgent] Failed to collect domain balancer state: {e}")

        success_sr = (
            float(np.mean(self._goal_success_window))
            if len(self._goal_success_window) > 0
            else None
        )
        print(
            f"[VecPPOAgent] checkpoint @ total rollout {self.total_rollouts}: "
            f"success rate "
            f"{'n/a' if success_sr is None else f'{success_sr:.0%}'} "
            f"(last {len(self._goal_success_window)} eps)"
        )

        should_update_best = self._should_update_best()
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
            # Merged per-domain reward-scale estimate (see save_model above
            # and DomainRewardBalancer.merge_states) — restores reward
            # scaling across a restart instead of resetting it.
            "domain_balancer_state": self._domain_balancer_state,
            # The save-state files live under the run output; this manifest is
            # what makes them usable after auto-resume.
            "frontier_pool": list(self.frontier_pool),
            # Durable first-honest checkpoint deduplication and replay audit.
            "checkpoint_recordings": dict(self.checkpoint_recordings),
            "best_success_rate": float(self.best_success_rate),
            "best_discoveries": int(self.best_discoveries),
            "best_selection_reward": float(self.best_reward),
            "total_rollouts": int(self.total_rollouts),
        }
        torch.save(info, f"{path}/info.pth")

        if should_update_best:
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
          2. Free-play (n_goals<=0) → cumulative count of first-honest durable
             checkpoint recordings. Never raw reward
             (which battle-farms), and never per-episode coverage counts
             like unique_maps — re-walking the same known maps every
             episode maximises those, i.e. they select FOR the farming
             loop this metric exists to select against. Only genuine
             frontier pushes (a flag/map/species/level the RUN had never
             seen) move this count.
          3. Fallback (directed stage, no success yet) → mean reward, so a
             best/ checkpoint always exists for the next stage to load.
             (Free-play gets the same guarantee from best_discoveries
             starting at -1: the first full window always writes best/.)
        """
        ma_buf = self.episode_data["moving_avg_reward"]
        if len(ma_buf) < ma_buf.maxlen:
            return False
        current_ma = float(np.mean(ma_buf))

        win = self._goal_success_window
        min_eps = int(
            self.config.get(
                "best_success_min_episodes", self.config.get("best_success_window", 100)
            )
        )

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

        # 2. Free-play: first-honest durable checkpoints, not reward or
        # seeded archive discoveries.
        if self.n_goals <= 0:
            honest_checkpoint_count = len(self.checkpoint_recordings)
            if honest_checkpoint_count > self.best_discoveries:
                self.best_discoveries = honest_checkpoint_count
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
            configured_checkpoint = self.config.get("checkpoint")
            same_run_resume = False
            if configured_checkpoint:
                configured_root = os.path.abspath(str(configured_checkpoint))
                loaded_root = os.path.abspath(str(path))
                try:
                    same_run_resume = (
                        os.path.commonpath([configured_root, loaded_root])
                        == configured_root
                    )
                except ValueError:
                    same_run_resume = False
            if same_run_resume:
                legacy_success = info.get("success_rate")
                self.best_success_rate = float(
                    info.get(
                        "best_success_rate",
                        (
                            legacy_success
                            if legacy_success is not None
                            else self.best_success_rate
                        ),
                    )
                )
                self.best_discoveries = int(
                    info.get(
                        "best_discoveries",
                        len(info.get("checkpoint_recordings") or {}),
                    )
                )
                self.best_reward = float(
                    info.get(
                        "best_selection_reward",
                        info.get("best_reward", self.best_reward),
                    )
                )
            else:
                # A new curriculum stage has a different objective/reward
                # scale and must establish its own best/ high-water marks.
                self.best_success_rate = -1.0
                self.best_discoveries = -1
                self.best_reward = float("-inf")
            self.total_rollouts = int(info.get("total_rollouts", self.total_rollouts))
            # Clear per-stage state for the new curriculum stage so a runner
            # that chains stages in one process behaves like separate
            # processes: the entropy offset, the early-stop latch, and the
            # success window.
            self.model.set_entropy_offset(0)
            self._early_stopped = False
            self._entropy_coef = float(self.config.get("ppo_entropy_coef", 0.02))
            if bool(self.config.get("entropy_servo_enabled", True)):
                # Fresh stage: restart the servo from the configured
                # coefficient; it re-converges onto the band within a few
                # rollouts either way.
                self.model.set_entropy_coef(self._entropy_coef)
            else:
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

            # Restore the merged domain-reward-balancer estimate so reward
            # scaling doesn't snap back to its under-informed initial state
            # on this restart (see DomainRewardBalancer.merge_states and the
            # one-time broadcast in _train_loop).
            self._pending_domain_balancer_state = info.get("domain_balancer_state")

            # Flag-based entries are keyed by int flag number; threshold-
            # checkpoint entries (e.g. "pokedex_owned_6") are keyed by a
            # non-numeric string -- only numeric keys get cast back to int,
            # so a saved threshold entry doesn't crash int() on reload.
            self.checkpoint_recordings = {
                (int(k) if (isinstance(k, int) or str(k).lstrip("-").isdigit()) else k): dict(v)
                for k, v in (info.get("checkpoint_recordings") or {}).items()
            }
            for flag_num in self._recapture_checkpoints:
                self.checkpoint_recordings.pop(int(flag_num), None)

            # Restore only snapshots whose state files still exist. If the run
            # directory moved, retry by basename under its current snapshot dir.
            restored_captures = []
            snapshot_dir = self.config.get("goexplore_snapshot_dir") or os.path.join(
                self.config.get("output_base_dir", "."), "goexplore_snapshots"
            )
            raw_pool = info.get("frontier_pool") or []
            for raw_cap in raw_pool:
                cap = dict(raw_cap)
                path = cap.get("path", "")
                if not os.path.isfile(path):
                    candidate = os.path.join(snapshot_dir, os.path.basename(path))
                    if os.path.isfile(candidate):
                        cap["path"] = candidate
                    else:
                        self._frontier_restore_missing = True
                        continue
                restored_captures.append(cap)
            if restored_captures:
                self._ingest_frontier_captures(restored_captures)
                print(
                    f"[GoExplore] restored {len(self.frontier_pool)} frontier snapshots",
                    flush=True,
                )
            if len(restored_captures) < len(raw_pool):
                self._frontier_restore_missing = True
            self._loaded_checkpoint = True

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
                _ma_window = int(self.config.get("best_success_window", 100))
                _deque_maxlens = {
                    "moving_avg_reward": _ma_window,
                    "moving_avg_length": _ma_window,
                    "moving_avg_loss": _ma_window,
                    "buttons_pressed": 1000,
                }
                fresh = {
                    "episode_rewards": [],
                    "episode_lengths": [],
                    "episode_losses": [],
                    "moving_avg_reward": deque(maxlen=_ma_window),
                    "moving_avg_length": deque(maxlen=_ma_window),
                    "moving_avg_loss": deque(maxlen=_ma_window),
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
                    "episode_truncation_cause": [],
                    "episode_final_stagnation_clock": [],
                    "episode_pokedex_seen": [],
                    "episode_pokedex_owned": [],
                    "episode_trainer_wins": [],
                    "episode_catches": [],
                    "episode_new_species_catches": [],
                    "episode_duplicate_catches": [],
                    "episode_seeded": [],
                    "episode_reward_sources": [],
                    "episode_goal_success": [],
                    "episode_goal_fire_steps": [],
                    "episode_flag_fire_steps": [],
                    "discovery_log": [],
                    "rollout_policy_entropy": [],
                    "rollout_behaviour_entropy": [],
                    "rollout_entropy_coef": [],
                    "rollout_lr": [],
                    "rollout_approx_kl": [],
                    "rollout_base_post_kl": [],
                    "rollout_behaviour_post_kl": [],
                    "rollout_base_replay_kl": [],
                    "rollout_behaviour_replay_kl": [],
                    "rollout_action_log_prob_abs_max": [],
                    "rollout_action_log_prob_abs_mean": [],
                    "rollout_replay_rejected": [],
                    "rollout_kl_rejected": [],
                    "rollout_clip_fraction": [],
                    "rollout_actor_loss": [],
                    "rollout_critic_loss": [],
                    "rollout_duration_s": [],
                    "probe_success_rate": [],
                    "probe_rollout_idx": [],
                    "probe_goal_type_rates": [],
                    "probe_goal_rates": [],
                    "probe_fire_steps": [],
                    "assisted_probe_success_rate": [],
                    "assisted_probe_rollout_idx": [],
                    "assisted_probe_goal_type_rates": [],
                    "assisted_probe_goal_rates": [],
                    "assisted_probe_fire_steps": [],
                    "long_probe_success_rate": [],
                    "long_probe_rollout_idx": [],
                    "long_probe_goal_type_rates": [],
                    "long_probe_goal_rates": [],
                    "long_probe_fire_steps": [],
                    "assisted_long_probe_success_rate": [],
                    "assisted_long_probe_rollout_idx": [],
                    "assisted_long_probe_goal_type_rates": [],
                    "assisted_long_probe_goal_rates": [],
                    "assisted_long_probe_fire_steps": [],
                }
                for key, value in loaded_episode_data.items():
                    if key not in fresh:
                        continue
                    if key in _deque_maxlens:
                        # ALWAYS rebuild with THIS stage's configured
                        # window, even if the loaded value already
                        # unpickled as a deque — otherwise a loaded deque
                        # object silently keeps the PREVIOUS stage's
                        # maxlen, and changing best_success_window between
                        # stages (e.g. a short long-episode stage that
                        # needs a smaller window to ever write best/) has
                        # no effect.
                        fresh[key] = deque(value, maxlen=_deque_maxlens[key])
                    else:
                        fresh[key] = value
                self.episode_data = fresh
                self._discovery_keys_seen = set()
                for discovery in self.episode_data.get("discovery_log", []):
                    raw_key = discovery.get("key")
                    stable_key = (
                        tuple(raw_key) if isinstance(raw_key, list) else raw_key
                    )
                    self._discovery_keys_seen.add(
                        (str(discovery.get("type")), stable_key)
                    )
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
                "goals_total": len(self.episode_data.get("episode_goals_total", [])),
                "goals_made": len(self.episode_data.get("episode_goals_made", [])),
                "goals_target": len(self.episode_data.get("episode_goals_target", [])),
                "flag_fires": len(self.episode_data.get("episode_flag_fires", [])),
                "unique_cells": len(self.episode_data.get("episode_unique_cells", [])),
                "unique_maps": len(self.episode_data.get("episode_unique_maps", [])),
                "archive_size": len(self.episode_data.get("episode_archive_size", [])),
                "reward_sources": len(
                    self.episode_data.get("episode_reward_sources", [])
                ),
                "rollouts": len(self.episode_data.get("rollout_policy_entropy", [])),
            }
        except FileNotFoundError:
            print(f"No checkpoint found at {path}, starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch.")
