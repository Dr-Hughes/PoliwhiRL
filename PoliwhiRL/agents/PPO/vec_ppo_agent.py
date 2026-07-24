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
)
from PoliwhiRL.environment.gym_env import STAGNATION_CLOCK_RAM_IDX
from PoliwhiRL.environment.vec_env import _load_actions_file, write_actions_file
from PoliwhiRL.environment.visit_archive import VisitArchive
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

        # Behaviour-time stuck-action temperature (reinstated 2026-07-19). It
        # was removed in the 2026-07-16 rebuild as "undirected noise", but the
        # ep_6395 diagnosis showed the real failure it addressed: once a pocket
        # is locally exhausted the reward goes flat, so there is no gradient to
        # prefer a productive move over mashing into a wall, and a peaked policy
        # then repeats one action until the watchdog cuts the episode. This
        # ramps the SAMPLING temperature per-env with the stagnation clock (the
        # fraction of the stall budget consumed, read from the RAM vector), so
        # extra exploration is injected ONLY where and when novelty has stalled
        # — not globally. The action's stored log-prob is taken under this
        # tempered behaviour distribution, so PPO's importance ratio corrects
        # for the injection at update time. 0 => temperature pinned at 1.0 =>
        # identical to plain sampling.
        self.stuck_temperature = float(config.get("stuck_action_temperature", 0.0))
        self.stuck_temperature_max = float(
            config.get("stuck_action_temperature_max", 3.0)
        )
        # Fraction of the stagnation budget consumed before the ramp starts (a
        # brief stall is normal); temperature climbs linearly from 1.0 here to
        # stuck_temperature_max as the clock reaches 1.0 (watchdog cut-off).
        self.stuck_temperature_threshold = float(
            config.get("stuck_action_temperature_threshold", 0.25)
        )
        self._stag_clock_idx = STAGNATION_CLOCK_RAM_IDX

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
        self._frontier_index = {}  # (bank, num) -> rarest capture dict
        self.frontier_pool = []  # curated list, rarest first
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
                self.save_model(self.config["checkpoint"])

            if self._check_early_stopping():
                break

        if (
            self.config.get("save_checkpoint", True)
            and self.config.get("checkpoint") is not None
        ):
            self.save_model(self.config["checkpoint"])

    def _apply_stuck_temperature(self, action_probs, ram_tensor, action_mask):
        """Per-env temperature-scale the sampling distribution by stall depth.

        ``action_probs`` is ``(N, A)`` from the (masked) policy. For each env
        we read the ``stagnation_clock`` feature (fraction of the stagnation-
        truncation budget consumed, already in [0, 1]) from the last RAM
        frame and set a temperature that ramps from 1.0 at
        ``stuck_temperature_threshold`` up to ``stuck_temperature_max`` as the
        clock reaches 1.0 (where the watchdog truncates). Softmax temperature
        scaling is ``normalize(probs ** (1/T))``; we then re-apply the (last-
        frame) action mask and renormalise so tempering can never resurrect a
        masked action (a masked prob is a clamped ~1e-10 that ``**(1/T)``
        would otherwise inflate). Returns the behaviour distribution to sample
        from. No-op returning ``action_probs`` unchanged when
        ``stuck_action_temperature <= 0``.
        """
        if self.stuck_temperature <= 0.0:
            return action_probs
        clock = ram_tensor[:, -1, self._stag_clock_idx].clamp(0.0, 1.0)  # (N,)
        thr = min(max(self.stuck_temperature_threshold, 0.0), 0.999)
        # 0 at/below the threshold, ramping to 1.0 as the clock hits 1.0.
        excess = (clock - thr).clamp(min=0.0) / (1.0 - thr)
        temp = (
            (1.0 + self.stuck_temperature * excess)
            .clamp(1.0, self.stuck_temperature_max)
            .unsqueeze(1)
        )  # (N,1)
        tempered = action_probs.pow(1.0 / temp)
        if action_mask is not None:
            # The rollout mask is per-position (N, L, A); temper the last frame.
            m = action_mask[:, -1] if action_mask.dim() == 3 else action_mask
            tempered = tempered * m
        denom = tempered.sum(dim=1, keepdim=True).clamp(min=1e-10)
        return tempered / denom

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
                action_probs = torch.nan_to_num(
                    action_probs, nan=0.0, posinf=0.0, neginf=0.0
                )
                action_probs = torch.clamp(action_probs, 1e-10, 1.0)
                # Diagnostic: ACTUAL policy entropy at behaviour time.
                step_entropy = (
                    -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1).mean()
                )
                self._rollout_entropy_sum += float(step_entropy)
                self._rollout_entropy_n += 1
                # Behaviour-time stuck-exploration: temper the sampling
                # distribution per-env when the stagnation clock is high, so a
                # stalled env actually TRIES different actions. The action is
                # drawn from — and its stored log-prob taken under — this
                # behaviour distribution, so PPO's importance ratio is correct
                # off-policy. The entropy diagnostic above stays on the
                # UN-tempered policy so it tracks the real policy, not the
                # injected noise.
                sample_probs = self._apply_stuck_temperature(
                    action_probs, ram_tensor, action_mask
                )
                actions_t = torch.multinomial(sample_probs, 1).squeeze(1)
                log_probs_t = torch.log(
                    sample_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1) + 1e-10
                )

            actions = actions_t.cpu().numpy().astype(np.int64)
            log_probs_np = log_probs_t.cpu().numpy().astype(np.float32)

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

            checkpoint_jobs = []
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
                    if self.visit_archive.merge_milestones(
                        **info.get(
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
                    ):
                        self._archive_dirty = True
                    goals_total = (
                        int(info.get("n_flag", 0))
                        + int(info.get("n_pokedex", 0))
                        + int(info.get("n_map", 0))
                    )
                    n_target = int(info.get("n_target", self.n_goals))
                    running_seeded = bool(
                        self.goexplore_enabled
                        and self.env_is_seeded is not None
                        and self.env_is_seeded[i]
                    )
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

            mems = new_mems

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
        if kind == "flag":
            return ("flag", int(cap["flag"]))
        if kind == "party":
            return ("party", int(cap["party_size"]))
        if kind == "cell":
            return self.visit_archive.cell_key(
                int(cap["bank"]), int(cap["num"]), int(cap["x"]), int(cap["y"])
            )
        return (int(cap["bank"]), int(cap["num"]))

    def _live_count(self, cap):
        """Current run-wide rarity of a snapshot's frontier, judged LIVE (not
        by the stale capture-time count): run-wide fire count of the event flag
        for a flag capture, cell count in "cell" mode, else map count. Live
        ranking lets early start-region snapshots sink and be evicted while the
        genuine frontier (rarely-visited cell / rarely-fired flag) stays on
        top and is seeded most (weight 1/(1+count))."""
        kind = cap.get("kind", self.goexplore_granularity)
        if kind == "flag":
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
            # Keep the newest snapshot for the location (all are frontier frames,
            # so effectively equivalent; newest keeps the freshest file path).
            self._frontier_index[key] = c
        self.frontier_pool = sorted(
            self._frontier_index.values(), key=self._live_count
        )[: self.goexplore_pool_size]
        if grew:
            print(
                f"[GoExplore] frontier locations={len(self._frontier_index)} "
                f"(pool={len(self.frontier_pool)}, mode={self.goexplore_granularity}); "
                f"pool live-counts: "
                f"{sorted(self._live_count(c) for c in self.frontier_pool)[:12]}",
                flush=True,
            )

    def _pick_frontier_snapshot(self):
        """Sample a frontier snapshot, biased toward rarer locations by LIVE
        count (weight 1/(1+live_count)), so seeding concentrates on the true
        frontier while still occasionally revisiting nearer, better-learned
        ground."""
        if not self.frontier_pool:
            return None
        weights = np.array(
            [1.0 / (1.0 + self._live_count(c)) for c in self.frontier_pool],
            dtype=np.float64,
        )
        total = weights.sum()
        if total <= 0:
            return self.frontier_pool[0]
        idx = int(np.random.choice(len(self.frontier_pool), p=weights / total))
        return self.frontier_pool[idx]

    def _goexplore_cycle(self, vec_env, env_idx):
        """Choose env_idx's NEXT start state (Go-Explore). Probe workers
        (env 0 .. probe_workers-1) always train from-scratch and are left
        untouched. Other workers restart from a frontier snapshot with
        probability seed_fraction, else return to the true start. Takes effect
        on the worker's next auto-reset; the seeded flag is queued in
        env_is_seeded_pending and promoted alongside env_state_indices."""
        seeded = False
        if env_idx >= self.goexplore_probe_workers:
            if self.frontier_pool and np.random.rand() < self.goexplore_seed_fraction:
                snap = self._pick_frontier_snapshot()
                if snap is not None:
                    try:
                        vec_env.set_env_state(env_idx, snap["path"])
                        seeded = True
                    except Exception as e:
                        print(f"[GoExplore] seed failed env {env_idx}: {e}")
            if not seeded and self.env_is_seeded[env_idx]:
                # Was running from a seed; hand it back to the true start.
                try:
                    vec_env.set_env_state(env_idx, self._true_start_path)
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
        self._probe_env = self._run_probe_pass(
            env=self._probe_env,
            probe_config=self._probe_config,
            episode_length=self.probe_episode_length,
            n_episodes=self.probe_episodes,
            keys=(
                "probe_success_rate",
                "probe_rollout_idx",
                "probe_goal_type_rates",
                "probe_goal_rates",
                "probe_fire_steps",
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
        self._long_probe_env = self._run_probe_pass(
            env=self._long_probe_env,
            probe_config=self._long_probe_config,
            episode_length=self.long_probe_episode_length,
            n_episodes=self.long_probe_episodes,
            keys=(
                "long_probe_success_rate",
                "long_probe_rollout_idx",
                "long_probe_goal_type_rates",
                "long_probe_goal_rates",
                "long_probe_fire_steps",
            ),
            label=self.probe_label + "_long",
        )

    def _run_probe_pass(
        self, env, probe_config, episode_length, n_episodes, keys, label
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
                    state[None], ram[None], mems
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
            vec_env.enable_record(folder, use_episode_number=False, env_idx=0)
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
            completed = 0
            observed_flag = False
            for action in actions:
                _obs, _reward, done, _truncated = env.step(int(action))
                completed += 1
                observed_flag = any(
                    int(flag) == int(entry["flag"])
                    for flag, _step in env.reward_calculator.flag_fire_step_log
                )
                if done:
                    break
            entry["replayed_steps"] = completed
            entry["checkpoint_observed"] = observed_flag
            entry["status"] = (
                "recorded"
                if completed == len(actions) and observed_flag
                else "checkpoint_not_reproduced"
            )
            print(
                f"[Checkpoint] {entry['status']} first honest '{entry['title']}' "
                f"({completed} steps) -> {entry['output_folder']}",
                flush=True,
            )
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
        old_log_probs = data["old_log_probs"]  # (S, N, L)
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

        with torch.no_grad():
            # Per-position values from the SAME segment forward the critic
            # loss trains against (keeps GAE targets and value-clip baselines
            # self-consistent with the training view).
            _, values_flat, _ = self.model.actor_critic(
                flat_states, flat_ram_states, flat_mems
            )
            values_seg = values_flat.squeeze(-1).reshape(S, N, L)  # (S, N, L)
            values_time = to_time(values_seg)  # (T, N)

            # Bootstrap V(s_T) per env from the post-rollout state, using the
            # memory going into the final stored step.
            tail_obs = data["last_next_obs"].unsqueeze(1)  # (N, 1, *input_shape)
            tail_ram = data["last_next_ram"].unsqueeze(1)  # (N, 1, ram_obs_dim)
            _, tail_v, _ = self.model.actor_critic(tail_obs, tail_ram, tail_mems)
            tail_values = tail_v[:, -1].squeeze(-1)  # (N,)

        rewards_time = to_time(rewards_seg)
        dones_time = to_time(dones_seg)
        trunc_time = to_time(trunc_seg) if trunc_seg is not None else None

        returns_time, adv_time = self._per_env_gae(
            rewards_time, values_time, dones_time, tail_values, truncated=trunc_time
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

    def _per_env_gae(self, rewards, values, dones, tail_values, truncated=None):
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

            self.checkpoint_recordings = {
                int(k): dict(v)
                for k, v in (info.get("checkpoint_recordings") or {}).items()
            }

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
                    "episode_seeded": [],
                    "episode_reward_sources": [],
                    "episode_goal_success": [],
                    "episode_goal_fire_steps": [],
                    "episode_flag_fire_steps": [],
                    "discovery_log": [],
                    "rollout_policy_entropy": [],
                    "rollout_entropy_coef": [],
                    "rollout_lr": [],
                    "rollout_approx_kl": [],
                    "rollout_clip_fraction": [],
                    "rollout_actor_loss": [],
                    "rollout_critic_loss": [],
                    "rollout_duration_s": [],
                    "probe_success_rate": [],
                    "probe_rollout_idx": [],
                    "probe_goal_type_rates": [],
                    "probe_goal_rates": [],
                    "probe_fire_steps": [],
                    "long_probe_success_rate": [],
                    "long_probe_rollout_idx": [],
                    "long_probe_goal_type_rates": [],
                    "long_probe_goal_rates": [],
                    "long_probe_fire_steps": [],
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
