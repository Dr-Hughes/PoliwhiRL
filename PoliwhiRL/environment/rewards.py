# -*- coding: utf-8 -*-
"""Per-step reward calculator — pure exploration.

Reward sources:

  Frontier novelty (run-wide decaying):
    r += bonus / (effective_count + 1)    # effective_count = min(visits, floor)
    Pays on first entry to a cell per episode. With frontier_novelty_count_floor>0
    a saturated cell still pays bonus/(floor+1) — preventing total gradient starvation.

  New-map discovery (run-wide decaying):
    r += new_map_reward / (map_visit_count + 1)  # depletes as map is re-entered

  Step penalty (constant):
    r += step_penalty  # applied every valid step; keeps gradient alive when novelty
                       # is exhausted; default 0.0 (off)

  Whiteout:
    r += whiteout_penalty  # one-shot on party HP -> 0

Frontier novelty uses run-wide visit counts backed by ``visit_archive``.
``frontier_novelty_count_floor`` (default 0) bounds the decay so a
frequently-visited cell still pays bonus/(floor+1) each episode when set.

New-map reward decays with the run-wide entry count so map-bouncing stops
paying after a handful of entries while genuine frontier maps still pay
on first discovery.
"""
import numpy as np
from .goals import GoalsManager
from .visit_archive import VisitArchive

_VALID_BATTLE_TYPES = (0, 1, 2)
_VALID_PLAYER_STATES = (0, 1, 2, 4)


def is_ram_state_valid(env_vars):
    if int(env_vars.get("battle_type", 0)) not in _VALID_BATTLE_TYPES:
        return False
    if int(env_vars.get("player_state", 0)) not in _VALID_PLAYER_STATES:
        return False
    if (
        int(env_vars.get("X", 0)) == 0
        and int(env_vars.get("Y", 0)) == 0
        and int(env_vars.get("map_num", 0)) == 0
        and int(env_vars.get("map_bank", 0)) == 0
    ):
        return False
    return True


class Rewards:
    def __init__(self, config, visit_archive=None):
        self.goals = GoalsManager(config)
        # The visit archive must outlive any individual Rewards instance
        # (env.reset() re-instantiates Rewards every episode). The owning
        # PyBoyEnvironment passes its archive in here; if no archive is
        # supplied — only happens in unit-test fixtures — we create a
        # private one whose counts will reset with this Rewards object.
        self.visit_archive = visit_archive if visit_archive is not None else VisitArchive()

        self.max_steps = config["episode_length"]

        # Run-wide decaying map-discovery reward. Pays new_map_reward /
        # (global_map_entry_count + 1) on first entry to a map each episode.
        # Decays across the run so map-bouncing stops paying, but genuine
        # frontier maps still pay fully on first discovery.
        self.new_map_reward = config.get("new_map_reward", 50)
        self.frontier_novelty_bonus = config.get("frontier_novelty_bonus", 25.0)
        # Optional floor on the visit count denominator. 0 (default) means
        # fully depleting: a saturated cell pays near-zero. A positive value
        # leaves a renewable per-cell trickle of bonus/(floor+1).
        self.frontier_novelty_count_floor = config.get(
            "frontier_novelty_count_floor", 0
        )
        self.whiteout_penalty = config.get("whiteout_penalty", -20.0)
        # Applied every valid step. Negative value keeps gradient non-zero
        # after the novelty landscape depletes. Default 0.0 (disabled).
        self.step_penalty = float(config.get("step_penalty", 0.0))

        self.clip = config.get("reward_clip", 1000)

        # Optional rounding of the per-step reward to this many decimal
        # places — keeps logs / PNG filenames legible. Default 2.
        self.reward_round_dp = config.get("reward_round_dp", 2)

        # Per-episode novelty / progress trackers.
        self._novel_cells_this_episode = set()
        # Pending archive records: cells/maps GENUINELY visited this episode.
        # The worker reports these in terminal_info at episode end; the AGENT
        # merges them into the canonical archive and broadcasts the table back.
        self._cells_to_record = set()
        self._maps_to_record = set()
        # Diagnostic: the step number at which each goal rung fired this episode.
        self.goal_fire_steps = []
        self._prev_rung = 0
        self.explored_tiles = set()
        # Per-episode. Cross-episode novelty is in visit_archive.
        self.explored_maps = set()

        # Per-source reward accumulators for diagnostic logging.
        self._episode_breakdown = {
            "frontier": 0.0,
            "new_map": 0.0,
            "step_penalty": 0.0,
            "whiteout": 0.0,
        }

        # Ordered list of unique (map_bank, map_num) pairs entered during
        # the training portion of this episode.
        self._recent_maps_n = int(config.get("ram_recent_maps_n", 6))
        self._recent_maps_list = []

        self.done = False
        self.truncated = False
        self.last_action = None
        self.steps = 0
        self.cumulative_reward = 0

        # Whiteout gating.
        self._prev_party_size = None
        self._prev_party_hp = None
        self.whiteouts = 0

    # ------------------------------------------------------------------ #
    # Properties (delegate to GoalsManager)                               #
    # ------------------------------------------------------------------ #

    @property
    def N_goals(self):
        return self.goals.N_goals

    @N_goals.setter
    def N_goals(self, value):
        self.goals.N_goals = value

    @property
    def flag_goals_completed(self):
        return self.goals.flag_goals_completed

    @property
    def map_goals_completed(self):
        return self.goals.map_goals_completed

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def start_new_episode(self):
        self.done = False
        self.truncated = False
        self.last_action = None
        self.steps = 0
        self.cumulative_reward = 0
        self._prev_party_size = None
        self._prev_party_hp = None
        self.whiteouts = 0
        self._novel_cells_this_episode = set()
        self._cells_to_record = set()
        self._maps_to_record = set()
        self.goal_fire_steps = []
        self._prev_rung = 0
        self.explored_maps = set()
        self._recent_maps_list = []
        for key in self._episode_breakdown:
            self._episode_breakdown[key] = 0.0
        self.goals.reset_episode_trackers()

    def seed_explored_maps(self, map_keys):
        """Pre-fill the per-episode ``explored_maps`` set with maps the
        action-replay walked through. Called by ``PyBoyEnvironment.replay_actions``
        after ``start_new_episode`` so the training segment's first step on
        an already-seen map doesn't fire a spurious new_map bonus."""
        for key in map_keys:
            self.explored_maps.add((int(key[0]), int(key[1])))

    def get_episode_breakdown(self):
        """Per-source reward totals for the current episode."""
        return dict(self._episode_breakdown)

    # ------------------------------------------------------------------ #
    # Main reward calculation                                             #
    # ------------------------------------------------------------------ #

    def calculate_reward(self, env_vars, button_press):
        self.steps += 1

        if is_ram_state_valid(env_vars):
            self.goals.note_map_visit(int(env_vars["map_bank"]), int(env_vars["map_num"]))
            self.goals.check_maps_visited_goals()
            self.goals.check_map_goals(int(env_vars["map_bank"]), int(env_vars["map_num"]))
            # Flag goals: advance GoalsManager counter for metrics only.
            # No reward is paid — pure exploration design has no flag reward.
            self.goals.check_flag_goals(env_vars["story_flags"])

            r_map = self._new_map_bonus(env_vars)
            self._episode_breakdown["new_map"] += r_map

            r_front = self._frontier_novelty_bonus(env_vars)
            self._episode_breakdown["frontier"] += r_front

            r_wo = self._check_whiteout(env_vars)
            self._episode_breakdown["whiteout"] += r_wo

            self._episode_breakdown["step_penalty"] += self.step_penalty

            total = r_map + r_front + r_wo + self.step_penalty

            # Log the step at which each map goal rung fired (diagnostic).
            rung = self.n_map_goals_completed()
            if rung > self._prev_rung:
                self.goal_fire_steps.extend(
                    [int(self.steps)] * (rung - self._prev_rung)
                )
                self._prev_rung = rung
        else:
            total = 0.0

        self.last_action = button_press

        if self.steps > self.max_steps:
            # Budget cut-off: truncated, not a natural terminal.
            self.done = True
            self.truncated = True

        if self.reward_round_dp is not None:
            total = round(float(total), int(self.reward_round_dp))

        self.cumulative_reward += total
        return np.clip(total, -self.clip, self.clip).astype(np.float32), self.done

    # ------------------------------------------------------------------ #
    # Reward subroutines                                                  #
    # ------------------------------------------------------------------ #

    def _check_whiteout(self, env_vars):
        """One-shot whiteout penalty on the ``prev_total_hp > 0 -> cur == 0``
        transition. Closes the "free Pokémon Center teleport" loophole."""
        party_size, _, party_hp, _ = env_vars["party_info"]
        if party_size <= 0:
            return 0
        # Reset baseline on party-size change (catching/joining a Pokémon
        # changes total HP without anyone taking damage).
        if (
            self._prev_party_size is not None
            and party_size != self._prev_party_size
        ):
            self._prev_party_size = party_size
            self._prev_party_hp = party_hp
            return 0
        if self._prev_party_hp is None:
            self._prev_party_hp = party_hp
            self._prev_party_size = party_size
            return 0
        if env_vars.get("script_active", False):
            self._prev_party_hp = party_hp
            return 0
        reward = 0.0
        if party_hp == 0 and self._prev_party_hp > 0:
            reward += self.whiteout_penalty
            self.whiteouts += 1
        self._prev_party_hp = party_hp
        return reward

    def _new_map_bonus(self, env_vars):
        """Run-wide decaying bonus for entering a map not yet visited this episode.

        Pays ``new_map_reward / (global_count + 1)`` on first entry to a
        (map_bank, map_num) this episode. The global_count is the run-wide
        number of training episodes that have entered this map, so the bonus
        depletes as the map becomes familiar while genuine frontier maps
        always pay fully on first discovery.

        Also updates ``explored_tiles`` for the observation layer.
        Skipped during scripted overlays where the player position is stale.
        """
        map_key = (int(env_vars["map_bank"]), int(env_vars["map_num"]))
        loc = (env_vars["X"], env_vars["Y"], map_key[0], map_key[1])
        self.explored_tiles.add(loc)
        if not self.new_map_reward:
            return 0
        if map_key in self.explored_maps:
            return 0
        if env_vars.get("script_active", False):
            self.explored_maps.add(map_key)
            return 0
        self.explored_maps.add(map_key)
        global_count = self.visit_archive.map_count(map_key[0], map_key[1])
        if map_key not in self._recent_maps_list:
            self._recent_maps_list.append(map_key)
        self._maps_to_record.add(map_key)
        return float(self.new_map_reward) / (global_count + 1)

    def _frontier_novelty_bonus(self, env_vars):
        """Persistent count-based cell novelty: pays
        ``bonus / (effective_count + 1)`` where ``count`` is the run-wide
        visit count for the quantised cell (the ``visit_archive`` ledger that
        persists across episodes).

        Because the count persists, the intrinsic landscape depletes over
        training: a region the policy has already explored pays near-zero,
        while a never-visited cell still pays full bonus. The novelty
        gradient therefore always points at the receding frontier.

        Each cell pays at most once per episode (``_novel_cells_this_episode``)
        so wiggling can't farm within an episode.

        Gated on ``script_active`` — during cutscenes / menus the player
        position is stale.
        """
        if self.frontier_novelty_bonus <= 0:
            return 0
        if env_vars.get("script_active", False):
            return 0
        mb, mn = env_vars["map_bank"], env_vars["map_num"]
        x, y = env_vars["X"], env_vars["Y"]
        cell = self.visit_archive.cell_key(mb, mn, x, y)
        if cell in self._novel_cells_this_episode:
            return 0
        self._novel_cells_this_episode.add(cell)
        prior_count = self.visit_archive.count(mb, mn, x, y)
        # Queue the genuine training visit for the agent's canonical merge.
        self._cells_to_record.add(cell)
        effective_count = prior_count
        if self.frontier_novelty_count_floor is not None and self.frontier_novelty_count_floor > 0:
            effective_count = min(prior_count, self.frontier_novelty_count_floor)
        return self.frontier_novelty_bonus / (effective_count + 1)

    # ------------------------------------------------------------------ #
    # Progress queries (used by RAM observation builder & plotting)       #
    # ------------------------------------------------------------------ #

    def get_progress(self):
        return {
            "Steps": self.steps,
            "Goals Reached": self.goals.N_goals,
            "Explored Tiles": len(self.explored_tiles),
            "Maps Visited (episode)": len(self.goals._maps_seen_this_episode),
            "Banks Visited (episode)": len(
                {bank for (bank, _) in self.goals._maps_seen_this_episode}
            ),
            "Flag Fires": self.goals.flag_goals_completed,
            "Whiteouts": self.whiteouts,
            "Cumulative Reward": self.cumulative_reward,
        }

    def explored_tile_count(self):
        return len(self.explored_tiles)

    def recent_maps_visited(self):
        """Last _recent_maps_n unique maps entered during the training portion
        of this episode, in visit order (oldest first). Padded with (0, 0)
        at the front when fewer maps have been entered, so the RAM vector
        index is stable across steps and episodes."""
        recent = self._recent_maps_list[-self._recent_maps_n:]
        pad = self._recent_maps_n - len(recent)
        return [(0, 0)] * pad + list(recent)

    def n_location_goals_completed(self):
        return 0

    def n_pokedex_goals_completed(self):
        return self.goals.n_pokedex_goals_completed()

    def n_level_goals_completed(self):
        return self.goals.n_level_goals_completed()

    def n_xp_goals_completed(self):
        return self.goals.n_xp_goals_completed()

    def n_flag_goals_completed(self):
        return self.goals.n_flag_goals_completed()

    def n_map_goals_completed(self):
        return self.goals.n_map_goals_completed()
