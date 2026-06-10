# -*- coding: utf-8 -*-
"""Per-step reward calculator.

    Extrinsic (directed milestones — the curriculum spine):
      r += 500 · flag_fires                           # binary story milestone
         + 150 · Δpokedex_owned                       # binary per species caught
         +  10 · Δpokedex_seen (first sighting only)  # one-shot per species seen
         +   5 · Δkey_items_count                     # picking up Pokéballs etc.
         + 250 · map_goal_reached                     # reached a configured town
         − 100 · whiteout                             # hard fail
    Intrinsic (count-based exploration + capped battle outcome):
      r +=  3 / (global_visits + 1) · new_cell        # PERSISTENT frontier
         + 50 / (global_map_entries + 1) · new_map    # decaying first-discovery
         +  3 · first_battle_entry_per_map · decay    # engage signal
         +  8 · first_battle_won_per_map · decay      # WIN, not damage
         + (0 · Δenemy_hp · decay)                    # damage OFF by default
         + 10 · Δparty_total_level                    # minor leveling
      with total battle reward clamped to battle_reward_episode_cap/episode.
    clipped to ±reward_clip.

Frontier novelty is PERSISTENT (count-based, backed by ``visit_archive``):
the denominator is the run-wide visit count for the quantised cell, so a
region the policy has already worked to death pays near-zero while
genuinely new ground still pays full. This makes "re-walking known cells
yields nothing; the only novelty left is past the frontier" an emergent,
area-invariant pressure — no hand-authored breadcrumb per stage. Each cell
is still paid at most once per episode (``_novel_cells_this_episode``) so
wiggling can't farm, and the count is recorded only on genuine training
steps (NOT during replay), so the evaluator's replay can't drain it.
The other non-stationary term is the ``new_map`` first-discovery bonus.

Battle reward is win-based: a small first-entry-per-map engagement bonus
plus a per-map win bonus (enemy HP→0, player survives, not a catch). Raw
damage is off by default (it was the farm vector). Per-map decay
``1/(1 + battle_decay_coef · n)`` still applies, and a hard per-episode cap
(``battle_reward_episode_cap``) is the decay-independent backstop.

Maps the replay visited are seeded into ``explored_maps`` before the
training episode starts (see ``Rewards.seed_explored_maps``) so the
``new_map`` term cannot pay for maps the replay already explored.
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

        # Event-flag progress reward. OFF by default: the flag-detection
        # system is WIP and never fired in the curriculum runs, so a dormant
        # 500-point reward would ambush the economy the moment it starts
        # firing. Enable explicitly per-config once flags are wired + tested.
        self.flag_progress_reward = config.get("flag_progress_reward", 0)
        self.pokedex_owned_reward = config.get("pokedex_owned_reward", 150)
        self.pokedex_first_sight_reward = config.get("pokedex_first_sight_reward", 10)
        self.key_item_pickup_reward = config.get("key_item_pickup_reward", 5)
        # Legacy flat-per-episode new-map bonus. Retained for back-compat /
        # config parsing but no longer the primary signal — the smoothly
        # decaying first-discovery bonus below replaces it. Default 0 so old
        # configs that still set it keep working, but fresh configs use
        # new_map_first_discovery_reward.
        self.new_map_reward = config.get("new_map_reward", 0)
        # One-time-ish discovery bonus: pays new_map_first_discovery_reward /
        # (run-wide entry count + 1) the first time a map is entered each
        # episode. Decays globally so map-bouncing stops paying within a few
        # entries; the stationary episodic frontier carries the bulk of the
        # "new areas are good" drive. Kept small and rare (see AGENTS.md /
        # plan: it is the only non-stationary reward term).
        self.new_map_first_discovery_reward = config.get(
            "new_map_first_discovery_reward", 50
        )
        # Reward for reaching a configured map-reach goal ("get to town X").
        # A coarse terminal milestone — sized comparably to a flag fire so
        # the policy treats "arrive at the target town" as a real objective.
        self.map_goal_reward = config.get("map_goal_reward", 250)
        self.frontier_novelty_bonus = config.get("frontier_novelty_bonus", 25.0)
        # Cap on visit count used in the novelty denominator so the signal
        # never drains to zero. A floor of 20 means a saturated cell still
        # pays bonus/21 ≈ 1.2 (config default: 25). Set to None or ≤ 0
        # to disable the floor (unbounded decay, legacy behaviour).
        self.frontier_novelty_count_floor = config.get(
            "frontier_novelty_count_floor", 20
        )
        # Hard ceiling on the TOTAL intrinsic reward (new_map + frontier +
        # battle + level) paid in a single episode. The dense intrinsic
        # stream is unbounded in episode length, so on long routes it can
        # out-pay the one-time milestone reward and the policy abandons the
        # goal to farm exploration. Capping the whole stream below a single
        # milestone payout removes the farming optimum. Sized as a fraction
        # of map_goal_reward; <= 0 disables the cap.
        self.intrinsic_reward_episode_cap = config.get(
            "intrinsic_reward_episode_cap", 100.0
        )
        # Per-step living cost (negative). Makes loitering net-negative so the
        # terminal goal — which ends the episode — is a relief, not a loss of
        # future income. Paid into the EXTRINSIC stream (directed signal).
        # Tune so |step_penalty| x steps-between-milestones < milestone reward.
        self.step_penalty = config.get("step_penalty", 0.0)
        # Small one-shot bonus on the FIRST battle entered on each map this
        # episode (not per battle — that was a farm surface). Teaches the
        # agent that engaging is sometimes necessary without paying to grind.
        self.battle_engagement_reward = config.get("battle_engagement_reward", 3.0)
        # Reward for WINNING the first battle on each map this episode
        # (enemy HP reached 0, player did not white out / catch). This is the
        # primary battle signal — outcome, not raw damage.
        self.battle_win_reward = config.get("battle_win_reward", 8.0)
        # Raw per-step damage reward. OFF by default — Δenemy_hp is the farm
        # vector that produced 2500-reward grinding episodes. Kept behind the
        # coef so it can be re-enabled experimentally.
        self.damage_dealt_reward = config.get("damage_dealt_reward", 0.0)
        # Hard ceiling on total battle reward (entry + win + damage) paid in a
        # single episode — a decay-independent backstop against farming.
        self.battle_reward_episode_cap = config.get("battle_reward_episode_cap", 30.0)
        # Coefficient in the per-map battle decay multiplier
        # ``1/(1 + battle_decay_coef · battles_on_this_map)``. Set to 0
        # to disable decay entirely. Default 0.2 picks a gentle slope.
        self.battle_decay_coef = config.get("battle_decay_coef", 0.2)
        self.whiteout_penalty = config.get("whiteout_penalty", -100)
        # Minor dense reward paid per total party level gained. Leveling is
        # an explicit *minor* progress goal (see project objective): kept
        # small so it can't out-pay exploration and re-create battle farming.
        # Suppressed on party-size change so a newly caught / received
        # Pokémon's existing levels are not paid out.
        self.level_up_reward = config.get("level_up_reward", 10)

        # Threshold used by GoalsManager.check_xp_goals — does not pay
        # reward directly; advances the per-source N_goals counter only.
        self.xp_goal_threshold = config.get("xp_goal_threshold", 10)

        self.clip = config.get("reward_clip", 1000)

        # Optional rounding of the per-step reward to this many decimal
        # places — keeps logs / PNG filenames legible (e.g. 8.33 instead of
        # 8.3333334922790527). None disables rounding. The breakdown
        # accumulators stay unrounded (diagnostic only). Default 2 preserves
        # two decimal places.
        self.reward_round_dp = config.get("reward_round_dp", 2)

        # Opt-in early termination: end the episode the moment every
        # configured goal has fired this episode.
        self.terminate_on_goal_complete = bool(
            config.get("terminate_on_goal_complete", False)
        )

        # When True, reward calculation must not write process-lifetime
        # state (the map-discovery ledger). Set by gym_env.replay_actions
        # around the warm-start replay loop so re-walking the corridor to the
        # goal every episode doesn't drain the first-discovery bonus.
        self._replaying = False

        # Per-episode novelty / progress trackers. The frontier bonus pays
        # each quantised cell at most once per episode; the persistent visit
        # count (cross-episode decay) lives in ``visit_archive``.
        self._novel_cells_this_episode = set()
        self.explored_tiles = set()
        # Per-episode. Cross-episode novelty is in visit_archive.
        # ``seed_explored_maps`` pre-fills this with the maps the replay
        # walked through, so new_map only pays for genuinely fresh maps.
        self.explored_maps = set()

        # Running total of intrinsic reward paid this episode, clamped to
        # ``intrinsic_reward_episode_cap``. Reset in ``start_new_episode``.
        self._intrinsic_reward_paid = 0.0

        # Per-source reward accumulators for diagnostic logging.
        self._episode_breakdown = {
            "flag": 0.0,
            "map_goal": 0.0,
            "pokedex_owned": 0.0,
            "pokedex_first_sight": 0.0,
            "key_item": 0.0,
            "new_map": 0.0,
            "frontier": 0.0,
            "battle_entry": 0.0,
            "battle_win": 0.0,
            "damage_dealt": 0.0,
            "level_up": 0.0,
            "whiteout": 0.0,
            "step": 0.0,
            # Negative clawback: the portion of gross intrinsic reward removed
            # by the per-episode cap. Keeps the breakdown summing to the true
            # reward and makes "how much farming the cap killed" visible.
            "intrinsic_capped": 0.0,
        }

        # Process-lifetime trackers (only reset on full re-init, not per-episode).
        self.pokedex_seen = 0
        self.pokedex_owned = 0
        self._key_items_count = None

        # Last per-step reward split (extrinsic milestones vs intrinsic
        # exploration/battle/leveling), read by the vec agent's two-stream
        # scaler. Refreshed every calculate_reward call.
        self._last_extrinsic = 0.0
        self._last_intrinsic = 0.0

        self.done = False
        # ``truncated`` distinguishes a budget cut-off (steps > max_steps)
        # from a genuine terminal (goal complete). GAE must bootstrap
        # V(s_{T+1}) on truncation but zero it on a true terminal — see
        # the agents' GAE routines.
        self.truncated = False
        self.last_action = None
        self.steps = 0
        self._prev_enemy_hp = None
        # Tracks battle_type across calls so the engagement bonus only
        # fires on the 0 → non-zero transition. Deliberately NOT reset in
        # ``start_new_episode`` — if action_replay walks into a battle,
        # training should not re-pay the entry bonus.
        self._prev_battle_type = 0
        self.cumulative_reward = 0

        # Whiteout gating: only ``_prev_party_size`` and ``_prev_party_hp``
        # are load-bearing.
        self._prev_party_size = None
        self._prev_party_hp = None
        # Level-up reward trackers (independent of the whiteout / goal
        # trackers so the shared-state coupling stays contained).
        self._level_reward_prev_size = None
        self._level_reward_prev_total = None
        self.whiteouts = 0
        self.battles_this_episode = 0
        # Per-map battle counter for decay. Keyed by (map_bank, map_num).
        # Preserved across same-map re-entry within an episode so the
        # policy can't oscillate between two routes to reset decay.
        self._battles_by_map = {}
        # The map where the current battle started — used for damage
        # decay so a flicker of the overworld map mid-battle doesn't
        # rescale the gradient. Cleared when battle_type returns to 0.
        self._current_battle_map = None
        # Maps on which a battle has already been WON this episode (win bonus
        # is first-win-per-map only). Per-episode.
        self._battle_won_maps = set()
        # Whether the enemy's HP has reached 0 during the current battle —
        # distinguishes a win from a flee / whiteout at battle exit.
        self._enemy_reached_zero = False
        # Running total of battle reward paid this episode (entry + win +
        # damage), clamped to ``battle_reward_episode_cap``.
        self._battle_reward_paid = 0.0

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
    def pokedex_goals(self):
        return self.goals.pokedex_goals

    @property
    def level_goals(self):
        return self.goals.level_goals

    @property
    def xp_goals(self):
        return self.goals.xp_goals

    @property
    def pokedex_goals_completed(self):
        return self.goals.pokedex_goals_completed

    @property
    def level_goals_completed(self):
        return self.goals.level_goals_completed

    @property
    def xp_goals_completed(self):
        return self.goals.xp_goals_completed

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
        # Level-up reward trackers (independent of the whiteout / goal
        # trackers so the shared-state coupling stays contained).
        self._level_reward_prev_size = None
        self._level_reward_prev_total = None
        self.whiteouts = 0
        self.battles_this_episode = 0
        self._battles_by_map = {}
        self._current_battle_map = None
        self._battle_won_maps = set()
        self._enemy_reached_zero = False
        self._battle_reward_paid = 0.0
        self._intrinsic_reward_paid = 0.0
        self._prev_enemy_hp = None
        self._novel_cells_this_episode = set()
        # ``explored_maps`` is per-episode now. ``seed_explored_maps`` is
        # called after this when a replay precedes the training segment.
        self.explored_maps = set()
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

    def seed_novel_cells(self, cell_keys):
        """Mark cells the replay walked through as already-credited this
        episode, so the frontier bonus does not pay for the corridor the
        warm-start replay already covered (used by the evaluator's replay
        path). The persistent ``visit_archive`` count is intentionally NOT
        bumped here — replay steps are not genuine training discoveries."""
        for key in cell_keys:
            self._novel_cells_this_episode.add(key)

    def seed_battle_counts(self, battles_by_map):
        """Pre-fill ``_battles_by_map`` with battle counts from replay.

        Prevents the training segment from receiving full-strength battle
        rewards on maps where replay already fought. Must be called after
        ``start_new_episode`` (which resets the dict).
        """
        for key, count in battles_by_map.items():
            self._battles_by_map[(int(key[0]), int(key[1]))] = count

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
            # Advance the breadth-of-exploration goal (visit N unique maps).
            # Pays no reward — new_map_reward is the signal; this drives the
            # N_goals metric and the termination predicate.
            self.goals.check_maps_visited_goals()

            # Per-step living cost (extrinsic, directed). Constant negative so
            # loitering bleeds reward and reaching the terminal goal is a
            # relief. ``total`` is re-derived from the two streams below.
            r_step = self.step_penalty
            self._episode_breakdown["step"] += r_step

            r_flag = self._check_flag_progress(env_vars)
            self._episode_breakdown["flag"] += r_flag

            r_mapgoal = self._check_map_goal(env_vars)
            self._episode_breakdown["map_goal"] += r_mapgoal

            r_pok, r_seen = self._check_pokedex_progress(env_vars)
            self._episode_breakdown["pokedex_owned"] += r_pok
            self._episode_breakdown["pokedex_first_sight"] += r_seen

            r_key = self._check_key_item_pickup(env_vars)
            self._episode_breakdown["key_item"] += r_key

            # Internal counter goals (advance N_goals; pay no direct reward).
            self._advance_internal_counters(env_vars)

            r_wo = self._check_whiteout(env_vars)
            self._episode_breakdown["whiteout"] += r_wo

            r_map = self._new_map_bonus(env_vars)
            self._episode_breakdown["new_map"] += r_map

            r_front = self._frontier_novelty_bonus(env_vars)
            self._episode_breakdown["frontier"] += r_front

            r_be, r_win, r_dmg = self._battle_engagement_reward(env_vars)
            self._episode_breakdown["battle_entry"] += r_be
            self._episode_breakdown["battle_win"] += r_win
            self._episode_breakdown["damage_dealt"] += r_dmg

            r_lvl = self._check_level_up(env_vars)
            self._episode_breakdown["level_up"] += r_lvl

            # Two-stream split for the agent's decoupled reward scaler.
            # Extrinsic = directed milestones (sparse, large) + the living
            # cost; intrinsic = exploration + battle outcome + leveling
            # (dense, small). The agent normalises each stream independently so
            # a milestone spike can't divide the dense signal toward zero (and
            # vice-versa). The intrinsic stream is clamped to a per-episode cap
            # here so dense exploration can never out-pay the milestone reward
            # on a long route; the clipped amount is logged as a negative
            # ``intrinsic_capped`` clawback so the breakdown still sums to the
            # true reward.
            extrinsic = r_flag + r_mapgoal + r_pok + r_seen + r_key + r_wo + r_step
            gross_intrinsic = r_map + r_front + r_be + r_win + r_dmg + r_lvl
            intrinsic = self._cap_intrinsic_reward(gross_intrinsic)
            self._episode_breakdown["intrinsic_capped"] += intrinsic - gross_intrinsic
            # Re-derive total from the streams so it reflects the living cost
            # and the intrinsic cap (the piecewise sums above predate both).
            total = extrinsic + intrinsic
        else:
            total = 0.0
            extrinsic = 0.0
            intrinsic = 0.0

        self._last_extrinsic = float(extrinsic)
        self._last_intrinsic = float(intrinsic)
        self.last_action = button_press

        if self.steps > self.max_steps:
            # Budget cut-off, not a natural terminal: the agent did not
            # "finish", the episode was merely truncated. The value of the
            # (unobserved) next state should be bootstrapped in GAE.
            self.done = True
            self.truncated = True
        elif self.terminate_on_goal_complete and self.goals.all_goal_thresholds_met():
            # Genuine terminal: the stage goal is met and the episode ends
            # here. truncated stays False so GAE zeroes V(s_{T+1}).
            self.done = True

        if self.reward_round_dp is not None:
            total = round(float(total), int(self.reward_round_dp))

        self.cumulative_reward += total
        return np.clip(total, -self.clip, self.clip).astype(np.float32), self.done

    # ------------------------------------------------------------------ #
    # Progress-signal subroutines                                         #
    # ------------------------------------------------------------------ #

    def _check_flag_progress(self, env_vars):
        new_fires = self.goals.check_flag_goals(env_vars["story_flags"])
        return new_fires * self.flag_progress_reward

    def _check_map_goal(self, env_vars):
        """Pay ``map_goal_reward`` per map-reach goal entered this step."""
        if not self.map_goal_reward:
            return 0.0
        new_fires = self.goals.check_map_goals(
            env_vars["map_bank"], env_vars["map_num"]
        )
        return new_fires * self.map_goal_reward

    def _check_pokedex_progress(self, env_vars):
        """Returns (owned_reward, first_sight_reward).

        - ``owned`` increment pays ``pokedex_owned_reward`` per species caught.
        - ``seen`` increment pays ``pokedex_first_sight_reward`` per fresh species
          encountered. Smaller than owned so the agent isn't farmed by
          repeated wild encounters of the same species.
        """
        owned_reward = 0.0
        seen_reward = 0.0
        seen = int(env_vars["pokedex_seen"])
        owned = int(env_vars["pokedex_owned"])

        if seen > self.pokedex_seen:
            seen_reward = (seen - self.pokedex_seen) * self.pokedex_first_sight_reward
            self.pokedex_seen = seen
        if owned > self.pokedex_owned:
            owned_reward = (owned - self.pokedex_owned) * self.pokedex_owned_reward
            self.pokedex_owned = owned
        # Funnel into threshold-based goals (advances N_goals).
        self.goals.check_pokedex_goals(seen, owned)
        return owned_reward, seen_reward

    def _check_key_item_pickup(self, env_vars):
        """One-shot bonus per new key item acquired. Bridges the gap
        between ``return egg`` (Pokéballs handed over by Elm) and ``catch
        wild Pokémon``."""
        if not self.key_item_pickup_reward:
            return 0.0
        count = int(env_vars.get("key_items_count", 0))
        if self._key_items_count is None:
            self._key_items_count = count
            return 0.0
        if count > self._key_items_count:
            delta = count - self._key_items_count
            self._key_items_count = count
            return delta * self.key_item_pickup_reward
        # Allow the count to decrease (item used) without penalty.
        self._key_items_count = count
        return 0.0

    def _advance_internal_counters(self, env_vars):
        """Advance level / xp counters via GoalsManager. Pays no reward."""
        battle_type = int(env_vars.get("battle_type", 0))
        if battle_type not in _VALID_BATTLE_TYPES:
            return
        party_size, party_level, _, party_exp = env_vars["party_info"]
        self.goals.check_xp_goals(party_size, party_exp, self.xp_goal_threshold)
        self.goals.check_level_goals(party_size, party_level)
        if self._prev_party_size is None:
            self._prev_party_size = party_size

    def _check_level_up(self, env_vars):
        """Minor dense reward per total party level gained.

        Pays ``level_up_reward`` for each level the party's total level
        increases by. Suppressed (baseline reseeded, no pay-out) when the
        party size changes, so a freshly caught / received Pokémon's
        existing levels don't trigger a windfall. Kept deliberately small
        relative to the exploration signals — leveling is a *minor* goal,
        and over-rewarding it re-creates the battle-farming local optimum.
        """
        if not self.level_up_reward:
            return 0.0
        party_size, party_level, _, _ = env_vars["party_info"]
        if party_size <= 0:
            return 0.0
        if self._level_reward_prev_size is None:
            self._level_reward_prev_size = party_size
            self._level_reward_prev_total = party_level
            return 0.0
        if party_size != self._level_reward_prev_size:
            self._level_reward_prev_size = party_size
            self._level_reward_prev_total = party_level
            return 0.0
        reward = 0.0
        if party_level > self._level_reward_prev_total:
            reward = (party_level - self._level_reward_prev_total) * self.level_up_reward
        self._level_reward_prev_total = party_level
        return reward

    def _check_whiteout(self, env_vars):
        """One-shot whiteout penalty on the ``prev_total_hp > 0 → cur == 0``
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
        """Discovery bonus for entering a (map_bank, map_num) not yet entered
        this episode.

        Two layered signals:
          - ``new_map_first_discovery_reward / (global_count + 1)``: a small,
            smoothly-decaying first-discovery top-up. ``global_count`` is the
            run-wide number of training entries to this map (the map ledger in
            ``VisitArchive``). Map-bouncing stops paying after a few entries;
            a genuinely fresh frontier map still pays on first discovery.
          - ``new_map_reward`` (legacy flat per-episode; default 0).

        ``explored_maps`` is pre-seeded with maps the action-replay visited
        (see ``seed_explored_maps``), so the bonus only pays for maps the
        *training* segment actually discovers. Also maintains
        ``explored_tiles`` (purely an observation signal, no reward attached).

        The global ledger is written only when not replaying (so re-walking
        the corridor every episode can't drain the bonus) and not during
        scripted overlays (a scripted warp can read a transient map).
        """
        map_key = (int(env_vars["map_bank"]), int(env_vars["map_num"]))
        loc = (env_vars["X"], env_vars["Y"], map_key[0], map_key[1])
        self.explored_tiles.add(loc)
        if not self.new_map_reward and not self.new_map_first_discovery_reward:
            return 0
        if map_key in self.explored_maps:
            return 0
        self.explored_maps.add(map_key)
        reward = float(self.new_map_reward)
        if self.new_map_first_discovery_reward:
            if env_vars.get("script_active", False):
                # Don't credit / record a discovery from a transient scripted
                # map read; still mark explored above so we don't re-check.
                return reward
            gcount = self.visit_archive.map_count(map_key[0], map_key[1])
            reward += self.new_map_first_discovery_reward / (gcount + 1)
            if not self._replaying:
                self.visit_archive.record_map(map_key[0], map_key[1])
        return reward

    def _frontier_novelty_bonus(self, env_vars):
        """Persistent count-based cell novelty: pays
        ``bonus / (effective_count + 1)`` where ``count`` is the **run-wide**
        visit count for the quantised cell (the ``visit_archive`` ledger that
        persists across episodes), not a per-episode counter.

        Because the count persists, the intrinsic landscape *depletes* over
        training: a region the policy has already explored to death pays
        near-zero, while a never-visited cell (e.g. the first room past a door
        it rarely opens) still pays full. The novelty gradient therefore
        always points at the receding frontier — the policy learns that
        re-treading known ground is pointless and the only novelty left is
        further out, with no per-stage breadcrumb. Combined with the per-step
        living cost this makes loitering in a saturated region net-negative.

        ``effective_count = min(count, floor)`` bounds how deep the decay
        goes (a small renewable trickle remains, which the step penalty
        swamps). Each cell pays at most once per episode
        (``_novel_cells_this_episode``) so wiggling can't farm, and the count
        is recorded only on genuine training steps (``not _replaying``) so the
        evaluator's replay can't drain the ledger.

        Gated on ``script_active`` — during cutscenes / menus the player
        position is stale, so paying a frontier bonus would be noise.
        """
        if self.frontier_novelty_bonus <= 0:
            return 0
        # Skip during scripted overlays — player position is stale.
        if env_vars.get("script_active", False):
            return 0
        mb, mn = env_vars["map_bank"], env_vars["map_num"]
        x, y = env_vars["X"], env_vars["Y"]
        cell = self.visit_archive.cell_key(mb, mn, x, y)
        # Once-per-cell-per-episode: re-treading a cell within one episode
        # can't farm (the persistent count handles cross-episode decay).
        if cell in self._novel_cells_this_episode:
            return 0
        self._novel_cells_this_episode.add(cell)
        prior_count = self.visit_archive.count(mb, mn, x, y)
        if not self._replaying:
            # Record the genuine training visit so the cell depletes for
            # future episodes. Replay steps must not drain the ledger.
            self.visit_archive.record(mb, mn, x, y)
        effective_count = prior_count
        if self.frontier_novelty_count_floor is not None and self.frontier_novelty_count_floor > 0:
            effective_count = min(prior_count, self.frontier_novelty_count_floor)
        return self.frontier_novelty_bonus / (effective_count + 1)

    def _battle_decay_multiplier(self, n):
        """Per-map battle decay. ``n`` is the number of battles already
        entered on the current map this episode (1-indexed at the first
        battle). ``battle_decay_coef = 0`` disables decay entirely."""
        if self.battle_decay_coef <= 0:
            return 1.0
        return 1.0 / (1.0 + self.battle_decay_coef * n)

    def _cap_battle_reward(self, amount):
        """Clamp a proposed positive battle reward to the per-episode cap.
        Tracks the running total so entry + win + damage together can never
        exceed ``battle_reward_episode_cap`` — a decay-independent backstop
        against grinding. Returns the portion actually paid."""
        if amount <= 0:
            return 0.0
        cap = self.battle_reward_episode_cap
        if cap is None or cap <= 0:
            return amount
        remaining = cap - self._battle_reward_paid
        allowed = max(0.0, min(amount, remaining))
        self._battle_reward_paid += allowed
        return allowed

    def _cap_intrinsic_reward(self, amount):
        """Clamp the per-step POSITIVE intrinsic total to the per-episode
        budget ``intrinsic_reward_episode_cap``. Tracks the running total so
        new_map + frontier + battle + level together can never exceed the cap
        in one episode — the backstop that stops exploration from out-paying
        the milestone reward on long routes. Negative intrinsic (none today)
        passes through untouched. Returns the portion actually paid."""
        cap = self.intrinsic_reward_episode_cap
        if cap is None or cap <= 0 or amount <= 0:
            return amount
        remaining = cap - self._intrinsic_reward_paid
        allowed = max(0.0, min(amount, remaining))
        self._intrinsic_reward_paid += allowed
        return allowed

    def _battle_engagement_reward(self, env_vars):
        """Battle reward = winning, not damage.

        - **Entry bonus** (``battle_engagement_reward``): paid once on the
          FIRST battle entered on each map this episode — enough to teach the
          agent that engaging is sometimes necessary, not enough to farm.
        - **Win bonus** (``battle_win_reward``): paid once per map this
          episode when a battle is WON — enemy HP reached 0 and the player did
          not white out (and did not catch: catching is paid by
          ``pokedex_owned_reward`` and rarely zeroes enemy HP). Outcome, not
          grind, is the signal.
        - **Damage** (``damage_dealt_reward``): OFF by default; the farm
          vector. Behind the coef for experiments.

        All three are multiplied by the per-map decay and clamped to
        ``battle_reward_episode_cap`` for the episode. Returns
        ``(entry_reward, win_reward, damage_reward)``.
        """
        battle_type = int(env_vars.get("battle_type", 0))
        if battle_type not in _VALID_BATTLE_TYPES:
            return 0.0, 0.0, 0.0

        win_reward = 0.0
        if battle_type == 0:
            # Battle just ended this step (or no battle active). If a battle
            # was in progress (``_current_battle_map`` set), decide win/loss.
            if self._current_battle_map is not None and self._enemy_reached_zero:
                _, _, party_hp, _ = env_vars["party_info"]
                won = party_hp > 0  # not a whiteout
                if won and self._current_battle_map not in self._battle_won_maps:
                    self._battle_won_maps.add(self._current_battle_map)
                    n = self._battles_by_map.get(self._current_battle_map, 1)
                    win_reward = self._cap_battle_reward(
                        self.battle_win_reward * self._battle_decay_multiplier(n)
                    )
            self._prev_enemy_hp = None
            self._prev_battle_type = 0
            self._current_battle_map = None
            self._enemy_reached_zero = False
            return 0.0, win_reward, 0.0

        entry_reward = 0.0
        if self._prev_battle_type == 0:
            map_key = (int(env_vars["map_bank"]), int(env_vars["map_num"]))
            first_on_map = map_key not in self._battles_by_map
            self._battles_by_map[map_key] = self._battles_by_map.get(map_key, 0) + 1
            self._current_battle_map = map_key
            self.battles_this_episode += 1
            self._enemy_reached_zero = False
            if first_on_map:
                n = self._battles_by_map[map_key]
                entry_reward = self._cap_battle_reward(
                    self.battle_engagement_reward * self._battle_decay_multiplier(n)
                )
        self._prev_battle_type = battle_type

        damage_reward = 0.0
        enemy_hp = int(env_vars.get("enemy_hp", 0))
        if self._prev_enemy_hp is not None and enemy_hp < self._prev_enemy_hp:
            if self.damage_dealt_reward:
                damage = self._prev_enemy_hp - enemy_hp
                n = self._battles_by_map.get(self._current_battle_map, 1)
                damage_reward = self._cap_battle_reward(
                    damage * self.damage_dealt_reward * self._battle_decay_multiplier(n)
                )
        if enemy_hp == 0:
            self._enemy_reached_zero = True
        self._prev_enemy_hp = enemy_hp
        return entry_reward, win_reward, damage_reward

    # ------------------------------------------------------------------ #
    # Progress queries (used by RAM observation builder & plotting)
    # ------------------------------------------------------------------ #

    def get_progress(self):
        return {
            "Steps": self.steps,
            "Goals Reached": self.goals.N_goals,
            "Pokédex Seen": self.pokedex_seen,
            "Pokédex Owned": self.pokedex_owned,
            "Explored Tiles": len(self.explored_tiles),
            "Maps Visited (episode)": len(self.goals._maps_seen_this_episode),
            "Banks Visited (episode)": len(
                {bank for (bank, _) in self.goals._maps_seen_this_episode}
            ),
            "Flag Fires": self.goals.flag_goals_completed,
            "Whiteouts": self.whiteouts,
            "Battles Entered": self.battles_this_episode,
            "Cumulative Reward": self.cumulative_reward,
        }

    def explored_tile_count(self):
        return len(self.explored_tiles)

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
