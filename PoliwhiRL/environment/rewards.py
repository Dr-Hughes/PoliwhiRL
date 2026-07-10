# -*- coding: utf-8 -*-
"""Per-step reward calculator.

Reward sources, in order of intended magnitude:

  Milestone rewards (directed, large, sparse) — the PRIMARY signal:
    ANY fresh 0->1 transition of a flag in gym_env._DERIVED_FLAG_TABLE, ANY
    new pokedex species (seen/owned), ANY party level gained, and ANY new
    key item pay reward UNCONDITIONALLY — regardless of whether a stage's
    `goals` list happens to mention that specific flag/species/level. This
    is deliberate: a hand-picked per-stage goal list doesn't scale (every
    new stage would need a human to look up the right flag number), and it
    caps what the agent can ever be rewarded for to what a human already
    anticipated. Making the whole curated table reward-eligible everywhere
    means the agent can pick up on ANY story-progress event it stumbles
    into while exploring, not just the one thing a stage config names — the
    same mechanism that pays for "get the starter" also pays for "give the
    egg back to Elm" and "beat the first gym", with zero additional config.
    A stage's own `goals` list still exists, but only drives termination /
    success-rate metrics for that stage (see goals.py) — it no longer gates
    reward. `map`-type goals are the one deliberate exception: reaching a
    SPECIFIC coordinate has no way to be "general" (there's no generic
    signal for "this location matters") so `map_goal_reward` still only
    pays for a stage's explicitly configured target, same as before.
    These are sized an order of magnitude above the exploration/battle
    terms below so grabbing a milestone dominates regardless of how long
    novelty has had to accumulate.

  Battle progress (capped, decaying) — SECONDARY, "some battling is good,
  farming isn't":
    entering a battle and knocking out the opponent each pay a small
    bonus, first-per-map-per-episode with per-map decay, and the combined
    total is clamped to `battle_reward_episode_cap` per episode so
    grinding plateaus in value fast while a handful of productive battles
    (including a mandatory trainer fight blocking the route) still pay.

  Level-up (small, dense):
    r += level_up_reward per total party level gained this episode.

  Exploration (secondary, cheap):
    frontier novelty: flat bonus for stepping onto a cell not yet visited
    THIS EPISODE. Deliberately NOT cross-episode/run-wide: a persistent
    decaying count previously let the whole reward landscape saturate and
    collapse mid-stage once a region got thoroughly walked (see
    AGENTS.md §10a). The per-episode version can't saturate because it
    resets every episode; long-run progress is anchored by the milestone
    rewards above (which persist as "already fired this stage", not
    "already fired ever") instead.
    new-map: bonus / (run-wide entry count + 1) for entering a map not yet
    visited this episode — this ledger IS run-wide/persistent (backed by
    the much smaller VisitArchive map-count table), so map-bouncing stops
    paying after a handful of entries while a genuinely new map still
    pays in full on first discovery.

  Step penalty (constant, small):
    time pressure — keeps grinding/wandering after a milestone from being
    free, without being large enough to make the agent rush past a
    milestone it hasn't reached yet.

  Whiteout: one-shot penalty on party HP -> 0.
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
        # Only the map-count side of the archive is used now (cell novelty
        # is per-episode — see module docstring), but it's still the right
        # place for the run-wide new-map ledger to live.
        self.visit_archive = visit_archive if visit_archive is not None else VisitArchive()

        self.max_steps = config["episode_length"]
        # When true, the episode ends the moment every configured goal has
        # fired (natural terminal, not a truncation). Off by default:
        # non-termination makes a milestone strictly additive ("reach goal
        # -> +reward AND keep exploring") rather than a trade-off against
        # the rest of the episode's reward, which is what stops the policy
        # from learning a goal and then abandoning it. Stage 1 (short,
        # simple) is the one place this is worth turning on.
        self.terminate_on_goal_complete = bool(
            config.get("terminate_on_goal_complete", False)
        )

        # ---- Milestone rewards (primary signal, always-on — see module docstring) ----
        self.flag_progress_reward = config.get("flag_progress_reward", 500)
        self.map_goal_reward = config.get("map_goal_reward", 250)
        self.maps_visited_reward = config.get("maps_visited_reward", 0)
        self.pokedex_owned_reward = config.get("pokedex_owned_reward", 150)
        self.pokedex_seen_reward = config.get("pokedex_first_sight_reward", 10)
        self.key_item_pickup_reward = config.get("key_item_pickup_reward", 5)

        # ---- Battle progress (capped, decaying) ----
        self.battle_engagement_reward = config.get("battle_engagement_reward", 3.0)
        self.battle_win_reward = config.get("battle_win_reward", 8.0)
        self.battle_decay_coef = config.get("battle_decay_coef", 0.2)
        self.battle_reward_episode_cap = config.get("battle_reward_episode_cap", 30.0)

        # ---- Level-up ----
        self.level_up_reward = config.get("level_up_reward", 10)

        # ---- Exploration (secondary) ----
        # Run-wide decaying map-discovery reward. Pays new_map_reward /
        # (global_map_entry_count + 1) on first entry to a map each episode.
        self.new_map_reward = config.get("new_map_reward", 50)
        # Per-episode-only cell novelty (see module docstring for why this
        # is not run-wide).
        self.frontier_novelty_bonus = config.get("frontier_novelty_bonus", 10.0)

        self.whiteout_penalty = config.get("whiteout_penalty", -100)
        # Applied every valid step. Small and negative — time pressure, not
        # a dominant term.
        self.step_penalty = float(config.get("step_penalty", -0.02))

        self.clip = config.get("reward_clip", 1000)

        # Optional rounding of the per-step reward to this many decimal
        # places — keeps logs / PNG filenames legible. Default 2.
        self.reward_round_dp = config.get("reward_round_dp", 2)

        # Per-episode novelty / progress trackers.
        self._novel_cells_this_episode = set()
        # Kept for terminal_info / vec-env plumbing compatibility, but never
        # populated: cell novelty is per-episode only now, so there is
        # nothing to merge into the (run-wide) archive at episode end.
        self._cells_to_record = set()
        self._maps_to_record = set()
        # RAM-vector history features (see gym_env._build_ram_vector): let
        # the POLICY see its own exploration frontier directly, not just
        # feel it via reward.
        self._steps_since_novel_cell = 0
        self._last_cell_novel = False
        # Diagnostic: the step number at which each goal rung fired this episode.
        self.goal_fire_steps = []
        self._prev_rung = 0
        self.explored_tiles = set()
        # Per-episode. Cross-episode map novelty lives in visit_archive.
        self.explored_maps = set()

        # Per-source reward accumulators for diagnostic logging.
        self._episode_breakdown = {
            "flag": 0.0,
            "map_goal": 0.0,
            "maps_visited": 0.0,
            "pokedex": 0.0,
            "key_item": 0.0,
            "battle": 0.0,
            "level": 0.0,
            "frontier": 0.0,
            "new_map": 0.0,
            "step_penalty": 0.0,
            "whiteout": 0.0,
        }

        # Always-on global-progress trackers (see module docstring). Baseline
        # is seeded on first call each episode, not hardcoded to 0/empty, so
        # a save-state that starts mid-progress doesn't get re-paid for
        # already-true state.
        self._flag_table_initial = None
        self._flag_table_fired = {}
        self._prev_pokedex_seen = None
        self._prev_pokedex_owned = None
        self._prev_level_party_size = None
        self._prev_level_total = None
        self._prev_key_items_count = None

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

        # Battle-progress trackers.
        self._prev_battle_type = None
        self._prev_enemy_hp = None
        self._battle_engaged_maps = {}
        self._battle_won_maps = {}
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
        self._steps_since_novel_cell = 0
        self._last_cell_novel = False
        self.goal_fire_steps = []
        self._prev_rung = 0
        self.explored_maps = set()
        self._recent_maps_list = []
        for key in self._episode_breakdown:
            self._episode_breakdown[key] = 0.0
        self._prev_battle_type = None
        self._prev_enemy_hp = None
        self._battle_engaged_maps = {}
        self._battle_won_maps = {}
        self._battle_reward_paid = 0.0
        self._flag_table_initial = None
        self._flag_table_fired = {}
        self._prev_pokedex_seen = None
        self._prev_pokedex_owned = None
        self._prev_level_party_size = None
        self._prev_level_total = None
        self._prev_key_items_count = None
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
            mb, mn = int(env_vars["map_bank"]), int(env_vars["map_num"])
            self.goals.note_map_visit(mb, mn)

            # GoalsManager calls below still run for their side effects on
            # this STAGE's configured goal counters (termination via
            # all_goal_thresholds_met, success-rate metrics) — their return
            # values no longer drive reward directly (see module docstring).
            # map_fires is the one exception: reaching a specific configured
            # coordinate has no general substitute, so it still pays here.
            maps_visited_fires = self.goals.check_maps_visited_goals()
            map_fires = self.goals.check_map_goals(mb, mn)
            self.goals.check_flag_goals(env_vars["story_flags"])
            party_size, party_level, _party_hp, party_exp = env_vars["party_info"]
            self.goals.check_pokedex_goals(
                env_vars["pokedex_seen"], env_vars["pokedex_owned"]
            )
            self.goals.check_level_goals(party_size, party_level)

            r_map_goal = map_fires * self.map_goal_reward
            r_maps_visited = maps_visited_fires * self.maps_visited_reward
            r_flag = self._global_flag_progress_bonus(env_vars["story_flags"])
            r_pokedex = self._global_pokedex_bonus(
                env_vars["pokedex_seen"], env_vars["pokedex_owned"]
            )
            r_level = self._global_level_bonus(party_size, party_level)
            r_key_item = self._global_key_item_bonus(
                env_vars.get("key_items_count", 0)
            )

            r_battle = self._battle_progress(env_vars)
            r_map = self._new_map_bonus(env_vars)
            r_front = self._frontier_novelty_bonus(env_vars)
            r_wo = self._check_whiteout(env_vars)

            self._episode_breakdown["flag"] += r_flag
            self._episode_breakdown["map_goal"] += r_map_goal
            self._episode_breakdown["maps_visited"] += r_maps_visited
            self._episode_breakdown["pokedex"] += r_pokedex
            self._episode_breakdown["key_item"] += r_key_item
            self._episode_breakdown["level"] += r_level
            self._episode_breakdown["battle"] += r_battle
            self._episode_breakdown["new_map"] += r_map
            self._episode_breakdown["frontier"] += r_front
            self._episode_breakdown["whiteout"] += r_wo
            self._episode_breakdown["step_penalty"] += self.step_penalty

            total = (
                r_flag
                + r_map_goal
                + r_maps_visited
                + r_pokedex
                + r_key_item
                + r_level
                + r_battle
                + r_map
                + r_front
                + r_wo
                + self.step_penalty
            )

            # Log the step at which each goal rung fired (diagnostic;
            # covers both map-reach and flag milestones).
            rung = self.n_map_goals_completed() + self.n_flag_goals_completed()
            if rung > self._prev_rung:
                self.goal_fire_steps.extend(
                    [int(self.steps)] * (rung - self._prev_rung)
                )
                self._prev_rung = rung

            if self.terminate_on_goal_complete and self.goals.all_goal_thresholds_met():
                self.done = True
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

    def _global_flag_progress_bonus(self, story_flags):
        """Pays ``flag_progress_reward`` for ANY fresh 0->1 transition among
        ALL flags in ``gym_env._DERIVED_FLAG_TABLE`` — not just ones a stage
        config names (see module docstring). Snapshots the initial
        flag-bytes on first call so flags already true at episode start
        (e.g. earlier-game story state baked into the save-state) don't fire.

        Lazy import: gym_env imports Rewards at module scope, so importing
        gym_env back at module scope here would be circular. By the time
        this method actually runs, gym_env is already fully loaded.
        """
        from .gym_env import _DERIVED_FLAG_TABLE

        if self._flag_table_initial is None:
            self._flag_table_initial = bytes(story_flags)

        new_fires = 0
        for flag_num, _name in _DERIVED_FLAG_TABLE:
            if self._flag_table_fired.get(flag_num, False):
                continue
            byte_idx, bit_idx = flag_num // 8, flag_num % 8
            now_bit = (story_flags[byte_idx] >> bit_idx) & 1
            start_bit = (self._flag_table_initial[byte_idx] >> bit_idx) & 1
            if now_bit == 1 and start_bit == 0:
                self._flag_table_fired[flag_num] = True
                new_fires += 1
        return new_fires * self.flag_progress_reward

    def _global_pokedex_bonus(self, pokedex_seen, pokedex_owned):
        """Pays for ANY new pokedex species (seen/owned), not just when a
        stage configures a pokedex goal. Baseline seeded on first call this
        episode (not hardcoded 0) so a mid-progress save-state isn't re-paid."""
        if self._prev_pokedex_seen is None:
            self._prev_pokedex_seen = int(pokedex_seen)
            self._prev_pokedex_owned = int(pokedex_owned)
            return 0.0
        seen_delta = max(0, int(pokedex_seen) - self._prev_pokedex_seen)
        owned_delta = max(0, int(pokedex_owned) - self._prev_pokedex_owned)
        self._prev_pokedex_seen = int(pokedex_seen)
        self._prev_pokedex_owned = int(pokedex_owned)
        return seen_delta * self.pokedex_seen_reward + owned_delta * self.pokedex_owned_reward

    def _global_level_bonus(self, party_size, party_level):
        """Pays per total party level gained, always — not gated behind a
        configured `level` goal. Suppressed across a party-size change
        (catching/joining a Pokémon changes total level without anyone
        actually leveling up)."""
        if self._prev_level_party_size is None:
            self._prev_level_party_size = party_size
            self._prev_level_total = party_level
            return 0.0
        if party_size != self._prev_level_party_size:
            self._prev_level_party_size = party_size
            self._prev_level_total = party_level
            return 0.0
        gained = max(0, party_level - self._prev_level_total)
        self._prev_level_total = party_level
        return gained * self.level_up_reward

    def _global_key_item_bonus(self, key_items_count):
        """Pays per new key item picked up, always."""
        if self._prev_key_items_count is None:
            self._prev_key_items_count = int(key_items_count)
            return 0.0
        gained = max(0, int(key_items_count) - self._prev_key_items_count)
        self._prev_key_items_count = int(key_items_count)
        return gained * self.key_item_pickup_reward

    def _battle_progress(self, env_vars):
        """Capped, decaying reward for battle engagement and wins.

        Engagement fires on the ``battle_type`` ``0 -> nonzero`` transition.
        Win fires when ``enemy_hp`` drops to 0 while still in battle (a KO,
        distinct from the player whiteing out or the enemy fleeing/being
        caught). Both are first-per-map-per-episode with per-map decay, and
        their combined total this episode is clamped to
        ``battle_reward_episode_cap`` — a decay-independent anti-farm
        backstop. This is deliberately generic: it doesn't special-case the
        early rival fight (there is no reliable "beat the rival" event flag
        for it — see gym_env.py's ``rival_cherrygrove`` comment) — winning
        ANY blocking trainer battle, including that one, pays through this
        same path.
        """
        cur_bt = int(env_vars.get("battle_type", 0))
        map_key = (int(env_vars["map_bank"]), int(env_vars["map_num"]))
        reward = 0.0

        if self._prev_battle_type is not None and self._prev_battle_type == 0 and cur_bt != 0:
            n = self._battle_engaged_maps.get(map_key, 0) + 1
            self._battle_engaged_maps[map_key] = n
            reward += self.battle_engagement_reward / (1 + self.battle_decay_coef * (n - 1))

        if cur_bt != 0:
            enemy_hp = int(env_vars.get("enemy_hp", 0))
            if self._prev_enemy_hp is not None and self._prev_enemy_hp > 0 and enemy_hp == 0:
                n = self._battle_won_maps.get(map_key, 0) + 1
                self._battle_won_maps[map_key] = n
                reward += self.battle_win_reward / (1 + self.battle_decay_coef * (n - 1))
            self._prev_enemy_hp = enemy_hp
        else:
            self._prev_enemy_hp = None

        self._prev_battle_type = cur_bt

        if reward <= 0:
            return 0.0
        headroom = max(0.0, self.battle_reward_episode_cap - self._battle_reward_paid)
        payable = min(reward, headroom)
        self._battle_reward_paid += payable
        return payable

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
        """Per-episode-only cell novelty: flat bonus the first time this
        episode steps onto a quantised cell (no cross-episode persistence —
        see module docstring). Also maintains the ``last_cell_novel`` /
        ``steps_since_novel_cell`` bookkeeping the RAM vector exposes to the
        policy, independent of whether the bonus itself is configured on,
        so those features stay meaningful even with ``frontier_novelty_bonus``
        disabled.

        Each cell pays at most once per episode (``_novel_cells_this_episode``)
        so wiggling can't farm within an episode. Gated on ``script_active``
        — during cutscenes / menus the player position is stale.
        """
        self._last_cell_novel = False
        self._steps_since_novel_cell += 1
        if env_vars.get("script_active", False):
            return 0
        mb, mn = env_vars["map_bank"], env_vars["map_num"]
        x, y = env_vars["X"], env_vars["Y"]
        cell = self.visit_archive.cell_key(mb, mn, x, y)
        if cell in self._novel_cells_this_episode:
            return 0
        self._novel_cells_this_episode.add(cell)
        self._last_cell_novel = True
        self._steps_since_novel_cell = 0
        if self.frontier_novelty_bonus <= 0:
            return 0
        return self.frontier_novelty_bonus

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

    def last_cell_novel_flag(self):
        """1.0 if the current cell had not been visited yet this episode
        (as of the most recent ``calculate_reward`` call), else 0.0."""
        return 1.0 if self._last_cell_novel else 0.0

    def steps_since_novel_cell(self):
        """Steps since a not-yet-visited-this-episode cell was last
        stepped onto. Lets the policy directly perceive "I've been
        retreading ground for a while" instead of only inferring it
        through reward."""
        return self._steps_since_novel_cell

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
