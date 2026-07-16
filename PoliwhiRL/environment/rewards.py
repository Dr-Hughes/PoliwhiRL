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
    novelty has had to accumulate. Re-fires DEPLETE, though: every episode
    restarts from the same save-state, so each milestone fires again every
    episode, and at full price the first one on the corridor is an annuity
    the policy can farm forever (observed 2026-07-11: "talked_to_mom",
    +500/episode, agent never left the house). Each fire is therefore
    scaled by 1/sqrt(1 + prior_episode_fire_count) from the run-wide
    archive — run-first fires pay in full, farmed re-fires decay toward
    (but never reach) zero, exactly the frontier-novelty rule applied to
    milestones.

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
    frontier novelty: `frontier_novelty_bonus * max(1/sqrt(1 +
    persistent_visits), frontier_novelty_floor)` for stepping onto a cell
    not yet visited THIS EPISODE, where `persistent_visits` is the
    run-wide VisitArchive count for that cell. The per-episode gate stops
    within-episode wiggling from farming; the persistent-count decay is
    what stops CROSS-episode farming — without it, re-covering
    already-known ground in a different order each episode paid the same
    as pushing into genuinely new territory, and an agent could (and did —
    stage 3, 2026-07) settle into a loop that harvests the whole known
    region every episode and never leaves. The episodic FLOOR is the other
    half of the balance (added 2026-07-12 after the gamearea_long run):
    decay directs, but must never extinguish — a fully-decayed region is
    otherwise a reward desert where every action pays the same, the local
    gradient vanishes, and absorbing loops (18k steps against one wall,
    ep_104) become stable. With the floor, first-visit-this-episode always
    pays at least `bonus * floor`, so "keep covering ground you haven't
    covered yet this episode" is self-sustaining income at any depth of
    the game — the curiosity behaviour itself stays rewarded, while the
    (higher) frontier payout still says where the genuinely new ground is.
    The always-on milestone rewards above anchor the corridor either way.
    new-map: bonus * max(1/(run-wide entry count + 1),
    frontier_novelty_floor) for entering a map not yet visited this
    episode — same decay-plus-floor idea at map granularity.

  Step penalty (constant, small):
    time pressure — keeps grinding/wandering after a milestone from being
    free, without being large enough to make the agent rush past a
    milestone it hasn't reached yet.

  Whiteout: one-shot penalty on party HP -> 0.

  Stagnation truncation (not a reward — an episode-budget rule):
    if `stagnation_truncation_steps` consecutive free-walking steps (not
    scripted, not in battle) pass without claiming a single
    first-this-episode cell, the episode is truncated (time-limit
    semantics — bootstrapped, NOT a terminal). No behaviour is punished;
    compute is just reallocated: an absorbed policy (wall-bumping,
    two-cell pacing, any future degenerate loop — all caught by the same
    "nothing novel is happening" test) costs at most the threshold instead
    of the whole episode budget. "auto" sizes it to
    max(256, episode_length // 16).
"""
import math

import numpy as np
from .goals import GoalsManager
from .visit_archive import VisitArchive, CELL_SIZE

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
        # Cell novelty, gated per episode AND decayed by the run-wide
        # persistent visit count (see module docstring).
        self.frontier_novelty_bonus = config.get("frontier_novelty_bonus", 10.0)
        # Episodic novelty floor (fraction of the full bonus): the run-wide
        # decay never suppresses a first-visit-THIS-EPISODE payout below
        # bonus * floor. Cross-run decay directs the gradient at the
        # frontier; the floor guarantees within-episode exploration stays
        # income everywhere, forever — without it, fully-decayed regions
        # become reward deserts where every action pays the same (nothing),
        # and the policy has no local gradient at all. Observed 2026-07-12
        # (gamearea_long ep_104): agent absorbed against a wall for 18k of
        # 20k steps in a fully-decayed room. NGU-style split: episodic
        # novelty never dies, lifetime novelty only amplifies the frontier.
        # Milestones deliberately do NOT get this floor — at 500-scale, a
        # floored milestone is the farming annuity all over again.
        self.frontier_novelty_floor = float(
            config.get("frontier_novelty_floor", 0.2)
        )

        self.whiteout_penalty = config.get("whiteout_penalty", -100)
        # Applied every valid step. Small and negative — time pressure, not
        # a dominant term.
        self.step_penalty = float(config.get("step_penalty", -0.02))

        # ---- Stagnation truncation (see module docstring) ----
        # Consecutive free-walking steps without a first-this-episode cell
        # before the episode is truncated. "auto" scales with the episode
        # budget; 0 / None disables. Scripted frames and battles freeze the
        # counter (position is stale / legitimately fixed there) so a long
        # cutscene or fight can never false-trigger it.
        raw_stagnation = config.get("stagnation_truncation_steps", "auto")
        if raw_stagnation == "auto":
            self._stagnation_limit = max(256, int(self.max_steps) // 16)
        else:
            self._stagnation_limit = int(raw_stagnation or 0)
        self._stagnation_steps = 0

        # ---- Battle-progress watchdog (see calculate_reward) ----
        # The free-walking stagnation counter above deliberately FREEZES
        # during battle (battle_type != 0) so a genuine multi-turn fight is
        # never truncated mid-swing. The unintended consequence: a battle
        # that stops making progress — PP exhausted, an unwinnable/looping
        # move-menu, the "mash A on a depleted move" attractor — has NO
        # ceiling short of the whole episode budget (observed in
        # 00_freeform_gamearea_xlong3 ep_3404: 36,202 consecutive battle
        # steps, 88% of the episode, ~98% of presses the A button). This is
        # a battle-specific analogue: it counts consecutive battle steps with
        # NO change in enemy HP ratio or party HP (the two ground-truth
        # signals that a turn actually resolved), and truncates (time-limit
        # semantics, value bootstraps — NOT a punished terminal) once the
        # threshold is crossed. Any real turn — dealing OR taking damage —
        # resets it, so a long but progressing fight is safe; only a battle
        # that is genuinely going nowhere is cut.
        #
        # This cap is NOT the primary escape route — the stuck-temperature
        # mechanism (behaviour-time increasing randomness, see vec_ppo_agent)
        # is, and is meant to break most stalls by learning to switch move /
        # run. The cap exists only so a pathological env can't pollute a
        # rollout with tens of thousands of identical dead transitions, so it
        # is deliberately LOOSE — set well above where the stuck-temperature
        # ramp saturates (~1k steps) so in-episode random recovery gets a long
        # window to find AND repeat the escape before the budget is
        # reallocated. Truncation only ever fires when escape did NOT happen,
        # so it never erases a successful-escape transition already in the
        # buffer. "auto" matches the free-walking stagnation scale
        # (episode_length // 16, floor 512); 0 / None disables.
        raw_battle_stag = config.get("battle_stagnation_truncation_steps", "auto")
        if raw_battle_stag == "auto":
            self._battle_stagnation_limit = max(512, int(self.max_steps) // 16)
        else:
            self._battle_stagnation_limit = int(raw_battle_stag or 0)
        self._battle_stagnation_steps = 0
        self._prev_battle_enemy_hp = None
        self._prev_battle_party_hp = None

        # ---- Battle exit (flee/escape) reward ----
        # Small positive for ENDING a wild battle without whiteout and
        # without a KO (i.e. successfully fleeing). Catching is out of scope
        # for this curriculum and wild battles are pure traversal obstacles,
        # so "get out and keep exploring" is exactly the aligned behaviour;
        # it shortens the credit path for the escape action the stuck-
        # temperature mechanism discovers. Trainer battles (battle_type 2)
        # cannot be fled, so this can never sabotage the mandatory rival
        # fight. Folded into the same per-episode battle cap as engagement/
        # win so it can't become a farm. Default 0.0 (off) — opt in per stage.
        self.battle_flee_reward = float(config.get("battle_flee_reward", 0.0))

        self.clip = config.get("reward_clip", 1000)

        # Optional rounding of the per-step reward to this many decimal
        # places — keeps logs / PNG filenames legible. Default 2.
        self.reward_round_dp = config.get("reward_round_dp", 2)

        # Per-episode novelty / progress trackers.
        self._novel_cells_this_episode = set()
        # Genuinely-novel-this-episode cell keys, queued by
        # _frontier_novelty_bonus and reported via terminal_info at episode
        # end so the agent can merge them into the persistent (run-wide)
        # VisitArchive — this is what backs global_rarity_coef and the
        # matching RAM observation feature with real cross-episode data.
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

        # Diagnostic-only, run-wide discovery log: one record per genuine
        # first-ever (this whole run, not just this episode) milestone
        # fire — {"type": "flag"|"map"|"pokedex_seen"|"pokedex_owned"|
        # "level"|"key_item", "key": <flag_num / (bank,num) / species-delta
        # / level-delta>, "step": step within this episode}. Reported via
        # terminal_info (like reward_breakdown) and stamped with the
        # episode index by the agent when merged — see
        # VecPPOAgent._commit_episode. Purely additive: never read by
        # reward/observation code, exists only so a discovery-order graph
        # (which milestone became reachable, and when, relative to others)
        # can be reconstructed after a run instead of hand-authored.
        self._discoveries_this_episode = []
        # This episode's pending milestone state, reported via
        # terminal_info and merged into visit_archive's run-wide ledger at
        # episode end (see get_milestone_state) — what makes the discovery
        # log above actually correct (run-wide first-ever, not per-episode
        # first-ever) for milestone types with no persistent in-game state.
        self._flags_fired_pending = set()
        # (kind, key) milestone events fired this episode, merged into the
        # archive's fire-count table at episode end. Backs the re-fire
        # decay in _milestone_refire_scale.
        self._milestone_fires_pending = set()
        self._episode_max_pokedex_seen = 0
        self._episode_max_pokedex_owned = 0
        self._episode_max_level = 0
        self._episode_max_key_items = 0
        # Diagnostic-only, per-episode: [flag_num, step] for EVERY derived-
        # table flag fire this episode, unconditioned on any stage's goal
        # list. goal_fire_steps above only covers configured goals, so a
        # goal-less (freeform) stage would otherwise have no time-to-rung
        # series at all — and time-to-rung vs episode budget is the trigger
        # metric for reviving snapshot seeding.
        self.flag_fire_step_log = []

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

        # Ground-truth dead-end tracking for directional_frontier_potential
        # (see that method's docstring) — which of up/down/left/right, if
        # any, was just tried and failed to move the player.
        self._prev_pos = None
        self._blocked_direction = None

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
        self._stagnation_steps = 0
        self._battle_stagnation_steps = 0
        self._prev_battle_enemy_hp = None
        self._prev_battle_party_hp = None
        self.goal_fire_steps = []
        self._prev_rung = 0
        self.explored_maps = set()
        self._recent_maps_list = []
        self._discoveries_this_episode = []
        self._flags_fired_pending = set()
        self._milestone_fires_pending = set()
        self._episode_max_pokedex_seen = 0
        self._episode_max_pokedex_owned = 0
        self._episode_max_level = 0
        self._episode_max_key_items = 0
        self.flag_fire_step_log = []
        for key in self._episode_breakdown:
            self._episode_breakdown[key] = 0.0
        self._prev_battle_type = None
        self._prev_enemy_hp = None
        self._battle_engaged_maps = {}
        self._battle_won_maps = {}
        self._battle_reward_paid = 0.0
        self._prev_pos = None
        self._blocked_direction = None
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

    def get_discoveries(self):
        """This episode's genuine run-wide first-ever milestone fires (see
        _discoveries_this_episode). Diagnostic only."""
        return list(self._discoveries_this_episode)

    def get_milestone_state(self):
        """This episode's pending milestone state, for the agent to merge
        into visit_archive's run-wide ledger (see VisitArchive.merge_milestones).
        The maxima/flags back the discovery log; ``milestone_fires`` backs
        the re-fire depletion the reward path reads
        (see _milestone_refire_scale)."""
        return {
            "flags_fired": sorted(self._flags_fired_pending),
            "pokedex_seen_max": self._episode_max_pokedex_seen,
            "pokedex_owned_max": self._episode_max_pokedex_owned,
            "level_max": self._episode_max_level,
            "key_items_max": self._episode_max_key_items,
            "milestone_fires": sorted(self._milestone_fires_pending),
        }

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
            # covers every configured goal type — pokédex/level rungs were
            # previously missed, so e.g. a "get starter" goal never logged
            # a fire step).
            rung = (
                self.n_map_goals_completed()
                + self.n_flag_goals_completed()
                + self.n_pokedex_goals_completed()
                + self.n_level_goals_completed()
            )
            if rung > self._prev_rung:
                self.goal_fire_steps.extend(
                    [int(self.steps)] * (rung - self._prev_rung)
                )
                self._prev_rung = rung

            if self.terminate_on_goal_complete and self.goals.all_goal_thresholds_met():
                self.done = True

            # Stagnation accounting: only free-walking steps count toward
            # the truncation threshold — scripted overlays and battles
            # freeze the counter rather than growing it (the player can't
            # claim cells there, and both can legitimately run long).
            # _last_cell_novel is maintained by _frontier_novelty_bonus
            # every valid step, independent of the bonus being enabled.
            if (
                not env_vars.get("script_active", False)
                and int(env_vars.get("battle_type", 0)) == 0
            ):
                if self._last_cell_novel:
                    self._stagnation_steps = 0
                else:
                    self._stagnation_steps += 1

                # Ground-truth dead-end detection: if the just-taken
                # directional press left (map, X, Y) unchanged, that
                # direction cannot pay off right now — regardless of what
                # the visit archive says about the coordinate one cell
                # over. directional_frontier_potential reads this to stop
                # forecasting reward for a wall/obstacle, which otherwise
                # reads as maximally fresh forever (see that method's
                # docstring). A non-directional action leaves whatever was
                # already known unchanged, since nothing about reachability
                # changed.
                cur_pos = (mb, mn, int(env_vars["X"]), int(env_vars["Y"]))
                if button_press in self._DIRECTION_BUTTONS:
                    self._blocked_direction = (
                        button_press if cur_pos == self._prev_pos else None
                    )
                self._prev_pos = cur_pos

            # Battle-progress watchdog accounting (see __init__). Runs on the
            # battle steps the free-walking counter above deliberately skips.
            # A turn that actually resolves changes either enemy HP (we dealt
            # damage / it fainted) or party HP (we took damage) — either
            # resets the counter. Consecutive battle steps that move neither
            # are the stuck signature. Outside battle the counter and its
            # baselines are cleared so the next fight starts fresh.
            if int(env_vars.get("battle_type", 0)) != 0:
                cur_enemy_hp = int(env_vars.get("enemy_hp", 0))
                cur_party_hp = int(env_vars["party_info"][2])
                progressed = (
                    self._prev_battle_enemy_hp is None
                    or cur_enemy_hp != self._prev_battle_enemy_hp
                    or cur_party_hp != self._prev_battle_party_hp
                )
                if progressed:
                    self._battle_stagnation_steps = 0
                else:
                    self._battle_stagnation_steps += 1
                self._prev_battle_enemy_hp = cur_enemy_hp
                self._prev_battle_party_hp = cur_party_hp
            else:
                self._battle_stagnation_steps = 0
                self._prev_battle_enemy_hp = None
                self._prev_battle_party_hp = None
        else:
            total = 0.0

        self.last_action = button_press

        if self.steps > self.max_steps:
            # Budget cut-off: truncated, not a natural terminal.
            self.done = True
            self.truncated = True

        if (
            self._stagnation_limit > 0
            and self._stagnation_steps >= self._stagnation_limit
        ):
            # Stagnation cut-off: the episode has gone a full threshold of
            # free-walking steps without touching a single new-this-episode
            # cell — it is absorbed in a loop (wall, pacing, or otherwise)
            # or has exhausted its reachable region. Truncated (time-limit
            # semantics, value bootstraps), NOT a terminal: nothing is
            # punished, the remaining budget is just reallocated to a
            # fresh attempt. See module docstring.
            self.done = True
            self.truncated = True

        if (
            self._battle_stagnation_limit > 0
            and self._battle_stagnation_steps >= self._battle_stagnation_limit
        ):
            # Battle-progress cut-off: a full threshold of consecutive battle
            # steps with no change in enemy or party HP — the fight is stuck
            # (see __init__). Truncated (time-limit, value bootstraps), NOT a
            # punished terminal, mirroring the free-walking stagnation cut-off
            # above. This is the backstop for the "36k steps mashing A in one
            # unwinnable battle" failure; the stuck-temperature mechanism is
            # what should break most stalls before this fires.
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

    def _milestone_refire_scale(self, kind, key):
        """Depletion factor for a milestone re-fire:
        ``1 / sqrt(1 + prior_episode_fire_count)`` from the run-wide
        archive — the same rule that gates cell/frontier income. A run-first
        fire pays in full; an every-episode re-fire decays toward zero but
        never fully dark (the corridor stays anchored, it just stops paying
        full rent). Without this, the first milestone on the corridor is a
        full-price annuity every episode and becomes a stable farming
        equilibrium that outbids all exploration (observed 2026-07-11).
        Replica staleness is bounded by one rollout, same as cells — worst
        case a genuinely-first fire pays full on a couple of parallel envs.
        """
        n = self.visit_archive.milestone_fire_count(kind, key)
        self._milestone_fires_pending.add((str(kind), int(key)))
        return 1.0 / math.sqrt(1.0 + n)

    def _global_flag_progress_bonus(self, story_flags):
        """Pays ``flag_progress_reward`` (scaled by the re-fire depletion,
        see ``_milestone_refire_scale``) for ANY fresh 0->1 transition among
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

        total = 0.0
        for flag_num, _name in _DERIVED_FLAG_TABLE:
            if self._flag_table_fired.get(flag_num, False):
                continue
            byte_idx, bit_idx = flag_num // 8, flag_num % 8
            now_bit = (story_flags[byte_idx] >> bit_idx) & 1
            start_bit = (self._flag_table_initial[byte_idx] >> bit_idx) & 1
            if now_bit == 1 and start_bit == 0:
                self._flag_table_fired[flag_num] = True
                total += self.flag_progress_reward * self._milestone_refire_scale(
                    "flag", flag_num
                )
                self._flags_fired_pending.add(int(flag_num))
                self.flag_fire_step_log.append([int(flag_num), int(self.steps)])
                # Reward-calc's own tracker above resets every episode by
                # design (flags re-fire every episode — no snapshot
                # seeding). The discovery log wants run-wide first-ever
                # instead, which only visit_archive (persistent across
                # episodes) can answer — its replica is at most one
                # rollout stale, same bound as new_map's global_count.
                if not self.visit_archive.flag_ever_fired(flag_num):
                    self._discoveries_this_episode.append(
                        {"type": "flag", "key": int(flag_num), "step": int(self.steps)}
                    )
        return total

    def _global_pokedex_bonus(self, pokedex_seen, pokedex_owned):
        """Pays for ANY new pokedex species (seen/owned), not just when a
        stage configures a pokedex goal. Baseline seeded on first call this
        episode (not hardcoded 0) so a mid-progress save-state isn't re-paid."""
        if self._prev_pokedex_seen is None:
            self._prev_pokedex_seen = int(pokedex_seen)
            self._prev_pokedex_owned = int(pokedex_owned)
            return 0.0
        prev_seen, prev_owned = self._prev_pokedex_seen, self._prev_pokedex_owned
        seen_delta = max(0, int(pokedex_seen) - prev_seen)
        owned_delta = max(0, int(pokedex_owned) - prev_owned)
        self._prev_pokedex_seen = int(pokedex_seen)
        self._prev_pokedex_owned = int(pokedex_owned)
        # Pay per count-threshold crossed, each depleted by how many prior
        # EPISODES already reached that threshold (keyed on the absolute
        # count, not the species — every episode restarts from the same
        # save-state, so "owned reached 1 again" is the re-fire being farmed,
        # while "owned reached 2 for the first time this run" pays in full).
        reward = 0.0
        for t in range(prev_seen + 1, int(pokedex_seen) + 1):
            reward += self.pokedex_seen_reward * self._milestone_refire_scale(
                "pokedex_seen", t
            )
        for t in range(prev_owned + 1, int(pokedex_owned) + 1):
            reward += self.pokedex_owned_reward * self._milestone_refire_scale(
                "pokedex_owned", t
            )
        # Discovery log wants "has ANY prior episode ever reached this many"
        # (run-wide, via visit_archive), not "did this episode's count just
        # increase" — every episode's counts start from 0 and climb the
        # same way, so a within-episode delta alone would (mis-)fire every
        # episode. "key" is the new absolute count reached, not the delta.
        if seen_delta and int(pokedex_seen) > self.visit_archive.pokedex_seen_max():
            self._discoveries_this_episode.append(
                {"type": "pokedex_seen", "key": int(pokedex_seen), "step": int(self.steps)}
            )
        if owned_delta and int(pokedex_owned) > self.visit_archive.pokedex_owned_max():
            self._discoveries_this_episode.append(
                {"type": "pokedex_owned", "key": int(pokedex_owned), "step": int(self.steps)}
            )
        self._episode_max_pokedex_seen = max(self._episode_max_pokedex_seen, int(pokedex_seen))
        self._episode_max_pokedex_owned = max(self._episode_max_pokedex_owned, int(pokedex_owned))
        return reward

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
        prev_total = self._prev_level_total
        gained = max(0, party_level - prev_total)
        self._prev_level_total = party_level
        if gained and party_level > self.visit_archive.level_max():
            self._discoveries_this_episode.append(
                {"type": "level", "key": int(party_level), "step": int(self.steps)}
            )
        self._episode_max_level = max(self._episode_max_level, int(party_level))
        # Per total-level threshold, depleted by prior-episode re-fires
        # (see _milestone_refire_scale) — grinding the same early levels
        # every episode stops being income; a new personal-best level
        # always pays in full.
        reward = 0.0
        for t in range(prev_total + 1, int(party_level) + 1):
            reward += self.level_up_reward * self._milestone_refire_scale("level", t)
        return reward

    def _global_key_item_bonus(self, key_items_count):
        """Pays per new key item picked up, always."""
        if self._prev_key_items_count is None:
            self._prev_key_items_count = int(key_items_count)
            return 0.0
        prev_count = self._prev_key_items_count
        gained = max(0, int(key_items_count) - prev_count)
        self._prev_key_items_count = int(key_items_count)
        if gained and int(key_items_count) > self.visit_archive.key_items_max():
            self._discoveries_this_episode.append(
                {"type": "key_item", "key": int(key_items_count), "step": int(self.steps)}
            )
        self._episode_max_key_items = max(self._episode_max_key_items, int(key_items_count))
        reward = 0.0
        for t in range(prev_count + 1, int(key_items_count) + 1):
            reward += self.key_item_pickup_reward * self._milestone_refire_scale(
                "key_item", t
            )
        return reward

    def _battle_progress(self, env_vars):
        """Capped, decaying reward for battle engagement, wins, and escapes.

        Engagement fires on the ``battle_type`` ``0 -> nonzero`` transition.
        Win fires when ``enemy_hp`` drops to 0 while still in battle (a KO,
        distinct from the player whiteing out or the enemy fleeing/being
        caught). Flee fires when a WILD battle (``battle_type == 1``) ends
        with the enemy still alive and the party not whited out — i.e. the
        agent successfully ran, the aligned outcome for a traversal-only
        curriculum where catching is out of scope (off unless
        ``battle_flee_reward > 0``). All three are clamped together to
        ``battle_reward_episode_cap`` — a decay-independent anti-farm
        backstop. This is deliberately generic: it doesn't special-case the
        early rival fight (there is no reliable "beat the rival" event flag
        for it — see gym_env.py's ``rival_cherrygrove`` comment) — winning
        ANY blocking trainer battle, including that one, pays through this
        same path. Flee cannot apply to trainer battles (you can't run from
        them), so it can never let the agent skip a mandatory fight.
        """
        cur_bt = int(env_vars.get("battle_type", 0))
        map_key = (int(env_vars["map_bank"]), int(env_vars["map_num"]))
        party_hp = int(env_vars["party_info"][2])
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
            # Wild battle just ended: reward a genuine escape (enemy still
            # alive => not a KO, party HP > 0 => not a whiteout). _prev_enemy_hp
            # still holds the last in-battle reading at this transition step.
            if (
                self.battle_flee_reward > 0
                and self._prev_battle_type == 1
                and self._prev_enemy_hp is not None
                and self._prev_enemy_hp > 0
                and party_hp > 0
            ):
                reward += self.battle_flee_reward
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
        Reward is skipped during scripted overlays (player position is
        stale), but the bookkeeping (``explored_maps`` / ``_recent_maps_list``
        / ``_maps_to_record``) always runs on first entry regardless of
        script state — otherwise a map whose first-ever entry happens to
        land on a scripted transition (a door/warp frame) would be silently
        and permanently dropped from the persistent archive, every episode
        it's rediscovered the exact same way.
        """
        map_key = (int(env_vars["map_bank"]), int(env_vars["map_num"]))
        loc = (env_vars["X"], env_vars["Y"], map_key[0], map_key[1])
        self.explored_tiles.add(loc)
        if not self.new_map_reward:
            return 0
        if map_key in self.explored_maps:
            return 0
        self.explored_maps.add(map_key)
        if map_key not in self._recent_maps_list:
            self._recent_maps_list.append(map_key)
        self._maps_to_record.add(map_key)
        global_count = self.visit_archive.map_count(map_key[0], map_key[1])
        if global_count == 0:
            # Genuinely never recorded before (as of the archive's last
            # broadcast) — a real run-wide discovery, not just new-to-this-
            # episode. This is the "map" event a discovery-order graph
            # would care about.
            self._discoveries_this_episode.append(
                {"type": "map", "key": [map_key[0], map_key[1]], "step": int(self.steps)}
            )
        if env_vars.get("script_active", False):
            return 0
        return float(self.new_map_reward) * max(
            1.0 / (global_count + 1), self.frontier_novelty_floor
        )

    def _frontier_novelty_bonus(self, env_vars):
        """Cell novelty, gated per episode and decayed by run-wide visits:
        ``frontier_novelty_bonus * max(1/sqrt(1 + persistent_visits),
        frontier_novelty_floor)``.

        A cell pays at most once per episode, on first entry that episode
        (``_novel_cells_this_episode``, so wiggling can't farm within an
        episode). The payout then decays with the persistent ``VisitArchive``
        count so re-covering known ground pays LESS across episodes — the
        gradient always points at the run's true frontier, where the count
        is 0 and the bonus pays in full — but never below the episodic
        floor: first-visit-this-episode is always income (NGU-style: the
        episodic signal never dies, the lifetime term only amplifies the
        frontier). The floor is what keeps a local gradient alive deep in
        fully-explored territory — without it every action in a decayed
        room pays identically (nothing) and absorbing loops become stable
        (see frontier_novelty_floor in __init__).

        The policy can SEE this decay: ``global_cell_visit_count`` in the
        RAM vector is the same archive count, so the reward stays a
        (near-)Markovian function of the observation rather than of hidden
        per-episode history.

        Also maintains the ``last_cell_novel`` / ``steps_since_novel_cell``
        bookkeeping the RAM vector exposes to the policy, independent of
        whether the bonus is configured on, and queues genuinely novel
        (first-this-episode) cells into ``_cells_to_record`` so the agent
        merges them into the persistent archive at episode end.

        Gated on ``script_active`` — during cutscenes / menus the player
        position is stale.
        """
        self._last_cell_novel = False
        self._steps_since_novel_cell += 1
        if env_vars.get("script_active", False):
            return 0.0
        mb, mn = env_vars["map_bank"], env_vars["map_num"]
        x, y = env_vars["X"], env_vars["Y"]
        cell = self.visit_archive.cell_key(mb, mn, x, y)
        if cell in self._novel_cells_this_episode:
            return 0.0
        self._novel_cells_this_episode.add(cell)
        self._cells_to_record.add(cell)
        self._last_cell_novel = True
        self._steps_since_novel_cell = 0

        if self.frontier_novelty_bonus <= 0:
            return 0.0
        persistent_visits = self.visit_archive.count(mb, mn, x, y)
        scale = 1.0 / math.sqrt(1.0 + persistent_visits)
        return float(self.frontier_novelty_bonus) * max(
            scale, self.frontier_novelty_floor
        )

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

    def global_cell_visit_count(self, env_vars):
        """Run-wide persistent visit count for the current cell (0 if never
        recorded). Lets the RAM vector expose the same cross-episode count
        that gates the frontier-novelty payout, so the policy can directly
        perceive how well-trodden a spot is across the whole run rather
        than having to infer it via reward."""
        return self.visit_archive.count(
            env_vars["map_bank"], env_vars["map_num"],
            env_vars["X"], env_vars["Y"],
        )

    # (dx, dy) in raw tile units, one CELL_SIZE step, in the order
    # (up, down, left, right). Y increases downward, X increases rightward
    # (screen convention) — an approximation for informational purposes
    # only (this never gates reward or truncates at a map edge), so a
    # wrong sign at worst mislabels the direction, it doesn't corrupt the
    # payout itself.
    _DIRECTION_OFFSETS = ((0, -1), (0, 1), (-1, 0), (1, 0))
    _DIRECTION_BUTTONS = ("up", "down", "left", "right")  # same order

    def directional_frontier_potential(self, env_vars):
        """Forecast, for each of (up, down, left, right), the frontier-
        novelty payout ``_frontier_novelty_bonus`` would give if the agent
        stepped one cell that way RIGHT NOW: 0.0 if that cell is already
        claimed this episode, else ``1 / sqrt(1 + persistent_visits)`` —
        the exact decay factor the reward pays (frontier_novelty_bonus
        itself is a scalar multiplier the policy doesn't need to see).

        Exists so the policy can PERCEIVE the exploration gradient
        directly instead of inferring "which way is still fresh" purely
        from correlating actions with scalar reward after the fact — the
        credit-assignment path the model otherwise has to learn is long
        (walk several steps, THEN get a reward, then work out which
        direction of travel caused it). Read-only: never mutates
        ``_novel_cells_this_episode`` — a lookahead, not a visit.

        Dead-end correction: the coordinate-arithmetic lookahead below has
        no idea whether the neighbouring cell is actually reachable. A
        wall/furniture/obstacle tile is a cell the player's (X, Y) can
        never equal, so ``visit_archive.count`` returns 0 for it — not just
        early in training, but PERMANENTLY, since nothing can ever visit
        it. That plugs into the same formula as the single freshest cell in
        the game (``max(1/sqrt(1+0), floor) == 1.0``, the ceiling), so an
        obstacle direction reads as maximally rewarding forever, an
        observation-level lure that no amount of training can decay away
        because it isn't a function of anything training affects. We
        correct this with ground truth instead of guessing at collision
        byte semantics: ``_blocked_direction`` (set in ``calculate_reward``)
        is whichever direction was just tried and failed to move the
        player, which self-corrects every step and needs no assumption
        about tile types — a ledge, NPC, cut-tree, or un-surfed water tile
        all read the same way (blocked now) and clear the same way (the
        first successful step through them).
        """
        mb, mn = env_vars["map_bank"], env_vars["map_num"]
        x, y = env_vars["X"], env_vars["Y"]
        out = []
        for (dx, dy), button in zip(self._DIRECTION_OFFSETS, self._DIRECTION_BUTTONS):
            if button == self._blocked_direction:
                out.append(0.0)
                continue
            nx, ny = x + dx * CELL_SIZE, y + dy * CELL_SIZE
            cell = self.visit_archive.cell_key(mb, mn, nx, ny)
            if cell in self._novel_cells_this_episode:
                out.append(0.0)
            else:
                n = self.visit_archive.count(mb, mn, nx, ny)
                # Mirror the floored payout exactly (see
                # _frontier_novelty_bonus) — the forecast must stay truthful
                # to what stepping there would actually pay.
                out.append(
                    max(1.0 / math.sqrt(1.0 + n), self.frontier_novelty_floor)
                )
        return out

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
