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

  Exploration (per-episode curiosity — the exploration DRIVE):
    frontier novelty: a flat `frontier_novelty_bonus` for the first step
    onto each cell THIS EPISODE (the per-episode gate stops within-episode
    wiggling from farming). By default there is NO cross-episode decay and
    NO floor: every episode is its own exploration problem, so covering
    genuinely-new-this-episode ground always pays the same and the only way
    to earn MORE is to reach further than you have this episode. This is a
    deliberate design choice (2026-07-16 rebuild): reward a transferable
    "keep finding new ground" skill, not a policy that memorises a single
    training-wide golden path. The prior design (run-wide 1/sqrt(visits)
    decay + a floor) inverted the incentive — the floor made re-sweeping
    the known region a permanent ~2/cell annuity that out-paid discovering
    a new region, so the agent camped one bank and never crossed.
    An OPTIONAL NGU lifelong term (`frontier_lifelong_decay`, default off)
    re-introduces the 1/sqrt(1 + run-wide visits) multiplier — the one
    place that uses training-wide global knowledge — for when pure
    per-episode coverage plateaus; it is pure reward-shaping (never in the
    observation, never teleports the agent).
    new-map / new-bank: flat `new_map_reward` for the first entry to a map
    this episode, plus a larger `new_bank_reward` the first time a whole
    new BANK (region) is entered — so pushing the frontier into new
    territory always beats re-covering the known region. Both flat and
    episodic (no run-wide decay).
    The policy also sees an egocentric per-episode visited mask
    (local_visited_mask) so it can perceive "which nearby cells are still
    fresh this episode" directly instead of only inferring it from reward.

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

from PoliwhiRL.checkpoints import DERIVED_FLAG_TABLE, is_recordable_checkpoint
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
        self.visit_archive = (
            visit_archive if visit_archive is not None else VisitArchive()
        )

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
        self.checkpoint_progress_reward = config.get(
            "checkpoint_progress_reward", self.flag_progress_reward
        )
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

        # ---- Exploration (per-episode curiosity — see module docstring) ----
        # Flat per-episode first-visit coverage reward. Each cell pays
        # ``frontier_novelty_bonus`` the first time it is stepped onto THIS
        # episode (the per-episode gate below stops within-episode wiggling
        # from farming it). There is NO floor and, by default, NO
        # cross-episode decay: every episode is its own exploration problem,
        # so the only way to earn MORE coverage reward is to reach further
        # than you already have this episode. This deliberately trains a
        # transferable "keep finding new ground" skill rather than a policy
        # that memorises a single training-wide golden path.
        self.frontier_novelty_bonus = float(config.get("frontier_novelty_bonus", 1.0))
        # First entry to a new map (and, bigger, a whole new BANK/region —
        # e.g. crossing Route 29 -> Cherrygrove) this episode. Flat and
        # episodic. new_bank_reward dwarfs a home re-sweep's marginal value
        # so pushing into genuinely new territory always beats re-covering
        # the known region.
        self.new_map_reward = float(config.get("new_map_reward", 5.0))
        self.new_bank_reward = float(config.get("new_bank_reward", 20.0))
        # OPTIONAL NGU-style lifelong term (default OFF — the one place that
        # uses training-wide global knowledge). When enabled, the per-cell
        # coverage payout is multiplied by 1/sqrt(1 + run-wide visit count),
        # so ground the whole run has swept many times stops paying and only
        # the frontier pays full. This is pure reward-shaping (never in the
        # observation, never teleports the agent) — flip it on only if pure
        # per-episode coverage plateaus. No floor: decay is allowed to go to
        # ~0 on stale ground (that's the point).
        self.frontier_lifelong_decay = bool(
            config.get("frontier_lifelong_decay", False)
        )
        # OPTIONAL NGU-style lifelong term for the MAP/BANK reward (default
        # OFF). The same novelty principle as frontier_lifelong_decay, one
        # level up: when enabled, new_map/new_bank payouts are multiplied by
        # 1/sqrt(1 + run-wide entry count), so re-entering a map the run has
        # toured thousands of times pays ~0 and only genuinely-new regions pay
        # full. Kills the "tour the known building cluster every episode for a
        # flat map bonus" local optimum without teaching a route — it is
        # reward-shaping only (never in the observation), so the policy still
        # sees purely egocentric/per-episode state.
        self.map_lifelong_decay = bool(config.get("map_lifelong_decay", False))
        # ---- Revisit re-reward on story progress ----
        # A story beat (any OBSERVABLE derived-table flag flip — the same
        # flags the policy sees in its RAM vector) genuinely changes the world:
        # a route opens, an NPC needs re-visiting. When that happens the whole
        # explored region stops being "done" and backtracking through it should
        # pay again — otherwise the return leg to the professor is a dead,
        # gradient-less corridor. On a flag flip we re-open coverage
        # (_novel_cells_this_episode is cleared) so cells can pay once more,
        # but a cell the agent has ALREADY visited this episode re-pays at
        # ``revisit_novelty_scale`` of full while genuinely-new-this-episode
        # ground still pays full. The strict ordering new(1.0) > revisit(scale)
        # > nothing(0) keeps the frontier gradient dominant — backtracking is
        # rewarded without out-bidding real exploration, and because the flag
        # bit is in the observation the policy can learn "something changed ->
        # re-explore". 0 disables the re-reward (a flip then only re-opens
        # full-price on truly-new ground).
        self.revisit_novelty_scale = float(config.get("revisit_novelty_scale", 0.5))
        self.flag_reopens_novelty = bool(config.get("flag_reopens_novelty", True))
        # ---- Menu / UI-state novelty (behavioural exploration) ----
        # Coverage novelty over the discrete UI/menu context, not just the map
        # cell: pays ``menu_novelty_bonus`` the first time each distinct
        # (battle_type, ui_byte, map_handler_byte) tuple is reached this
        # episode (revisit-discounted and flag-re-opened exactly like tiles).
        # A saturated "always attack" battle policy has no reason to open the
        # BAG and find the ball; making unseen menu contexts intrinsically
        # worth reaching pulls it to explore the menu tree, where the pokedex/
        # party payoff for actually catching then takes over. General across
        # every menu (bag, party-switch, item) — not a hardcoded "throw ball"
        # reward. 0 disables. Bounded per episode (finite distinct contexts,
        # each pays once per epoch), so it cannot be farmed.
        self.menu_novelty_bonus = float(config.get("menu_novelty_bonus", 0.0))
        # ---- Event-flag novelty (the interaction/story-gate drive, OFF by
        # default). The curated milestone reward above only pays for the ~48
        # flags a human hand-listed in gym_env._DERIVED_FLAG_TABLE; the dense
        # INTERMEDIATE event flags Crystal sets for small interactions (talked
        # to NPC, received item, script step reached) pay nothing and give no
        # gradient toward the interactions that cross a story gate. When
        # enabled, this pays event_novelty_bonus for the first 0->1 flip THIS
        # episode of ANY bit in the whole wEventFlags region (0xDA72-0xDB71),
        # depleted by 1/sqrt(1 + run-wide fire count) so a genuinely-new
        # interaction pays full while a bit that flips every episode (clock /
        # sprite-visibility churn — the noisy-TV failure mode) is depleted to
        # ~0 within a handful of episodes. Area-invariant and hand-authors no
        # golden path — it just rewards "make the world change in a way it
        # hasn't before". Off by default because a few flag regions are
        # non-monotonic; exclude the documented-bad bits (26 transient,
        # 1726 sprite-visibility) and any others via event_novelty_exclude_flags.
        self.event_novelty_enabled = bool(config.get("event_novelty_enabled", False))
        self.event_novelty_bonus = float(config.get("event_novelty_bonus", 2.0))
        self._event_novelty_exclude = set(
            int(f) for f in config.get("event_novelty_exclude_flags", [26, 1726])
        )
        # Track which event-flag bits fire each step whenever EITHER the
        # event-novelty reward OR flag-state Go-Explore capture needs it
        # (gym_env reads _last_step_event_fires to snapshot "verge of the next
        # event" states). Decoupled from the reward so capture works even if
        # the reward weight is 0.
        self._track_event_fires = self.event_novelty_enabled or bool(
            config.get("goexplore_flag_capture", False)
        )
        # [(bit, run_wide_fire_count), ...] fired THIS step (see
        # _detect_event_fires). Cleared every step in calculate_reward.
        self._last_step_event_fires = []
        # Egocentric local visited-this-episode mask exposed to the policy
        # (see local_visited_mask): a (2R+1)x(2R+1) grid of cells centred on
        # the player, 1 where already visited this episode. Resets every
        # episode; fully egocentric so the "move toward the fresh cells"
        # skill transfers to any map.
        self._visited_mask_radius = int(config.get("visited_mask_radius", 2))
        # Egocentric per-episode frontier-direction sense (see
        # frontier_direction): a wider window than the fixed mask, collapsed
        # into a single unit vector pointing at unexplored-this-episode
        # ground plus a local-saturation scalar. 0 disables the feature.
        self._frontier_sense_radius = int(config.get("frontier_sense_radius", 6))

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

        # Battle-flee reward REMOVED: rewarding a successful escape from a
        # wild battle was easy, farmable income that taught permanent battle
        # avoidance (never builds a party — bites at the rival/gyms). Wild
        # battles are handled purely by the battle-stagnation watchdog above.

        self.clip = config.get("reward_clip", 1000)

        # Optional rounding of the per-step reward to this many decimal
        # places — keeps logs / PNG filenames legible. Default 2.
        self.reward_round_dp = config.get("reward_round_dp", 2)

        # Per-episode novelty / progress trackers.
        self._novel_cells_this_episode = set()
        # True per-episode visit history (never cleared until episode end),
        # separate from _novel_cells_this_episode which is the "paid at the
        # current reward epoch" gate that a flag flip re-opens. Determines
        # whether a re-paid cell is a revisit (discounted) vs genuinely new,
        # and backs the honest visited-mask / frontier-direction observation
        # features so they stay monotonic across a flag re-open.
        self._ever_visited_cells_this_episode = set()
        # Menu/UI-state novelty (see _menu_novelty_bonus): paid-this-epoch gate
        # and true-history set, mirroring the cell sets above.
        self._novel_menu_states_this_episode = set()
        self._ever_menu_states_this_episode = set()
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
        # Per-episode set of banks (regions) entered — first entry to a new
        # bank pays new_bank_reward.
        self.explored_banks = set()

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
        # (kind, key) milestone thresholds already PAID this episode — the
        # within-episode dedup that stops a non-monotonic counter (or a farm
        # loop) from re-paying the same threshold every up-tick. See
        # _milestone_refire_scale.
        self._milestone_thresholds_paid = set()
        # Event-flag novelty per-episode state (see _event_novelty_bonus):
        # snapshot of the wEventFlags bytes at episode start, the set of bits
        # already paid this episode, and the pending set of bits fired this
        # episode (merged into the archive's run-wide fire-count ledger).
        self._event_flags_initial = None
        self._event_flags_fired = set()
        self._event_flags_fired_pending = set()
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
            "checkpoint": 0.0,
            "map_goal": 0.0,
            "maps_visited": 0.0,
            "pokedex": 0.0,
            "key_item": 0.0,
            "battle": 0.0,
            "level": 0.0,
            "frontier": 0.0,
            "menu": 0.0,
            "new_map": 0.0,
            "event": 0.0,
            "step_penalty": 0.0,
            "whiteout": 0.0,
        }

        # Always-on global-progress trackers (see module docstring). Baseline
        # is seeded on first call each episode, not hardcoded to 0/empty, so
        # a save-state that starts mid-progress doesn't get re-paid for
        # already-true state.
        self._flag_table_initial = None
        self._flag_table_fired = {}
        self._last_checkpoint_progress_reward = 0.0
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
        # Which cut-off ended the episode: None (natural terminal / still
        # running), "budget" (time-limit — bootstraps), or "stagnation" /
        # "battle_stagnation" (stuck — zero-bootstrap terminal). Only "budget"
        # is bootstrapped in the GAE (see vec_env / _per_env_gae).
        self.truncation_cause = None
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
        self.truncation_cause = None
        self.last_action = None
        self.steps = 0
        self.cumulative_reward = 0
        self._prev_party_size = None
        self._prev_party_hp = None
        self.whiteouts = 0
        self._novel_cells_this_episode = set()
        self._ever_visited_cells_this_episode = set()
        self._novel_menu_states_this_episode = set()
        self._ever_menu_states_this_episode = set()
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
        self.explored_banks = set()
        self._recent_maps_list = []
        self._discoveries_this_episode = []
        self._flags_fired_pending = set()
        self._milestone_fires_pending = set()
        self._milestone_thresholds_paid = set()
        self._event_flags_initial = None
        self._event_flags_fired = set()
        self._event_flags_fired_pending = set()
        self._last_step_event_fires = []
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
            # Event-flag bits that fired 0->1 this episode over the whole
            # wEventFlags region — merged into the archive's run-wide
            # fire-count ledger that backs the event-novelty decay.
            "event_flags_fired": sorted(self._event_flags_fired_pending),
        }

    # ------------------------------------------------------------------ #
    # Main reward calculation                                             #
    # ------------------------------------------------------------------ #

    def calculate_reward(self, env_vars, button_press):
        self.steps += 1
        # Only holds THIS step's event-flag fires; invalid frames report none.
        self._last_step_event_fires = []

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
            r_checkpoint = self._last_checkpoint_progress_reward
            r_pokedex = self._global_pokedex_bonus(
                env_vars["pokedex_seen"], env_vars["pokedex_owned"]
            )
            r_level = self._global_level_bonus(party_size, party_level)
            r_key_item = self._global_key_item_bonus(env_vars.get("key_items_count", 0))

            r_battle = self._battle_progress(env_vars)
            r_map = self._new_map_bonus(env_vars)
            r_front = self._frontier_novelty_bonus(env_vars)
            r_menu = self._menu_novelty_bonus(env_vars)
            if self._track_event_fires:
                self._last_step_event_fires = self._detect_event_fires(
                    env_vars["story_flags"]
                )
            r_event = self._event_novelty_bonus()
            r_wo = self._check_whiteout(env_vars)

            self._episode_breakdown["flag"] += r_flag - r_checkpoint
            self._episode_breakdown["checkpoint"] += r_checkpoint
            self._episode_breakdown["map_goal"] += r_map_goal
            self._episode_breakdown["maps_visited"] += r_maps_visited
            self._episode_breakdown["pokedex"] += r_pokedex
            self._episode_breakdown["key_item"] += r_key_item
            self._episode_breakdown["level"] += r_level
            self._episode_breakdown["battle"] += r_battle
            self._episode_breakdown["new_map"] += r_map
            self._episode_breakdown["frontier"] += r_front
            self._episode_breakdown["menu"] += r_menu
            self._episode_breakdown["event"] += r_event
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
                + r_menu
                + r_event
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
            # Budget cut-off: a genuine time-limit truncation. The episode was
            # still making progress and merely ran out of clock, so the value
            # SHOULD bootstrap the (unobserved) continuation. truncation_cause
            # == "budget" is the only cause the agent bootstraps on (see
            # vec_env terminal_info / _per_env_gae).
            self.done = True
            self.truncated = True
            self.truncation_cause = "budget"

        if (
            self._stagnation_limit > 0
            and self._stagnation_steps >= self._stagnation_limit
        ):
            # Stagnation cut-off: the episode has gone a full threshold of
            # free-walking steps without touching a single new-this-episode
            # cell — it is absorbed in a loop (wall, pacing, or otherwise) or
            # has exhausted its reachable region. This is NOT bootstrapped: it
            # is treated as a zero-bootstrap terminal (truncation_cause is
            # "stagnation", which the agent excludes from the GAE bootstrap
            # mask). Rationale: bootstrapping here made the critic value the
            # stuck state at ~V(post-reset) — a fresh, novelty-rich episode —
            # so being absorbed in a dead pocket looked like a cheap route to
            # a high-value reset, and there was zero advantage to leaving.
            # Cutting the bootstrap gives the stuck region a low (near-zero)
            # value, so escaping it finally has positive advantage.
            self.done = True
            self.truncated = True
            # Do not overwrite a budget cause set the same step (budget wins;
            # it is the honest continuation semantics).
            if self.truncation_cause is None:
                self.truncation_cause = "stagnation"

        if (
            self._battle_stagnation_limit > 0
            and self._battle_stagnation_steps >= self._battle_stagnation_limit
        ):
            # Battle-progress cut-off: a full threshold of consecutive battle
            # steps with no change in enemy or party HP — the fight is stuck
            # (see __init__). Like the free-walking stagnation cut-off above
            # this is a zero-bootstrap terminal (truncation_cause
            # "battle_stagnation"), NOT a bootstrapped time-limit: a fight
            # going nowhere should not inherit the value of the fresh episode
            # that follows the reset. This is the backstop for the "36k steps
            # mashing A in one unwinnable battle" failure; the stuck-
            # temperature mechanism is what should break most stalls first.
            self.done = True
            self.truncated = True
            if self.truncation_cause is None:
                self.truncation_cause = "battle_stagnation"

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
        if self._prev_party_size is not None and party_size != self._prev_party_size:
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

        Within a single episode, each ``(kind, key)`` milestone threshold pays
        at most ONCE. A monotonic counter crosses each threshold exactly once,
        so this is a no-op for well-behaved progress. A *re-cross* means the
        underlying count wobbled down and back up — a non-monotonic RAM read,
        an item tossed and re-picked-up, a menu-reorder, or an outright farm
        loop. Without this guard the threshold-crossing loops
        (``_global_key_item_bonus`` / ``_global_pokedex_bonus`` /
        ``_global_level_bonus``) re-paid the same threshold every up-tick,
        while the run-wide depletion above could never catch up: the
        ``_milestone_fires_pending`` set records only ONE fire per episode no
        matter how many times it re-crossed, so ``n`` grew ~1/episode while the
        farm paid thousands of times at that same barely-depleted scale.
        Observed 2026-07-20: ``key_items_count`` oscillation drove the
        ``key_item`` source to ~17k/episode and collapsed exploration (honest
        unique_maps 16 -> 5.6). Deduping per episode aligns payment with the
        ledger (one fire == one pending entry == +1 archive count).
        """
        mkey = (str(kind), int(key))
        if mkey in self._milestone_thresholds_paid:
            return 0.0
        self._milestone_thresholds_paid.add(mkey)
        n = self.visit_archive.milestone_fire_count(kind, key)
        self._milestone_fires_pending.add(mkey)
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
        if self._flag_table_initial is None:
            self._flag_table_initial = bytes(story_flags)

        total = 0.0
        self._last_checkpoint_progress_reward = 0.0
        for flag_num, _name in DERIVED_FLAG_TABLE:
            if self._flag_table_fired.get(flag_num, False):
                continue
            byte_idx, bit_idx = flag_num // 8, flag_num % 8
            now_bit = (story_flags[byte_idx] >> bit_idx) & 1
            start_bit = (self._flag_table_initial[byte_idx] >> bit_idx) & 1
            if now_bit == 1 and start_bit == 0:
                self._flag_table_fired[flag_num] = True
                reward_value = (
                    self.checkpoint_progress_reward
                    if is_recordable_checkpoint(flag_num)
                    else self.flag_progress_reward
                )
                paid = reward_value * self._milestone_refire_scale("flag", flag_num)
                total += paid
                if is_recordable_checkpoint(flag_num):
                    self._last_checkpoint_progress_reward += paid
                self._flags_fired_pending.add(int(flag_num))
                self.flag_fire_step_log.append([int(flag_num), int(self.steps)])
                # Story beat -> the world changed. Re-open coverage / menu
                # novelty so backtracking through already-explored ground pays
                # again (discounted for revisits, full for genuinely-new
                # ground — see _frontier_novelty_bonus / _menu_novelty_bonus).
                # The _ever_* history sets are intentionally NOT cleared, so
                # revisits stay identifiable and the observation features stay
                # monotonic. Clearing is idempotent across multiple flags in
                # one step.
                if self.flag_reopens_novelty:
                    self._novel_cells_this_episode.clear()
                    self._novel_menu_states_this_episode.clear()
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
                {
                    "type": "pokedex_seen",
                    "key": int(pokedex_seen),
                    "step": int(self.steps),
                }
            )
        if owned_delta and int(pokedex_owned) > self.visit_archive.pokedex_owned_max():
            self._discoveries_this_episode.append(
                {
                    "type": "pokedex_owned",
                    "key": int(pokedex_owned),
                    "step": int(self.steps),
                }
            )
        self._episode_max_pokedex_seen = max(
            self._episode_max_pokedex_seen, int(pokedex_seen)
        )
        self._episode_max_pokedex_owned = max(
            self._episode_max_pokedex_owned, int(pokedex_owned)
        )
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
            # Register a party-size milestone on GROWTH (caught / received a
            # Pokemon) so the Go-Explore "caught" capture (gym_env) can rank
            # post-catch snapshots by run-wide rarity — same milestone ledger
            # the reward decay uses. Registration only; pays no reward here
            # (rewarding party growth directly would be farmable via
            # catch/release). A shrink (whiteout, release) registers nothing.
            if party_size > self._prev_level_party_size:
                self._milestone_fires_pending.add(("party_size", int(party_size)))
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
                {
                    "type": "key_item",
                    "key": int(key_items_count),
                    "step": int(self.steps),
                }
            )
        self._episode_max_key_items = max(
            self._episode_max_key_items, int(key_items_count)
        )
        reward = 0.0
        for t in range(prev_count + 1, int(key_items_count) + 1):
            reward += self.key_item_pickup_reward * self._milestone_refire_scale(
                "key_item", t
            )
        return reward

    def _battle_progress(self, env_vars):
        """Capped, decaying reward for battle engagement and wins.

        Engagement fires on the ``battle_type`` ``0 -> nonzero`` transition.
        Win fires when ``enemy_hp`` drops to 0 while still in battle (a KO,
        distinct from the player whiteing out or the enemy fleeing/being
        caught). Both terms are clamped together to
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

        if (
            self._prev_battle_type is not None
            and self._prev_battle_type == 0
            and cur_bt != 0
        ):
            n = self._battle_engaged_maps.get(map_key, 0) + 1
            self._battle_engaged_maps[map_key] = n
            reward += self.battle_engagement_reward / (
                1 + self.battle_decay_coef * (n - 1)
            )

        if cur_bt != 0:
            enemy_hp = int(env_vars.get("enemy_hp", 0))
            if (
                self._prev_enemy_hp is not None
                and self._prev_enemy_hp > 0
                and enemy_hp == 0
            ):
                n = self._battle_won_maps.get(map_key, 0) + 1
                self._battle_won_maps[map_key] = n
                reward += self.battle_win_reward / (
                    1 + self.battle_decay_coef * (n - 1)
                )
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
        """Flat, per-episode bonus for reaching new territory this episode.

        Pays ``new_map_reward`` on first entry to a (map_bank, map_num) this
        episode, plus an additional ``new_bank_reward`` the first time a whole
        new BANK (region) is entered this episode. Both fire once per episode.

        By default both are FLAT (no run-wide decay). With ``map_lifelong_decay``
        on, each payout is multiplied by ``1/sqrt(1 + run-wide entry count)``
        (map count for the map term, bank count for the bank term), so touring
        a map/region the run has already entered thousands of times pays ~0
        while a genuinely-new map/bank still pays full. This dissolves the
        "re-tour the known building cluster every episode for a flat map
        bonus" optimum. Unlike the old depletion-plus-floor scheme, there is
        NO floor and it is a strict novelty signal (rarer territory pays more),
        so it can't invert into re-sweeping out-paying discovery.

        Also updates ``explored_tiles`` for the observation layer.
        Reward is skipped during scripted overlays (player position is
        stale), but the bookkeeping (``explored_maps`` / ``_recent_maps_list``
        / ``_maps_to_record``) always runs on first entry regardless of
        script state — otherwise a map whose first-ever entry happens to
        land on a scripted transition (a door/warp frame) would be silently
        and permanently dropped from the persistent archive, every episode
        it's rediscovered the exact same way.
        """
        bank, num = int(env_vars["map_bank"]), int(env_vars["map_num"])
        map_key = (bank, num)
        loc = (env_vars["X"], env_vars["Y"], bank, num)
        self.explored_tiles.add(loc)
        reward = 0.0
        new_bank = bank not in self.explored_banks
        self.explored_banks.add(bank)
        if map_key not in self.explored_maps:
            self.explored_maps.add(map_key)
            if map_key not in self._recent_maps_list:
                self._recent_maps_list.append(map_key)
            self._maps_to_record.add(map_key)
            if self.visit_archive.map_count(bank, num) == 0:
                # Genuinely never recorded before (as of the archive's last
                # broadcast) — a real run-wide discovery for the diagnostic
                # discovery-order log (never read by reward/observation).
                self._discoveries_this_episode.append(
                    {"type": "map", "key": [bank, num], "step": int(self.steps)}
                )
            if not env_vars.get("script_active", False):
                if self.map_lifelong_decay:
                    reward += self.new_map_reward / math.sqrt(
                        1.0 + self.visit_archive.map_count(bank, num)
                    )
                    if new_bank:
                        reward += self.new_bank_reward / math.sqrt(
                            1.0 + self.visit_archive.bank_count(bank)
                        )
                else:
                    reward += self.new_map_reward
                    if new_bank:
                        reward += self.new_bank_reward
        return reward

    def _frontier_novelty_bonus(self, env_vars):
        """Per-episode coverage reward: ``frontier_novelty_bonus`` for the
        first step onto each cell THIS episode.

        A cell pays at most once per episode (``_novel_cells_this_episode``,
        so wiggling can't farm it within an episode). By default the payout
        is FLAT — every episode is its own exploration problem and covering
        genuinely-new-this-episode ground always pays the same, so the only
        way to earn more is to reach further than you have this episode.

        Optional NGU lifelong term (``frontier_lifelong_decay``, default
        off): multiply by ``1/sqrt(1 + run-wide visit count)`` so ground the
        whole run has swept many times stops paying and only the frontier
        pays full. No floor — decay is allowed to reach ~0 on stale ground.

        Also maintains the ``last_cell_novel`` / ``steps_since_novel_cell``
        bookkeeping the stagnation-truncation counter reads, and queues
        genuinely novel (first-this-episode) cells into ``_cells_to_record``
        so the agent can merge them into the persistent archive at episode
        end (used for the optional lifelong term and the discovery log).

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
        # Already paid at the current reward epoch (since episode start, or
        # since the last story-flag re-open): no further pay until the next
        # re-open. Blocks in-episode wiggling from farming a cell.
        if cell in self._novel_cells_this_episode:
            return 0.0
        # A cell in the true-history set has been walked before this episode:
        # after a flag re-open it re-pays at the discounted revisit rate;
        # genuinely-new-this-episode ground pays full. Before any re-open the
        # two sets are identical, so this branch never triggers and behaviour
        # is exactly the old flat per-episode coverage.
        is_revisit = cell in self._ever_visited_cells_this_episode
        self._novel_cells_this_episode.add(cell)
        self._ever_visited_cells_this_episode.add(cell)
        self._cells_to_record.add(cell)
        # Any PAID step (full or discounted) counts as progress for the
        # stagnation watchdog, so a productive backtrack through re-opened
        # ground is not truncated as a stall.
        self._last_cell_novel = True
        self._steps_since_novel_cell = 0

        if self.frontier_novelty_bonus <= 0:
            return 0.0
        scale = self.revisit_novelty_scale if is_revisit else 1.0
        if self.frontier_lifelong_decay:
            persistent_visits = self.visit_archive.count(mb, mn, x, y)
            return (
                float(self.frontier_novelty_bonus)
                * scale
                / math.sqrt(1.0 + persistent_visits)
            )
        return float(self.frontier_novelty_bonus) * scale

    def _menu_novelty_bonus(self, env_vars):
        """Per-episode coverage novelty over the UI/menu context.

        Keys on ``(battle_type, ui_byte, map_handler_byte)`` — the discrete
        mode bytes that distinguish overworld from a battle's fight menu, bag,
        ball pocket, party-switch screen, etc. Pays ``menu_novelty_bonus`` the
        first time each distinct context is reached this reward epoch, at the
        discounted rate for a context seen earlier this episode (mirroring the
        cell revisit logic, and re-opened by a story-flag flip the same way).
        This makes exploring the menu tree intrinsically worthwhile so a
        saturated battle policy is pulled to try the bag / switch, without any
        hardcoded per-action reward. Bounded (finite contexts, once per epoch
        each) so it cannot be farmed. Not gated on ``script_active`` — the menu
        bytes are exactly the states we want to reward reaching."""
        if self.menu_novelty_bonus <= 0:
            return 0.0
        key = (
            int(env_vars.get("battle_type", 0)),
            int(env_vars.get("ui_byte", 0)),
            int(env_vars.get("map_handler_byte", 0)),
        )
        if key in self._novel_menu_states_this_episode:
            return 0.0
        is_revisit = key in self._ever_menu_states_this_episode
        self._novel_menu_states_this_episode.add(key)
        self._ever_menu_states_this_episode.add(key)
        scale = self.revisit_novelty_scale if is_revisit else 1.0
        return float(self.menu_novelty_bonus) * scale

    def _detect_event_fires(self, story_flags):
        """Return ``[(bit, run_wide_fire_count), ...]`` for every event-flag
        bit that flipped 0->1 for the FIRST time THIS episode, over the whole
        wEventFlags region (0xDA72-0xDB71) — the dense interaction signal the
        curated ~48-flag milestone table lacks. Marks each as fired (so it pays
        at most once per episode) and queues it for the archive's run-wide
        ledger. Snapshots the flag bytes on the first call so flags already set
        at episode start (baked into the save-state — e.g. a Go-Explore seed)
        never fire. Excludes non-monotonic bits (``event_novelty_exclude_flags``).

        Backs BOTH the event-novelty reward (_event_novelty_bonus) and
        flag-state Go-Explore capture (gym_env reads _last_step_event_fires),
        so it runs whenever either is on. Not gated on ``script_active`` — an
        event flag flipping DURING a cutscene is exactly the signal we want
        (that's when NPC/story flags are set).
        """
        cur = np.asarray(story_flags, dtype=np.uint8)
        if self._event_flags_initial is None:
            self._event_flags_initial = cur.copy()
            return []
        # Bytes with at least one bit newly set vs episode start. unpackbits is
        # little-endian so bit index == flag_num convention (LSB-first within a
        # byte), matching _DERIVED_FLAG_TABLE's flag_num // 8, % 8.
        newly_set_bytes = cur & ~self._event_flags_initial
        if not newly_set_bytes.any():
            return []
        fired_bits = np.flatnonzero(np.unpackbits(newly_set_bytes, bitorder="little"))
        fires = []
        for bit in fired_bits.tolist():
            if bit in self._event_flags_fired or bit in self._event_novelty_exclude:
                continue
            self._event_flags_fired.add(bit)
            self._event_flags_fired_pending.add(int(bit))
            fires.append((int(bit), self.visit_archive.event_flag_fire_count(bit)))
        return fires

    def _event_novelty_bonus(self):
        """Pay ``event_novelty_bonus`` for each event-flag bit that fired this
        step (see _detect_event_fires), depleted by ``1/sqrt(1 + run-wide fire
        count)`` — full for a genuinely-new interaction, ~0 for a bit that
        flips every episode (clock/sprite churn). Reads the fires detected in
        calculate_reward; off unless ``event_novelty_enabled``."""
        if not self.event_novelty_enabled or self.event_novelty_bonus <= 0:
            return 0.0
        reward = 0.0
        for _bit, n in self._last_step_event_fires:
            reward += self.event_novelty_bonus / math.sqrt(1.0 + n)
        return reward

    def rare_event_fires(self, count_max):
        """Event-flag bits that fired THIS step whose run-wide fire count is
        <= count_max — the rare-flag signal gym_env uses to snapshot a
        'verge of the next event' Go-Explore state."""
        return [(b, n) for (b, n) in self._last_step_event_fires if n <= count_max]

    def event_fires(self):
        """All event-flag transitions detected on the current step."""
        return list(self._last_step_event_fires)

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
        recent = self._recent_maps_list[-self._recent_maps_n :]
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

    def stagnation_fraction(self):
        """Fraction of the stagnation-truncation budget consumed, in [0, 1].

        ``_stagnation_steps`` counts consecutive FREE-WALKING steps with no
        new-this-episode cell (it freezes during battles/scripts, exactly
        like the watchdog that truncates on it), so this reaches 1.0 the step
        the watchdog fires. Exposed in the RAM vector as the ``stagnation_
        clock`` feature: it de-aliases "just arrived at this tile" from "stuck
        here for hundreds of steps" (identical otherwise), and is the same
        signal the behaviour-time stuck-temperature ramp keys off. Returns 0.0
        when the watchdog is disabled (no meaningful clock)."""
        if self._stagnation_limit <= 0:
            return 0.0
        return min(1.0, self._stagnation_steps / float(self._stagnation_limit))

    def local_visited_mask(self, env_vars):
        """Egocentric (2R+1)x(2R+1) grid, row-major, of whether each nearby
        cell has been visited THIS episode (1.0) or is still fresh (0.0),
        centred on the player's current cell (which is always 1.0 once
        stepped onto). ``R = _visited_mask_radius``.

        This is the policy's per-episode "where have I been near me" memory:
        it resets every episode and is fully egocentric, so the skill it
        teaches — "move toward the 0s" — transfers to any map rather than
        encoding a fixed route. Replaces the old run-wide global visit count
        and directional frontier-potential features (deleted: they exposed
        training-wide global knowledge and an obstacle-lure artefact).

        Read-only: a lookahead over ``_ever_visited_cells_this_episode`` (the
        true, monotonic episode history — NOT the paid-this-epoch gate a story
        flag re-opens), never a visit. During a scripted overlay the player
        position is stale, so return all-zeros (nothing meaningful to
        report)."""
        R = self._visited_mask_radius
        n = 2 * R + 1
        if env_vars.get("script_active", False):
            return [0.0] * (n * n)
        mb, mn = env_vars["map_bank"], env_vars["map_num"]
        x, y = env_vars["X"], env_vars["Y"]
        out = []
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                cell = self.visit_archive.cell_key(
                    mb, mn, x + dx * CELL_SIZE, y + dy * CELL_SIZE
                )
                out.append(
                    1.0 if cell in self._ever_visited_cells_this_episode else 0.0
                )
        return out

    def frontier_direction(self, env_vars):
        """Egocentric, per-episode exploration gradient. Returns
        ``[dir_x, dir_y, local_saturation]``:

        - ``(dir_x, dir_y)``: a unit vector, in egocentric map axes, pointing
          toward the mass of cells NOT yet visited this episode within
          ``_frontier_sense_radius`` cells, inverse-square weighted so the
          nearest unexplored opening dominates. ``(0, 0)`` when the window is
          fully fresh or symmetrically swept (no directional gradient).
        - ``local_saturation`` in ``[0, 1]``: fraction of the window already
          visited this episode — how boxed-in the agent is by its own trail.

        This is the longer-range companion to ``local_visited_mask``: the mask
        is a fixed 5x5 the transformer reads cell-by-cell; this collapses a
        wider window into one directional push so the agent can escape a
        fully-covered pocket toward unexplored ground. It reads only
        ``_novel_cells_this_episode`` (which resets every episode) and emits a
        relative direction, so it teaches a transferable "head toward the
        unexplored" skill rather than a fixed route. Wall cells read as
        unvisited (no walkability info here); symmetric walls cancel and the
        CNN sees the real tilemap, so this stays a soft prior.

        Read-only lookahead; all-zeros during scripted overlays (position is
        stale) or when the feature is disabled (radius 0)."""
        R = self._frontier_sense_radius
        if R <= 0 or env_vars.get("script_active", False):
            return [0.0, 0.0, 0.0]
        mb, mn = env_vars["map_bank"], env_vars["map_num"]
        x, y = env_vars["X"], env_vars["Y"]
        vx = vy = 0.0
        visited = 0
        total = 0
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                if dx == 0 and dy == 0:
                    continue
                total += 1
                cell = self.visit_archive.cell_key(
                    mb, mn, x + dx * CELL_SIZE, y + dy * CELL_SIZE
                )
                # True episode history (not the flag-re-opened paid gate), so
                # the gradient keeps pointing at genuinely-unvisited ground.
                if cell in self._ever_visited_cells_this_episode:
                    visited += 1
                else:
                    w = 1.0 / float(dx * dx + dy * dy)
                    vx += w * dx
                    vy += w * dy
        norm = math.hypot(vx, vy)
        if norm > 1e-8:
            vx /= norm
            vy /= norm
        else:
            vx = vy = 0.0
        saturation = float(visited) / float(total) if total else 0.0
        return [vx, vy, saturation]

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
