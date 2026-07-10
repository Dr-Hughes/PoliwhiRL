# -*- coding: utf-8 -*-
"""Progress-signal accumulator (Phase 4: no termination predicate).

The episode terminator is *always* ``episode_length`` — no
``terminate_on``, no goal checklist, no targeted milestone. Reward fuel
comes from continuous game-state progress signals plus the novelty
landscape exposed via the visit archive in ``environment/visit_archive.py``.

Configurable progress signals
-----------------------------
Each enabled signal contributes to per-step reward and bumps the
``N_goals`` training metric (which is now a generic "progress fires"
count, not a tile-touch count). Coordinate-based goals (location /
soft-waypoint) are intentionally NOT supported — the design goal is to
stop the reward signal from encoding the curriculum's hand-authored path.
Map / tile novelty live on the ``rewards`` side and are not part of the
goal definition.

- **pokedex**  – ``{"type": "pokedex", "kind": "seen"|"owned", "threshold": N}``.
  Multi-fires; pays per integer increment up to ``threshold``.
- **flag**     – ``{"type": "flag", "flag_num": N}``.  Pays
  ``flag_progress_reward`` when the named event-flag bit (offset into
  0xDA72–0xDB71) transitions ``0 → 1`` this episode. Flags already set at
  episode start (e.g. via replay) do NOT fire — only fresh transitions
  count.
- **level**    – ``{"type": "level", "threshold": N}``.  Fires N times as
  the party gains N total levels.
- **xp**       – ``{"type": "xp", "threshold": N, "xp_per_fire": K}``.
  Fires once per K XP gained, capped at N fires total.
- **map**      – ``{"type": "map", "map_bank": B, "map_num": N}``.  Fires
  once when the player enters ``(B, N)`` *during* the training portion.
  The episode's starting map (post-replay) does NOT fire — only a fresh
  entry counts, mirroring the flag semantics. This is a coarse terminal
  *milestone* ("reached town X"), not a per-tile path-shaping signal, so
  it does not reintroduce the oracle-teacher problem that banned the old
  ``location`` goals.
- **maps_visited** – ``{"type": "maps_visited", "threshold": N}``.  Fires
  once per unique ``(map_bank, map_num)`` the player has been on this
  episode, up to ``threshold`` fires. Unlike ``map`` it doesn't name a
  specific destination — it just counts breadth of exploration, so it's
  the natural goal for "leave the house" style early stages (e.g. visit
  bedroom → downstairs → outside = 3 maps). The starting map counts as
  the first visit. Completion (for ``terminate_on_goal_complete``) is
  ``unique maps this episode >= threshold``.
"""

import copy


class GoalsManager:
    """Track game-progress signals and evaluate the episode-termination
    predicate. Has no coordinate-tile checklist and no curriculum index."""

    def __init__(self, config):
        raw_goals = config.get("goals") or []
        self._goals_raw = copy.deepcopy(raw_goals)
        self._parse_goals()
        if config.get("terminate_on") is not None:
            raise ValueError(
                "`terminate_on` is no longer supported — episodes run to "
                "`episode_length` always (Phase 4 free-play). Remove the "
                "key from your stage config."
            )

        # Per-episode progress counters. Aggregated into N_goals for
        # downstream metric plotting.
        self.N_goals = 0
        self.pokedex_goals_completed = 0
        self.level_goals_completed = 0
        self.xp_goals_completed = 0
        self.flag_goals_completed = 0
        self.map_goals_completed = 0
        self.maps_visited_goals_completed = 0
        self._pokedex_progress = {}
        self._flag_progress = {}  # flag_num -> fired (bool)
        self._map_fired = set()   # indices into self._map_goals that have fired
        self._map_initial = None  # (bank, num) at episode start (post-replay)
        self._maps_seen_this_episode = set()

        # XP / level trackers (seeded on first call).
        self._xp_starting_total = None
        self._xp_prev_size = None
        self._level_starting_total = None
        self._level_prev_size = None

        # Snapshot of the story-flag bytes at episode start. Seeded on the
        # first ``check_flag_goals`` call so we only count fresh 0→1
        # transitions, not flags already true via replay.
        self._flag_initial_state = None

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #

    def _parse_goals(self):
        self._pokedex_goals = []
        self._level_goals = []
        self._xp_goals = []
        self._flag_goals = []
        self._map_goals = []
        self._maps_visited_goals = []

        for goal in self._goals_raw:
            gtype = goal.get("type")
            if gtype == "pokedex":
                self._pokedex_goals.append(
                    {"kind": goal["kind"], "threshold": goal["threshold"]}
                )
            elif gtype == "level":
                self._level_goals.append(
                    {"kind": goal.get("kind", "total_level"), "threshold": goal["threshold"]}
                )
            elif gtype == "xp":
                self._xp_goals.append({
                    "kind": goal.get("kind", "total_xp"),
                    "threshold": goal["threshold"],
                    "xp_per_fire": goal.get("xp_per_fire", 10),
                })
            elif gtype == "flag":
                if "flag_num" not in goal:
                    raise ValueError(f"flag goal needs 'flag_num': {goal!r}")
                self._flag_goals.append({"flag_num": int(goal["flag_num"])})
            elif gtype == "map":
                if "map_num" not in goal:
                    raise ValueError(f"map goal needs 'map_num': {goal!r}")
                self._map_goals.append({
                    "map_bank": int(goal["map_bank"]) if "map_bank" in goal else None,
                    "map_num": int(goal["map_num"]),
                })
            elif gtype == "maps_visited":
                if "threshold" not in goal:
                    raise ValueError(f"maps_visited goal needs 'threshold': {goal!r}")
                self._maps_visited_goals.append({"threshold": int(goal["threshold"])})
            else:
                raise ValueError(
                    "Unknown goal type: "
                    f"{gtype!r} (supported: pokedex, level, xp, flag, map, maps_visited)"
                )

    # ------------------------------------------------------------------ #
    # Public properties (kept for back-compat with reward / plotting code)
    # ------------------------------------------------------------------ #

    @property
    def location_goals(self):
        # Empty — no coordinate goals supported. Reward & plotting layers
        # still query this to populate the legacy ``target_*`` slots; with
        # no goals here they get zeroed, which is exactly what we want.
        return {}

    @property
    def pokedex_goals(self):
        result = {}
        for g in self._pokedex_goals:
            fired = self._pokedex_progress.get(g["kind"], 0)
            if fired < g["threshold"]:
                result[g["kind"]] = g["threshold"]
        return result

    @property
    def level_goals(self):
        return {g["kind"]: g["threshold"] for g in self._level_goals}

    @property
    def xp_goals(self):
        return {g["kind"]: g["threshold"] for g in self._xp_goals}

    @property
    def flag_goals(self):
        return [g["flag_num"] for g in self._flag_goals]

    @property
    def map_goals(self):
        return [(g["map_bank"], g["map_num"]) for g in self._map_goals]

    # ------------------------------------------------------------------ #
    # Per-step progress checks
    # ------------------------------------------------------------------ #

    def check_pokedex_goals(self, pokedex_seen, pokedex_owned):
        """Returns (seen_fires, owned_fires) — new increments this step for
        each kind, separated so callers can reward them at different rates
        (owning a species matters far more than merely sighting one)."""
        seen_fires = 0
        owned_fires = 0

        for g in self._pokedex_goals:
            kind = g["kind"]
            threshold = g["threshold"]
            current_value = pokedex_seen if kind == "seen" else pokedex_owned
            fired = self._pokedex_progress.get(kind, 0)
            fires_now = min(int(current_value), threshold) - fired
            if fires_now > 0:
                self.N_goals += fires_now
                self.pokedex_goals_completed += fires_now
                self._pokedex_progress[kind] = fired + fires_now
                if kind == "seen":
                    seen_fires += fires_now
                else:
                    owned_fires += fires_now

        self._pokedex_goals = [
            g for g in self._pokedex_goals
            if self._pokedex_progress.get(g["kind"], 0) < g["threshold"]
        ]
        return seen_fires, owned_fires

    def check_flag_goals(self, story_flags):
        """Check whether any configured flag bit transitioned 0→1 this step.

        Snapshots the initial flag-bytes on first call so that flags
        already set at episode start (e.g. via action_replay walking the
        env into a state where ``has_starter`` is already true) do NOT
        count — only fresh transitions are rewarded.
        """
        if not self._flag_goals:
            return 0
        if self._flag_initial_state is None:
            self._flag_initial_state = bytes(story_flags)

        new_fires = 0
        for g in self._flag_goals:
            fnum = g["flag_num"]
            if self._flag_progress.get(fnum, False):
                continue  # already fired this episode
            byte_idx, bit_idx = fnum // 8, fnum % 8
            now_bit = (story_flags[byte_idx] >> bit_idx) & 1
            start_bit = (self._flag_initial_state[byte_idx] >> bit_idx) & 1
            if now_bit == 1 and start_bit == 0:
                self._flag_progress[fnum] = True
                self.flag_goals_completed += 1
                self.N_goals += 1
                new_fires += 1
        return new_fires

    def check_map_goals(self, map_bank, map_num):
        """Fire any configured map-reach goal whose target the player has
        just entered. Snapshots the episode's starting map on the first
        call so a goal whose target equals the start position (e.g. a
        replay that ends on the target map) does not fire spuriously —
        only a genuine entry during the training portion counts. Returns
        the number of map goals that fired this step.
        """
        if not self._map_goals:
            return 0
        cur = (int(map_bank), int(map_num))
        if self._map_initial is None:
            self._map_initial = cur
        if cur == self._map_initial:
            return 0
        new_fires = 0
        for idx, g in enumerate(self._map_goals):
            if idx in self._map_fired:
                continue
            if g["map_num"] != cur[1]:
                continue
            if g["map_bank"] is not None and g["map_bank"] != cur[0]:
                continue
            self._map_fired.add(idx)
            self.map_goals_completed += 1
            self.N_goals += 1
            new_fires += 1
        return new_fires

    def note_map_visit(self, map_bank, map_num):
        """Record that the player is currently on (map_bank, map_num). Idempotent.
        Used by the maps_visited goal and map-novelty metrics. Doesn't pay
        reward — rewards.py owns the new_map_reward path."""
        self._maps_seen_this_episode.add((int(map_bank), int(map_num)))

    def seed_seen_maps(self, map_keys):
        """Pre-fill ``_maps_seen_this_episode`` with maps the replay walked
        through. Called after ``reset_episode_trackers`` so the maps_visited
        goal credits replay progress and is consistent with new_map_reward.

        Does NOT advance ``maps_visited_goals_completed`` — those counters
        track training-episode progress only. The seeded maps simply prevent
        the replay's maps from counting as "new" during training.
        """
        for key in map_keys:
            self._maps_seen_this_episode.add((int(key[0]), int(key[1])))

    # ------------------------------------------------------------------ #
    # Snapshot seed facts (save-state restarts)                           #
    # ------------------------------------------------------------------ #

    def fired_map_goal_keys(self):
        """(map_bank, map_num) keys of map goals that have fired this
        episode. Exported into snapshot seed facts — keys, not indices, so
        the facts survive a change of goal list between stages."""
        return [
            (self._map_goals[i]["map_bank"], self._map_goals[i]["map_num"])
            for i in sorted(self._map_fired)
        ]

    def fired_flag_nums(self):
        """Flag numbers of flag goals that have fired this episode."""
        return [f for f, fired in self._flag_progress.items() if fired]

    def apply_seed_facts(self, map_goal_keys, flag_nums, pokedex_seen, pokedex_owned):
        """Mark goals completed by a snapshot's source path as fired.

        Matches the exported facts against the CURRENT goal config: a map
        goal is marked fired if its key appears in ``map_goal_keys``; flag
        goals likewise; pokedex goals are advanced from the restored
        seen/owned counts. Counters (``N_goals``, per-type completed) are
        advanced so the RAM progress features and the termination predicate
        see the seeded progress — the agent's goals-at-start snapshot then
        excludes it from ``goals_made``.

        Must be called on a fresh episode (after ``reset_episode_trackers``)
        and before the first per-step check.
        """
        keys = {
            (None if b is None else int(b), int(n))
            for b, n in (tuple(k) for k in map_goal_keys)
        }
        for idx, g in enumerate(self._map_goals):
            if idx in self._map_fired:
                continue
            for b, n in keys:
                if g["map_num"] != n:
                    continue
                if (
                    g["map_bank"] is not None
                    and b is not None
                    and g["map_bank"] != b
                ):
                    continue
                self._map_fired.add(idx)
                self.map_goals_completed += 1
                self.N_goals += 1
                break
        configured_flags = {g["flag_num"] for g in self._flag_goals}
        for fnum in flag_nums:
            fnum = int(fnum)
            if fnum in configured_flags and not self._flag_progress.get(fnum, False):
                self._flag_progress[fnum] = True
                self.flag_goals_completed += 1
                self.N_goals += 1
        # Pokedex goals complete from the restored counts (pays no reward —
        # the Rewards baselines are set from the same facts).
        self.check_pokedex_goals(int(pokedex_seen), int(pokedex_owned))

    def check_maps_visited_goals(self):
        """Advance the maps_visited counter toward its threshold.

        Fires once per unique (map_bank, map_num) seen this episode, capped
        at the configured threshold. Call after ``note_map_visit`` so the
        count reflects the current step. Pays no reward directly — the
        ``new_map_reward`` path in rewards.py is the learning signal; this
        only drives N_goals (for metrics) and the termination predicate.
        Folded into ``n_map_goals_completed`` so existing plotting /
        terminal-info plumbing surfaces it without changes elsewhere.
        """
        if not self._maps_visited_goals:
            return 0
        threshold = max(g["threshold"] for g in self._maps_visited_goals)
        seen = len(self._maps_seen_this_episode)
        target = min(seen, threshold)
        new_fires = target - self.maps_visited_goals_completed
        if new_fires <= 0:
            return 0
        self.maps_visited_goals_completed += new_fires
        self.N_goals += new_fires
        return new_fires

    def check_xp_goals(self, party_size, party_exp, xp_per_fire):
        if not self._xp_goals:
            return 0
        if self._xp_prev_size is None:
            self._xp_prev_size = party_size
            self._xp_starting_total = party_exp
            return 0
        if party_size != self._xp_prev_size:
            self._xp_prev_size = party_size
            self._xp_starting_total = party_exp
            return 0
        if xp_per_fire <= 0:
            return 0
        xp_gained = party_exp - self._xp_starting_total
        if xp_gained <= 0:
            return 0
        total_threshold = sum(g["threshold"] for g in self._xp_goals)
        chunks_crossed = min(xp_gained // xp_per_fire, total_threshold)
        new_fires = chunks_crossed - self.xp_goals_completed
        if new_fires <= 0:
            return 0
        self.xp_goals_completed += new_fires
        self.N_goals += new_fires
        return new_fires

    def check_level_goals(self, party_size, party_level):
        if not self._level_goals:
            return 0
        if self._level_prev_size is None:
            self._level_prev_size = party_size
            self._level_starting_total = party_level
            return 0
        if party_size != self._level_prev_size:
            self._level_prev_size = party_size
            self._level_starting_total = party_level
            return 0
        levels_gained = party_level - self._level_starting_total
        if levels_gained <= 0:
            return 0
        total_threshold = sum(g["threshold"] for g in self._level_goals)
        new_fires = min(levels_gained, total_threshold) - self.level_goals_completed
        if new_fires <= 0:
            return 0
        self.level_goals_completed += new_fires
        self.N_goals += new_fires
        return new_fires

    # ------------------------------------------------------------------ #
    # Episode reset
    # ------------------------------------------------------------------ #

    def reset_episode_trackers(self):
        self._xp_starting_total = None
        self._xp_prev_size = None
        self._level_starting_total = None
        self._level_prev_size = None
        self._flag_initial_state = None
        self._flag_progress = {}
        self._map_fired = set()
        self._map_initial = None
        self._maps_seen_this_episode = set()
        self.flag_goals_completed = 0
        self.map_goals_completed = 0
        self.maps_visited_goals_completed = 0

    def reset_all(self):
        self.N_goals = 0
        self.pokedex_goals_completed = 0
        self.level_goals_completed = 0
        self.xp_goals_completed = 0
        self.map_goals_completed = 0
        self.maps_visited_goals_completed = 0
        self._pokedex_progress = {}
        self.reset_episode_trackers()
        self._parse_goals()

    # ------------------------------------------------------------------ #
    # Progress queries (used by RAM observation builder & plotting)
    # ------------------------------------------------------------------ #

    def n_pokedex_goals_completed(self):
        return self.pokedex_goals_completed

    def n_level_goals_completed(self):
        return self.level_goals_completed

    def n_xp_goals_completed(self):
        return self.xp_goals_completed

    def n_flag_goals_completed(self):
        return self.flag_goals_completed

    def n_map_goals_completed(self):
        # maps_visited fires are folded into the map-goal count so existing
        # metric / terminal-info plumbing (which only knows about n_map)
        # surfaces breadth-of-exploration progress without extra wiring.
        return self.map_goals_completed + self.maps_visited_goals_completed

    def all_goal_thresholds_met(self):
        """True when every configured goal has fully fired this episode.

        Used by the opt-in early-termination path (config flag
        ``terminate_on_goal_complete``) so navigation stages can end the
        episode the moment the target milestone is reached, while keeping
        the Phase-4 default "run to ``episode_length``" semantics for
        everything else. Returns False if no goals are configured — an
        empty goal list should not be considered "solved".
        """
        # Use the immutable original spec: check_pokedex_goals prunes
        # completed entries from _pokedex_goals, so completing a
        # pokedex-only stage would otherwise empty every mutable list and
        # make this wrongly report "no goals configured" (never terminate).
        if not self._goals_raw:
            return False
        # Flag goals: each must have fired this episode.
        for g in self._flag_goals:
            if not self._flag_progress.get(g["flag_num"], False):
                return False
        # Map-reach goals: every configured target must have been entered
        # this episode (a genuine entry during the training portion). A
        # save-state that *starts* on the target does not count — it achieved
        # nothing — so such an episode runs to episode_length rather than
        # terminating instantly with a spurious success. Per-stage state pools
        # must therefore not start the agent on that stage's target map.
        if len(self._map_fired) < len(self._map_goals):
            return False
        # Breadth goals: enough unique maps visited this episode.
        for g in self._maps_visited_goals:
            if len(self._maps_seen_this_episode) < g["threshold"]:
                return False
        # Pokedex: check_pokedex_goals prunes completed entries from
        # _pokedex_goals as they hit threshold. Non-empty list => incomplete.
        if self._pokedex_goals:
            return False
        # Level / xp goals: tally vs total threshold sum.
        total_level = sum(g["threshold"] for g in self._level_goals)
        if self.level_goals_completed < total_level:
            return False
        total_xp = sum(g["threshold"] for g in self._xp_goals)
        if self.xp_goals_completed < total_xp:
            return False
        return True

    def per_goal_status(self, pokedex_seen=0, pokedex_owned=0):
        """Per-goal met/not-met against the ORIGINAL spec, for eval reporting.

        Returns a list of ``(label, met)`` in the order goals were configured.
        Re-derives met-ness from STABLE signals (``_map_fired``,
        ``_flag_progress``, the live pokedex counts) rather than the mutable
        ``_pokedex_goals`` list, which is pruned as goals complete. Used by the
        from-scratch end-to-end eval to show WHICH milestone was forgotten,
        not just an all-or-nothing boolean.
        """
        statuses = []
        map_idx = 0
        for goal in self._goals_raw:
            gtype = goal.get("type")
            if gtype == "map":
                met = map_idx in self._map_fired
                label = f"map {goal.get('map_bank')}/{goal['map_num']}"
                map_idx += 1
            elif gtype == "pokedex":
                kind = goal["kind"]
                thr = goal["threshold"]
                have = pokedex_owned if kind == "owned" else pokedex_seen
                met = have >= thr
                label = f"pokedex {kind}>={thr}"
            elif gtype == "flag":
                fnum = int(goal["flag_num"])
                met = bool(self._flag_progress.get(fnum, False))
                label = f"flag {fnum}"
            elif gtype == "maps_visited":
                thr = int(goal["threshold"])
                met = len(self._maps_seen_this_episode) >= thr
                label = f"maps_visited>={thr}"
            elif gtype == "level":
                thr = int(goal["threshold"])
                met = self.level_goals_completed >= thr
                label = f"level>={thr}"
            elif gtype == "xp":
                thr = int(goal["threshold"])
                met = self.xp_goals_completed >= thr
                label = f"xp>={thr}"
            else:
                met = False
                label = str(gtype)
            statuses.append((label, bool(met)))
        return statuses
