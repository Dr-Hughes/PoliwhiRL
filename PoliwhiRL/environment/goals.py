# -*- coding: utf-8 -*-
"""Typed goal parser and progress tracker.

Separates *curriculum structure* (what goals exist, which are required)
from *reward magnitudes* (how much each goal type pays).

Goal types
----------
- **location**  – tile match.  ``hard`` (default ``True``) controls whether
  the goal advances the curriculum index and counts toward the hard-goal
  termination target.  ``soft`` goals pay a smaller bonus and never block
  or advance the curriculum.
- **pokedex**   – ``{"kind": "seen"|"owned", "threshold": N}``.  Multi-fires:
  a threshold of N contributes N goal slots (one per integer increment).
- **xp**        – ``{"threshold": N, "xp_per_fire": K}``.  Fires once per
  ``xp_per_fire`` XP gained, capped at N fires total.
- **level**     – ``{"threshold": N}``.  Fires N times as the party gains N
  levels from the starting point.
"""

import copy
from collections import OrderedDict

# Track config objects already validated so we don't warn repeatedly.
_VALIDATED_CONFIGS = set()


def _normalise_positions(option):
    """Convert a single position spec (list or dict) to [x, y, bank, map]."""
    if isinstance(option, dict):
        x = option.get("x", 0)
        y = option.get("y", 0)
        bank = option.get("map_bank")
        map_num = option.get("map", 0)
        return [x, y, bank, map_num]
    if isinstance(option, list):
        if len(option) < 3:
            raise ValueError(f"Position list must have at least 3 elements (x, y, map): {option!r}")
        x, y, map_num = option[0], option[1], option[2]
        bank = option[4] if len(option) >= 5 else None
        return [x, y, bank, map_num]
    raise ValueError(f"Unknown goal position format: {option!r}")


class GoalsManager:
    """Parse typed goal configs, validate them, and track progress."""

    def __init__(self, config):
        raw_goals = config.get("goals")
        if raw_goals is None:
            raw_goals = []

        self._goals = copy.deepcopy(raw_goals)
        self._parse()

        # Progress state.
        self.current_goal_index = 0
        self.N_goals = 0
        self.pokedex_goals_completed = 0
        self.level_goals_completed = 0
        self.xp_goals_completed = 0
        self._pokedex_progress = {}

        # XP / level trackers (seeded on first call).
        self._xp_starting_total = None
        self._xp_prev_size = None
        self._level_starting_total = None
        self._level_prev_size = None

        # Curriculum target.
        self.hard_goal_count_target = config.get(
            "hard_goal_count_target", -1
        )
        self.break_on_target = config.get("break_on_goal", True)

        if self.hard_goal_count_target == -1:
            self.hard_goal_count_target = (
                len(self._hard_goals)
                + sum(g["threshold"] for g in self._pokedex_goals)
                + sum(g["threshold"] for g in self._level_goals)
                + sum(g["threshold"] for g in self._xp_goals)
            )

        self._validate(config)

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #

    def _parse(self):
        self._hard_goals = OrderedDict()
        self._soft_goals = []
        self._pokedex_goals = []
        self._level_goals = []
        self._xp_goals = []

        for goal in self._goals:
            gtype = goal.get("type")
            if gtype == "location":
                positions = []
                check_bank = False
                for opt in goal.get("positions", []):
                    pos = _normalise_positions(opt)
                    positions.append(pos)
                    check_bank = check_bank or pos[2] is not None
                parsed = {"positions": positions, "check_bank": check_bank}
                if goal.get("hard", True):
                    self._hard_goals[len(self._hard_goals)] = parsed
                else:
                    self._soft_goals.append(parsed)
            elif gtype == "pokedex":
                self._pokedex_goals.append({"kind": goal["kind"], "threshold": goal["threshold"]})
            elif gtype == "level":
                self._level_goals.append({"kind": goal.get("kind", "total_level"), "threshold": goal["threshold"]})
            elif gtype == "xp":
                self._xp_goals.append({
                    "kind": goal.get("kind", "total_xp"),
                    "threshold": goal["threshold"],
                    "xp_per_fire": goal.get("xp_per_fire", 10),
                })
            else:
                raise ValueError(f"Unknown goal type: {gtype!r}")

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate(self, config):
        config_id = id(config)
        if config_id in _VALIDATED_CONFIGS:
            return
        _VALIDATED_CONFIGS.add(config_id)
        total_possible = (
            len(self._hard_goals)
            + len(self._soft_goals)
            + sum(g["threshold"] for g in self._pokedex_goals)
            + sum(g["threshold"] for g in self._level_goals)
            + sum(g["threshold"] for g in self._xp_goals)
        )
        target = self.hard_goal_count_target

        if target > total_possible:
            print(
                f"WARNING: hard_goal_count_target ({target}) exceeds "
                f"total possible fires ({total_possible}) — "
                f"{len(self._hard_goals)} hard + "
                f"{len(self._soft_goals)} soft + "
                f"{len(self._pokedex_goals)} pokedex + "
                f"{len(self._level_goals)} level + "
                f"{len(self._xp_goals)} xp. "
                f"Episode will timeout."
            )
        if target < len(self._hard_goals):
            skipped = len(self._hard_goals) - target
            print(
                f"WARNING: hard_goal_count_target ({target}) < "
                f"hard location goals ({len(self._hard_goals)}) — "
                f"last {skipped} hard goal(s) may never train."
            )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def location_goals(self):
        return self._hard_goals

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

    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #

    def is_target_reached(self):
        return self.N_goals >= self.hard_goal_count_target

    @property
    def target_reached(self):
        return self.is_target_reached()

    def completion_fraction(self):
        if self.hard_goal_count_target <= 0:
            return 1.0
        return min(self.N_goals / self.hard_goal_count_target, 1.0)

    # ------------------------------------------------------------------ #
    # Active goal access
    # ------------------------------------------------------------------ #

    def active_hard_goal(self):
        if self.current_goal_index < len(self._hard_goals):
            return list(self._hard_goals.values())[self.current_goal_index]
        return None

    def active_hard_goal_positions(self):
        goal = self.active_hard_goal()
        return goal["positions"] if goal else None

    def has_active_hard_goal(self):
        return self.current_goal_index < len(self._hard_goals)

    # ------------------------------------------------------------------ #
    # Goal achievement checks
    # ------------------------------------------------------------------ #

    def check_hard_goal_achievement(self, cur_x, cur_y, cur_map, cur_bank):
        if self.current_goal_index >= len(self._hard_goals):
            return False, None

        goal = list(self._hard_goals.values())[self.current_goal_index]
        check_bank = goal["check_bank"]

        for opt in goal["positions"]:
            if opt[0] == cur_x and opt[1] == cur_y and opt[3] == cur_map:
                if not check_bank or opt[2] == cur_bank:
                    self.current_goal_index += 1
                    self.N_goals += 1
                    return True, goal
        return False, None

    def check_soft_goals(self, cur_x, cur_y, cur_map, cur_bank):
        hits = []
        remaining = []
        for goal in self._soft_goals:
            check_bank = goal["check_bank"]
            matched = False
            for opt in goal["positions"]:
                if opt[0] == cur_x and opt[1] == cur_y and opt[3] == cur_map:
                    if not check_bank or opt[2] == cur_bank:
                        matched = True
                        break
            if matched:
                hits.append(goal)
                self.N_goals += 1
            else:
                remaining.append(goal)
        self._soft_goals = remaining
        return hits

    def check_pokedex_goals(self, pokedex_seen, pokedex_owned):
        new_fires = 0
        seen_increased = False
        owned_increased = False

        for g in self._pokedex_goals:
            kind = g["kind"]
            threshold = g["threshold"]
            current_value = pokedex_seen if kind == "seen" else pokedex_owned
            fired = self._pokedex_progress.get(kind, 0)
            fires_now = min(int(current_value), threshold) - fired
            if fires_now > 0:
                new_fires += fires_now
                self.N_goals += fires_now
                self.pokedex_goals_completed += fires_now
                self._pokedex_progress[kind] = fired + fires_now
                if kind == "seen":
                    seen_increased = True
                else:
                    owned_increased = True

        self._pokedex_goals = [
            g for g in self._pokedex_goals
            if self._pokedex_progress.get(g["kind"], 0) < g["threshold"]
        ]
        return new_fires, seen_increased, owned_increased

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

    def reset_all(self):
        self.current_goal_index = 0
        self.N_goals = 0
        self.pokedex_goals_completed = 0
        self.level_goals_completed = 0
        self.xp_goals_completed = 0
        self._pokedex_progress = {}
        self._xp_starting_total = None
        self._xp_prev_size = None
        self._level_starting_total = None
        self._level_prev_size = None
        self._parse()

    # ------------------------------------------------------------------ #
    # Progress queries
    # ------------------------------------------------------------------ #

    def n_hard_goals_completed(self):
        return self.current_goal_index

    def n_pokedex_goals_completed(self):
        return self.pokedex_goals_completed

    def n_level_goals_completed(self):
        return self.level_goals_completed

    def n_xp_goals_completed(self):
        return self.xp_goals_completed

    def total_hard_goals(self):
        return len(self._hard_goals)
