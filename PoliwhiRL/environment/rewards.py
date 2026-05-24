# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict

# Module-level set of config object ids that have been validated.
# Prevents the warning from firing on every env reset.
_VALIDATED_CONFIGS = set()

# Valid raw byte values for the RAM fields we use as the canary for a
# clean RAM snapshot. Anything else means the engine is mid-state-change
# (e.g. the wild-battle init window) and the snapshot must not be trusted.
_VALID_BATTLE_TYPES = (0, 1, 2)            # overworld, wild, trainer
_VALID_PLAYER_STATES = (0, 1, 2, 4)        # walk, bike, skate, surf


def is_ram_state_valid(env_vars):
    """Heuristic: does this RAM snapshot look like a clean post-tick read?

    Returns False on the two junk signatures observed in stage 5 traces:

    1. ``battle_type`` outside ``{0, 1, 2}`` or ``player_state`` outside
       ``{0, 1, 2, 4}`` — these are individual bytes that only ever take
       those values during normal play; anything else is a mid-write read.
    2. ``X``, ``Y``, ``map_num`` and ``map_bank`` all simultaneously zero —
       the screen-scroll-out for battle init briefly zeroes the overworld
       coords; the player never legitimately stands at (0, 0) of map 0
       in bank 0 in this curriculum.

    Used to gate the reward calculation and the PNG recorder so junk
    frames cannot pollute ``explored_tiles`` / ``explored_maps``, advance
    ``pokedex_seen`` past its real value, or write garbage filenames.
    """
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
    def __init__(self, config):
        # Configuration parameters
        self.max_steps = config["episode_length"]
        self.N_goals_target = config["N_goals_target"]
        self.break_on_goal = config["break_on_goal"]
        self.punish_steps = config["punish_steps"]

        # Configurable rewards
        self.goal_reward = config.get("goal_reward", 100)
        self.sequence_bonus = config.get("sequence_bonus", 50)
        self.checkpoint_bonus = config.get("checkpoint_bonus", 200)
        self.all_goals_bonus = config.get("all_goals_bonus", 500)
        self.early_completion_bonus = config.get("early_completion_bonus", 0)

        # Fixed penalties
        self.step_penalty = config.get("step_penalty", -1) if self.punish_steps else 0
        self.button_penalty = -5  # Fixed -5 for start/select

        # Pokedex rewards. Configurable so curriculum stages can damp the
        # "see one new species" path without removing the underlying tracking.
        # Stages that need to push the policy toward fighting (rather than
        # fleeing-for-seen-credit) can set pokedex_seen_reward to 0.
        self.pokedex_seen_reward = config.get("pokedex_seen_reward", 50)
        self.pokedex_owned_reward = config.get("pokedex_owned_reward", 150)

        # Clipping
        self.clip = 1000  # Higher to accommodate integer rewards

        # Goal sequencing
        self.require_sequential = config.get("require_sequential", True)
        self.checkpoint_goals = config.get(
            "checkpoint_goals", [2, 4, 6]
        )  # Major milestones

        # Exploration parameters
        self.exploration_reward = config.get("exploration_reward", 0.0)
        # First-visit-per-map bonus, separate from the per-tile exploration
        # reward. Sized so that entering a new town/route is a meaningful
        # signal without competing with the curriculum location goals
        # (goal_reward + sequence_bonus = 150 by default).
        self.new_map_reward = config.get("new_map_reward", 0.0)
        # Reward per step spent inside a battle (battle_type != 0). Designed
        # to offset the step_penalty so the policy isn't actively punished
        # for engaging combat. Independent of damage dealt.
        self.battle_engagement_reward = config.get("battle_engagement_reward", 0.0)
        # Reward per HP point of damage dealt to the enemy Pokemon during a
        # battle. Provides a dense within-battle signal so the policy can
        # learn the attack mechanic before any level-up fires.
        self.damage_dealt_reward = config.get("damage_dealt_reward", 0.0)

        # Party progress rewards: small dense signals for XP gains and larger
        # bonuses when the party's total level increases. Helps prevent the
        # policy from gaming the system by swapping low-level Pokemon in/out.
        self.party_level_reward = config.get("party_level_reward", 0)
        self.party_exp_reward = config.get("party_exp_reward", 0)

        # XP milestone reward: fires once per threshold of cumulative XP
        # gained from battles. Suppressed when party size changes (capture /
        # swap) to avoid counting the new Pokemon's existing EXP. The
        # accumulator is NOT reset on size change — it pauses for the step
        # and resumes next step.
        self.xp_milestone_threshold = config.get("xp_milestone_threshold", 0)
        self.xp_milestone_reward = config.get("xp_milestone_reward", 0)

        # When True, allows party progress rewards even when party size
        # changes, provided the agent is in a battle (battle_type != 0).
        # This prevents false suppression during mid-battle captures or
        # party swaps. Disabled by default until we verify that battle_type
        # and XP gain timing align correctly with frame stepping.
        self.party_reward_check_battle = config.get("party_reward_check_battle", False)

        # XP-goal mechanism: pays a goal-tier reward per chunk of cumulative
        # XP gained, capped at the configured max fires. Designed so winning
        # a wild battle (≈ 10–25 XP at low level) fires once and pays a
        # location-goal-sized reward. xp_goal_threshold = chunk size in XP;
        # xp_goal_reward = reward per fire; xp_goals = {"total_xp": N} caps
        # the number of fires (matches level_goals shape).
        self.xp_goal_threshold = config.get("xp_goal_threshold", 10)
        self.xp_goal_reward = config.get("xp_goal_reward", self.goal_reward)

        # Distance shaping: potential-based reward for getting closer to the
        # active location goal. Reward-neutral in expectation (optimal policy
        # unchanged). Reset on goal hit or map change.
        self.distance_shaping_coef = config.get("distance_shaping_coef", 0.0)
        self._d_prev = None

        # State variables
        self.pokedex_seen = 0
        self.pokedex_owned = 0
        self.done = False
        self.last_action = None
        self.steps = 0
        self.N_goals = 0
        self.explored_tiles = set()
        # First-visit-per-map tracking, keyed by (map_bank, map_num). Lives
        # alongside explored_tiles and is preserved across the replay → training
        # boundary so re-entering a map the replay already discovered does not
        # pay a fresh bonus.
        self.explored_maps = set()
        # Tracks the enemy HP across consecutive in-battle steps so we can
        # credit the policy for damage dealt. Cleared whenever battle_type
        # returns to 0 (overworld) so HP from a prior battle never carries
        # into the next. Also reset by start_new_episode() so replay-time
        # damage cannot leak into the training episode.
        self._prev_enemy_hp = None
        self.cumulative_reward = 0
        self.allowed_pokedex_goals = ["seen", "owned"]
        # Independent counter for pokedex-goal hits. self.pokedex_goals
        # mutates as goals get consumed, so we need a separate count for
        # the policy's RAM-vector progress feature.
        self.pokedex_goals_completed = 0
        # Per-type count of pokedex goal fires. A threshold of N contributes
        # N goal slots (one per integer increment) rather than a single fire
        # at the threshold. Tracked separately from pokedex_goals so we can
        # tell when a type is fully consumed.
        self._pokedex_goal_progress = {}

        # Party progress tracking (previous party size, total level and EXP).
        # None means "not yet seeded" so we don't fire a phantom reward on step 0.
        # Party size is tracked to suppress rewards when a new Pokemon joins,
        # preventing compound rewards with pokedex-goal / level-up bonuses.
        self._prev_party_size = None
        self._prev_party_level = None
        self._prev_party_exp = None

        # XP milestone accumulator: tracks cumulative XP gained since the last
        # milestone fired. Reset on start_new_episode() and when a milestone
        # fires. NOT reset on party size change (we pause accumulation for
        # that step but keep the running total).
        self._xp_since_milestone = 0

        # Level goals: multi-fire based on total party level increases from
        # the starting point. {"total_level": 3} fires 3 times as the party
        # gains 3 levels. Suppressed on party size change.
        self.allowed_level_goals = ["total_level"]
        self.level_goals_completed = 0
        self._level_starting_total = None  # Seeded from first step
        self._level_goal_prev_size = None  # Track size change independently

        # XP goals: multi-fire based on chunks of cumulative XP gained from
        # the starting baseline. {"total_xp": 1} caps fires at 1 (≈ 1 win).
        # Independent from level_goals so a stage can target either signal.
        # Suppressed on party size change to avoid crediting a captured
        # mon's existing EXP.
        self.allowed_xp_goals = ["total_xp"]
        self.xp_goals_completed = 0
        self._xp_goal_prev_size = None
        self._xp_goal_starting_total = None

        # Variables for ordered goals
        self.location_goals = OrderedDict()
        self.current_goal_index = 0

        self.set_goals(
            config["location_goals"],
            config.get("pokedex_goals", {}),
            config.get("level_goals", {}),
            config.get("xp_goals", {}),
        )

        if self.N_goals_target == -1:
            total_level = sum(self.level_goals.values())
            total_xp_fires = sum(self.xp_goals.values())
            self.N_goals_target = (
                len(self.location_goals)
                + len(self.pokedex_goals)
                + total_level
                + total_xp_fires
            )

        # Validate goal configuration once per config object (not per reset).
        if id(config) not in _VALIDATED_CONFIGS:
            _VALIDATED_CONFIGS.add(id(config))
            self._validate_config()

    def update_targets(self, n_goals_target, max_steps):
        self.N_goals_target = n_goals_target
        self.max_steps = max_steps
        self.done = False

    def _validate_config(self):
        """One-time validation of goal configuration. Fires once per config
        object (not per reset) to catch unreachable targets or wasted goals."""
        total_possible = len(self.location_goals)
        for threshold in self.pokedex_goals.values():
            total_possible += int(threshold)  # multi-fire: threshold N = N slots
        for threshold in self.level_goals.values():
            total_possible += int(threshold)  # multi-fire: threshold N = N slots
        for threshold in self.xp_goals.values():
            total_possible += int(threshold)  # multi-fire: threshold N = N slots

        if self.N_goals_target > total_possible:
            print(
                f"WARNING: N_goals_target ({self.N_goals_target}) exceeds "
                f"total possible fires ({total_possible}) — "
                f"{len(self.location_goals)} location + "
                f"{dict(self.pokedex_goals)} pokedex + "
                f"{dict(self.level_goals)} level. "
                f"Episode will timeout before reaching target."
            )
        if self.N_goals_target < len(self.location_goals):
            skipped = len(self.location_goals) - self.N_goals_target
            print(
                f"WARNING: N_goals_target ({self.N_goals_target}) < "
                f"location goals ({len(self.location_goals)}) — "
                f"last {skipped} location goal(s) may never train "
                f"(episode breaks before reaching them)."
            )

    def _distance_shaping(self, cur_x, cur_y, cur_map, cur_bank):
        """Potential-based shaping: reward when the player gets closer to the
        active location goal on the same map. Reset when the map changes (the
        target might be on a different map now). Also reset in _check_goal_achievement
        when a goal fires."""
        if self.distance_shaping_coef <= 0:
            return 0
        if self.current_goal_index >= len(self.location_goals):
            self._d_prev = None
            return 0

        goal = list(self.location_goals.values())[self.current_goal_index]
        # positions is [[x, y, bank, map], ...]. Use the first option as the
        # canonical target for distance shaping.
        target_x, target_y, target_bank, target_map = goal["positions"][0]

        # Only shape when on the same map (and bank if specified).
        if cur_map != target_map:
            self._d_prev = None
            return 0
        if goal["check_bank"] and cur_bank != target_bank:
            self._d_prev = None
            return 0

        d_curr = abs(cur_x - target_x) + abs(cur_y - target_y)

        if self._d_prev is not None and self._d_prev > d_curr:
            shaping = self.distance_shaping_coef * (self._d_prev - d_curr)
            self._d_prev = d_curr
            return shaping

        self._d_prev = d_curr
        return 0

    def start_new_episode(self):
        """Reset per-episode bookkeeping while preserving everything that
        encodes curriculum / exploration progress.

        Preserved across replay → training:
          - current_goal_index, N_goals, pokedex_goals_completed,
            level_goals_completed, pokedex_seen, pokedex_owned
            (curriculum progress).
          - explored_tiles, explored_maps (so re-walking replay-visited
            tiles/maps does not pay a fresh exploration / new-map bonus).
          - _d_prev (distance-shaping potential, so shaping fires from
            step 0).

        Cleared:
          - done (a fresh episode starts not-done).
          - last_action (no carry-over of the replay's last button press).
          - steps (clean budget against max_steps).
          - cumulative_reward (we only want to track training-episode
            reward; replay rewards don't count).
          - _prev_party_level, _prev_party_exp (re-seed from step 0 of
            the training episode so we don't fire a phantom reward).
          - _level_starting_total, _level_goal_prev_size (re-seed from
            step 0 so level goals track from the training episode start).
          - _prev_enemy_hp (re-seed in battle on the first training step
            so replay-time damage cannot be re-credited).
          - _xp_goal_starting_total, _xp_goal_prev_size (re-seed from step 0
            of the training episode so xp_goals fire on training-episode
            XP gain only, not on the replay's XP accumulation).
        """
        self.done = False
        self.last_action = None
        self.steps = 0
        self.cumulative_reward = 0
        self._prev_party_size = None
        self._prev_party_level = None
        self._prev_party_exp = None
        self._xp_since_milestone = 0
        self._level_starting_total = None
        self._level_goal_prev_size = None
        self._prev_enemy_hp = None
        self._xp_goal_prev_size = None
        self._xp_goal_starting_total = None

    def set_goals(self, location_goals, pokedex_goals, level_goals=None, xp_goals=None):
        self.location_goals = OrderedDict()
        self.pokedex_goals = {}
        self.level_goals = {}
        self.xp_goals = {}
        if location_goals:
            for idx, goal in enumerate(location_goals):
                # Parse each goal entry. Supports two formats:
                #   List: [x, y, map_num, room?, map_bank?]
                #   Dict: {"x": ..., "y": ..., "map": ..., "map_bank"?: ...}
                # map_bank (5th list element or "map_bank" key) is optional.
                # When absent, matching ignores map_bank (backwards compat).
                positions = []
                check_bank = False
                for option in goal:
                    if isinstance(option, dict):
                        x = option.get("x", 0)
                        y = option.get("y", 0)
                        bank = option.get("map_bank")
                        map_num = option.get("map", 0)
                        positions.append([x, y, bank, map_num])
                        check_bank = check_bank or bank is not None
                    elif isinstance(option, list):
                        x, y, map_num = option[0], option[1], option[2]
                        bank = option[4] if len(option) >= 5 else None
                        positions.append([x, y, bank, map_num])
                        check_bank = check_bank or bank is not None
                    else:
                        raise ValueError(f"Unknown goal format: {option}")
                self.location_goals[idx] = {
                    "positions": positions,
                    "check_bank": check_bank,
                }
        if pokedex_goals:
            if isinstance(pokedex_goals, dict):
                for k, v in pokedex_goals.items():
                    if k in self.allowed_pokedex_goals:
                        self.pokedex_goals[k] = v
            else:
                raise ValueError("Pokedex goals must be a dictionary")
        if level_goals:
            if isinstance(level_goals, dict):
                for k, v in level_goals.items():
                    if k in self.allowed_level_goals:
                        self.level_goals[k] = v
            else:
                raise ValueError("Level goals must be a dictionary")
        if xp_goals:
            if isinstance(xp_goals, dict):
                for k, v in xp_goals.items():
                    if k in self.allowed_xp_goals:
                        self.xp_goals[k] = v
            else:
                raise ValueError("XP goals must be a dictionary")

    def calculate_reward(self, env_vars, button_press):
        self.steps += 1

        if is_ram_state_valid(env_vars):
            cur_x = env_vars["X"]
            cur_y = env_vars["Y"]
            cur_map = env_vars["map_num"]
            cur_bank = env_vars["map_bank"]

            macro = self._macro_reward(env_vars)
            micro = self._micro_reward(env_vars, button_press, cur_x, cur_y, cur_map, cur_bank)
            total_reward = macro + micro
        else:
            # Transitional / junk RAM snapshot. Skip everything except the
            # signals that depend only on the step happening and the button
            # pressed (not on RAM contents), so we don't pollute
            # explored_tiles, explored_maps, pokedex_seen / owned, or the
            # prev trackers with values that may be garbage. The step still
            # counts toward max_steps.
            total_reward = self._step_penalty()
            if button_press in ["start", "select"]:
                total_reward += self.button_penalty

        self.last_action = button_press

        if self.done or self.steps > self.max_steps:
            self.done = True

        self.cumulative_reward += total_reward
        clipped_reward = np.clip(total_reward, -self.clip, self.clip).astype(np.float32)

        return clipped_reward, self.done

    # ------------------------------------------------------------------ #
    # Macro rewards: sparse, large-magnitude, curriculum-driven signals   #
    # (goal hits, pokedex milestones, checkpoint / completion bonuses)    #
    # ------------------------------------------------------------------ #

    def _macro_reward(self, env_vars):
        reward = 0
        reward += self._check_goal_achievement(env_vars)
        reward += self._check_pokedex_rewards(env_vars)
        reward += self._check_level_goals(env_vars)
        reward += self._check_xp_goals(env_vars)
        return reward

    def _check_goal_achievement(self, env_vars):
        """Match current position against the active location goal.

        Includes map_bank in the match key to prevent collisions between
        maps with the same number in different groups.
        """
        if self.current_goal_index >= len(self.location_goals):
            return 0

        cur_x = env_vars["X"]
        cur_y = env_vars["Y"]
        cur_bank = env_vars["map_bank"]
        cur_map = env_vars["map_num"]

        goal = list(self.location_goals.values())[self.current_goal_index]
        goal_check_bank = goal["check_bank"]

        matched = False
        for opt in goal["positions"]:
            if opt[0] == cur_x and opt[1] == cur_y and opt[3] == cur_map:
                if not goal_check_bank or opt[2] == cur_bank:
                    matched = True
                    break

        if not matched:
            return 0

        self.current_goal_index += 1
        self.N_goals += 1
        # Reset distance shaping on goal hit.
        self._d_prev = None

        reward = self.goal_reward

        if self.require_sequential:
            reward += self.sequence_bonus

        if self.N_goals in self.checkpoint_goals:
            reward += self.checkpoint_bonus

        if self.N_goals >= self.N_goals_target:
            reward += self.all_goals_bonus
            reward += self.early_completion_bonus
            if self.break_on_goal:
                self.done = True

        return reward

    def _check_pokedex_rewards(self, env_vars):
        """Base pokedex seen/owned rewards plus pokedex-goal achievement bonuses."""
        reward = 0
        for goal_type in ["seen", "owned"]:
            if env_vars[f"pokedex_{goal_type}"] > getattr(self, f"pokedex_{goal_type}"):
                reward += (
                    self.pokedex_seen_reward
                    if goal_type == "seen"
                    else self.pokedex_owned_reward
                )
                setattr(self, f"pokedex_{goal_type}", env_vars[f"pokedex_{goal_type}"])
                reward += self._check_pokedex_goal_achievement(
                    env_vars[f"pokedex_{goal_type}"], goal_type
                )
        return reward

    # ------------------------------------------------------------------ #
    # Micro rewards: dense, small-magnitude, per-step signals            #
    # (exploration, step penalty, button penalty, distance shaping)      #
    # ------------------------------------------------------------------ #

    def _micro_reward(self, env_vars, button_press, cur_x, cur_y, cur_map, cur_bank):
        reward = 0
        reward += self._exploration_reward(env_vars)
        reward += self._step_penalty()
        reward += self._distance_shaping(cur_x, cur_y, cur_map, cur_bank)
        reward += self._check_party_progress(env_vars)
        reward += self._battle_engagement_reward(env_vars)

        if button_press in ["start", "select"]:
            reward += self.button_penalty

        return reward

    def _battle_engagement_reward(self, env_vars):
        """Per-step battle engagement + damage-dealt reward.

        Outside of battles (``battle_type == 0``), this is a no-op and
        clears ``_prev_enemy_hp`` so HP readings from a finished battle
        cannot leak into the next one.

        Inside a battle, pays:
          - ``battle_engagement_reward`` flat per step (offsets the step
            penalty so engaging combat is no longer net-negative).
          - ``damage_dealt_reward * Δhp`` whenever the enemy's HP drops
            relative to the previous step. ``_prev_enemy_hp`` is seeded
            on the first observed step of a battle, so the initial
            full-HP reading is never credited as damage. HP increases
            (potions / healing items) are ignored.

        ``battle_type`` values outside the valid set ``{0, 1, 2}``
        indicate a mid-transition / garbage RAM snapshot (the engine is
        between overworld and battle states). On those steps we treat
        the snapshot as untrusted: no engagement reward, no damage
        credit, and ``_prev_enemy_hp`` is left untouched so the next
        valid step is compared against the last valid reading.
        """
        battle_type = int(env_vars.get("battle_type", 0))
        if battle_type not in (0, 1, 2):
            return 0
        if battle_type == 0:
            self._prev_enemy_hp = None
            return 0

        enemy_hp = int(env_vars.get("enemy_hp", 0))
        reward = self.battle_engagement_reward

        if self._prev_enemy_hp is not None and enemy_hp < self._prev_enemy_hp:
            damage = self._prev_enemy_hp - enemy_hp
            reward += damage * self.damage_dealt_reward

        self._prev_enemy_hp = enemy_hp
        return reward

    def _check_pokedex_goal_achievement(self, current_value, goal_type):
        if goal_type not in self.pokedex_goals:
            return 0

        threshold = self.pokedex_goals[goal_type]
        fired = self._pokedex_goal_progress.get(goal_type, 0)
        new_fires = min(int(current_value), threshold) - fired
        if new_fires <= 0:
            return 0

        prev_n_goals = self.N_goals
        self.N_goals += new_fires
        self.pokedex_goals_completed += new_fires
        self._pokedex_goal_progress[goal_type] = fired + new_fires
        if fired + new_fires >= threshold:
            del self.pokedex_goals[goal_type]

        # Per-fire reward is already paid once by _check_pokedex_rewards
        # when the underlying count increased; goal-achievement only adds
        # the all-goals bonus on the crossing.
        reward = 0
        if prev_n_goals < self.N_goals_target <= self.N_goals:
            reward += self.all_goals_bonus
            if self.break_on_goal:
                self.done = True

        return reward

    def _exploration_reward(self, env_vars):
        # Include map_bank in the key to prevent collisions between maps
        # with the same number in different bank groups.
        reward = 0
        current_location = (
            env_vars["X"],
            env_vars["Y"],
            env_vars["map_bank"],
            env_vars["map_num"],
        )
        if current_location not in self.explored_tiles:
            self.explored_tiles.add(current_location)
            reward += self.exploration_reward

        # First-visit-per-map bonus. Sized via new_map_reward to be
        # noticeable without dwarfing a goal hit. explored_maps is
        # preserved across the replay → training boundary so maps the
        # replay already discovered do not pay a fresh bonus.
        if self.new_map_reward:
            map_key = (env_vars["map_bank"], env_vars["map_num"])
            if map_key not in self.explored_maps:
                self.explored_maps.add(map_key)
                reward += self.new_map_reward

        return reward

    def _step_penalty(self):
        # Fixed step penalty, no time dependency
        return self.step_penalty

    def _check_level_goals(self, env_vars):
        """Multi-fire level goals based on total party level increases from
        the starting point. {'total_level': 3} fires 3 times as the party
        gains 3 levels. Suppressed on party size change.

        Returns the number of N_goals increments (0, 1, or more).
        """
        if not self.level_goals:
            return 0

        party_size, party_level, _, _ = env_vars["party_info"]

        if self._level_goal_prev_size is None:
            # Seed from the first observation.
            self._level_goal_prev_size = party_size
            self._level_starting_total = party_level
            return 0

        size_changed = party_size != self._level_goal_prev_size

        if size_changed:
            # Party composition changed — update baseline to avoid phantom
            # level gains from a new Pokemon's existing levels.
            self._level_goal_prev_size = party_size
            self._level_starting_total = party_level
            return 0

        # Calculate total levels gained since starting point.
        levels_gained = party_level - self._level_starting_total
        if levels_gained <= 0:
            return 0

        # Fire goals for each level gained, up to the configured threshold.
        reward = 0
        for goal_type, threshold in self.level_goals.items():
            if levels_gained > threshold:
                continue  # Already fully consumed (shouldn't happen in practice)

        # Total fires so far.
        fired_so_far = self.level_goals_completed

        # How many new fires this step.
        new_fires = min(levels_gained, sum(self.level_goals.values())) - fired_so_far
        if new_fires <= 0:
            return 0

        prev_n_goals = self.N_goals
        self.N_goals += new_fires
        self.level_goals_completed += new_fires

        if prev_n_goals < self.N_goals_target <= self.N_goals:
            reward += self.all_goals_bonus
            reward += self.early_completion_bonus
            if self.break_on_goal:
                self.done = True

        return reward

    def _check_xp_goals(self, env_vars):
        """Multi-fire XP goals: fire once per ``xp_goal_threshold`` XP
        gained from the starting baseline, capped at the configured max
        fires. Each fire pays ``xp_goal_reward`` (defaults to
        ``goal_reward``) and counts toward ``N_goals_target``. The all-
        goals and early-completion bonuses are paid when a fire crosses
        the curriculum target.

        Designed as a "won a battle" proxy: with the default
        ``xp_goal_threshold: 10``, a wild-mon win (~10–25 XP at low
        level) typically fires once. Set ``xp_goals: {"total_xp": N}``
        to cap fires at N.

        Suppressed on party size change so a captured Pokemon's existing
        EXP cannot be credited as a win — the baseline is re-seeded and
        future XP gains track from the new total.

        Defense-in-depth: short-circuits on transitional RAM snapshots
        (battle_type outside ``{0, 1, 2}``) even though
        ``calculate_reward`` already gates the whole reward path on
        ``is_ram_state_valid``.
        """
        if not self.xp_goals:
            return 0

        battle_type = int(env_vars.get("battle_type", 0))
        if battle_type not in (0, 1, 2):
            return 0

        party_size, _, _, party_exp = env_vars["party_info"]

        if self._xp_goal_prev_size is None:
            self._xp_goal_prev_size = party_size
            self._xp_goal_starting_total = party_exp
            return 0

        if party_size != self._xp_goal_prev_size:
            # Capture / swap: re-seed baseline so the new mon's EXP isn't
            # credited as a win on the next step.
            self._xp_goal_prev_size = party_size
            self._xp_goal_starting_total = party_exp
            return 0

        if self.xp_goal_threshold <= 0:
            return 0

        xp_gained = party_exp - self._xp_goal_starting_total
        if xp_gained <= 0:
            return 0

        threshold_fires = sum(self.xp_goals.values())
        chunks_crossed = min(
            xp_gained // self.xp_goal_threshold, threshold_fires
        )
        new_fires = chunks_crossed - self.xp_goals_completed
        if new_fires <= 0:
            return 0

        prev_n_goals = self.N_goals
        self.N_goals += new_fires
        self.xp_goals_completed += new_fires

        reward = new_fires * self.xp_goal_reward

        if prev_n_goals < self.N_goals_target <= self.N_goals:
            reward += self.all_goals_bonus + self.early_completion_bonus
            if self.break_on_goal:
                self.done = True

        return reward

    def _check_party_progress(self, env_vars):
        """Reward increases in the party's total level and total EXP.

        When party size changes, skips reward to avoid compounding with
        pokedex-goal / level-up bonuses from captures or party swaps.

        If ``party_reward_check_battle`` is True and the agent is in a battle
        (battle_type != 0), allows rewards even on size change — the XP gain
        is likely from a battle rather than a box swap.

        XP milestone reward: fires once per ``xp_milestone_threshold`` of
        cumulative XP gained. Suppressed on party size change (the new
        Pokemon's existing EXP would inflate the total). The accumulator
        pauses for the size-change step but is NOT reset — it resumes next
        step.

        NOTE: party_reward_check_battle defaults to False until we verify
        that battle_type and XP gain timing align correctly with frame
        stepping. With it disabled, any party size change suppresses the
        reward for that step. This means legitimate mid-battle captures or
        auto-party changes will also be skipped, but it's the safer default
        for testing.
        """
        reward = 0
        # Guard against transitional RAM snapshots: battle_type outside the
        # valid set {0, 1, 2} is only ever seen mid-state-change (e.g. the
        # frames between overworld and the wild-battle UI). Reading party
        # data from those snapshots can return junk values that drive
        # Δexp / Δlevel through the reward clip in a single step. Skip the
        # update entirely on those steps — including the prev trackers —
        # so the next valid step compares against the last valid reading.
        battle_type = int(env_vars.get("battle_type", 0))
        if battle_type not in (0, 1, 2):
            return 0

        party_size, party_level, _, party_exp = env_vars["party_info"]

        if self._prev_party_size is None:
            # Seed from the first observation — no reward yet.
            self._prev_party_size = party_size
            self._prev_party_level = party_level
            self._prev_party_exp = party_exp
            return 0

        size_changed = party_size != self._prev_party_size

        if size_changed:
            # Party composition changed. By default we skip the reward this
            # step to avoid compounding with capture / swap bonuses. If
            # party_reward_check_battle is on AND we're in a battle, allow
            # the reward through (likely legitimate battle XP).
            in_battle = False
            if self.party_reward_check_battle:
                in_battle = env_vars.get("battle_type", 0) != 0

            # Always advance the size tracker so we don't keep hitting this
            # branch on subsequent steps with the same new size.
            self._prev_party_size = party_size

            if not in_battle:
                # Re-seed level/exp too — XP delta across a size change is
                # not meaningful (it includes the new Pokemon's contribution).
                self._prev_party_level = party_level
                self._prev_party_exp = party_exp
                # XP milestone accumulator is NOT reset — it pauses for this
                # step and resumes next step (the new Pokemon's EXP is part
                # of the new baseline, tracked via _prev_party_exp above).
                return 0
            # Fall through: in battle and flag enabled, treat as normal step.

        if self.party_level_reward and party_level > self._prev_party_level:
            reward += (party_level - self._prev_party_level) * self.party_level_reward

        exp_gain = party_exp - self._prev_party_exp
        if self.party_exp_reward and exp_gain > 0:
            reward += exp_gain * self.party_exp_reward

        # XP milestone: accumulate only when party size didn't change (or
        # we're in-battle with party_reward_check_battle). Prevents the
        # new Pokemon's existing EXP from triggering a false milestone.
        if self.xp_milestone_threshold > 0 and exp_gain > 0:
            self._xp_since_milestone += exp_gain
            while self._xp_since_milestone >= self.xp_milestone_threshold:
                self._xp_since_milestone -= self.xp_milestone_threshold
                reward += self.xp_milestone_reward

        self._prev_party_level = party_level
        self._prev_party_exp = party_exp
        return reward

    def get_progress(self):
        return {
            "Steps": self.steps,
            "Goals Reached": self.N_goals,
            "Pokédex Seen": self.pokedex_seen,
            "Pokédex Owned": self.pokedex_owned,
            "Explored Tiles": len(self.explored_tiles),
            "Cumulative Reward": self.cumulative_reward,
        }

    def is_checkpoint_reached(self):
        """Check if the current goal count is a checkpoint"""
        return self.N_goals in self.checkpoint_goals

    def get_current_goal_info(self):
        """Get information about the current goal"""
        if self.current_goal_index < len(self.location_goals):
            goal = list(self.location_goals.values())[self.current_goal_index]
            return {
                "index": self.current_goal_index,
                "locations": goal["positions"],
                "is_checkpoint": (self.current_goal_index + 1) in self.checkpoint_goals,
            }
        return None

    def get_current_target_vector(self):
        """Goal-conditioning signal for the policy.

        Returns (target_x, target_y, target_map, target_map_bank,
        has_active_target). When all location goals are complete the
        target is the final goal's coords + bank, and has_active_target
        is 0 — so the input stays numerically stable but the model can
        route on the flag. `target_map_bank` is 0.0 when the goal entry
        didn't specify a bank (the matching code likewise ignores bank
        in that case).
        """
        if self.current_goal_index < len(self.location_goals):
            goal = list(self.location_goals.values())[self.current_goal_index]
            # positions is [[x, y, bank, map], ...]. Use the first option.
            opt = goal["positions"][0]
            bank = float(opt[2]) if opt[2] is not None else 0.0
            return float(opt[0]), float(opt[1]), float(opt[3]), bank, 1.0
        # All goals done — return the final goal's coords as a stable
        # neutral value, flagged inactive.
        if self.location_goals:
            last_opt = list(self.location_goals.values())[-1]["positions"][0]
            bank = float(last_opt[2]) if last_opt[2] is not None else 0.0
            return float(last_opt[0]), float(last_opt[1]), float(last_opt[3]), bank, 0.0
        return 0.0, 0.0, 0.0, 0.0, 0.0

    def explored_tile_count(self):
        return len(self.explored_tiles)

    def n_location_goals_completed(self):
        return self.current_goal_index

    def n_pokedex_goals_completed(self):
        return self.pokedex_goals_completed

    def n_level_goals_completed(self):
        return self.level_goals_completed

    def n_xp_goals_completed(self):
        return self.xp_goals_completed
