# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict


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

        # Pokedex rewards
        self.pokedex_seen_reward = 50
        self.pokedex_owned_reward = 150

        # Clipping
        self.clip = 1000  # Higher to accommodate integer rewards

        # Goal sequencing
        self.require_sequential = config.get("require_sequential", True)
        self.checkpoint_goals = config.get(
            "checkpoint_goals", [2, 4, 6]
        )  # Major milestones

        # Exploration parameters
        self.exploration_reward = config.get("exploration_reward", 0.0)

        # Party progress rewards: small dense signals for XP gains and larger
        # bonuses when the party's total level increases. Helps prevent the
        # policy from gaming the system by swapping low-level Pokemon in/out.
        self.party_level_reward = config.get("party_level_reward", 0)
        self.party_exp_reward = config.get("party_exp_reward", 0)

        # When True, allows party progress rewards even when party size
        # changes, provided the agent is in a battle (battle_type != 0).
        # This prevents false suppression during mid-battle captures or
        # party swaps. Disabled by default until we verify that battle_type
        # and XP gain timing align correctly with frame stepping.
        self.party_reward_check_battle = config.get("party_reward_check_battle", False)

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

        # Variables for ordered goals
        self.location_goals = OrderedDict()
        self.current_goal_index = 0

        self.set_goals(config["location_goals"], config["pokedex_goals"])

        if self.N_goals_target == -1:
            self.N_goals_target = len(self.location_goals) + len(self.pokedex_goals)

    def update_targets(self, n_goals_target, max_steps):
        self.N_goals_target = n_goals_target
        self.max_steps = max_steps
        self.done = False

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
            pokedex_seen, pokedex_owned (curriculum progress).
          - explored_tiles (so re-walking replay-visited tiles does not
            pay a fresh exploration_reward).
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
        """
        self.done = False
        self.last_action = None
        self.steps = 0
        self.cumulative_reward = 0
        self._prev_party_size = None
        self._prev_party_level = None
        self._prev_party_exp = None

    def set_goals(self, location_goals, pokedex_goals):
        self.location_goals = OrderedDict()
        self.pokedex_goals = {}
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

    def calculate_reward(self, env_vars, button_press):
        self.steps += 1

        cur_x = env_vars["X"]
        cur_y = env_vars["Y"]
        cur_map = env_vars["map_num"]
        cur_bank = env_vars["map_bank"]

        macro = self._macro_reward(env_vars)
        micro = self._micro_reward(env_vars, button_press, cur_x, cur_y, cur_map, cur_bank)
        total_reward = macro + micro

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

        if button_press in ["start", "select"]:
            reward += self.button_penalty

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
        current_location = (
            env_vars["X"],
            env_vars["Y"],
            env_vars["map_bank"],
            env_vars["map_num"],
        )
        if current_location not in self.explored_tiles:
            self.explored_tiles.add(current_location)
            return self.exploration_reward
        return 0

    def _step_penalty(self):
        # Fixed step penalty, no time dependency
        return self.step_penalty

    def _check_party_progress(self, env_vars):
        """Reward increases in the party's total level and total EXP.

        When party size changes, skips reward to avoid compounding with
        pokedex-goal / level-up bonuses from captures or party swaps.

        If ``party_reward_check_battle`` is True and the agent is in a battle
        (battle_type != 0), allows rewards even on size change — the XP gain
        is likely from a battle rather than a box swap.

        NOTE: party_reward_check_battle defaults to False until we verify
        that battle_type and XP gain timing align correctly with frame
        stepping. With it disabled, any party size change suppresses the
        reward for that step. This means legitimate mid-battle captures or
        auto-party changes will also be skipped, but it's the safer default
        for testing. A more robust fix would track per-slot species + EXP to
        isolate which Pokemon gained XP, but that's over-engineered for the
        dense signal we're trying to provide.
        """
        reward = 0
        party_size, party_level, _, party_exp = env_vars["party_info"]

        if self._prev_party_size is None:
            # Seed from the first observation — no reward yet.
            self._prev_party_size = party_size
            self._prev_party_level = party_level
            self._prev_party_exp = party_exp
            return 0

        if party_size != self._prev_party_size:
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
                return 0
            # Fall through: in battle and flag enabled, treat as normal step.

        if self.party_level_reward and party_level > self._prev_party_level:
            reward += (party_level - self._prev_party_level) * self.party_level_reward
        if self.party_exp_reward and party_exp > self._prev_party_exp:
            reward += (party_exp - self._prev_party_exp) * self.party_exp_reward

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
