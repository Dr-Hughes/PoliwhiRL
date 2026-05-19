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
        self.pokedex_seen_reward = 25
        self.pokedex_owned_reward = 50

        # Clipping
        self.clip = 1000  # Higher to accommodate integer rewards

        # Goal sequencing
        self.require_sequential = config.get("require_sequential", True)
        self.checkpoint_goals = config.get(
            "checkpoint_goals", [2, 4, 6]
        )  # Major milestones

        # Exploration parameters
        self.exploration_reward = config.get("exploration_reward", 0.0)

        # State variables
        self.pokedex_seen = 0
        self.pokedex_owned = 0
        self.done = False
        self.last_action = None
        self.steps = 0
        self.N_goals = 0
        self.explored_tiles = set()
        self.last_location = None
        self.cumulative_reward = 0
        self.allowed_pokedex_goals = ["seen", "owned"]

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

    def set_goals(self, location_goals, pokedex_goals):
        self.location_goals = OrderedDict()
        self.pokedex_goals = {}
        if location_goals:
            for idx, goal in enumerate(location_goals):
                # Handle both list format [[x,y,map,room]] and dict format [{"x":1,"y":2,"map":3}]
                processed_goal = []
                for option in goal:
                    if isinstance(option, dict):
                        # Dict format - extract x, y, map values
                        processed_goal.append(
                            [
                                option.get("x", 0),
                                option.get("y", 0),
                                option.get("map", 0),
                            ]
                        )
                    elif isinstance(option, list):
                        # List format - take first 3 elements (x, y, map)
                        processed_goal.append(option[:3])
                    else:
                        raise ValueError(f"Unknown goal format: {option}")
                self.location_goals[idx] = processed_goal
        if pokedex_goals:
            if isinstance(pokedex_goals, dict):
                for k, v in pokedex_goals.items():
                    if k in self.allowed_pokedex_goals:
                        self.pokedex_goals[k] = v
            else:
                raise ValueError("Pokedex goals must be a dictionary")

    def calculate_reward(self, env_vars, button_press):
        self.steps += 1
        total_reward = 0

        total_reward += self._check_goals(env_vars)
        total_reward += self._exploration_reward(env_vars)
        total_reward += self._step_penalty()

        if button_press in ["start", "select"]:
            total_reward += self.button_penalty

        self.last_action = button_press

        if self.done or self.steps > self.max_steps:
            self.done = True

        self.cumulative_reward += total_reward
        clipped_reward = np.clip(total_reward, -self.clip, self.clip).astype(np.float32)

        return clipped_reward, self.done

    def _check_goals(self, env_vars):
        reward = 0

        # Check location goals
        cur_x, cur_y, cur_loc, _ = (
            env_vars["X"],
            env_vars["Y"],
            env_vars["map_num"],
            env_vars["room"],
        )
        xyl = [cur_x, cur_y, cur_loc]
        reward += self._check_goal_achievement(xyl)

        # Check pokedex goals
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

    def _check_goal_achievement(self, current_value):
        if self.current_goal_index < len(self.location_goals):
            goal = list(self.location_goals.values())[self.current_goal_index]
            if current_value in goal:
                self.current_goal_index += 1
                self.N_goals += 1

                # Fixed integer reward
                reward = self.goal_reward

                # Sequential bonus for completing in order
                if self.require_sequential:
                    reward += self.sequence_bonus

                # Checkpoint bonus for major milestones
                if self.N_goals in self.checkpoint_goals:
                    reward += self.checkpoint_bonus

                # All goals completed bonus
                if self.N_goals >= self.N_goals_target:
                    reward += self.all_goals_bonus
                    reward += self.early_completion_bonus
                    if self.break_on_goal:
                        self.done = True

                return reward
        return 0

    def _check_pokedex_goal_achievement(self, current_value, goal_type):
        if (
            goal_type in self.pokedex_goals
            and current_value >= self.pokedex_goals[goal_type]
        ):
            del self.pokedex_goals[goal_type]
            self.N_goals += 1

            # Fixed integer reward (already given base reward above)
            reward = 0

            # All goals completed bonus
            if self.N_goals >= self.N_goals_target:
                reward += self.all_goals_bonus
                if self.break_on_goal:
                    self.done = True

            return reward
        return 0

    def _exploration_reward(self, env_vars):
        current_location = ((env_vars["X"], env_vars["Y"]), env_vars["map_num"])
        if current_location not in self.explored_tiles:
            self.explored_tiles.add(current_location)
            return self.exploration_reward
        return 0

    def _step_penalty(self):
        # Fixed step penalty, no time dependency
        return self.step_penalty

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
                "locations": goal,
                "is_checkpoint": (self.current_goal_index + 1) in self.checkpoint_goals,
            }
        return None

    def get_current_target_vector(self):
        """Goal-conditioning signal for the policy.

        Returns (target_x, target_y, target_map, has_active_target).
        When all location goals are complete the target is the player's
        most recently expected location (last goal in the sequence) and
        has_active_target is 0 — so the input remains numerically stable
        but the model can route on the flag.
        """
        if self.current_goal_index < len(self.location_goals):
            goal = list(self.location_goals.values())[self.current_goal_index]
            # `goal` is a list of acceptable [x, y, map] options. Use the
            # first option as the canonical representative target.
            x, y, map_num = goal[0][0], goal[0][1], goal[0][2]
            return float(x), float(y), float(map_num), 1.0
        # All goals done — return the final goal's coords as a stable
        # neutral value, flagged inactive.
        if self.location_goals:
            last = list(self.location_goals.values())[-1][0]
            return float(last[0]), float(last[1]), float(last[2]), 0.0
        return 0.0, 0.0, 0.0, 0.0

    def explored_tile_count(self):
        return len(self.explored_tiles)
