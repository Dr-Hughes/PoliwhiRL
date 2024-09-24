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

        # Simplified reward values
        self.small_reward = 0.1
        self.medium_reward = 0.5
        self.large_reward = 1.0
        self.small_penalty = -0.1
        self.medium_penalty = -0.5
        self.large_penalty = -1.0

        # Goal achievement rewards
        self.goal_reward_max = 10.0
        self.goal_reward_min = 1.0

        # Clipping
        self.clip = 20

        # Other rewards and penalties
        self.exploration_reward = self.small_reward
        self.step_penalty = self.small_penalty if self.punish_steps else 0
        self.button_penalty = self.medium_penalty
        self.pokedex_seen_reward = self.medium_reward
        self.pokedex_owned_reward = self.large_reward

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

        # New variables for ordered goals
        self.location_goals = OrderedDict()
        self.current_goal_index = 0

        # New parameter for distance-based reward
        self.distance_reward_factor = self.medium_reward

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
                self.location_goals[idx] = [option[:-1] for option in goal]
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

        return np.clip(total_reward, -self.clip, self.clip, dtype=np.float16), self.done

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
                # Calculate decaying reward
                progress = self.steps / self.max_steps
                reward = self.goal_reward_max - (
                    (self.goal_reward_max - self.goal_reward_min) * progress
                )
                reward = max(self.goal_reward_min, min(self.goal_reward_max, reward))
                if self.N_goals >= self.N_goals_target:
                    reward *= 2  # Double reward for reaching all goals
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
            # Calculate decaying reward
            progress = self.steps / self.max_steps
            reward = self.goal_reward_max - (
                (self.goal_reward_max - self.goal_reward_min) * progress
            )
            reward = max(self.goal_reward_min, min(self.goal_reward_max, reward))
            if self.N_goals >= self.N_goals_target:
                reward *= 2  # Double reward for reaching all goals
                if self.break_on_goal:
                    self.done = True
            return reward
        return 0

    def _exploration_reward(self, env_vars):
        current_location = ((env_vars["X"], env_vars["Y"]), env_vars["map_num"])
        if current_location not in self.explored_tiles:
            self.explored_tiles.add(current_location)
            dist_reward = self._distance_based_reward(env_vars)
            return self.exploration_reward + dist_reward
        return 0

    def _step_penalty(self):
        return self.step_penalty

    def _distance_based_reward(self, env_vars):
        current_location = (env_vars["X"], env_vars["Y"])
        goal = list(self.location_goals.values())[self.current_goal_index][0]
        if env_vars["map_num"] == goal[2]:  # Check if on the same map
            distance = np.sqrt(
                (current_location[0] - goal[0]) ** 2
                + (current_location[1] - goal[1]) ** 2
            )
            return self.distance_reward_factor * (1 / (distance + 1))
        return 0

    def get_progress(self):
        return {
            "Steps": self.steps,
            "Goals Reached": self.N_goals,
            "Pokédex Seen": self.pokedex_seen,
            "Pokédex Owned": self.pokedex_owned,
            "Explored Tiles": len(self.explored_tiles),
            "Cumulative Reward": self.cumulative_reward,
        }
