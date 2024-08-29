# -*- coding: utf-8 -*-
import numpy as np
from collections import deque


class Rewards:
    def __init__(
        self,
        goals=None,
        N_goals_target=2,
        max_steps=1000,
        break_on_goal=True,
        use_cumu_reward=False,
    ):
        self.max_steps = max_steps
        self.N_goals_target = N_goals_target
        self.break_on_goal = break_on_goal
        self.use_cumu_reward = use_cumu_reward
        self.reset()
        if goals:
            self.set_goals(goals)

    def reset(self):
        self.pkdex_seen = 0
        self.pkdex_owned = 0
        self.money = 0
        self.done = False
        self.reward_goals = {}
        self.steps = 0
        self.N_goals = 0
        self.explored_tiles = set()
        self.last_location = None
        self.cumulative_reward = 0

    def set_goals(self, goals):
        self.reward_goals = {}
        for idx, goal in enumerate(goals):
            self.reward_goals[idx] = [option[:-1] for option in goal]

    def calculate_reward(self, env_vars):
        self.steps += 1
        total_reward = 0

        total_reward += self._goal_reward(env_vars)

        total_reward += self._exploration_reward(env_vars)

        total_reward += self._pokedex_reward(env_vars)

        # Increased step penalty to encourage speed
        total_reward -= 1  # Adjusted penalty for each step

        if self.steps >= self.max_steps:
            self.done = True

        # Check for episode termination
        if self.done or self.steps >= self.max_steps:
            total_reward += self._episode_end_reward()

        self.cumulative_reward += total_reward

        # Clip and normalize the reward
        normalized_reward = np.clip(
            total_reward, -500, 500, dtype=np.float64
        )  # Adjusted clipping range
        if self.use_cumu_reward:
            return (
                np.clip(self.cumulative_reward, -10000, 10000, dtype=np.float64),
                self.done,
            )
        return normalized_reward, self.done

    def _goal_reward(self, env_vars):
        cur_x, cur_y, cur_loc = env_vars["X"], env_vars["Y"], env_vars["map_num_loc"]
        xyl = [cur_x, cur_y, cur_loc]
        for key, value in list(self.reward_goals.items()):
            if xyl in value:
                del self.reward_goals[key]
                self.N_goals += 1
                if self.N_goals >= self.N_goals_target:
                    if self.break_on_goal:
                        self.done = True
                # Reward decreases more rapidly over time, but scaled down
                step_factor = max(0, 1 - (self.steps / (self.max_steps * 0.5)))
                return 500 * step_factor  # Reward now ranges from 0 to 500
        return 0

    def _exploration_reward(self, env_vars):
        current_location = ((env_vars["X"], env_vars["Y"]), env_vars["map_num_loc"])
        if current_location not in self.explored_tiles:
            self.explored_tiles.add(current_location)
            return 2
        return 0

    def _pokedex_reward(self, env_vars):
        reward = 0
        if env_vars["pkdex_seen"] > self.pkdex_seen:
            reward += 10  # Scaled down reward for seeing a new Pokémon
            self.pkdex_seen = env_vars["pkdex_seen"]
        if env_vars["pkdex_owned"] > self.pkdex_owned:
            self.pkdex_owned = env_vars["pkdex_owned"]
            reward += 50  # Scaled down reward for capturing a new Pokémon
        return reward

    def _episode_end_reward(self):
        if self.N_goals >= self.N_goals_target:
            # Bonus for completing all goals quickly, scaled down
            time_bonus = max(0, 1 - (self.steps / self.max_steps))
            return 500 * (1 + time_bonus)  # Bonus now ranges from 500 to 1000
        elif self.steps >= self.max_steps:
            return -200  # Scaled down penalty for timeout
        return 0
