# -*- coding: utf-8 -*-
import numpy as np


class Rewards:
    def __init__(
        self,
        goals=None,
        N_goals_target=2,
        max_steps=1000,
        break_on_goal=True,
        break_on_leveling=False,
    ):
        self.max_steps = max_steps
        self.N_goals_target = N_goals_target
        self.break_on_goal = break_on_goal
        self.reset()
        if goals:
            self.set_goals(goals)

    def reset(self):
        self.pkdex_seen = 0
        self.pkdex_owned = 0
        self.done = False
        self.reward_goals = {}
        self.steps = 0
        self.N_goals = 0
        self.explored_tiles = set()
        self.last_location = None
        self.cumulative_reward = 0
        self.exploration_decay = 1.0  # New: for decaying exploration reward

    def set_goals(self, goals):
        self.reward_goals = {}
        for idx, goal in enumerate(goals):
            self.reward_goals[idx] = [option[:-1] for option in goal]

    def calculate_reward(self, env_vars, button_press):
        self.steps += 1
        total_reward = 0

        # Goal reward
        total_reward += self._goal_reward(env_vars)

        # Exploration reward (now with decay)
        total_reward += self._exploration_reward(env_vars) * self.exploration_decay
        self.exploration_decay *= 0.999  # Decay the exploration reward

        # Pokedex reward (now scaled by rarity)
        total_reward += self._pokedex_reward(env_vars)

        # Step penalty
        total_reward += self._step_penalty()

        # Punish for select and start
        if button_press in ["start", "select"]:
            total_reward -= 0.5

        # Check for episode termination
        if self.done or self.steps >= self.max_steps:
            total_reward += self._episode_end_reward()

        self.cumulative_reward += total_reward

        # Normalize the reward
        normalized_reward = np.clip(total_reward, -5, 5)  # Increased range
        return normalized_reward, self.done

    def _goal_reward(self, env_vars):
        cur_x, cur_y, cur_loc = env_vars["X"], env_vars["Y"], env_vars["map_num_loc"]
        xyl = [cur_x, cur_y, cur_loc]
        for key, value in list(self.reward_goals.items()):
            if xyl in value:
                del self.reward_goals[key]
                self.N_goals += 1
                if self.N_goals >= self.N_goals_target and self.break_on_goal:
                    self.done = True
                # Increased reward for reaching a goal
                return 5.0 * (1 - self.steps / self.max_steps)
        return 0

    def _exploration_reward(self, env_vars):
        current_location = ((env_vars["X"], env_vars["Y"]), env_vars["map_num_loc"])
        if current_location not in self.explored_tiles:
            self.explored_tiles.add(current_location)
            return 0.1  # Small positive reward for exploring new tiles
        return 0

    def _pokedex_reward(self, env_vars):
        reward = 0
        if env_vars["pkdex_seen"] > self.pkdex_seen:
            reward += 0.1  # Reward for seeing a new Pokémon
            self.pkdex_seen = env_vars["pkdex_seen"]
        if env_vars["pkdex_owned"] > self.pkdex_owned:
            reward += 0.3  # Larger reward for capturing a new Pokémon
            self.pkdex_owned = env_vars["pkdex_owned"]
        return reward

    def _step_penalty(self):
        return -0.01  # Small negative reward for each step to encourage efficiency

    def _episode_end_reward(self):
        if self.N_goals >= self.N_goals_target:
            # Increased bonus for completing all goals
            return 10.0 * (1 - self.steps / self.max_steps)
        elif self.steps > self.max_steps:
            return -1.0  # Increased penalty for timeout
        return 0

    def get_progress(self):
        return {
            "Steps": self.steps,
            "Goals Reached": self.N_goals,
            "Pokédex Seen": self.pkdex_seen,
            "Pokédex Owned": self.pkdex_owned,
            "Explored Tiles": len(self.explored_tiles),
            "Cumulative Reward": self.cumulative_reward,
        }
