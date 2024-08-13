import numpy as np
from collections import deque

class Rewards:
    def __init__(self, goals=None, N_goals_target=2, max_steps=1000):
        self.xy = set()
        self.pkdex_seen = 0
        self.pkdex_owned = 0
        self.money = 0
        self.max_steps = max_steps
        self.total_level = 0
        self.total_hp = 0
        self.total_exp = 0
        self.done = False
        self.reward_goals = {}
        self.reward_goals_rewards = {}
        self.steps = 0
        self.N_goals_target = N_goals_target
        self.N_goals = 0
        self.explored_tiles = set()
        self.action_history = {}
        self.last_distance = float("inf")
        self.last_location = None
        self.consecutive_non_explore_steps = 0
        self.pokemon_caught = set()
        self.recent_locations = deque(maxlen=10)  # Store recent locations
        self.location_frequency = {}  # Track frequency of visits to each location
        if goals:
            self.set_goals(goals)

    def set_goals(self, goals):
        self.reward_goals = {}
        self.reward_goals_rewards = {}
        for idx, goal in enumerate(goals):
            self.reward_goals[idx] = [option[:-1] for option in goal]
            self.reward_goals_rewards[idx] = goal[0][-1]

    def update_for_goals(self, ram):
        cur_x, cur_y, cur_loc = ram["X"], ram["Y"], ram["map_num_loc"]
        xyl = [cur_x, cur_y, cur_loc]
        for key, value in list(self.reward_goals.items()):
            if xyl in value:
                del self.reward_goals[key]
                time_bonus = (1 - (self.steps / self.max_steps)) ** 3  # Cubic time bonus for quicker completion
                reward = 500 + (2500 * time_bonus)  # Increased reward for reaching significant locations
                self.N_goals += 1
                if self.N_goals == self.N_goals_target:
                    reward += 10000  # Very high reward for completing all goals
                return reward
        return 0

    def update_for_party_pokemon(self, ram):
        total_level, total_hp, total_exp = ram["party_info"]
        reward = 0
        if total_level > np.sum(self.total_level):
            reward += 20  # Increased reward for leveling up
        self.total_level = total_level
        if total_hp > np.sum(self.total_hp):
            reward += 5
        self.total_hp = total_hp
        if total_exp > np.sum(self.total_exp):
            reward += 5
        self.total_exp = total_exp
        return reward

    def update_for_movement(self, ram):
        cur_xy = (ram["X"], ram["Y"])
        cur_loc = ram["map_num_loc"]
        current_location = (cur_xy, cur_loc)
        
        # Update location frequency
        self.location_frequency[current_location] = self.location_frequency.get(current_location, 0) + 1
        
        # Check for repetitive movement
        if len(self.recent_locations) >= 2:
            if current_location == self.recent_locations[-2]:
                return -5  # Significant penalty for going back and forth
        
        self.recent_locations.append(current_location)
        
        if current_location != self.last_location:
            self.consecutive_non_explore_steps = 0
            if current_location not in self.explored_tiles:
                self.explored_tiles.add(current_location)
                self.last_location = current_location
                return 10  # Reward for exploring new tiles
            self.last_location = current_location
            return max(1 - (self.location_frequency[current_location] / 10), 0)  # Diminishing returns for revisits
        
        self.consecutive_non_explore_steps += 1
        return -0.5 * self.consecutive_non_explore_steps  # Increasing penalty for staying in the same place

    def update_for_pokedex(self, ram):
        reward = 0
        if ram["pkdex_seen"] > self.pkdex_seen:
            reward += 50  # Significant reward for seeing new Pokémon
        self.pkdex_seen = ram["pkdex_seen"]
        if ram["pkdex_owned"] > self.pkdex_owned:
            self.pkdex_owned = ram["pkdex_owned"]
            reward += 5000
            self.done = True # moved to done here
        return reward

    def update_for_money(self, ram):
        if ram["money"] > self.money:
            reward = 0.05 * (ram["money"] - self.money)  # Proportional reward for earning money
        elif ram["money"] < self.money:
            reward = -0.05 * (self.money - ram["money"])  # Proportional penalty for losing money
        else:
            reward = 0
        self.money = ram["money"]
        return reward

    def step_penalty(self):
        return -1  # Increased step penalty to encourage quicker actions

    def update_for_timeout(self):
        return -5000  # Severe penalty for timeout

    def calculate_exploration_efficiency(self):
        unique_locations = len(self.explored_tiles)
        total_moves = sum(self.location_frequency.values())
        return unique_locations / max(total_moves, 1)  # Avoid division by zero

    def calc_rewards(self, env_vars, steps):
        self.steps = steps

        time_decay = max(0, 1 - (self.steps / self.max_steps) ** 1.2)  # Sharper time decay

        total_reward = self.step_penalty()

        goal_reward = self.update_for_goals(env_vars) * time_decay
        movement_reward = self.update_for_movement(env_vars)
        pokedex_reward = self.update_for_pokedex(env_vars)

        other_rewards = (
            sum(
                [
                    self.update_for_party_pokemon(env_vars),
                    self.update_for_money(env_vars),
                ]
            )
            * time_decay
            * 0.5  # Reduced weight for less critical rewards
        )

        exploration_efficiency = self.calculate_exploration_efficiency()
        exploration_bonus = exploration_efficiency * 100  # Reward efficient exploration

        total_reward += goal_reward + movement_reward + pokedex_reward + other_rewards + exploration_bonus

        if self.steps == self.max_steps:
            timeout_penalty = self.update_for_timeout()
            total_reward += timeout_penalty

        elif self.done and self.steps < self.max_steps:
            early_completion_bonus = 5000 * (1 - (self.steps / self.max_steps)) ** 3  # Cubic early completion bonus
            total_reward += early_completion_bonus

        return max(min(total_reward, 5000), -5000), self.done  # Increased reward/penalty limits