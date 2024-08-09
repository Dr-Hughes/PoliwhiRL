# -*- coding: utf-8 -*-
import math
import numpy as np


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
        self.goal = ""
        self.done = False
        self.reward_goals = {}
        self.reward_goals_rewards = {}
        self.default_reward = 0.01
        self.steps = 0
        self.steps_since_goal = 0
        self.N_goals_target = N_goals_target
        self.goal_completion_times = []
        self.N_goals = 0
        if goals:
            self.set_goals(goals)

    def set_goals(self, goals):

        self.reward_goals = {}
        self.reward_goals_rewards = {}

        for idx, goal in enumerate(goals):
            self.reward_goals[idx] = []
            for idy, option in enumerate(goal):
                self.reward_goals[idx].append(option[:-1])
                self.reward_goals_rewards[idx] = option[-1]
        self.start_time = self.steps  # Record the start time when goals are set

    def update_for_goals(self, ram):
        reward = 0
        cur_x, cur_y, cur_loc = ram["X"], ram["Y"], ram["map_num_loc"]
        xyl = [cur_x, cur_y, cur_loc]

        for key, value in list(
            self.reward_goals.items()
        ):  # Use list() to avoid runtime error
            for idx, goal in enumerate(value):
                if xyl == goal:
                    del self.reward_goals[key]
                    time_taken = self.steps - self.start_time
                    decay_factor = math.exp(-0.005 * time_taken)  # Exponential decay
                    reward = self.reward_goals_rewards[key] * decay_factor
                    reward = max(reward, self.default_reward * 25)
                    self.N_goals += 1
                    self.goal_completion_times.append(time_taken)
                    print(
                        f"Completed goal {key}, reward: {reward}, time taken: {time_taken}"
                    )
                    if self.N_goals == self.N_goals_target:
                        print("Completed all required goals")
                        self.done = True
                        # Add bonus for completing all goals
                        avg_time = sum(self.goal_completion_times) / len(
                            self.goal_completion_times
                        )
                        bonus = self.default_reward * 100 * math.exp(-0.001 * avg_time)
                        reward += bonus
                        print(f"All goals bonus: {bonus}")
                    return reward
        return reward

    def update_for_party_pokemon(self, ram):
        total_level, total_hp, total_exp = ram["party_info"]
        reward = 0
        if total_level > np.sum(self.total_level):
            reward += self.default_reward * 1000
        self.total_level = total_level
        if total_hp > np.sum(self.total_hp):
            reward += self.default_reward * 100
        self.total_hp = total_hp
        if total_exp > np.sum(self.total_exp):
            reward += self.default_reward * 100
        self.total_exp = total_exp
        return reward

    def update_for_movement(self, ram):
        reward = 0
        cur_xy = (ram["X"], ram["Y"])
        if cur_xy not in self.xy:
            reward += self.default_reward * 5
        self.xy.add(cur_xy)
        return reward

    def update_for_pokedex(self, ram):
        reward = 0
        if ram["pkdex_seen"] > self.pkdex_seen:
            reward += self.default_reward * 100
        self.pkdex_seen = ram["pkdex_seen"]
        if ram["pkdex_owned"] > self.pkdex_owned:
            self.pkdex_owned = ram["pkdex_owned"]
            reward += self.default_reward * 200
            # set as goal complete
            if self.goal == "pkmn":
                print("Got a new pokemon!")
                self.done = True
        return reward

    def update_for_money(self, ram):
        reward = 0
        if ram["money"] > self.money:
            reward += self.default_reward * 3
        elif ram["money"] < self.money:
            reward -= self.default_reward * 3
        self.money = ram["money"]
        return reward

    def calc_rewards(self, env_vars, steps):
        self.steps = steps
        total_reward = -self.default_reward  # negative reward for not doing anything

        # Time-based scaling factor
        time_factor = 1 - (self.steps / self.max_steps)

        for f in [
            self.update_for_party_pokemon,
            self.update_for_movement,
            self.update_for_pokedex,
            self.update_for_money,
            self.update_for_goals,
        ]:
            total_reward += f(env_vars) * time_factor

        # Clip reward between -1 and 1
        total_reward = max(min(total_reward, 1), -1)

        return total_reward, self.done
