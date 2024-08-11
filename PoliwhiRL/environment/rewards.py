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
        self.done = False
        self.reward_goals = {}
        self.reward_goals_rewards = {}
        self.default_reward = 0.001
        self.steps = 0
        self.N_goals_target = N_goals_target
        self.N_goals = 0
        self.explored_tiles = set()
        self.action_history = {}
        self.last_distance = float('inf')
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
                time_bonus = 1 - (self.steps / self.max_steps)
                reward = 100 + (900 * time_bonus)  # Scales from 100 to 1000 based on time
                self.N_goals += 1
                print(f"Completed goal {key}, reward: {reward}")
                if self.N_goals == self.N_goals_target:
                    print("Completed all required goals")
                    self.done = True
                    reward += 2000  # Extra reward for completing all goals
                return reward
        return 0

    def update_for_goal_progress(self, ram):
        cur_x, cur_y, cur_loc = ram["X"], ram["Y"], ram["map_num_loc"]
        progress_reward = 0
        for goal in self.reward_goals.values():
            for x, y, loc in goal:
                if loc == cur_loc:
                    distance = abs(cur_x - x) + abs(cur_y - y)
                    progress_reward += 1 / (distance + 1)  # Avoid division by zero
        return progress_reward * 0.1  # Scale as needed

    def update_for_party_pokemon(self, ram):
        total_level, total_hp, total_exp = ram["party_info"]
        reward = 0
        if total_level > np.sum(self.total_level):
            reward += 0.5
        self.total_level = total_level
        if total_hp > np.sum(self.total_hp):
            reward += 0.1
        self.total_hp = total_hp
        if total_exp > np.sum(self.total_exp):
            reward += 0.1
        self.total_exp = total_exp
        return reward

    def update_for_movement(self, ram):
        cur_xy = (ram["X"], ram["Y"])
        if cur_xy not in self.xy:
            self.xy.add(cur_xy)
            return 0.05
        return 0

    def update_for_pokedex(self, ram):
        reward = 0
        if ram["pkdex_seen"] > self.pkdex_seen:
            reward += 1
        self.pkdex_seen = ram["pkdex_seen"]
        if ram["pkdex_owned"] > self.pkdex_owned:
            self.pkdex_owned = ram["pkdex_owned"]
            reward += 2
        return reward

    def update_for_money(self, ram):
        if ram["money"] > self.money:
            reward = 0.01
        elif ram["money"] < self.money:
            reward = -0.01
        else:
            reward = 0
        self.money = ram["money"]
        return reward

    def step_penalty(self):
        return -0.1  # Small penalty for each step

    def reward_efficient_exploration(self, ram):
        cur_xy = (ram["X"], ram["Y"])
        if cur_xy not in self.explored_tiles:
            self.explored_tiles.add(cur_xy)
            return 0.1 * (1 - (self.steps / self.max_steps))
        return 0
    
    def update_for_timeout(self):
        return -1000

    def calc_rewards(self, env_vars, steps):
        step_diff = steps - self.steps
        self.steps = steps
        
        time_decay = max(0, 1 - (self.steps / self.max_steps)**2)
        
        total_reward = 0#self.step_penalty() * step_diff
        
        goal_reward = self.update_for_goals(env_vars) * time_decay
        progress_reward = self.update_for_goal_progress(env_vars) * time_decay
        exploration_reward = self.reward_efficient_exploration(env_vars)
        
        other_rewards = sum([
            self.update_for_party_pokemon(env_vars),
            self.update_for_movement(env_vars),
            self.update_for_pokedex(env_vars),
            self.update_for_money(env_vars)
        ]) * time_decay * 0.1
        
        total_reward += (goal_reward + progress_reward + exploration_reward + 
                        other_rewards)
        
        if self.steps == self.max_steps:
            timeout_penalty = self.update_for_timeout()
            total_reward += timeout_penalty
            print(f"Timeout penalty: {timeout_penalty}")
        
        elif self.done and self.steps < self.max_steps:
            early_completion_bonus = 1000 * (1 - (self.steps / self.max_steps))
            total_reward += early_completion_bonus
            print(f"Early completion bonus: {early_completion_bonus}")
        
        return max(min(total_reward, 1000), -1000), self.done