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
        self.default_reward = 0.01
        self.steps = 0
        self.N_goals_target = N_goals_target
        self.N_goals = 0
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
                reward = self.reward_goals_rewards[key]
                self.N_goals += 1
                print(f"Completed goal {key}, reward: {reward}")
                if self.N_goals == self.N_goals_target:
                    print("Completed all required goals")
                    self.done = True
                return reward
        return 0

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
        cur_xy = (ram["X"], ram["Y"])
        if cur_xy not in self.xy:
            self.xy.add(cur_xy)
            return self.default_reward * 5
        return 0

    def update_for_pokedex(self, ram):
        reward = 0
        if ram["pkdex_seen"] > self.pkdex_seen:
            reward += self.default_reward * 100
        self.pkdex_seen = ram["pkdex_seen"]
        if ram["pkdex_owned"] > self.pkdex_owned:
            self.pkdex_owned = ram["pkdex_owned"]
            reward += self.default_reward * 200
        return reward

    def update_for_money(self, ram):
        if ram["money"] > self.money:
            reward = self.default_reward * 3
        elif ram["money"] < self.money:
            reward = -self.default_reward * 3
        else:
            reward = 0
        self.money = ram["money"]
        return reward

    def calc_rewards(self, env_vars, steps):
        self.steps = steps
        total_reward = 0

        time_factor = 1 - (self.steps / self.max_steps)

        for f in [
            self.update_for_party_pokemon,
            self.update_for_movement,
            self.update_for_pokedex,
            self.update_for_money,
            self.update_for_goals,
        ]:
            total_reward += f(env_vars) * time_factor

        return max(min(total_reward, 1), -1), self.done