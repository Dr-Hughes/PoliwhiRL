# -*- coding: utf-8 -*-
import numpy as np
from .goals import GoalsManager

_VALID_BATTLE_TYPES = (0, 1, 2)
_VALID_PLAYER_STATES = (0, 1, 2, 4)


def is_ram_state_valid(env_vars):
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
        self.goals = GoalsManager(config)

        # Episode control
        self.max_steps = config["episode_length"]
        self.punish_steps = config.get("punish_steps", True)
        self.require_sequential = config.get("require_sequential", True)
        self.checkpoint_goals = config.get("checkpoint_goals", [2, 4, 6])

        # Reward magnitudes
        self.goal_reward = config.get("goal_reward", 100)
        self.sequence_bonus = config.get("sequence_bonus", 50)
        self.checkpoint_bonus = config.get("checkpoint_bonus", 200)
        self.all_goals_bonus = config.get("all_goals_bonus", 500)
        self.early_completion_bonus = config.get("early_completion_bonus", 0)
        self.soft_waypoint_reward = config.get("soft_waypoint_reward", 25)

        # Penalties
        self.step_penalty = config.get("step_penalty", -1) if self.punish_steps else 0
        self.button_penalty = -5

        # Pokedex rewards
        self.pokedex_seen_reward = config.get("pokedex_seen_reward", 50)
        self.pokedex_owned_reward = config.get("pokedex_owned_reward", 150)

        # Exploration
        self.exploration_reward = config.get("exploration_reward", 0.0)
        self.new_map_reward = config.get("new_map_reward", 0.0)

        # Battle
        self.battle_engagement_reward = config.get("battle_engagement_reward", 0.0)
        self.damage_dealt_reward = config.get("damage_dealt_reward", 0.0)

        # Party progress
        self.party_level_reward = config.get("party_level_reward", 0)
        self.party_exp_reward = config.get("party_exp_reward", 0)
        self.party_reward_check_battle = config.get("party_reward_check_battle", False)

        # XP milestones
        self.xp_milestone_threshold = config.get("xp_milestone_threshold", 0)
        self.xp_milestone_reward = config.get("xp_milestone_reward", 0)

        # XP goals
        self.xp_goal_threshold = config.get("xp_goal_threshold", 10)
        self.xp_goal_reward = config.get("xp_goal_reward", self.goal_reward)

        # Distance shaping
        self.distance_shaping_coef = config.get("distance_shaping_coef", 0.0)
        self._d_prev = None

        # Clipping
        self.clip = 1000

        # State
        self.pokedex_seen = 0
        self.pokedex_owned = 0
        self.done = False
        self.last_action = None
        self.steps = 0
        self.explored_tiles = set()
        self.explored_maps = set()
        self._prev_enemy_hp = None
        self.cumulative_reward = 0

        # Party progress tracking
        self._prev_party_size = None
        self._prev_party_level = None
        self._prev_party_exp = None
        self._xp_since_milestone = 0

    # ------------------------------------------------------------------ #
    # Properties (delegate to GoalsManager)                               #
    # ------------------------------------------------------------------ #

    @property
    def N_goals(self):
        return self.goals.N_goals

    @N_goals.setter
    def N_goals(self, value):
        self.goals.N_goals = value

    @property
    def current_goal_index(self):
        return self.goals.current_goal_index

    @property
    def location_goals(self):
        return self.goals.location_goals

    @property
    def pokedex_goals(self):
        return self.goals.pokedex_goals

    @property
    def level_goals(self):
        return self.goals.level_goals

    @property
    def xp_goals(self):
        return self.goals.xp_goals

    @property
    def pokedex_goals_completed(self):
        return self.goals.pokedex_goals_completed

    @property
    def level_goals_completed(self):
        return self.goals.level_goals_completed

    @property
    def xp_goals_completed(self):
        return self.goals.xp_goals_completed

    @property
    def N_goals_target(self):
        return self.goals.hard_goal_count_target

    @property
    def break_on_goal(self):
        return self.goals.break_on_target

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def update_targets(self, hard_goal_count_target, max_steps):
        self.goals.hard_goal_count_target = hard_goal_count_target
        self.max_steps = max_steps
        self.done = False

    def start_new_episode(self):
        self.done = False
        self.last_action = None
        self.steps = 0
        self.cumulative_reward = 0
        self._prev_party_size = None
        self._prev_party_level = None
        self._prev_party_exp = None
        self._xp_since_milestone = 0
        self._prev_enemy_hp = None
        self.goals.reset_episode_trackers()

    # ------------------------------------------------------------------ #
    # Main reward calculation                                             #
    # ------------------------------------------------------------------ #

    def calculate_reward(self, env_vars, button_press):
        self.steps += 1

        if is_ram_state_valid(env_vars):
            cur_x = env_vars["X"]
            cur_y = env_vars["Y"]
            cur_map = env_vars["map_num"]
            cur_bank = env_vars["map_bank"]
            macro = self._macro_reward(env_vars, cur_x, cur_y, cur_map, cur_bank)
            micro = self._micro_reward(env_vars, button_press, cur_x, cur_y, cur_map, cur_bank)
            total_reward = macro + micro
        else:
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
    # Macro rewards                                                       #
    # ------------------------------------------------------------------ #

    def _macro_reward(self, env_vars, cur_x, cur_y, cur_map, cur_bank):
        reward = 0
        reward += self._check_hard_goal_achievement(cur_x, cur_y, cur_map, cur_bank)
        reward += self._check_soft_goals(cur_x, cur_y, cur_map, cur_bank)
        reward += self._check_pokedex_rewards(env_vars)
        reward += self._check_level_goals(env_vars)
        reward += self._check_xp_goals(env_vars)
        return reward

    def _check_hard_goal_achievement(self, cur_x, cur_y, cur_map, cur_bank):
        hit, goal_info = self.goals.check_hard_goal_achievement(cur_x, cur_y, cur_map, cur_bank)
        if not hit:
            return 0
        self._d_prev = None

        reward = self.goal_reward
        if self.require_sequential:
            reward += self.sequence_bonus
        if self.goals.N_goals in self.checkpoint_goals:
            reward += self.checkpoint_bonus
        if self.goals.is_target_reached():
            reward += self.all_goals_bonus
            reward += self.early_completion_bonus
            if self.goals.break_on_target:
                self.done = True
        return reward

    def _check_soft_goals(self, cur_x, cur_y, cur_map, cur_bank):
        hits = self.goals.check_soft_goals(cur_x, cur_y, cur_map, cur_bank)
        if not hits:
            return 0
        return len(hits) * self.soft_waypoint_reward

    def _check_pokedex_rewards(self, env_vars):
        reward = 0
        seen_increased = env_vars["pokedex_seen"] > self.pokedex_seen
        owned_increased = env_vars["pokedex_owned"] > self.pokedex_owned

        if seen_increased:
            reward += self.pokedex_seen_reward
            self.pokedex_seen = env_vars["pokedex_seen"]
        if owned_increased:
            reward += self.pokedex_owned_reward
            self.pokedex_owned = env_vars["pokedex_owned"]

        if not (seen_increased or owned_increased):
            return reward

        self.goals.check_pokedex_goals(env_vars["pokedex_seen"], env_vars["pokedex_owned"])

        if self.goals.is_target_reached():
            reward += self.all_goals_bonus
            if self.goals.break_on_target:
                self.done = True
        return reward

    # ------------------------------------------------------------------ #
    # Micro rewards                                                       #
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

    def _distance_shaping(self, cur_x, cur_y, cur_map, cur_bank):
        if self.distance_shaping_coef <= 0:
            return 0
        positions = self.goals.active_hard_goal_positions()
        if positions is None:
            self._d_prev = None
            return 0
        target_x, target_y, target_bank, target_map = positions[0]
        goal = self.goals.active_hard_goal()
        if cur_map != target_map:
            self._d_prev = None
            return 0
        if goal and goal["check_bank"] and cur_bank != target_bank:
            self._d_prev = None
            return 0
        d_curr = abs(cur_x - target_x) + abs(cur_y - target_y)
        if self._d_prev is not None and self._d_prev > d_curr:
            shaping = self.distance_shaping_coef * (self._d_prev - d_curr)
            self._d_prev = d_curr
            return shaping
        self._d_prev = d_curr
        return 0

    def _battle_engagement_reward(self, env_vars):
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

    def _exploration_reward(self, env_vars):
        reward = 0
        current_location = (env_vars["X"], env_vars["Y"], env_vars["map_bank"], env_vars["map_num"])
        if current_location not in self.explored_tiles:
            self.explored_tiles.add(current_location)
            reward += self.exploration_reward
        if self.new_map_reward:
            map_key = (env_vars["map_bank"], env_vars["map_num"])
            if map_key not in self.explored_maps:
                self.explored_maps.add(map_key)
                reward += self.new_map_reward
        return reward

    def _step_penalty(self):
        return self.step_penalty

    def _check_level_goals(self, env_vars):
        battle_type = int(env_vars.get("battle_type", 0))
        if battle_type not in (0, 1, 2):
            return 0
        party_size, party_level, _, _ = env_vars["party_info"]
        new_fires = self.goals.check_level_goals(party_size, party_level)
        if new_fires <= 0:
            return 0
        reward = 0
        if self.goals.is_target_reached():
            reward += self.all_goals_bonus
            reward += self.early_completion_bonus
            if self.goals.break_on_target:
                self.done = True
        return reward

    def _check_xp_goals(self, env_vars):
        battle_type = int(env_vars.get("battle_type", 0))
        if battle_type not in (0, 1, 2):
            return 0
        party_size, _, _, party_exp = env_vars["party_info"]
        new_fires = self.goals.check_xp_goals(party_size, party_exp, self.xp_goal_threshold)
        if new_fires <= 0:
            return 0
        reward = new_fires * self.xp_goal_reward
        if self.goals.is_target_reached():
            reward += self.all_goals_bonus + self.early_completion_bonus
            if self.goals.break_on_target:
                self.done = True
        return reward

    def _check_party_progress(self, env_vars):
        reward = 0
        battle_type = int(env_vars.get("battle_type", 0))
        if battle_type not in (0, 1, 2):
            return 0
        party_size, party_level, _, party_exp = env_vars["party_info"]
        if self._prev_party_size is None:
            self._prev_party_size = party_size
            self._prev_party_level = party_level
            self._prev_party_exp = party_exp
            return 0
        size_changed = party_size != self._prev_party_size
        if size_changed:
            in_battle = False
            if self.party_reward_check_battle:
                in_battle = env_vars.get("battle_type", 0) != 0
            self._prev_party_size = party_size
            if not in_battle:
                self._prev_party_level = party_level
                self._prev_party_exp = party_exp
                return 0
        if self.party_level_reward and party_level > self._prev_party_level:
            reward += (party_level - self._prev_party_level) * self.party_level_reward
        exp_gain = party_exp - self._prev_party_exp
        if self.party_exp_reward and exp_gain > 0:
            reward += exp_gain * self.party_exp_reward
        if self.xp_milestone_threshold > 0 and exp_gain > 0:
            self._xp_since_milestone += exp_gain
            while self._xp_since_milestone >= self.xp_milestone_threshold:
                self._xp_since_milestone -= self.xp_milestone_threshold
                reward += self.xp_milestone_reward
        self._prev_party_level = party_level
        self._prev_party_exp = party_exp
        return reward

    # ------------------------------------------------------------------ #
    # Progress queries                                                    #
    # ------------------------------------------------------------------ #

    def get_progress(self):
        return {
            "Steps": self.steps,
            "Goals Reached": self.goals.N_goals,
            "Pokédex Seen": self.pokedex_seen,
            "Pokédex Owned": self.pokedex_owned,
            "Explored Tiles": len(self.explored_tiles),
            "Cumulative Reward": self.cumulative_reward,
        }

    def is_checkpoint_reached(self):
        return self.goals.N_goals in self.checkpoint_goals

    def get_current_goal_info(self):
        if self.goals.has_active_hard_goal():
            goal = self.goals.active_hard_goal()
            return {
                "index": self.goals.current_goal_index,
                "locations": goal["positions"],
                "is_checkpoint": (self.goals.current_goal_index + 1) in self.checkpoint_goals,
            }
        return None

    def get_current_target_vector(self):
        positions = self.goals.active_hard_goal_positions()
        if positions is not None:
            opt = positions[0]
            bank = float(opt[2]) if opt[2] is not None else 0.0
            return float(opt[0]), float(opt[1]), float(opt[3]), bank, 1.0
        if self.goals.location_goals:
            last_goal = list(self.goals.location_goals.values())[-1]
            last_opt = last_goal["positions"][0]
            bank = float(last_opt[2]) if last_opt[2] is not None else 0.0
            return float(last_opt[0]), float(last_opt[1]), float(last_opt[3]), bank, 0.0
        return 0.0, 0.0, 0.0, 0.0, 0.0

    def explored_tile_count(self):
        return len(self.explored_tiles)

    def n_location_goals_completed(self):
        return self.goals.n_hard_goals_completed()

    def n_pokedex_goals_completed(self):
        return self.goals.n_pokedex_goals_completed()

    def n_level_goals_completed(self):
        return self.goals.n_level_goals_completed()

    def n_xp_goals_completed(self):
        return self.goals.n_xp_goals_completed()
