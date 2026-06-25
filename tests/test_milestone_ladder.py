# -*- coding: utf-8 -*-
"""Cumulative milestone-ladder goal tracking (from-scratch curriculum).

Pinned behaviours:

- GoalsManager tracks map/pokedex goal completions via calculate_reward.
- ``all_goal_thresholds_met`` returns True only once every configured goal
  fires — intermediate milestones do not end the episode (no
  terminate_on_goal_complete in the simplified reward system).
- ``per_goal_status`` reports which rungs were reached (forgetting detector).
- goal_fire_steps logs the step number of each map-goal rung advance.
"""
import unittest
import numpy as np

from PoliwhiRL.environment.rewards import Rewards


def _zero_flags():
    return np.zeros(256, dtype=np.uint8)


def _base_config(**overrides):
    cfg = {
        "episode_length": 1000,
        "new_map_reward": 0,
        "frontier_novelty_bonus": 0,
        "whiteout_penalty": 0,
        "reward_round_dp": None,
    }
    cfg.update(overrides)
    return cfg


def _env_vars(map_bank=24, map_num=0, pokedex_owned=0):
    return {
        "X": 4, "Y": 3, "map_num": map_num, "map_bank": map_bank,
        "room": 0, "warp_number": 0, "money": 0,
        "pokedex_seen": pokedex_owned, "pokedex_owned": pokedex_owned,
        "collision_down": 0, "collision_up": 0,
        "collision_left": 0, "collision_right": 0,
        "story_flags": _zero_flags(),
        "battle_type": 0, "johto_badges": 0, "player_state": 0,
        "key_items_count": 0, "game_hour": 0, "bgm_id": 0,
        "enemy_hp": 0, "enemy_max_hp": 20,
        "party_info": (1, 5, 20, 0),
        "script_active": False,
    }


class TestMilestoneLadder(unittest.TestCase):
    def _ladder_config(self):
        # Three-map ladder.
        return _base_config(goals=[
            {"type": "map", "map_bank": 24, "map_num": 4},
            {"type": "map", "map_bank": 24, "map_num": 5},
            {"type": "map", "map_bank": 24, "map_num": 3},
        ])

    def test_goals_tracked_progressively(self):
        """GoalsManager advances as map milestones are hit; not all at once."""
        rw = Rewards(self._ladder_config())
        rw.calculate_reward(_env_vars(map_bank=24, map_num=0), "")
        self.assertFalse(rw.goals.all_goal_thresholds_met())

        rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        self.assertFalse(rw.goals.all_goal_thresholds_met())
        self.assertEqual(rw.n_map_goals_completed(), 1)

        rw.calculate_reward(_env_vars(map_bank=24, map_num=5), "")
        self.assertEqual(rw.n_map_goals_completed(), 2)
        self.assertFalse(rw.goals.all_goal_thresholds_met())

        rw.calculate_reward(_env_vars(map_bank=24, map_num=3), "")
        self.assertTrue(rw.goals.all_goal_thresholds_met())

    def test_map_goal_not_double_counted(self):
        """Re-entering a milestone map does not advance the goal counter again."""
        rw = Rewards(self._ladder_config())
        rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        count_after_first = rw.n_map_goals_completed()
        rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        self.assertEqual(rw.n_map_goals_completed(), count_after_first)

    def test_per_goal_status_reports_progress(self):
        rw = Rewards(self._ladder_config())
        rw.calculate_reward(_env_vars(map_bank=24, map_num=0), "")
        rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        status = dict(rw.goals.per_goal_status())
        self.assertTrue(status["map 24/4"])
        self.assertFalse(status["map 24/5"])
        self.assertFalse(status["map 24/3"])

    def test_episode_only_truncates_on_budget(self):
        """Without terminate_on_goal_complete, done is only set by step budget.
        steps > max_steps fires at step max_steps+1."""
        rw = Rewards(_base_config(episode_length=3, goals=[
            {"type": "map", "map_bank": 24, "map_num": 4},
        ]))
        _, d1 = rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        self.assertFalse(d1)  # goal complete but no early termination
        _, d2 = rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        self.assertFalse(d2)
        _, d3 = rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        self.assertFalse(d3)  # step 3 == max_steps, not yet over budget
        _, d4 = rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        self.assertTrue(d4)   # step 4 > max_steps: budget hit
        self.assertTrue(rw.truncated)


if __name__ == "__main__":
    unittest.main()


class TestGoalFireSteps(unittest.TestCase):
    """goal_fire_steps logs the episode step at which each map-goal rung fired —
    the bottleneck-rung / time-budget diagnostic shipped in terminal_info."""

    def test_fire_steps_recorded_for_map_goals(self):
        rw = Rewards(_base_config(goals=[
            {"type": "map", "map_bank": 24, "map_num": 4},
            {"type": "map", "map_bank": 24, "map_num": 3},
        ]))
        rw.start_new_episode()
        rw.calculate_reward(_env_vars(map_num=0), "")   # step 1: nothing
        rw.calculate_reward(_env_vars(map_num=4), "")   # step 2: first map goal
        rw.calculate_reward(_env_vars(map_num=4), "")   # step 3: nothing
        rw.calculate_reward(_env_vars(map_num=3), "")   # step 4: second map goal
        self.assertEqual(rw.goal_fire_steps, [2, 4])

    def test_seeded_progress_not_logged_as_fires(self):
        rw = Rewards(_base_config(goals=[
            {"type": "map", "map_bank": 24, "map_num": 4},
            {"type": "map", "map_bank": 24, "map_num": 3},
        ]))
        # Seed progress manually (as gym_env.restore_snapshot does).
        rw.seed_explored_maps([(24, 4)])
        rw.goals.seed_seen_maps([(24, 4)])
        rw.goals.apply_seed_facts([(24, 4)], [], 0, 0)
        rw._prev_rung = rw.n_flag_goals_completed() + rw.n_map_goals_completed()
        rw.calculate_reward(_env_vars(map_num=4), "")   # step 1: seeded, no fire
        self.assertEqual(rw.goal_fire_steps, [])
        rw.calculate_reward(_env_vars(map_num=3), "")   # step 2: new map goal fires
        self.assertEqual(rw.goal_fire_steps, [2])
