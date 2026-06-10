# -*- coding: utf-8 -*-
"""Cumulative milestone-ladder termination (from-scratch curriculum).

Pinned behaviours:

- A stage listing several goals terminates ONLY when the full set has fired
  (``terminate_on_goal_complete`` + ``all_goal_thresholds_met``), not on the
  first / any intermediate milestone — so intermediate milestones never end the
  episode.
- Each milestone pays its reward exactly once as it is passed.
- ``per_goal_status`` reports which rungs were reached (forgetting detector).
"""
import unittest
import numpy as np

from PoliwhiRL.environment.rewards import Rewards


def _zero_flags():
    return np.zeros(256, dtype=np.uint8)


def _base_config(**overrides):
    cfg = {
        "episode_length": 1000,
        "pokedex_owned_reward": 150,
        "pokedex_first_sight_reward": 0,
        "key_item_pickup_reward": 0,
        "new_map_reward": 0,
        "new_map_first_discovery_reward": 0,
        "frontier_novelty_bonus": 0,
        "battle_engagement_reward": 0,
        "battle_win_reward": 0,
        "damage_dealt_reward": 0,
        "battle_decay_coef": 0,
        "level_up_reward": 0,
        "flag_progress_reward": 0,
        "whiteout_penalty": 0,
        "intrinsic_reward_episode_cap": 0,
        "step_penalty": 0.0,
        "map_goal_reward": 250,
        "reward_round_dp": None,
        "terminate_on_goal_complete": True,
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
        # Map / pokedex / map ladder mirroring the from-scratch stage 3 shape.
        return _base_config(goals=[
            {"type": "map", "map_bank": 24, "map_num": 4},
            {"type": "pokedex", "kind": "owned", "threshold": 1},
            {"type": "map", "map_bank": 24, "map_num": 3},
        ])

    def test_terminates_only_when_full_set_fired(self):
        rw = Rewards(self._ladder_config())
        # Start map (24/0) snapshots the initial map; nothing fires.
        _, d0 = rw.calculate_reward(_env_vars(map_bank=24, map_num=0), "")
        self.assertFalse(d0)
        # Milestone 1: enter New Bark (24/4) -> pays 250, NOT terminal.
        r1, d1 = rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        self.assertAlmostEqual(float(r1), 250.0, places=4)
        self.assertFalse(d1)
        # Milestone 2: own a starter -> pays 150, still NOT terminal.
        r2, d2 = rw.calculate_reward(
            _env_vars(map_bank=24, map_num=4, pokedex_owned=1), "")
        self.assertAlmostEqual(float(r2), 150.0, places=4)
        self.assertFalse(d2)
        # Milestone 3 (deepest): enter Route 29 (24/3) -> pays 250 AND ends.
        r3, d3 = rw.calculate_reward(
            _env_vars(map_bank=24, map_num=3, pokedex_owned=1), "")
        self.assertAlmostEqual(float(r3), 250.0, places=4)
        self.assertTrue(d3)
        self.assertTrue(rw.goals.all_goal_thresholds_met())

    def test_each_map_milestone_pays_once(self):
        rw = Rewards(self._ladder_config())
        rw.calculate_reward(_env_vars(map_bank=24, map_num=0), "")
        rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")  # +250
        # Re-entering the same milestone map does not pay again.
        r, _ = rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        self.assertAlmostEqual(float(r), 0.0, places=4)

    def test_per_goal_status_reports_progress(self):
        rw = Rewards(self._ladder_config())
        rw.calculate_reward(_env_vars(map_bank=24, map_num=0), "")
        rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        status = dict(rw.goals.per_goal_status(pokedex_owned=0))
        self.assertTrue(status["map 24/4"])
        self.assertFalse(status["pokedex owned>=1"])
        self.assertFalse(status["map 24/3"])


if __name__ == "__main__":
    unittest.main()
