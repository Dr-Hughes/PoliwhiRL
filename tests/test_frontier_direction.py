# -*- coding: utf-8 -*-
"""Egocentric per-episode frontier-direction sense (Rewards.frontier_direction,
added 2026-07-16 to break the bank-24 coverage plateau).

Pinned behaviours:

- Returns [dir_x, dir_y, local_saturation]; the direction is a UNIT vector in
  egocentric map axes and stays purely relative (no absolute position leaks).
- A fresh, all-unvisited neighbourhood has no gradient -> (0, 0, 0).
- After laying a trail on one side, the vector points AWAY from the trail
  toward unexplored-this-episode ground, and saturation rises above 0.
- Fully per-episode: start_new_episode wipes the gradient back to fresh.
- radius 0 disables it; scripted frames report all-zeros (stale position).
"""
import math
import unittest
import numpy as np

from PoliwhiRL.environment.rewards import Rewards
from PoliwhiRL.environment.visit_archive import VisitArchive, CELL_SIZE


def _cfg(**overrides):
    cfg = {
        "episode_length": 1000,
        "new_map_reward": 0,
        "new_bank_reward": 0,
        "frontier_novelty_bonus": 1.0,
        "whiteout_penalty": 0,
        "step_penalty": 0.0,
        "reward_round_dp": None,
        "goals": [],
        "frontier_sense_radius": 6,
    }
    cfg.update(overrides)
    return cfg


def _ev(x, y, script_active=False):
    return {
        "X": x, "Y": y, "map_num": 7, "map_bank": 24,
        "room": 0, "warp_number": 0, "money": 0,
        "pokedex_seen": 0, "pokedex_owned": 0,
        "collision_down": 0, "collision_up": 0,
        "collision_left": 0, "collision_right": 0,
        "story_flags": np.zeros(256, dtype=np.uint8),
        "battle_type": 0, "johto_badges": 0, "player_state": 0,
        "key_items_count": 0, "game_hour": 0, "bgm_id": 0,
        "enemy_hp": 0, "enemy_max_hp": 20,
        "party_info": (1, 5, 20, 0),
        "script_active": script_active,
    }


class TestFrontierDirection(unittest.TestCase):
    def _rw(self, **overrides):
        return Rewards(_cfg(**overrides), visit_archive=VisitArchive())

    def test_fresh_neighbourhood_has_no_gradient(self):
        rw = self._rw()
        self.assertEqual(rw.frontier_direction(_ev(20, 20)), [0.0, 0.0, 0.0])

    def test_points_away_from_trail_and_is_unit_length(self):
        rw = self._rw()
        # Lay a trail to the LEFT of (20, 20); unexplored ground is to the right.
        for cx in range(0, 21, CELL_SIZE):
            rw.calculate_reward(_ev(cx, 20), "")
        dx, dy, sat = rw.frontier_direction(_ev(20, 20))
        self.assertGreater(dx, 0.0)                      # pushed toward the fresh side
        self.assertAlmostEqual(dy, 0.0, places=6)        # trail is horizontal
        self.assertAlmostEqual(math.hypot(dx, dy), 1.0, places=6)
        self.assertGreater(sat, 0.0)
        self.assertLess(sat, 1.0)

    def test_resets_each_episode(self):
        rw = self._rw()
        for cx in range(0, 21, CELL_SIZE):
            rw.calculate_reward(_ev(cx, 20), "")
        self.assertNotEqual(rw.frontier_direction(_ev(20, 20)), [0.0, 0.0, 0.0])
        rw.start_new_episode()
        self.assertEqual(rw.frontier_direction(_ev(20, 20)), [0.0, 0.0, 0.0])

    def test_radius_zero_disables(self):
        rw = self._rw(frontier_sense_radius=0)
        for cx in range(0, 21, CELL_SIZE):
            rw.calculate_reward(_ev(cx, 20), "")
        self.assertEqual(rw.frontier_direction(_ev(20, 20)), [0.0, 0.0, 0.0])

    def test_scripted_frame_reports_zero(self):
        rw = self._rw()
        for cx in range(0, 21, CELL_SIZE):
            rw.calculate_reward(_ev(cx, 20), "")
        self.assertEqual(
            rw.frontier_direction(_ev(20, 20, script_active=True)), [0.0, 0.0, 0.0]
        )


if __name__ == "__main__":
    unittest.main()
