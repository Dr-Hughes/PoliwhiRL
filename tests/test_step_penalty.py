# -*- coding: utf-8 -*-
"""Episode breakdown and reward stream sanity tests.

Pinned behaviours (post-simplification — step_penalty removed):

- get_episode_breakdown() returns only the three pure-exploration keys:
  frontier, new_map, whiteout.
- A valid step with no novelty and no whiteout contributes 0.0 to all
  breakdown entries.
- Cumulative reward increments correctly.
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
        "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _env_vars(x=4, y=3, map_bank=24, map_num=7):
    return {
        "X": x, "Y": y, "map_num": map_num, "map_bank": map_bank,
        "room": 0, "warp_number": 0, "money": 0,
        "pokedex_seen": 0, "pokedex_owned": 0,
        "collision_down": 0, "collision_up": 0,
        "collision_left": 0, "collision_right": 0,
        "story_flags": _zero_flags(),
        "battle_type": 0, "johto_badges": 0, "player_state": 0,
        "key_items_count": 0, "game_hour": 0, "bgm_id": 0,
        "enemy_hp": 0, "enemy_max_hp": 20,
        "party_info": (1, 5, 20, 0),
        "script_active": False,
    }


class TestEpisodeBreakdown(unittest.TestCase):
    def test_breakdown_has_exactly_three_keys(self):
        rw = Rewards(_base_config())
        bd = rw.get_episode_breakdown()
        self.assertEqual(set(bd.keys()), {"frontier", "new_map", "whiteout"})

    def test_zero_reward_step_contributes_nothing(self):
        """All signals off: breakdown stays 0 after a valid step."""
        rw = Rewards(_base_config())
        # Use same cell every step so frontier also returns 0 after first hit.
        rw.calculate_reward(_env_vars(), "")  # first step pays frontier once
        rw.start_new_episode()
        # After the merge happens in a real run the frontier pays 0.5x next ep,
        # but without a merge, on a fresh episode it pays full again. So use
        # a "visited this episode" cell to get true zero.
        rw.calculate_reward(_env_vars(x=4, y=3), "")  # pays frontier
        r, _ = rw.calculate_reward(_env_vars(x=4, y=3), "")  # same cell: 0
        self.assertAlmostEqual(float(r), 0.0, places=4)
        # new_map and whiteout still 0.
        bd = rw.get_episode_breakdown()
        self.assertAlmostEqual(bd["new_map"], 0.0, places=4)
        self.assertAlmostEqual(bd["whiteout"], 0.0, places=4)

    def test_cumulative_reward_increments(self):
        rw = Rewards(_base_config(
            frontier_novelty_bonus=10.0, reward_round_dp=None
        ))
        rw.calculate_reward(_env_vars(x=4, y=3), "")
        rw.calculate_reward(_env_vars(x=6, y=3), "")
        # Both cells pay 10.0 (fresh run, no archive hits).
        self.assertAlmostEqual(rw.cumulative_reward, 20.0, places=4)

    def test_no_step_key_in_breakdown(self):
        """step_penalty is removed; 'step' must not appear in breakdown."""
        rw = Rewards(_base_config())
        self.assertNotIn("step", rw.get_episode_breakdown())


if __name__ == "__main__":
    unittest.main()
