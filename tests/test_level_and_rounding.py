# -*- coding: utf-8 -*-
"""Coverage for per-step reward rounding and map-goal GoalsManager tracking.

Pinned behaviours:

- ``reward_round_dp`` rounds the returned per-step reward to N decimals.
- Map-type goals track entry into named (map_bank, map_num) targets.
- The starting map is snapshotted on first step so a goal whose target equals
  the start position does not fire spuriously.
"""
import unittest
import numpy as np

from PoliwhiRL.environment.rewards import Rewards


def _zero_flags():
    return np.zeros(256, dtype=np.uint8)


def _base_config(**overrides):
    cfg = {
        "episode_length": 100,
        "new_map_reward": 0,
        "new_bank_reward": 0,
        "frontier_novelty_bonus": 0,
        "whiteout_penalty": 0,
        "step_penalty": 0.0,
        "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _env_vars(
    party_info=(1, 5, 20, 0), battle_type=0, enemy_hp=0, map_bank=24, map_num=7
):
    return {
        "X": 4,
        "Y": 3,
        "map_num": map_num,
        "map_bank": map_bank,
        "room": 0,
        "warp_number": 0,
        "money": 0,
        "pokedex_seen": 0,
        "pokedex_owned": 0,
        "collision_down": 0,
        "collision_up": 0,
        "collision_left": 0,
        "collision_right": 0,
        "story_flags": _zero_flags(),
        "battle_type": battle_type,
        "johto_badges": 0,
        "player_state": 0,
        "key_items_count": 0,
        "game_hour": 0,
        "bgm_id": 0,
        "enemy_hp": enemy_hp,
        "enemy_max_hp": 20,
        "party_info": party_info,
        "script_active": False,
    }


class TestRewardRounding(unittest.TestCase):
    def test_rounds_to_configured_dp(self):
        """Per-step reward rounds to the configured dp. Frontier novelty is
        a flat per-episode bonus now (no cross-episode count-based decay),
        so a non-round bonus value is the simplest way to exercise rounding."""
        rw = Rewards(_base_config(frontier_novelty_bonus=8.336, reward_round_dp=2))
        r, _ = rw.calculate_reward(_env_vars(map_bank=24, map_num=7), "")
        self.assertAlmostEqual(float(r), 8.34, places=6)

    def test_unrounded_when_disabled(self):
        """reward_round_dp=None returns the raw float."""
        rw = Rewards(_base_config(frontier_novelty_bonus=8.336, reward_round_dp=None))
        r, _ = rw.calculate_reward(_env_vars(map_bank=24, map_num=7), "")
        self.assertAlmostEqual(float(r), 8.336, places=5)


class TestMapGoal(unittest.TestCase):
    def _cfg(self):
        return _base_config(
            goals=[{"type": "map", "map_bank": 26, "map_num": 3}],
        )

    def test_fires_on_entering_target_map(self):
        rw = Rewards(self._cfg())
        # Start on New Bark (24, 4) — snapshots the start map, no fire.
        r0, done0 = rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        self.assertAlmostEqual(float(r0), 0.0, places=4)
        self.assertFalse(done0)
        # Enter Cherrygrove City (26, 3) — goal fires, pays map_goal_reward
        # (default 250) — milestones are the primary reward signal.
        r1, done1 = rw.calculate_reward(_env_vars(map_bank=26, map_num=3), "")
        self.assertAlmostEqual(float(r1), 250.0, places=4)
        self.assertFalse(done1)  # no terminate_on_goal_complete by default
        self.assertEqual(rw.n_map_goals_completed(), 1)

    def test_does_not_fire_if_target_is_start_map(self):
        rw = Rewards(self._cfg())
        r0, done0 = rw.calculate_reward(_env_vars(map_bank=26, map_num=3), "")
        self.assertAlmostEqual(float(r0), 0.0, places=4)
        self.assertFalse(done0)
        self.assertEqual(rw.n_map_goals_completed(), 0)

    def test_bank_must_match(self):
        rw = Rewards(self._cfg())
        rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")  # start
        # Same map_num (3) but wrong bank (24) must NOT fire.
        r1, _ = rw.calculate_reward(_env_vars(map_bank=24, map_num=3), "")
        self.assertAlmostEqual(float(r1), 0.0, places=4)
        self.assertEqual(rw.n_map_goals_completed(), 0)


if __name__ == "__main__":
    unittest.main()
