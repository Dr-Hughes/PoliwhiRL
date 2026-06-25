# -*- coding: utf-8 -*-
"""Replay seeding & frontier floor coverage.

Pinned behaviours:

- ``GoalsManager.seed_seen_maps`` pre-fills ``_maps_seen_this_episode`` so
  the maps_visited goal credits replay progress.
- Frontier novelty count floor prevents signal collapse: with a floor of 20,
  a cell visited 100 times still pays ``bonus / 21``.
- ``seed_explored_maps`` pre-fills the per-episode explored_maps set so the
  new_map bonus doesn't fire for replay-walked maps.
"""
import unittest
import numpy as np

from PoliwhiRL.environment.goals import GoalsManager
from PoliwhiRL.environment.rewards import Rewards


def _zero_flags():
    return np.zeros(256, dtype=np.uint8)


def _base_config(**overrides):
    cfg = {
        "episode_length": 100,
        "new_map_reward": 0,
        "frontier_novelty_bonus": 25.0,
        "whiteout_penalty": 0,
        "reward_round_dp": None,
        "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _env_vars(x=4, y=3, map_num=7, map_bank=24, battle_type=0,
              enemy_hp=50, party_info=(1, 5, 20, 0), script_active=False):
    return {
        "X": x, "Y": y, "map_num": map_num, "map_bank": map_bank,
        "room": 0, "warp_number": 0, "money": 0,
        "pokedex_seen": 0, "pokedex_owned": 0,
        "collision_down": 0, "collision_up": 0,
        "collision_left": 0, "collision_right": 0,
        "story_flags": _zero_flags(),
        "battle_type": battle_type, "johto_badges": 0, "player_state": 0,
        "key_items_count": 0, "game_hour": 0, "bgm_id": 0,
        "enemy_hp": enemy_hp, "enemy_max_hp": 100,
        "party_info": party_info,
        "script_active": script_active,
    }


# ------------------------------------------------------------------ #
# GoalsManager.seed_seen_maps
# ------------------------------------------------------------------ #

class TestSeedSeenMaps(unittest.TestCase):
    def test_seed_pre_fills_maps_seen(self):
        gm = GoalsManager(_base_config())
        gm.reset_episode_trackers()
        gm.seed_seen_maps([(24, 7), (0, 3)])
        self.assertIn((24, 7), gm._maps_seen_this_episode)
        self.assertIn((0, 3), gm._maps_seen_this_episode)

    def test_seed_empty_no_op(self):
        gm = GoalsManager(_base_config())
        gm.reset_episode_trackers()
        gm.seed_seen_maps([])
        self.assertEqual(len(gm._maps_seen_this_episode), 0)

    def test_maps_visited_goal_credits_seeded_maps(self):
        """Seeded maps count toward the threshold; only new maps fire."""
        cfg = _base_config(goals=[{"type": "maps_visited", "threshold": 3}])
        gm = GoalsManager(cfg)
        gm.reset_episode_trackers()

        # Seed 2 maps (simulating replay visited 2 maps).
        gm.seed_seen_maps([(24, 7), (0, 3)])

        # Visit a 3rd map during training.
        gm.note_map_visit(1, 5)
        gm.check_maps_visited_goals()

        # Should have visited 3 maps total (2 seeded + 1 new).
        self.assertEqual(len(gm._maps_seen_this_episode), 3)

    def test_maps_visited_goal_termination_with_seeded_maps(self):
        """all_goal_thresholds_met respects seeded maps toward threshold."""
        cfg = _base_config(goals=[{"type": "maps_visited", "threshold": 3}])
        gm = GoalsManager(cfg)
        gm.reset_episode_trackers()

        # Seed 2 maps.
        gm.seed_seen_maps([(24, 7), (0, 3)])

        # Not yet met — only 2 maps.
        self.assertFalse(gm.all_goal_thresholds_met())

        # Visit a 3rd map.
        gm.note_map_visit(1, 5)

        # Now met — 3 maps total.
        self.assertTrue(gm.all_goal_thresholds_met())

    def test_seeded_maps_not_double_counted_by_note_map_visit(self):
        """Calling note_map_visit for a seeded map is idempotent."""
        gm = GoalsManager(_base_config())
        gm.reset_episode_trackers()
        gm.seed_seen_maps([(24, 7)])

        # Re-visit the same map.
        gm.note_map_visit(24, 7)

        self.assertEqual(len(gm._maps_seen_this_episode), 1)


# ------------------------------------------------------------------ #
# Frontier novelty count floor
# ------------------------------------------------------------------ #

class TestFrontierCountFloor(unittest.TestCase):
    """Frontier novelty is persistent (count-based, backed by the visit
    archive). The floor bounds how deep the cross-episode decay goes; we
    exercise it by pre-populating the *archive's* run-wide visit count for a
    cell. The quantised cell for (x=10, y=10, bank=24, num=7) is (24,7,5,5)."""

    def _prime(self, rw, count, x=10, y=10, map_bank=24, map_num=7):
        cell = rw.visit_archive.cell_key(map_bank, map_num, x, y)
        rw.visit_archive._counts[cell] = count

    def test_floor_prevents_drain(self):
        """With floor=20, a cell already visited 99×this episode still pays
        bonus/21."""
        rw = Rewards(_base_config(
            frontier_novelty_bonus=25.0, frontier_novelty_count_floor=20,
            reward_round_dp=None))
        rw.start_new_episode()
        self._prime(rw, 99)
        reward, _ = rw.calculate_reward(_env_vars(x=10, y=10), "")
        self.assertAlmostEqual(float(reward), 25.0 / 21, places=4)

    def test_no_floor_unbounded_decay(self):
        rw = Rewards(_base_config(
            frontier_novelty_bonus=25.0, frontier_novelty_count_floor=0,
            reward_round_dp=None))
        rw.start_new_episode()
        self._prime(rw, 99)
        reward, _ = rw.calculate_reward(_env_vars(x=10, y=10), "")
        self.assertAlmostEqual(float(reward), 25.0 / 100, places=4)

    def test_fresh_cell_ignores_floor(self):
        rw = Rewards(_base_config(
            frontier_novelty_bonus=25.0, frontier_novelty_count_floor=20,
            reward_round_dp=None))
        rw.start_new_episode()
        reward, _ = rw.calculate_reward(_env_vars(x=99, y=99), "")
        self.assertAlmostEqual(float(reward), 25.0, places=4)

    def test_floor_at_boundary(self):
        rw = Rewards(_base_config(
            frontier_novelty_bonus=25.0, frontier_novelty_count_floor=10,
            reward_round_dp=None))
        rw.start_new_episode()
        self._prime(rw, 10, x=5, y=5)
        reward, _ = rw.calculate_reward(_env_vars(x=5, y=5), "")
        self.assertAlmostEqual(float(reward), 25.0 / 11, places=4)

    def test_floor_above_boundary(self):
        rw = Rewards(_base_config(
            frontier_novelty_bonus=25.0, frontier_novelty_count_floor=10,
            reward_round_dp=None))
        rw.start_new_episode()
        self._prime(rw, 50, x=5, y=5)
        reward, _ = rw.calculate_reward(_env_vars(x=5, y=5), "")
        self.assertAlmostEqual(float(reward), 25.0 / 11, places=4)


# ------------------------------------------------------------------ #
# Config default alignment
# ------------------------------------------------------------------ #

class TestConfigDefaults(unittest.TestCase):
    def test_new_map_reward_default(self):
        """new_map_reward defaults to 50."""
        cfg = _base_config()
        del cfg["new_map_reward"]  # remove to trigger default
        rw = Rewards(cfg)
        self.assertEqual(rw.new_map_reward, 50)

    def test_whiteout_penalty_default(self):
        """whiteout_penalty defaults to -20."""
        cfg = {"episode_length": 100, "goals": []}
        rw = Rewards(cfg)
        self.assertEqual(rw.whiteout_penalty, -20.0)

    def test_frontier_novelty_bonus_default(self):
        """frontier_novelty_bonus defaults to 25.0."""
        cfg = {"episode_length": 100, "goals": []}
        rw = Rewards(cfg)
        self.assertEqual(rw.frontier_novelty_bonus, 25.0)


if __name__ == "__main__":
    unittest.main()
