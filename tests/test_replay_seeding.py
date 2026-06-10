# -*- coding: utf-8 -*-
"""Replay seeding & frontier floor coverage.

Pinned behaviours:

- ``seed_battle_counts`` pre-fills ``_battles_by_map`` so the training
  segment's first battle on a replay-fought map receives decayed rewards.
- ``GoalsManager.seed_seen_maps`` pre-fills ``_maps_seen_this_episode`` so
  the maps_visited goal credits replay progress.
- Frontier novelty count floor prevents signal collapse: with a floor of 20,
  a cell visited 100 times still pays ``bonus / 21``.
- Full replay → seed → training cycle preserves battle counts and map seeds
  through the ``start_new_episode`` boundary.
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
        "pokedex_owned_reward": 0,
        "pokedex_first_sight_reward": 0,
        "key_item_pickup_reward": 0,
        "new_map_reward": 0,
        "new_map_first_discovery_reward": 0,
        "frontier_novelty_bonus": 25.0,
        "battle_engagement_reward": 10.0,
        "damage_dealt_reward": 1.0,
        "battle_decay_coef": 0.2,
        "flag_progress_reward": 0,
        "map_goal_reward": 0,
        "whiteout_penalty": 0,
        "level_up_reward": 0,
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
# seed_battle_counts
# ------------------------------------------------------------------ #

class TestSeedBattleCounts(unittest.TestCase):
    def test_seed_pre_fills_battles_by_map(self):
        rw = Rewards(_base_config())
        rw.start_new_episode()
        rw.seed_battle_counts({(24, 7): 3})
        self.assertEqual(rw._battles_by_map[(24, 7)], 3)

    def test_seed_multiple_maps(self):
        rw = Rewards(_base_config())
        rw.start_new_episode()
        rw.seed_battle_counts({(24, 7): 1, (0, 3): 5})
        self.assertEqual(rw._battles_by_map[(24, 7)], 1)
        self.assertEqual(rw._battles_by_map[(0, 3)], 5)

    def test_seed_empty_no_op(self):
        rw = Rewards(_base_config())
        rw.start_new_episode()
        rw.seed_battle_counts({})
        self.assertEqual(rw._battles_by_map, {})

    def test_seeded_map_pays_no_entry_bonus(self):
        """A replay-seeded map already counts as engaged, so the
        first-per-map entry bonus does not pay there (anti-double-pay)."""
        rw = Rewards(_base_config(
            battle_engagement_reward=10.0,
            battle_decay_coef=0.2,
            battle_reward_episode_cap=0,
            frontier_novelty_bonus=0,  # isolate battle reward
        ))
        rw.start_new_episode()
        # Seed 1 battle already fought on map (24, 7).
        rw.seed_battle_counts({(24, 7): 1})

        # Enter a battle on the same map — not the first engagement here.
        ev = _env_vars(battle_type=1, map_num=7, map_bank=24)
        reward, _ = rw.calculate_reward(ev, "")
        self.assertAlmostEqual(float(reward), 0.0, places=4)

    def test_seed_does_not_affect_unseeded_map(self):
        """First battle on a map not in the seed pays the full (n=1 decayed)
        entry bonus."""
        rw = Rewards(_base_config(
            battle_engagement_reward=10.0,
            battle_decay_coef=0.2,
            battle_reward_episode_cap=0,
            frontier_novelty_bonus=0,  # isolate battle reward
        ))
        rw.start_new_episode()
        rw.seed_battle_counts({(24, 7): 3})

        # Enter battle on a different map — fresh, first-on-map.
        ev = _env_vars(battle_type=1, map_num=50, map_bank=3)
        reward, _ = rw.calculate_reward(ev, "")

        # Full engagement: 10 * 1/(1 + 0.2*1) = 10/1.2 ≈ 8.33
        expected = 10.0 / (1.0 + 0.2 * 1)
        self.assertAlmostEqual(float(reward), expected, places=4)

    def test_seed_after_reset_preserves_counts(self):
        """Simulate: replay fights → start_new_episode → seed restores."""
        rw = Rewards(_base_config(frontier_novelty_bonus=0))

        # Simulate replay fighting 2 battles on map (24, 7).
        ev = _env_vars(battle_type=0)
        rw.calculate_reward(ev, "")  # baseline non-battle

        for _ in range(2):
            # Toggle: battle → no-battle → battle to trigger entry twice.
            ev_battle = _env_vars(battle_type=1, map_num=7, map_bank=24)
            rw.calculate_reward(ev_battle, "")
            ev_out = _env_vars(battle_type=0)
            rw.calculate_reward(ev_out, "")

        self.assertEqual(rw._battles_by_map.get((24, 7), 0), 2)

        # Capture counts before reset (simulating replay_actions).
        captured = dict(rw._battles_by_map)

        # start_new_episode wipes the dict.
        rw.start_new_episode()
        self.assertEqual(rw._battles_by_map, {})

        # seed restores.
        rw.seed_battle_counts(captured)
        self.assertEqual(rw._battles_by_map[(24, 7)], 2)


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
        """Flat per-episode new_map_reward now defaults to 0 (legacy path
        retired in favour of the first-discovery bonus)."""
        cfg = _base_config()
        del cfg["new_map_reward"]  # remove to trigger default
        rw = Rewards(cfg)
        self.assertEqual(rw.new_map_reward, 0)

    def test_new_map_first_discovery_default(self):
        """First-discovery bonus default."""
        cfg = _base_config()
        del cfg["new_map_first_discovery_reward"]
        rw = Rewards(cfg)
        self.assertEqual(rw.new_map_first_discovery_reward, 50)

    def test_battle_decay_coef_default(self):
        """Hardcoded default matches curriculum_base.json (0.2)."""
        cfg = _base_config()
        del cfg["battle_decay_coef"]
        rw = Rewards(cfg)
        self.assertEqual(rw.battle_decay_coef, 0.2)

    def test_level_up_reward_default(self):
        """Hardcoded default matches curriculum_base.json (10)."""
        cfg = _base_config()
        del cfg["level_up_reward"]
        rw = Rewards(cfg)
        self.assertEqual(rw.level_up_reward, 10)


if __name__ == "__main__":
    unittest.main()
