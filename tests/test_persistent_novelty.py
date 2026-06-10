# -*- coding: utf-8 -*-
"""Persistent (cross-episode) frontier novelty.

Pinned behaviour after the episodic→persistent rework:

- The frontier bonus decays with the *run-wide* visit count (the
  ``visit_archive`` ledger), so a cell pays less every episode it is
  re-visited — the town depletes and the frontier recedes outward, with no
  hand-authored breadcrumb.
- A cell is still paid at most once per episode (no within-episode farming).
- Replay steps (``_replaying``) do NOT record into the archive, so the
  evaluator's replay can't drain the ledger.
- ``visit_archive.n_cells_seen()`` (the ``archive_size`` progress signal)
  grows as genuinely new cells are discovered.
"""
import unittest

import numpy as np

from PoliwhiRL.environment.rewards import Rewards


def _zero_flags():
    return np.zeros(256, dtype=np.uint8)


def _cfg(**overrides):
    cfg = {
        "episode_length": 100,
        "pokedex_owned_reward": 0, "pokedex_first_sight_reward": 0,
        "key_item_pickup_reward": 0, "new_map_reward": 0,
        "new_map_first_discovery_reward": 0,
        "frontier_novelty_bonus": 12.0, "frontier_novelty_count_floor": 20,
        "battle_engagement_reward": 0.0, "damage_dealt_reward": 0.0,
        "flag_progress_reward": 0, "map_goal_reward": 0,
        "whiteout_penalty": 0, "level_up_reward": 0, "battle_win_reward": 0,
        "intrinsic_reward_episode_cap": 0.0, "step_penalty": 0.0,
        "reward_round_dp": None, "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _env_vars(x=4, y=3, map_num=7, map_bank=24, script_active=False):
    return {
        "X": x, "Y": y, "map_num": map_num, "map_bank": map_bank,
        "room": 0, "warp_number": 0, "money": 0,
        "pokedex_seen": 0, "pokedex_owned": 0,
        "collision_down": 0, "collision_up": 0,
        "collision_left": 0, "collision_right": 0,
        "story_flags": _zero_flags(),
        "battle_type": 0, "johto_badges": 0, "player_state": 0,
        "key_items_count": 0, "game_hour": 0, "bgm_id": 0,
        "enemy_hp": 0, "enemy_max_hp": 100,
        "party_info": (1, 5, 20, 0),
        "script_active": script_active,
    }


class TestPersistentNovelty(unittest.TestCase):
    def test_decays_across_episodes(self):
        """The same cell pays bonus/1, then bonus/2, then bonus/3 on
        successive episodes — the town depletes."""
        rw = Rewards(_cfg())
        ev = _env_vars(x=4, y=3)
        expected = [12.0 / 1, 12.0 / 2, 12.0 / 3]
        for exp in expected:
            rw.start_new_episode()
            reward, _ = rw.calculate_reward(ev, "")
            self.assertAlmostEqual(float(reward), exp, places=4)

    def test_once_per_episode(self):
        """Re-treading a cell within one episode pays only once."""
        rw = Rewards(_cfg())
        rw.start_new_episode()
        ev = _env_vars(x=4, y=3)
        first, _ = rw.calculate_reward(ev, "")
        second, _ = rw.calculate_reward(ev, "")
        self.assertAlmostEqual(float(first), 12.0, places=4)
        self.assertAlmostEqual(float(second), 0.0, places=4)

    def test_fresh_cell_still_pays_full(self):
        """A never-visited cell pays full even after the town is saturated —
        the frontier is where the novelty lives."""
        rw = Rewards(_cfg())
        # Saturate one cell over several episodes.
        for _ in range(5):
            rw.start_new_episode()
            rw.calculate_reward(_env_vars(x=4, y=3), "")
        rw.start_new_episode()
        # A distinct, unseen cell still pays full bonus.
        reward, _ = rw.calculate_reward(_env_vars(x=40, y=40), "")
        self.assertAlmostEqual(float(reward), 12.0, places=4)

    def test_replay_does_not_record(self):
        """Replay steps must not drain the persistent ledger."""
        rw = Rewards(_cfg())
        rw._replaying = True
        rw.start_new_episode()
        rw.calculate_reward(_env_vars(x=4, y=3), "")
        rw._replaying = False
        # Cell was never recorded, so a genuine visit still pays full.
        rw.start_new_episode()
        reward, _ = rw.calculate_reward(_env_vars(x=4, y=3), "")
        self.assertAlmostEqual(float(reward), 12.0, places=4)

    def test_archive_size_grows_with_new_cells(self):
        """n_cells_seen (the archive_size stall signal) tracks distinct
        cells discovered."""
        rw = Rewards(_cfg())
        rw.start_new_episode()
        self.assertEqual(rw.visit_archive.n_cells_seen(), 0)
        rw.calculate_reward(_env_vars(x=4, y=3), "")
        rw.calculate_reward(_env_vars(x=40, y=40), "")
        self.assertEqual(rw.visit_archive.n_cells_seen(), 2)
        # Re-visiting known cells next episode does not grow the count.
        rw.start_new_episode()
        rw.calculate_reward(_env_vars(x=4, y=3), "")
        self.assertEqual(rw.visit_archive.n_cells_seen(), 2)


if __name__ == "__main__":
    unittest.main()
