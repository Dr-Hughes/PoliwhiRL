# -*- coding: utf-8 -*-
"""Persistent (cross-episode) frontier novelty with the AGENT-owned archive.

Pinned behaviour after the global-archive rework:

- ``Rewards`` never writes archive counts. It queues the episode's genuine
  visits in ``_cells_to_record`` / ``_maps_to_record``; the agent merges
  them into the canonical archive (+1 per cell/map per episode) and
  broadcasts the table back to every worker's read-only replica.
- The frontier bonus decays with the canonical count, so a cell pays less
  every episode ANY worker visits it — the town depletes globally and the
  frontier recedes outward, with no hand-authored breadcrumb.
- A cell is still paid at most once per episode (no within-episode farming).
- Replay/seeded steps do NOT queue records, so the evaluator's replay and
  snapshot prefixes can't drain the ledger.
"""
import unittest

import numpy as np

from PoliwhiRL.environment.rewards import Rewards
from PoliwhiRL.environment.visit_archive import VisitArchive


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


def _end_episode_merge(rw):
    """Simulate the worker→agent→worker round trip at episode end: the
    agent merges the episode's pending records into the canonical table
    (here: the same archive instance the Rewards replica reads)."""
    rw.visit_archive.merge_visits(rw._cells_to_record, rw._maps_to_record)


class TestPersistentNovelty(unittest.TestCase):
    def test_decays_across_episodes(self):
        """The same cell pays bonus/1, then bonus/2, then bonus/3 on
        successive episodes once the agent's merge lands — the town
        depletes."""
        rw = Rewards(_cfg())
        ev = _env_vars(x=4, y=3)
        expected = [12.0 / 1, 12.0 / 2, 12.0 / 3]
        for exp in expected:
            rw.start_new_episode()
            reward, _ = rw.calculate_reward(ev, "")
            self.assertAlmostEqual(float(reward), exp, places=4)
            _end_episode_merge(rw)

    def test_no_depletion_without_merge(self):
        """Rewards itself never writes counts: without the agent's merge the
        replica is untouched (a worker's replica is stale by at most one
        rollout — correctness lives in the once-per-episode gate)."""
        rw = Rewards(_cfg())
        for _ in range(3):
            rw.start_new_episode()
            reward, _ = rw.calculate_reward(_env_vars(x=4, y=3), "")
            self.assertAlmostEqual(float(reward), 12.0, places=4)
        self.assertEqual(rw.visit_archive.n_cells_seen(), 0)
        # The pending set queued the visit for the agent.
        self.assertEqual(len(rw._cells_to_record), 1)

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
        # Saturate one cell over several merged episodes.
        for _ in range(5):
            rw.start_new_episode()
            rw.calculate_reward(_env_vars(x=4, y=3), "")
            _end_episode_merge(rw)
        rw.start_new_episode()
        # A distinct, unseen cell still pays full bonus.
        reward, _ = rw.calculate_reward(_env_vars(x=40, y=40), "")
        self.assertAlmostEqual(float(reward), 12.0, places=4)

    def test_global_depletion_across_workers(self):
        """A cell visited by worker A pays less for worker B after the
        broadcast — the landscape is global, not 16 private ones."""
        canonical = VisitArchive()
        rw_a = Rewards(_cfg())
        rw_b = Rewards(_cfg())
        rw_a.start_new_episode()
        rw_a.calculate_reward(_env_vars(x=4, y=3), "")
        # Agent merges A's episode and broadcasts to B's replica.
        canonical.merge_visits(rw_a._cells_to_record, rw_a._maps_to_record)
        rw_b.visit_archive.load_state(canonical.to_state())
        rw_b.start_new_episode()
        reward, _ = rw_b.calculate_reward(_env_vars(x=4, y=3), "")
        self.assertAlmostEqual(float(reward), 12.0 / 2, places=4)

    def test_state_round_trip(self):
        """to_state/load_state preserves both tables (checkpoint path)."""
        a = VisitArchive()
        a.merge_visits([(24, 7, 2, 1), (24, 7, 2, 2)], [(24, 7)])
        a.merge_visits([(24, 7, 2, 1)], [(24, 7)])
        b = VisitArchive()
        b.load_state(a.to_state())
        self.assertEqual(b.count(24, 7, 4, 3), 2)   # cell (24,7,2,1) at CELL_SIZE 2
        self.assertEqual(b.count(24, 7, 4, 5), 1)
        self.assertEqual(b.map_count(24, 7), 2)
        self.assertEqual(b.n_cells_seen(), 2)


if __name__ == "__main__":
    unittest.main()
