# -*- coding: utf-8 -*-
"""Frontier novelty accumulation and episode budget tests.

Post-simplification: the intrinsic reward cap (intrinsic_reward_episode_cap)
has been removed. These tests verify the pure frontier-novelty behaviour
without any cap — the gradient is always uncapped.

Pinned behaviours:

- N fresh cells each paying ``frontier_novelty_bonus`` accumulate without a cap.
- The frontier breakdown entry tracks the total paid.
- After start_new_episode the novel_cells set resets so new visits pay again.
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
        "frontier_novelty_bonus": 10.0,
        "frontier_novelty_count_floor": 0,
        "whiteout_penalty": 0,
        "reward_round_dp": None,
        "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _env_vars(x=0, map_bank=24, map_num=7):
    return {
        "X": x, "Y": 0, "map_num": map_num, "map_bank": map_bank,
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


def _walk_fresh_cells(rw, n, x_start=0):
    """Visit n fresh quantised cells (CELL_SIZE=2 so step X by 2). Returns the
    list of per-step total rewards."""
    rewards = []
    for i in range(n):
        r, _ = rw.calculate_reward(_env_vars(x=x_start + 2 * i), "")
        rewards.append(float(r))
    return rewards


class TestFrontierAccumulation(unittest.TestCase):
    def test_n_fresh_cells_pay_full(self):
        """4 fresh cells × 10 each = 40; no cap."""
        rw = Rewards(_base_config())
        rewards = _walk_fresh_cells(rw, 4)
        self.assertAlmostEqual(sum(rewards), 40.0, places=4)
        bd = rw.get_episode_breakdown()
        self.assertAlmostEqual(bd["frontier"], 40.0, places=4)

    def test_no_intrinsic_capped_key(self):
        """The breakdown no longer has an intrinsic_capped entry."""
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=0), "")
        self.assertNotIn("intrinsic_capped", rw.get_episode_breakdown())

    def test_episode_reset_clears_novel_cells(self):
        """After start_new_episode the same cells pay full again (archive
        not yet merged — simulates the stale-replica scenario)."""
        rw = Rewards(_base_config())
        _walk_fresh_cells(rw, 4)
        rw.start_new_episode()
        rewards = _walk_fresh_cells(rw, 4)
        # Same cells again on fresh episode, archive not merged → still pay full.
        self.assertAlmostEqual(sum(rewards), 40.0, places=4)

    def test_frontier_breakdown_resets_on_new_episode(self):
        rw = Rewards(_base_config())
        _walk_fresh_cells(rw, 4)
        rw.start_new_episode()
        bd = rw.get_episode_breakdown()
        self.assertAlmostEqual(bd["frontier"], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
