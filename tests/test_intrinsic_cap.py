# -*- coding: utf-8 -*-
"""Per-episode intrinsic reward cap.

Pinned behaviours:

- The total intrinsic reward (new_map + frontier + battle + level) paid in a
  single episode is clamped to ``intrinsic_reward_episode_cap`` — the backstop
  that stops dense exploration out-paying the one-time milestone reward on a
  long route.
- The clipped amount is logged as a negative ``intrinsic_capped`` breakdown
  entry, so gross frontier + the clawback still sum to the actual reward.
- The budget resets every episode.
- ``intrinsic_reward_episode_cap <= 0`` disables the cap (no clamping).
"""
import unittest
import numpy as np

from PoliwhiRL.environment.rewards import Rewards


def _zero_flags():
    return np.zeros(256, dtype=np.uint8)


def _base_config(**overrides):
    cfg = {
        "episode_length": 1000,
        "pokedex_owned_reward": 0,
        "pokedex_first_sight_reward": 0,
        "key_item_pickup_reward": 0,
        "new_map_reward": 0,
        "new_map_first_discovery_reward": 0,
        "frontier_novelty_bonus": 10.0,
        "frontier_novelty_count_floor": 20,
        "battle_engagement_reward": 0,
        "battle_win_reward": 0,
        "damage_dealt_reward": 0,
        "battle_decay_coef": 0,
        "level_up_reward": 0,
        "flag_progress_reward": 0,
        "whiteout_penalty": 0,
        "step_penalty": 0.0,
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


class TestIntrinsicCap(unittest.TestCase):
    def test_cap_clamps_episode_intrinsic_total(self):
        rw = Rewards(_base_config(intrinsic_reward_episode_cap=25.0))
        # 4 fresh cells × 10 frontier each = 40 gross, capped at 25.
        rewards = _walk_fresh_cells(rw, 4)
        self.assertAlmostEqual(sum(rewards), 25.0, places=4)
        self.assertAlmostEqual(rw._intrinsic_reward_paid, 25.0, places=4)
        bd = rw.get_episode_breakdown()
        # Gross frontier still logged in full; clawback makes the sum honest.
        self.assertAlmostEqual(bd["frontier"], 40.0, places=4)
        self.assertAlmostEqual(bd["intrinsic_capped"], -15.0, places=4)
        self.assertAlmostEqual(
            bd["frontier"] + bd["intrinsic_capped"], 25.0, places=4)

    def test_cap_resets_each_episode(self):
        rw = Rewards(_base_config(intrinsic_reward_episode_cap=25.0))
        _walk_fresh_cells(rw, 4)
        self.assertAlmostEqual(rw._intrinsic_reward_paid, 25.0, places=4)
        rw.start_new_episode()
        self.assertAlmostEqual(rw._intrinsic_reward_paid, 0.0, places=4)
        # Fresh budget next episode: a genuinely novel cell (not one walked
        # last episode — frontier counts now persist) pays full frontier again.
        r, _ = rw.calculate_reward(_env_vars(x=100), "")
        self.assertAlmostEqual(float(r), 10.0, places=4)

    def test_cap_disabled_pays_full(self):
        rw = Rewards(_base_config(intrinsic_reward_episode_cap=0))
        rewards = _walk_fresh_cells(rw, 4)
        self.assertAlmostEqual(sum(rewards), 40.0, places=4)
        bd = rw.get_episode_breakdown()
        self.assertAlmostEqual(bd["intrinsic_capped"], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
