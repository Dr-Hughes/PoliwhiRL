# -*- coding: utf-8 -*-
"""Episode breakdown and reward stream sanity tests.

Pinned behaviours:

- get_episode_breakdown() returns one key per reward source (milestones,
  battle, level, frontier, new_map, step_penalty, whiteout).
- A valid step with no novelty, no whiteout, and step_penalty=0 contributes
  0.0 to all breakdown entries.
- A valid step with step_penalty=-0.3 accumulates the penalty each step.
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
        "new_bank_reward": 0,
        "frontier_novelty_bonus": 0,
        "whiteout_penalty": 0,
        "step_penalty": 0.0,
        "reward_round_dp": None,
        "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _env_vars(x=4, y=3, map_bank=24, map_num=7):
    return {
        "X": x,
        "Y": y,
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
        "battle_type": 0,
        "johto_badges": 0,
        "player_state": 0,
        "key_items_count": 0,
        "game_hour": 0,
        "bgm_id": 0,
        "enemy_hp": 0,
        "enemy_max_hp": 20,
        "party_info": (1, 5, 20, 0),
        "script_active": False,
    }


class TestEpisodeBreakdown(unittest.TestCase):
    def test_breakdown_has_expected_keys(self):
        rw = Rewards(_base_config())
        bd = rw.get_episode_breakdown()
        self.assertEqual(
            set(bd.keys()),
            {
                "flag",
                "checkpoint_honest",
                "checkpoint_seeded",
                "map_goal",
                "maps_visited",
                "pokedex",
                "key_item",
                "battle",
                "first_trainer_battle",
                "level",
                "frontier",
                "menu",
                "new_map",
                "event",
                "step_penalty",
                "whiteout",
                "rounding_adjustment",
                "clip_adjustment",
                "delivery_cast_adjustment",
                "delivered_total",
                "domain_scale_adjustment",
            },
        )

    def test_zero_reward_step_contributes_nothing(self):
        """All signals off (including step_penalty=0): breakdown stays 0 after a valid step."""
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
        self.assertAlmostEqual(bd["step_penalty"], 0.0, places=4)

    def test_step_penalty_accumulates(self):
        """A negative step_penalty is added every valid step."""
        rw = Rewards(_base_config(step_penalty=-0.3))
        for _ in range(5):
            rw.calculate_reward(_env_vars(x=4, y=3), "")
        bd = rw.get_episode_breakdown()
        self.assertAlmostEqual(bd["step_penalty"], -0.3 * 5, places=4)

    def test_step_penalty_in_total_reward(self):
        """step_penalty is reflected in the returned per-step reward."""
        rw = Rewards(_base_config(step_penalty=-0.3, reward_round_dp=None))
        # Move to a new cell so frontier=0 and new_map=0 (bonus disabled).
        r, _ = rw.calculate_reward(_env_vars(x=4, y=3), "")
        # Only step_penalty contributes (frontier_novelty_bonus=0).
        self.assertAlmostEqual(float(r), -0.3, places=4)

    def test_cumulative_reward_increments(self):
        rw = Rewards(_base_config(frontier_novelty_bonus=10.0, reward_round_dp=None))
        rw.calculate_reward(_env_vars(x=4, y=3), "")
        rw.calculate_reward(_env_vars(x=6, y=3), "")
        # Both cells pay 10.0 (fresh run, no archive hits).
        self.assertAlmostEqual(rw.cumulative_reward, 20.0, places=4)

    def test_unrounded_negative_half_cent_is_conserved(self):
        rw = Rewards(_base_config(step_penalty=-0.005, reward_round_dp=None))
        returned = [float(rw.calculate_reward(_env_vars(), "")[0]) for _ in range(3)]
        bd = rw.get_episode_breakdown()
        self.assertAlmostEqual(bd["delivered_total"], sum(returned), places=12)
        self.assertAlmostEqual(rw.cumulative_reward, sum(returned), places=12)
        self.assertAlmostEqual(bd["rounding_adjustment"], 0.0, places=12)
        self.assertAlmostEqual(bd["clip_adjustment"], 0.0, places=12)
        raw_total = bd["step_penalty"]
        accounted = (
            raw_total
            + bd["rounding_adjustment"]
            + bd["clip_adjustment"]
            + bd["delivery_cast_adjustment"]
        )
        self.assertAlmostEqual(accounted, bd["delivered_total"], places=12)
        self.assertAlmostEqual(bd["step_penalty"], -0.015, places=12)

    def test_rounding_and_clipping_adjustments_are_explicit(self):
        rounded = Rewards(_base_config(step_penalty=-0.005, reward_round_dp=2))
        rounded_reward, _ = rounded.calculate_reward(_env_vars(), "")
        rounded_bd = rounded.get_episode_breakdown()
        self.assertAlmostEqual(float(rounded_reward), -0.01, places=7)
        self.assertAlmostEqual(rounded_bd["rounding_adjustment"], -0.005, places=12)
        self.assertAlmostEqual(rounded_bd["clip_adjustment"], 0.0, places=12)

        clipped = Rewards(
            _base_config(
                frontier_novelty_bonus=20.0,
                reward_clip=5.0,
                reward_round_dp=None,
            )
        )
        clipped_reward, _ = clipped.calculate_reward(_env_vars(), "")
        clipped_bd = clipped.get_episode_breakdown()
        self.assertAlmostEqual(float(clipped_reward), 5.0, places=12)
        self.assertAlmostEqual(clipped_bd["clip_adjustment"], -15.0, places=12)
        self.assertAlmostEqual(clipped_bd["delivered_total"], 5.0, places=12)
        self.assertAlmostEqual(clipped.cumulative_reward, 5.0, places=12)


if __name__ == "__main__":
    unittest.main()
