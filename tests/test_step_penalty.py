# -*- coding: utf-8 -*-
"""Per-step living cost (step_penalty).

Pinned behaviours:

- A negative ``step_penalty`` is paid every valid step and accumulates in the
  ``step`` breakdown entry.
- It lands in the EXTRINSIC stream (``_last_extrinsic``), not intrinsic, so the
  two-stream scaler treats it as a directed signal.
- Default (0.0) is a no-op.
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
        "frontier_novelty_bonus": 0,
        "battle_engagement_reward": 0,
        "battle_win_reward": 0,
        "damage_dealt_reward": 0,
        "battle_decay_coef": 0,
        "level_up_reward": 0,
        "flag_progress_reward": 0,
        "whiteout_penalty": 0,
        "intrinsic_reward_episode_cap": 0,
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


class TestStepPenalty(unittest.TestCase):
    def test_applied_every_step(self):
        rw = Rewards(_base_config(step_penalty=-0.5))
        rewards = [float(rw.calculate_reward(_env_vars(x=2 * i), "")[0])
                   for i in range(3)]
        for r in rewards:
            self.assertAlmostEqual(r, -0.5, places=4)
        self.assertAlmostEqual(rw.get_episode_breakdown()["step"], -1.5, places=4)

    def test_lives_in_extrinsic_stream(self):
        rw = Rewards(_base_config(step_penalty=-0.5))
        rw.calculate_reward(_env_vars(), "")
        self.assertAlmostEqual(rw._last_extrinsic, -0.5, places=4)
        self.assertAlmostEqual(rw._last_intrinsic, 0.0, places=4)

    def test_default_is_noop(self):
        rw = Rewards(_base_config())  # step_penalty unset -> 0.0
        r, _ = rw.calculate_reward(_env_vars(), "")
        self.assertAlmostEqual(float(r), 0.0, places=4)
        self.assertAlmostEqual(rw.get_episode_breakdown()["step"], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
