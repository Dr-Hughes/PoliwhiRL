# -*- coding: utf-8 -*-
"""Coverage for the minor level-up reward and per-step reward rounding.

Pinned behaviours:

- ``level_up_reward`` pays per total-party-level gained, once per level.
- A flat party level pays nothing on subsequent steps.
- A party-size change reseeds the baseline and pays nothing (a freshly
  caught / received Pokémon's levels are not a windfall).
- ``reward_round_dp`` rounds the returned per-step reward to N decimals.
"""
import unittest
import numpy as np

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
        "frontier_novelty_bonus": 0,
        "battle_engagement_reward": 0,
        "damage_dealt_reward": 0,
        "battle_decay_coef": 0,
        "flag_progress_reward": 0,
        "whiteout_penalty": 0,
        "level_up_reward": 0,
        "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _env_vars(party_info=(1, 5, 20, 0), battle_type=0, enemy_hp=0,
              map_bank=24, map_num=7):
    return {
        "X": 4, "Y": 3, "map_num": map_num, "map_bank": map_bank,
        "room": 0, "warp_number": 0, "money": 0,
        "pokedex_seen": 0, "pokedex_owned": 0,
        "collision_down": 0, "collision_up": 0,
        "collision_left": 0, "collision_right": 0,
        "story_flags": _zero_flags(),
        "battle_type": battle_type, "johto_badges": 0, "player_state": 0,
        "key_items_count": 0, "game_hour": 0, "bgm_id": 0,
        "enemy_hp": enemy_hp, "enemy_max_hp": 20,
        "party_info": party_info,
        "script_active": False,
    }


class TestLevelUpReward(unittest.TestCase):
    def test_pays_per_level_gained(self):
        rw = Rewards(_base_config(level_up_reward=10))
        # First call seeds the baseline (no pay-out).
        r0, _ = rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0)), "")
        self.assertAlmostEqual(float(r0), 0.0, places=4)
        # +2 total levels → 2 × 10.
        r1, _ = rw.calculate_reward(_env_vars(party_info=(1, 7, 20, 0)), "")
        self.assertAlmostEqual(float(r1), 20.0, places=4)
        # Flat level → nothing.
        r2, _ = rw.calculate_reward(_env_vars(party_info=(1, 7, 20, 0)), "")
        self.assertAlmostEqual(float(r2), 0.0, places=4)

    def test_party_size_change_suppressed(self):
        rw = Rewards(_base_config(level_up_reward=10))
        rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0)), "")
        # Caught a Pokémon: size 1 → 2, total level jumps — must not pay.
        r1, _ = rw.calculate_reward(_env_vars(party_info=(2, 12, 40, 0)), "")
        self.assertAlmostEqual(float(r1), 0.0, places=4)
        # Subsequent genuine level gain at the new size pays normally.
        r2, _ = rw.calculate_reward(_env_vars(party_info=(2, 13, 40, 0)), "")
        self.assertAlmostEqual(float(r2), 10.0, places=4)

    def test_disabled_by_default(self):
        rw = Rewards(_base_config())  # level_up_reward = 0
        rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0)), "")
        r1, _ = rw.calculate_reward(_env_vars(party_info=(1, 9, 20, 0)), "")
        self.assertAlmostEqual(float(r1), 0.0, places=4)


class TestRewardRounding(unittest.TestCase):
    def test_rounds_to_configured_dp(self):
        # damage 1 × decay 1/(1+0.2·1) = 0.8333… → 0.83 at 2 dp.
        cfg = _base_config(
            damage_dealt_reward=1.0, battle_decay_coef=0.2, reward_round_dp=2,
        )
        rw = Rewards(cfg)
        rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=20), "")  # seed
        r1, _ = rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=19), "")
        self.assertAlmostEqual(float(r1), 0.83, places=6)

    def test_unrounded_when_disabled(self):
        cfg = _base_config(damage_dealt_reward=1.0, battle_decay_coef=0.2, reward_round_dp=None)
        rw = Rewards(cfg)
        rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=20), "")
        r1, _ = rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=19), "")
        self.assertAlmostEqual(float(r1), 1.0 / 1.2, places=5)


class TestMapGoal(unittest.TestCase):
    def _cfg(self):
        return _base_config(
            map_goal_reward=250,
            terminate_on_goal_complete=True,
            goals=[{"type": "map", "map_bank": 26, "map_num": 3}],
        )

    def test_fires_on_entering_target_map(self):
        rw = Rewards(self._cfg())
        # Start on New Bark (24, 4) — snapshots the start map, no fire.
        r0, done0 = rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        self.assertAlmostEqual(float(r0), 0.0, places=4)
        self.assertFalse(done0)
        # Walk onto Route 29 (24, ?) — still no fire.
        r1, _ = rw.calculate_reward(_env_vars(map_bank=24, map_num=12), "")
        self.assertAlmostEqual(float(r1), 0.0, places=4)
        # Enter Cherrygrove City (26, 3) — fires once and terminates.
        r2, done2 = rw.calculate_reward(_env_vars(map_bank=26, map_num=3), "")
        self.assertAlmostEqual(float(r2), 250.0, places=4)
        self.assertTrue(done2)
        self.assertEqual(rw.n_map_goals_completed(), 1)

    def test_does_not_fire_if_target_is_start_map(self):
        # If a replay leaves us standing on the target, it must not fire on
        # step 0 (mirrors flag semantics).
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


class TestPokedexGoalTermination(unittest.TestCase):
    """Regression: a pokedex-only stage must terminate when its threshold
    is met. check_pokedex_goals prunes completed entries, so the
    'configured?' guard in all_goal_thresholds_met must read the immutable
    spec, not the mutable (pruned) lists — otherwise it never terminates."""

    def test_pokedex_owned_goal_terminates(self):
        rw = Rewards(_base_config(
            pokedex_owned_reward=150,
            terminate_on_goal_complete=True,
            goals=[{"type": "pokedex", "kind": "owned", "threshold": 1}],
        ))
        ev0 = _env_vars(party_info=(0, 0, 0, 0)); ev0["pokedex_owned"] = 0; ev0["pokedex_seen"] = 0
        _, d0 = rw.calculate_reward(ev0, "")
        self.assertFalse(d0)
        ev1 = _env_vars(party_info=(1, 5, 20, 0)); ev1["pokedex_owned"] = 1; ev1["pokedex_seen"] = 1
        _, d1 = rw.calculate_reward(ev1, "")
        self.assertTrue(d1, "pokedex_owned>=1 goal should terminate the episode")
        self.assertTrue(rw.goals.all_goal_thresholds_met())


if __name__ == "__main__":
    unittest.main()
