# -*- coding: utf-8 -*-
"""Whiteout one-shot penalty for the reward calculator.

Pinned behaviours:

- ``prev_total_hp > 0 → cur_total_hp == 0`` transition pays
  ``whiteout_penalty`` exactly once and increments ``whiteouts``.
- HP staying at 0 across subsequent steps does not re-fire.
- Party-size change resets the HP baseline (prevents catching/joining
  a Pokémon from looking like an HP loss).
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
        "frontier_novelty_bonus": 0,
        "whiteout_penalty": -100,
        "step_penalty": 0.0,
        "reward_round_dp": None,
        "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _env_vars(party_info=(1, 5, 20, 0), battle_type=0, script_active=False,
              enemy_hp=0, map_bank=24, map_num=7):
    """Minimal env_vars dict. ``party_info`` is (size, level, hp, exp)."""
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
        "script_active": script_active,
    }


class TestWhiteout(unittest.TestCase):
    def test_whiteout_transition_fires_once(self):
        rw = Rewards(_base_config(whiteout_penalty=-100))
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0)), "")
        # 5 → 0 transition: -100 whiteout.
        r1, _ = rw.calculate_reward(_env_vars(party_info=(1, 5, 0, 0)), "")
        self.assertAlmostEqual(float(r1), -100.0, places=4)
        self.assertEqual(rw.whiteouts, 1)
        # Staying at HP=0: prev_hp is 0, whiteout does not re-fire.
        r2, _ = rw.calculate_reward(_env_vars(party_info=(1, 5, 0, 0)), "")
        self.assertAlmostEqual(float(r2), 0.0, places=4)
        self.assertEqual(rw.whiteouts, 1)

    def test_whiteout_does_not_terminate(self):
        """Whiteout itself does not set ``done`` — only the episode budget does."""
        rw = Rewards(_base_config(episode_length=100))
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0)), "")
        _, done = rw.calculate_reward(_env_vars(party_info=(1, 5, 0, 0)), "")
        self.assertFalse(done)

    def test_whiteout_blocked_by_script_active(self):
        """Mid-battle HP=0 reads under a script overlay shouldn't false-fire."""
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0)), "")
        r1, _ = rw.calculate_reward(
            _env_vars(party_info=(1, 5, 0, 0), script_active=True), ""
        )
        self.assertAlmostEqual(float(r1), 0.0, places=4)
        self.assertEqual(rw.whiteouts, 0)

    def test_whiteout_default_penalty(self):
        """Default penalty is -100 — a whiteout is a real hard-fail signal
        now that milestone rewards dominate the reward magnitude."""
        cfg = {"episode_length": 100, "goals": []}
        rw = Rewards(cfg)
        self.assertEqual(rw.whiteout_penalty, -100.0)


class TestPartySizeChange(unittest.TestCase):
    def test_party_size_change_resets_hp_baseline(self):
        """Catching a mon (party_size 1→2) should NOT trigger a whiteout
        false-positive — the HP baseline must reset on roster change."""
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0)), "")
        # Party size goes 1 → 2; HP changes from 20 to 35. No reward.
        r1, _ = rw.calculate_reward(_env_vars(party_info=(2, 10, 35, 0)), "")
        self.assertAlmostEqual(float(r1), 0.0, places=4)
        self.assertEqual(rw.whiteouts, 0)

    def test_zero_party_size_does_not_fire(self):
        """Pre-starter the party is empty; we must not score that as HP=0."""
        rw = Rewards(_base_config())
        r0, _ = rw.calculate_reward(_env_vars(party_info=(0, 0, 0, 0)), "")
        self.assertAlmostEqual(float(r0), 0.0, places=4)
        r1, _ = rw.calculate_reward(_env_vars(party_info=(0, 0, 0, 0)), "")
        self.assertAlmostEqual(float(r1), 0.0, places=4)
        self.assertEqual(rw.whiteouts, 0)


class TestProgressMetrics(unittest.TestCase):
    def test_whiteouts_in_progress(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0)), "")
        rw.calculate_reward(_env_vars(party_info=(1, 5, 0, 0)), "")
        progress = rw.get_progress()
        self.assertEqual(progress["Whiteouts"], 1)

    def test_episode_reset_clears_whiteouts(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0)), "")
        rw.calculate_reward(_env_vars(party_info=(1, 5, 0, 0)), "")
        rw.start_new_episode()
        self.assertEqual(rw.whiteouts, 0)


if __name__ == "__main__":
    unittest.main()
