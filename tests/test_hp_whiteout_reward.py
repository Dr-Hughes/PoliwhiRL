# -*- coding: utf-8 -*-
"""Whiteout one-shot penalty for the reward calculator.

Pinned behaviours:

- Party HP reaching 0 arms the whiteout; the step that has HP back on a
  DIFFERENT map (the warp home) pays ``whiteout_penalty`` exactly once and
  increments ``whiteouts`` — also when the faint happens under a battle
  script, which the old transition check never saw.
- HP back on the same map (a transient 0-HP read) never fires.
- Party-size change resets the baseline (prevents catching/joining a
  Pokémon from looking like an HP loss).
- The warp's heal pays no ``resource_potential`` refund; a real heal does.
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
        "new_bank_reward": 0,
        "frontier_novelty_bonus": 0,
        "whiteout_penalty": -100,
        "step_penalty": 0.0,
        "reward_round_dp": None,
        "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _env_vars(
    party_info=(1, 5, 20, 0),
    battle_type=0,
    script_active=False,
    enemy_hp=0,
    map_bank=24,
    map_num=7,
    party_resources=None,
    party_levels=None,
    x=4,
    y=3,
    player_direction=1,
):
    """Minimal env_vars dict. ``party_info`` is (size, level, hp, exp)."""
    return {
        "X": x,
        "Y": y,
        "player_direction": player_direction,
        "party_resources": party_resources,
        "party_levels": party_levels if party_levels is not None else (5,) * party_info[0],
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
        "battle_type": battle_type,
        "johto_badges": 0,
        "player_state": 0,
        "key_items_count": 0,
        "game_hour": 0,
        "bgm_id": 0,
        "enemy_hp": enemy_hp,
        "enemy_max_hp": 20,
        "party_info": party_info,
        "script_active": script_active,
    }


class TestWhiteout(unittest.TestCase):
    def _faint_then_warp(self, rw, script_active=False):
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0)), "")
        r_faint, _ = rw.calculate_reward(
            _env_vars(party_info=(1, 5, 0, 0), script_active=script_active), ""
        )
        r_text, _ = rw.calculate_reward(_env_vars(party_info=(1, 5, 0, 0), script_active=True), "")
        r_warp, _ = rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0), map_num=8), "")
        return r_faint, r_text, r_warp

    def test_whiteout_pays_once_on_the_warp(self):
        rw = Rewards(_base_config(whiteout_penalty=-100))
        r_faint, r_text, r_warp = self._faint_then_warp(rw)
        self.assertAlmostEqual(float(r_faint), 0.0, places=4)
        self.assertAlmostEqual(float(r_text), 0.0, places=4)
        self.assertAlmostEqual(float(r_warp), -100.0, places=4)
        self.assertEqual(rw.whiteouts, 1)
        r_after, _ = rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0), map_num=8), "")
        self.assertAlmostEqual(float(r_after), 0.0, places=4)
        self.assertEqual(rw.whiteouts, 1)

    def test_faint_under_battle_script_still_pays(self):
        """The faint lands under a battle script; the old check missed it."""
        rw = Rewards(_base_config(whiteout_penalty=-100))
        _, _, r_warp = self._faint_then_warp(rw, script_active=True)
        self.assertAlmostEqual(float(r_warp), -100.0, places=4)
        self.assertEqual(rw.whiteouts, 1)

    def test_heal_before_warp_still_pays(self):
        """The heal may land a step before the map changes."""
        rw = Rewards(_base_config(whiteout_penalty=-100))
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0)), "")
        rw.calculate_reward(_env_vars(party_info=(1, 5, 0, 0), script_active=True), "")
        r_heal, _ = rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0), script_active=True), "")
        r_warp, _ = rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0), map_num=8), "")
        self.assertAlmostEqual(float(r_heal), 0.0, places=4)
        self.assertAlmostEqual(float(r_warp), -100.0, places=4)

    def test_transient_zero_on_same_map_does_not_fire(self):
        """A garbage 0-HP read that recovers on the same map is not a whiteout."""
        rw = Rewards(_base_config(whiteout_penalty=-100))
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0)), "")
        rw.calculate_reward(_env_vars(party_info=(1, 5, 0, 0)), "")
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0)), "")
        r, _ = rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0), map_num=8), "")
        self.assertAlmostEqual(float(r), 0.0, places=4)
        self.assertEqual(rw.whiteouts, 0)

    def test_whiteout_does_not_terminate(self):
        """Whiteout itself does not set ``done`` — only the episode budget does."""
        rw = Rewards(_base_config(episode_length=100))
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0)), "")
        rw.calculate_reward(_env_vars(party_info=(1, 5, 0, 0)), "")
        _, done = rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0), map_num=8), "")
        self.assertFalse(done)

    def test_whiteout_default_penalty(self):
        """Default penalty is -100 — a whiteout is a real hard-fail signal
        now that milestone rewards dominate the reward magnitude."""
        cfg = {"episode_length": 100, "goals": []}
        rw = Rewards(cfg)
        self.assertEqual(rw.whiteout_penalty, -100.0)


class TestWhiteoutResourceRefund(unittest.TestCase):
    """The whiteout warp's heal earns no resource_potential; a nurse heal does."""

    def _cfg(self):
        return _base_config(whiteout_penalty=0, resource_potential_coef=10.0)

    def test_warp_heal_pays_no_refund(self):
        rw = Rewards(self._cfg())
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0), party_resources=(0.25, 0.5)), "")
        rw.calculate_reward(_env_vars(party_info=(1, 5, 0, 0), party_resources=(0.0, 0.5)), "")
        r_warp, _ = rw.calculate_reward(
            _env_vars(party_info=(1, 5, 20, 0), party_resources=(1.0, 1.0), map_num=8), ""
        )
        self.assertAlmostEqual(float(r_warp), 0.0, places=4)
        # the potential lost while fainting stands as the cost
        self.assertLess(rw._episode_breakdown["resource_potential"], 0.0)

    def test_real_heal_pays_and_counts(self):
        rw = Rewards(self._cfg())
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0), party_resources=(0.25, 0.5)), "")
        r_heal, _ = rw.calculate_reward(
            _env_vars(party_info=(1, 5, 20, 0), party_resources=(1.0, 1.0)), ""
        )
        self.assertGreater(float(r_heal), 0.0)
        self.assertEqual(rw.get_episode_event_counts()["heals"], 1)
        self.assertEqual(rw.get_episode_event_counts()["whiteouts"], 0)


class TestEpisodeDiagnostics(unittest.TestCase):
    def test_wall_bump_counts_only_a_press_that_stays_put(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=4, y=3, player_direction=1), "")
        # facing up, press up, same tile: a bump
        rw.calculate_reward(_env_vars(x=4, y=3, player_direction=1), "up")
        # press left while facing up: only turns
        rw.calculate_reward(_env_vars(x=4, y=3, player_direction=3), "left")
        # press left again and move: a walk
        rw.calculate_reward(_env_vars(x=3, y=3, player_direction=3), "left")
        counts = rw.get_episode_event_counts()
        self.assertEqual(counts["wall_bumps"], 1)
        self.assertEqual(counts["walk_presses"], 3)

    def test_no_bump_under_script_or_in_battle(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(player_direction=1), "")
        rw.calculate_reward(_env_vars(player_direction=1, script_active=True), "up")
        rw.calculate_reward(_env_vars(player_direction=1, battle_type=1), "up")
        self.assertEqual(rw.get_episode_event_counts()["wall_bumps"], 0)

    def test_level_gains_split_lead_and_nonlead(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(party_info=(2, 20, 20, 0), party_levels=(15, 5)), "")
        rw.calculate_reward(_env_vars(party_info=(2, 21, 20, 0), party_levels=(16, 5)), "")
        rw.calculate_reward(_env_vars(party_info=(2, 23, 20, 0), party_levels=(16, 7)), "")
        # a reorder moves levels between slots and is not a level-up
        rw.calculate_reward(_env_vars(party_info=(2, 23, 20, 0), party_levels=(7, 16)), "")
        counts = rw.get_episode_event_counts()
        self.assertEqual(counts["lead_level_gains"], 1)
        self.assertEqual(counts["nonlead_level_gains"], 2)
        self.assertEqual(counts["top3_levels"], [16, 7])

    def test_counters_reset_each_episode(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(player_direction=1), "")
        rw.calculate_reward(_env_vars(player_direction=1), "up")
        rw.start_new_episode()
        self.assertEqual(rw.get_episode_event_counts()["wall_bumps"], 0)


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
    def _whiteout(self, rw):
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0)), "")
        rw.calculate_reward(_env_vars(party_info=(1, 5, 0, 0)), "")
        rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0), map_num=8), "")

    def test_whiteouts_in_progress(self):
        rw = Rewards(_base_config())
        self._whiteout(rw)
        self.assertEqual(rw.get_progress()["Whiteouts"], 1)

    def test_episode_reset_clears_whiteouts(self):
        rw = Rewards(_base_config())
        self._whiteout(rw)
        rw.start_new_episode()
        self.assertEqual(rw.whiteouts, 0)


if __name__ == "__main__":
    unittest.main()
