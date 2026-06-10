# -*- coding: utf-8 -*-
"""Whiteout one-shot penalty + battle entry/damage coverage for the
reward calculator.

Pinned behaviours:

- ``prev_total_hp > 0 → cur_total_hp == 0`` transition pays
  ``whiteout_penalty`` exactly once and increments ``whiteouts``.
- HP staying at 0 across subsequent steps does not re-fire.
- Party-size change resets the HP baseline (prevents catching/joining
  a Pokémon from looking like an HP loss).
- ``battle_engagement_reward`` fires once on the 0 → non-zero
  ``battle_type`` transition; subsequent steps inside the same battle
  pay zero entry bonus.
- ``damage_dealt_reward`` pays ``coef × Δenemy_hp × decay`` per step
  while enemy HP drops.
- Decay is **per-map**: ``1/(1 + battle_decay_coef · n)`` where ``n``
  is the count of battles already entered on the current
  ``(map_bank, map_num)`` this episode. Walking to a fresh map gives
  a fresh budget; re-entering a map already fought on preserves its
  counter (no oscillation exploit).
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
        # Disable decay by default for tests that don't exercise it,
        # so we can assert on raw reward values without arithmetic.
        # Tests that target decay opt back in.
        "battle_decay_coef": 0,
        "flag_progress_reward": 0,
        "whiteout_penalty": -100,
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

    def test_whiteout_does_not_terminate_unless_goal_complete(self):
        """Whiteout itself doesn't set ``done`` — only the episode budget
        or ``terminate_on_goal_complete`` does."""
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
        # Now drop HP by 5 — no per-step penalty in the new design.
        r2, _ = rw.calculate_reward(_env_vars(party_info=(2, 10, 30, 0)), "")
        self.assertAlmostEqual(float(r2), 0.0, places=4)
        self.assertEqual(rw.whiteouts, 0)

    def test_zero_party_size_does_not_fire(self):
        """Pre-starter the party is empty; we must not score that as HP=0."""
        rw = Rewards(_base_config())
        r0, _ = rw.calculate_reward(_env_vars(party_info=(0, 0, 0, 0)), "")
        self.assertAlmostEqual(float(r0), 0.0, places=4)
        r1, _ = rw.calculate_reward(_env_vars(party_info=(0, 0, 0, 0)), "")
        self.assertAlmostEqual(float(r1), 0.0, places=4)
        self.assertEqual(rw.whiteouts, 0)


class TestBattleEngagement(unittest.TestCase):
    def _battle_vars(self, enemy_hp, battle_type=1, party_info=(1, 5, 20, 0),
                     map_bank=24, map_num=7):
        return _env_vars(
            party_info=party_info, battle_type=battle_type, enemy_hp=enemy_hp,
            map_bank=map_bank, map_num=map_num,
        )

    def _exit_battle(self, party_hp=20, map_bank=24, map_num=7):
        return _env_vars(
            party_info=(1, 5, party_hp, 0), battle_type=0,
            map_bank=map_bank, map_num=map_num,
        )

    def test_battles_counter_increments_on_entry(self):
        rw = Rewards(_base_config(battle_engagement_reward=5.0))
        rw.calculate_reward(self._battle_vars(20), "a")
        self.assertEqual(rw.battles_this_episode, 1)
        # Stay in battle — counter does not re-increment.
        rw.calculate_reward(self._battle_vars(20), "a")
        self.assertEqual(rw.battles_this_episode, 1)
        # Exit battle, re-enter.
        rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0)), "")
        rw.calculate_reward(self._battle_vars(20), "a")
        self.assertEqual(rw.battles_this_episode, 2)

    def test_entry_bonus_is_first_per_map_only(self):
        """Entry bonus pays once on the FIRST battle of each map this
        episode; subsequent battles on the same map pay 0 entry (anti-farm).
        A different map pays again."""
        rw = Rewards(_base_config(
            battle_engagement_reward=5.0, damage_dealt_reward=0,
            battle_decay_coef=0, battle_reward_episode_cap=0,  # cap off
        ))
        # First battle on map 7 → pays 5.
        r, _ = rw.calculate_reward(self._battle_vars(20, map_num=7), "a")
        self.assertAlmostEqual(float(r), 5.0, places=4)
        rw.calculate_reward(self._exit_battle(map_num=7), "")
        # Second battle on map 7 → entry 0.
        r, _ = rw.calculate_reward(self._battle_vars(20, map_num=7), "a")
        self.assertAlmostEqual(float(r), 0.0, places=4)
        rw.calculate_reward(self._exit_battle(map_num=7), "")
        # First battle on a fresh map 8 → pays 5 again.
        r, _ = rw.calculate_reward(self._battle_vars(20, map_num=8), "a")
        self.assertAlmostEqual(float(r), 5.0, places=4)

    def test_damage_off_by_default(self):
        """damage_dealt_reward defaults to 0 — no reward as enemy HP drops."""
        rw = Rewards(_base_config(battle_engagement_reward=0))
        rw.calculate_reward(self._battle_vars(20), "a")
        r, _ = rw.calculate_reward(self._battle_vars(10), "a")
        self.assertAlmostEqual(float(r), 0.0, places=4)


class TestBattleWin(unittest.TestCase):
    """Winning (enemy HP→0, player survives) pays battle_win_reward once per
    map; fleeing / not finishing pays nothing."""

    def _battle_vars(self, enemy_hp, map_num=7):
        return _env_vars(battle_type=1, enemy_hp=enemy_hp, map_num=map_num)

    def test_win_pays_once_per_map(self):
        rw = Rewards(_base_config(
            battle_engagement_reward=0, battle_win_reward=8.0,
            battle_decay_coef=0, battle_reward_episode_cap=0,
        ))
        # Enter, knock enemy to 0, exit alive → win.
        rw.calculate_reward(self._battle_vars(20), "a")
        rw.calculate_reward(self._battle_vars(0), "a")
        r, _ = rw.calculate_reward(
            _env_vars(party_info=(1, 5, 18, 0), battle_type=0, map_num=7), "")
        self.assertAlmostEqual(float(r), 8.0, places=4)
        # A second won battle on the SAME map pays no win bonus.
        rw.calculate_reward(self._battle_vars(20, map_num=7), "a")
        rw.calculate_reward(self._battle_vars(0, map_num=7), "a")
        r, _ = rw.calculate_reward(
            _env_vars(party_info=(1, 5, 18, 0), battle_type=0, map_num=7), "")
        self.assertAlmostEqual(float(r), 0.0, places=4)

    def test_no_win_if_enemy_not_defeated(self):
        """Exiting a battle without zeroing enemy HP (fled) pays no win."""
        rw = Rewards(_base_config(
            battle_engagement_reward=0, battle_win_reward=8.0,
            battle_decay_coef=0,
        ))
        rw.calculate_reward(self._battle_vars(20), "a")
        rw.calculate_reward(self._battle_vars(10), "a")  # enemy never hits 0
        r, _ = rw.calculate_reward(
            _env_vars(party_info=(1, 5, 18, 0), battle_type=0), "")
        self.assertAlmostEqual(float(r), 0.0, places=4)

    def test_no_win_on_whiteout(self):
        """Enemy at 0 but player also at 0 HP at exit → not a win."""
        rw = Rewards(_base_config(
            battle_engagement_reward=0, battle_win_reward=8.0,
            battle_decay_coef=0, whiteout_penalty=0,
        ))
        rw.calculate_reward(self._battle_vars(20), "a")
        rw.calculate_reward(self._battle_vars(0), "a")
        r, _ = rw.calculate_reward(
            _env_vars(party_info=(1, 5, 0, 0), battle_type=0), "")
        self.assertAlmostEqual(float(r), 0.0, places=4)


class TestBattleRewardCap(unittest.TestCase):
    def test_cap_limits_total_battle_reward(self):
        """Total battle reward across an episode cannot exceed the cap even
        with damage farming enabled."""
        rw = Rewards(_base_config(
            battle_engagement_reward=0, damage_dealt_reward=1.0,
            battle_decay_coef=0, battle_reward_episode_cap=15.0,
        ))
        paid = 0.0
        # Farm 5 battles of 10 damage each = 50 raw, capped at 15.
        for _ in range(5):
            rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=20), "a")
            r, _ = rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=10), "a")
            paid += float(r)
            rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0), battle_type=0), "")
        self.assertAlmostEqual(paid, 15.0, places=4)


class TestPerMapBattleDecay(unittest.TestCase):
    """Decay still applies to the (first-per-map) entry and win bonuses."""

    def _battle_vars(self, enemy_hp, battle_type=1, map_bank=24, map_num=7):
        return _env_vars(
            battle_type=battle_type, enemy_hp=enemy_hp,
            map_bank=map_bank, map_num=map_num,
        )

    def _exit_battle(self, map_bank=24, map_num=7):
        return _env_vars(
            party_info=(1, 5, 20, 0), battle_type=0,
            map_bank=map_bank, map_num=map_num,
        )

    def test_first_entry_decays_by_map_count(self):
        """The first entry on a map decays by that map's run count. Two
        separate maps each pay their n=1 rate (5/1.1)."""
        rw = Rewards(_base_config(
            battle_engagement_reward=5.0, damage_dealt_reward=0,
            battle_decay_coef=0.1, battle_reward_episode_cap=0,
        ))
        r, _ = rw.calculate_reward(self._battle_vars(20, map_num=7), "a")
        self.assertAlmostEqual(float(r), 5.0 / 1.1, places=4)
        rw.calculate_reward(self._exit_battle(map_num=7), "")
        r, _ = rw.calculate_reward(self._battle_vars(20, map_num=8), "a")
        self.assertAlmostEqual(float(r), 5.0 / 1.1, places=4)

    def test_episode_reset_clears_per_map_counter(self):
        rw = Rewards(_base_config(
            battle_engagement_reward=5.0, damage_dealt_reward=0,
            battle_decay_coef=0.1, battle_reward_episode_cap=0,
        ))
        rw.calculate_reward(self._battle_vars(20), "a")
        rw.calculate_reward(self._exit_battle(), "")
        rw.start_new_episode()
        # First battle of the new episode → n=1 → decay 1/1.1.
        r, _ = rw.calculate_reward(self._battle_vars(20), "a")
        self.assertAlmostEqual(float(r), 5.0 / 1.1, places=4)


class TestProgressMetrics(unittest.TestCase):
    def test_whiteouts_and_battles_in_progress(self):
        rw = Rewards(_base_config(battle_engagement_reward=5.0))
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0), battle_type=1), "a")
        rw.calculate_reward(_env_vars(party_info=(1, 5, 0, 0), battle_type=1), "a")
        progress = rw.get_progress()
        self.assertEqual(progress["Whiteouts"], 1)
        self.assertEqual(progress["Battles Entered"], 1)

    def test_episode_reset_clears_counters(self):
        rw = Rewards(_base_config(battle_engagement_reward=5.0))
        rw.calculate_reward(_env_vars(party_info=(1, 5, 5, 0), battle_type=1), "a")
        rw.calculate_reward(_env_vars(party_info=(1, 5, 0, 0), battle_type=1), "a")
        rw.start_new_episode()
        self.assertEqual(rw.whiteouts, 0)
        self.assertEqual(rw.battles_this_episode, 0)


if __name__ == "__main__":
    unittest.main()
