# -*- coding: utf-8 -*-
"""Battle-progress watchdog, flee reward, and enemy-HP byte-order tests.

Pinned behaviours:

- The battle-progress watchdog truncates an episode after a threshold of
  consecutive battle steps with NO change in enemy or party HP, and does NOT
  truncate a battle that keeps making progress (HP changing).
- The counter is battle-gated: free-walking steps never advance it, and
  leaving battle resets it.
- The flee reward pays on a genuine wild-battle escape (enemy alive, party
  alive) and is withheld on a KO, a whiteout, or a trainer battle.
- RAMManagement reads enemy HP big-endian (0xD216 high, 0xD217 low), matching
  the emulator ground truth verified against the wild-battle save state.
"""
import unittest
import numpy as np

from PoliwhiRL.environment.rewards import Rewards
from PoliwhiRL.environment.RAM import RAMManagement


def _zero_flags():
    return np.zeros(256, dtype=np.uint8)


def _config(**overrides):
    cfg = {
        "episode_length": 100000,  # large so only the watchdog can truncate
        "new_map_reward": 0,
        "frontier_novelty_bonus": 0,
        "whiteout_penalty": 0,
        "step_penalty": 0.0,
        "reward_round_dp": None,
        "stagnation_truncation_steps": 0,  # disable the free-walk one here
        "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _ev(battle_type=0, enemy_hp=0, party_hp=20, party_size=1):
    return {
        "X": 4, "Y": 3, "map_num": 7, "map_bank": 24,
        "room": 0, "warp_number": 0, "money": 0,
        "pokedex_seen": 0, "pokedex_owned": 0,
        "collision_down": 0, "collision_up": 0,
        "collision_left": 0, "collision_right": 0,
        "story_flags": _zero_flags(),
        "battle_type": battle_type, "johto_badges": 0, "player_state": 0,
        "key_items_count": 0, "game_hour": 0, "bgm_id": 0,
        "enemy_hp": enemy_hp, "enemy_max_hp": 20,
        "party_info": (party_size, 5, party_hp, 0),
        "script_active": False,
    }


class TestBattleWatchdog(unittest.TestCase):
    def test_auto_limit_scales_with_episode_length(self):
        rw = Rewards(_config(episode_length=40960,
                             battle_stagnation_truncation_steps="auto"))
        self.assertEqual(rw._battle_stagnation_limit, 40960 // 16)

    def test_stuck_battle_truncates(self):
        limit = 10
        rw = Rewards(_config(battle_stagnation_truncation_steps=limit))
        rw.start_new_episode()
        done = False
        # Enter battle then hold everything static (no HP change).
        for i in range(limit + 5):
            _, done = rw.calculate_reward(_ev(battle_type=1, enemy_hp=14), "a")
            if done:
                break
        self.assertTrue(done)
        self.assertTrue(rw.truncated)
        self.assertGreaterEqual(rw._battle_stagnation_steps, limit)

    def test_progressing_battle_not_truncated(self):
        limit = 10
        rw = Rewards(_config(battle_stagnation_truncation_steps=limit))
        rw.start_new_episode()
        # Enemy HP ticks down every step: a real, progressing fight.
        hp = 30
        for _ in range(limit * 3):
            hp = max(0, hp - 1)
            _, done = rw.calculate_reward(_ev(battle_type=1, enemy_hp=hp), "a")
            self.assertFalse(done, "a progressing battle must never truncate")

    def test_party_damage_also_counts_as_progress(self):
        limit = 5
        rw = Rewards(_config(battle_stagnation_truncation_steps=limit))
        rw.start_new_episode()
        php = 20
        for _ in range(limit * 3):
            php = max(1, php - 1)  # we keep taking damage: still progressing
            _, done = rw.calculate_reward(
                _ev(battle_type=1, enemy_hp=14, party_hp=php), "a")
            self.assertFalse(done)

    def test_leaving_battle_resets_counter(self):
        limit = 10
        rw = Rewards(_config(battle_stagnation_truncation_steps=limit))
        rw.start_new_episode()
        for _ in range(limit - 1):
            rw.calculate_reward(_ev(battle_type=1, enemy_hp=14), "a")
        self.assertGreater(rw._battle_stagnation_steps, 0)
        rw.calculate_reward(_ev(battle_type=0), "down")  # battle ends
        self.assertEqual(rw._battle_stagnation_steps, 0)

    def test_free_walking_never_advances_battle_counter(self):
        rw = Rewards(_config(battle_stagnation_truncation_steps=5))
        rw.start_new_episode()
        for _ in range(50):
            _, done = rw.calculate_reward(_ev(battle_type=0), "up")
            self.assertFalse(done)
        self.assertEqual(rw._battle_stagnation_steps, 0)

    def test_disabled_when_zero(self):
        rw = Rewards(_config(battle_stagnation_truncation_steps=0))
        rw.start_new_episode()
        for _ in range(200):
            _, done = rw.calculate_reward(_ev(battle_type=1, enemy_hp=14), "a")
            self.assertFalse(done)


class TestFleeReward(unittest.TestCase):
    def _run_exit(self, prev_bt, enemy_hp_in_battle, party_hp_on_exit,
                  flee_reward=1.0):
        rw = Rewards(_config(battle_flee_reward=flee_reward,
                             battle_engagement_reward=0.0,
                             battle_win_reward=0.0))
        rw.start_new_episode()
        # One in-battle step to seed _prev_battle_type / _prev_enemy_hp.
        rw.calculate_reward(_ev(battle_type=prev_bt, enemy_hp=enemy_hp_in_battle),
                            "a")
        before = rw.get_episode_breakdown()["battle"]
        rw.calculate_reward(_ev(battle_type=0, party_hp=party_hp_on_exit), "b")
        return rw.get_episode_breakdown()["battle"] - before

    def test_flee_pays_on_wild_escape(self):
        self.assertAlmostEqual(
            self._run_exit(prev_bt=1, enemy_hp_in_battle=14, party_hp_on_exit=20),
            1.0)

    def test_no_flee_on_ko(self):
        # Enemy at 0 HP on the last in-battle step = a KO, not an escape.
        self.assertAlmostEqual(
            self._run_exit(prev_bt=1, enemy_hp_in_battle=0, party_hp_on_exit=20),
            0.0)

    def test_no_flee_on_whiteout(self):
        self.assertAlmostEqual(
            self._run_exit(prev_bt=1, enemy_hp_in_battle=14, party_hp_on_exit=0),
            0.0)

    def test_no_flee_from_trainer_battle(self):
        self.assertAlmostEqual(
            self._run_exit(prev_bt=2, enemy_hp_in_battle=14, party_hp_on_exit=20),
            0.0)

    def test_off_by_default(self):
        self.assertAlmostEqual(
            self._run_exit(prev_bt=1, enemy_hp_in_battle=14, party_hp_on_exit=20,
                           flee_reward=0.0),
            0.0)


class _FakePyBoy:
    def __init__(self, mem):
        self.memory = mem


class TestEnemyHPByteOrder(unittest.TestCase):
    def _ram(self, d216, d217, d218, d219):
        mem = {0xD216: d216, 0xD217: d217, 0xD218: d218, 0xD219: d219}
        return RAMManagement(_FakePyBoy(mem))

    def test_big_endian_current_hp(self):
        # Ground truth from in_wild_battle_route_13.state: L2 Sentret, 14 HP,
        # stored 0xD216=0 (high), 0xD217=14 (low).
        ram = self._ram(0, 14, 0, 14)
        self.assertEqual(ram.get_enemy_hp(), 14)
        self.assertEqual(ram.get_enemy_max_hp(), 14)

    def test_high_byte_contributes_256(self):
        ram = self._ram(1, 44, 1, 44)  # 0x012C = 300
        self.assertEqual(ram.get_enemy_hp(), 300)
        self.assertEqual(ram.get_enemy_max_hp(), 300)

    def test_zero_is_zero(self):
        ram = self._ram(0, 0, 0, 20)
        self.assertEqual(ram.get_enemy_hp(), 0)


if __name__ == "__main__":
    unittest.main()
