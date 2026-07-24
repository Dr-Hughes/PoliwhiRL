# -*- coding: utf-8 -*-
"""Battle-progress, level-up, pokedex, and terminate-on-goal-complete
reward wiring — added when milestone/battle rewards were reconnected to
the reward function (previously tracked for metrics only).

Pinned behaviours:

- Battle engagement pays once per fresh 0->nonzero battle_type transition,
  first-per-map-per-episode, decaying on repeat engagements on the same map.
- Battle win pays once per enemy_hp>0->0 transition while still in battle
  (a KO), same first-per-map-per-episode decay.
- Combined battle reward this episode is clamped to battle_reward_episode_cap.
- level_up_reward pays per total party level gained (suppressed across a
  party-size change).
- pokedex_owned_reward / pokedex_first_sight_reward pay independently per
  owned/seen increment.
- terminate_on_goal_complete, when configured true, ends the episode
  (done=True, not truncated) the step every goal is satisfied; false
  (default) lets the episode run to budget regardless.

- Flag/pokedex/level/key-item reward is ALWAYS-ON: it fires for ANY curated
  flag transition / new species / level gain / new key item even when the
  stage's `goals` list is completely empty — reward no longer requires a
  human to have anticipated and configured that specific milestone.
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
        "whiteout_penalty": 0,
        "step_penalty": 0.0,
        "flag_progress_reward": 0,
        "map_goal_reward": 0,
        "pokedex_owned_reward": 0,
        "pokedex_first_sight_reward": 0,
        "level_up_reward": 0,
        "battle_engagement_reward": 0.0,
        "battle_win_reward": 0.0,
        "reward_round_dp": None,
        "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _env_vars(
    x=4,
    y=3,
    map_bank=24,
    map_num=7,
    battle_type=0,
    enemy_hp=0,
    enemy_max_hp=100,
    party_info=(1, 5, 20, 0),
    pokedex_seen=0,
    pokedex_owned=0,
    story_flags=None,
    key_items_count=0,
):
    return {
        "X": x,
        "Y": y,
        "map_num": map_num,
        "map_bank": map_bank,
        "room": 0,
        "warp_number": 0,
        "money": 0,
        "pokedex_seen": pokedex_seen,
        "pokedex_owned": pokedex_owned,
        "collision_down": 0,
        "collision_up": 0,
        "collision_left": 0,
        "collision_right": 0,
        "story_flags": story_flags if story_flags is not None else _zero_flags(),
        "battle_type": battle_type,
        "johto_badges": 0,
        "player_state": 0,
        "key_items_count": key_items_count,
        "game_hour": 0,
        "bgm_id": 0,
        "enemy_hp": enemy_hp,
        "enemy_max_hp": enemy_max_hp,
        "party_info": party_info,
        "script_active": False,
    }


def _set_flag(arr, flag_num):
    arr = arr.copy()
    byte_idx, bit_idx = flag_num // 8, flag_num % 8
    arr[byte_idx] |= 1 << bit_idx
    return arr


class TestBattleEngagement(unittest.TestCase):
    def test_fires_on_entering_battle(self):
        rw = Rewards(_base_config(battle_engagement_reward=3.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        r, _ = rw.calculate_reward(_env_vars(battle_type=1), "")
        self.assertAlmostEqual(float(r), 3.0, places=4)

    def test_does_not_refire_while_still_in_battle(self):
        rw = Rewards(_base_config(battle_engagement_reward=3.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=1), "")
        r, _ = rw.calculate_reward(_env_vars(battle_type=1), "")
        self.assertAlmostEqual(float(r), 0.0, places=4)

    def test_decays_on_second_battle_same_map(self):
        rw = Rewards(_base_config(battle_engagement_reward=3.0, battle_decay_coef=0.2))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=1), "")  # engagement 1: pays 3.0
        rw.calculate_reward(_env_vars(battle_type=0), "")  # battle ends
        r, _ = rw.calculate_reward(_env_vars(battle_type=1), "")  # engagement 2
        self.assertAlmostEqual(float(r), 3.0 / 1.2, places=4)


class TestBattleWin(unittest.TestCase):
    def test_fires_on_enemy_ko(self):
        rw = Rewards(_base_config(battle_win_reward=8.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=50), "")
        r, _ = rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=0), "")
        self.assertAlmostEqual(float(r), 8.0, places=4)

    def test_does_not_fire_on_whiteout_or_flee(self):
        """Leaving battle (battle_type -> 0) without an enemy_hp>0->0
        transition while still IN battle must not pay a win."""
        rw = Rewards(_base_config(battle_win_reward=8.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=50), "")
        r, _ = rw.calculate_reward(_env_vars(battle_type=0, enemy_hp=0), "")
        self.assertAlmostEqual(float(r), 0.0, places=4)

    def test_engagement_and_win_combine(self):
        rw = Rewards(_base_config(battle_engagement_reward=3.0, battle_win_reward=8.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        r_engage, _ = rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=50), "")
        self.assertAlmostEqual(float(r_engage), 3.0, places=4)
        r_win, _ = rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=0), "")
        self.assertAlmostEqual(float(r_win), 8.0, places=4)


class TestBattleRewardCap(unittest.TestCase):
    def test_total_battle_reward_capped_per_episode(self):
        rw = Rewards(
            _base_config(
                battle_engagement_reward=10.0,
                battle_win_reward=10.0,
                battle_reward_episode_cap=15.0,
            )
        )
        rw.calculate_reward(_env_vars(battle_type=0), "")
        r1, _ = rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=50), "")  # +10
        self.assertAlmostEqual(float(r1), 10.0, places=4)
        r2, _ = rw.calculate_reward(
            _env_vars(battle_type=1, enemy_hp=0), ""
        )  # wants +10, only 5 left
        self.assertAlmostEqual(float(r2), 5.0, places=4)
        r3, _ = rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(
            _env_vars(battle_type=1, enemy_hp=50), ""
        )  # would pay more, capped to 0
        r4, _ = rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=0), "")
        bd = rw.get_episode_breakdown()
        self.assertAlmostEqual(bd["battle"], 15.0, places=4)


class TestLevelUpReward(unittest.TestCase):
    def test_pays_per_level_gained(self):
        rw = Rewards(
            _base_config(
                level_up_reward=10,
                goals=[
                    {"type": "level", "threshold": 5},
                ],
            )
        )
        rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0)), "")
        r, _ = rw.calculate_reward(_env_vars(party_info=(1, 7, 20, 0)), "")
        self.assertAlmostEqual(float(r), 20.0, places=4)  # +2 levels * 10

    def test_suppressed_on_party_size_change(self):
        rw = Rewards(
            _base_config(
                level_up_reward=10,
                goals=[
                    {"type": "level", "threshold": 5},
                ],
            )
        )
        rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0)), "")
        r, _ = rw.calculate_reward(_env_vars(party_info=(2, 15, 20, 0)), "")
        self.assertAlmostEqual(float(r), 0.0, places=4)


class TestPokedexReward(unittest.TestCase):
    def test_owned_and_seen_pay_independently(self):
        rw = Rewards(
            _base_config(
                pokedex_owned_reward=150,
                pokedex_first_sight_reward=10,
                goals=[
                    {"type": "pokedex", "kind": "seen", "threshold": 5},
                    {"type": "pokedex", "kind": "owned", "threshold": 5},
                ],
            )
        )
        rw.calculate_reward(_env_vars(pokedex_seen=0, pokedex_owned=0), "")
        r, _ = rw.calculate_reward(_env_vars(pokedex_seen=2, pokedex_owned=1), "")
        self.assertAlmostEqual(float(r), 2 * 10 + 1 * 150, places=4)


class TestTerminateOnGoalComplete(unittest.TestCase):
    def test_off_by_default_runs_to_budget(self):
        rw = Rewards(
            _base_config(
                episode_length=3,
                goals=[{"type": "map", "map_bank": 26, "map_num": 3}],
            )
        )
        rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        _, done = rw.calculate_reward(_env_vars(map_bank=26, map_num=3), "")
        self.assertFalse(done)

    def test_true_ends_episode_on_goal_completion(self):
        rw = Rewards(
            _base_config(
                episode_length=100,
                terminate_on_goal_complete=True,
                goals=[{"type": "map", "map_bank": 26, "map_num": 3}],
            )
        )
        rw.calculate_reward(_env_vars(map_bank=24, map_num=4), "")
        _, done = rw.calculate_reward(_env_vars(map_bank=26, map_num=3), "")
        self.assertTrue(done)
        self.assertFalse(rw.truncated)  # natural terminal, not a budget cut-off


class TestAlwaysOnMilestones(unittest.TestCase):
    """Flag/pokedex/level/key-item reward must fire even with an EMPTY
    `goals` list — this is the whole point of generalizing them away from
    per-stage hand-authored targets."""

    def test_flag_fires_with_no_goals_configured(self):
        rw = Rewards(_base_config(flag_progress_reward=500, goals=[]))
        rw.calculate_reward(_env_vars(), "")  # seeds baseline
        flags = _set_flag(_zero_flags(), 26)  # got_starter — in the curated table
        r, _ = rw.calculate_reward(_env_vars(story_flags=flags), "")
        self.assertAlmostEqual(float(r), 500.0, places=4)

    def test_uncurated_flag_bit_does_not_fire(self):
        """Sanity check this isn't literally all 2048 bits — only the
        curated ~50 in _DERIVED_FLAG_TABLE are reward-eligible."""
        rw = Rewards(_base_config(flag_progress_reward=500, goals=[]))
        rw.calculate_reward(_env_vars(), "")
        flags = _set_flag(_zero_flags(), 2000)  # not in _DERIVED_FLAG_TABLE
        r, _ = rw.calculate_reward(_env_vars(story_flags=flags), "")
        self.assertAlmostEqual(float(r), 0.0, places=4)

    def test_pokedex_fires_with_no_goals_configured(self):
        rw = Rewards(
            _base_config(
                pokedex_owned_reward=150,
                pokedex_first_sight_reward=10,
                goals=[],
            )
        )
        rw.calculate_reward(_env_vars(pokedex_seen=0, pokedex_owned=0), "")
        r, _ = rw.calculate_reward(_env_vars(pokedex_seen=1, pokedex_owned=1), "")
        self.assertAlmostEqual(float(r), 10.0 + 150.0, places=4)

    def test_level_fires_with_no_goals_configured(self):
        rw = Rewards(_base_config(level_up_reward=10, goals=[]))
        rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0)), "")
        r, _ = rw.calculate_reward(_env_vars(party_info=(1, 6, 20, 0)), "")
        self.assertAlmostEqual(float(r), 10.0, places=4)

    def test_key_item_fires_with_no_goals_configured(self):
        rw = Rewards(_base_config(key_item_pickup_reward=5, goals=[]))
        rw.calculate_reward(_env_vars(key_items_count=0), "")
        r, _ = rw.calculate_reward(_env_vars(key_items_count=1), "")
        self.assertAlmostEqual(float(r), 5.0, places=4)

    def test_already_true_at_episode_start_does_not_fire(self):
        """Baseline is seeded from the first call, not hardcoded to
        false/zero — a save-state that already has a flag set must not
        re-pay it every episode."""
        flags = _set_flag(_zero_flags(), 26)
        rw = Rewards(_base_config(flag_progress_reward=500, goals=[]))
        rw.calculate_reward(
            _env_vars(story_flags=flags), ""
        )  # seeds baseline as already-true
        r, _ = rw.calculate_reward(_env_vars(story_flags=flags), "")
        self.assertAlmostEqual(float(r), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
