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
import math
import unittest
import numpy as np

from PoliwhiRL.environment.rewards import Rewards
from PoliwhiRL.environment.visit_archive import VisitArchive
from PoliwhiRL.environment.domain_reward_balancer import discovery_domain


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

    def test_wild_cap_does_not_starve_trainer_pool(self):
        """Wild and trainer battles draw from independent per-episode caps,
        so a wild-battle budget that's already exhausted must not block a
        trainer engagement from paying."""
        rw = Rewards(
            _base_config(
                battle_engagement_reward=10.0,
                battle_win_reward=0.0,
                battle_reward_episode_cap=10.0,
                trainer_battle_reward_episode_cap=10.0,
                trainer_battle_premium_multiplier=1.0,
                first_trainer_battle_reward=0.0,
                first_trainer_battle_seeded_reward=0.0,
            )
        )
        rw.calculate_reward(_env_vars(battle_type=0), "")
        r_wild, _ = rw.calculate_reward(_env_vars(battle_type=1), "")  # exhausts wild pool
        self.assertAlmostEqual(float(r_wild), 10.0, places=4)
        rw.calculate_reward(_env_vars(battle_type=0), "")
        r_trainer, _ = rw.calculate_reward(_env_vars(battle_type=2), "")
        self.assertAlmostEqual(float(r_trainer), 10.0, places=4)

    def test_trainer_cap_independent_of_wild_cap_value(self):
        """trainer_battle_reward_episode_cap, not battle_reward_episode_cap,
        bounds trainer-battle income."""
        rw = Rewards(
            _base_config(
                battle_engagement_reward=10.0,
                battle_win_reward=0.0,
                battle_reward_episode_cap=5.0,
                trainer_battle_reward_episode_cap=25.0,
                trainer_battle_premium_multiplier=1.0,
                first_trainer_battle_reward=0.0,
                first_trainer_battle_seeded_reward=0.0,
            )
        )
        rw.calculate_reward(_env_vars(battle_type=0), "")
        r_trainer, _ = rw.calculate_reward(_env_vars(battle_type=2), "")
        self.assertAlmostEqual(float(r_trainer), 10.0, places=4)


class TestTrainerBattlePremium(unittest.TestCase):
    """Every trainer battle (not just the first) earns a premium over the
    generic engagement/win reward, and wild/trainer decay is tracked
    separately so grinding one can't deplete the other's payout."""

    def test_trainer_engagement_gets_premium(self):
        rw = Rewards(
            _base_config(
                battle_engagement_reward=3.0,
                trainer_battle_premium_multiplier=2.5,
                first_trainer_battle_reward=0.0,
                first_trainer_battle_seeded_reward=0.0,
            )
        )
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2), "")
        self.assertAlmostEqual(rw.get_episode_breakdown()["battle"], 7.5, places=4)

    def test_trainer_win_gets_premium(self):
        rw = Rewards(
            _base_config(
                battle_win_reward=8.0,
                trainer_battle_premium_multiplier=2.5,
                first_trainer_battle_reward=0.0,
                first_trainer_battle_seeded_reward=0.0,
            )
        )
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=50), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=0), "")
        self.assertAlmostEqual(rw.get_episode_breakdown()["battle"], 20.0, places=4)

    def test_wild_battle_unaffected_by_premium(self):
        rw = Rewards(
            _base_config(
                battle_engagement_reward=3.0,
                trainer_battle_premium_multiplier=2.5,
            )
        )
        rw.calculate_reward(_env_vars(battle_type=0), "")
        r, _ = rw.calculate_reward(_env_vars(battle_type=1), "")
        self.assertAlmostEqual(float(r), 3.0, places=4)

    def test_wild_and_trainer_decay_tracked_independently_same_map(self):
        """A wild engagement on a map must not deplete what a trainer
        engagement on that SAME map pays (or vice versa) — they're keyed
        by (map, battle_type), not just map."""
        rw = Rewards(
            _base_config(
                battle_engagement_reward=3.0,
                battle_decay_coef=0.2,
                trainer_battle_premium_multiplier=1.0,
                first_trainer_battle_reward=0.0,
                first_trainer_battle_seeded_reward=0.0,
            )
        )
        rw.calculate_reward(_env_vars(battle_type=0), "")
        r_wild, _ = rw.calculate_reward(_env_vars(battle_type=1), "")  # wild n=1
        self.assertAlmostEqual(float(r_wild), 3.0, places=4)
        rw.calculate_reward(_env_vars(battle_type=0), "")  # battle ends
        rw.calculate_reward(
            _env_vars(battle_type=2), ""
        )  # trainer n=1 on same map, NOT decayed by the wild fire
        self.assertAlmostEqual(rw.get_episode_breakdown()["battle"], 3.0 + 3.0, places=4)

    def test_repeat_trainer_engagement_same_map_still_decays(self):
        rw = Rewards(
            _base_config(
                battle_engagement_reward=3.0,
                battle_decay_coef=0.2,
                trainer_battle_premium_multiplier=1.0,
                first_trainer_battle_reward=0.0,
                first_trainer_battle_seeded_reward=0.0,
            )
        )
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2), "")  # trainer n=1: pays 3.0
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2), "")  # trainer n=2
        self.assertAlmostEqual(rw.get_episode_breakdown()["battle"], 3.0 + 3.0 / 1.2, places=4)


class TestFirstTrainerBattleReward(unittest.TestCase):
    """Big, checkpoint-style, one-time bonus for the first-ever TRAINER
    battle (battle_type 0 -> 2) — a reliable proxy for the mandatory Route
    29 rival fight, bridging the long gap between picking up the Mystery
    Egg and delivering it to Elm (see rewards.py's _battle_progress)."""

    def test_wild_battle_does_not_fire(self):
        rw = Rewards(_base_config(first_trainer_battle_reward=40.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=1), "")
        self.assertAlmostEqual(rw.get_episode_breakdown()["first_trainer_battle"], 0.0, places=4)

    def test_trainer_battle_fires_full_reward_when_never_fired_before(self):
        rw = Rewards(_base_config(first_trainer_battle_reward=40.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        r, _ = rw.calculate_reward(_env_vars(battle_type=2), "")
        self.assertAlmostEqual(float(r), 40.0, places=4)
        self.assertAlmostEqual(rw.get_episode_breakdown()["first_trainer_battle"], 40.0, places=4)

    def test_does_not_refire_within_same_episode(self):
        rw = Rewards(_base_config(first_trainer_battle_reward=40.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2), "")  # fires
        rw.calculate_reward(_env_vars(battle_type=0), "")  # battle ends
        rw.calculate_reward(_env_vars(battle_type=2), "")  # re-enter, same episode
        self.assertAlmostEqual(rw.get_episode_breakdown()["first_trainer_battle"], 40.0, places=4)

    def test_bypasses_battle_reward_episode_cap(self):
        rw = Rewards(
            _base_config(
                battle_engagement_reward=10.0,
                battle_reward_episode_cap=0.0,
                first_trainer_battle_reward=40.0,
            )
        )
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2), "")
        bd = rw.get_episode_breakdown()
        self.assertAlmostEqual(bd["battle"], 0.0, places=4)
        self.assertAlmostEqual(bd["first_trainer_battle"], 40.0, places=4)


class TestFirstTrainerBattleHonestSeededDecay(unittest.TestCase):
    def test_honest_decays_with_prior_honest_fires(self):
        archive = VisitArchive()
        rw1 = Rewards(
            _base_config(
                first_trainer_battle_reward=40.0,
                first_trainer_battle_honest_floor=0.1,
            ),
            visit_archive=archive,
        )
        rw1.calculate_reward(_env_vars(battle_type=0), "")
        r1, _ = rw1.calculate_reward(_env_vars(battle_type=2), "")
        self.assertAlmostEqual(float(r1), 40.0, places=4)
        archive.merge_milestones(
            flags_fired=[],
            pokedex_seen_max=0,
            pokedex_owned_max=0,
            level_max=0,
            key_items_max=0,
            first_trainer_battle_fired=rw1.get_milestone_state()["first_trainer_battle_fired"],
            checkpoint_seeded=False,
        )

        rw2 = Rewards(
            _base_config(
                first_trainer_battle_reward=40.0,
                first_trainer_battle_honest_floor=0.1,
            ),
            visit_archive=archive,
        )
        rw2.calculate_reward(_env_vars(battle_type=0), "")
        r2, _ = rw2.calculate_reward(_env_vars(battle_type=2), "")
        self.assertAlmostEqual(float(r2), 40.0 / math.sqrt(2.0), places=4)

    def test_seeded_uses_seeded_reward_and_decay(self):
        archive = VisitArchive()
        rw = Rewards(
            _base_config(
                episode_seeded=True,
                first_trainer_battle_seeded_reward=10.0,
            ),
            visit_archive=archive,
        )
        rw.calculate_reward(_env_vars(battle_type=0), "")
        r, _ = rw.calculate_reward(_env_vars(battle_type=2), "")
        self.assertAlmostEqual(float(r), 10.0, places=4)


class TestFirstTrainerBattleDiscoveryAndDomain(unittest.TestCase):
    def test_logs_run_wide_first_ever_discovery(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2), "")
        discoveries = rw.get_discoveries()
        self.assertTrue(any(d["type"] == "first_trainer_battle" for d in discoveries))

    def test_counted_in_progressed_domains(self):
        rw = Rewards(_base_config(first_trainer_battle_reward=40.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2), "")
        self.assertIn("milestone", rw.progressed_domains())

    def test_attributed_to_milestone_domain(self):
        self.assertEqual(discovery_domain("first_trainer_battle"), "milestone")

    def test_fire_counts_survive_state_round_trip(self):
        archive = VisitArchive()
        rw = Rewards(_base_config(first_trainer_battle_reward=40.0), visit_archive=archive)
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2), "")
        archive.merge_milestones(
            flags_fired=[],
            pokedex_seen_max=0,
            pokedex_owned_max=0,
            level_max=0,
            key_items_max=0,
            first_trainer_battle_fired=rw.get_milestone_state()["first_trainer_battle_fired"],
            checkpoint_seeded=False,
        )
        replica = VisitArchive()
        replica.load_state(archive.to_state())
        self.assertEqual(replica.first_trainer_battle_fire_count(seeded=False), 1)


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
        rw.calculate_reward(_env_vars(story_flags=flags), "")  # seeds baseline as already-true
        r, _ = rw.calculate_reward(_env_vars(story_flags=flags), "")
        self.assertAlmostEqual(float(r), 0.0, places=4)


class TestDuplicateCatchReward(unittest.TestCase):
    """Small reward for catching a species already owned — the
    complementary case to pokedex_owned_reward (which only pays on a
    genuinely NEW species, see RAM.py.get_pokedex_owned). See
    Rewards._catch_progress."""

    def test_new_species_catch_pays_nothing_from_this_channel(self):
        rw = Rewards(_base_config(duplicate_catch_reward=20.0))
        rw.calculate_reward(_env_vars(battle_type=0, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        rw.calculate_reward(_env_vars(battle_type=1, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        r, _ = rw.calculate_reward(
            _env_vars(battle_type=1, party_info=(2, 5, 20, 0), pokedex_owned=2), ""
        )
        self.assertAlmostEqual(float(r), 0.0, places=4)
        self.assertEqual(rw.get_episode_event_counts()["new_species_catches"], 1)
        self.assertEqual(rw.get_episode_event_counts()["duplicate_catches"], 0)

    def test_duplicate_species_catch_pays_the_small_reward(self):
        rw = Rewards(_base_config(duplicate_catch_reward=20.0))
        rw.calculate_reward(_env_vars(battle_type=0, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        rw.calculate_reward(_env_vars(battle_type=1, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        r, _ = rw.calculate_reward(
            _env_vars(battle_type=1, party_info=(2, 5, 20, 0), pokedex_owned=1), ""
        )
        self.assertAlmostEqual(float(r), 20.0, places=4)
        counts = rw.get_episode_event_counts()
        self.assertEqual(counts["catches"], 1)
        self.assertEqual(counts["duplicate_catches"], 1)
        self.assertEqual(counts["new_species_catches"], 0)

    def test_overworld_gift_does_not_count_as_a_catch(self):
        """party_size rising with no adjacent wild battle_type (a starter
        pickup, the Mystery Egg hand-in, a trade) must never be mistaken
        for a catch."""
        rw = Rewards(_base_config(duplicate_catch_reward=20.0))
        rw.calculate_reward(_env_vars(battle_type=0, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        r, _ = rw.calculate_reward(
            _env_vars(battle_type=0, party_info=(2, 5, 20, 0), pokedex_owned=1), ""
        )
        self.assertAlmostEqual(float(r), 0.0, places=4)
        counts = rw.get_episode_event_counts()
        self.assertEqual(counts["catches"], 0)
        self.assertEqual(counts["duplicate_catches"], 0)

    def test_trainer_battle_party_change_does_not_count_as_a_catch(self):
        """You cannot catch a trainer's Pokémon — battle_type == 2 party
        changes (e.g. a swap/heal quirk) must not pay or count."""
        rw = Rewards(_base_config(duplicate_catch_reward=20.0))
        rw.calculate_reward(_env_vars(battle_type=0, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        rw.calculate_reward(_env_vars(battle_type=2, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        r, _ = rw.calculate_reward(
            _env_vars(battle_type=2, party_info=(2, 5, 20, 0), pokedex_owned=1), ""
        )
        self.assertAlmostEqual(float(r), 0.0, places=4)
        self.assertEqual(rw.get_episode_event_counts()["catches"], 0)

    def test_decays_on_repeated_duplicate_catches_same_map(self):
        rw = Rewards(
            _base_config(duplicate_catch_reward=20.0, battle_decay_coef=0.2)
        )
        rw.calculate_reward(_env_vars(battle_type=0, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        rw.calculate_reward(_env_vars(battle_type=1, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        r1, _ = rw.calculate_reward(
            _env_vars(battle_type=1, party_info=(2, 5, 20, 0), pokedex_owned=1), ""
        )
        rw.calculate_reward(_env_vars(battle_type=0, party_info=(2, 5, 20, 0), pokedex_owned=1), "")
        rw.calculate_reward(_env_vars(battle_type=1, party_info=(2, 5, 20, 0), pokedex_owned=1), "")
        r2, _ = rw.calculate_reward(
            _env_vars(battle_type=1, party_info=(3, 5, 20, 0), pokedex_owned=1), ""
        )
        self.assertAlmostEqual(float(r1), 20.0, places=4)
        self.assertAlmostEqual(float(r2), 20.0 / 1.2, places=4)

    def test_total_duplicate_catch_reward_capped_per_episode(self):
        rw = Rewards(
            _base_config(duplicate_catch_reward=20.0, catch_reward_episode_cap=25.0)
        )
        rw.calculate_reward(_env_vars(battle_type=0, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        rw.calculate_reward(_env_vars(battle_type=1, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        rw.calculate_reward(
            _env_vars(battle_type=1, party_info=(2, 5, 20, 0), pokedex_owned=1), ""
        )  # +20, cap now has 5 left
        rw.calculate_reward(_env_vars(battle_type=0, party_info=(2, 5, 20, 0), pokedex_owned=1), "")
        rw.calculate_reward(_env_vars(battle_type=1, party_info=(2, 5, 20, 0), pokedex_owned=1), "")
        r2, _ = rw.calculate_reward(
            _env_vars(battle_type=1, party_info=(3, 5, 20, 0), pokedex_owned=1), ""
        )  # wants 20/1.2=16.67, only 5 left
        self.assertAlmostEqual(float(r2), 5.0, places=4)
        self.assertAlmostEqual(rw.get_episode_breakdown()["catch"], 25.0, places=4)

    def test_catch_landing_one_step_after_battle_type_reverts_still_counts(self):
        """A catch's party_size bump can land the step AFTER battle_type has
        already reverted to 0 (frame-skip/RAM-read timing) -- this must
        still be detected as a wild catch via the "immediately preceding
        step" half of the gate, tracked independently of
        Rewards._battle_progress's own _prev_battle_type (see
        Rewards._catch_progress docstring)."""
        rw = Rewards(_base_config(duplicate_catch_reward=20.0))
        rw.calculate_reward(_env_vars(battle_type=0, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        rw.calculate_reward(_env_vars(battle_type=1, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        r, _ = rw.calculate_reward(
            _env_vars(battle_type=0, party_info=(2, 5, 20, 0), pokedex_owned=1), ""
        )
        self.assertAlmostEqual(float(r), 20.0, places=4)
        counts = rw.get_episode_event_counts()
        self.assertEqual(counts["catches"], 1)
        self.assertEqual(counts["duplicate_catches"], 1)

    def test_separate_from_battle_reward_episode_cap(self):
        rw = Rewards(
            _base_config(
                duplicate_catch_reward=20.0,
                battle_engagement_reward=10.0,
                battle_reward_episode_cap=0.0,
                catch_reward_episode_cap=60.0,
            )
        )
        rw.calculate_reward(_env_vars(battle_type=0, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        r, _ = rw.calculate_reward(
            _env_vars(battle_type=1, party_info=(2, 5, 20, 0), pokedex_owned=1), ""
        )
        self.assertAlmostEqual(float(r), 20.0, places=4)
        bd = rw.get_episode_breakdown()
        self.assertAlmostEqual(bd["battle"], 0.0, places=4)
        self.assertAlmostEqual(bd["catch"], 20.0, places=4)


class TestFirstTrainerBattleWinReward(unittest.TestCase):
    """Rare, high-value event: mirrors TestFirstTrainerBattleReward exactly
    but keyed to an actual trainer-battle WIN (enemy_hp -> 0), not merely
    entering one."""

    def test_trainer_engagement_alone_does_not_fire(self):
        rw = Rewards(_base_config(first_trainer_battle_win_reward=50.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=50), "")
        self.assertAlmostEqual(
            rw.get_episode_breakdown()["first_trainer_battle_win"], 0.0, places=4
        )

    def test_wild_win_does_not_fire(self):
        rw = Rewards(_base_config(first_trainer_battle_win_reward=50.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=50), "")
        rw.calculate_reward(_env_vars(battle_type=1, enemy_hp=0), "")
        self.assertAlmostEqual(
            rw.get_episode_breakdown()["first_trainer_battle_win"], 0.0, places=4
        )

    def test_trainer_win_fires_full_reward_when_never_fired_before(self):
        rw = Rewards(_base_config(first_trainer_battle_win_reward=50.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=50), "")
        r, _ = rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=0), "")
        self.assertAlmostEqual(float(r), 50.0, places=4)
        self.assertAlmostEqual(
            rw.get_episode_breakdown()["first_trainer_battle_win"], 50.0, places=4
        )
        self.assertEqual(rw.get_episode_event_counts()["trainer_wins"], 1)

    def test_does_not_refire_within_same_episode(self):
        rw = Rewards(_base_config(first_trainer_battle_win_reward=50.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=50), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=0), "")  # fires
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=50), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=0), "")  # second win, same ep
        self.assertAlmostEqual(
            rw.get_episode_breakdown()["first_trainer_battle_win"], 50.0, places=4
        )
        self.assertEqual(rw.get_episode_event_counts()["trainer_wins"], 2)

    def test_bypasses_battle_reward_episode_cap(self):
        rw = Rewards(
            _base_config(
                battle_win_reward=10.0,
                battle_reward_episode_cap=0.0,
                first_trainer_battle_win_reward=50.0,
            )
        )
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=50), "")
        r, _ = rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=0), "")
        self.assertAlmostEqual(float(r), 50.0, places=4)
        bd = rw.get_episode_breakdown()
        self.assertAlmostEqual(bd["battle"], 0.0, places=4)
        self.assertAlmostEqual(bd["first_trainer_battle_win"], 50.0, places=4)


class TestFirstTrainerBattleWinHonestSeededDecay(unittest.TestCase):
    def test_honest_decays_with_prior_honest_fires(self):
        archive = VisitArchive()
        rw1 = Rewards(
            _base_config(
                first_trainer_battle_win_reward=50.0,
                first_trainer_battle_win_honest_floor=0.1,
            ),
            visit_archive=archive,
        )
        rw1.calculate_reward(_env_vars(battle_type=0), "")
        rw1.calculate_reward(_env_vars(battle_type=2, enemy_hp=50), "")
        r1, _ = rw1.calculate_reward(_env_vars(battle_type=2, enemy_hp=0), "")
        self.assertAlmostEqual(float(r1), 50.0, places=4)
        archive.merge_milestones(
            flags_fired=[],
            pokedex_seen_max=0,
            pokedex_owned_max=0,
            level_max=0,
            key_items_max=0,
            first_trainer_battle_win_fired=rw1.get_milestone_state()[
                "first_trainer_battle_win_fired"
            ],
            checkpoint_seeded=False,
        )

        rw2 = Rewards(
            _base_config(
                first_trainer_battle_win_reward=50.0,
                first_trainer_battle_win_honest_floor=0.1,
            ),
            visit_archive=archive,
        )
        rw2.calculate_reward(_env_vars(battle_type=0), "")
        rw2.calculate_reward(_env_vars(battle_type=2, enemy_hp=50), "")
        r2, _ = rw2.calculate_reward(_env_vars(battle_type=2, enemy_hp=0), "")
        self.assertAlmostEqual(float(r2), 50.0 / math.sqrt(2.0), places=4)

    def test_seeded_uses_seeded_reward_and_decay(self):
        archive = VisitArchive()
        rw = Rewards(
            _base_config(
                episode_seeded=True,
                first_trainer_battle_win_seeded_reward=15.0,
            ),
            visit_archive=archive,
        )
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=50), "")
        r, _ = rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=0), "")
        self.assertAlmostEqual(float(r), 15.0, places=4)


class TestFirstTrainerBattleWinDiscoveryAndDomain(unittest.TestCase):
    def test_logs_run_wide_first_ever_discovery(self):
        rw = Rewards(_base_config(first_trainer_battle_win_reward=50.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=50), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=0), "")
        discoveries = rw.get_discoveries()
        self.assertTrue(any(d["type"] == "first_trainer_battle_win" for d in discoveries))

    def test_counted_in_progressed_domains_as_battle(self):
        rw = Rewards(_base_config(first_trainer_battle_win_reward=50.0))
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=50), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=0), "")
        self.assertIn("battle", rw.progressed_domains())

    def test_attributed_to_battle_domain(self):
        self.assertEqual(discovery_domain("first_trainer_battle_win"), "battle")

    def test_fire_counts_survive_state_round_trip(self):
        archive = VisitArchive()
        rw = Rewards(
            _base_config(first_trainer_battle_win_reward=50.0), visit_archive=archive
        )
        rw.calculate_reward(_env_vars(battle_type=0), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=50), "")
        rw.calculate_reward(_env_vars(battle_type=2, enemy_hp=0), "")
        archive.merge_milestones(
            flags_fired=[],
            pokedex_seen_max=0,
            pokedex_owned_max=0,
            level_max=0,
            key_items_max=0,
            first_trainer_battle_win_fired=rw.get_milestone_state()[
                "first_trainer_battle_win_fired"
            ],
            checkpoint_seeded=False,
        )
        replica = VisitArchive()
        replica.load_state(archive.to_state())
        self.assertEqual(replica.first_trainer_battle_win_fire_count(seeded=False), 1)
        self.assertTrue(replica.first_trainer_battle_win_ever_fired())


class TestEpisodeEventCounts(unittest.TestCase):
    def test_zero_by_default(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(battle_type=0), "")
        counts = rw.get_episode_event_counts()
        self.assertEqual(
            counts,
            {
                "trainer_wins": 0,
                "catches": 0,
                "new_species_catches": 0,
                "duplicate_catches": 0,
            },
        )

    def test_resets_on_new_episode(self):
        rw = Rewards(_base_config(duplicate_catch_reward=20.0))
        rw.calculate_reward(_env_vars(battle_type=0, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        rw.calculate_reward(_env_vars(battle_type=1, party_info=(1, 5, 20, 0), pokedex_owned=1), "")
        rw.calculate_reward(
            _env_vars(battle_type=1, party_info=(2, 5, 20, 0), pokedex_owned=1), ""
        )
        self.assertEqual(rw.get_episode_event_counts()["catches"], 1)
        rw.start_new_episode()
        self.assertEqual(rw.get_episode_event_counts()["catches"], 0)


if __name__ == "__main__":
    unittest.main()
