# -*- coding: utf-8 -*-
"""Domain reward balancing (see PoliwhiRL/environment/domain_reward_balancer.py).

Pinned behaviours:

- Disabled by default: Rewards() with no domain_reward_balancing_enabled key
  (or set False) delivers the exact same total as before this mechanism
  existed, and domain_scale_adjustment is always 0.0.
- Enabled: DomainRewardBalancer.scale() equalises typical fire magnitude
  across domains (a small frequent signal and a large rare one converge to
  comparable scaled magnitude), is a no-op on exact zero, and respects
  min_std as a floor so a near-constant stream doesn't blow up 1/std.
- The progress-activity multiplier decays toward the configured floor after
  a domain goes `domain_progress_window_episodes` episodes without genuine
  (run-wide first-ever, or for battle: any-this-episode) progress, and
  recovers to 1.0 the instant it produces progress again.
- Rewards.progressed_domains() correctly attributes discoveries and battle
  activity to their domains.
- End-to-end through Rewards.calculate_reward, delivered_total still
  reconciles exactly against raw sources + all adjustment keys (including
  the new domain_scale_adjustment).
"""
import unittest

import numpy as np

from PoliwhiRL.environment.domain_reward_balancer import DomainRewardBalancer
from PoliwhiRL.environment.rewards import Rewards


def _zero_flags():
    return np.zeros(256, dtype=np.uint8)


def _with_flag_set(flags, flag_num):
    """story_flags is bit-packed: flag_num -> (byte_idx, bit_idx)."""
    flags = flags.copy()
    byte_idx, bit_idx = flag_num // 8, flag_num % 8
    flags[byte_idx] |= 1 << bit_idx
    return flags


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


def _env_vars(x=4, y=3, map_bank=24, map_num=7, story_flags=None):
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
        "story_flags": _zero_flags() if story_flags is None else story_flags,
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


class TestDomainRewardBalancerUnit(unittest.TestCase):
    def test_disabled_is_a_no_op(self):
        balancer = DomainRewardBalancer({"domain_reward_balancing_enabled": False})
        self.assertEqual(balancer.scale("frontier", 0.25), 0.25)
        self.assertEqual(balancer.scale("milestone", 20.0), 20.0)

    def test_zero_is_always_a_no_op_even_when_enabled(self):
        balancer = DomainRewardBalancer({"domain_reward_balancing_enabled": True})
        self.assertEqual(balancer.scale("frontier", 0.0), 0.0)

    def test_equalises_typical_fire_magnitude(self):
        balancer = DomainRewardBalancer(
            {
                "domain_reward_balancing_enabled": True,
                "domain_reward_min_magnitude": 1e-6,
            }
        )
        # A small, constant-magnitude stream (frontier) vs a large,
        # constant-magnitude stream (milestone) — after enough observations
        # each domain's own running mean converges to its magnitude, so both
        # scale to unit size.
        frontier_scaled = [balancer.scale("frontier", 0.25) for _ in range(200)]
        milestone_scaled = [balancer.scale("milestone", 20.0) for _ in range(200)]
        self.assertAlmostEqual(frontier_scaled[-1], 1.0, places=4)
        self.assertAlmostEqual(milestone_scaled[-1], 1.0, places=4)

    def test_variable_stream_keeps_internal_spread(self):
        balancer = DomainRewardBalancer(
            {
                "domain_reward_balancing_enabled": True,
                "domain_reward_min_magnitude": 1e-6,
            }
        )
        for _ in range(100):
            balancer.scale("milestone", 10.0)
        small = balancer.scale("milestone", 10.0)
        for _ in range(100):
            balancer.scale("milestone", 10.0)
        big = balancer.scale("milestone", 30.0)
        # A 3x-larger fire still scales to noticeably more than a typical
        # one — cross-domain scale is equalised, not within-domain spread.
        self.assertGreater(big, small * 1.5)

    def test_min_magnitude_floors_a_near_zero_stream(self):
        balancer = DomainRewardBalancer(
            {
                "domain_reward_balancing_enabled": True,
                "domain_reward_min_magnitude": 0.05,
            }
        )
        # First-ever fire: running mean is still exactly this value, so
        # scale_factor = 1/max(0.001, 0.05) = 20 (floored), not 1/0.001.
        scaled = balancer.scale("frontier", 0.001)
        self.assertAlmostEqual(scaled, 0.001 * (1.0 / 0.05), places=6)

    def test_activity_multiplier_decays_then_recovers(self):
        balancer = DomainRewardBalancer(
            {
                "domain_reward_balancing_enabled": True,
                "domain_progress_window_episodes": 10,
                "domain_progress_floor": 0.2,
            }
        )
        self.assertAlmostEqual(balancer.activity_multiplier("milestone"), 1.0)
        for _ in range(10):
            balancer.note_episode_end(set())  # no domain progressed
        self.assertAlmostEqual(balancer.activity_multiplier("milestone"), 0.2)
        balancer.note_episode_end({"milestone"})
        self.assertAlmostEqual(balancer.activity_multiplier("milestone"), 1.0)

    def test_catching_domain_scales_independently_of_milestone(self):
        """Raising a catching-domain source's magnitude must not change the
        milestone domain's scale factor — the whole point of splitting them
        out (see domain_reward_balancer.py's 2026-10-16 note)."""
        balancer = DomainRewardBalancer(
            {
                "domain_reward_balancing_enabled": True,
                "domain_reward_min_magnitude": 1e-6,
            }
        )
        for _ in range(50):
            balancer.scale("milestone", 20.0)
        milestone_scale_before = balancer.diagnostics()["milestone"]["scale_factor"]
        for _ in range(50):
            balancer.scale("catching", 90.0)
        milestone_scale_after = balancer.diagnostics()["milestone"]["scale_factor"]
        self.assertAlmostEqual(milestone_scale_before, milestone_scale_after, places=6)

    def test_priority_weight_defaults_to_one_and_is_configurable(self):
        default_balancer = DomainRewardBalancer(
            {"domain_reward_balancing_enabled": True, "domain_reward_min_magnitude": 1e-6}
        )
        weighted_balancer = DomainRewardBalancer(
            {
                "domain_reward_balancing_enabled": True,
                "domain_reward_min_magnitude": 1e-6,
                "domain_priority_weights": {"catching": 2.0},
            }
        )
        default_value = default_balancer.scale("catching", 10.0)
        weighted_value = weighted_balancer.scale("catching", 10.0)
        self.assertAlmostEqual(weighted_value, default_value * 2.0, places=6)
        # Unweighted domains on the same balancer are unaffected.
        self.assertAlmostEqual(
            weighted_balancer.scale("milestone", 10.0),
            default_balancer.scale("milestone", 10.0),
            places=6,
        )

    def test_activity_multiplier_disabled_window_is_always_full(self):
        balancer = DomainRewardBalancer(
            {
                "domain_reward_balancing_enabled": True,
                "domain_progress_window_episodes": 0,
            }
        )
        for _ in range(100):
            balancer.note_episode_end(set())
        self.assertAlmostEqual(balancer.activity_multiplier("frontier"), 1.0)


class TestProgressedDomains(unittest.TestCase):
    def test_flag_discovery_attributes_to_milestone(self):
        rw = Rewards(_base_config(flag_progress_reward=10))
        # First call seeds the episode's flag baseline (flag 30 = 0) —
        # _global_flag_progress_bonus only counts an observed 0->1
        # transition, not "already true at episode start".
        rw.calculate_reward(_env_vars(), "")
        flags = _with_flag_set(_zero_flags(), 30)
        rw.calculate_reward(_env_vars(story_flags=flags), "")
        self.assertIn("milestone", rw.progressed_domains())

    def test_map_discovery_attributes_to_frontier(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(map_bank=99, map_num=1), "")
        self.assertIn("frontier", rw.progressed_domains())

    def test_battle_reward_attributes_to_battle(self):
        rw = Rewards(_base_config(battle_engagement_reward=5.0))
        # First call seeds _prev_battle_type=0 — engagement reward fires on
        # the observed 0->nonzero transition, not an episode that starts
        # already mid-battle.
        rw.calculate_reward(_env_vars(), "")
        env = _env_vars()
        env["battle_type"] = 1
        rw.calculate_reward(env, "")
        self.assertIn("battle", rw.progressed_domains())

    def test_pokedex_discovery_attributes_to_catching_not_milestone(self):
        rw = Rewards(_base_config(pokedex_owned_reward=10))
        # Seed the pokedex baseline (owned=0), then cross owned=1.
        rw.calculate_reward(_env_vars(), "")
        env = _env_vars()
        env["pokedex_owned"] = 1
        rw.calculate_reward(env, "")
        domains = rw.progressed_domains()
        self.assertIn("catching", domains)
        self.assertNotIn("milestone", domains)

    def test_no_further_activity_beyond_initial_map_discovery(self):
        rw = Rewards(_base_config())
        # Visiting the very first map/cell of a fresh archive is itself a
        # one-off run-wide "map" discovery -> frontier. With no flags,
        # battle, or new map on the second step, nothing else should join.
        rw.calculate_reward(_env_vars(), "")
        rw.calculate_reward(_env_vars(), "")
        self.assertEqual(rw.progressed_domains(), {"frontier"})


class TestEndToEndReconciliation(unittest.TestCase):
    def test_disabled_total_matches_pre_existing_flat_sum(self):
        rw = Rewards(
            _base_config(
                frontier_novelty_bonus=1.0,
                flag_progress_reward=10,
                domain_reward_balancing_enabled=False,
            )
        )
        flags = _zero_flags()
        flags[30] = 1
        r, _ = rw.calculate_reward(_env_vars(x=4, y=3, story_flags=flags), "")
        bd = rw.get_episode_breakdown()
        self.assertEqual(bd["domain_scale_adjustment"], 0.0)
        self.assertAlmostEqual(float(r), bd["frontier"] + bd["flag"], places=6)

    def test_enabled_reconciles_delivered_total(self):
        rw = Rewards(
            _base_config(
                frontier_novelty_bonus=1.0,
                flag_progress_reward=10,
                domain_reward_balancing_enabled=True,
            )
        )
        flags = _zero_flags()
        flags[30] = 1
        rw.calculate_reward(_env_vars(x=4, y=3, story_flags=flags), "")
        bd = rw.get_episode_breakdown()
        raw_total = (
            bd["frontier"]
            + bd["new_map"]
            + bd["menu"]
            + bd["battle"]
            + bd["flag"]
            + bd["checkpoint_honest"]
            + bd["checkpoint_seeded"]
            + bd["map_goal"]
            + bd["maps_visited"]
            + bd["pokedex"]
            + bd["key_item"]
            + bd["level"]
            + bd["event"]
            + bd["whiteout"]
            + bd["step_penalty"]
        )
        accounted = (
            raw_total
            + bd["rounding_adjustment"]
            + bd["clip_adjustment"]
            + bd["delivery_cast_adjustment"]
            + bd["domain_scale_adjustment"]
        )
        self.assertAlmostEqual(accounted, bd["delivered_total"], places=6)
        # With both frontier and flag firing on the same step, scaling
        # should actually have changed the delivered total (not a no-op).
        self.assertNotAlmostEqual(bd["domain_scale_adjustment"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
