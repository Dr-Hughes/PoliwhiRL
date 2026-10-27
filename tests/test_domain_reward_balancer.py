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
- The progress-activity multiplier (decay toward a floor after
  `domain_progress_window_episodes` without progress) was REMOVED 2026-10-23:
  its per-domain progress proxies were asymmetric — battle counted any
  this-episode fire while every other domain needed a run-wide first-ever
  discovery — so on a plateau it transferred weight onto wild-battle
  grinding. `domain_priority_weights` remain the only durable lever.
- "event" is its own domain (2026-10-23), so event-novelty magnitude can no
  longer shrink the milestone domain's shared scale factor.
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
                # Permissive clamp bounds so this test isolates the
                # equalisation mechanism from the 2026-10-24 scale-factor
                # clamp (see test_scale_factor_is_clamped_to_configured_range).
                "domain_reward_scale_factor_min": 1e-6,
                "domain_reward_scale_factor_max": 1e6,
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
                # Permissive clamp so this isolates min_magnitude's own
                # floor from the 2026-10-24 scale-factor clamp.
                "domain_reward_scale_factor_max": 1e6,
            }
        )
        # First-ever fire: running mean is still exactly this value, so
        # scale_factor = 1/max(0.001, 0.05) = 20 (floored), not 1/0.001.
        scaled = balancer.scale("frontier", 0.001)
        self.assertAlmostEqual(scaled, 0.001 * (1.0 / 0.05), places=6)

    def test_scale_factor_is_clamped_to_configured_range(self):
        """2026-10-24 review finding 2: delivered reward tracks fire COUNT
        (per-fire normalisation turns many small fires into many unit
        fires), so on a long/deep episode an un-clamped scale factor can
        drift far enough from parity that the balancer's own correction
        (domain_scale_adjustment) outweighs the configured raw reward
        prices. The clamp bounds this independently of min_magnitude."""
        balancer = DomainRewardBalancer(
            {
                "domain_reward_balancing_enabled": True,
                "domain_reward_min_magnitude": 0.05,
                "domain_reward_scale_factor_min": 0.25,
                "domain_reward_scale_factor_max": 4.0,
            }
        )
        # A near-zero stream wants scale_factor = 1/0.05 = 20 — clamped to 4.
        tiny_scaled = balancer.scale("frontier", 0.001)
        self.assertAlmostEqual(tiny_scaled, 0.001 * 4.0, places=6)
        # A large-magnitude stream wants scale_factor = 1/20 = 0.05 —
        # clamped up to 0.25.
        large_scaled = [balancer.scale("milestone", 20.0) for _ in range(200)][-1]
        self.assertAlmostEqual(large_scaled, 20.0 * 0.25, places=4)

    def test_event_domain_scales_independently_of_milestone(self):
        """Raising event-novelty magnitude must not shrink the milestone
        domain's scale factor — the reason event left the milestone domain
        (2026-10-23 review F4)."""
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
            balancer.scale("event", 10.0)
        milestone_scale_after = balancer.diagnostics()["milestone"]["scale_factor"]
        self.assertAlmostEqual(milestone_scale_before, milestone_scale_after, places=6)

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

    def test_priority_weight_cannot_push_effective_multiplier_past_the_ceiling(self):
        """2026-10-25 fix: a priority weight > 1.0 must not let the combined
        multiplier exceed scale_factor_max even when the domain's own
        (pre-weight) scale_factor is already clamped there — this is the
        loophole that let a live run's frontier domain (scale_factor 3.67,
        near the 4.0 ceiling) times its 1.3 priority weight reach an
        effective 4.77x multiplier, silently exceeding the documented
        bound the clamp claims to enforce."""
        balancer = DomainRewardBalancer(
            {
                "domain_reward_balancing_enabled": True,
                "domain_reward_min_magnitude": 0.05,
                "domain_reward_scale_factor_min": 0.25,
                "domain_reward_scale_factor_max": 4.0,
                "domain_priority_weights": {"frontier": 1.3},
            }
        )
        # A near-zero stream drives the pre-weight scale_factor to the 4.0
        # ceiling on its own (same setup as test_scale_factor_is_clamped_to_configured_range).
        scaled = balancer.scale("frontier", 0.001)
        diag = balancer.diagnostics()["frontier"]
        self.assertAlmostEqual(diag["scale_factor"], 4.0, places=6)
        self.assertAlmostEqual(diag["effective_multiplier"], 4.0, places=6)
        self.assertAlmostEqual(scaled, 0.001 * 4.0, places=6)

    def test_priority_weight_cannot_push_effective_multiplier_past_the_floor(self):
        """Symmetric case: a priority weight < 1.0 must not push the combined
        multiplier below scale_factor_min."""
        balancer = DomainRewardBalancer(
            {
                "domain_reward_balancing_enabled": True,
                "domain_reward_min_magnitude": 1e-6,
                "domain_reward_scale_factor_min": 0.25,
                "domain_reward_scale_factor_max": 4.0,
                "domain_priority_weights": {"milestone": 0.5},
            }
        )
        # A large-magnitude stream drives the pre-weight scale_factor down to
        # the 0.25 floor on its own.
        scaled = [balancer.scale("milestone", 20.0) for _ in range(200)][-1]
        diag = balancer.diagnostics()["milestone"]
        self.assertAlmostEqual(diag["scale_factor"], 0.25, places=6)
        self.assertAlmostEqual(diag["effective_multiplier"], 0.25, places=6)
        self.assertAlmostEqual(scaled, 20.0 * 0.25, places=4)

    def test_priority_weight_defaults_to_one_and_is_configurable(self):
        default_balancer = DomainRewardBalancer(
            {
                "domain_reward_balancing_enabled": True,
                "domain_reward_min_magnitude": 1e-6,
            }
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


class TestDomainRewardBalancerPersistence(unittest.TestCase):
    """DomainRewardBalancer.state_dict/load_state_dict/merge_states — added
    2026-10-20 so reward-scale state survives a process restart instead of
    resetting to its under-informed initial state every time (the balancer
    itself has no multiprocessing-aware broadcast; this is a one-time
    checkpoint-save / process-start seed, not a live cross-worker sync)."""

    def test_round_trip_preserves_scale(self):
        balancer = DomainRewardBalancer(
            {
                "domain_reward_balancing_enabled": True,
                "domain_reward_min_magnitude": 1e-6,
            }
        )
        for _ in range(50):
            balancer.scale("milestone", 20.0)
        before = balancer.diagnostics()

        restored = DomainRewardBalancer(
            {
                "domain_reward_balancing_enabled": True,
                "domain_reward_min_magnitude": 1e-6,
            }
        )
        restored.load_state_dict(balancer.state_dict())
        after = restored.diagnostics()

        for d in ("frontier", "battle", "milestone", "catching", "event"):
            self.assertAlmostEqual(
                before[d]["scale_factor"], after[d]["scale_factor"], places=6
            )

    def test_old_checkpoint_state_with_staleness_keys_is_ignored(self):
        """Checkpoints written before the progress-activity multiplier was
        removed (2026-10-23) carry an episodes_since_progress key; loading
        them must not raise and must not resurrect the removed behaviour."""
        balancer = DomainRewardBalancer({"domain_reward_balancing_enabled": True})
        balancer.load_state_dict(
            {
                "scales": {},
                "episodes_since_progress": {"milestone": 42},
            }
        )
        self.assertNotIn("activity", balancer.diagnostics()["milestone"])

    def test_merge_states_combines_multiple_workers(self):
        workers = []
        for _ in range(3):
            w = DomainRewardBalancer(
                {
                    "domain_reward_balancing_enabled": True,
                    "domain_reward_min_magnitude": 1e-6,
                }
            )
            for _ in range(40):
                w.scale("milestone", 20.0)
            workers.append(w)

        merged_state = DomainRewardBalancer.merge_states(
            [w.state_dict() for w in workers]
        )
        merged = DomainRewardBalancer(
            {
                "domain_reward_balancing_enabled": True,
                "domain_reward_min_magnitude": 1e-6,
            }
        )
        merged.load_state_dict(merged_state)
        # All three workers saw an identical stream, so the merged estimate
        # should land on the same typical magnitude (~20) as any one of them.
        self.assertAlmostEqual(
            merged.diagnostics()["milestone"]["scale_factor"],
            workers[0].diagnostics()["milestone"]["scale_factor"],
            places=3,
        )

    def test_merge_states_empty_input_is_empty_state(self):
        self.assertEqual(DomainRewardBalancer.merge_states([]), {})
        self.assertEqual(DomainRewardBalancer.merge_states([None, {}]), {})

    def test_load_state_dict_tolerates_missing_or_empty_state(self):
        balancer = DomainRewardBalancer({"domain_reward_balancing_enabled": True})
        before = balancer.diagnostics()
        balancer.load_state_dict({})
        balancer.load_state_dict(None)
        self.assertEqual(balancer.diagnostics(), before)


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
        rw.calculate_reward(_env_vars(map_bank=25, map_num=1), "")
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
