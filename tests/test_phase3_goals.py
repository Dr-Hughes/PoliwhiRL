# -*- coding: utf-8 -*-
"""Reward / goal coverage:

- ``flag`` goal type fires exactly once on a 0→1 transition and not on
  flags already true at episode start. No reward is paid (pure exploration),
  but the flag_goals_completed counter advances for metrics.
- Frontier novelty pays ``bonus / (prior_count + 1)`` — smooth decay,
  no separate refire constant.
- RAM observation carries script/UI state buckets and no longer carries
  ``target_*`` slots.
- maps_visited goal fires once per unique map up to threshold.
- Episode terminates only on step budget (no terminate_on_goal_complete).
"""
import math
import unittest
import numpy as np

from PoliwhiRL.environment.goals import GoalsManager
from PoliwhiRL.environment.rewards import Rewards
from PoliwhiRL.environment.gym_env import RAM_FEATURE_KEYS, RAM_FEATURE_INDEX


# ----------------------------- fixtures ------------------------------ #

def _zero_flags():
    return np.zeros(256, dtype=np.uint8)


def _set_flag(arr, flag_num):
    arr = arr.copy()
    byte_idx, bit_idx = flag_num // 8, flag_num % 8
    arr[byte_idx] |= (1 << bit_idx)
    return arr


def _base_config(**overrides):
    cfg = {
        "episode_length": 100,
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


def _env_vars(x=4, y=3, map_num=7, map_bank=24, pokedex_seen=0,
              pokedex_owned=0, story_flags=None, battle_type=0,
              party_info=(0, 0, 0, 0), key_items_count=0):
    return {
        "X": x, "Y": y, "map_num": map_num, "map_bank": map_bank,
        "room": 0, "warp_number": 0, "money": 0,
        "pokedex_seen": pokedex_seen, "pokedex_owned": pokedex_owned,
        "collision_down": 0, "collision_up": 0,
        "collision_left": 0, "collision_right": 0,
        "story_flags": story_flags if story_flags is not None else _zero_flags(),
        "battle_type": battle_type, "johto_badges": 0, "player_state": 0,
        "key_items_count": key_items_count, "game_hour": 0, "bgm_id": 0,
        "enemy_hp": 0, "enemy_max_hp": 0,
        "party_info": party_info,
        "script_active": False,
    }


# ----------------------------- tests --------------------------------- #

class TestFlagGoals(unittest.TestCase):
    def test_flag_fires_on_zero_to_one_transition(self):
        """Flag goals advance the counter on 0→1 transition and pay
        flag_progress_reward exactly once — milestones are the primary
        reward signal now."""
        rw = Rewards(_base_config(goals=[{"type": "flag", "flag_num": 25}]))
        # Initial step with flag at 0 — seeds initial state, no fire.
        r0, _ = rw.calculate_reward(_env_vars(), button_press="")
        self.assertEqual(rw.flag_goals_completed, 0)
        self.assertAlmostEqual(float(r0), 0.0, places=4)

        # Step with the flag now set — counter advances, pays the milestone.
        flags = _set_flag(_zero_flags(), 25)
        r1, _ = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        self.assertEqual(rw.flag_goals_completed, 1)
        self.assertAlmostEqual(float(r1), 500.0, places=4)

        # Step again with the flag still set — should NOT fire a second time.
        r2, _ = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        self.assertEqual(rw.flag_goals_completed, 1)
        self.assertAlmostEqual(float(r2), 0.0, places=4)

    def test_flag_already_set_at_episode_start_does_not_fire(self):
        """Flags already true via replay must not advance the counter."""
        rw = Rewards(_base_config(goals=[{"type": "flag", "flag_num": 25}]))
        flags = _set_flag(_zero_flags(), 25)
        r0, _ = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        self.assertEqual(rw.flag_goals_completed, 0)
        r1, _ = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        self.assertEqual(rw.flag_goals_completed, 0)


class TestTerminateOnGoalComplete(unittest.TestCase):
    def test_terminate_on_legacy_key_rejected(self):
        cfg = _base_config(terminate_on={"type": "flag", "flag_num": 25})
        with self.assertRaisesRegex(ValueError, "terminate_on.*no longer supported"):
            Rewards(cfg)

    def test_default_runs_to_budget(self):
        """Episode must run to episode_length regardless of goal completion."""
        cfg = _base_config(
            episode_length=3,
            goals=[{"type": "flag", "flag_num": 25}],
        )
        rw = Rewards(cfg)
        flags = _set_flag(_zero_flags(), 25)
        _, done = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        self.assertFalse(done)
        _, done = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        self.assertFalse(done)
        _, done = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        self.assertFalse(done)
        # 4th step exceeds budget — terminates.
        _, done = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        self.assertTrue(done)


class TestMapGoal(unittest.TestCase):
    def test_map_goal_fires_on_entry(self):
        """A ``map`` goal fires the GoalsManager counter on entering the
        target and pays map_goal_reward — milestones are the primary
        reward signal now."""
        cfg = _base_config(
            goals=[{"type": "map", "map_bank": 24, "map_num": 3}],
        )
        rw = Rewards(cfg)
        # Start on (24, 7): not the target.
        r, done = rw.calculate_reward(
            _env_vars(map_bank=24, map_num=7), button_press=""
        )
        self.assertFalse(done)
        self.assertEqual(rw.n_map_goals_completed(), 0)
        # Enter the target map.
        r, done = rw.calculate_reward(
            _env_vars(map_bank=24, map_num=3, x=99, y=1), button_press=""
        )
        self.assertAlmostEqual(float(r), 250.0, places=4)  # default map_goal_reward
        self.assertFalse(done)  # no terminate_on_goal_complete by default
        self.assertEqual(rw.n_map_goals_completed(), 1)

    def test_map_goal_not_met_when_starting_on_target(self):
        """Starting on the target map must NOT fire the goal."""
        cfg = _base_config(
            goals=[{"type": "map", "map_bank": 24, "map_num": 7}],
        )
        rw = Rewards(cfg)
        r, done = rw.calculate_reward(
            _env_vars(map_bank=24, map_num=7), button_press=""
        )
        self.assertAlmostEqual(float(r), 0.0, places=4)
        self.assertFalse(done)
        self.assertEqual(rw.n_map_goals_completed(), 0)


class TestFrontierNovelty(unittest.TestCase):
    def test_first_visit_pays_full_bonus(self):
        rw = Rewards(_base_config(frontier_novelty_bonus=5.0))
        r0, _ = rw.calculate_reward(_env_vars(x=4, y=3), button_press="")
        self.assertEqual(float(r0), 5.0)

    def test_same_cell_same_episode_no_repeat(self):
        rw = Rewards(_base_config(frontier_novelty_bonus=5.0))
        rw.calculate_reward(_env_vars(x=4, y=3), button_press="")
        r1, _ = rw.calculate_reward(_env_vars(x=5, y=3), button_press="")
        self.assertEqual(float(r1), 0.0)

    def test_depletes_across_episodes_by_sqrt_of_persistent_visits(self):
        """OPTIONAL NGU lifelong term (frontier_lifelong_decay=True): the
        per-cell payout decays across episodes with the persistent archive
        count, bonus / sqrt(1 + visits). Re-covering known ground stops
        being income while a never-visited cell always pays in full. sqrt
        keeps a residual (never hits zero). This is OFF by default (pure
        per-episode coverage) — see the rewards.py module docstring."""
        from PoliwhiRL.environment.visit_archive import VisitArchive
        archive = VisitArchive()
        expected = [5.0, 5.0 / math.sqrt(2), 5.0 / math.sqrt(3)]
        for want in expected:
            rw = Rewards(
                _base_config(frontier_novelty_bonus=5.0, frontier_lifelong_decay=True),
                visit_archive=archive,
            )
            r, _ = rw.calculate_reward(_env_vars(x=4, y=3), button_press="")
            self.assertAlmostEqual(float(r), want, places=4)
            archive.merge_visits(rw._cells_to_record, rw._maps_to_record)
        # One increment per cell per episode.
        self.assertEqual(archive.count(24, 7, 4, 3), 3)

    def test_blocked_during_script_active(self):
        ev = _env_vars(x=4, y=3)
        ev["script_active"] = True
        rw = Rewards(_base_config(frontier_novelty_bonus=5.0))
        r0, _ = rw.calculate_reward(ev, button_press="")
        self.assertEqual(float(r0), 0.0)

    def test_fires_when_script_inactive(self):
        ev = _env_vars(x=4, y=3)
        ev["script_active"] = False
        rw = Rewards(_base_config(frontier_novelty_bonus=5.0))
        r0, _ = rw.calculate_reward(ev, button_press="")
        self.assertEqual(float(r0), 5.0)


class TestMapLifelongDecay(unittest.TestCase):
    """OPTIONAL NGU lifelong term for the MAP/BANK reward
    (map_lifelong_decay=True): re-entering a map the run has already toured
    many times pays new_map_reward / sqrt(1 + run-wide entry count), so a
    heavily-toured cluster stops being income while a genuinely-new region
    still pays full. Dissolves the "tour the known buildings every episode"
    optimum. OFF by default (flat episodic map reward)."""

    def test_map_reward_decays_across_episodes(self):
        from PoliwhiRL.environment.visit_archive import VisitArchive
        archive = VisitArchive()
        # Same bank each episode, so only the new_map term is exercised (the
        # bank is "new to the episode" every time but its decay is checked
        # separately below); use new_bank_reward=0 to isolate the map term.
        expected = [5.0, 5.0 / math.sqrt(2), 5.0 / math.sqrt(3)]
        for want in expected:
            rw = Rewards(
                _base_config(new_map_reward=5.0, new_bank_reward=0.0,
                             map_lifelong_decay=True),
                visit_archive=archive,
            )
            r = rw._new_map_bonus(_env_vars(x=4, y=3, map_bank=24, map_num=7))
            self.assertAlmostEqual(float(r), want, places=4)
            archive.merge_visits(rw._cells_to_record, rw._maps_to_record)
        self.assertEqual(archive.map_count(24, 7), 3)

    def test_new_region_pays_full_while_toured_region_pays_little(self):
        from PoliwhiRL.environment.visit_archive import VisitArchive
        archive = VisitArchive()
        for _ in range(99):  # tour bank 24 map 7 hard
            archive.merge_visits([], [(24, 7)])
        rw = Rewards(
            _base_config(new_map_reward=5.0, new_bank_reward=20.0,
                         map_lifelong_decay=True),
            visit_archive=archive,
        )
        r_known = rw._new_map_bonus(_env_vars(map_bank=24, map_num=7))
        rw.start_new_episode()
        r_new = rw._new_map_bonus(_env_vars(map_bank=25, map_num=1))
        # Known map+bank both decayed (bank_count(24) == 99 here, one toured
        # map): 5/sqrt(100) + 20/sqrt(100) = 2.5. Brand-new bank 25 (count 0)
        # pays the full 5 + 20 = 25 — a 10x gradient toward new territory.
        self.assertAlmostEqual(r_known, 5.0 / 10 + 20.0 / 10, places=4)
        self.assertAlmostEqual(r_new, 25.0, places=4)
        self.assertGreater(r_new, r_known * 9)

    def test_flat_when_disabled(self):
        from PoliwhiRL.environment.visit_archive import VisitArchive
        archive = VisitArchive()
        for _ in range(99):
            archive.merge_visits([], [(24, 7)])
        rw = Rewards(
            _base_config(new_map_reward=5.0, new_bank_reward=0.0),  # decay off
            visit_archive=archive,
        )
        r = rw._new_map_bonus(_env_vars(map_bank=24, map_num=7))
        self.assertAlmostEqual(float(r), 5.0, places=4)  # no decay


class TestNewMapSeedingFromReplay(unittest.TestCase):
    def test_seeded_map_does_not_pay(self):
        rw = Rewards(_base_config(new_map_reward=25))
        rw.start_new_episode()
        rw.seed_explored_maps([(24, 7)])
        r0, _ = rw.calculate_reward(_env_vars(map_bank=24, map_num=7), "")
        self.assertEqual(float(r0), 0.0)
        # A genuinely new map still pays.
        r1, _ = rw.calculate_reward(_env_vars(map_bank=24, map_num=8), "")
        self.assertEqual(float(r1), 25.0)


class TestRAMObservationShape(unittest.TestCase):
    def test_no_target_features_present(self):
        for key in ("target_x", "target_y", "target_map",
                    "target_map_bank", "has_active_target"):
            self.assertNotIn(key, RAM_FEATURE_KEYS, f"{key} should be removed")

    def test_script_state_features_present(self):
        for key in (
            "script_active",
            "ui_state_walking_indoor", "ui_state_walking_outdoor",
            "ui_state_text_box", "ui_state_transition", "ui_state_other",
            "map_handler_indoor", "map_handler_outdoor",
            "map_handler_script_active", "map_handler_transition",
            "map_handler_other",
        ):
            self.assertIn(key, RAM_FEATURE_KEYS, f"{key} should be in feature list")

    def test_enemy_hp_present(self):
        self.assertIn("enemy_hp", RAM_FEATURE_KEYS)

    def test_no_location_progress_counter(self):
        self.assertNotIn("n_location_goals_completed", RAM_FEATURE_KEYS)
        self.assertIn("n_flag_goals_completed", RAM_FEATURE_KEYS)

    def test_feature_index_is_consistent(self):
        self.assertEqual(len(set(RAM_FEATURE_INDEX.values())), len(RAM_FEATURE_KEYS))
        for k in RAM_FEATURE_KEYS:
            self.assertIn(k, RAM_FEATURE_INDEX)


class TestMapsVisitedGoal(unittest.TestCase):
    def test_fires_once_per_unique_map_up_to_threshold(self):
        rw = Rewards(_base_config(goals=[{"type": "maps_visited", "threshold": 3}]))
        rw.calculate_reward(_env_vars(map_bank=24, map_num=1), button_press="")
        self.assertEqual(rw.goals.maps_visited_goals_completed, 1)
        rw.calculate_reward(_env_vars(map_bank=24, map_num=1), button_press="")
        self.assertEqual(rw.goals.maps_visited_goals_completed, 1)
        rw.calculate_reward(_env_vars(map_bank=24, map_num=2), button_press="")
        self.assertEqual(rw.goals.maps_visited_goals_completed, 2)
        rw.calculate_reward(_env_vars(map_bank=24, map_num=3), button_press="")
        self.assertEqual(rw.goals.maps_visited_goals_completed, 3)
        rw.calculate_reward(_env_vars(map_bank=24, map_num=4), button_press="")
        self.assertEqual(rw.goals.maps_visited_goals_completed, 3)
        self.assertEqual(rw.n_map_goals_completed(), 3)

    def test_runs_past_threshold_without_terminating(self):
        """No terminate_on_goal_complete: episode continues past threshold."""
        cfg = _base_config(
            episode_length=100,
            goals=[{"type": "maps_visited", "threshold": 2}],
        )
        rw = Rewards(cfg)
        _, done = rw.calculate_reward(_env_vars(map_bank=24, map_num=1), button_press="")
        self.assertFalse(done)
        _, done = rw.calculate_reward(_env_vars(map_bank=24, map_num=2), button_press="")
        self.assertFalse(done)  # threshold met but no early termination
        self.assertTrue(rw.goals.all_goal_thresholds_met())

    def test_requires_threshold(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            GoalsManager({"goals": [{"type": "maps_visited"}]})


class TestGoalsManagerParserRejection(unittest.TestCase):
    def test_location_goal_type_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown goal type"):
            GoalsManager({
                "goals": [{"type": "location", "positions": [[1, 2, 3]]}],
            })

    def test_flag_goal_requires_flag_num(self):
        with self.assertRaisesRegex(ValueError, "flag_num"):
            GoalsManager({"goals": [{"type": "flag"}]})


if __name__ == "__main__":
    unittest.main()
