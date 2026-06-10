# -*- coding: utf-8 -*-
"""Reward / goal coverage:

- ``flag`` goal type fires exactly once on a 0→1 transition and not on
  flags already true at episode start.
- ``pokedex_first_sight_reward`` pays on Δ seen; ``pokedex_owned_reward``
  on Δ owned. Both honour cross-episode preservation of the maxima.
- ``key_item_pickup_reward`` pays on Δ key_items_count.
- Frontier novelty pays ``bonus / (prior_count + 1)`` — smooth decay,
  no separate refire constant.
- ``terminate_on_goal_complete`` ends the episode when every configured
  goal has fired this episode.
- RAM observation carries script/UI state buckets and no longer carries
  ``target_*`` slots.
"""
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
        "pokedex_owned_reward": 0,
        "pokedex_first_sight_reward": 0,
        "key_item_pickup_reward": 0,
        "new_map_reward": 0,
        "new_map_first_discovery_reward": 0,
        "frontier_novelty_bonus": 0,
        "battle_engagement_reward": 0,
        "damage_dealt_reward": 0,
        "flag_progress_reward": 200,
        "whiteout_penalty": 0,
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
    }


# ----------------------------- tests --------------------------------- #

class TestFlagGoals(unittest.TestCase):
    def test_flag_fires_on_zero_to_one_transition(self):
        rw = Rewards(_base_config(goals=[{"type": "flag", "flag_num": 25}]))
        # Initial step with flag at 0 — seeds initial state, no fire.
        r0, _ = rw.calculate_reward(_env_vars(), button_press="")
        self.assertEqual(rw.flag_goals_completed, 0)
        self.assertEqual(float(r0), 0.0)

        # Step with the flag now set — should fire.
        flags = _set_flag(_zero_flags(), 25)
        r1, _ = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        self.assertEqual(rw.flag_goals_completed, 1)
        self.assertEqual(float(r1), 200.0)

        # Step again with the flag still set — should NOT fire a second time.
        r2, _ = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        self.assertEqual(rw.flag_goals_completed, 1)
        self.assertEqual(float(r2), 0.0)

    def test_flag_already_set_at_episode_start_does_not_fire(self):
        """Flags already true via replay must not pay — only fresh 0→1
        transitions count toward the agent's training reward."""
        rw = Rewards(_base_config(goals=[{"type": "flag", "flag_num": 25}]))
        flags = _set_flag(_zero_flags(), 25)
        # First call seeds the initial-state snapshot; flag is already 1.
        r0, _ = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        self.assertEqual(rw.flag_goals_completed, 0)
        self.assertEqual(float(r0), 0.0)
        # Subsequent steps with the same value also don't fire.
        r1, _ = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        self.assertEqual(rw.flag_goals_completed, 0)
        self.assertEqual(float(r1), 0.0)


class TestTerminateOnGoalComplete(unittest.TestCase):
    def test_terminate_on_legacy_key_rejected(self):
        cfg = _base_config(terminate_on={"type": "flag", "flag_num": 25})
        with self.assertRaisesRegex(ValueError, "terminate_on.*no longer supported"):
            Rewards(cfg)

    def test_default_runs_to_budget(self):
        """Without ``terminate_on_goal_complete`` the episode must run to
        episode_length even after the flag fires."""
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
        # 4th step exceeds budget — terminates.
        _, done = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        _, done = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        self.assertTrue(done)

    def test_terminate_on_goal_complete_ends_early(self):
        """With ``terminate_on_goal_complete: true`` the episode ends on
        the step the last configured goal fires."""
        cfg = _base_config(
            episode_length=100,
            terminate_on_goal_complete=True,
            goals=[{"type": "flag", "flag_num": 25}],
        )
        rw = Rewards(cfg)
        # Step 1: flag still 0, episode continues.
        _, done = rw.calculate_reward(_env_vars(), button_press="")
        self.assertFalse(done)
        # Step 2: flag fires, episode ends.
        flags = _set_flag(_zero_flags(), 25)
        _, done = rw.calculate_reward(_env_vars(story_flags=flags), button_press="")
        self.assertTrue(done)


class TestAdditiveMilestoneNoTerminate(unittest.TestCase):
    def test_goal_met_without_terminating(self):
        """Stages 2-5 run with terminate_on_goal_complete=False: reaching the
        goal pays its reward and is recorded as success (all_goal_thresholds_met
        → True, for best/ selection) but does NOT end the episode, so the agent
        keeps exploring beyond the milestone."""
        cfg = _base_config(
            terminate_on_goal_complete=False,
            pokedex_owned_reward=150,
            goals=[{"type": "pokedex", "kind": "owned", "threshold": 1}],
        )
        rw = Rewards(cfg)
        # Before the starter: not satisfied, episode continues.
        _, done = rw.calculate_reward(_env_vars(pokedex_owned=0), button_press="")
        self.assertFalse(done)
        self.assertFalse(rw.goals.all_goal_thresholds_met())
        # Obtain the starter: pays 150, episode does NOT end, success recorded.
        r, done = rw.calculate_reward(_env_vars(pokedex_owned=1), button_press="")
        self.assertEqual(r, 150)
        self.assertFalse(done)
        self.assertTrue(rw.goals.all_goal_thresholds_met())
        # Keep exploring afterwards — still not done, success stays recorded.
        _, done = rw.calculate_reward(_env_vars(pokedex_owned=1), button_press="")
        self.assertFalse(done)
        self.assertTrue(rw.goals.all_goal_thresholds_met())


class TestMapGoal(unittest.TestCase):
    def test_map_goal_fires_on_entry_and_terminates(self):
        """A ``map`` goal pays map_goal_reward on entering the target
        (bank, num) at any coordinate, and ends the episode."""
        cfg = _base_config(
            terminate_on_goal_complete=True,
            map_goal_reward=250,
            goals=[{"type": "map", "map_bank": 24, "map_num": 3}],
        )
        rw = Rewards(cfg)
        # Start on (24, 7): not the target, episode continues.
        r, done = rw.calculate_reward(
            _env_vars(map_bank=24, map_num=7), button_press=""
        )
        self.assertFalse(done)
        # Enter the target map at a different x/y — coordinate-independent.
        r, done = rw.calculate_reward(
            _env_vars(map_bank=24, map_num=3, x=99, y=1), button_press=""
        )
        self.assertEqual(r, 250)
        self.assertTrue(done)

    def test_map_goal_not_met_when_starting_on_target(self):
        """Starting on the target map achieves nothing, so it must NOT
        terminate the episode (avoids zero-length training segments). Per-
        stage state pools must not start the agent on the stage's target."""
        cfg = _base_config(
            terminate_on_goal_complete=True,
            map_goal_reward=250,
            goals=[{"type": "map", "map_bank": 24, "map_num": 7}],
        )
        rw = Rewards(cfg)
        r, done = rw.calculate_reward(
            _env_vars(map_bank=24, map_num=7), button_press=""
        )
        self.assertEqual(r, 0)
        self.assertFalse(done)


class TestPokedexRewards(unittest.TestCase):
    def test_first_sight_pays_per_delta(self):
        rw = Rewards(_base_config(pokedex_first_sight_reward=10))
        # Seed at 0.
        r0, _ = rw.calculate_reward(_env_vars(pokedex_seen=0), "")
        self.assertEqual(float(r0), 0.0)
        # Δ seen = +2 ⇒ 20.
        r1, _ = rw.calculate_reward(_env_vars(pokedex_seen=2), "")
        self.assertEqual(float(r1), 20.0)
        # No further Δ ⇒ 0.
        r2, _ = rw.calculate_reward(_env_vars(pokedex_seen=2), "")
        self.assertEqual(float(r2), 0.0)

    def test_owned_pays_per_delta(self):
        rw = Rewards(_base_config(pokedex_owned_reward=150))
        rw.calculate_reward(_env_vars(pokedex_owned=0), "")
        r1, _ = rw.calculate_reward(_env_vars(pokedex_owned=1), "")
        self.assertEqual(float(r1), 150.0)
        # No further Δ ⇒ 0.
        r2, _ = rw.calculate_reward(_env_vars(pokedex_owned=1), "")
        self.assertEqual(float(r2), 0.0)


class TestKeyItemPickup(unittest.TestCase):
    def test_pickup_pays_per_delta(self):
        rw = Rewards(_base_config(key_item_pickup_reward=5))
        # Seed at 0.
        rw.calculate_reward(_env_vars(key_items_count=0), "")
        # +5 items at once (e.g. Pokéball x5 from Elm).
        r1, _ = rw.calculate_reward(_env_vars(key_items_count=5), "")
        self.assertEqual(float(r1), 25.0)
        # No further Δ.
        r2, _ = rw.calculate_reward(_env_vars(key_items_count=5), "")
        self.assertEqual(float(r2), 0.0)

    def test_pickup_does_not_pay_on_decrement(self):
        """Using an item lowers the count but mustn't refund reward."""
        rw = Rewards(_base_config(key_item_pickup_reward=5))
        rw.calculate_reward(_env_vars(key_items_count=5), "")
        r, _ = rw.calculate_reward(_env_vars(key_items_count=4), "")
        self.assertEqual(float(r), 0.0)


class TestFrontierNovelty(unittest.TestCase):
    def test_first_visit_pays_full_bonus(self):
        rw = Rewards(_base_config(frontier_novelty_bonus=5.0))
        r0, _ = rw.calculate_reward(_env_vars(x=4, y=3), button_press="")
        # prior_count == 0 ⇒ 5 / 1 = 5.
        self.assertEqual(float(r0), 5.0)

    def test_same_cell_same_episode_no_repeat(self):
        rw = Rewards(_base_config(frontier_novelty_bonus=5.0))
        rw.calculate_reward(_env_vars(x=4, y=3), button_press="")
        # x stays in the same quantised cell (4 // 2 == 5 // 2 == 2).
        r1, _ = rw.calculate_reward(_env_vars(x=5, y=3), button_press="")
        self.assertEqual(float(r1), 0.0)

    def test_depletes_across_episodes(self):
        """Frontier novelty is PERSISTENT (count-based): the same cell pays
        less every episode it is re-visited, because the run-wide visit count
        in the shared archive grows. This depletion is what makes the policy
        treat known ground as exhausted and push the frontier outward.
        (See test_persistent_novelty.py for the full contract.)"""
        from PoliwhiRL.environment.visit_archive import VisitArchive
        archive = VisitArchive()  # shared across episodes
        expected = [5.0 / 1, 5.0 / 2, 5.0 / 3]
        for exp in expected:
            rw = Rewards(
                _base_config(frontier_novelty_bonus=5.0),
                visit_archive=archive,
            )
            r, _ = rw.calculate_reward(_env_vars(x=4, y=3), button_press="")
            self.assertAlmostEqual(float(r), exp, places=4)
        # The genuine training visits are now recorded in the archive.
        self.assertEqual(archive.count(24, 7, 4, 3), 3)

    def test_blocked_during_script_active(self):
        """Frontier novelty is gated on script_active=False — during cutscenes
        and menus, player position is stale so paying a frontier bonus would be noise."""
        ev = _env_vars(x=4, y=3)
        ev["script_active"] = True
        rw = Rewards(_base_config(frontier_novelty_bonus=5.0))
        r0, _ = rw.calculate_reward(ev, button_press="")
        self.assertEqual(float(r0), 0.0)

    def test_fires_when_script_inactive(self):
        """Frontier novelty fires normally when not in a script."""
        ev = _env_vars(x=4, y=3)
        ev["script_active"] = False
        rw = Rewards(_base_config(frontier_novelty_bonus=5.0))
        r0, _ = rw.calculate_reward(ev, button_press="")
        self.assertEqual(float(r0), 5.0)


class TestNewMapSeedingFromReplay(unittest.TestCase):
    def test_seeded_map_does_not_pay(self):
        """seed_explored_maps emulates the action-replay walking through a
        map: the training segment's first step on that map must not fire
        the new_map bonus (flat legacy reward path)."""
        rw = Rewards(_base_config(new_map_reward=25))
        rw.start_new_episode()
        rw.seed_explored_maps([(24, 7)])
        r0, _ = rw.calculate_reward(_env_vars(map_bank=24, map_num=7), "")
        self.assertEqual(float(r0), 0.0)
        # A genuinely new map still pays.
        r1, _ = rw.calculate_reward(_env_vars(map_bank=24, map_num=8), "")
        self.assertEqual(float(r1), 25.0)


class TestNewMapFirstDiscovery(unittest.TestCase):
    def test_first_discovery_decays_by_global_count(self):
        """The first-discovery bonus pays bonus/(global_count+1) and the
        global ledger increments per training entry, so re-discovering a map
        across episodes pays progressively less (anti map-bouncing)."""
        from PoliwhiRL.environment.visit_archive import VisitArchive
        archive = VisitArchive()
        for expected in (50.0, 25.0, 50.0 / 3.0):
            rw = Rewards(
                _base_config(new_map_first_discovery_reward=50),
                visit_archive=archive,
            )
            rw.start_new_episode()
            r, _ = rw.calculate_reward(_env_vars(map_bank=26, map_num=3), "")
            self.assertAlmostEqual(float(r), expected, places=4)

    def test_replay_does_not_pump_ledger(self):
        """With _replaying=True the discovery bonus still computes but the
        global ledger is NOT written, so the corridor the replay walks every
        episode can't drain the first-discovery bonus."""
        from PoliwhiRL.environment.visit_archive import VisitArchive
        archive = VisitArchive()
        rw = Rewards(
            _base_config(new_map_first_discovery_reward=50),
            visit_archive=archive,
        )
        rw._replaying = True
        rw.start_new_episode()
        rw.calculate_reward(_env_vars(map_bank=26, map_num=3), "")
        self.assertEqual(archive.map_count(26, 3), 0)
        # After replay ends, a genuine training entry pays full and records.
        rw._replaying = False
        rw.start_new_episode()
        r, _ = rw.calculate_reward(_env_vars(map_bank=26, map_num=3), "")
        self.assertAlmostEqual(float(r), 50.0, places=4)
        self.assertEqual(archive.map_count(26, 3), 1)


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
        """Policy must see enemy HP so it can choose Catch vs Attack."""
        self.assertIn("enemy_hp", RAM_FEATURE_KEYS)

    def test_no_location_progress_counter(self):
        self.assertNotIn("n_location_goals_completed", RAM_FEATURE_KEYS)
        self.assertIn("n_flag_goals_completed", RAM_FEATURE_KEYS)

    def test_feature_index_is_consistent(self):
        self.assertEqual(len(set(RAM_FEATURE_INDEX.values())), len(RAM_FEATURE_KEYS))
        for k in RAM_FEATURE_KEYS:
            self.assertIn(k, RAM_FEATURE_INDEX)


class TestMapsVisitedGoal(unittest.TestCase):
    """Stage-1 "visit N unique maps" breadth goal."""

    def test_fires_once_per_unique_map_up_to_threshold(self):
        rw = Rewards(_base_config(goals=[{"type": "maps_visited", "threshold": 3}]))
        # Map 1 (the starting map) — first unique map, one fire.
        rw.calculate_reward(_env_vars(map_bank=24, map_num=1), button_press="")
        self.assertEqual(rw.goals.maps_visited_goals_completed, 1)
        # Re-visiting the same map does not fire again.
        rw.calculate_reward(_env_vars(map_bank=24, map_num=1), button_press="")
        self.assertEqual(rw.goals.maps_visited_goals_completed, 1)
        # Map 2.
        rw.calculate_reward(_env_vars(map_bank=24, map_num=2), button_press="")
        self.assertEqual(rw.goals.maps_visited_goals_completed, 2)
        # Map 3 — threshold reached.
        rw.calculate_reward(_env_vars(map_bank=24, map_num=3), button_press="")
        self.assertEqual(rw.goals.maps_visited_goals_completed, 3)
        # A 4th distinct map does not over-count past the threshold.
        rw.calculate_reward(_env_vars(map_bank=24, map_num=4), button_press="")
        self.assertEqual(rw.goals.maps_visited_goals_completed, 3)
        # Folded into the map-goal metric for plotting / terminal info.
        self.assertEqual(rw.n_map_goals_completed(), 3)

    def test_terminate_on_goal_complete_ends_at_threshold(self):
        cfg = _base_config(
            episode_length=100,
            terminate_on_goal_complete=True,
            goals=[{"type": "maps_visited", "threshold": 3}],
        )
        rw = Rewards(cfg)
        _, done = rw.calculate_reward(_env_vars(map_bank=24, map_num=1), button_press="")
        self.assertFalse(done)
        _, done = rw.calculate_reward(_env_vars(map_bank=24, map_num=2), button_press="")
        self.assertFalse(done)
        # Third unique map this step — last threshold met, episode ends here.
        _, done = rw.calculate_reward(_env_vars(map_bank=24, map_num=3), button_press="")
        self.assertTrue(done)
        self.assertFalse(rw.truncated)

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
