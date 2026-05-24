# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
import numpy as np
from PoliwhiRL.reward_evaluation import evaluate_reward_system
from PoliwhiRL.environment.rewards import Rewards, is_ram_state_valid
from main import load_default_config, load_user_config, merge_configs


def _goals(*items):
    """Build a `goals` list from typed goal dicts (new format)."""
    return list(items)


def _loc(positions, hard=True):
    """Shortcut for a location goal."""
    return {"type": "location", "positions": positions, "hard": hard}


def _pokedex(kind, threshold):
    """Shortcut for a pokedex goal."""
    return {"type": "pokedex", "kind": kind, "threshold": threshold}


def _level(threshold):
    """Shortcut for a level goal."""
    return {"type": "level", "kind": "total_level", "threshold": threshold}


def _xp(threshold, xp_per_fire=10):
    """Shortcut for an xp goal."""
    return {"type": "xp", "kind": "total_xp", "threshold": threshold, "xp_per_fire": xp_per_fire}


class TestRewardSystem(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_loc = "./configs/evaluate_reward_system.json"
        self.config = load_default_config()
        self.user_config = load_user_config(self.config_loc)
        self.config = merge_configs(self.config, self.user_config)
        self.config["output_path"] = self.temp_dir

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_reward_system_functionality_1_goal(self):
        """Test that reward system works correctly with 1 goal"""
        self.config["hard_goal_count_target"] = 1
        rewards = evaluate_reward_system(self.config)
        # Test that rewards are computed (non-empty list)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        # Test that all rewards are finite numbers
        self.assertTrue(all(np.isfinite(r) for r in rewards))

    def test_reward_system_functionality_2_goals(self):
        """Test that reward system works correctly with 2 goals"""
        self.config["hard_goal_count_target"] = 2
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))

    def test_reward_system_functionality_3_goals(self):
        """Test that reward system works correctly with 3 goals"""
        self.config["hard_goal_count_target"] = 3
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        # Test that negative total is reasonable for inefficient play
        total_reward = np.sum(rewards)
        self.assertGreater(total_reward, -1000)

    def test_reward_system_functionality_4_goals(self):
        """Test that reward system works correctly with 4 goals"""
        self.config["hard_goal_count_target"] = 4
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        total_reward = np.sum(rewards)
        self.assertGreater(total_reward, -1000)

    def test_reward_system_functionality_5_goals(self):
        """Test that reward system works correctly with 5 goals"""
        self.config["hard_goal_count_target"] = 5
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        total_reward = np.sum(rewards)
        self.assertGreater(total_reward, -1000)

    def test_reward_system_functionality_6_goals(self):
        """Test that reward system works correctly with 6 goals"""
        self.config["hard_goal_count_target"] = 6
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        total_reward = np.sum(rewards)
        self.assertGreater(total_reward, -1000)

    def test_reward_system_functionality_7_goals(self):
        """Test that reward system works correctly with 7 goals"""
        self.config["hard_goal_count_target"] = 7
        rewards = evaluate_reward_system(self.config)
        self.assertIsInstance(rewards, list)
        self.assertGreater(len(rewards), 0)
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        total_reward = np.sum(rewards)
        self.assertGreater(total_reward, -1000)

    def test_reward_scaling_and_clipping(self):
        """Test that rewards are properly scaled and clipped"""
        config = load_default_config()
        config["hard_goal_count_target"] = 1
        config["episode_length"] = 100
        config["goals"] = [_loc([[1, 1, 1]])]

        reward_system = Rewards(config)

        self.assertEqual(reward_system.clip, 1000)

        self.assertLessEqual(reward_system.step_penalty, 0)
        self.assertGreaterEqual(reward_system.step_penalty, -1.0)

        self.assertGreater(reward_system.goal_reward, 0)
        self.assertGreater(reward_system.sequence_bonus, 0)
        self.assertGreater(reward_system.checkpoint_bonus, 0)
        self.assertGreater(reward_system.all_goals_bonus, 0)

    def test_reward_goal_achievement(self):
        """Test that goal achievement produces positive rewards"""
        config = load_default_config()
        config["hard_goal_count_target"] = 1
        config["episode_length"] = 1000
        config["goals"] = [_loc([[5, 5, 1]])]

        reward_system = Rewards(config)

        # Simulate reaching the goal location
        env_vars = {
            "X": 5,
            "Y": 5,
            "map_num": 1,
            "map_bank": 0,
            "room": 0,
            "pokedex_seen": 0,
            "pokedex_owned": 0,
            "party_info": (1, 5, 20, 0),
        }

        reward, done = reward_system.calculate_reward(env_vars, "A")

        # Should get a positive reward for achieving goal
        self.assertGreater(reward, 0)
        # Should complete the episode since break_on_goal defaults to True
        self.assertTrue(done)

    def test_reward_step_penalty_is_constant(self):
        """Step penalty is fixed across an episode (no time dependency)."""
        config = load_default_config()
        config["hard_goal_count_target"] = 1
        config["episode_length"] = 100
        config["punish_steps"] = True
        config["goals"] = [_loc([[999, 999, 999]])]  # Unreachable goal

        reward_system = Rewards(config)

        env_vars = {
            "X": 1,
            "Y": 1,
            "map_num": 1,
            "map_bank": 0,
            "room": 0,
            "pokedex_seen": 0,
            "pokedex_owned": 0,
            "party_info": (1, 5, 20, 0),
        }

        early_reward, _ = reward_system.calculate_reward(env_vars, "A")
        for _ in range(40):
            reward_system.calculate_reward(env_vars, "A")
        mid_reward, _ = reward_system.calculate_reward(env_vars, "A")

        self.assertEqual(mid_reward, early_reward)
        self.assertEqual(float(early_reward), float(reward_system.step_penalty))

    def test_reward_step_penalty_disabled(self):
        """punish_steps=False zeroes out the per-step penalty."""
        config = load_default_config()
        config["hard_goal_count_target"] = 1
        config["episode_length"] = 100
        config["punish_steps"] = False
        config["goals"] = [_loc([[999, 999, 999]])]

        reward_system = Rewards(config)
        self.assertEqual(reward_system.step_penalty, 0)

        env_vars = {
            "X": 1,
            "Y": 1,
            "map_num": 1,
            "map_bank": 0,
            "room": 0,
            "pokedex_seen": 0,
            "pokedex_owned": 0,
            "party_info": (1, 5, 20, 0),
        }
        reward, _ = reward_system.calculate_reward(env_vars, "A")
        self.assertEqual(float(reward), 0.0)


class TestRewardsBranches(unittest.TestCase):
    """Direct unit tests on Rewards class branches not exercised end-to-end."""

    def _base_config(self, goals=None):
        config = load_default_config()
        config["episode_length"] = 100
        config["punish_steps"] = False  # isolate goal/bonus rewards
        if goals is not None:
            config["goals"] = goals
        else:
            config["goals"] = [_loc([[999, 999, 999]])]
        return config

    def _env_vars(self, x=1, y=1, map_num=1, room=0, seen=0, owned=0,
                  party_level=5, party_exp=0, party_size=1, battle_type=0,
                  enemy_hp=0, enemy_max_hp=0, map_bank=0, player_state=0):
        return {
            "X": x,
            "Y": y,
            "map_num": map_num,
            "map_bank": map_bank,
            "room": room,
            "pokedex_seen": seen,
            "pokedex_owned": owned,
            "party_info": (party_size, party_level, 20, party_exp),
            "battle_type": battle_type,
            "enemy_hp": enemy_hp,
            "enemy_max_hp": enemy_max_hp,
            "player_state": player_state,
        }

    def test_sequential_goals_must_be_in_order(self):
        """Reaching goal 2 before goal 1 gives no reward."""
        config = self._base_config(goals=[_loc([[1, 1, 1]]), _loc([[2, 2, 1]])])
        config["hard_goal_count_target"] = 2
        config["require_sequential"] = True

        rewards = Rewards(config)

        # Land on goal 2 first — should not advance.
        reward, _ = rewards.calculate_reward(self._env_vars(x=2, y=2), "a")
        self.assertEqual(float(reward), 0.0)
        self.assertEqual(rewards.N_goals, 0)

        # Land on goal 1 — should advance.
        reward, _ = rewards.calculate_reward(self._env_vars(x=1, y=1), "a")
        self.assertGreater(float(reward), 0)
        self.assertEqual(rewards.N_goals, 1)

    def test_sequential_bonus_added(self):
        config = self._base_config(goals=[
            _loc([[1, 1, 1]]),
            _loc([[2, 2, 1]]),
            _loc([[3, 3, 1]]),
            _loc([[4, 4, 1]]),
            _loc([[5, 5, 1]]),
        ])
        config["hard_goal_count_target"] = 5  # avoid all-goals bonus on first hit
        config["require_sequential"] = True
        config["checkpoint_goals"] = [99]  # avoid checkpoint bonus

        rewards = Rewards(config)
        reward, _ = rewards.calculate_reward(self._env_vars(x=1, y=1), "a")
        expected = rewards.goal_reward + rewards.sequence_bonus
        self.assertEqual(float(reward), float(expected))

    def test_checkpoint_bonus_triggers_at_milestone(self):
        config = self._base_config(goals=[
            _loc([[1, 1, 1]]),
            _loc([[2, 2, 1]]),
            _loc([[3, 3, 1]]),
            _loc([[4, 4, 1]]),
            _loc([[5, 5, 1]]),
        ])
        config["hard_goal_count_target"] = 5
        config["require_sequential"] = False
        config["checkpoint_goals"] = [2]

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(x=1, y=1), "a")  # goal 1
        reward, _ = rewards.calculate_reward(self._env_vars(x=2, y=2), "a")  # goal 2

        # Second goal should include the checkpoint bonus.
        self.assertGreaterEqual(float(reward), float(rewards.checkpoint_bonus))

    def test_all_goals_bonus_and_break_on_goal(self):
        config = self._base_config(goals=[_loc([[1, 1, 1]])])
        config["hard_goal_count_target"] = 1
        config["break_on_goal"] = True
        config["checkpoint_goals"] = [99]

        rewards = Rewards(config)
        reward, done = rewards.calculate_reward(self._env_vars(x=1, y=1), "a")
        self.assertTrue(done)
        # Reward should include goal + sequence + all_goals bonus.
        self.assertGreaterEqual(
            float(reward),
            float(rewards.goal_reward + rewards.all_goals_bonus),
        )

    def test_button_penalty_for_start_select(self):
        config = self._base_config()
        config["hard_goal_count_target"] = 1

        rewards = Rewards(config)
        r_start, _ = rewards.calculate_reward(self._env_vars(), "start")
        rewards2 = Rewards(config)
        r_select, _ = rewards2.calculate_reward(self._env_vars(), "select")
        rewards3 = Rewards(config)
        r_a, _ = rewards3.calculate_reward(self._env_vars(), "a")

        self.assertEqual(float(r_start), float(rewards.button_penalty))
        self.assertEqual(float(r_select), float(rewards2.button_penalty))
        self.assertEqual(float(r_a), 0.0)

    def test_exploration_reward_first_visit_only(self):
        config = self._base_config()
        config["hard_goal_count_target"] = 1
        config["exploration_reward"] = 1.0

        rewards = Rewards(config)
        r1, _ = rewards.calculate_reward(self._env_vars(x=5, y=5), "a")
        r2, _ = rewards.calculate_reward(self._env_vars(x=5, y=5), "a")
        r3, _ = rewards.calculate_reward(self._env_vars(x=6, y=5), "a")

        self.assertEqual(float(r1), 1.0)
        self.assertEqual(float(r2), 0.0)
        self.assertEqual(float(r3), 1.0)

    def test_pokedex_seen_reward(self):
        config = self._base_config()
        config["hard_goal_count_target"] = 99  # don't trigger all-goals

        rewards = Rewards(config)
        # Bump seen from 0 -> 3 in one step.
        r, _ = rewards.calculate_reward(self._env_vars(seen=3), "a")
        self.assertEqual(float(r), float(rewards.pokedex_seen_reward))

        # Subsequent step with same seen count — no extra pokedex reward.
        r2, _ = rewards.calculate_reward(self._env_vars(seen=3), "a")
        self.assertEqual(float(r2), 0.0)

    def test_pokedex_goal_threshold_increments_n_goals(self):
        config = self._base_config(goals=[
            _loc([[999, 999, 999]]),
            _pokedex("seen", 2),
        ])
        config["hard_goal_count_target"] = 99

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(seen=1), "a")
        # Each integer increment toward the threshold counts as a goal fire.
        self.assertEqual(rewards.N_goals, 1)
        rewards.calculate_reward(self._env_vars(seen=2), "a")
        self.assertEqual(rewards.N_goals, 2)
        # Threshold fully consumed: entry removed.
        self.assertNotIn("seen", rewards.pokedex_goals)

    def test_pokedex_goal_multi_fire_single_step(self):
        config = self._base_config(goals=[
            _loc([[999, 999, 999]]),
            _pokedex("seen", 3),
        ])
        config["hard_goal_count_target"] = 99

        rewards = Rewards(config)
        # Jumping from 0 -> 3 in one step should fire all 3 goal slots.
        rewards.calculate_reward(self._env_vars(seen=3), "a")
        self.assertEqual(rewards.N_goals, 3)
        self.assertNotIn("seen", rewards.pokedex_goals)

    def test_pokedex_goal_multi_fire_caps_at_threshold(self):
        config = self._base_config(goals=[
            _loc([[999, 999, 999]]),
            _pokedex("seen", 2),
        ])
        config["hard_goal_count_target"] = 99

        rewards = Rewards(config)
        # Overshooting the threshold (0 -> 5) caps at threshold (2 fires).
        rewards.calculate_reward(self._env_vars(seen=5), "a")
        self.assertEqual(rewards.N_goals, 2)
        self.assertNotIn("seen", rewards.pokedex_goals)

    def test_reward_clipping(self):
        config = self._base_config(goals=[_loc([[1, 1, 1]])])
        config["hard_goal_count_target"] = 1
        config["goal_reward"] = 100000
        config["all_goals_bonus"] = 100000

        rewards = Rewards(config)
        r, _ = rewards.calculate_reward(self._env_vars(x=1, y=1), "a")
        self.assertEqual(float(r), float(rewards.clip))

    def test_goal_dict_format(self):
        """Dict-format position is normalised correctly."""
        config = self._base_config(goals=[_loc([{"x": 3, "y": 4, "map": 5}])])
        config["hard_goal_count_target"] = 1

        rewards = Rewards(config)
        # Dict format should have been normalised.
        goal = list(rewards.location_goals.values())[0]
        self.assertEqual(goal["positions"], [[3, 4, None, 5]])
        self.assertFalse(goal["check_bank"])

        # And the goal should be reachable as a location.
        r, _ = rewards.calculate_reward(self._env_vars(x=3, y=4, map_num=5), "a")
        self.assertGreater(float(r), 0)

    def test_distance_shaping_runs_when_enabled(self):
        """Regression: _distance_shaping used to unpack the position tuple
        with `target_x, _, target_bank, target_map = ...` and then read
        `target_y`, which raised NameError as soon as any user enabled
        distance shaping with a target on the same map as the player.
        """
        config = self._base_config(goals=[_loc([[8, 5, 1]])])
        config["hard_goal_count_target"] = 1
        config["distance_shaping_coef"] = 1.0

        rewards = Rewards(config)
        # First call seeds _d_prev (no shaping yet). Should not raise.
        r1, _ = rewards.calculate_reward(self._env_vars(x=5, y=5, map_num=1), "a")
        # Second call gets closer (5 -> 6 on x axis). Should produce a
        # positive shaping bonus, not raise.
        r2, _ = rewards.calculate_reward(self._env_vars(x=6, y=5, map_num=1), "a")
        self.assertGreaterEqual(float(r2), 1.0)

    def test_goal_rejects_unknown_format(self):
        config = self._base_config()
        config["hard_goal_count_target"] = 1
        config["goals"] = [{"type": "location", "positions": [[42]]}]
        with self.assertRaises(ValueError):
            Rewards(config)

    def test_party_level_reward_fires_on_increase(self):
        """party_level_reward fires when total party level increases."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["party_level_reward"] = 10

        rewards = Rewards(config)
        # Step 1 seeds (no reward).
        r, _ = rewards.calculate_reward(self._env_vars(party_level=5), "a")
        self.assertEqual(float(r), 0.0)

        # Level goes from 5 → 8 (3 levels gained).
        r, _ = rewards.calculate_reward(self._env_vars(party_level=8), "a")
        self.assertEqual(float(r), 30.0)

    def test_party_level_reward_no_decrease(self):
        """No reward when party level decreases (e.g. swapping out a Pokemon)."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["party_level_reward"] = 10

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(party_level=10), "a")  # seed
        r, _ = rewards.calculate_reward(self._env_vars(party_level=5), "a")
        self.assertEqual(float(r), 0.0)

    def test_party_exp_reward_fires_on_increase(self):
        """party_exp_reward fires when total party EXP increases."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["party_exp_reward"] = 0.01

        rewards = Rewards(config)
        # Step 1 seeds (no reward).
        r, _ = rewards.calculate_reward(self._env_vars(party_exp=100), "a")
        self.assertEqual(float(r), 0.0)

        # EXP goes from 100 → 250 (150 gained).
        r, _ = rewards.calculate_reward(self._env_vars(party_exp=250), "a")
        self.assertEqual(float(r), 1.5)

    def test_party_exp_reward_no_decrease(self):
        """No reward when party EXP decreases."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["party_exp_reward"] = 0.01

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(party_exp=500), "a")  # seed
        r, _ = rewards.calculate_reward(self._env_vars(party_exp=200), "a")
        self.assertEqual(float(r), 0.0)

    def test_party_rewards_disabled_by_default(self):
        """When config values are 0 (default), no party progress reward fires."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        # party_level_reward and party_exp_reward default to 0

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(party_level=5, party_exp=0), "a")  # seed
        r, _ = rewards.calculate_reward(
            self._env_vars(party_level=20, party_exp=500), "a"
        )
        self.assertEqual(float(r), 0.0)

    def test_party_progress_resets_on_new_episode(self):
        """_prev_party_level/exp are reset on start_new_episode so the
        training episode re-seeds from step 0."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["party_level_reward"] = 10

        rewards = Rewards(config)
        # Simulate a replay episode: level goes 5 → 10.
        rewards.calculate_reward(self._env_vars(party_level=5), "a")  # seed
        rewards.calculate_reward(self._env_vars(party_level=10), "a")  # +50

        # Now start_new_episode (e.g. after replay).
        rewards.start_new_episode()

        # New episode seeds from current level (10), so no phantom reward.
        r, _ = rewards.calculate_reward(self._env_vars(party_level=10), "a")
        self.assertEqual(float(r), 0.0)

        # But a genuine increase still fires.
        r, _ = rewards.calculate_reward(self._env_vars(party_level=12), "a")
        self.assertEqual(float(r), 20.0)

    def test_party_size_change_suppresses_reward(self):
        """When party size changes, reward is suppressed to avoid compound
        bonuses from captures / box swaps."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["party_exp_reward"] = 0.1

        rewards = Rewards(config)
        # Seed with party of 1.
        rewards.calculate_reward(self._env_vars(party_size=1, party_exp=0), "a")

        # Party grows to 2 with extra EXP — reward should be suppressed.
        r, _ = rewards.calculate_reward(
            self._env_vars(party_size=2, party_exp=100), "a"
        )
        self.assertEqual(float(r), 0.0)

    def test_party_reward_check_battle_allows_in_battle(self):
        """When party_reward_check_battle is True and battle_type != 0,
        rewards are allowed even on party size change."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["party_exp_reward"] = 0.1
        config["party_reward_check_battle"] = True

        rewards = Rewards(config)
        # Seed with party of 1, not in battle.
        rewards.calculate_reward(
            self._env_vars(party_size=1, party_exp=0), "a"
        )

        # Party grows to 2 with extra EXP while in battle (battle_type=1).
        r, _ = rewards.calculate_reward(
            self._env_vars(party_size=2, party_exp=100, battle_type=1), "a"
        )
        # Reward should fire: 100 * 0.1 = 10.
        self.assertEqual(float(r), 10.0)

    def test_party_reward_check_battle_suppresses_outside_battle(self):
        """When party_reward_check_battle is True but battle_type == 0,
        rewards are still suppressed on party size change."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["party_exp_reward"] = 0.1
        config["party_reward_check_battle"] = True

        rewards = Rewards(config)
        # Seed with party of 1.
        rewards.calculate_reward(
            self._env_vars(party_size=1, party_exp=0), "a"
        )

        # Party grows to 2 with extra EXP while NOT in battle.
        r, _ = rewards.calculate_reward(
            self._env_vars(party_size=2, party_exp=100), "a"
        )
        self.assertEqual(float(r), 0.0)

    # ---- XP milestone reward tests ----

    def test_xp_milestone_disabled_by_default(self):
        """xp_milestone_threshold defaults to 0 (disabled)."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99

        rewards = Rewards(config)
        self.assertEqual(rewards.xp_milestone_threshold, 0)
        self.assertEqual(rewards.xp_milestone_reward, 0)

    def test_xp_milestone_fires_on_threshold(self):
        """xp_milestone_reward fires once per threshold of cumulative XP."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["xp_milestone_threshold"] = 100
        config["xp_milestone_reward"] = 25

        rewards = Rewards(config)
        # Step 1 seeds (no reward).
        r, _ = rewards.calculate_reward(self._env_vars(party_exp=0), "a")
        self.assertEqual(float(r), 0.0)

        # Gain 90 XP — below threshold, no milestone.
        r, _ = rewards.calculate_reward(self._env_vars(party_exp=90), "a")
        self.assertEqual(float(r), 0.0)

        # Gain 20 more XP (total 110) — crosses threshold once.
        r, _ = rewards.calculate_reward(self._env_vars(party_exp=110), "a")
        self.assertEqual(float(r), 25.0)

    def test_xp_milestone_multi_fire(self):
        """Multiple milestones fire if enough XP gained in one step."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["xp_milestone_threshold"] = 100
        config["xp_milestone_reward"] = 25

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(party_exp=0), "a")  # seed

        # Gain 250 XP in one step — crosses threshold twice (200), 50 left over.
        r, _ = rewards.calculate_reward(self._env_vars(party_exp=250), "a")
        self.assertEqual(float(r), 50.0)  # 2 × 25

    def test_xp_milestone_suppressed_on_party_size_change(self):
        """XP milestone accumulator pauses when party size changes (e.g. capture)."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["xp_milestone_threshold"] = 100
        config["xp_milestone_reward"] = 25

        rewards = Rewards(config)
        # Seed with party of 1.
        rewards.calculate_reward(
            self._env_vars(party_size=1, party_exp=0), "a"
        )

        # Gain 60 XP.
        rewards.calculate_reward(
            self._env_vars(party_size=1, party_exp=60), "a"
        )

        # Party grows to 2 with extra EXP (e.g. capture) — accumulator pauses,
        # no milestone fires.
        r, _ = rewards.calculate_reward(
            self._env_vars(party_size=2, party_exp=200), "a"
        )
        self.assertEqual(float(r), 0.0)

    def test_xp_milestone_resumes_after_size_change(self):
        """XP milestone accumulator resumes after party size stabilises."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["xp_milestone_threshold"] = 100
        config["xp_milestone_reward"] = 25

        rewards = Rewards(config)
        # Seed with party of 1.
        rewards.calculate_reward(
            self._env_vars(party_size=1, party_exp=0), "a"
        )

        # Gain 60 XP.
        rewards.calculate_reward(
            self._env_vars(party_size=1, party_exp=60), "a"
        )

        # Party grows to 2 — accumulator pauses.
        rewards.calculate_reward(
            self._env_vars(party_size=2, party_exp=200), "a"
        )

        # Party stable at 2, gain 50 more XP (200→250) — crosses threshold.
        r, _ = rewards.calculate_reward(
            self._env_vars(party_size=2, party_exp=250), "a"
        )
        self.assertEqual(float(r), 25.0)

    def test_xp_milestone_resets_on_new_episode(self):
        """XP milestone accumulator resets on start_new_episode."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["xp_milestone_threshold"] = 100
        config["xp_milestone_reward"] = 25

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(party_exp=0), "a")  # seed
        rewards.calculate_reward(self._env_vars(party_exp=80), "a")  # 80 XP

        # Start new episode — accumulator resets.
        rewards.start_new_episode()
        self.assertEqual(rewards._xp_since_milestone, 0)

        # Gain 80 XP in new episode — below threshold.
        r, _ = rewards.calculate_reward(self._env_vars(party_exp=80), "a")
        self.assertEqual(float(r), 0.0)

    # ---- Config validation tests ----

    def test_config_validation_exceeds_total_fires(self):
        """Warns when hard_goal_count_target exceeds total possible fires."""
        config = self._base_config(goals=[
            _loc([[1, 1, 1]]), _loc([[2, 2, 1]]),  # 2 hard
            _pokedex("seen", 1),  # 1 pokedex fire = 3 total
        ])
        config["hard_goal_count_target"] = 5

        import io, sys
        captured = io.StringIO()
        sys.stdout = captured
        try:
            Rewards(config)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("exceeds total possible fires", output)

    def test_config_validation_below_hard_goals(self):
        """Warns when hard_goal_count_target is less than hard goals count."""
        config = self._base_config(goals=[
            _loc([[1, 1, 1]]), _loc([[2, 2, 1]]), _loc([[3, 3, 1]])
        ])
        config["hard_goal_count_target"] = 2

        import io, sys
        captured = io.StringIO()
        sys.stdout = captured
        try:
            Rewards(config)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("may never train", output)

    def test_config_validation_fires_once_per_config(self):
        """Validation warning fires only once per config object (not per reset)."""
        config = self._base_config(goals=[_loc([[1, 1, 1]])])
        config["hard_goal_count_target"] = 5

        import io, sys
        captured = io.StringIO()
        sys.stdout = captured
        try:
            Rewards(config)  # first instantiation — should warn
            Rewards(config)  # same config object — should NOT warn again
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        # Count how many times the warning appears.
        self.assertEqual(output.count("exceeds total possible fires"), 1)

    def test_config_validation_no_warning_valid(self):
        """No warning when hard_goal_count_target is within valid range."""
        config = self._base_config(goals=[
            _loc([[1, 1, 1]]), _loc([[2, 2, 1]]),  # 2 hard
            _pokedex("seen", 2),  # 2 pokedex fires = 4 total
        ])
        config["hard_goal_count_target"] = 3

        import io, sys
        captured = io.StringIO()
        sys.stdout = captured
        try:
            Rewards(config)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertNotIn("WARNING", output)

    # --------------------------------------------------------------- #
    # Level goals                                                     #
    # --------------------------------------------------------------- #

    def test_level_goals_disabled_by_default(self):
        """Level goals are empty by default."""
        config = self._base_config(goals=[_loc([[1, 1, 1]])])
        config["hard_goal_count_target"] = 1
        rewards = Rewards(config)
        self.assertEqual(rewards.level_goals, {})
        self.assertEqual(rewards.level_goals_completed, 0)

    def test_level_goals_fires_on_threshold(self):
        """Level goal fires when party gains levels from starting point."""
        config = self._base_config(goals=[
            _loc([[1, 1, 1]]),
            _level(1),
        ])
        config["hard_goal_count_target"] = 2  # 1 location + 1 level
        config["break_on_goal"] = False

        rewards = Rewards(config)
        env_vars = self._env_vars(x=1, y=1, map_num=1, party_level=5)
        rewards.calculate_reward(env_vars, "a")  # seed step, hit location goal

        # Party gains 1 level → level goal fires, N_goals reaches target
        env_vars = self._env_vars(x=1, y=1, map_num=1, party_level=6)
        reward, done = rewards.calculate_reward(env_vars, "a")
        self.assertEqual(rewards.N_goals, 2)
        self.assertEqual(rewards.level_goals_completed, 1)
        self.assertEqual(rewards.n_level_goals_completed(), 1)

    def test_level_goals_multi_fire(self):
        """Multiple level fires in one step when level jumps."""
        config = self._base_config(goals=[_level(3)])
        config["hard_goal_count_target"] = 10
        config["break_on_goal"] = False

        rewards = Rewards(config)
        env_vars = self._env_vars(party_level=5)
        rewards.calculate_reward(env_vars, "a")  # seed

        # Jump from level 5 to 8 (3 levels gained) → 3 fires
        env_vars = self._env_vars(party_level=8)
        rewards.calculate_reward(env_vars, "a")
        self.assertEqual(rewards.level_goals_completed, 3)
        self.assertEqual(rewards.N_goals, 3)

    def test_level_goals_suppressed_on_party_size_change(self):
        """Level goals don't fire when party size changes."""
        config = self._base_config(goals=[_level(3)])
        config["hard_goal_count_target"] = 10
        config["break_on_goal"] = False

        rewards = Rewards(config)
        env_vars = self._env_vars(party_level=5, party_size=1)
        rewards.calculate_reward(env_vars, "a")  # seed

        # Party size changes AND level jumps — should be suppressed
        env_vars = self._env_vars(party_level=15, party_size=2)
        rewards.calculate_reward(env_vars, "a")
        self.assertEqual(rewards.level_goals_completed, 0)

        # Now party is stable at size 2, baseline is 15. Gain 1 more level.
        env_vars = self._env_vars(party_level=16, party_size=2)
        rewards.calculate_reward(env_vars, "a")
        self.assertEqual(rewards.level_goals_completed, 1)

    def test_level_goals_resets_on_new_episode(self):
        """Level goal trackers reset on start_new_episode."""
        config = self._base_config(goals=[_level(3)])
        config["hard_goal_count_target"] = 10
        config["break_on_goal"] = False

        rewards = Rewards(config)
        env_vars = self._env_vars(party_level=5)
        rewards.calculate_reward(env_vars, "a")  # seed

        # Gain some levels
        env_vars = self._env_vars(party_level=7)
        rewards.calculate_reward(env_vars, "a")
        self.assertEqual(rewards.level_goals_completed, 2)

        # New episode resets trackers (but NOT level_goals_completed counter)
        rewards.start_new_episode()
        self.assertIsNone(rewards.goals._level_starting_total)
        self.assertIsNone(rewards.goals._level_prev_size)

    def test_level_goals_break_on_target(self):
        """Episode breaks when level goal crossing hits hard_goal_count_target."""
        config = self._base_config(goals=[_level(1)])
        config["hard_goal_count_target"] = 1
        config["break_on_goal"] = True

        rewards = Rewards(config)
        env_vars = self._env_vars(party_level=5)
        rewards.calculate_reward(env_vars, "a")  # seed

        # Level up → N_goals reaches target → done
        env_vars = self._env_vars(party_level=6)
        reward, done = rewards.calculate_reward(env_vars, "a")
        self.assertTrue(done)
        self.assertEqual(rewards.N_goals, 1)

    def test_config_validation_includes_level_goals(self):
        """Config validation counts level goals in total possible fires."""
        config = self._base_config(goals=[
            _loc([[1, 1, 1]]),  # 1 hard
            _pokedex("seen", 1),  # 1 pokedex
            _level(2),  # 2 level fires = 4 total
        ])
        config["hard_goal_count_target"] = 10

        import io, sys
        captured = io.StringIO()
        sys.stdout = captured
        try:
            Rewards(config)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("exceeds total possible fires", output)
        self.assertIn("4", output)  # total possible

    # --------------------------------------------------------------- #
    # XP goals                                                        #
    # --------------------------------------------------------------- #

    def test_xp_goals_disabled_by_default(self):
        """XP goals dict is empty unless configured."""
        config = self._base_config(goals=[_loc([[1, 1, 1]])])
        config["hard_goal_count_target"] = 1
        rewards = Rewards(config)
        self.assertEqual(rewards.xp_goals, {})
        self.assertEqual(rewards.xp_goals_completed, 0)
        self.assertEqual(rewards.n_xp_goals_completed(), 0)

    def test_xp_goals_fires_on_chunk_crossing(self):
        """One XP-goal fire pays xp_goal_reward when chunk threshold is crossed."""
        config = self._base_config(goals=[
            _loc([[999, 999, 999]]),
            _xp(3, xp_per_fire=10),
        ])
        config["hard_goal_count_target"] = 99  # don't end episode
        config["xp_goal_threshold"] = 10
        config["xp_goal_reward"] = 100
        config["break_on_goal"] = False

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(party_exp=0), "a")  # seed
        # Gain 12 XP → 1 chunk crossed → 1 fire of 100, plus exp delta reward.
        r, _ = rewards.calculate_reward(self._env_vars(party_exp=12), "a")
        self.assertEqual(rewards.xp_goals_completed, 1)
        self.assertEqual(rewards.n_xp_goals_completed(), 1)
        self.assertGreaterEqual(float(r), 100.0)  # at minimum the goal payout

    def test_xp_goals_multi_fire_in_one_step(self):
        """A large XP jump fires multiple chunks at once, capped at max."""
        config = self._base_config(goals=[
            _loc([[999, 999, 999]]),
            _xp(3, xp_per_fire=10),
        ])
        config["hard_goal_count_target"] = 99
        config["xp_goal_threshold"] = 10
        config["xp_goal_reward"] = 100
        config["break_on_goal"] = False

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(party_exp=0), "a")  # seed
        # Gain 45 XP → 4 chunks but capped at 3.
        rewards.calculate_reward(self._env_vars(party_exp=45), "a")
        self.assertEqual(rewards.xp_goals_completed, 3)
        self.assertEqual(rewards.N_goals, 3)

    def test_xp_goals_capped_at_threshold(self):
        """Further XP gain after cap is reached does not produce more fires."""
        config = self._base_config(goals=[
            _loc([[999, 999, 999]]),
            _xp(1, xp_per_fire=10),
        ])
        config["hard_goal_count_target"] = 99
        config["xp_goal_threshold"] = 10
        config["break_on_goal"] = False

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(party_exp=0), "a")  # seed
        rewards.calculate_reward(self._env_vars(party_exp=15), "a")
        self.assertEqual(rewards.xp_goals_completed, 1)
        # Big additional gain, but cap holds.
        rewards.calculate_reward(self._env_vars(party_exp=200), "a")
        self.assertEqual(rewards.xp_goals_completed, 1)

    def test_xp_goals_suppressed_on_party_size_change(self):
        """Captures / swaps re-seed the baseline and don't fire."""
        config = self._base_config(goals=[
            _loc([[999, 999, 999]]),
            _xp(3, xp_per_fire=10),
        ])
        config["hard_goal_count_target"] = 99
        config["xp_goal_threshold"] = 10
        config["break_on_goal"] = False

        rewards = Rewards(config)
        rewards.calculate_reward(
            self._env_vars(party_exp=0, party_size=1), "a"
        )  # seed
        # Size jumps from 1 → 2 with a big EXP jump (new mon's EXP) — no fire.
        rewards.calculate_reward(
            self._env_vars(party_exp=500, party_size=2), "a"
        )
        self.assertEqual(rewards.xp_goals_completed, 0)
        # Stable size now; gain 10 more XP from new baseline → 1 fire.
        rewards.calculate_reward(
            self._env_vars(party_exp=510, party_size=2), "a"
        )
        self.assertEqual(rewards.xp_goals_completed, 1)

    def test_xp_goals_resets_on_new_episode(self):
        """start_new_episode clears the XP-goal trackers."""
        config = self._base_config(goals=[
            _loc([[999, 999, 999]]),
            _xp(3, xp_per_fire=10),
        ])
        config["hard_goal_count_target"] = 99
        config["xp_goal_threshold"] = 10
        config["break_on_goal"] = False

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(party_exp=100), "a")
        self.assertEqual(rewards.goals._xp_starting_total, 100)

        rewards.start_new_episode()
        self.assertIsNone(rewards.goals._xp_starting_total)
        self.assertIsNone(rewards.goals._xp_prev_size)

    def test_xp_goals_skipped_on_transitional_battle_type(self):
        """Junk battle_type (e.g. 122) short-circuits XP-goal firing."""
        config = self._base_config(goals=[
            _loc([[999, 999, 999]]),
            _xp(3, xp_per_fire=10),
        ])
        config["hard_goal_count_target"] = 99
        config["xp_goal_threshold"] = 10
        config["break_on_goal"] = False

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(party_exp=0), "a")  # seed
        # Junk battle_type with apparent XP jump — must not fire.
        # Top-level gate (is_ram_state_valid) catches it first, but the
        # per-method guard is defense-in-depth.
        rewards._check_xp_goals(
            {"battle_type": 122, "party_info": (1, 5, 20, 500)}
        )
        self.assertEqual(rewards.xp_goals_completed, 0)

    def test_xp_goals_break_on_target(self):
        """Episode ends when XP goal crossing hits hard_goal_count_target."""
        config = self._base_config(goals=[_xp(1, xp_per_fire=10)])
        config["hard_goal_count_target"] = 1
        config["xp_goal_threshold"] = 10
        config["break_on_goal"] = True

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(party_exp=0), "a")  # seed
        _, done = rewards.calculate_reward(
            self._env_vars(party_exp=10), "a"
        )
        self.assertTrue(done)
        self.assertEqual(rewards.N_goals, 1)

    def test_config_validation_includes_xp_goals(self):
        """Config validation counts xp goals in total possible fires."""
        config = self._base_config(goals=[
            _loc([[1, 1, 1]]),  # 1 hard
            _xp(2, xp_per_fire=10),  # 2 xp fires = 3 total
        ])
        config["hard_goal_count_target"] = 10

        import io, sys
        captured = io.StringIO()
        sys.stdout = captured
        try:
            Rewards(config)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("exceeds total possible fires", output)

    # --------------------------------------------------------------- #
    # Battle engagement + damage rewards                              #
    # --------------------------------------------------------------- #

    def test_battle_engagement_silent_outside_battle(self):
        """battle_engagement_reward and damage_dealt_reward only fire when
        battle_type != 0."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["battle_engagement_reward"] = 1.0
        config["damage_dealt_reward"] = 0.5

        rewards = Rewards(config)
        r, _ = rewards.calculate_reward(
            self._env_vars(battle_type=0, enemy_hp=20), "a"
        )
        self.assertEqual(float(r), 0.0)
        self.assertIsNone(rewards._prev_enemy_hp)

    def test_battle_engagement_pays_per_step_in_battle(self):
        """Per-step engagement reward fires every step inside a battle."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["battle_engagement_reward"] = 1.0
        config["damage_dealt_reward"] = 0.0  # isolate engagement

        rewards = Rewards(config)
        # First in-battle step seeds _prev_enemy_hp; engagement still pays.
        r1, _ = rewards.calculate_reward(
            self._env_vars(battle_type=1, enemy_hp=30), "a"
        )
        # Second step at same HP — engagement only, no damage.
        r2, _ = rewards.calculate_reward(
            self._env_vars(battle_type=1, enemy_hp=30), "a"
        )
        self.assertEqual(float(r1), 1.0)
        self.assertEqual(float(r2), 1.0)

    def test_damage_dealt_reward_scales_with_hp_drop(self):
        """Damage reward is damage_dealt_reward × (prev_hp - curr_hp)."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["battle_engagement_reward"] = 0.0  # isolate damage
        config["damage_dealt_reward"] = 0.5

        rewards = Rewards(config)
        # Seed at full HP.
        rewards.calculate_reward(
            self._env_vars(battle_type=1, enemy_hp=30), "a"
        )
        # Drop HP by 10 → 10 × 0.5 = 5.
        r, _ = rewards.calculate_reward(
            self._env_vars(battle_type=1, enemy_hp=20), "a"
        )
        self.assertEqual(float(r), 5.0)

    def test_damage_first_battle_step_is_not_credited(self):
        """The first step of a battle seeds prev_hp but does not pay damage —
        the initial enemy_hp reading must never be treated as a drop from 0."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["battle_engagement_reward"] = 0.0
        config["damage_dealt_reward"] = 1.0

        rewards = Rewards(config)
        # First reading of the battle: enemy at 25 HP. No damage should be credited.
        r, _ = rewards.calculate_reward(
            self._env_vars(battle_type=1, enemy_hp=25), "a"
        )
        self.assertEqual(float(r), 0.0)
        self.assertEqual(rewards._prev_enemy_hp, 25)

    def test_damage_counter_resets_when_battle_ends(self):
        """When battle_type returns to 0, _prev_enemy_hp clears so HP from a
        finished battle cannot leak into the next one."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["battle_engagement_reward"] = 0.0
        config["damage_dealt_reward"] = 1.0

        rewards = Rewards(config)
        rewards.calculate_reward(
            self._env_vars(battle_type=1, enemy_hp=30), "a"
        )  # seed
        # Damage to 10 → 20 reward.
        rewards.calculate_reward(
            self._env_vars(battle_type=1, enemy_hp=10), "a"
        )
        # Battle ends; tracker clears.
        r, _ = rewards.calculate_reward(
            self._env_vars(battle_type=0, enemy_hp=0), "a"
        )
        self.assertEqual(float(r), 0.0)
        self.assertIsNone(rewards._prev_enemy_hp)

        # A NEW battle starts at full HP. This must NOT credit damage from
        # (10 → 40) as a negative or from prior battle.
        r, _ = rewards.calculate_reward(
            self._env_vars(battle_type=1, enemy_hp=40), "a"
        )
        self.assertEqual(float(r), 0.0)  # first-step seed
        self.assertEqual(rewards._prev_enemy_hp, 40)

    def test_damage_ignores_enemy_hp_increase(self):
        """Healing items / state quirks that increase HP must not yield
        negative reward."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["battle_engagement_reward"] = 0.0
        config["damage_dealt_reward"] = 1.0

        rewards = Rewards(config)
        rewards.calculate_reward(
            self._env_vars(battle_type=1, enemy_hp=10), "a"
        )  # seed
        # HP goes UP (potion). Should be no reward.
        r, _ = rewards.calculate_reward(
            self._env_vars(battle_type=1, enemy_hp=25), "a"
        )
        self.assertEqual(float(r), 0.0)
        # _prev_enemy_hp tracks the new (higher) value so subsequent damage
        # is credited correctly.
        self.assertEqual(rewards._prev_enemy_hp, 25)

    def test_damage_counter_resets_on_new_episode(self):
        """start_new_episode clears _prev_enemy_hp so replay-time damage
        cannot be re-credited into the training episode."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["battle_engagement_reward"] = 0.0
        config["damage_dealt_reward"] = 1.0

        rewards = Rewards(config)
        rewards.calculate_reward(
            self._env_vars(battle_type=1, enemy_hp=30), "a"
        )  # seed
        rewards.calculate_reward(
            self._env_vars(battle_type=1, enemy_hp=15), "a"
        )  # +15
        self.assertEqual(rewards._prev_enemy_hp, 15)

        rewards.start_new_episode()
        self.assertIsNone(rewards._prev_enemy_hp)

        # First step of new episode (still in battle at 15 HP) seeds again
        # without crediting damage.
        r, _ = rewards.calculate_reward(
            self._env_vars(battle_type=1, enemy_hp=15), "a"
        )
        self.assertEqual(float(r), 0.0)

    # --------------------------------------------------------------- #
    # New-map first-visit bonus                                       #
    # --------------------------------------------------------------- #

    def test_new_map_bonus_fires_on_first_visit(self):
        """new_map_reward fires once per (map_bank, map_num) pair."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["new_map_reward"] = 50
        config["exploration_reward"] = 0.0  # isolate new-map bonus

        rewards = Rewards(config)
        r1, _ = rewards.calculate_reward(
            self._env_vars(x=5, y=5, map_num=3, map_bank=50), "a"
        )
        # Same map, different tile — no new-map bonus, no per-tile reward.
        r2, _ = rewards.calculate_reward(
            self._env_vars(x=6, y=5, map_num=3, map_bank=50), "a"
        )
        # Different map — fires again.
        r3, _ = rewards.calculate_reward(
            self._env_vars(x=1, y=1, map_num=4, map_bank=50), "a"
        )

        self.assertEqual(float(r1), 50.0)
        self.assertEqual(float(r2), 0.0)
        self.assertEqual(float(r3), 50.0)

    def test_new_map_bonus_distinguishes_map_bank(self):
        """Same map_num in different map_banks counts as distinct maps."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["new_map_reward"] = 50
        config["exploration_reward"] = 0.0

        rewards = Rewards(config)
        r1, _ = rewards.calculate_reward(
            self._env_vars(map_num=3, map_bank=50), "a"
        )
        r2, _ = rewards.calculate_reward(
            self._env_vars(map_num=3, map_bank=51), "a"
        )
        self.assertEqual(float(r1), 50.0)
        self.assertEqual(float(r2), 50.0)

    def test_new_map_bonus_preserved_across_new_episode(self):
        """explored_maps is preserved across start_new_episode (replay→train
        boundary) so the replay's discovered maps don't re-fire."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["new_map_reward"] = 50
        config["exploration_reward"] = 0.0

        rewards = Rewards(config)
        rewards.calculate_reward(self._env_vars(map_num=3, map_bank=50), "a")
        rewards.start_new_episode()

        # Re-entering the same map after start_new_episode pays nothing.
        r, _ = rewards.calculate_reward(self._env_vars(map_num=3, map_bank=50), "a")
        self.assertEqual(float(r), 0.0)

    def test_new_map_bonus_disabled_when_zero(self):
        """new_map_reward=0 disables the bonus entirely (no explored_maps
        accounting overhead either)."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["new_map_reward"] = 0
        config["exploration_reward"] = 0.0

        rewards = Rewards(config)
        r, _ = rewards.calculate_reward(self._env_vars(map_num=3, map_bank=50), "a")
        self.assertEqual(float(r), 0.0)
        self.assertEqual(rewards.explored_maps, set())

    # --------------------------------------------------------------- #
    # Soft waypoint goals                                              #
    # --------------------------------------------------------------- #

    def test_soft_goal_pays_waypoint_reward(self):
        """A soft goal fires soft_waypoint_reward but does not advance
        the curriculum index."""
        config = self._base_config(goals=[
            _loc([[1, 1, 1]]),  # hard goal 0
            _loc([[2, 2, 1]], hard=False),  # soft goal
        ])
        config["hard_goal_count_target"] = 1
        config["soft_waypoint_reward"] = 25

        rewards = Rewards(config)
        # Hit the soft goal first — should pay soft_waypoint_reward.
        r, _ = rewards.calculate_reward(self._env_vars(x=2, y=2), "a")
        self.assertEqual(float(r), 25.0)
        # Soft goal increments N_goals but does not advance curriculum.
        self.assertEqual(rewards.N_goals, 1)
        self.assertEqual(rewards.current_goal_index, 0)

    def test_soft_goal_does_not_trigger_termination(self):
        """Hitting a soft goal should not break the episode on its own."""
        config = self._base_config(goals=[
            _loc([[999, 999, 999]]),  # hard goal (unreachable)
            _loc([[2, 2, 1]], hard=False),  # soft goal
        ])
        config["hard_goal_count_target"] = 1
        config["soft_waypoint_reward"] = 25
        config["break_on_goal"] = True

        rewards = Rewards(config)
        # Hit the soft goal — N_goals becomes 1 but target is on hard goals.
        r, done = rewards.calculate_reward(self._env_vars(x=2, y=2), "a")
        self.assertFalse(done)  # should NOT break
        self.assertEqual(float(r), 25.0)

    def test_soft_goal_consumed_after_hit(self):
        """A soft goal is removed after being hit (no re-fire)."""
        config = self._base_config(goals=[
            _loc([[999, 999, 999]]),  # hard goal (unreachable)
            _loc([[5, 5, 1]], hard=False),  # soft goal
        ])
        config["hard_goal_count_target"] = 99
        config["soft_waypoint_reward"] = 25

        rewards = Rewards(config)
        r1, _ = rewards.calculate_reward(self._env_vars(x=5, y=5), "a")
        self.assertEqual(float(r1), 25.0)
        # Second visit — no re-fire.
        r2, _ = rewards.calculate_reward(self._env_vars(x=5, y=5), "a")
        self.assertEqual(float(r2), 0.0)

    def test_soft_goal_in_validation(self):
        """Soft goals count toward total possible fires in validation."""
        config = self._base_config(goals=[
            _loc([[1, 1, 1]]),  # 1 hard
            _loc([[2, 2, 1]], hard=False),  # 1 soft
        ])
        config["hard_goal_count_target"] = 2

        import io, sys
        captured = io.StringIO()
        sys.stdout = captured
        try:
            Rewards(config)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        # Target (2) == total (1 hard + 1 soft) — no warning.
        self.assertNotIn("WARNING", output)

    # --------------------------------------------------------------- #
    # Configurable pokedex rewards                                    #
    # --------------------------------------------------------------- #

    # --------------------------------------------------------------- #
    # Transitional / garbage-RAM guards                               #
    # --------------------------------------------------------------- #

    def test_party_progress_skipped_on_invalid_battle_type(self):
        """A battle_type outside {0,1,2} marks a transitional RAM snapshot.
        On those steps party_progress must not credit any Δexp/Δlevel and
        must not move the prev trackers — so the next valid step still
        compares against the last valid reading rather than against junk.
        """
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["party_level_reward"] = 10
        config["party_exp_reward"] = 0.01

        rewards = Rewards(config)
        # Seed at a valid pre-transition state.
        rewards.calculate_reward(
            self._env_vars(party_level=5, party_exp=125, battle_type=0), "a"
        )
        prev_level = rewards._prev_party_level
        prev_exp = rewards._prev_party_exp

        # Transitional snapshot with junk party_info (the bug scenario).
        # battle_type=122 is exactly what we saw in the stage-5 trace.
        r, _ = rewards.calculate_reward(
            self._env_vars(
                party_level=200, party_exp=10_000_000, battle_type=122
            ),
            "a",
        )
        self.assertEqual(float(r), 0.0)
        # Trackers must be untouched — junk values from this step cannot
        # be set as the baseline for the next valid step.
        self.assertEqual(rewards._prev_party_level, prev_level)
        self.assertEqual(rewards._prev_party_exp, prev_exp)

        # Next valid step (small genuine increase) credits correctly against
        # the pre-transition baseline, not against the junk we just rejected.
        # battle_engagement_reward and damage_dealt_reward both default to 0
        # in _base_config (they're only set by curriculum_base.json).
        r, _ = rewards.calculate_reward(
            self._env_vars(party_level=6, party_exp=200, battle_type=1), "a"
        )
        # Δlevel = 1 × 10 = 10, Δexp = 75 × 0.01 = 0.75.
        self.assertAlmostEqual(float(r), 10.75, places=5)

    def test_battle_engagement_skipped_on_invalid_battle_type(self):
        """battle_type values outside {0,1,2} are mid-transition garbage.
        Engagement reward must not fire, damage tracker must not update."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["battle_engagement_reward"] = 1.0
        config["damage_dealt_reward"] = 1.0

        rewards = Rewards(config)
        # Establish a valid prev_enemy_hp in a real battle.
        rewards.calculate_reward(
            self._env_vars(battle_type=1, enemy_hp=30), "a"
        )
        self.assertEqual(rewards._prev_enemy_hp, 30)

        # Transitional snapshot: battle_type=122, junk enemy_hp.
        r, _ = rewards.calculate_reward(
            self._env_vars(battle_type=122, enemy_hp=5), "a"
        )
        self.assertEqual(float(r), 0.0)
        # _prev_enemy_hp untouched — the next valid in-battle step still
        # compares against the last real reading.
        self.assertEqual(rewards._prev_enemy_hp, 30)

    def test_party_progress_seed_skipped_on_invalid_battle_type(self):
        """A garbage-RAM step seen BEFORE any seeding must not seed the
        trackers — the first valid step seeds them."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["party_level_reward"] = 10

        rewards = Rewards(config)
        # First step is transitional garbage — must not seed.
        rewards.calculate_reward(
            self._env_vars(party_level=200, battle_type=99), "a"
        )
        self.assertIsNone(rewards._prev_party_size)

        # Next step is the real first observation — seeds, no reward.
        r, _ = rewards.calculate_reward(
            self._env_vars(party_level=5, battle_type=0), "a"
        )
        self.assertEqual(float(r), 0.0)
        self.assertEqual(rewards._prev_party_level, 5)


    # --------------------------------------------------------------- #
    # RAM-validity helper + full reward-path gating                   #
    # --------------------------------------------------------------- #

    def test_is_ram_state_valid_accepts_normal_play(self):
        """The helper accepts ordinary post-tick reads — overworld, in a
        wild battle, and in a trainer battle, across the valid player
        states."""
        for battle_type in (0, 1, 2):
            for player_state in (0, 1, 2, 4):
                env_vars = self._env_vars(
                    x=10, y=8, map_num=3, map_bank=50,
                    battle_type=battle_type, player_state=player_state,
                )
                self.assertTrue(
                    is_ram_state_valid(env_vars),
                    f"rejected valid combo battle={battle_type} player={player_state}",
                )

    def test_is_ram_state_valid_rejects_unknown_battle_type(self):
        for bt in (3, 5, 50, 122, 200, 255):
            env_vars = self._env_vars(battle_type=bt)
            self.assertFalse(
                is_ram_state_valid(env_vars), f"accepted bogus battle_type={bt}"
            )

    def test_is_ram_state_valid_rejects_unknown_player_state(self):
        for ps in (3, 5, 50, 99, 255):
            env_vars = self._env_vars(player_state=ps)
            self.assertFalse(
                is_ram_state_valid(env_vars), f"accepted bogus player_state={ps}"
            )

    def test_is_ram_state_valid_rejects_all_zero_coords(self):
        """x=y=map=bank=0 is the screen-scroll-out junk signature."""
        env_vars = self._env_vars(x=0, y=0, map_num=0, map_bank=0)
        self.assertFalse(is_ram_state_valid(env_vars))

    def test_is_ram_state_valid_accepts_partial_zeros(self):
        """A single zero coord on its own is normal play (e.g. y=0 at the
        top edge of a map). Only all four zero simultaneously is junk."""
        self.assertTrue(is_ram_state_valid(self._env_vars(x=0, y=5, map_num=3)))
        self.assertTrue(is_ram_state_valid(self._env_vars(x=5, y=0, map_num=3)))
        self.assertTrue(is_ram_state_valid(self._env_vars(x=5, y=5, map_num=0, map_bank=50)))

    def test_calculate_reward_skips_macro_micro_on_junk(self):
        """A junk frame returns just step_penalty + button_penalty and
        does not touch exploration / pokedex / prev trackers."""
        config = self._base_config()
        config["punish_steps"] = True
        config["step_penalty"] = -0.5
        config["hard_goal_count_target"] = 99
        config["exploration_reward"] = 1.0
        config["new_map_reward"] = 50.0
        config["pokedex_seen_reward"] = 50
        config["pokedex_owned_reward"] = 150
        config["battle_engagement_reward"] = 1.0

        rewards = Rewards(config)
        # Junk frame with otherwise-rewarding readings: novel tile, novel
        # map, pokedex jumped, claimed to be in a battle. All must be
        # ignored — only step_penalty pays.
        r, _ = rewards.calculate_reward(
            self._env_vars(
                x=0, y=0, map_num=0, map_bank=0,    # all-zero junk signature
                seen=5, owned=2,
                battle_type=122,                     # bogus battle byte
                player_state=50,                     # bogus player byte
            ),
            "a",
        )
        self.assertEqual(float(r), -0.5)
        self.assertEqual(rewards.explored_tiles, set())
        self.assertEqual(rewards.explored_maps, set())
        self.assertEqual(rewards.pokedex_seen, 0)
        self.assertEqual(rewards.pokedex_owned, 0)
        # Steps still counted toward the episode budget.
        self.assertEqual(rewards.steps, 1)

    def test_calculate_reward_junk_then_valid_seeds_cleanly(self):
        """Following a junk frame, the next valid frame should seed
        trackers normally — junk values must not be the baseline."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["exploration_reward"] = 1.0
        config["new_map_reward"] = 50.0

        rewards = Rewards(config)
        # Junk first.
        rewards.calculate_reward(
            self._env_vars(x=0, y=0, map_num=0, map_bank=0, seen=99), "a"
        )
        # First valid step — should pay both exploration and new-map
        # bonuses for a real tile.
        r, _ = rewards.calculate_reward(
            self._env_vars(x=10, y=10, map_num=3, map_bank=50), "a"
        )
        self.assertEqual(float(r), 51.0)  # 1 (tile) + 50 (new map)
        self.assertIn((10, 10, 50, 3), rewards.explored_tiles)
        self.assertIn((50, 3), rewards.explored_maps)
        # pokedex_seen is still untouched by the junk frame.
        self.assertEqual(rewards.pokedex_seen, 0)

    def test_calculate_reward_junk_button_penalty_still_applies(self):
        """A start/select press on a junk frame still incurs button_penalty
        — the penalty depends on the action, not RAM contents."""
        config = self._base_config()
        config["punish_steps"] = False  # isolate button penalty
        config["hard_goal_count_target"] = 99

        rewards = Rewards(config)
        r, _ = rewards.calculate_reward(
            self._env_vars(x=0, y=0, map_num=0, map_bank=0), "start"
        )
        self.assertEqual(float(r), float(rewards.button_penalty))


    def test_pokedex_seen_reward_configurable(self):
        """pokedex_seen_reward can be overridden via config (e.g. set to 0 to
        damp the 'flee for seen-credit' incentive)."""
        config = self._base_config()
        config["hard_goal_count_target"] = 99
        config["pokedex_seen_reward"] = 0
        config["pokedex_owned_reward"] = 200

        rewards = Rewards(config)
        self.assertEqual(rewards.pokedex_seen_reward, 0)
        self.assertEqual(rewards.pokedex_owned_reward, 200)

        # seen++ should produce no reward when pokedex_seen_reward=0.
        r, _ = rewards.calculate_reward(self._env_vars(seen=1), "a")
        self.assertEqual(float(r), 0.0)

        # owned++ still pays the configured 200.
        r, _ = rewards.calculate_reward(self._env_vars(seen=1, owned=1), "a")
        self.assertEqual(float(r), 200.0)


if __name__ == "__main__":
    unittest.main()
