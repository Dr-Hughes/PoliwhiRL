# -*- coding: utf-8 -*-
"""Stagnation truncation — the episode-budget rule added 2026-07-12 after
the gamearea_long run's ep_104 spent 18k of 20k steps absorbed against a
wall in a fully-decayed room.

Pinned behaviours:

- If `stagnation_truncation_steps` consecutive FREE-WALKING steps pass
  without claiming a single first-this-episode cell, the episode ends as
  a TRUNCATION (time-limit semantics — the agent's GAE bootstraps the
  tail value), never a natural terminal.
- Scripted frames and battle frames FREEZE the counter (neither grow nor
  reset it): a long cutscene or fight can never false-trigger the cut-off.
- Claiming any novel-this-episode cell resets the counter — an agent that
  keeps covering fresh ground is never truncated, regardless of how it
  moves.
- "auto" sizes the threshold to max(256, episode_length // 16); 0
  disables the mechanism entirely.
- The counter resets across episodes (start_new_episode).
"""
import unittest
import numpy as np

from PoliwhiRL.environment.rewards import Rewards


def _zero_flags():
    return np.zeros(256, dtype=np.uint8)


def _base_config(**overrides):
    cfg = {
        "episode_length": 1000,
        "new_map_reward": 0,
        "new_bank_reward": 0,
        "frontier_novelty_bonus": 1.0,
        "whiteout_penalty": 0,
        "step_penalty": 0.0,
        "reward_round_dp": None,
        "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _env_vars(x=4, y=3, map_bank=24, map_num=7, battle_type=0, script_active=False):
    return {
        "X": x, "Y": y, "map_num": map_num, "map_bank": map_bank,
        "room": 0, "warp_number": 0, "money": 0,
        "pokedex_seen": 0, "pokedex_owned": 0,
        "collision_down": 0, "collision_up": 0,
        "collision_left": 0, "collision_right": 0,
        "story_flags": _zero_flags(),
        "battle_type": battle_type, "johto_badges": 0, "player_state": 0,
        "key_items_count": 0, "game_hour": 0, "bgm_id": 0,
        "enemy_hp": 0, "enemy_max_hp": 20,
        "party_info": (1, 5, 20, 0),
        "script_active": script_active,
    }


def _config(**overrides):
    cfg = _base_config()
    cfg.update(overrides)
    return cfg


class TestStagnationTruncation(unittest.TestCase):
    def test_absorbed_loop_truncates_at_threshold(self):
        rw = Rewards(_config(stagnation_truncation_steps=10))
        # First step claims the cell (novel -> counter resets to 0)...
        _, done = rw.calculate_reward(_env_vars(x=0), "")
        self.assertFalse(done)
        # ...then 9 more steps on the same cell: still under threshold.
        for _ in range(9):
            _, done = rw.calculate_reward(_env_vars(x=0), "")
            self.assertFalse(done)
        # The 10th consecutive stagnant step crosses it.
        _, done = rw.calculate_reward(_env_vars(x=0), "")
        self.assertTrue(done)
        self.assertTrue(rw.truncated)

    def test_two_cell_pacing_also_truncates(self):
        """The failure class is absorption, not wall-bumping specifically:
        oscillating between two already-claimed cells is caught by the
        same nothing-novel-is-happening test."""
        rw = Rewards(_config(stagnation_truncation_steps=10))
        rw.calculate_reward(_env_vars(x=0), "")
        rw.calculate_reward(_env_vars(x=10), "")  # second cell, also novel
        done = False
        for i in range(10):
            _, done = rw.calculate_reward(_env_vars(x=0 if i % 2 else 10), "")
        self.assertTrue(done)
        self.assertTrue(rw.truncated)

    def test_covering_fresh_ground_never_truncates(self):
        rw = Rewards(_config(stagnation_truncation_steps=5))
        # Each step lands on a new cell (CELL_SIZE-spaced x) -> counter
        # resets every step, no truncation despite many steps.
        for i in range(30):
            _, done = rw.calculate_reward(_env_vars(x=i * 10), "")
            self.assertFalse(done)

    def test_novel_cell_resets_the_counter(self):
        rw = Rewards(_config(stagnation_truncation_steps=6))
        rw.calculate_reward(_env_vars(x=0), "")
        for _ in range(4):
            rw.calculate_reward(_env_vars(x=0), "")
        # Fresh cell just before the threshold: counter back to 0.
        rw.calculate_reward(_env_vars(x=50), "")
        for _ in range(5):
            _, done = rw.calculate_reward(_env_vars(x=50), "")
            self.assertFalse(done)
        _, done = rw.calculate_reward(_env_vars(x=50), "")
        self.assertTrue(done)

    def test_scripted_frames_freeze_the_counter(self):
        """A cutscene of any length must not push the counter over the
        threshold — scripted frames don't count as stagnation."""
        rw = Rewards(_config(stagnation_truncation_steps=5))
        rw.calculate_reward(_env_vars(x=0), "")
        for _ in range(20):
            _, done = rw.calculate_reward(_env_vars(x=0, script_active=True), "")
            self.assertFalse(done)

    def test_battle_frames_freeze_the_counter(self):
        """Position is legitimately fixed during a battle — battle frames
        don't count as stagnation either."""
        rw = Rewards(_config(stagnation_truncation_steps=5))
        rw.calculate_reward(_env_vars(x=0), "")
        ev = _env_vars(x=0)
        ev["battle_type"] = 1
        for _ in range(20):
            _, done = rw.calculate_reward(ev, "")
            self.assertFalse(done)

    def test_freeze_does_not_reset_progress_toward_threshold(self):
        """Frozen frames pause the count; they must not wipe it (that
        would let alternating stall/script frames evade the cut-off)."""
        rw = Rewards(_config(stagnation_truncation_steps=5))
        rw.calculate_reward(_env_vars(x=0), "")
        for _ in range(4):
            rw.calculate_reward(_env_vars(x=0), "")
        rw.calculate_reward(_env_vars(x=0, script_active=True), "")
        # One more free-walking stagnant step crosses the threshold.
        _, done = rw.calculate_reward(_env_vars(x=0), "")
        self.assertTrue(done)
        self.assertTrue(rw.truncated)

    def test_truncation_is_not_a_natural_terminal(self):
        """done=True must arrive WITH truncated=True — downstream GAE
        treats it as a time-limit (bootstraps V(s')), not a death."""
        rw = Rewards(_config(stagnation_truncation_steps=3))
        rw.calculate_reward(_env_vars(x=0), "")
        done = False
        while not done:
            _, done = rw.calculate_reward(_env_vars(x=0), "")
        self.assertTrue(rw.truncated)

    def test_auto_threshold_scales_with_episode_length(self):
        rw_short = Rewards(_config(episode_length=1000))
        self.assertEqual(rw_short._stagnation_limit, 256)  # floor
        rw_long = Rewards(_config(episode_length=20480))
        self.assertEqual(rw_long._stagnation_limit, 1280)  # 20480 // 16

    def test_zero_disables(self):
        rw = Rewards(_config(stagnation_truncation_steps=0))
        rw.calculate_reward(_env_vars(x=0), "")
        for _ in range(500):
            _, done = rw.calculate_reward(_env_vars(x=0), "")
            self.assertFalse(done)

    def test_counter_resets_across_episodes(self):
        rw = Rewards(_config(stagnation_truncation_steps=5))
        rw.calculate_reward(_env_vars(x=0), "")
        for _ in range(4):
            rw.calculate_reward(_env_vars(x=0), "")
        rw.start_new_episode()
        # The old episode's near-threshold count must not leak in.
        rw.calculate_reward(_env_vars(x=0), "")  # novel again (new episode)
        for _ in range(4):
            _, done = rw.calculate_reward(_env_vars(x=0), "")
            self.assertFalse(done)


if __name__ == "__main__":
    unittest.main()
