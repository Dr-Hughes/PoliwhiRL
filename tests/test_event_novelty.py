# -*- coding: utf-8 -*-
"""Event-flag novelty reward (rewards.py _event_novelty_bonus).

Pinned behaviours:
- OFF by default: flipping an event bit pays nothing.
- ON: the first 0->1 flip THIS episode of a non-excluded bit over the whole
  wEventFlags region pays event_novelty_bonus (full, run-first).
- Flags already set at episode start (baked into the save-state) never fire.
- Excluded bits (event_novelty_exclude_flags) never fire.
- A bit pays at most once per episode.
- Re-fires deplete by 1/sqrt(1 + run-wide fire count) via the shared archive,
  and get_milestone_state -> merge_milestones records the run-wide count.
"""
import math
import unittest

import numpy as np

from PoliwhiRL.environment.rewards import Rewards
from PoliwhiRL.environment.visit_archive import VisitArchive


def _zero_flags():
    return np.zeros(256, dtype=np.uint8)


def _set_flag(flags, flag_num):
    """Set event-flag bit `flag_num` (LSB-first within a byte, matching
    _DERIVED_FLAG_TABLE's flag_num // 8, % 8 convention)."""
    flags = flags.copy()
    flags[flag_num // 8] |= 1 << (flag_num % 8)
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
        "flag_progress_reward": 0,
        "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _env_vars(story_flags):
    return {
        "X": 4,
        "Y": 3,
        "map_num": 7,
        "map_bank": 24,
        "room": 0,
        "warp_number": 0,
        "money": 0,
        "pokedex_seen": 0,
        "pokedex_owned": 0,
        "collision_down": 0,
        "collision_up": 0,
        "collision_left": 0,
        "collision_right": 0,
        "story_flags": story_flags,
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


# A safe, non-excluded, non-transient event bit for the tests.
_TEST_FLAG = 100


class TestEventNovelty(unittest.TestCase):
    def _step(self, rw, flags):
        rw.calculate_reward(_env_vars(flags), button_press=0)
        return rw.get_episode_breakdown()["event"]

    def test_off_by_default(self):
        rw = Rewards(_base_config())
        self._step(rw, _zero_flags())  # baseline
        ev = self._step(rw, _set_flag(_zero_flags(), _TEST_FLAG))
        self.assertEqual(ev, 0.0)

    def test_fresh_flip_pays_full(self):
        rw = Rewards(_base_config(event_novelty_enabled=True, event_novelty_bonus=2.0))
        self._step(rw, _zero_flags())  # baseline snapshot
        ev = self._step(rw, _set_flag(_zero_flags(), _TEST_FLAG))
        self.assertAlmostEqual(ev, 2.0, places=5)

    def test_flag_set_at_start_never_fires(self):
        rw = Rewards(_base_config(event_novelty_enabled=True, event_novelty_bonus=2.0))
        pre = _set_flag(_zero_flags(), _TEST_FLAG)
        self._step(rw, pre)  # baseline already has it set
        ev = self._step(rw, pre)
        self.assertEqual(ev, 0.0)

    def test_excluded_flag_never_fires(self):
        rw = Rewards(
            _base_config(
                event_novelty_enabled=True,
                event_novelty_bonus=2.0,
                event_novelty_exclude_flags=[_TEST_FLAG],
            )
        )
        self._step(rw, _zero_flags())
        ev = self._step(rw, _set_flag(_zero_flags(), _TEST_FLAG))
        self.assertEqual(ev, 0.0)

    def test_pays_once_per_episode(self):
        rw = Rewards(_base_config(event_novelty_enabled=True, event_novelty_bonus=2.0))
        self._step(rw, _zero_flags())
        flags = _set_flag(_zero_flags(), _TEST_FLAG)
        first = self._step(rw, flags)  # pays 2.0
        total_after_second = self._step(rw, flags)  # still set, must not re-pay
        self.assertAlmostEqual(first, 2.0, places=5)
        self.assertAlmostEqual(total_after_second, 2.0, places=5)  # unchanged

    def test_refire_depletes_across_episodes(self):
        archive = VisitArchive()
        cfg = _base_config(event_novelty_enabled=True, event_novelty_bonus=2.0)

        # Episode 1: fresh flip pays full, then merge into the archive ledger.
        rw1 = Rewards(cfg, visit_archive=archive)
        self._step(rw1, _zero_flags())
        ev1 = self._step(rw1, _set_flag(_zero_flags(), _TEST_FLAG))
        self.assertAlmostEqual(ev1, 2.0, places=5)
        state = rw1.get_milestone_state()
        self.assertIn(_TEST_FLAG, state["event_flags_fired"])
        archive.merge_milestones(**state)
        self.assertEqual(archive.event_flag_fire_count(_TEST_FLAG), 1)

        # Episode 2: same flip now pays 2.0 / sqrt(1 + 1).
        rw2 = Rewards(cfg, visit_archive=archive)
        self._step(rw2, _zero_flags())
        ev2 = self._step(rw2, _set_flag(_zero_flags(), _TEST_FLAG))
        self.assertAlmostEqual(ev2, 2.0 / math.sqrt(2.0), places=5)

    def test_rare_event_fires_reports_this_step(self):
        rw = Rewards(_base_config(event_novelty_enabled=True, event_novelty_bonus=10.0))
        self._step(rw, _zero_flags())
        self._step(rw, _set_flag(_zero_flags(), _TEST_FLAG))
        self.assertEqual(rw.rare_event_fires(3), [(_TEST_FLAG, 0)])
        self.assertEqual(rw.rare_event_fires(0), [(_TEST_FLAG, 0)])  # count 0 <= 0
        # Same flag still set next step: no new fire.
        self._step(rw, _set_flag(_zero_flags(), _TEST_FLAG))
        self.assertEqual(rw.rare_event_fires(3), [])

    def test_capture_detection_runs_even_when_reward_disabled(self):
        # goexplore_flag_capture on, event reward OFF -> detection still runs
        # (so gym_env can snapshot verge states) but pays no reward.
        rw = Rewards(_base_config(goexplore_flag_capture=True))
        self._step(rw, _zero_flags())
        ev = self._step(rw, _set_flag(_zero_flags(), _TEST_FLAG))
        self.assertEqual(ev, 0.0)
        self.assertEqual(rw.rare_event_fires(3), [(_TEST_FLAG, 0)])

    def test_no_tracking_when_both_off(self):
        rw = Rewards(_base_config())
        self._step(rw, _zero_flags())
        self._step(rw, _set_flag(_zero_flags(), _TEST_FLAG))
        self.assertEqual(rw.rare_event_fires(3), [])

    def test_archive_state_roundtrips_event_counts(self):
        archive = VisitArchive()
        archive.merge_milestones(
            flags_fired=[],
            pokedex_seen_max=0,
            pokedex_owned_max=0,
            level_max=0,
            key_items_max=0,
            event_flags_fired=[_TEST_FLAG, 200],
        )
        replica = VisitArchive()
        replica.load_state(archive.to_state())
        self.assertEqual(replica.event_flag_fire_count(_TEST_FLAG), 1)
        self.assertEqual(replica.event_flag_fire_count(200), 1)


if __name__ == "__main__":
    unittest.main()
