# -*- coding: utf-8 -*-
"""Story milestones (config ``story_milestones``) and the milestone clip carry.

Pinned behaviours:

- A flag milestone re-prices its durable checkpoint in honest episodes; a
  seeded episode keeps the normal checkpoint price.
- A map milestone pays once per episode on first entry, in any order, never
  for the map the episode starts on; seeded entries pay the depleting seeded
  price.
- A milestone payout above ``reward_clip`` is delivered over the following
  steps; a non-milestone spike over the clip is still cut.
- A flag milestone must be a durable checkpoint.
"""
import json
import os
import unittest

import numpy as np

from PoliwhiRL.environment.rewards import Rewards

FALKNER = 1213


def _flags(*set_flags):
    flags = np.zeros(256, dtype=np.uint8)
    for f in set_flags:
        flags[f // 8] |= 1 << (f % 8)
    return flags


def _cfg(**overrides):
    cfg = {
        "episode_length": 1000,
        "new_map_reward": 0,
        "new_bank_reward": 0,
        "frontier_novelty_bonus": 0,
        "whiteout_penalty": 0,
        "step_penalty": 0.0,
        "reward_round_dp": None,
        "goals": [],
        "flag_progress_reward": 0,
        "checkpoint_progress_reward": 40,
        "seeded_checkpoint_progress_reward": 10,
        "reward_clip": 50,
        "story_milestones": [
            {"type": "flag", "flag_num": FALKNER, "reward": 200},
            {"type": "map", "map_bank": 3, "map_num": 37, "reward": 150},
            {"type": "map", "map_bank": 8, "map_num": 7, "reward": 120},
        ],
    }
    cfg.update(overrides)
    return cfg


def _vars(map_bank=10, map_num=7, flags=None):
    return {
        "X": 4,
        "Y": 3,
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
        "story_flags": flags if flags is not None else _flags(),
        "battle_type": 0,
        "johto_badges": 0,
        "player_state": 0,
        "key_items_count": 0,
        "game_hour": 0,
        "bgm_id": 0,
        "enemy_hp": 0,
        "enemy_max_hp": 20,
        "party_info": (1, 15, 40, 0),
        "script_active": False,
    }


def _run(rw, steps):
    return [float(rw.calculate_reward(v, "")[0]) for v in steps]


class TestFlagMilestone(unittest.TestCase):
    def test_honest_pays_the_milestone_price_over_several_steps(self):
        rw = Rewards(_cfg())
        out = _run(rw, [_vars()] + [_vars(flags=_flags(FALKNER))] * 6)
        self.assertEqual(out[1], 50.0)  # clipped on the fire step...
        self.assertAlmostEqual(sum(out), 200.0, places=4)  # ...but fully delivered
        self.assertAlmostEqual(rw.get_episode_breakdown()["checkpoint_honest"], 200.0, places=4)

    def test_seeded_keeps_the_normal_checkpoint_price(self):
        rw = Rewards(_cfg(episode_seeded=True))
        out = _run(rw, [_vars()] + [_vars(flags=_flags(FALKNER))] * 3)
        self.assertAlmostEqual(sum(out), 10.0, places=4)

    def test_unlisted_checkpoint_keeps_the_default_price(self):
        rw = Rewards(_cfg())
        out = _run(rw, [_vars()] + [_vars(flags=_flags(1214))] * 3)
        self.assertAlmostEqual(sum(out), 40.0, places=4)

    def test_non_checkpoint_flag_is_rejected(self):
        with self.assertRaises(ValueError):
            Rewards(_cfg(story_milestones=[{"type": "flag", "flag_num": 8, "reward": 10}]))


class TestMapMilestone(unittest.TestCase):
    def test_first_entry_pays_once_in_any_order(self):
        rw = Rewards(_cfg(reward_clip=1000))
        out = _run(
            rw,
            [
                _vars(10, 7),
                _vars(8, 7),  # Azalea before Union Cave: order is free
                _vars(3, 37),
                _vars(8, 7),  # re-entry pays nothing
                _vars(3, 37),
            ],
        )
        self.assertEqual(out, [0.0, 120.0, 150.0, 0.0, 0.0])

    def test_start_map_never_pays(self):
        rw = Rewards(_cfg(reward_clip=1000))
        out = _run(rw, [_vars(3, 37), _vars(3, 38), _vars(3, 37)])
        self.assertEqual(out, [0.0, 0.0, 0.0])

    def test_seeded_entry_pays_the_seeded_price(self):
        rw = Rewards(_cfg(reward_clip=1000, episode_seeded=True))
        out = _run(rw, [_vars(10, 7), _vars(3, 37)])
        self.assertAlmostEqual(out[1], 10.0, places=4)

    def test_resets_each_episode(self):
        rw = Rewards(_cfg(reward_clip=1000))
        _run(rw, [_vars(10, 7), _vars(3, 37)])
        rw.start_new_episode()
        out = _run(rw, [_vars(10, 7), _vars(3, 37)])
        self.assertEqual(out[1], 150.0)


class TestClipCarry(unittest.TestCase):
    def test_non_milestone_spike_is_still_clipped(self):
        rw = Rewards(_cfg(story_milestones=[], level_up_reward=500))
        rw.calculate_reward(_vars(), "")
        spike = dict(_vars(), party_info=(1, 16, 40, 0))
        out = _run(rw, [spike, spike, spike])
        self.assertEqual(out[0], 50.0)
        self.assertEqual(out[1:], [0.0, 0.0])

    def test_breakdown_conserves_delivered_reward(self):
        rw = Rewards(_cfg())
        _run(rw, [_vars()] + [_vars(flags=_flags(FALKNER))] * 2)
        bd = rw.get_episode_breakdown()
        sources = sum(
            v
            for k, v in bd.items()
            if k
            not in (
                "delivered_total",
                "rounding_adjustment",
                "clip_adjustment",
                "delivery_cast_adjustment",
                "domain_scale_adjustment",
            )
        )
        adjustments = bd["rounding_adjustment"] + bd["clip_adjustment"]
        self.assertAlmostEqual(bd["delivered_total"], sources + adjustments, places=3)


class TestOpenWorldConfig(unittest.TestCase):
    def test_milestones_and_continuation_state_resolve(self):
        repo = os.path.join(os.path.dirname(__file__), "..")
        with open(os.path.join(repo, "configs", "stages", "open_world.json")) as f:
            cfg = json.load(f)
        Rewards(dict(_cfg(), story_milestones=cfg["story_milestones"]))
        self.assertTrue(
            os.path.exists(os.path.join(repo, cfg["continuation_probe_state_path"])),
            "continuation probe save-state missing (tools/save_state_from_actions.py)",
        )


if __name__ == "__main__":
    unittest.main()
