# -*- coding: utf-8 -*-
"""Milestone re-fire depletion: milestones fire again every episode (every
episode restarts from the same save-state), and each fire pays
``base / sqrt(1 + prior_episode_fire_count)`` from the run-wide archive —
the frontier-novelty rule applied to milestones.

Pinned behaviours:

- A run-first milestone fire pays the full configured reward.
- A re-fire pays decayed by the archive's episode-fire count for that
  exact (kind, key) — farming the first corridor milestone stops being a
  full-price annuity (the 2026-07-11 "talked_to_mom" equilibrium) while
  never going fully dark (the corridor stays anchored).
- Fired (kind, key) events reach the archive via get_milestone_state →
  merge_milestones (one increment per milestone per episode), and
  merge_milestones reports a change so the agent re-broadcasts.
- Fire counts survive a to_state/load_state round trip (checkpoint +
  worker broadcast path).
- Pokédex/level/key-item thresholds are keyed on the absolute count
  reached: the re-farmed early thresholds decay while a new personal-best
  threshold still pays in full.
"""
import unittest
import numpy as np

from PoliwhiRL.environment.rewards import Rewards
from PoliwhiRL.environment.visit_archive import VisitArchive

MOM_FLAG = 1735  # EVENT_PLAYERS_HOUSE_MOM_1 — the farmed milestone


def _flags(*set_flags):
    arr = np.zeros(256, dtype=np.uint8)
    for f in set_flags:
        arr[f // 8] |= 1 << (f % 8)
    return arr


def _base_config(**overrides):
    cfg = {
        "episode_length": 1000,
        "new_map_reward": 0,
        "frontier_novelty_bonus": 0.0,
        "flag_progress_reward": 500,
        "pokedex_owned_reward": 150,
        "pokedex_first_sight_reward": 10,
        "level_up_reward": 10,
        "key_item_pickup_reward": 5,
        "whiteout_penalty": 0,
        "step_penalty": 0.0,
        "reward_round_dp": None,
        "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _env_vars(**overrides):
    ev = {
        "X": 0, "Y": 0, "map_num": 7, "map_bank": 24,
        "room": 0, "warp_number": 0, "money": 0,
        "pokedex_seen": 0, "pokedex_owned": 0,
        "collision_down": 0, "collision_up": 0,
        "collision_left": 0, "collision_right": 0,
        "story_flags": _flags(),
        "battle_type": 0, "johto_badges": 0, "player_state": 0,
        "key_items_count": 0, "game_hour": 0, "bgm_id": 0,
        "enemy_hp": 0, "enemy_max_hp": 20,
        "party_info": (1, 5, 20, 0),
        "script_active": False,
    }
    ev.update(overrides)
    return ev


def _run_mom_episode(archive):
    """One episode: baseline step, then the mom flag fires. Returns the
    flag-channel reward and merges the episode's milestones into archive
    the way VecPPOAgent does at episode end."""
    rw = Rewards(_base_config(), visit_archive=archive)
    rw.calculate_reward(_env_vars(), "")  # seeds the flag-byte baseline
    rw.calculate_reward(_env_vars(story_flags=_flags(MOM_FLAG)), "")
    paid = rw.get_episode_breakdown()["flag"]
    archive.merge_milestones(**rw.get_milestone_state())
    return paid


class TestFlagRefireDecay(unittest.TestCase):
    def test_first_fire_pays_full(self):
        self.assertAlmostEqual(_run_mom_episode(VisitArchive()), 500.0, places=4)

    def test_refire_decays_by_sqrt_of_episode_fires(self):
        archive = VisitArchive()
        paid = [_run_mom_episode(archive) for _ in range(4)]
        for k, p in enumerate(paid):
            self.assertAlmostEqual(p, 500.0 / np.sqrt(1 + k), places=4)

    def test_refire_never_goes_fully_dark(self):
        archive = VisitArchive()
        for _ in range(100):
            paid = _run_mom_episode(archive)
        self.assertGreater(paid, 0.0)

    def test_merge_reports_change_for_refire_of_known_flag(self):
        # Re-fires change no max/ever-fired entry, but the fire COUNT moved
        # — the agent must still mark the archive dirty and re-broadcast,
        # or worker replicas would keep paying stale (higher) values.
        archive = VisitArchive()
        _run_mom_episode(archive)
        rw = Rewards(_base_config(), visit_archive=archive)
        rw.calculate_reward(_env_vars(), "")
        rw.calculate_reward(_env_vars(story_flags=_flags(MOM_FLAG)), "")
        self.assertTrue(archive.merge_milestones(**rw.get_milestone_state()))

    def test_fire_counts_survive_state_round_trip(self):
        archive = VisitArchive()
        _run_mom_episode(archive)
        _run_mom_episode(archive)
        replica = VisitArchive()
        replica.load_state(archive.to_state())
        self.assertEqual(replica.milestone_fire_count("flag", MOM_FLAG), 2)
        self.assertAlmostEqual(
            _run_mom_episode(replica), 500.0 / np.sqrt(3), places=4
        )


class TestThresholdChannelsRefireDecay(unittest.TestCase):
    def _pokedex_episode(self, archive, owned):
        rw = Rewards(_base_config(), visit_archive=archive)
        rw.calculate_reward(_env_vars(), "")  # seeds baselines
        rw.calculate_reward(_env_vars(pokedex_owned=owned, pokedex_seen=owned), "")
        paid = rw.get_episode_breakdown()["pokedex"]
        archive.merge_milestones(**rw.get_milestone_state())
        return paid

    def test_pokedex_refarmed_threshold_decays_new_threshold_pays_full(self):
        archive = VisitArchive()
        first = self._pokedex_episode(archive, owned=1)
        self.assertAlmostEqual(first, 150.0 + 10.0, places=4)  # owned=1, seen=1
        second = self._pokedex_episode(archive, owned=1)
        self.assertAlmostEqual(second, (150.0 + 10.0) / np.sqrt(2), places=4)
        # Third episode reaches owned=2: threshold 1 decays further,
        # threshold 2 is a run-first and pays full.
        third = self._pokedex_episode(archive, owned=2)
        self.assertAlmostEqual(
            third, (150.0 + 10.0) / np.sqrt(3) + (150.0 + 10.0), places=4
        )

    def test_level_thresholds_pay_per_level_with_decay(self):
        archive = VisitArchive()
        rw = Rewards(_base_config(), visit_archive=archive)
        rw.calculate_reward(_env_vars(party_info=(1, 5, 20, 0)), "")
        # 5 -> 7 crosses thresholds 6 and 7, both run-first: full price.
        rw.calculate_reward(_env_vars(party_info=(1, 7, 20, 0)), "")
        self.assertAlmostEqual(rw.get_episode_breakdown()["level"], 20.0, places=4)
        archive.merge_milestones(**rw.get_milestone_state())

        rw2 = Rewards(_base_config(), visit_archive=archive)
        rw2.calculate_reward(_env_vars(party_info=(1, 5, 20, 0)), "")
        rw2.calculate_reward(_env_vars(party_info=(1, 6, 20, 0)), "")
        self.assertAlmostEqual(
            rw2.get_episode_breakdown()["level"], 10.0 / np.sqrt(2), places=4
        )

    def test_key_item_threshold_decay(self):
        archive = VisitArchive()
        for expected_scale in (1.0, 1.0 / np.sqrt(2)):
            rw = Rewards(_base_config(), visit_archive=archive)
            rw.calculate_reward(_env_vars(), "")
            rw.calculate_reward(_env_vars(key_items_count=1), "")
            self.assertAlmostEqual(
                rw.get_episode_breakdown()["key_item"],
                5.0 * expected_scale,
                places=4,
            )
            archive.merge_milestones(**rw.get_milestone_state())


if __name__ == "__main__":
    unittest.main()
