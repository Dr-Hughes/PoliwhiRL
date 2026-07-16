# -*- coding: utf-8 -*-
"""Persistent-count-gated frontier novelty + observation feature, and the
persistent-archive plumbing behind it.

Pinned behaviours:

- Frontier novelty pays `frontier_novelty_bonus / sqrt(1 + persistent_visits)`
  on first entry to a cell each episode, reading the run-wide VisitArchive
  count — so re-covering known ground stops being income across episodes
  while a never-visited cell always pays in full.
- sqrt decay: known ground keeps a small residual payout forever (never
  fully dark), it just decays toward zero as visits accumulate.
- _cells_to_record is populated (previously dead code) so the persistent
  VisitArchive has real cell data to read; it resets every episode and
  only records each cell once.
- _new_map_bonus now queues a map's bookkeeping (explored_maps /
  _recent_maps_list / _maps_to_record) even when the first entry happens
  during a script_active frame — only the reward itself is skipped.
- The new `global_cell_visit_count` RAM feature reflects the persistent
  archive count for the current cell, log1p-scaled, appended at the end of
  RAM_FEATURE_KEYS per the append-only contract.
"""
import unittest
import numpy as np

from PoliwhiRL.environment.rewards import Rewards
from PoliwhiRL.environment.visit_archive import VisitArchive
from PoliwhiRL.environment.gym_env import RAM_FEATURE_KEYS, RAM_FEATURE_INDEX, _build_ram_vector


def _zero_flags():
    return np.zeros(256, dtype=np.uint8)


def _base_config(**overrides):
    cfg = {
        "episode_length": 1000,
        "new_map_reward": 0,
        "frontier_novelty_bonus": 10.0,
        "whiteout_penalty": 0,
        "step_penalty": 0.0,
        "reward_round_dp": None,
        "goals": [],
    }
    cfg.update(overrides)
    return cfg


def _env_vars(x=0, y=0, map_bank=24, map_num=7, script_active=False):
    return {
        "X": x, "Y": y, "map_num": map_num, "map_bank": map_bank,
        "room": 0, "warp_number": 0, "money": 0,
        "pokedex_seen": 0, "pokedex_owned": 0,
        "collision_down": 0, "collision_up": 0,
        "collision_left": 0, "collision_right": 0,
        "story_flags": _zero_flags(),
        "battle_type": 0, "johto_badges": 0, "player_state": 0,
        "key_items_count": 0, "game_hour": 0, "bgm_id": 0,
        "enemy_hp": 0, "enemy_max_hp": 20,
        "party_info": (1, 5, 20, 0),
        "script_active": script_active,
    }


class TestFrontierGating(unittest.TestCase):
    def test_never_visited_cell_pays_full_bonus(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=0), "")
        bd = rw.get_episode_breakdown()
        self.assertAlmostEqual(bd["frontier"], 10.0, places=4)

    def test_persistent_visits_decay_the_payout_by_sqrt(self):
        archive = VisitArchive()
        cell = archive.cell_key(24, 7, 0, 0)
        for _ in range(3):
            archive.merge_visits([cell], [])
        rw = Rewards(_base_config(), visit_archive=archive)
        rw.calculate_reward(_env_vars(x=0), "")
        bd = rw.get_episode_breakdown()
        # 10 / sqrt(1 + 3) = 5.0
        self.assertAlmostEqual(bd["frontier"], 5.0, places=4)

    def test_heavily_visited_cell_pays_the_episodic_floor(self):
        """Cross-run decay never suppresses a first-visit-this-episode
        payout below bonus * frontier_novelty_floor: with 999 visits the
        sqrt term (~0.32) is under the floor (2.0 at the 0.2 default), so
        the floor pays. This is the episodic-curiosity guarantee — known
        ground decays toward the floor, never toward zero (the ep_104
        reward-desert fix, 2026-07-12)."""
        archive = VisitArchive()
        cell = archive.cell_key(24, 7, 0, 0)
        for _ in range(999):
            archive.merge_visits([cell], [])
        rw = Rewards(_base_config(), visit_archive=archive)
        rw.calculate_reward(_env_vars(x=0), "")
        bd = rw.get_episode_breakdown()
        self.assertAlmostEqual(bd["frontier"], 10.0 * 0.2, places=4)

    def test_fresh_ground_outpays_known_ground(self):
        """The point of the gating: on any marginal step, a never-visited
        cell must pay strictly more than a run-familiar one (floored at
        bonus * floor, so the max preference ratio is 1/floor = 5x)."""
        archive = VisitArchive()
        known = archive.cell_key(24, 7, 0, 0)
        for _ in range(50):
            archive.merge_visits([known], [])
        rw = Rewards(_base_config(), visit_archive=archive)
        rw.calculate_reward(_env_vars(x=0), "")
        known_pay = rw.get_episode_breakdown()["frontier"]
        # x=10 quantises to a different, never-visited cell.
        rw.calculate_reward(_env_vars(x=10), "")
        fresh_pay = rw.get_episode_breakdown()["frontier"] - known_pay
        self.assertGreater(fresh_pay, known_pay * 4)

    def test_same_cell_pays_at_most_once_per_episode(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=0), "")
        first = rw.get_episode_breakdown()["frontier"]
        rw.calculate_reward(_env_vars(x=0), "")
        self.assertAlmostEqual(rw.get_episode_breakdown()["frontier"], first, places=4)

    def test_disabled_via_zero_bonus(self):
        rw = Rewards(_base_config(frontier_novelty_bonus=0))
        rw.calculate_reward(_env_vars(x=0), "")
        self.assertAlmostEqual(rw.get_episode_breakdown()["frontier"], 0.0, places=4)

    def test_gated_on_script_active(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=0, script_active=True), "")
        self.assertAlmostEqual(rw.get_episode_breakdown()["frontier"], 0.0, places=4)


class TestCellsToRecordPlumbing(unittest.TestCase):
    def test_novel_cell_is_queued_for_recording(self):
        rw = Rewards(_base_config())
        cell = rw.visit_archive.cell_key(24, 7, 0, 0)
        rw.calculate_reward(_env_vars(x=0), "")
        self.assertIn(cell, rw._cells_to_record)

    def test_same_cell_only_queued_once_per_episode(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=0), "")
        rw.calculate_reward(_env_vars(x=0), "")
        cell = rw.visit_archive.cell_key(24, 7, 0, 0)
        self.assertEqual(
            sum(1 for c in rw._cells_to_record if c == cell), 1
        )

    def test_cells_to_record_resets_on_new_episode(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=0), "")
        self.assertTrue(rw._cells_to_record)
        rw.start_new_episode()
        self.assertEqual(rw._cells_to_record, set())

    def test_script_active_cell_not_queued(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=0, script_active=True), "")
        self.assertEqual(rw._cells_to_record, set())


class TestNewMapBonusScriptActiveBookkeeping(unittest.TestCase):
    """The independent bug: a map's first-ever entry landing on a
    script_active frame must not silently drop it from the persistent
    archive forever."""

    def _config(self, **overrides):
        return _base_config(
            new_map_reward=50, frontier_novelty_bonus=0,
            **overrides,
        )

    def test_script_active_first_entry_pays_no_reward(self):
        rw = Rewards(self._config())
        r, _ = rw.calculate_reward(_env_vars(map_bank=26, map_num=3, script_active=True), "")
        self.assertAlmostEqual(float(r), 0.0, places=4)

    def test_script_active_first_entry_still_recorded_in_maps_to_record(self):
        rw = Rewards(self._config())
        rw.calculate_reward(_env_vars(map_bank=26, map_num=3, script_active=True), "")
        self.assertIn((26, 3), rw._maps_to_record)

    def test_script_active_first_entry_still_marks_explored(self):
        """A second entry (even without script_active) must not re-pay —
        proves explored_maps bookkeeping happened on the scripted step."""
        rw = Rewards(self._config())
        rw.calculate_reward(_env_vars(map_bank=26, map_num=3, script_active=True), "")
        r, _ = rw.calculate_reward(_env_vars(map_bank=26, map_num=3, script_active=False), "")
        self.assertAlmostEqual(float(r), 0.0, places=4)

    def test_non_script_active_entry_still_pays(self):
        """Regression guard: the normal (non-scripted) first-entry path is
        unchanged by the fix."""
        rw = Rewards(self._config())
        r, _ = rw.calculate_reward(_env_vars(map_bank=26, map_num=3, script_active=False), "")
        self.assertAlmostEqual(float(r), 50.0, places=4)

    def test_run_familiar_map_first_entry_pays_the_episodic_floor(self):
        """Same decay-plus-floor rule at map granularity: a map entered in
        hundreds of prior episodes still pays new_map_reward * floor on
        first entry each episode, never ~zero."""
        archive = VisitArchive()
        for _ in range(199):
            archive.merge_visits([], [(26, 3)])
        rw = Rewards(self._config(), visit_archive=archive)
        r, _ = rw.calculate_reward(
            _env_vars(map_bank=26, map_num=3, script_active=False), ""
        )
        # 50 / 200 = 0.25 < 50 * 0.2 = 10.0 -> floor pays.
        self.assertAlmostEqual(float(r), 10.0, places=4)


class TestGlobalCellVisitCountRamFeature(unittest.TestCase):
    def test_feature_present_immediately_after_steps_since_novel_cell(self):
        """Append-only contract: the new feature must sit right after the
        last previously-final base feature, not be inserted earlier."""
        self.assertIn("global_cell_visit_count", RAM_FEATURE_KEYS)
        prev_idx = RAM_FEATURE_INDEX["steps_since_novel_cell"]
        new_idx = RAM_FEATURE_INDEX["global_cell_visit_count"]
        self.assertEqual(new_idx, prev_idx + 1)

    def test_zero_visits_gives_zero_feature(self):
        rw = Rewards(_base_config())
        env_vars = _env_vars(x=0)
        count = rw.global_cell_visit_count(env_vars)
        self.assertEqual(count, 0)

    def test_high_visits_gives_large_positive_feature(self):
        archive = VisitArchive()
        cell = archive.cell_key(24, 7, 0, 0)
        for _ in range(999):
            archive.merge_visits([cell], [])
        rw = Rewards(_base_config(), visit_archive=archive)
        env_vars = _env_vars(x=0)
        count = rw.global_cell_visit_count(env_vars)
        self.assertEqual(count, 999)

    def test_build_ram_vector_scales_the_feature(self):
        idx = RAM_FEATURE_INDEX["global_cell_visit_count"]

        env_vars = _env_vars(x=0)
        vec_zero = _build_ram_vector(
            env_vars, 0, 0, 0, 0, 0, 0, (0, 5, 128),
            global_cell_visit_count=0,
        )
        vec_many = _build_ram_vector(
            env_vars, 0, 0, 0, 0, 0, 0, (0, 5, 128),
            global_cell_visit_count=999,
        )
        self.assertAlmostEqual(vec_zero[idx], 0.0, places=4)
        self.assertGreater(vec_many[idx], vec_zero[idx])


def _set_flag(arr, flag_num):
    arr = arr.copy()
    byte_idx, bit_idx = flag_num // 8, flag_num % 8
    arr[byte_idx] |= (1 << bit_idx)
    return arr


class TestDiscoveryLog(unittest.TestCase):
    """Diagnostic-only run-wide first-ever-milestone log (see
    Rewards._discoveries_this_episode / get_discoveries) — never affects
    reward or observations, only feeds post-hoc discovery-order analysis."""

    def test_flag_fire_is_logged(self):
        rw = Rewards(_base_config())
        ev = _env_vars(x=0)
        ev["story_flags"] = _zero_flags()
        rw.calculate_reward(ev, "")  # seeds baseline
        ev["story_flags"] = _set_flag(_zero_flags(), 26)
        rw.calculate_reward(ev, "")
        d = rw.get_discoveries()
        flags = [r for r in d if r["type"] == "flag"]
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["key"], 26)

    def test_map_logged_only_on_true_first_ever_entry(self):
        archive = VisitArchive()
        # First-ever entry to (26, 3): logged.
        rw1 = Rewards(_base_config(new_map_reward=50), visit_archive=archive)
        rw1.calculate_reward(_env_vars(map_bank=26, map_num=3), "")
        maps1 = [r for r in rw1.get_discoveries() if r["type"] == "map"]
        self.assertEqual(len(maps1), 1)
        self.assertEqual(maps1[0]["key"], [26, 3])
        archive.merge_visits([], rw1._maps_to_record)

        # A later episode's first-this-episode (but not first-ever) entry
        # to the same map must NOT be logged as a discovery.
        rw2 = Rewards(_base_config(new_map_reward=50), visit_archive=archive)
        rw2.calculate_reward(_env_vars(map_bank=26, map_num=3), "")
        maps2 = [r for r in rw2.get_discoveries() if r["type"] == "map"]
        self.assertEqual(len(maps2), 0)

    def test_pokedex_owned_logged(self):
        rw = Rewards(_base_config())
        ev = _env_vars(x=0)
        ev["pokedex_owned"] = 0
        rw.calculate_reward(ev, "")
        ev2 = _env_vars(x=0)
        ev2["pokedex_owned"] = 1
        rw.calculate_reward(ev2, "")
        owned = [r for r in rw.get_discoveries() if r["type"] == "pokedex_owned"]
        self.assertEqual(len(owned), 1)
        self.assertEqual(owned[0]["key"], 1)

    def test_level_gain_logged(self):
        """key is the new absolute level reached, not the delta gained —
        deltas reset every episode (no snapshot seeding) so a delta alone
        can't distinguish genuine run-wide progress from routine replay."""
        rw = Rewards(_base_config())
        ev = dict(_env_vars(x=0))
        ev["party_info"] = (1, 5, 20, 0)
        rw.calculate_reward(ev, "")
        ev2 = dict(_env_vars(x=0))
        ev2["party_info"] = (1, 6, 20, 0)
        rw.calculate_reward(ev2, "")
        levels = [r for r in rw.get_discoveries() if r["type"] == "level"]
        self.assertEqual(len(levels), 1)
        self.assertEqual(levels[0]["key"], 6)

    def test_flag_not_relogged_once_merged_into_persistent_archive(self):
        """The bug this class exists to pin: every episode restarts from
        scratch, so flag/pokedex/level/key_item fire fresh EVERY episode —
        that's correct for reward, but the discovery log must only log the
        true run-wide first time, using visit_archive (persistent across
        episodes), not the per-episode reward tracker (which resets)."""
        archive = VisitArchive()

        rw1 = Rewards(_base_config(), visit_archive=archive)
        ev = _env_vars(x=0)
        ev["story_flags"] = _zero_flags()
        rw1.calculate_reward(ev, "")
        ev["story_flags"] = _set_flag(_zero_flags(), 26)
        rw1.calculate_reward(ev, "")
        self.assertEqual(len([d for d in rw1.get_discoveries() if d["type"] == "flag"]), 1)
        archive.merge_milestones(**rw1.get_milestone_state())

        # A second, independent episode re-fires the exact same flag from
        # scratch (as every episode does) — must NOT be logged again.
        rw2 = Rewards(_base_config(), visit_archive=archive)
        ev2 = _env_vars(x=0)
        ev2["story_flags"] = _zero_flags()
        rw2.calculate_reward(ev2, "")
        ev2["story_flags"] = _set_flag(_zero_flags(), 26)
        rw2.calculate_reward(ev2, "")
        self.assertEqual(len([d for d in rw2.get_discoveries() if d["type"] == "flag"]), 0)

    def test_pokedex_owned_not_relogged_once_merged(self):
        archive = VisitArchive()

        rw1 = Rewards(_base_config(), visit_archive=archive)
        ev = _env_vars(x=0)
        ev["pokedex_owned"] = 0
        rw1.calculate_reward(ev, "")
        ev["pokedex_owned"] = 1
        rw1.calculate_reward(ev, "")
        self.assertEqual(len([d for d in rw1.get_discoveries() if d["type"] == "pokedex_owned"]), 1)
        archive.merge_milestones(**rw1.get_milestone_state())

        rw2 = Rewards(_base_config(), visit_archive=archive)
        ev2 = _env_vars(x=0)
        ev2["pokedex_owned"] = 0
        rw2.calculate_reward(ev2, "")
        ev2["pokedex_owned"] = 1
        rw2.calculate_reward(ev2, "")
        self.assertEqual(len([d for d in rw2.get_discoveries() if d["type"] == "pokedex_owned"]), 0)

    def test_get_milestone_state_reflects_episode_maxima(self):
        rw = Rewards(_base_config())
        ev = dict(_env_vars(x=0))
        ev["party_info"] = (1, 5, 20, 0)
        ev["pokedex_owned"] = 2
        ev["key_items_count"] = 1
        ev["story_flags"] = _zero_flags()
        rw.calculate_reward(ev, "")  # seeds baselines
        ev["story_flags"] = _set_flag(_zero_flags(), 26)
        rw.calculate_reward(ev, "")
        state = rw.get_milestone_state()
        self.assertEqual(state["flags_fired"], [26])
        self.assertEqual(state["pokedex_owned_max"], 2)
        self.assertEqual(state["level_max"], 5)
        self.assertEqual(state["key_items_max"], 1)

    def test_visit_archive_merge_milestones_round_trip_through_checkpoint_state(self):
        archive = VisitArchive()
        archive.merge_milestones([26, 27], 3, 2, 7, 1)
        state = archive.to_state()
        restored = VisitArchive()
        restored.load_state(state)
        self.assertTrue(restored.flag_ever_fired(26))
        self.assertTrue(restored.flag_ever_fired(27))
        self.assertFalse(restored.flag_ever_fired(99))
        self.assertEqual(restored.pokedex_seen_max(), 3)
        self.assertEqual(restored.pokedex_owned_max(), 2)
        self.assertEqual(restored.level_max(), 7)
        self.assertEqual(restored.key_items_max(), 1)

    def test_merge_milestones_reports_change_for_dirty_broadcast(self):
        """The agent only re-broadcasts the archive to worker replicas when
        it's dirty. merge_milestones must report whether anything changed —
        otherwise a milestone achieved after the cell archive saturates
        (when nothing else sets the dirty flag) never reaches the workers
        and dedup re-logs it every episode forever."""
        archive = VisitArchive()
        self.assertTrue(archive.merge_milestones([26], 1, 0, 5, 0))
        # Identical merge: nothing new → no broadcast needed.
        self.assertFalse(archive.merge_milestones([26], 1, 0, 5, 0))
        # Strictly-lower maxima: also no change.
        self.assertFalse(archive.merge_milestones([], 0, 0, 3, 0))
        # A new flag alone is a change.
        self.assertTrue(archive.merge_milestones([27], 0, 0, 0, 0))
        # A higher maximum alone is a change.
        self.assertTrue(archive.merge_milestones([], 2, 0, 0, 0))

    def test_dedup_end_to_end_through_broadcast_replica(self):
        """Full loop: worker fires flag → agent merges ledger → broadcast
        state → a WORKER REPLICA loaded from that state must suppress the
        re-log when a later episode fires the same flag again."""
        canonical = VisitArchive()

        # Episode 1 on some worker: flag 26 fires, discovery logged.
        rw1 = Rewards(_base_config())
        ev = _env_vars(x=0)
        ev["story_flags"] = _zero_flags()
        rw1.calculate_reward(ev, "")
        ev["story_flags"] = _set_flag(_zero_flags(), 26)
        rw1.calculate_reward(ev, "")
        self.assertEqual(
            len([d for d in rw1.get_discoveries() if d["type"] == "flag"]), 1)

        # Agent side: merge + broadcast (to_state → replica load_state).
        changed = canonical.merge_milestones(**rw1.get_milestone_state())
        self.assertTrue(changed)
        replica = VisitArchive()
        replica.load_state(canonical.to_state())

        # Episode 2 on another worker, reading the refreshed replica: the
        # same flag re-fires (every episode does, from scratch) but must
        # NOT be re-logged as a discovery.
        rw2 = Rewards(_base_config(), visit_archive=replica)
        ev2 = _env_vars(x=0)
        ev2["story_flags"] = _zero_flags()
        rw2.calculate_reward(ev2, "")
        ev2["story_flags"] = _set_flag(_zero_flags(), 26)
        rw2.calculate_reward(ev2, "")
        self.assertEqual(
            len([d for d in rw2.get_discoveries() if d["type"] == "flag"]), 0)
        # The reward itself still pays (per-episode by design).
        self.assertEqual(rw2.get_milestone_state()["flags_fired"], [26])

    def test_discovery_log_resets_on_new_episode(self):
        rw = Rewards(_base_config(new_map_reward=50))
        rw.calculate_reward(_env_vars(map_bank=26, map_num=3), "")
        self.assertTrue(rw.get_discoveries())
        rw.start_new_episode()
        self.assertEqual(rw.get_discoveries(), [])

    def test_get_discoveries_returns_a_copy(self):
        rw = Rewards(_base_config(new_map_reward=50))
        rw.calculate_reward(_env_vars(map_bank=26, map_num=3), "")
        d = rw.get_discoveries()
        d.append({"type": "fake", "key": None, "step": 0})
        self.assertNotEqual(rw.get_discoveries(), d)


class TestDirectionalFrontierPotential(unittest.TestCase):
    """Rewards.directional_frontier_potential — the [up, down, left, right]
    forecast surfaced to the policy via the RAM vector's
    frontier_potential_* features, so it can perceive the exploration
    gradient directly instead of inferring it from reward after the fact."""

    def test_all_unvisited_directions_forecast_full_value(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=10, y=10), "")
        up, down, left, right = rw.directional_frontier_potential(
            _env_vars(x=10, y=10)
        )
        self.assertAlmostEqual(up, 1.0, places=4)
        self.assertAlmostEqual(down, 1.0, places=4)
        self.assertAlmostEqual(left, 1.0, places=4)
        self.assertAlmostEqual(right, 1.0, places=4)

    def test_heavily_visited_direction_forecasts_the_floor(self):
        """The forecast must mirror the floored payout exactly — a
        run-familiar (but unclaimed-this-episode) cell forecasts the
        episodic floor, not ~zero, matching what stepping there pays."""
        archive = VisitArchive()
        # Up from (10, 10) is (10, 8) at CELL_SIZE 2.
        up_cell = archive.cell_key(24, 7, 10, 8)
        for _ in range(99):
            archive.merge_visits([up_cell], [])
        rw = Rewards(_base_config(), visit_archive=archive)
        up, down, left, right = rw.directional_frontier_potential(
            _env_vars(x=10, y=10)
        )
        self.assertAlmostEqual(up, 0.2, places=4)
        self.assertAlmostEqual(down, 1.0, places=4)
        self.assertAlmostEqual(left, 1.0, places=4)
        self.assertAlmostEqual(right, 1.0, places=4)

    def test_zero_floor_restores_pure_sqrt_decay(self):
        """frontier_novelty_floor=0 must fully disable the floor (both in
        the payout and the forecast) — the pure-decay behaviour stays
        reachable via config."""
        archive = VisitArchive()
        cell = archive.cell_key(24, 7, 0, 0)
        up_cell = archive.cell_key(24, 7, 10, 8)
        for _ in range(999):
            archive.merge_visits([cell], [])
            archive.merge_visits([up_cell], [])
        rw = Rewards(
            _base_config(frontier_novelty_floor=0.0), visit_archive=archive
        )
        rw.calculate_reward(_env_vars(x=0), "")
        self.assertLess(rw.get_episode_breakdown()["frontier"], 0.5)
        up, _down, _left, _right = rw.directional_frontier_potential(
            _env_vars(x=10, y=10)
        )
        self.assertLess(up, 0.11)

    def test_direction_already_claimed_this_episode_forecasts_zero(self):
        rw = Rewards(_base_config())
        # Step onto (10, 8) this episode — the cell "up" from (10, 10).
        rw.calculate_reward(_env_vars(x=10, y=8), "")
        up, down, left, right = rw.directional_frontier_potential(
            _env_vars(x=10, y=10)
        )
        self.assertAlmostEqual(up, 0.0, places=4)
        self.assertAlmostEqual(down, 1.0, places=4)

    def test_lookahead_does_not_mutate_episode_state(self):
        # Read-only: calling the forecast must not itself claim the cell.
        rw = Rewards(_base_config())
        rw.directional_frontier_potential(_env_vars(x=10, y=10))
        rw.calculate_reward(_env_vars(x=10, y=8), "")
        self.assertAlmostEqual(rw.get_episode_breakdown()["frontier"], 10.0, places=4)


if __name__ == "__main__":
    unittest.main()
