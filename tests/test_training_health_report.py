# -*- coding: utf-8 -*-
"""tools/training_health_report.py — offline info.pth health-check.

Pure synthetic-data tests: no PyBoy, no training run, no torch.save/load
round trip (the report functions take an already-loaded episode_data
dict). Validates the plateau/farming/frontier heuristics against
hand-built series that pin the exact failure signatures this project has
hit before (milestone re-fire farming, archive-growth plateau, a probe
horizon too short to see an already-learned capability).
"""
import io
import statistics
import unittest
from contextlib import redirect_stdout

from tools.training_health_report import (
    report_reward_mix, report_archive_growth, report_time_to_frontier,
    report_probes, _quartile_slices,
)


def _capture(fn, *args, **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


class TestQuartileSlices(unittest.TestCase):
    def test_splits_into_four_contiguous_chunks(self):
        chunks = _quartile_slices(list(range(12)), 4)
        self.assertEqual([list(c) for c in chunks], [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])

    def test_empty_series_returns_empty(self):
        self.assertEqual(_quartile_slices([], 4), [])

    def test_non_divisible_length_still_returns_four_slices(self):
        """Regression (2026-07-15): stepping range(0, n, size) instead of
        indexing by slice number produced a stray 5th slice whenever n
        wasn't an exact multiple of n_parts (e.g. n=13 -> size=3 ->
        slices at 0,3,6,9,12 -> a 1-element final sliver). Every caller
        treats quarters[-1] as "the last quarter" for a trend comparison;
        a stray sliver silently turned that into "the single last
        episode" instead — on a real 3825-episode run this reported one
        pathological outlier episode's stats as if they were a 956-episode
        quarter average."""
        chunks = _quartile_slices(list(range(13)), 4)
        self.assertEqual(len(chunks), 4)
        # last slice absorbs the remainder rather than spilling into a 5th.
        self.assertEqual([list(c) for c in chunks],
                          [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11, 12]])


class TestRewardMixTrend(unittest.TestCase):
    def _episode_data(self, sources):
        return {"episode_reward_sources": sources}

    def test_flags_farming_when_milestone_holds_flat_relative_to_frontier(self):
        # The 2026-07-11 "talked_to_mom" signature: flag reward never
        # decays and frontier stays tiny throughout.
        sources = [
            {"flag": 500.0, "frontier": 10.0} for _ in range(200)
        ]
        out = _capture(report_reward_mix, self._episode_data(sources))
        self.assertIn("WARNING", out)
        self.assertIn("farming", out)

    def test_ok_when_frontier_overtakes_decaying_milestone(self):
        sources = (
            [{"flag": 500.0, "frontier": 20.0} for _ in range(50)]
            + [{"flag": 30.0, "frontier": 90.0} for _ in range(50)]
        )
        out = _capture(report_reward_mix, self._episode_data(sources))
        self.assertIn("OK", out)
        self.assertNotIn("WARNING", out)

    def test_ok_when_both_channels_decline_but_milestone_is_depleting(self):
        # The real shape seen in every run so far: episode 0 is a one-time
        # bootstrap bonanza (everything is "first visit", full price), then
        # BOTH milestone and frontier fall together as re-fires/re-visits
        # deplete — frontier is NOT expected to grow monotonically (it only
        # pays for genuinely fresh cells), so "frontier hasn't overtaken
        # milestone" alone must not read as a farming warning.
        sources = (
            [{"flag": 143.0, "frontier": 144.0} for _ in range(50)]
            + [{"flag": 51.0, "frontier": 51.0} for _ in range(50)]
        )
        out = _capture(report_reward_mix, self._episode_data(sources))
        self.assertIn("OK", out)
        self.assertIn("depleting as designed", out)
        self.assertNotIn("WARNING", out)


class TestArchiveGrowth(unittest.TestCase):
    def test_flags_plateau_when_final_quarter_is_flat(self):
        sizes = list(range(0, 50, 2)) + [50] * 25  # grows, then flatlines
        out = _capture(report_archive_growth, {"episode_archive_size": sizes})
        self.assertIn("WARNING", out)
        self.assertIn("plateau", out)

    def test_no_warning_when_still_growing(self):
        sizes = list(range(0, 100, 2))
        out = _capture(report_archive_growth, {"episode_archive_size": sizes})
        self.assertNotIn("WARNING", out)


class TestTimeToFrontier(unittest.TestCase):
    def test_max_not_median_drives_the_trigger(self):
        # Hundreds of shallow (mom-only) episodes plus a handful of deep
        # breakthroughs — median alone would hide the real frontier depth
        # (this was a bug caught by running the tool against a real run).
        fire_series = [[[1735, 190]] for _ in range(260)]
        fire_series += [[[1735, 190], [26, 1900]] for _ in range(6)]
        out = _capture(report_time_to_frontier, {"episode_flag_fire_steps": fire_series}, 2048)
        self.assertIn("MAX deepest fire step ever reached: 1900", out)
        self.assertIn("TRIGGER", out)

    def test_no_trigger_when_frontier_well_within_budget(self):
        fire_series = [[[1735, 190]] for _ in range(50)]
        out = _capture(report_time_to_frontier, {"episode_flag_fire_steps": fire_series}, 2048)
        self.assertNotIn("TRIGGER", out)

    def test_no_crash_with_no_fires(self):
        out = _capture(report_time_to_frontier, {"episode_flag_fire_steps": []}, 2048)
        self.assertIn("no flag fires", out)


class TestProbeComparison(unittest.TestCase):
    def test_notes_when_long_probe_reveals_hidden_capability(self):
        ed = {
            "probe_rollout_idx": [100],
            "probe_goal_type_rates": [{"pokedex": 0.0}],
            "long_probe_rollout_idx": [100],
            "long_probe_goal_type_rates": [{"pokedex": 0.8}],
        }
        out = _capture(report_probes, ed, 512, 4096)
        self.assertIn("NOTE", out)
        self.assertIn("pokedex", out)

    def test_no_note_when_rates_track_each_other(self):
        ed = {
            "probe_rollout_idx": [100],
            "probe_goal_type_rates": [{"flag": 1.0}],
            "long_probe_rollout_idx": [100],
            "long_probe_goal_type_rates": [{"flag": 1.0}],
        }
        out = _capture(report_probes, ed, 512, 4096)
        self.assertNotIn("NOTE", out)

    def test_handles_missing_long_probe_gracefully(self):
        ed = {"probe_rollout_idx": [1], "probe_goal_type_rates": [{"flag": 1.0}]}
        out = _capture(report_probes, ed, 512, None)
        self.assertIn("no long-horizon probe data", out)


if __name__ == "__main__":
    unittest.main()
