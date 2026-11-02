# -*- coding: utf-8 -*-
"""tools/training_health_report.py — offline info.pth health-check.

Pure synthetic-data tests: no PyBoy, no training run, no torch.save/load
round trip (the report functions take an already-loaded episode_data
dict). Validates the plateau/farming/frontier heuristics against
hand-built series that pin the exact failure signatures this project has
hit before (milestone re-fire farming, archive-growth plateau, a probe
horizon too short to see an already-learned capability). The Capacity tests
additionally pin report_capacity's present-data branches — run_config-derived
geometry, NaN exclusion, and the refusal to guess a threshold the run never
persisted — which no checkpoint on disk can reach yet.
"""
import argparse
import io
import re
import unittest
from contextlib import redirect_stdout
from unittest import mock

from tools.training_health_report import (
    report_entropy,
    report_reward_mix,
    report_archive_growth,
    report_time_to_frontier,
    report_probes,
    report_checkpoints,
    report_capacity,
    report_stall_location,
    report_trust_region,
    _quartile_slices,
    _map_name_table,
)


def _capture(fn, *args, **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


NAN = float("nan")


class _FakeTensor:
    """Stand-in for a torch tensor inside a synthetic state_dict.

    report_capacity only ever asks a state_dict entry for numel(), dim() and
    shape, so the Capacity tests need neither a torch import nor a
    torch.save/load round trip against a real checkpoint.
    """

    def __init__(self, shape):
        self.shape = tuple(shape)

    def numel(self):
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def dim(self):
        return len(self.shape)


def _state_dict(shapes):
    """Build a synthetic state_dict from ``{key: shape}``."""
    return {key: _FakeTensor(shape) for key, shape in shapes.items()}


def _capacity_args(**overrides):
    """Namespace holding every flag report_capacity reads, at main()'s defaults.

    The optional ones stay None so a test that expects a threshold to come from
    the run's own config can prove run_config — not a CLI default — supplied it.
    """
    base = dict(
        model_path=None,
        scaling_factor=None,
        conv_strides=None,
        update_frequency=256,
        target_kl=0.02,
        hard_kl=0.06,
        max_grad_norm=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _capacity(episode_data, d=None, state_dict=None, **arg_overrides):
    """Run report_capacity on synthetic inputs and return its stdout.

    ``_load_state_dict`` is patched out so no test touches torch or the
    filesystem. Passing no state_dict simulates the common real case: a
    checkpoint with no readable sibling actor_critic.pth.
    """
    loaded = (state_dict, None) if state_dict else (None, "synthetic test: no model file")
    with mock.patch("tools.training_health_report._load_state_dict", return_value=loaded):
        return _capture(
            report_capacity,
            episode_data,
            {} if d is None else d,
            _capacity_args(**arg_overrides),
            "/synthetic/Checkpoints/info.pth",
        )


_TABLE_ROW = re.compile(r"^  (\S.*?)\s+([\d,]+)\s+([\d.]+)%$")


def _table_rows(out):
    """The (label, count, share%) rows of any label/count/share table printed.

    Skips the ``transformer_blocks (all N)`` summary row, which restates two
    per-block rows and would otherwise double-count toward 200%.
    """
    rows = []
    for line in out.splitlines():
        match = _TABLE_ROW.match(line)
        if match and "(all" not in match.group(1):
            label, count, share = match.groups()
            rows.append((label, int(count.replace(",", "")), float(share)))
    return rows


def _subsection(out, header):
    """The body of one ``-- header --`` subsection, up to the next one.

    Three Capacity tests assert their own subsection emitted no WARNING at
    all. Since the Architecture provenance block now warns about a missing
    obs_resize_filter on EVERY pre-provenance run — which is every synthetic
    run_config in this file — that blanket check has to be scoped to the
    subsection under test rather than to the whole Capacity output.
    """
    body = []
    inside = False
    for line in out.splitlines():
        stripped = line.strip()
        if stripped == header:
            inside = True
            continue
        if inside and stripped.startswith("-- ") and stripped.endswith(" --"):
            break
        if inside:
            body.append(line)
    return "\n".join(body)


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
        self.assertEqual(
            [list(c) for c in chunks],
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11, 12]],
        )


class TestOptimisationReport(unittest.TestCase):
    def test_reports_kl_clip_and_learning_rate(self):
        episode_data = {
            "rollout_policy_entropy": [1.0, 0.9],
            "rollout_entropy_coef": [0.01, 0.01],
            "rollout_approx_kl": [0.002, 0.004],
            "rollout_clip_fraction": [0.03, 0.05],
            "rollout_lr": [3e-4, 2e-4],
        }

        out = _capture(report_entropy, episode_data, 0.6, 1.2, 1e-3, 0.1)

        self.assertIn("approximate KL", out)
        self.assertIn("clip fraction", out)
        self.assertIn("learning rate", out)

    def test_reports_replay_mismatch_and_rejection_trend(self):
        episode_data = {
            "rollout_policy_entropy": [1.0] * 120,
            "rollout_entropy_coef": [0.01] * 120,
            "rollout_base_replay_kl": [1e-8] * 60 + [2e-5] * 60,
            "rollout_behaviour_replay_kl": [2e-8] * 60 + [3e-5] * 60,
            "rollout_replay_rejected": [False] * 90 + [True] * 30,
        }

        out = _capture(report_entropy, episode_data, 0.6, 1.2, 1e-3, 0.1)

        self.assertIn("base collection/replay KL: mean=1.000e-05", out)
        self.assertIn("replay-rejected updates: 30/120 (25.0%)", out)
        self.assertIn("final 100: 30/100 (30.0%)", out)
        self.assertIn("replay rejection by quarter: 0.0%, 0.0%, 0.0%, 100.0%", out)
        self.assertIn("WARNING: replay rejection rate increased", out)

    def test_omits_replay_diagnostics_for_legacy_checkpoints(self):
        episode_data = {
            "rollout_policy_entropy": [1.0],
            "rollout_entropy_coef": [0.01],
        }

        out = _capture(report_entropy, episode_data, 0.6, 1.2, 1e-3, 0.1)

        self.assertNotIn("collection/replay KL", out)
        self.assertNotIn("replay-rejected updates", out)


class TestRewardMixTrend(unittest.TestCase):
    def _episode_data(self, sources):
        return {"episode_reward_sources": sources}

    def test_flags_farming_when_milestone_holds_flat_relative_to_frontier(self):
        # The 2026-07-11 "talked_to_mom" signature: flag reward never
        # decays and frontier stays tiny throughout.
        sources = [{"flag": 500.0, "frontier": 10.0} for _ in range(200)]
        out = _capture(report_reward_mix, self._episode_data(sources))
        self.assertIn("WARNING", out)
        self.assertIn("farming", out)

    def test_ok_when_frontier_overtakes_decaying_milestone(self):
        sources = [{"flag": 500.0, "frontier": 20.0} for _ in range(50)] + [
            {"flag": 30.0, "frontier": 90.0} for _ in range(50)
        ]
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
        sources = [{"flag": 143.0, "frontier": 144.0} for _ in range(50)] + [
            {"flag": 51.0, "frontier": 51.0} for _ in range(50)
        ]
        out = _capture(report_reward_mix, self._episode_data(sources))
        self.assertIn("OK", out)
        self.assertIn("depleting as designed", out)
        self.assertNotIn("WARNING", out)

    def test_exposes_delivery_adjustments_separately_from_sources(self):
        sources = [
            {
                "step_penalty": -0.005,
                "delivered_total": -0.01,
                "rounding_adjustment": -0.005,
                "clip_adjustment": 0.0,
                "delivery_cast_adjustment": 0.0,
            }
        ]
        out = _capture(report_reward_mix, self._episode_data(sources))
        self.assertIn("delivery accounting", out)
        self.assertIn("rounding_adjustment", out)
        self.assertIn("clip_adjustment", out)
        self.assertIn("delivery_cast_adjustment", out)
        self.assertIn("conservation error", out)
        self.assertNotIn("does not reconcile", out)

    def test_warns_when_delivery_accounting_does_not_conserve_reward(self):
        sources = [
            {
                "frontier": 1.0,
                "delivered_total": 0.5,
                "rounding_adjustment": 0.0,
                "clip_adjustment": 0.0,
            }
        ]
        out = _capture(report_reward_mix, self._episode_data(sources))
        self.assertIn("does not reconcile", out)


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
    def test_reports_named_checkpoint_goals(self):
        ed = {
            "probe_rollout_idx": [100],
            "probe_goal_rates": [
                {
                    "Got Mystery Egg From Mr Pokemon": 0.4,
                    "Gave Mystery Egg To Elm": 0.1,
                }
            ],
        }
        out = _capture(report_probes, ed, 8192, 24576)
        self.assertIn("Got Mystery Egg From Mr Pokemon=40%", out)
        self.assertIn("Gave Mystery Egg To Elm=10%", out)

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


class TestCheckpointReport(unittest.TestCase):
    def test_splits_honest_and_seeded_checkpoint_reach(self):
        ed = {
            "episode_flag_fire_steps": [
                [[30, 100]],
                [[30, 40], [31, 200]],
            ],
            "episode_seeded": [False, True],
        }
        out = _capture(report_checkpoints, ed, {"frontier_pool": [{}]})
        self.assertIn("Got Mystery Egg From Mr. Pokémon", out)
        self.assertIn("Gave Mystery Egg To Elm", out)
        self.assertIn("persisted Go-Explore frontier pool: 1", out)


class TestCapacityArchitecture(unittest.TestCase):
    """-- Architecture --: the group shares must account for the printed total,
    and the >=50% single-dense-matrix WARNING must fire on the shape the real
    open_world_v3 checkpoint has (cnn.fc holding 73%) but not on a balanced net.
    """

    BALANCED = _state_dict(
        {
            "cnn.block1.conv.weight": (16, 1, 3, 3),  # 144
            "cnn.block2.conv.weight": (32, 16, 3, 3),  # 4,608
            "cnn.fc.weight": (64, 128),  # 8,192
            "ram_encoder.0.weight": (64, 350),  # 22,400
            "transformer_blocks.0.att.q.weight": (128, 128),  # 16,384
            "transformer_blocks.1.att.q.weight": (128, 128),  # 16,384
        }
    )  # total 68,112; largest dense matrix is 32.9% of it

    CONCENTRATED = _state_dict(
        {
            "cnn.block1.conv.weight": (16, 1, 3, 3),  # 144
            "cnn.block2.conv.weight": (32, 16, 3, 3),  # 4,608
            "cnn.fc.weight": (256, 1024),  # 262,144  <- 72.2% of the total
            "ram_encoder.0.weight": (64, 350),  # 22,400
            "transformer_blocks.0.att.q.weight": (192, 192),  # 36,864
            "transformer_blocks.1.att.q.weight": (192, 192),  # 36,864
        }
    )  # total 363,024

    def test_group_shares_account_for_every_parameter(self):
        out = _capacity({}, state_dict=self.BALANCED)

        self.assertIn("total parameters: 68,112", out)
        rows = _table_rows(out)
        self.assertEqual(sum(count for _label, count, _s in rows), 68112)
        # Rounding each share to 0.1% independently can leave the column a few
        # tenths off 100; a whole missing group could not hide inside that.
        self.assertAlmostEqual(sum(share for _l, _c, share in rows), 100.0, delta=0.5)
        self.assertEqual(
            [label for label, _count, _share in rows],
            [
                "ram_encoder",
                "transformer_blocks[0]",
                "transformer_blocks[1]",
                "cnn.fc",
                "cnn convs",
            ],
        )

    def test_transformer_blocks_are_summed_in_an_extra_row_not_double_counted(self):
        out = _capacity({}, state_dict=self.BALANCED)

        summary = [line for line in out.splitlines() if "(all 2)" in line]
        self.assertEqual(len(summary), 1)
        self.assertEqual(" ".join(summary[0].split()), "transformer_blocks (all 2) 32,768 48.1%")
        # The summary row is on top of the two per-block rows, so the shares of
        # ALL printed rows sum to ~148%, not 100 — _table_rows drops it.
        self.assertAlmostEqual(sum(share for _l, _c, share in _table_rows(out)), 100.0, delta=0.5)

    def test_warns_when_one_dense_layer_holds_most_parameters(self):
        out = _capacity({}, state_dict=self.CONCENTRATED)

        self.assertIn("WARNING: one dense layer, cnn.fc.weight, holds 72.2% of all", out)
        self.assertIn("largest tensor: cnn.fc.weight (256x1024) = 262,144 params (72.2%)", out)

    def test_no_dense_concentration_warning_when_parameters_are_spread(self):
        out = _capacity({}, state_dict=self.BALANCED)

        self.assertNotIn("WARNING: one dense layer", out)
        self.assertIn("largest tensor: ram_encoder.0.weight (64x350) = 22,400 params (32.9%)", out)

    def test_unloadable_model_degrades_to_unavailable_rather_than_raising(self):
        out = _capacity({})

        self.assertIn("parameter counts: unavailable", out)
        self.assertIn("no loadable model state_dict next to info.pth", out)


class TestCapacityReceptiveField(unittest.TestCase):
    """-- Receptive field --: geometry comes from run_config when the run
    persisted it, from --conv-strides when it did not, and is refused outright
    when the stride count cannot be matched to the conv layers.
    """

    TWO_CONV_3X3 = _state_dict(
        {
            "cnn.block1.conv.weight": (16, 1, 3, 3),
            "cnn.block2.conv.weight": (32, 16, 3, 3),
        }
    )
    THREE_CONV_3X3 = _state_dict(
        {
            "cnn.block1.conv.weight": (16, 1, 3, 3),
            "cnn.block2.conv.weight": (32, 16, 3, 3),
            "cnn.block3.conv.weight": (64, 32, 3, 3),
        }
    )
    RUN_CONFIG = {"conv_strides": [2, 2], "scaling_factor": 0.5}

    def test_run_config_alone_derives_the_field_with_no_cli_flags(self):
        out = _capacity({}, d={"run_config": dict(self.RUN_CONFIG)}, state_dict=self.TWO_CONV_3X3)

        self.assertIn("strides used, both axes (from run_config): [2, 2]", out)
        # RF = 1 + (3-1)*1 + (3-1)*2 = 7 scaled px; /0.5 = 14 original px; /8 tiles.
        self.assertIn("receptive field: 7x7 px of the scaled CNN input", out)
        self.assertIn("receptive field: 14.0x14.0 px of the original screen", out)
        self.assertIn("receptive field: 1.75x1.75 Game Boy tiles", out)
        self.assertIn("WARNING: the visual stack sees under 2 tiles", out)

    def test_run_config_strides_beat_conflicting_cli_strides(self):
        out = _capacity(
            {},
            d={"run_config": dict(self.RUN_CONFIG)},
            state_dict=self.TWO_CONV_3X3,
            conv_strides=[3, 3],
        )

        self.assertIn("strides used, both axes (from run_config): [2, 2]", out)
        self.assertIn("receptive field: 7x7 px of the scaled CNN input", out)
        self.assertNotIn("9x9", out)

    def test_cli_strides_are_still_used_when_run_config_has_none(self):
        out = _capacity(
            {},
            d={"run_config": {"scaling_factor": 0.5}},
            state_dict=self.TWO_CONV_3X3,
            conv_strides=[2, 2],
            scaling_factor=None,
        )

        self.assertIn("strides used, both axes (from CLI): [2, 2]", out)
        self.assertIn("receptive field: 7x7 px of the scaled CNN input", out)

    def test_null_empty_or_mistyped_run_config_strides_fall_through_to_the_cli(self):
        # A Conv2d-free net persists conv_strides as None; an older run has no
        # key at all. Neither may be read as "zero strides".
        for persisted in (None, [], "2,2"):
            with self.subTest(persisted=persisted):
                out = _capacity(
                    {},
                    d={"run_config": {"conv_strides": persisted, "scaling_factor": 0.5}},
                    state_dict=self.TWO_CONV_3X3,
                    conv_strides=[2, 2],
                )
                self.assertIn("strides used, both axes (from CLI): [2, 2]", out)
                self.assertIn("receptive field: 7x7 px of the scaled CNN input", out)

    def test_stride_count_mismatch_refuses_to_guess(self):
        run_config = {"conv_strides": [2, 2, 2], "scaling_factor": 0.5}
        out = _capacity({}, d={"run_config": run_config}, state_dict=self.TWO_CONV_3X3)

        self.assertIn("run_config supplied 3 strides for 2 conv layers", out)
        self.assertIn("not guessing which layer takes which", out)
        self.assertNotIn("px of the scaled CNN input", out)
        self.assertNotIn("Game Boy tiles", out)

    def test_not_derivable_without_strides_and_names_run_config_as_the_fix(self):
        out = _capacity({}, state_dict=self.TWO_CONV_3X3)

        self.assertIn("receptive field: not derivable from a state_dict", out)
        self.assertIn(
            "kernels in definition order: cnn.block1.conv.weight=3x3, cnn.block2.conv.weight=3x3",
            out,
        )
        self.assertIn("a run_config carrying both conv_strides and scaling_factor", out)
        self.assertIn("--conv-strides with 2 comma-separated ints", out)
        self.assertIn("--scaling-factor", out)
        self.assertNotIn("px of the scaled CNN input", out)
        # The automatic path is offered ahead of the two hand-passed flags.
        self.assertLess(
            out.index("a run_config carrying both conv_strides"),
            out.index("--conv-strides with 2 comma-separated ints"),
        )

    def test_no_warning_once_the_field_covers_two_or_more_tiles(self):
        # 1 + 2*1 + 2*2 + 2*4 = 15 scaled px; /0.5 = 30 original px = 3.75 tiles.
        run_config = {"conv_strides": [2, 2, 2], "scaling_factor": 0.5}
        out = _capacity({}, d={"run_config": run_config}, state_dict=self.THREE_CONV_3X3)

        self.assertIn("receptive field: 15x15 px of the scaled CNN input", out)
        self.assertIn("receptive field: 30.0x30.0 px of the original screen", out)
        self.assertIn("receptive field: 3.75x3.75 Game Boy tiles", out)
        self.assertNotIn("WARNING: the visual stack sees under 2 tiles", out)

    def test_unreadable_conv_layers_report_why(self):
        out = _capacity({})

        self.assertIn("receptive field: unavailable (the model state_dict did not load)", out)


class TestCapacityValueFit(unittest.TestCase):
    """-- Value fit --: NaN means "degenerate minibatch", so it is dropped from
    the mean AND the count, and explained variance is never proxied from
    critic_loss when the series is missing or entirely NaN.
    """

    def test_nan_rollouts_are_excluded_from_both_mean_and_count(self):
        out = _capacity({"rollout_explained_variance": [0.9, NAN, 1.0, NAN]})

        self.assertIn("explained variance: mean 0.950 over 2/4 finite rollouts", out)
        self.assertNotIn("over 4/4", out)
        self.assertNotIn("mean 0.475", out)  # the NaN-as-zero reading

    def test_high_explained_variance_is_ok(self):
        out = _capacity({"rollout_explained_variance": [0.9, NAN, 1.0, NAN]})

        self.assertIn("OK: critic fit is not capacity-limited", out)
        self.assertNotIn("WARNING: critic underfits returns", out)

    def test_low_explained_variance_warns(self):
        out = _capacity({"rollout_explained_variance": [0.20, 0.30]})

        self.assertIn("explained variance: mean 0.250 over 2/2 finite rollouts", out)
        self.assertIn("WARNING: critic underfits returns", out)
        self.assertNotIn("OK: critic fit", out)

    def test_mid_explained_variance_is_neither_ok_nor_warning(self):
        out = _capacity({"rollout_explained_variance": [0.6, 0.7]})

        self.assertIn("critic only partially fits returns", out)
        self.assertNotIn("OK: critic fit", out)
        self.assertNotIn("WARNING: critic underfits returns", out)

    def test_all_nan_series_reports_the_count_and_never_a_numeric_proxy(self):
        ed = {"rollout_explained_variance": [NAN, NAN], "rollout_critic_loss": [0.5, 0.25]}
        out = _capacity(ed)

        self.assertIn("explained variance: all 2 logged rollouts are NaN", out)
        self.assertNotIn("explained variance: mean", out)
        self.assertNotIn("OK: critic fit", out)
        # critic loss is still quoted, but explicitly stripped of interpretation.
        self.assertIn("raw critic loss, mean 0.3750", out)
        self.assertIn("quoted with no interpretation", out)

    def test_absent_series_is_unavailable_and_not_reconstructed_from_critic_loss(self):
        out = _capacity({"rollout_critic_loss": [0.5, 0.25]})

        self.assertIn("explained variance: unavailable for this run", out)
        self.assertIn("not reconstructed from critic_loss", out)
        self.assertIn("any such ratio here would be fabricated", out)
        self.assertNotIn("explained variance: mean", out)

    def test_cross_check_spans_only_the_rollouts_where_variance_is_finite(self):
        ed = {
            "rollout_explained_variance": [0.9, NAN, 0.9],
            "rollout_value_error_rmse": [0.1, NAN, 0.1],
            "rollout_return_std": [1.0, NAN, 1.0],
        }
        out = _capacity(ed)

        self.assertIn(
            "cross-check 1 - mean(value_error^2)/mean(return_std^2) over the "
            "same 2 rollouts: 0.990",
            out,
        )


class TestCapacityGradientAllocation(unittest.TestCase):
    """-- Gradient allocation --: NaN norms are dropped before any statistic,
    and the over-threshold fraction is skipped rather than computed against a
    guessed ppo_max_grad_norm.
    """

    def test_nan_grad_norms_are_excluded_from_mean_max_and_count(self):
        ed = {
            "rollout_pv_grad_norm": [0.5, NAN, 1.5, 2.0],
            "rollout_ent_grad_norm": [0.1, 0.1, NAN, 0.2],
        }
        out = _capacity(ed)

        self.assertIn(
            "policy-value grad norm: mean 1.333e+00, max 2.000e+00 (3/4 finite rollouts)", out
        )
        self.assertIn("mean 1.333e-01, max 2.000e-01 (3/4 finite rollouts)", out)

    def test_over_threshold_fraction_skipped_when_max_grad_norm_unknown(self):
        out = _capacity({"rollout_pv_grad_norm": [0.5, NAN, 1.5, 2.0]})

        self.assertIn("fraction above ppo_max_grad_norm: threshold unknown", out)
        self.assertIn("not guessing it", out)
        self.assertNotIn("rollouts with grad norm above ppo_max_grad_norm", out)

    def test_over_threshold_fraction_comes_from_run_config_and_finite_rollouts(self):
        ed = {"rollout_pv_grad_norm": [0.5, NAN, 1.5, 2.0]}
        out = _capacity(ed, d={"run_config": {"ppo_max_grad_norm": 1.0}})

        # 2 of the 3 FINITE rollouts exceed 1.0 — the NaN is not counted as under.
        self.assertIn("rollouts with grad norm above ppo_max_grad_norm (1): 2/3 (66.7%)", out)
        self.assertNotIn("threshold unknown", out)

    def test_cli_max_grad_norm_used_when_run_config_carries_none(self):
        out = _capacity({"rollout_pv_grad_norm": [0.5, 1.5]}, max_grad_norm=1.0)

        self.assertIn("rollouts with grad norm above ppo_max_grad_norm (1): 1/2 (50.0%)", out)

    def test_per_group_shares_renormalise_over_finite_entries_only(self):
        ed = {
            "rollout_pv_grad_norm": [1.0],
            "rollout_grad_norm_cnn": [1.0, 1.0],
            "rollout_grad_norm_ram": [NAN, 3.0],
            "rollout_grad_norm_trunk": [2.0, 2.0],
            "rollout_grad_norm_heads": [NAN, NAN],
        }
        out = _capacity(ed)

        # means cnn 1.0, ram 3.0, trunk 2.0 -> total 6.0; heads is all-NaN so it
        # gets no row at all rather than a fabricated 0.0 share.
        self.assertIn("mean share of the summed group norms", out)
        self.assertIn("16.7%", out)
        self.assertIn("50.0%", out)
        self.assertIn("33.3%", out)
        self.assertNotIn("heads", out)

    def test_unavailable_when_policy_value_norms_were_not_logged(self):
        out = _capacity({})

        self.assertIn("gradient allocation: unavailable for this run", out)


class TestCapacityOptimisationPressure(unittest.TestCase):
    """-- Optimisation pressure --: the one question the Optimisation section
    leaves open, whether the trust region rather than the network bounds progress.
    """

    def test_constant_lr_pressing_the_kl_ceiling_is_trust_region_limited(self):
        ed = {"rollout_approx_kl": [0.05] * 4 + [0.001] * 6, "rollout_lr": [3e-4] * 10}
        out = _capacity(ed)

        self.assertIn("rollouts above ppo_target_kl (0.02): 4/10 (40.0%)", out)
        self.assertIn("learning rate, first vs last finite value: did not change", out)
        self.assertIn("OK: trust-region-limited", out)
        self.assertIn("presses the KL ceiling on 40% of updates", out)
        self.assertNotIn("WARNING", _subsection(out, "-- Optimisation pressure --"))

    def test_updates_far_inside_the_trust_region_warn(self):
        ed = {"rollout_approx_kl": [0.001] * 10, "rollout_lr": [3e-4] * 10}
        out = _capacity(ed)

        self.assertIn("rollouts above ppo_target_kl (0.02): 0/10 (0.0%)", out)
        self.assertIn("WARNING: updates are far inside the trust region", out)
        self.assertNotIn("OK: trust-region-limited", out)

    def test_mid_band_kl_sits_near_the_target(self):
        ed = {"rollout_approx_kl": [0.05] * 2 + [0.001] * 8, "rollout_lr": [3e-4] * 10}
        out = _capacity(ed)

        self.assertIn("rollouts above ppo_target_kl (0.02): 2/10 (20.0%)", out)
        self.assertIn("updates sit near the trust-region target", out)
        self.assertNotIn("OK: trust-region-limited", out)
        self.assertNotIn("WARNING: updates are far inside", out)

    def test_a_decaying_lr_denies_the_trust_region_limited_verdict(self):
        # Same 100%-over-target KL, but the LR moved, so "constant LR" is false
        # and the section must not claim update size alone bounds progress.
        ed = {"rollout_approx_kl": [0.05] * 10, "rollout_lr": [3e-4] * 5 + [1e-5] * 5}
        out = _capacity(ed)

        self.assertIn("learning rate, first vs last finite value: changed", out)
        self.assertNotIn("OK: trust-region-limited", out)

    def test_run_config_kl_thresholds_override_the_cli_defaults(self):
        ed = {"rollout_approx_kl": [0.03] * 10, "rollout_lr": [3e-4] * 10}
        run_config = {"ppo_target_kl": 0.05, "ppo_hard_kl": 0.2}
        out = _capacity(ed, d={"run_config": run_config})

        self.assertIn("rollouts above ppo_target_kl (0.05): 0/10 (0.0%)", out)
        self.assertIn("rollouts above ppo_hard_kl (0.2): 0/10 (0.0%)", out)
        self.assertIn("WARNING: updates are far inside the trust region", out)

    def test_persisted_none_target_kl_prints_no_ceiling_and_no_verdict(self):
        # vec_ppo_agent.save_model keeps ppo_target_kl: None on purpose — there
        # None means the run configured NO KL early-stop, so --target-kl's 0.02
        # default must not be substituted for it and judged against.
        ed = {"rollout_approx_kl": [0.05] * 10, "rollout_lr": [3e-4] * 10}
        out = _capacity(ed, d={"run_config": {"ppo_target_kl": None}})

        self.assertIn("KL ceilings: ppo_target_kl none configured (run_config)", out)
        self.assertIn("fraction above ppo_target_kl: undefined", out)
        self.assertIn("this run configured no ppo_target_kl", out)
        self.assertIn("no KL early-stop ceiling", out)
        # No numeric fraction...
        self.assertNotIn("rollouts above ppo_target_kl", out)
        # ...and no verdict: a 100%-over-target series at a constant LR is
        # exactly what would otherwise be called trust-region-limited.
        self.assertNotIn("trust-region-limited", out)
        self.assertNotIn("WARNING: updates are far inside", out)
        self.assertNotIn("updates sit near the trust-region target", out)
        self.assertIn("trust-region verdict: skipped", out)

    def test_persisted_none_target_kl_is_listed_under_missing_diagnostics(self):
        ed = {"rollout_approx_kl": [0.05] * 4, "rollout_lr": [3e-4] * 4}
        out = _capacity(ed, d={"run_config": {"ppo_target_kl": None}})

        self.assertIn("what this run's artefacts cannot answer:", out)
        self.assertIn(
            "optimisation pressure: run persisted ppo_target_kl None, i.e. no "
            "KL early-stop ceiling, so the over-target fraction is undefined",
            out,
        )
        # A run_config WAS persisted, so neither the blanket "absent" entry nor
        # the CLI-defaults disclaimer may fire — the gap is this one key.
        self.assertNotIn("run_config: absent", out)
        self.assertNotIn("every threshold above came from CLI defaults", out)

    def test_absent_target_kl_still_falls_back_to_the_cli_default(self):
        # An older checkpoint with no ppo_target_kl key at all is the one case
        # where the CLI default legitimately supplies the ceiling.
        ed = {"rollout_approx_kl": [0.05] * 4 + [0.001] * 6, "rollout_lr": [3e-4] * 10}
        out = _capacity(ed, d={"run_config": {"ppo_max_grad_norm": 1.0}})

        self.assertIn("KL ceilings: ppo_target_kl 0.02 (CLI)", out)
        self.assertIn("rollouts above ppo_target_kl (0.02): 4/10 (40.0%)", out)
        self.assertIn("OK: trust-region-limited", out)
        self.assertNotIn("fraction above ppo_target_kl: undefined", out)

    def test_run_config_kl_thresholds_are_labelled_as_run_config(self):
        ed = {"rollout_approx_kl": [0.03] * 10, "rollout_lr": [3e-4] * 10}
        run_config = {"ppo_target_kl": 0.05, "ppo_hard_kl": 0.2}
        out = _capacity(ed, d={"run_config": run_config})

        self.assertIn("ppo_target_kl 0.05 (run_config)", out)
        self.assertIn("ppo_hard_kl 0.2 (run_config)", out)
        self.assertNotIn("ppo_target_kl 0.02 (CLI)", out)
        self.assertNotIn("ppo_hard_kl 0.06 (CLI)", out)

    def test_persisted_none_hard_kl_skips_only_the_hard_fraction(self):
        ed = {"rollout_approx_kl": [0.05] * 4 + [0.001] * 6, "rollout_lr": [3e-4] * 10}
        out = _capacity(ed, d={"run_config": {"ppo_hard_kl": None}})

        self.assertIn("ppo_hard_kl none configured (run_config)", out)
        self.assertIn("fraction above ppo_hard_kl: undefined", out)
        self.assertNotIn("rollouts above ppo_hard_kl", out)
        self.assertIn("run persisted ppo_hard_kl None", out)
        # ppo_target_kl is merely absent here, so the target-side fraction and
        # its verdict are still computed from the CLI default.
        self.assertIn("ppo_target_kl 0.02 (CLI)", out)
        self.assertIn("rollouts above ppo_target_kl (0.02): 4/10 (40.0%)", out)
        self.assertIn("OK: trust-region-limited", out)

    def test_unavailable_when_approx_kl_was_not_logged(self):
        out = _capacity({})

        self.assertIn("KL pressure: unavailable (no finite rollout_approx_kl)", out)


class TestCapacityTerminationMix(unittest.TestCase):
    """-- Termination mix --: stagnation and battle_stagnation are one idea-
    limited signature, but each cause keeps its own row in the table.
    """

    BOTH_STAGNATIONS = ["budget"] * 2 + ["stagnation"] * 3 + ["battle_stagnation"] * 2

    def test_the_two_stagnation_causes_are_summed_for_the_warning(self):
        out = _capacity({"episode_truncation_cause": list(self.BOTH_STAGNATIONS)})

        self.assertIn(
            "WARNING: 71% of episodes ended on stagnation (battle_stagnation, stagnation)", out
        )
        self.assertIn("idea-limited, not step-limited", out)

    def test_each_cause_is_still_listed_separately(self):
        out = _capacity({"episode_truncation_cause": list(self.BOTH_STAGNATIONS)})

        rows = {label: count for label, count, _share in _table_rows(out)}
        self.assertEqual(rows, {"budget": 2, "stagnation": 3, "battle_stagnation": 2})

    def test_no_warning_when_stagnation_is_under_60_percent(self):
        causes = ["budget"] * 5 + ["stagnation"] * 3 + ["battle_stagnation"] * 2
        out = _capacity({"episode_truncation_cause": causes})

        self.assertNotIn("WARNING", _subsection(out, "-- Termination mix --"))
        self.assertEqual(
            {label: count for label, count, _s in _table_rows(out)},
            {"budget": 5, "stagnation": 3, "battle_stagnation": 2},
        )

    def test_unavailable_when_causes_were_not_logged(self):
        out = _capacity({})

        self.assertIn("truncation causes: unavailable for this run", out)


class TestCapacityHonestCoverage(unittest.TestCase):
    """-- Honest coverage trend --: seeded episodes are excluded, empty buckets
    are dropped rather than averaged to 0.0, and fewer than two real buckets
    prints no trend at all instead of inventing a collapse.
    """

    def test_rising_first_to_last_is_ok(self):
        ed = {
            "episode_seeded": [False] * 8,
            "episode_unique_maps": [1, 2, 3, 4, 5, 6, 7, 8],
            "episode_unique_cells": [10, 20, 30, 40, 50, 60, 70, 80],
        }
        out = _capacity(ed)

        self.assertIn("8 honest episodes, oldest first:", out)
        self.assertIn("unique_maps   8 buckets: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]", out)
        self.assertIn("OK: honest coverage still rising first-to-last in both series", out)
        self.assertNotIn("WARNING", _subsection(out, "-- Honest coverage trend --"))

    def test_falling_first_to_last_warns(self):
        ed = {
            "episode_seeded": [False] * 8,
            "episode_unique_maps": [8, 7, 6, 5, 4, 3, 2, 1],
            "episode_unique_cells": [80, 70, 60, 50, 40, 30, 20, 10],
        }
        out = _capacity(ed)

        self.assertIn("WARNING: honest coverage declining first-to-last in both series", out)
        self.assertNotIn("OK: honest coverage", out)

    def test_one_series_each_way_is_reported_as_mixed_by_name(self):
        ed = {
            "episode_seeded": [False] * 8,
            "episode_unique_maps": [1, 2, 3, 4, 5, 6, 7, 8],
            "episode_unique_cells": [80, 70, 60, 50, 40, 30, 20, 10],
        }
        out = _capacity(ed)

        self.assertIn("mixed: unique_maps rose first-to-last while unique_cells fell", out)

    def test_seeded_episodes_are_excluded_from_the_trend(self):
        # The first four are Go-Explore seeds starting deep in solved territory.
        # Counted, they would turn a rising honest run into an apparent collapse.
        ed = {
            "episode_seeded": [True] * 4 + [False] * 4,
            "episode_unique_maps": [40, 41, 42, 43, 1, 2, 3, 4],
            "episode_unique_cells": [400, 410, 420, 430, 10, 20, 30, 40],
        }
        out = _capacity(ed)

        self.assertIn("4 honest episodes, oldest first:", out)
        self.assertIn("unique_maps   4 buckets: [1.0, 2.0, 3.0, 4.0]", out)
        self.assertIn("OK: honest coverage still rising first-to-last in both series", out)

    def test_too_few_honest_episodes_prints_no_trend_and_no_zero_padding(self):
        ed = {"episode_seeded": [False], "episode_unique_maps": [5], "episode_unique_cells": [50]}
        out = _capacity(ed)

        self.assertIn("1 honest episodes, oldest first:", out)
        # One real bucket, not [5.0, 0.0, 0.0, ...] padded out to eight.
        self.assertIn("unique_maps   1 buckets: [5.0]", out)
        self.assertIn("not enough honest episodes for a first-to-last trend in both series", out)
        self.assertNotIn("OK: honest coverage", out)
        self.assertNotIn("WARNING: honest coverage", out)

    def test_unavailable_when_both_coverage_series_are_absent(self):
        out = _capacity({"episode_seeded": [False, False]})

        self.assertIn("honest coverage: unavailable for this run", out)


class TestCapacityMissingDiagnostics(unittest.TestCase):
    """-- Missing diagnostics --: the section's own honesty ledger, and the
    end-to-end case where a run persisted everything and no CLI flag is needed.
    """

    FULL_RUN_CONFIG = {
        "num_envs": 8,
        "ppo_update_frequency": 128,
        "ppo_target_kl": 0.01,
        "ppo_hard_kl": 0.05,
        "ppo_max_grad_norm": 1.0,
        "scaling_factor": 0.5,
        "conv_strides": [2, 2],
    }
    FULL_ED = {
        "rollout_explained_variance": [0.95, NAN, 0.92, 0.97],
        "rollout_value_error_rmse": [0.10, NAN, 0.12, 0.09],
        "rollout_return_std": [1.0, NAN, 1.0, 1.0],
        "rollout_critic_loss": [0.5, NAN, 0.25],
        "rollout_pv_grad_norm": [0.5, NAN, 1.5, 2.0],
        "rollout_ent_grad_norm": [0.1, 0.1, NAN, 0.2],
        "rollout_grad_norm_cnn": [1.0] * 4,
        "rollout_grad_norm_ram": [0.5] * 4,
        "rollout_grad_norm_trunk": [2.0] * 4,
        "rollout_grad_norm_heads": [0.25, NAN, 0.25, 0.25],
        "rollout_approx_kl": [0.02] * 10,
        "rollout_lr": [3e-4] * 10,
        "episode_truncation_cause": ["budget"] * 7 + ["stagnation"] * 3,
        "episode_seeded": [False] * 8,
        "episode_unique_maps": [1, 2, 3, 4, 5, 6, 7, 8],
        "episode_unique_cells": [10, 20, 30, 40, 50, 60, 70, 80],
    }
    TWO_CONV_3X3 = _state_dict(
        {
            "cnn.block1.conv.weight": (16, 1, 3, 3),
            "cnn.block2.conv.weight": (32, 16, 3, 3),
        }
    )

    def test_absent_run_config_is_reported_and_blamed_for_the_thresholds(self):
        out = _capacity({})

        self.assertIn("run_config: absent", out)
        self.assertIn("every threshold above came from CLI defaults", out)
        self.assertIn("what this run's artefacts cannot answer:", out)

    def test_a_run_that_persisted_everything_needs_no_cli_flag_and_gaps_nothing(self):
        out = _capacity(
            self.FULL_ED,
            d={"run_config": dict(self.FULL_RUN_CONFIG)},
            state_dict=self.TWO_CONV_3X3,
        )

        self.assertIn("none — every capacity signal this section can read was present", out)
        self.assertNotIn("every threshold above came from CLI defaults", out)
        # Geometry, data rate and thresholds all resolved from run_config alone.
        self.assertIn("strides used, both axes (from run_config): [2, 2]", out)
        self.assertIn("receptive field: 7x7 px of the scaled CNN input", out)
        self.assertIn(
            "fresh transitions per rollout: 1,024 = ppo_update_frequency 128 (run_config) "
            "x num_envs 8 (run_config)",
            out,
        )
        self.assertIn("parameters per fresh transition: 4.6", out)
        self.assertIn("explained variance: mean 0.947 over 3/4 finite rollouts", out)
        self.assertIn("rollouts above ppo_target_kl (0.01): 10/10 (100.0%)", out)
        self.assertIn("rollouts with grad norm above ppo_max_grad_norm (1): 2/3 (66.7%)", out)


def _stall(episode_data, d=None):
    """Run report_stall_location on synthetic episode_data, return its stdout."""
    return _capture(report_stall_location, episode_data, {} if d is None else d)


def _stall_ed(causes, maps, seeded=None, flags=None):
    """Index-aligned synthetic episode_data for report_stall_location.

    ``maps`` entries may carry None components: the real series appends
    ``[None, num]`` when a RAM read failed, and None is never coordinate 0.
    """
    n = len(causes)
    return {
        "episode_truncation_cause": list(causes),
        "episode_final_map": [list(m) for m in maps],
        "episode_seeded": [False] * n if seeded is None else list(seeded),
        "episode_flag_fire_steps": ([[] for _ in range(n)] if flags is None else list(flags)),
    }


def _trust_args(**overrides):
    """Namespace holding the one flag report_trust_region reads."""
    base = dict(ppo_epochs=4)
    base.update(overrides)
    return argparse.Namespace(**base)


def _trust(episode_data, d=None, **arg_overrides):
    """Run report_trust_region on synthetic episode_data, return its stdout."""
    return _capture(
        report_trust_region,
        episode_data,
        {} if d is None else d,
        _trust_args(**arg_overrides),
    )


def _body_lines(out):
    """A section's printed lines, minus blanks and its ``=== header ===``."""
    return [
        line for line in out.splitlines() if line.strip() and not line.strip().startswith("===")
    ]


class TestStallLocationConcentration(unittest.TestCase):
    """=== Stall location ===: the verdict on the single most common final map —
    >=50% is one gate, 25-50% is a lean, <25% is spread.
    """

    def test_half_or_more_on_one_map_is_a_single_gate(self):
        ed = _stall_ed(
            ["stagnation"] * 4,
            [[24, 4], [24, 4], [24, 4], [24, 5]],
            flags=[[[26, 100]], [[26, 120]], [[26, 90], [30, 400]], [[30, 80]]],
        )
        out = _stall(ed)

        self.assertIn("4 honest episodes ended on this cause; 4 carry a usable final map", out)
        self.assertIn("most common final map: (24, 4)", out)
        self.assertIn("75.0%", out)
        self.assertIn(
            "WARNING: stalls concentrate on (24, 4) — one gate, not a general "
            "exploration failure",
            out,
        )
        self.assertNotIn("stalls lean toward", out)
        self.assertNotIn("OK: stalls are spread", out)
        # The deepest flag reached, from the FINAL entry of each fire-step list.
        self.assertIn("deepest flag reached before stalling", out)
        self.assertIn("flag 26", out)
        self.assertIn("flag 30", out)

    def test_between_a_quarter_and_a_half_leans_but_is_spread(self):
        # 3/12 = 25.0% exactly, and (24, 4) is the unambiguous top map.
        maps = (
            [[24, 4]] * 3
            + [[24, 5]] * 2
            + [[24, 3]] * 2
            + [[26, 1]] * 2
            + [[26, 3]] * 2
            + [[11, 2]]
        )
        ed = _stall_ed(["stagnation"] * 12, maps)
        out = _stall(ed)

        self.assertIn("stalls lean toward (24, 4) but are spread", out)
        self.assertNotIn("WARNING: stalls concentrate", out)
        self.assertNotIn("OK: stalls are spread", out)

    def test_under_a_quarter_is_spread_with_no_single_gate(self):
        maps = [[24, 4], [24, 5], [24, 3], [26, 1], [26, 3]]
        out = _stall(_stall_ed(["stagnation"] * 5, maps))

        self.assertIn("OK: stalls are spread across 5 maps, no single gate", out)
        self.assertNotIn("WARNING: stalls concentrate", out)
        self.assertNotIn("stalls lean toward", out)

    def test_more_than_ten_maps_collapse_into_an_other_row(self):
        maps = [[24, 4]] * 6 + [[3, i] for i in range(1, 15)]
        out = _stall(_stall_ed(["stagnation"] * len(maps), maps))

        self.assertIn("other (5 maps)", out)
        self.assertIn("15 distinct over 20 episodes", out)
        # 6/20 = 30%, so the ten printed rows plus 'other' carry all 20.
        self.assertIn("stalls lean toward (24, 4) but are spread", out)


class TestStallLocationFiltering(unittest.TestCase):
    """Honest-only, bucketed on the EXACT cause value, and None-never-zero."""

    def test_a_seeded_episode_does_not_appear(self):
        ed = _stall_ed(["stagnation", "stagnation"], [[24, 4], [3, 49]], seeded=[True, False])
        out = _stall(ed)

        self.assertIn("1 honest episodes ended on this cause", out)
        self.assertIn("(3, 49)", out)
        self.assertNotIn("(24, 4)", out)

    def test_an_all_seeded_cause_reports_no_honest_episode(self):
        ed = _stall_ed(["stagnation"] * 2, [[24, 4], [24, 5]], seeded=[True, True])
        out = _stall(ed)

        self.assertIn("no honest episode ended on this cause", out)
        self.assertNotIn("(24, 4)", out)

    def test_the_two_stagnation_causes_are_never_pooled(self):
        ed = _stall_ed(
            [
                "stagnation",
                "stagnation",
                "battle_stagnation",
                "battle_stagnation",
                "budget",
                None,
                "corrupt",
            ],
            [[24, 4], [24, 5], [3, 49], [3, 50], [26, 1], [26, 3], [11, 2]],
        )
        out = _stall(ed)
        over = _subsection(out, "-- stagnation --")
        battle = _subsection(out, "-- battle_stagnation --")

        self.assertIn("2 honest episodes ended on this cause", over)
        self.assertIn("2 honest episodes ended on this cause", battle)
        self.assertIn("(24, 4)", over)
        self.assertNotIn("(3, 49)", over)
        self.assertIn("(3, 49)", battle)
        self.assertNotIn("(24, 4)", battle)
        # budget, None (a genuine terminal) and corrupt are not stalls at all.
        for pair in ("(26, 1)", "(26, 3)", "(11, 2)"):
            self.assertNotIn(pair, out)

    def test_none_map_components_are_skipped_and_counted_never_read_as_zero(self):
        ed = _stall_ed(["stagnation"] * 3, [[24, 4], [None, 5], [24, None]])
        out = _stall(ed)

        self.assertIn("3 honest episodes ended on this cause; 1 carry a usable final map", out)
        self.assertIn("2 episode(s) skipped: a final-map component was None or unreadable", out)
        self.assertIn("(24, 4)", out)
        self.assertNotIn("(0, 0)", out)
        self.assertNotIn("(None", out)

    def test_a_bucket_with_no_usable_map_says_so(self):
        out = _stall(_stall_ed(["stagnation"], [[None, None]]))

        self.assertIn("1 episode(s) skipped", out)
        self.assertIn("no episode in this bucket carries a usable pair", out)
        self.assertNotIn("WARNING: stalls concentrate", out)

    def test_absent_series_degrades_to_exactly_one_line(self):
        out = _stall({})

        self.assertIn(
            "episode_final_map: not logged before this change, so stall location is "
            "unavailable for this run",
            out,
        )
        self.assertEqual(len(_body_lines(out)), 1)

    def test_visited_maps_and_discovery_log_are_not_substituted(self):
        """Everywhere an episode went, and the run-wide firsts log, are neither of
        them where an episode ENDED — so the degrade line stands even when both
        are present.
        """
        ed = {
            "visited_maps": [[24, 4], [26, 1]],
            "discovery_log": [{"type": "map", "key": 1}],
            "episode_truncation_cause": ["stagnation"],
        }
        out = _stall(ed)

        self.assertIn("stall location is unavailable for this run", out)
        self.assertNotIn("(24, 4)", out)
        self.assertNotIn("(26, 1)", out)

    def test_an_absent_cause_series_cannot_be_bucketed(self):
        out = _stall({"episode_final_map": [[24, 4]]})

        self.assertIn(
            "episode_truncation_cause: not logged by this run, so its episodes cannot "
            "be bucketed by cause",
            out,
        )
        self.assertNotIn("(24, 4)", out)


class TestStallLocationMapNames(unittest.TestCase):
    """Names come from the verified table in docs/reference/ram-mapping.md and
    are never guessed: an undocumented pair prints as the bare (bank, num).
    """

    def test_an_undocumented_pair_prints_bare_with_no_invented_name(self):
        out = _stall(_stall_ed(["stagnation"], [[99, 99]]))

        rows = [
            line
            for line in out.splitlines()
            if line.strip().startswith("(99, 99)") and line.strip().endswith("100.0%")
        ]
        self.assertEqual(len(rows), 1)
        # The label field holds the bare pair and nothing else: no invented name.
        self.assertEqual(rows[0].split(), ["(99,", "99)", "1", "100.0%"])

    @unittest.skipUnless(_map_name_table(), "ram-mapping.md map table unreadable")
    def test_a_documented_pair_resolves_to_its_map_constant(self):
        out = _stall(_stall_ed(["stagnation"], [[24, 4]]))

        self.assertIn("(24, 4) NEW_BARK_TOWN", out)

    @unittest.skipUnless(_map_name_table(), "ram-mapping.md map table unreadable")
    def test_the_table_is_parsed_once_and_cached(self):
        table = _map_name_table()

        self.assertIs(table, _map_name_table())
        self.assertEqual(table[(24, 5)], "ELMS_LAB")
        self.assertNotIn((99, 99), table)


class TestTrustRegionBuckets(unittest.TestCase):
    """=== Trust region ===: three DISJOINT buckets, so a data-integrity
    rejection that applied no gradient is never counted as a trust-region cut.
    """

    TEN = {
        # Two replay rejections and one KL rollback all land on 0 accepted
        # epochs; only the three 2.0 rollouts are genuine early stops.
        "rollout_accepted_epochs": [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0],
        "rollout_replay_rejected": [True, True] + [False] * 8,
        "rollout_kl_rejected": [False, False, True] + [False] * 7,
    }

    def test_replay_and_kl_rejections_are_excluded_from_the_early_stop_bucket(self):
        out = _trust(dict(self.TEN))

        self.assertIn("configured ppo_epochs: 4 (CLI)", out)
        self.assertIn("(i)   replay-rejected, no gradient applied at all: 2/10 (20.0%)", out)
        self.assertIn("(ii)  KL-rejected / rolled back: 1/10 (10.0%)", out)
        self.assertIn(
            "(iii) genuine early stop, fewer than ppo_epochs=4 for any other "
            "reason: 3/10 (30.0%)",
            out,
        )
        self.assertIn("the three buckets are disjoint", out)
        self.assertIn("updates are occasionally cut short (30%)", out)
        self.assertNotIn("WARNING: the KL early stop", out)

    def test_the_epoch_distribution_is_ascending_over_the_finite_values(self):
        out = _trust(dict(self.TEN))

        self.assertIn("10/10 rollouts carry a finite accepted-epoch count", out)
        self.assertIn("0: 3 (30.0%)", out)
        self.assertIn("2: 3 (30.0%)", out)
        self.assertIn("4: 4 (40.0%)", out)
        self.assertLess(out.index("0: 3"), out.index("2: 3"))
        self.assertLess(out.index("2: 3"), out.index("4: 4"))

    def test_forty_percent_cut_short_makes_the_update_budget_the_constraint(self):
        ed = {
            "rollout_accepted_epochs": [1.0] * 4 + [4.0] * 6,
            "rollout_replay_rejected": [False] * 10,
            "rollout_kl_rejected": [False] * 10,
        }
        out = _trust(ed)

        self.assertIn("WARNING: the KL early stop cuts 40% of updates short of ppo_epochs=4", out)
        self.assertIn("the update budget, not the network, is the binding constraint", out)
        self.assertIn("consider LR, ppo_epochs or ppo_target_kl", out)

    def test_almost_every_update_running_the_full_budget_is_ok(self):
        ed = {
            "rollout_accepted_epochs": [4.0] * 10,
            "rollout_replay_rejected": [False] * 10,
            "rollout_kl_rejected": [False] * 10,
        }
        out = _trust(ed)

        self.assertIn("OK: updates almost always run the full 4 epochs", out)
        self.assertIn(
            "(iii) genuine early stop, fewer than ppo_epochs=4 for any other "
            "reason: 0/10 (0.0%)",
            out,
        )

    def test_nan_is_excluded_from_every_count_share_and_bucket(self):
        ed = {
            "rollout_accepted_epochs": [NAN, NAN, 4.0, 4.0, 2.0, NAN],
            "rollout_replay_rejected": [False] * 6,
            "rollout_kl_rejected": [False] * 6,
        }
        out = _trust(ed)

        self.assertIn("3/6 rollouts carry a finite accepted-epoch count", out)
        self.assertIn("the other 3 are NaN", out)
        self.assertIn("2: 1 (33.3%)", out)
        self.assertIn("4: 2 (66.7%)", out)
        self.assertNotIn("nan:", out.lower())
        # The three NaN rollouts are not read as "0 epochs completed".
        self.assertIn(
            "(iii) genuine early stop, fewer than ppo_epochs=4 for any other "
            "reason: 1/6 (16.7%)",
            out,
        )
        self.assertIn("updates are occasionally cut short (17%)", out)


class TestTrustRegionEpochSource(unittest.TestCase):
    """ppo_epochs is labelled with its source exactly as report_capacity labels
    its own thresholds: run_config wins, --ppo-epochs is the fallback.
    """

    ED = {
        "rollout_accepted_epochs": [8.0] * 4,
        "rollout_replay_rejected": [False] * 4,
        "rollout_kl_rejected": [False] * 4,
    }

    def test_run_config_overrides_the_cli_flag(self):
        out = _trust(dict(self.ED), d={"run_config": {"ppo_epochs": 8}}, ppo_epochs=2)

        self.assertIn("configured ppo_epochs: 8 (run_config)", out)
        self.assertIn("OK: updates almost always run the full 8 epochs", out)

    def test_the_cli_flag_is_used_and_labelled_when_run_config_has_no_key(self):
        out = _trust(dict(self.ED), ppo_epochs=2)

        self.assertIn("configured ppo_epochs: 2 (CLI)", out)
        self.assertIn("OK: updates almost always run the full 2 epochs", out)

    def test_a_run_config_that_persisted_none_falls_back_to_the_cli_default(self):
        out = _trust(dict(self.ED), d={"run_config": {"ppo_epochs": None}})

        self.assertIn("configured ppo_epochs: 4 (CLI)", out)


class TestTrustRegionDegrade(unittest.TestCase):
    """Absent or misaligned series degrade to an explicit line, never to 0."""

    def test_absent_series_degrades_to_exactly_one_line(self):
        out = _trust({})

        self.assertIn(
            "rollout_accepted_epochs: not logged before this change, so the epoch "
            "budget is unavailable for this run",
            out,
        )
        self.assertEqual(len(_body_lines(out)), 1)
        # No epoch budget is claimed from a CLI default on the degrade path.
        self.assertNotIn("ppo_epochs", out)

    def test_misaligned_rejection_series_is_flagged_as_a_lower_bound(self):
        ed = {
            "rollout_accepted_epochs": [4.0, 4.0, 2.0],
            "rollout_replay_rejected": [True],
            "rollout_kl_rejected": [],
        }
        out = _trust(ed)

        self.assertIn("are not index-aligned with the 3 accepted-epoch entries", out)
        self.assertIn("lower bounds", out)
        self.assertIn("(i)   replay-rejected, no gradient applied at all: 1/3 (33.3%)", out)

    def test_the_over_target_kl_number_is_cross_referenced_not_restated(self):
        ed = {"rollout_accepted_epochs": [4.0] * 3, "rollout_approx_kl": [0.5] * 3}
        out = _trust(ed, d={"run_config": {"ppo_target_kl": 0.01}})

        self.assertIn("cross-reference: the Capacity section's -- Optimisation pressure", out)
        self.assertIn("that number is not restated here", out)
        self.assertNotIn("rollouts above ppo_target_kl", out)


class TestCapacityArchitectureProvenance(unittest.TestCase):
    """-- Architecture provenance --: conv_strides keeps its KEY while changing
    MEANING between the legacy [2, 2] encoder and the tile-grid [4, 1, 1, 1, 2, 1]
    one, and the observation resize filter moved LANCZOS -> BOX, so the section
    must say which model and which observation pipeline produced the numbers —
    or admit that it cannot.
    """

    TILE_GRID = _state_dict(
        {
            "visual.stem.weight": (32, 3, 4, 4),  # 1,536
            "visual.blocks.0.conv.weight": (32, 32, 3, 3),  # 9,216
            "visual.blocks.1.conv.weight": (32, 32, 3, 3),  # 9,216
            "visual.blocks.2.conv.weight": (32, 32, 3, 3),  # 9,216
            "visual.down.weight": (64, 32, 3, 3),  # 18,432
            "visual.squeeze.weight": (16, 64, 1, 1),  # 1,024
        }
    )  # total 48,640
    FULL = {
        "param_count": 48640,
        "obs_resize_filter": "box",
        "ram_hidden_dim": 192,
        "ram_input_norm": False,
        "git_rev": "abc1234",
        "config_sha256": "deadbeef",
        "conv_strides": [4, 1, 1, 1, 2, 1],
        "scaling_factor": 0.5,
    }

    def test_every_recorded_key_is_printed_and_the_param_count_agrees(self):
        run_config = dict(self.FULL)
        out = _capacity({}, d={"run_config": run_config}, state_dict=self.TILE_GRID)

        self.assertIn("param_count (run_config): 48,640 — agrees with the state_dict count", out)
        self.assertIn("obs_resize_filter: box", out)
        self.assertNotIn("LANCZOS", out)
        self.assertIn("ram_hidden_dim: 192", out)
        # False is a setting, not an absence.
        self.assertIn("ram_input_norm: False", out)
        self.assertIn("git_rev: abc1234", out)
        self.assertIn("config_sha256: deadbeef", out)

    def test_the_encoder_is_named_from_the_stride_schedule_not_a_param_count(self):
        run_config = dict(self.FULL)
        out = _capacity({}, d={"run_config": run_config}, state_dict=self.TILE_GRID)

        self.assertIn("encoder identity: tile-grid encoder", out)
        self.assertIn(
            "from the conv stride schedule (run_config), never from a parameter count",
            out,
        )

    def test_the_legacy_stride_two_schedule_is_named_as_legacy(self):
        run_config = {"conv_strides": [2, 2], "scaling_factor": 0.5}
        out = _capacity({}, d={"run_config": run_config}, state_dict=self.TILE_GRID)

        self.assertIn("encoder identity: legacy stride-2 encoder", out)

    def test_an_unknown_schedule_is_called_unrecognised_rather_than_guessed(self):
        run_config = {"conv_strides": [2, 2, 2]}
        out = _capacity({}, d={"run_config": run_config}, state_dict=self.TILE_GRID)

        self.assertIn("encoder identity: unrecognised encoder (strides [2, 2, 2])", out)

    def test_cli_strides_name_the_encoder_when_run_config_recorded_none(self):
        out = _capacity({}, state_dict=self.TILE_GRID, conv_strides=[4, 1, 1, 1, 2, 1])

        self.assertIn(
            "encoder identity: tile-grid encoder — from the conv stride schedule " "(CLI)",
            out,
        )

    def test_absent_provenance_says_not_recorded_and_warns_about_lanczos(self):
        out = _capacity({})

        for key in (
            "param_count (run_config)",
            "obs_resize_filter",
            "ram_hidden_dim",
            "ram_input_norm",
            "git_rev",
            "config_sha256",
            "encoder identity",
        ):
            self.assertIn(f"{key}: not recorded", out)
        self.assertIn("WARNING: this run predates the filter being recorded", out)
        self.assertIn("LANCZOS (which leaked 0.75 tile at 5.490% of peak)", out)
        self.assertIn("comparisons across that boundary are not like-for-like", out)

    def test_a_recorded_param_count_that_disagrees_is_flagged(self):
        out = _capacity({}, d={"run_config": {"param_count": 48000}}, state_dict=self.TILE_GRID)

        self.assertIn(
            "WARNING: run_config's param_count 48,000 disagrees with the " "state_dict's 48,640",
            out,
        )
        self.assertIn("not the same model", out)

    def test_a_recorded_param_count_with_no_state_dict_is_not_cross_checked(self):
        out = _capacity({}, d={"run_config": {"param_count": 48640}})

        self.assertIn("param_count (run_config): 48,640 — the state_dict did not load", out)
        self.assertIn("not cross-checked", out)
        self.assertNotIn("agrees with the state_dict count", out)


class TestCapacityTileGridReceptiveField(unittest.TestCase):
    """The tile-grid field: a stride-4 patchify stem and a 1x1 squeeze must both
    fall out of RF = 1 + sum over layers of (k - 1) * cumulative jump before it.
    """

    TILE_GRID = TestCapacityArchitectureProvenance.TILE_GRID
    RUN_CONFIG = {"conv_strides": [4, 1, 1, 1, 2, 1], "scaling_factor": 0.5}

    def test_kernels_4_3_3_3_3_1_give_36_scaled_72_original_px_and_9_tiles(self):
        run_config = dict(self.RUN_CONFIG)
        out = _capacity({}, d={"run_config": run_config}, state_dict=self.TILE_GRID)

        # 1 + 3*1 + 8*4 + 8*4 + 8*4 + 8*4 + 0*8 = 36 scaled px; the 1x1 squeeze
        # adds nothing. /0.5 = 72 original px; /8 = 9 tiles.
        self.assertIn("receptive field: 36x36 px of the scaled CNN input", out)
        self.assertIn("receptive field: 72.0x72.0 px of the original screen", out)
        self.assertIn("receptive field: 9.00x9.00 Game Boy tiles", out)
        self.assertNotIn("WARNING: the visual stack sees under 2 tiles", out)


if __name__ == "__main__":
    unittest.main()
