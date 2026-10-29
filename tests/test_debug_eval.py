# -*- coding: utf-8 -*-
"""Tests for the debug evaluator: extended RAM probes, scripted menu
sequence, and PNG/JSON sidecar emission."""
import glob
import json
import os
import shutil
import tempfile
import unittest

from PoliwhiRL.environment.gym_env import PyBoyEnvironment
from PoliwhiRL.evaluator import run_debug_inference
from main import load_default_config


_EXTENDED_KEYS = {
    "johto_badges_raw",
    "johto_badges_count",
    "kanto_badges_raw",
    "kanto_badges_count",
    "unowndex_status",
    "bug_catching_contest",
    "day_of_week",
    "game_minute",
    "play_time_hours",
    "wild_battle_type",
    "enemy_trainer_type",
    "item_pocket_count",
    "pokeball_count",
    "coins",
    "repel_steps",
    "blue_card_points",
    "hooh_captured",
    "lugia_captured",
    "sudowoodo_captured",
    "red_gyarados_captured",
    "snorlax_captured",
    "raikou_map_bank",
    "raikou_map_num",
    "entei_map_bank",
    "entei_map_num",
}


class TestExtendedRAMProbes(unittest.TestCase):
    def setUp(self):
        self.config = load_default_config()
        self.env = PyBoyEnvironment(self.config)

    def tearDown(self):
        self.env.close()

    def test_extended_variables_all_int(self):
        ext = self.env.ram.get_extended_variables()
        self.assertEqual(set(ext.keys()), _EXTENDED_KEYS)
        for key, val in ext.items():
            self.assertIsInstance(val, int, f"{key} is not int: {type(val)}")
            self.assertGreaterEqual(val, 0, f"{key} negative: {val}")

    def test_debug_byte_windows_shape(self):
        windows = self.env.ram.get_debug_byte_windows()
        self.assertIn("audio_c2a0", windows)
        self.assertIn("map_d145", windows)
        self.assertIn("text_cf00", windows)
        self.assertIn("script_d430", windows)
        for name, bs in windows.items():
            self.assertGreater(len(bs), 0, f"empty window {name}")
            for b in bs:
                self.assertIsInstance(b, int)
                self.assertGreaterEqual(b, 0)
                self.assertLess(b, 256)


class TestMenuProbeRun(unittest.TestCase):
    """End-to-end: run menu_probe and confirm PNGs + JSONs are emitted."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="poliwhirl_debug_test_")
        self.config = load_default_config()
        self.config.update(
            {
                "model": "debug_eval",
                "debug_mode": "menu_probe",
                "record_path": self.tmpdir,
                "ignored_buttons": [""],
                "action_replay_paths": [],
                "episode_length": 64,
                "vision": True,
                "record": True,
            }
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_menu_probe_writes_files(self):
        run_debug_inference(self.config)
        folder = os.path.join(self.tmpdir, "menu_probe")
        self.assertTrue(os.path.isdir(folder), f"missing folder {folder}")

        pngs = glob.glob(os.path.join(folder, "*.png"))
        # Per-step JSONs + the run_summary.json finalize() writes.
        step_jsons = [
            p
            for p in glob.glob(os.path.join(folder, "*.json"))
            if os.path.basename(p) != "run_summary.json"
        ]
        # Initial frame + 8 scripted steps = 9 expected.
        self.assertEqual(len(pngs), 9, f"unexpected PNG count: {sorted(pngs)}")
        self.assertEqual(
            len(step_jsons), 9, f"unexpected per-step JSON count: {sorted(step_jsons)}"
        )
        self.assertTrue(os.path.isfile(os.path.join(folder, "run_summary.json")))

    def test_menu_probe_sidecar_schema(self):
        run_debug_inference(self.config)
        folder = os.path.join(self.tmpdir, "menu_probe")
        # run_summary.json sits alongside the per-step sidecars; exclude it.
        jsons = sorted(
            p
            for p in glob.glob(os.path.join(folder, "*.json"))
            if os.path.basename(p) != "run_summary.json"
        )
        self.assertTrue(jsons)

        with open(jsons[0]) as f:
            payload = json.load(f)

        for top_key in (
            "step",
            "episode",
            "button",
            "reward",
            "ram_state_valid",
            "base",
            "extended",
            "byte_windows_hex",
            "byte_windows_changed",
        ):
            self.assertIn(top_key, payload, f"missing {top_key}")

        # Extended dict should carry every documented probe.
        self.assertEqual(set(payload["extended"].keys()), _EXTENDED_KEYS)

    def test_menu_probe_change_deltas(self):
        """The first frame has no prior so byte_windows_changed must be
        empty. Later frames must record at least one delta around the
        START press (step 4) — the menu state has to flip something."""
        run_debug_inference(self.config)
        folder = os.path.join(self.tmpdir, "menu_probe")
        sidecars = sorted(
            p
            for p in glob.glob(os.path.join(folder, "*.json"))
            if os.path.basename(p) != "run_summary.json"
        )
        with open(sidecars[0]) as f:
            first = json.load(f)
        self.assertEqual(first["byte_windows_changed"], {})

        # Find the start sidecar (step_4) by filename.
        start_path = [p for p in sidecars if "step_4_" in os.path.basename(p)][0]
        with open(start_path) as f:
            start = json.load(f)
        # At least one of the byte windows must have flipped on the START.
        self.assertGreater(
            sum(len(v) for v in start["byte_windows_changed"].values()),
            0,
            f"no byte-window changes detected at step 4: {start['byte_windows_changed']}",
        )

    def test_run_summary_emitted(self):
        run_debug_inference(self.config)
        folder = os.path.join(self.tmpdir, "menu_probe")
        summary_path = os.path.join(folder, "run_summary.json")
        self.assertTrue(os.path.isfile(summary_path), f"missing {summary_path}")
        with open(summary_path) as f:
            summary = json.load(f)
        self.assertIn("frames_written", summary)
        self.assertIn("n_changed_addresses", summary)
        self.assertIn("addresses", summary)
        self.assertEqual(summary["frames_written"], 9)
        # The START press flips at least one address.
        self.assertGreater(summary["n_changed_addresses"], 0)
        # The address table must sort by changes descending.
        counts = [e["changes"] for e in summary["addresses"]]
        self.assertEqual(counts, sorted(counts, reverse=True))
        # Each entry should carry the structured fields.
        for entry in summary["addresses"]:
            for key in (
                "addr",
                "window",
                "offset",
                "changes",
                "first_change_step",
                "values_seen",
                "n_distinct_values",
            ):
                self.assertIn(key, entry)

    def test_menu_probe_button_signature(self):
        """Verify the start press actually fires (i.e. ignored_buttons
        override is being honoured). We do this by reading the recorded
        filename of the 4th scripted step, which must carry btn_start."""
        run_debug_inference(self.config)
        folder = os.path.join(self.tmpdir, "menu_probe")
        pngs = glob.glob(os.path.join(folder, "*.png"))
        names = [os.path.basename(p) for p in pngs]
        # step_4 = 4th scripted action = START (sequence is noop,noop,noop,
        # START,noop,noop,noop,B and the initial baseline frame is step_0).
        step4 = [n for n in names if n.startswith("step_4_")]
        self.assertEqual(len(step4), 1, f"step_4 missing from {names}")
        self.assertIn("btn_start", step4[0])
        step8 = [n for n in names if n.startswith("step_8_")]
        self.assertEqual(len(step8), 1)
        self.assertIn("btn_b", step8[0])


if __name__ == "__main__":
    unittest.main()
