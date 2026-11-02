# -*- coding: utf-8 -*-
"""tools/preflight.py — the pre-launch gate, and the demo stamp it enforces.

Pure tests: no emulator, no training run. The demo checks are exercised
through ``demo_failures`` on hand-built .npz handles, the resume guard through
``resume_note`` on temp dirs, and the writer side through ``demos.build`` with
``load_run``/``replay`` mocked so the stamp is asserted on a real file without
replaying anything.

What is being pinned: gym_env.get_screen_image() moved LANCZOS -> BOX, and SIL
feeds a demo's stored ``images`` straight to the network
(vec_ppo_agent._load_sil_demos -> PPOModel.sil_update), so a demo whose pixels
were made under the old filter trains the policy on an observation
distribution it never sees at rollout time. An ABSENT stamp means the file
predates the stamp and was recorded under LANCZOS — a failure, not a pass.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_CWD = os.getcwd()
import tools.preflight as preflight  # noqa: E402  (module import chdirs to REPO)

os.chdir(_CWD)

from PoliwhiRL.environment.gym_env import RAM_OBS_DIM  # noqa: E402
from tools.llm_agent import demos as demos_mod  # noqa: E402


class _FakeNpz:
    """Stand-in for an np.load() handle: .files plus __getitem__ is all
    demo_failures asks of one, so the gate is testable without a real file."""

    def __init__(self, arrays):
        self._arrays = dict(arrays)
        self.files = sorted(self._arrays)

    def __getitem__(self, key):
        return self._arrays[key]

    def __len__(self):
        return len(self._arrays)


def _demo(ram_dim=RAM_OBS_DIM, stamp=preflight.env_resize_filter()):
    arrays = {"ram_dim": np.int64(ram_dim), "actions": np.arange(4, dtype=np.int64)}
    if stamp is not None:
        arrays[preflight.DEMO_STAMP_KEY] = np.asarray(stamp)
    return _FakeNpz(arrays)


class TestDemoFilterGate(unittest.TestCase):
    def test_matching_stamp_passes(self):
        self.assertEqual(preflight.demo_failures(_demo(), "d.npz", RAM_OBS_DIM, "box"), [])

    def test_different_stamp_fails_naming_both_values(self):
        fails = preflight.demo_failures(_demo(stamp="lanczos"), "d.npz", RAM_OBS_DIM, "box")
        self.assertEqual(len(fails), 1)
        msg = fails[0]
        self.assertIn("d.npz", msg)
        self.assertIn("'lanczos'", msg)  # the file's value
        self.assertIn("'box'", msg)  # the env's value
        self.assertIn("tools/llm_agent/demos.py", msg)  # the rebuild command

    def test_absent_stamp_fails(self):
        fails = preflight.demo_failures(_demo(stamp=None), "d.npz", RAM_OBS_DIM, "box")
        self.assertEqual(len(fails), 1)
        self.assertIn("no obs_resize_filter stamp", fails[0])
        self.assertIn("LANCZOS", fails[0])
        self.assertIn("'box'", fails[0])
        self.assertIn("tools/llm_agent/demos.py", fails[0])

    def test_bytes_stamp_is_compared_decoded(self):
        fails = preflight.demo_failures(_demo(stamp=b"box"), "d.npz", RAM_OBS_DIM, "box")
        self.assertEqual(fails, [])

    def test_ram_dim_mismatch_still_fails(self):
        fails = preflight.demo_failures(_demo(ram_dim=RAM_OBS_DIM - 1), "d.npz", RAM_OBS_DIM, "box")
        self.assertEqual(len(fails), 1)
        self.assertIn("ram_dim", fails[0])

    def test_ram_dim_and_stamp_mismatch_report_both(self):
        d = _demo(ram_dim=RAM_OBS_DIM - 1, stamp="lanczos")
        self.assertEqual(len(preflight.demo_failures(d, "d.npz", RAM_OBS_DIM, "box")), 2)

    def test_expected_filter_is_the_writers_literal(self):
        # The gate must not drift from the writer: both name the filter
        # gym_env.get_screen_image() hardcodes.
        self.assertEqual(preflight.env_resize_filter(), demos_mod.OBS_RESIZE_FILTER)
        self.assertEqual(preflight.env_resize_filter(), "box")

    def test_real_npz_handle_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.npz"
            np.savez(path, ram_dim=RAM_OBS_DIM, obs_resize_filter="box")
            with np.load(path) as d:
                self.assertEqual(preflight.demo_failures(d, str(path), RAM_OBS_DIM, "box"), [])


class TestResumeNote(unittest.TestCase):
    def test_empty_load_checkpoint_does_not_test_the_cwd(self):
        # The bug: Path("") / "info.pth" is the RELATIVE info.pth, and
        # preflight chdirs to REPO, so a repo-root info.pth warned spuriously
        # about a resume that was never going to happen.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "info.pth").write_bytes(b"not a checkpoint")
            here = os.getcwd()
            os.chdir(tmp)
            try:
                note = preflight.resume_note("", None)
            finally:
                os.chdir(here)
        self.assertNotIn("WARNING", note)
        self.assertIn("starts fresh", note)

    def test_empty_load_checkpoint_with_no_checkpoint_dir(self):
        for empty in ("", None):
            self.assertIn("starts fresh", preflight.resume_note(empty, ""))
            self.assertIn("starts fresh", preflight.resume_note(empty, None))

    def test_empty_load_checkpoint_still_catches_auto_resume(self):
        # main.py's _resolve_auto_resume points an empty load_checkpoint at
        # config["checkpoint"] when that dir holds the pair, so the guard has
        # to look there or it misses a real resume.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("starts fresh", preflight.resume_note("", tmp))
            (Path(tmp) / "info.pth").write_bytes(b"x")
            self.assertIn("starts fresh", preflight.resume_note("", tmp))  # pair incomplete
            (Path(tmp) / "actor_critic.pth").write_bytes(b"x")
            note = preflight.resume_note("", tmp)
        self.assertIn("WARNING", note)
        self.assertIn("auto-resumes", note)
        self.assertIn("RESUME", note)

    def test_explicit_load_checkpoint_with_info_pth_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "info.pth").write_bytes(b"x")
            note = preflight.resume_note(tmp, "somewhere/else")
        self.assertIn("WARNING", note)
        self.assertIn("RESUME", note)

    def test_explicit_load_checkpoint_without_info_pth_is_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            note = preflight.resume_note(tmp, None)
        self.assertNotIn("WARNING", note)
        self.assertIn("starts fresh", note)
        self.assertIn(repr(tmp), note)


class TestEncoderIdentity(unittest.TestCase):
    def test_tile_grid_encoder(self):
        self.assertEqual(preflight.encoder_identity([4, 1, 1, 1, 2, 1]), "tile-grid encoder")

    def test_legacy_stride_2_encoder(self):
        self.assertEqual(preflight.encoder_identity([2, 2]), "legacy stride-2 encoder")

    def test_unrecognised_encoder_names_the_strides(self):
        self.assertIn("[3, 3]", preflight.encoder_identity([3, 3]))

    def test_no_convs_is_admitted_not_guessed(self):
        self.assertIn("unknown", preflight.encoder_identity([]))


class TestModelNote(unittest.TestCase):
    def test_note_reports_params_and_encoder(self):
        from main import load_default_config, load_user_config, merge_configs

        user = load_user_config("configs/stages/open_world.json")
        cfg = merge_configs(load_default_config(), user)
        note = preflight.model_note(cfg, (3, 72, 80), 9, RAM_OBS_DIM)
        self.assertTrue(note.startswith("model: "), note)
        self.assertIn("tile-grid encoder", note)
        self.assertIn("[4, 1, 1, 1, 2, 1]", note)
        count = int(note.split("model: ")[1].split(" params")[0].replace(",", ""))
        self.assertGreater(count, 100_000)


class _FakeGame:
    """What demos.build needs of a replay env: story flags and a close()."""

    def __init__(self):
        self.closed = False
        self.env = mock.Mock()

    def vars(self):
        return {"story_flags": bytes(512)}

    def close(self):
        self.closed = True


def _fake_load_run(run_dir):
    return {
        "dir": Path(run_dir),
        "meta": {"config": {}, "state": None},
        "attempts": [
            {"goal": "stairs", "attempt": 1, "success": True},
            {"goal": "skipped", "attempt": 1, "success": False},
        ],
        "steps": [{"step": 1}],
    }


def _fake_replay(steps):
    def replay(run_dir, until=None, game=None, on_step=None):
        g = game if game is not None else _FakeGame()
        for row, sample in steps:
            on_step(g, row, sample)
        return g, None

    return replay


class TestDemosWriteTheStamp(unittest.TestCase):
    def test_build_stamps_the_filter_and_keeps_the_old_arrays(self):
        T = 5
        steps = [
            (
                {"goal": "stairs", "attempt": 1},
                {
                    "image": np.full((3, 4, 4), t, np.uint8),
                    "ram": np.full((RAM_OBS_DIM,), float(t), np.float32),
                    "action": t,
                    "reward": float(t),
                },
            )
            for t in range(T)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "llm_demos"
            run_dir = Path(tmp) / "fake_run"
            with (
                mock.patch.object(demos_mod, "load_run", _fake_load_run),
                mock.patch.object(demos_mod, "replay", _fake_replay(steps)),
            ):
                written = demos_mod.build([(str(run_dir), None)], str(out))
            self.assertEqual(len(written), 1)  # only the successful attempt
            path = written[0]
            self.assertEqual(path.parent, out)
            self.assertIn("fake_run_01_stairs_a1.npz", path.name)
            with np.load(path) as d:
                self.assertIn("obs_resize_filter", d.files)
                self.assertEqual(str(d["obs_resize_filter"]), "box")
                self.assertEqual(str(d["obs_resize_filter"]), preflight.env_resize_filter())
                self.assertEqual(preflight.demo_failures(d, str(path), RAM_OBS_DIM, "box"), [])
                # Unchanged arrays and dtypes, so old readers still work.
                self.assertEqual(
                    sorted(d.files),
                    [
                        "actions",
                        "images",
                        "obs_resize_filter",
                        "ram",
                        "ram_dim",
                        "rewards",
                        "source",
                        "target_flag",
                    ],
                )
                self.assertEqual(d["images"].shape, (T, 3, 4, 4))
                self.assertEqual(d["images"].dtype, np.uint8)
                self.assertEqual(d["ram"].shape, (T, RAM_OBS_DIM))
                self.assertEqual(d["ram"].dtype, np.float32)
                self.assertEqual(d["actions"].dtype, np.int64)
                self.assertEqual(d["rewards"].dtype, np.float32)
                self.assertEqual(int(d["ram_dim"]), RAM_OBS_DIM)
                self.assertEqual(int(d["target_flag"]), -1)
                self.assertEqual(str(d["source"]), "llm_demo")

    def test_a_demo_written_before_the_stamp_fails_the_gate(self):
        # The eight files in Training Outputs/llm_demos were recorded under
        # LANCZOS with no stamp; this is the shape preflight must reject.
        T = 3
        steps = [
            (
                {"goal": "stairs", "attempt": 1},
                {
                    "image": np.zeros((3, 4, 4), np.uint8),
                    "ram": np.zeros((RAM_OBS_DIM,), np.float32),
                    "action": 0,
                    "reward": 0.0,
                },
            )
            for _ in range(T)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "llm_demos"
            with (
                mock.patch.object(demos_mod, "load_run", _fake_load_run),
                mock.patch.object(demos_mod, "replay", _fake_replay(steps)),
            ):
                path = demos_mod.build([(str(Path(tmp) / "fake_run"), None)], str(out))[0]
            with np.load(path) as d:
                arrays = {k: d[k] for k in d.files if k != "obs_resize_filter"}
            stale = _FakeNpz(arrays)
            fails = preflight.demo_failures(stale, str(path), RAM_OBS_DIM, "box")
        self.assertEqual(len(fails), 1)
        self.assertIn("no obs_resize_filter stamp", fails[0])


if __name__ == "__main__":
    unittest.main()
