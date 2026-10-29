# -*- coding: utf-8 -*-
"""Replay-pool path resolution + missing-file policy.

Pinned behaviour (guards the silent failure that let a curriculum stage train
with no warm-start because its seed file was missing):

- An explicit (non-glob) ``action_replay_paths`` entry that does not exist is a
  HARD error.
- A glob pattern matching zero files is tolerated (empty pool, warning only).
- Existing files load their trajectories.
"""
import os
import tempfile
import unittest

from PoliwhiRL.environment.vec_env import _load_replay_pool, write_actions_file


class TestReplayPoolLoading(unittest.TestCase):
    def test_missing_explicit_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            _load_replay_pool(["./definitely/not/here/actions.steps"])

    def test_missing_glob_is_tolerated(self):
        # A glob matching nothing returns an empty pool, no exception.
        expanded, trajectories = _load_replay_pool(["./no_such_dir/*.steps"])
        self.assertEqual(trajectories, [])

    def test_existing_file_loads(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "actions.steps")
            write_actions_file(path, [[0, 1, 2], [3, 4]])
            expanded, trajectories = _load_replay_pool([path])
            self.assertEqual(len(trajectories), 2)
            self.assertEqual(trajectories[0], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
