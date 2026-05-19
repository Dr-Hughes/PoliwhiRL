# -*- coding: utf-8 -*-
import json
import os
import shutil
import tempfile
import unittest

from main import load_user_config


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


class TestExtendsConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_missing_path_returns_empty(self):
        self.assertEqual(load_user_config(None), {})
        self.assertEqual(load_user_config(os.path.join(self.tmp, "nope.json")), {})

    def test_no_extends_returns_file_as_is(self):
        path = os.path.join(self.tmp, "a.json")
        _write_json(path, {"x": 1, "y": "hi"})
        self.assertEqual(load_user_config(path), {"x": 1, "y": "hi"})

    def test_single_extends_merges_with_override(self):
        base = os.path.join(self.tmp, "base.json")
        child = os.path.join(self.tmp, "child.json")
        _write_json(base, {"x": 1, "y": 2, "z": 3})
        _write_json(child, {"extends": "base.json", "y": 20, "w": 4})

        merged = load_user_config(child)
        self.assertEqual(merged, {"x": 1, "y": 20, "z": 3, "w": 4})
        # 'extends' must not leak into the resolved config.
        self.assertNotIn("extends", merged)

    def test_list_values_are_fully_replaced(self):
        base = os.path.join(self.tmp, "base.json")
        child = os.path.join(self.tmp, "child.json")
        _write_json(base, {"goals": [[1, 2], [3, 4]]})
        _write_json(child, {"extends": "base.json", "goals": [[9, 9]]})

        self.assertEqual(load_user_config(child), {"goals": [[9, 9]]})

    def test_chained_extends(self):
        a = os.path.join(self.tmp, "a.json")
        b = os.path.join(self.tmp, "b.json")
        c = os.path.join(self.tmp, "c.json")
        _write_json(a, {"x": 1, "y": 1, "z": 1})
        _write_json(b, {"extends": "a.json", "y": 2, "w": 2})
        _write_json(c, {"extends": "b.json", "z": 3})

        self.assertEqual(load_user_config(c), {"x": 1, "y": 2, "z": 3, "w": 2})

    def test_extends_resolves_relative_to_file_not_cwd(self):
        sub = os.path.join(self.tmp, "stages")
        os.makedirs(sub)
        base = os.path.join(self.tmp, "base.json")
        child = os.path.join(sub, "first.json")
        _write_json(base, {"x": 1})
        _write_json(child, {"extends": "../base.json", "y": 2})

        cwd = os.getcwd()
        try:
            os.chdir(self.tmp)
            self.assertEqual(load_user_config(child), {"x": 1, "y": 2})
        finally:
            os.chdir(cwd)

    def test_circular_extends_raises(self):
        a = os.path.join(self.tmp, "a.json")
        b = os.path.join(self.tmp, "b.json")
        _write_json(a, {"extends": "b.json"})
        _write_json(b, {"extends": "a.json"})

        with self.assertRaises(ValueError):
            load_user_config(a)

    def test_missing_parent_raises(self):
        child = os.path.join(self.tmp, "child.json")
        _write_json(child, {"extends": "missing.json"})

        with self.assertRaises(FileNotFoundError):
            load_user_config(child)


if __name__ == "__main__":
    unittest.main()
