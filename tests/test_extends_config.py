# -*- coding: utf-8 -*-
import json
import os
import shutil
import tempfile
import unittest

from main import load_default_config, load_user_config, merge_configs


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

    def test_canonical_open_world_config_resolves(self):
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "stages", "open_world.json")

        merged = load_user_config(path)

        self.assertEqual(merged["output_base_dir"], "./Training Outputs/open_world_v3/")
        self.assertTrue(merged["goexplore_enabled"])
        self.assertTrue(merged["event_novelty_enabled"])
        self.assertIn(1213, [g.get("flag_num") for g in merged["probe_goals"]])
        self.assertEqual(
            (merged["probe_goals"][-1]["map_bank"], merged["probe_goals"][-1]["map_num"]),
            (3, 37),  # Union Cave 1F, the first map past the Zephyr Badge gate
        )

    def test_open_world_battle_cap_covers_one_full_trainer_win_cycle(self):
        """2026-10-22 regression guard: an undocumented 'bump' commit once cut
        battle_reward_episode_cap from 15.0 to 3.0 (below the cost of even a
        single wild engagement), which silently clipped a trainer-battle win
        (the signal this project most wants to reinforce on the way to a gym
        badge) down to almost nothing whenever anything else touched the cap
        first this episode. Assert the FULLY merged config (stage + defaults,
        the same path the live trainer uses) leaves enough headroom for one
        complete trainer engage+win sequence to pay in full, so this class of
        accidental regression fails CI instead of silently degrading a run."""
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "stages", "open_world.json")
        merged = merge_configs(load_default_config(), load_user_config(path))

        premium = merged["trainer_battle_premium_multiplier"]
        one_full_trainer_cycle = premium * (
            merged["battle_engagement_reward"] + merged["battle_win_reward"]
        )
        self.assertGreaterEqual(merged["battle_reward_episode_cap"], one_full_trainer_cycle)

    def test_open_world_disables_per_fire_domain_balancing(self):
        """2026-10-25: per-fire domain normalisation priced one new cell at
        1.0 delivered and a story checkpoint or new species at ~1.3 — 93.5%
        of honest reward was coverage. open_world now uses explicit raw
        prices (balancer off) so rare progress events dominate by size."""
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "stages", "open_world.json")
        merged = merge_configs(load_default_config(), load_user_config(path))
        self.assertFalse(merged["domain_reward_balancing_enabled"])
        # Run-wide coverage decay starved the flag-reopened walk back to Elm
        # (honest flag 31 fell 40% -> 0% within ~120 episodes); keep it off.
        self.assertFalse(merged["frontier_lifelong_decay"])
        self.assertTrue(merged["species_catch_reward_enabled"])
        self.assertTrue(merged["checkpoint_competence_gating"])


if __name__ == "__main__":
    unittest.main()
