# -*- coding: utf-8 -*-
"""tools/world_map.py — offline, post-hoc tile/discovery analysis.

Pure synthetic-data tests: no PyBoy, no training run. Validates the
inference rules (walkable / wall / warp / encounter-rate) against
scripted step sequences, plus filename parsing and the discovery-log
loader's sort/format behaviour.
"""
import unittest

from tools.world_map import (
    StepRecord, WorldMap, parse_filename, format_discovery_timeline,
)


def _step(step, x, y, action, map_bank=24, map_num=7, battle_type="none",
          player_state="walk", warp=0):
    return StepRecord(
        step=step, map_bank=map_bank, map_num=map_num, x=x, y=y,
        action=action, battle_type=battle_type, player_state=player_state,
        warp=warp,
    )


class TestFilenameParsing(unittest.TestCase):
    def test_parses_current_format_with_warp(self):
        name = (
            "step_5_x_3_y_4_map_7_bank_24_room_0_battlestate_none"
            "_playerstate_walking_warp_2_btn_up_reward_0.5.png"
        )
        rec = parse_filename(name)
        self.assertIsNotNone(rec)
        self.assertEqual((rec.step, rec.x, rec.y, rec.map_num, rec.map_bank), (5, 3, 4, 7, 24))
        self.assertEqual(rec.warp, 2)
        self.assertEqual(rec.action, "up")

    def test_parses_older_format_without_warp(self):
        """Recordings made before the warp field existed must still parse
        (missing warp -> None, not a crash)."""
        name = (
            "step_5_x_3_y_4_map_7_bank_24_room_0_battlestate_none"
            "_playerstate_walking_btn_up_reward_0.5.png"
        )
        rec = parse_filename(name)
        self.assertIsNotNone(rec)
        self.assertIsNone(rec.warp)

    def test_non_matching_filename_returns_none(self):
        self.assertIsNone(parse_filename("not_a_step_file.png"))


class TestWalkabilityInference(unittest.TestCase):
    def test_successful_move_marks_walkable(self):
        world = WorldMap()
        prev = _step(0, x=5, y=5, action="right")
        cur = _step(1, x=6, y=5, action="")
        world.update(prev, cur)
        tile = world.tiles[(24, 7, 5, 5)]
        self.assertTrue(tile.walkable["right"])

    def test_blocked_move_marks_wall_on_target_cell(self):
        world = WorldMap()
        prev = _step(0, x=5, y=5, action="right")
        cur = _step(1, x=5, y=5, action="")  # position unchanged -> blocked
        world.update(prev, cur)
        prev_tile = world.tiles[(24, 7, 5, 5)]
        self.assertFalse(prev_tile.walkable["right"])
        wall_tile = world.tiles[(24, 7, 6, 5)]
        self.assertTrue(wall_tile.known_wall)
        # The bumped-into cell was never actually stood on.
        self.assertFalse(wall_tile.discovered)

    def test_battle_state_does_not_infer_a_wall(self):
        """Standing still during a battle must not be read as a blocked
        movement — the player wasn't trying to walk. Mirrors the
        battle_type gate Rewards uses for stagnation/blocked-direction
        accounting (see rewards.py)."""
        world = WorldMap()
        prev = _step(0, x=5, y=5, action="right", battle_type="wild")
        cur = _step(1, x=5, y=5, action="", battle_type="wild")
        world.update(prev, cur)
        self.assertNotIn("right", world.tiles[(24, 7, 5, 5)].walkable)
        self.assertFalse(world.tiles.get((24, 7, 6, 5), type("T", (), {"known_wall": False})).known_wall)

    def test_recorded_player_state_labels_are_never_gated_on(self):
        """Regression guard: recorded PNGs use PLAYER_STATE_LABELS values
        ("walk"/"bike"/"skate"/"surf"), never the literal "walking". A
        previous version of this tool gated walkability inference on
        `player_state == "walking"`, which never matched real data and
        silently disabled the entire feature — undetected because these
        fixtures' default also used the wrong literal. Any of the real
        labels must still infer walkability/walls."""
        for label in ("walk", "bike", "skate", "surf"):
            world = WorldMap()
            prev = _step(0, x=5, y=5, action="right", player_state=label)
            cur = _step(1, x=5, y=5, action="", player_state=label)
            world.update(prev, cur)
            self.assertTrue(
                world.tiles[(24, 7, 6, 5)].known_wall,
                f"wall not inferred for player_state={label!r}",
            )

    def test_non_directional_action_infers_nothing(self):
        world = WorldMap()
        prev = _step(0, x=5, y=5, action="a")
        cur = _step(1, x=5, y=5, action="")
        world.update(prev, cur)
        self.assertEqual(world.tiles[(24, 7, 5, 5)].walkable, {})

    def test_map_transition_is_not_treated_as_a_walkability_edge(self):
        world = WorldMap()
        prev = _step(0, x=5, y=5, action="right", map_num=7)
        world.update(None, prev)
        cur = _step(1, x=0, y=0, action="", map_num=9)
        world.update(prev, cur)
        self.assertEqual(world.tiles[(24, 7, 5, 5)].walkable, {})


class TestWarpAndEncounterInference(unittest.TestCase):
    def test_warp_number_change_tags_prior_tile(self):
        world = WorldMap()
        prev = _step(0, x=5, y=5, action="down", warp=1)
        cur = _step(1, x=5, y=6, action="", warp=2)
        world.update(prev, cur)
        self.assertTrue(world.tiles[(24, 7, 5, 5)].warp)

    def test_encounter_rate_tracks_battle_visits(self):
        world = WorldMap()
        world.update(None, _step(0, x=5, y=5, action="", battle_type="wild"))
        world.update(_step(0, x=5, y=5, action=""), _step(1, x=5, y=5, action="", battle_type="none"))
        tile = world.tiles[(24, 7, 5, 5)]
        self.assertEqual(tile.encounter_visits, 2)
        self.assertEqual(tile.encounter_hits, 1)
        self.assertAlmostEqual(tile.encounter_rate, 0.5)


class TestRenderAscii(unittest.TestCase):
    def test_render_shows_discovered_and_wall_cells(self):
        world = WorldMap()
        world.load_episode([
            _step(0, x=0, y=0, action="right"),
            _step(1, x=1, y=0, action="right"),
            _step(2, x=1, y=0, action=""),  # blocked -> wall at (2,0)
        ])
        rendered = world.render_ascii(24, 7)
        rows = rendered.split("\n")
        self.assertEqual(rows[0], "..#")

    def test_no_data_returns_placeholder(self):
        world = WorldMap()
        self.assertEqual(world.render_ascii(24, 99), "(no data for this map)")

    def test_local_patch_marks_center(self):
        world = WorldMap()
        world.update(None, _step(0, x=5, y=5, action=""))
        patch = world.local_patch(24, 7, 5, 5, radius=1)
        rows = patch.split("\n")
        self.assertEqual(rows[1][1], "P")


class TestDiscoveryTimelineFormatting(unittest.TestCase):
    def test_format_is_stable_and_readable(self):
        log = [
            {"episode": 10, "rollout_idx": 2, "step": 50, "type": "flag", "key": 26},
            {"episode": 5, "rollout_idx": 1, "step": 5, "type": "map", "key": [24, 3]},
        ]
        text = format_discovery_timeline(log)
        self.assertIn("flag", text)
        self.assertIn("map", text)
        self.assertEqual(len(text.split("\n")), 2)


if __name__ == "__main__":
    unittest.main()
