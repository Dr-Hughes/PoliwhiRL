# -*- coding: utf-8 -*-
"""Snapshot seed-state round trip (save-state restarts).

A snapshot carries *facts* about its source episode (maps entered, goal
keys fired, per-episode novelty sets). Restoring must:

- mark already-achieved goals complete against the CURRENT stage's goal
  list (facts are keys, not indices — stage 3 snapshots remain valid under
  stage 4's longer ladder);
- seed per-episode novelty sets (explored_maps, frontier cells) so the
  already-walked path cannot re-pay.
"""
import unittest

from PoliwhiRL.environment.goals import GoalsManager
from PoliwhiRL.environment.rewards import Rewards


def _config(goals, **over):
    cfg = {
        "episode_length": 100,
        "goals": goals,
        "new_map_reward": 0,
        "frontier_novelty_bonus": 0,
        "whiteout_penalty": 0,
    }
    cfg.update(over)
    return cfg


STAGE3_GOALS = [
    {"type": "map", "map_bank": 24, "map_num": 4},
    {"type": "pokedex", "kind": "owned", "threshold": 1},
    {"type": "map", "map_bank": 24, "map_num": 3},
]
STAGE4_GOALS = STAGE3_GOALS + [{"type": "map", "map_bank": 26, "map_num": 3}]


class TestGoalsSeedFacts(unittest.TestCase):
    def test_cross_stage_marking(self):
        # Facts from a stage-3 episode that completed the full stage-3
        # ladder, applied to a stage-4 goal list: 3 of 4 complete.
        g = GoalsManager(_config(STAGE4_GOALS))
        g.apply_seed_facts(
            map_goal_keys=[(24, 4), (24, 3)],
            flag_nums=[],
            pokedex_seen=3,
            pokedex_owned=1,
        )
        self.assertEqual(g.map_goals_completed, 2)
        self.assertEqual(g.pokedex_goals_completed, 1)
        self.assertEqual(g.N_goals, 3)
        self.assertFalse(g.all_goal_thresholds_met())  # Cherrygrove remains
        # The remaining map goal still fires on a genuine entry.
        g.check_map_goals(24, 4)  # establishes _map_initial (start map)
        fires = g.check_map_goals(26, 3)
        self.assertEqual(fires, 1)
        self.assertTrue(g.all_goal_thresholds_met())

    def test_fully_complete_facts_detectable(self):
        # Same-stage restore of a final-rung snapshot: everything already
        # met — restore_snapshot uses this to reject the start state.
        g = GoalsManager(_config(STAGE3_GOALS))
        g.apply_seed_facts([(24, 4), (24, 3)], [], 1, 1)
        self.assertTrue(g.all_goal_thresholds_met())

    def test_unknown_keys_ignored(self):
        g = GoalsManager(_config(STAGE3_GOALS))
        g.apply_seed_facts([(99, 99)], [123], 0, 0)
        self.assertEqual(g.N_goals, 0)


class TestRewardsSeedRoundTrip(unittest.TestCase):
    """Simulate what gym_env._capture_snapshot + restore_snapshot does."""

    def test_seed_explored_maps_prevents_double_pay(self):
        """seed_explored_maps pre-fills explored_maps so new_map bonus does
        not fire for maps the replay walked."""
        src = Rewards(_config(STAGE3_GOALS, new_map_reward=50))
        src.goals.seed_seen_maps([(24, 5), (24, 4)])
        src.explored_maps = {(24, 5), (24, 4)}
        src._novel_cells_this_episode = {(24, 4, 1, 2), (24, 5, 0, 0)}

        # Build facts dict (as _capture_snapshot does).
        facts = {
            "maps_seen": sorted(src.goals._maps_seen_this_episode),
            "explored_maps": sorted(src.explored_maps),
            "novel_cells": sorted(src._novel_cells_this_episode),
            "map_goals_fired": src.goals.fired_map_goal_keys(),
            "flag_goals_fired": src.goals.fired_flag_nums(),
        }

        # Restore into a fresh Rewards (as restore_snapshot does).
        dst = Rewards(_config(STAGE3_GOALS, new_map_reward=50))
        dst.seed_explored_maps(facts.get("explored_maps", []))
        for cell in facts.get("novel_cells", []):
            dst._novel_cells_this_episode.add(
                tuple(cell) if isinstance(cell, list) else cell
            )
        dst.goals.seed_seen_maps(facts.get("maps_seen", []))
        dst.goals.apply_seed_facts(
            facts.get("map_goals_fired", []),
            facts.get("flag_goals_fired", []),
            0, 0,
        )
        dst._prev_rung = dst.n_flag_goals_completed() + dst.n_map_goals_completed()

        # Per-episode novelty sets seeded — the walked path can't re-pay.
        self.assertEqual(dst.explored_maps, {(24, 5), (24, 4)})
        self.assertEqual(
            dst._novel_cells_this_episode, {(24, 4, 1, 2), (24, 5, 0, 0)}
        )
        self.assertEqual(
            dst.goals._maps_seen_this_episode, {(24, 5), (24, 4)}
        )

    def test_goal_progress_carried_across_restore(self):
        """Goals achieved on the source path are marked complete, and only
        the remaining goals can fire."""
        src = Rewards(_config(STAGE3_GOALS))
        src.goals.apply_seed_facts([(24, 4)], [], 0, 0)
        facts = {
            "maps_seen": [],
            "explored_maps": [],
            "novel_cells": [],
            "map_goals_fired": src.goals.fired_map_goal_keys(),
            "flag_goals_fired": [],
        }

        dst = Rewards(_config(STAGE3_GOALS))
        dst.goals.seed_seen_maps(facts.get("maps_seen", []))
        dst.goals.apply_seed_facts(
            facts.get("map_goals_fired", []),
            facts.get("flag_goals_fired", []),
            0, 0,
        )

        self.assertEqual(dst.goals.map_goals_completed, 1)
        self.assertFalse(dst.goals.all_goal_thresholds_met())

    def test_facts_survive_json_like_lists(self):
        # pickle keeps tuples, but be robust to list-ified keys anyway.
        dst = Rewards(_config(STAGE3_GOALS))
        explored = [[24, 4]]
        cells = [[24, 4, 1, 2]]
        dst.seed_explored_maps(explored)
        for cell in cells:
            dst._novel_cells_this_episode.add(
                tuple(cell) if isinstance(cell, list) else cell
            )
        dst.goals.apply_seed_facts([[24, 4]], [], 0, 0)

        self.assertIn((24, 4), dst.explored_maps)
        self.assertIn((24, 4, 1, 2), dst._novel_cells_this_episode)
        self.assertEqual(dst.goals.map_goals_completed, 1)


if __name__ == "__main__":
    unittest.main()
