# -*- coding: utf-8 -*-
"""Dead-end correction for directional_frontier_potential (2026-07-14).

Root cause: the compass forecasts each neighbouring cell's payout from raw
coordinate arithmetic against the persistent VisitArchive, with no check
that the cell is actually reachable. A wall/obstacle tile is a cell the
player's (X, Y) can never equal, so its archive visit count is — and
permanently stays — 0, which plugs into the same formula as the single
freshest cell in the game (1.0, the ceiling). That reads as maximally
rewarding forever, an observation-level lure no amount of training decays,
because it isn't a function of anything training affects.

Fix: track, from ground truth, whichever direction was just tried and
failed to move the player (``_blocked_direction``), and force the compass
to report 0.0 for that direction instead of the archive-derived value.
Self-corrects every step; needs no assumption about collision-byte
semantics, so ledges/NPCs/cut-trees/un-surfed water all read "blocked now"
and clear the same way (the first successful step through them).

Pinned behaviours:

- A direction whose most recent attempt left (map, X, Y) unchanged reads
  0.0 in the compass, regardless of what the archive says about that cell.
- A direction that successfully moved the player is never forced to 0 by
  this mechanism — it uses the normal archive-derived forecast.
- A non-directional action (e.g. "a") leaves the previously-known blocked
  direction as-is — nothing about reachability changed.
- Scripted/dialogue frames and battle frames never register a block (the
  same gate already used for stagnation accounting) — movement is
  legitimately locked there for reasons unrelated to collision, and
  mistaking that for "wall" would reintroduce a false signal on every
  cutscene.
- start_new_episode clears the tracked block; a fresh episode never
  inherits a stale blocked-direction from the previous one.
"""
import unittest

from PoliwhiRL.environment.rewards import Rewards
from tests.test_frontier_gating import _base_config, _env_vars


class TestWallDeadEndCompass(unittest.TestCase):
    def test_blocked_direction_reads_zero_in_the_compass(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=10, y=10), "")
        # "up" pressed, but position doesn't change -> blocked.
        rw.calculate_reward(_env_vars(x=10, y=10), "up")
        out = rw.directional_frontier_potential(_env_vars(x=10, y=10))
        up, down, left, right = out
        self.assertEqual(up, 0.0)
        # The other three directions are untouched by the fix — fresh,
        # never-visited neighbours still read the ceiling value.
        self.assertEqual(down, 1.0)
        self.assertEqual(left, 1.0)
        self.assertEqual(right, 1.0)

    def test_successful_move_is_not_forced_to_zero(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=10, y=10), "")
        # "up" pressed and position actually changes -> not blocked.
        rw.calculate_reward(_env_vars(x=10, y=9), "up")
        out = rw.directional_frontier_potential(_env_vars(x=10, y=9))
        up, _down, _left, _right = out
        self.assertEqual(up, 1.0)

    def test_non_directional_action_preserves_prior_block(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=10, y=10), "")
        rw.calculate_reward(_env_vars(x=10, y=10), "up")  # blocked
        # Pressing "a" (interact) — position unchanged, but this isn't a
        # directional attempt, so the known block must survive untouched.
        rw.calculate_reward(_env_vars(x=10, y=10), "a")
        out = rw.directional_frontier_potential(_env_vars(x=10, y=10))
        up, _down, _left, _right = out
        self.assertEqual(up, 0.0)

    def test_a_later_successful_move_clears_the_block(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=10, y=10), "")
        rw.calculate_reward(_env_vars(x=10, y=10), "up")  # blocked
        # Player turns and successfully walks left instead.
        rw.calculate_reward(_env_vars(x=9, y=10), "left")
        out = rw.directional_frontier_potential(_env_vars(x=9, y=10))
        up, _down, _left, _right = out
        self.assertEqual(up, 1.0)

    def test_script_active_never_registers_a_false_block(self):
        """Dialogue legitimately locks movement regardless of collision —
        mistaking that for a wall would falsely zero the compass on every
        cutscene."""
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=10, y=10), "")
        rw.calculate_reward(
            _env_vars(x=10, y=10, script_active=True), "up"
        )
        out = rw.directional_frontier_potential(_env_vars(x=10, y=10))
        up, _down, _left, _right = out
        self.assertEqual(up, 1.0)

    def test_battle_never_registers_a_false_block(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=10, y=10), "")
        ev = _env_vars(x=10, y=10)
        ev["battle_type"] = 1
        rw.calculate_reward(ev, "up")
        out = rw.directional_frontier_potential(_env_vars(x=10, y=10))
        up, _down, _left, _right = out
        self.assertEqual(up, 1.0)

    def test_new_episode_clears_stale_block(self):
        rw = Rewards(_base_config())
        rw.calculate_reward(_env_vars(x=10, y=10), "")
        rw.calculate_reward(_env_vars(x=10, y=10), "up")  # blocked
        rw.start_new_episode()
        rw.calculate_reward(_env_vars(x=10, y=10), "")
        out = rw.directional_frontier_potential(_env_vars(x=10, y=10))
        up, _down, _left, _right = out
        self.assertEqual(up, 1.0)


if __name__ == "__main__":
    unittest.main()
