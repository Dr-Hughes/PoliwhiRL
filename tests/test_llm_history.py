# -*- coding: utf-8 -*-
"""tools/llm_agent/history.py and the route commitment: the harness-written
history the model sees, and the check that it presses the route it names."""
import unittest

from tools.llm_agent.agent import StallCounter, check_following, route_orders
from tools.llm_agent.history import EpisodeSummary, TurnLog, compact_result, goal_names, route_named


def _s(map_name="ROUTE_29", x=5, y=5, mode="walk", map_id=(24, 3), **kw):
    s = {
        "map": map_name,
        "map_id": map_id,
        "x": x,
        "y": y,
        "mode": mode,
        "text": [],
        "in_battle": False,
    }
    s.update(kw)
    return s


ROUTES = [
    {"to": "CHERRYGROVE_CITY", "via": "west edge", "steps": 71, "first": "up", "path": "up 1"},
    {"to": "NEW_BARK_TOWN", "via": "east edge", "steps": 51, "first": "up", "path": "up 1"},
    {
        "to": "PROF.ELM",
        "via": "at (5,2)",
        "steps": 0,
        "first": "",
        "path": "",
        "face": "right",
        "person": True,
    },
]


class TestCompactResult(unittest.TestCase):
    def test_moves_collapse_to_where_they_ended(self):
        self.assertEqual(
            compact_result(
                "left: moved you to (23,15); left: moved you to (22,15); left: BLOCKED (blocked), "
                "now facing left; (stopped early: blocked; skipped left)"
            ),
            "moved to (22,15); left: BLOCKED (blocked), now facing left; (stopped: blocked)",
        )

    def test_dialogue_presses_collapse(self):
        self.assertEqual(
            compact_result("a: ok; a: ok; a: ok; (the scene/text ended, you can walk)"),
            "ok; (the scene/text ended, you can walk)",
        )
        self.assertEqual(
            compact_result("down: entered NEW_BARK_TOWN at (6,4); (map changed)"),
            "down: entered NEW_BARK_TOWN at (6,4); (map changed)",
        )


class TestTurnLog(unittest.TestCase):
    def test_the_window_only_grows_within_a_block(self):
        log = TurnLog(block=25)
        for i in range(1, 80):
            log.add(i, _s(x=i), ["left"], "left: moved you to (1,1)")
        # 79 turns: the last full block (turns 51-75) and the current one (76-79)
        shown = log.window()
        self.assertEqual(len(shown), 29)
        self.assertTrue(shown[0].startswith("t51 "))
        log.add(80, _s(), ["up"], "up: ok")
        self.assertTrue(log.window()[0].startswith("t51 "))  # same start: cacheable
        self.assertIn("50 earlier turns are summarised above", log.format())
        self.assertIn("t80 ROUTE_29 (5,5): up -> ok", log.format())


class TestEpisodeSummary(unittest.TestCase):
    def test_summarises_per_map_and_names_loops(self):
        ep = EpisodeSummary(block=25)
        a, b = _s(x=27, y=14), _s(x=26, y=14)
        for _ in range(3):
            ep.turn(a)
            ep.step(a, b)
            ep.step(b, a)
        ep.step(_s(x=59, y=8), _s("NEW_BARK_TOWN", 0, 8, map_id=(24, 4)))
        ep.step(a, _s(x=27, y=14, in_battle=True))
        ep.talk("ROUTE_29", "youngster", 27, 15, ["I'm waiting for", "my friend."])
        text = ep.text(3, [("ROUTE_29 (27,14)", 9)])
        self.assertIn(
            "- ROUTE_29: 3 turns, stood on 3 different cells (x 26-59, y 8-14), 1 battles", text
        )
        self.assertIn("left it via: (59,8) -> NEW_BARK_TOWN", text)
        self.assertIn('talked to the youngster at (27,15): "I\'m waiting for my friend."', text)
        self.assertIn("  - ROUTE_29 (27,14): 9 times", text)
        self.assertIn("ROUTE_29 3 turns", ep.one_line())

    def test_rebuilt_only_every_block(self):
        ep = EpisodeSummary(block=25)
        ep.turn(_s())
        first = ep.text(1)
        ep.turn(_s("NEW_BARK_TOWN"))
        self.assertEqual(ep.text(10), first)
        self.assertIn("NEW_BARK_TOWN", ep.text(26))


class TestRoutes(unittest.TestCase):
    def test_goal_names(self):
        goal = "Travel west along Route 29 until you reach Cherrygrove City."
        self.assertTrue(goal_names(goal, "CHERRYGROVE_CITY"))
        self.assertTrue(goal_names(goal, "ROUTE_29"))
        self.assertFalse(goal_names(goal, "ROUTE_30"))
        self.assertFalse(goal_names(goal, "NEW_BARK_TOWN"))
        self.assertTrue(goal_names("Talk to Elm and give it to him.", "PROF.ELM"))
        self.assertTrue(goal_names("Bring it back to Professor Elm's lab", "ELMS_LAB"))

    def test_route_named(self):
        self.assertEqual(route_named(ROUTES, "CHERRYGROVE_CITY")["via"], "west edge")
        self.assertEqual(route_named(ROUTES, "cherrygrove city")["to"], "CHERRYGROVE_CITY")
        self.assertEqual(route_named(ROUTES, "Elm")["to"], "PROF.ELM")
        self.assertIsNone(route_named(ROUTES, "none"))
        self.assertIsNone(route_named(ROUTES, ""))

    def test_check_following(self):
        s = _s(routes=ROUTES)
        ok = check_following(s, {"following": "CHERRYGROVE_CITY", "buttons": ["up", "right"]})
        self.assertTrue(ok["complied"])
        bad = check_following(s, {"following": "CHERRYGROVE_CITY", "buttons": ["left"]})
        self.assertEqual((bad["complied"], bad["expected"], bad["pressed"]), (False, "up", "left"))
        # next to a person: face them, or talk if already facing them
        self.assertTrue(check_following(s, {"following": "PROF.ELM", "buttons": ["a"]})["complied"])
        self.assertIsNone(check_following(s, {"following": "none", "buttons": ["a"]}))
        self.assertIsNone(
            check_following(
                _s(mode="battle", routes=ROUTES),
                {"following": "CHERRYGROVE_CITY", "buttons": ["a"]},
            )
        )

    def test_route_orders_say_closer_or_further(self):
        followed = {
            "to": "CHERRYGROVE_CITY",
            "via": "west edge",
            "steps": 71,
            "expected": "up",
            "pressed": "left",
            "complied": False,
        }
        now = _s(routes=[dict(ROUTES[0], steps=73)])
        lines = route_orders(followed, now)
        self.assertIn("but you pressed LEFT; that route's first move was UP", lines[0])
        self.assertIn("73 steps now, 71 before your last turn (FURTHER AWAY", lines[1])
        self.assertIn("Its next move is UP.", lines[1])
        followed.update(complied=True)
        lines = route_orders(followed, _s(routes=[dict(ROUTES[0], steps=69)]))
        self.assertEqual(len(lines), 1)
        self.assertIn("(closer)", lines[0])
        self.assertEqual(route_orders(None, now), [])


class TestLoops(unittest.TestCase):
    def test_loops_are_named_most_frequent_first(self):
        c = StallCounter()
        for _ in range(7):
            c.observe(("a",), label="ROUTE_29 (27,14)")
        for _ in range(9):
            c.observe(("b", None), label="ROUTE_29 (14,15)")
        c.observe(("c",), label="rare")
        self.assertEqual(c.loops(at_least=6), [("ROUTE_29 (14,15)", 9), ("ROUTE_29 (27,14)", 7)])


if __name__ == "__main__":
    unittest.main()
