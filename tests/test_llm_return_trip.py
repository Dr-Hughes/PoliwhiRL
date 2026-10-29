# -*- coding: utf-8 -*-
"""The return trip from Mr. Pokémon's (leave, Cherrygrove, heal, rival) and
the harness's healing prompt."""
import os
import unittest

from tools.llm_agent.agent import heal_facts, heal_reasons

ROM = "./emu_files/Pokemon - Crystal Version.gbc"
EGG_STATE = "./Manual Investigation States/mr_pokemons_house_with_egg.state"


def _s(**kw):
    s = {
        "map": "ROUTE_30",
        "mode": "walk",
        "party": 1,
        "battle": None,
        "fainted": 0,
        "party_hp_frac": 1.0,
        "party_pp_frac": 1.0,
        "routes": [],
    }
    s.update(kw)
    return s


class TestHealPrompt(unittest.TestCase):
    def test_healthy_party_gets_nothing(self):
        self.assertEqual(heal_reasons(_s()), [])
        self.assertEqual(heal_facts(_s(), {}), [])
        self.assertEqual(heal_facts(_s(party=0, party_hp_frac=0.0), {}), [])

    def test_each_reason(self):
        self.assertIn("1 of your Pokémon fainted", heal_reasons(_s(fainted=1)))
        self.assertIn("your party has 40% of its HP", heal_reasons(_s(party_hp_frac=0.4)))
        self.assertIn("your moves have 30% of their PP left", heal_reasons(_s(party_pp_frac=0.3)))
        self.assertEqual(heal_reasons(_s(party_hp_frac=0.5, party_pp_frac=0.5)), [])

    def test_route_to_a_center_on_this_map(self):
        s = _s(
            map="CHERRYGROVE_CITY",
            party_hp_frac=0.3,
            routes=[
                {
                    "to": "CHERRYGROVE_POKECENTER_1F",
                    "via": "door at (29,3)",
                    "steps": 7,
                    "first": "up",
                },
                {"to": "ROUTE_29", "via": "east edge", "steps": 20, "first": "right"},
            ],
        )
        facts = heal_facts(s, {})
        self.assertIn("SUGGESTED SUBGOAL", facts[0])
        self.assertIn(
            "follow the route to CHERRYGROVE_POKECENTER_1F (door at (29,3)), "
            "FIRST MOVE UP, 7 steps",
            facts[1],
        )

    def test_inside_a_center_and_centers_seen(self):
        inside = heal_facts(_s(map="CHERRYGROVE_POKECENTER_1F", fainted=1), {})
        self.assertIn("You are in a Pokémon Center", inside[1])
        far = heal_facts(_s(party_pp_frac=0.1), {"CHERRYGROVE_POKECENTER_1F": "CHERRYGROVE_CITY"})
        self.assertIn("CHERRYGROVE_POKECENTER_1F (in CHERRYGROVE_CITY)", far[1])

    def test_not_in_battle(self):
        self.assertEqual(heal_facts(_s(fainted=1, battle={"type": "wild"}), {}), [])


@unittest.skipUnless(
    os.path.exists(ROM) and os.path.exists(EGG_STATE), "ROM/save-state not available"
)
class TestReturnTrip(unittest.TestCase):
    """Walk the return trip by the harness's own routes, pressing a through
    every battle and scene, and check each goal fires where it should."""

    def test_goals_fire_along_the_way_and_out_of_order(self):
        from tools.llm_agent.game import Game
        from tools.llm_agent.goals import GOALS

        g = Game("configs/stages/open_world.json", state_path=EGG_STATE)
        keys = ("leave_mr_pokemon", "cherrygrove_return", "heal_cherrygrove", "rival")
        try:
            done = lambda: {k: GOALS[k].done(g, g.vars()) for k in keys}  # noqa: E731
            self.assertEqual(done(), dict.fromkeys(keys, False))
            first_done = {}
            for target in ("ROUTE_30", "CHERRYGROVE_CITY", "ROUTE_29"):
                for i in range(1500):
                    s = g.status()
                    if s["map"] == target and s["mode"] == "walk":
                        break
                    if s["mode"] == "walk":
                        r = next(
                            (r for r in g.routes() if r["to"] == target and r["steps"] is not None),
                            None,
                        )
                        exit_ = ((s.get("on_exit") or {}).get("leave") or ["down"])[0]
                        g.press((r["first"] or exit_) if r else "a")
                    else:
                        g.press("a")
                    for k, v in done().items():
                        if v and k not in first_done:
                            first_done[k] = g.status()["map"]
            self.assertEqual(first_done["leave_mr_pokemon"], "ROUTE_30")
            self.assertEqual(first_done["cherrygrove_return"], "CHERRYGROVE_CITY")
            # The rival battle happens in Cherrygrove; skipping the Pokémon
            # Center there marks the heal step done with it, not before.
            self.assertEqual(first_done["rival"], "CHERRYGROVE_CITY")
            self.assertEqual(first_done["heal_cherrygrove"], "CHERRYGROVE_CITY")
            self.assertTrue(all(done().values()))
            self.assertFalse(GOALS["back_to_elm"].done(g, g.vars()))
        finally:
            g.close()

    def test_the_route_to_the_nurse_heals_the_party(self):
        from tools.llm_agent.game import Game
        from tools.llm_agent.goals import GOALS

        g = Game("configs/stages/open_world.json", state_path=EGG_STATE)
        try:
            for target in ("ROUTE_30", "CHERRYGROVE_CITY", "CHERRYGROVE_POKECENTER_1F"):
                for i in range(1500):
                    s = g.status()
                    if s["map"] == target and s["mode"] == "walk":
                        break
                    if s["mode"] == "walk":
                        r = next(
                            (r for r in g.routes() if r["to"] == target and r["steps"] is not None),
                            None,
                        )
                        exit_ = ((s.get("on_exit") or {}).get("leave") or ["down"])[0]
                        g.press((r["first"] or exit_) if r else "a")
                    else:
                        g.press("a")
            s = g.status()
            self.assertEqual(s["map"], "CHERRYGROVE_POKECENTER_1F")
            # the wild battles on Route 30 cost HP
            self.assertLess(s["party_hp_frac"], 1.0)
            self.assertFalse(GOALS["heal_cherrygrove"].done(g, g.vars()))
            # the nurse is behind the counter: the route goes across it
            r = next(r for r in g.routes() if r["to"] == "nurse")
            self.assertTrue(r["across"])
            for leg in r["path"].split(", "):
                d, n = leg.split(" ")
                for _ in range(int(n)):
                    g.press(d)
            g.press(r["face"])
            for i in range(60):
                g.press("a")
                if g.status()["mode"] == "walk" and i > 3:
                    break
            s = g.status()
            self.assertEqual((s["party_hp_frac"], s["party_pp_frac"]), (1.0, 1.0))
            self.assertTrue(GOALS["heal_cherrygrove"].done(g, g.vars()))
            self.assertFalse(GOALS["rival"].done(g, g.vars()))
        finally:
            g.close()


if __name__ == "__main__":
    unittest.main()
