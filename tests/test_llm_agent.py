# -*- coding: utf-8 -*-
"""tools/llm_agent — the LLM game-playing harness.

Pure tests (no model server): decision parsing, prompt state formatting and
the reward audit. The emulator-backed checks (warp decode, goal predicates
along a scripted path) skip cleanly without the ROM.
"""
import os
import unittest

from tools.llm_agent.agent import format_state
from tools.llm_agent.llm import parse_decision
from tools.llm_agent.report import reward_audit
from tools.llm_agent.agent import Memory
from tools.llm_agent.notebook import MAX_WORDS, Notebook
from tools.llm_agent.verify import RewardVerifier

ROM = "./emu_files/Pokemon - Crystal Version.gbc"
STATE = "./emu_files/states/start.state"


class TestParseDecision(unittest.TestCase):
    def test_tool_arguments(self):
        d, err = parse_decision('{"observation": "o", "buttons": ["Up", "a"]}')
        self.assertIsNone(err)
        self.assertEqual(d["buttons"], ["up", "a"])

    def test_json_embedded_in_content(self):
        d, err = parse_decision('Sure!\n{"buttons": ["down"], "plan": "p"}\nDone')
        self.assertIsNone(err)
        self.assertEqual(d["buttons"], ["down"])

    def test_filters_and_truncates_buttons(self):
        d, _ = parse_decision('{"buttons": ["select", "up", "up", "up", "up", "up"]}')
        self.assertEqual(d["buttons"], ["up"] * 4)

    def test_string_buttons(self):
        d, _ = parse_decision('{"buttons": "left, left, a"}')
        self.assertEqual(d["buttons"], ["left", "left", "a"])

    def test_rejects_no_buttons(self):
        d, err = parse_decision('{"buttons": ["select"]}')
        self.assertIsNone(d)
        self.assertIn("no valid buttons", err)
        self.assertEqual(parse_decision("")[0], None)
        self.assertEqual(parse_decision("no json here")[0], None)


class TestRememberRecallParsing(unittest.TestCase):
    def test_remember_and_recall_normalised(self):
        d, _ = parse_decision('{"buttons": ["a"], "remember": "Mr. Pokemon is north", "recall": " egg "}')
        self.assertEqual(d["remember"], ["Mr. Pokemon is north"])
        self.assertEqual(d["recall"], "egg")
        d, _ = parse_decision('{"buttons": ["a"]}')
        self.assertEqual((d["remember"], d["recall"]), ([], ""))


class TestNotebook(unittest.TestCase):
    def test_add_dedupes_and_search_ranks(self):
        nb = Notebook()
        nb.add(["Mr. Pokemon lives north of Cherrygrove on Route 30"], goal="mr_pokemon",
               map_name="CHERRYGROVE_CITY", turn=1)
        nb.add(["mr pokemon lives north of cherrygrove on route 30!"], goal="g", map_name="m", turn=2)
        nb.add(["Press B to decline the nickname", ""], goal="starter", map_name="ELMS_LAB", turn=3)
        self.assertEqual(len(nb.facts), 2)
        hits = nb.search("where does Mr Pokemon live", turn=4)
        self.assertIn("Mr. Pokemon", hits[0]["text"])
        self.assertEqual(hits[0]["uses"], 1)
        self.assertEqual(nb.search("xyzzy"), [])

    def test_export_load_roundtrip(self):
        nb = Notebook()
        nb.add(["Elm's lab is west of your house"], goal="g", map_name="m", turn=1)
        nb2 = Notebook()
        nb2.load(nb.export())
        self.assertEqual(nb2.search("lab")[0]["text"], "Elm's lab is west of your house")
        nb2.add(["a different fact entirely"], goal="g", map_name="m", turn=2)
        self.assertEqual(sorted(f["id"] for f in nb2.facts), [1, 2])

    def test_word_cap_drops_least_used_oldest(self):
        nb = Notebook()
        nb.add(["keep this fact about the gym leader"], goal="g", map_name="m", turn=0)
        nb.search("gym leader")  # used once: survives
        for i in range(60):
            nb.add([" ".join(f"w{i}x{j}" for j in range(14))], goal="g", map_name="m", turn=i)
        self.assertLessEqual(nb.words(), MAX_WORDS)
        self.assertTrue(any("gym leader" in f["text"] for f in nb.facts))
        self.assertFalse(any(f["text"].startswith("w0x0 ") for f in nb.facts))
        self.assertTrue(any(f["text"].startswith("w59x0 ") for f in nb.facts))

    def test_memory_shows_recall_once_and_auto_recalls_when_stuck(self):
        m = Memory()
        m.notebook.add(["Elm's lab is the building west of your house"], goal="starter",
                       map_name="NEW_BARK_TOWN", turn=1)
        m.recall_query, m.recall_hits = "lab", m.notebook.search("lab")
        text = m.format("Get your first Pokemon from Elm", "NEW_BARK_TOWN")
        self.assertIn('Results of your recall "lab"', text)
        self.assertNotIn("Results of your recall", m.format("x", "y"))
        stuck = m.format("Get your first Pokemon from Elm", "NEW_BARK_TOWN", stall=6)
        self.assertIn("You are stuck; facts you saved", stuck)


class TestFormatState(unittest.TestCase):
    def test_relative_offsets(self):
        s = {
            "map": "PLAYERS_HOUSE_2F", "map_id": (24, 7), "x": 3, "y": 3,
            "facing": "down", "text_open": False, "in_battle": False, "party": 0,
            "around": {"up": "open", "down": "blocked", "left": "open", "right": "exit"},
            "exits": [{"x": 7, "y": 0, "to": "PLAYERS_HOUSE_1F", "dx": 4, "dy": -3}],
            "people": [{"x": 3, "y": 5, "dx": 0, "dy": 2}],
        }
        text = format_state(s)
        self.assertIn("to PLAYERS_HOUSE_1F at (x=7, y=0): 4 right, 3 up from you", text)
        self.assertIn("at (x=3, y=5): 2 down from you", text)
        self.assertIn("you can walk", text)

    def _route29(self, **kw):
        s = {
            "map": "ROUTE_29", "map_id": (24, 3), "x": 5, "y": 5, "facing": "up",
            "mode": "walk", "text_open": False, "in_battle": False, "party": 1,
            "party_mons": [{"level": 5, "hp": 18, "max_hp": 20}],
            "pokeballs": 5, "money": 3000, "badges": 0, "pokedex_owned": 1,
            "around": {"up": "open", "down": "open", "left": "open", "right": "open"},
            "exits": [], "people": [],
            "connections": {"west": {"to": "CHERRYGROVE_CITY", "cross": "y=6-9"},
                            "north": {"to": "ROUTE_46", "cross": ""}},
            "local_map": [". . @ , ,"],
        }
        s.update(kw)
        return s

    def test_walk_shows_map_and_edge_crossings(self):
        text = format_state(self._route29())
        self.assertIn("#1 Lv5 HP 18/20", text)
        self.assertIn("west edge leads to CHERRYGROVE_CITY (walkable at y=6-9)", text)
        self.assertIn("north edge leads to ROUTE_46 (no walkable crossing", text)
        self.assertIn(". . @ , ,", text)

    def test_walking_routes(self):
        text = format_state(self._route29(routes=[
            {"to": "CHERRYGROVE_CITY", "via": "west edge", "steps": 12,
             "path": "left 4, up (hop the ledge), left 6, then left off the edge"},
            {"to": "NEW_BARK_TOWN", "via": "east edge", "steps": None, "path": ""}]))
        self.assertIn("to CHERRYGROVE_CITY (west edge): FIRST MOVE LEFT; 12 steps: left 4, up (hop the ledge)", text)
        self.assertIn("to NEW_BARK_TOWN (east edge): no walking route from here", text)

    def test_battle_hides_overworld_and_names_enemy(self):
        mon = {"species": "RATTATA", "level": 3, "hp": 8, "max_hp": 14}
        you = {"species": "CYNDAQUIL", "level": 5, "hp": 18, "max_hp": 20}
        text = format_state(self._route29(
            mode="battle", text_open=True, in_battle=True,
            battle={"type": "wild", "enemy": mon, "you": you, "move_pp": [34, 30, 0, 0]}))
        self.assertIn("wild BATTLE is running", text)
        self.assertIn("enemy RATTATA Lv3 HP 8/14; your CYNDAQUIL Lv5 HP 18/20", text)
        self.assertNotIn("LOCAL MAP", text)
        self.assertNotIn("Next to you", text)
        self.assertNotIn("CHERRYGROVE_CITY", text)


class _FakeGame:
    def party(self):
        return [{"level": 5, "hp": 20, "max_hp": 20}]


def _vars(**kw):
    v = {"map_bank": 24, "map_num": 4, "story_flags": [0] * 256, "battle_type": 0,
         "enemy_hp": 0, "pokedex_owned": 1, "pokedex_seen": 1, "party_info": (1, 5, 20, 0),
         "key_items_count": 0, "script_active": False}
    v.update(kw)
    return v


class TestRewardVerifier(unittest.TestCase):
    def test_paid_missed_and_unexplained(self):
        closed = []
        rv = RewardVerifier(on_event=closed.append)
        g = _FakeGame()
        rv.step(1, "g", 1, "up", _vars(), g, {})
        # new map entered during a scene, new_map never pays -> MISSED
        rv.step(2, "g", 1, "up", _vars(map_num=5, script_active=True), g, {})
        # derived story flag 26 (got_starter) with checkpoint paid -> paid
        flags = [0] * 256
        flags[3] = 1 << 2
        rv.step(3, "g", 1, "a", _vars(map_num=5, story_flags=flags), g, {"checkpoint_honest": 40.0})
        # catch paid by pokedex
        rv.step(4, "g", 1, "a", _vars(map_num=5, story_flags=flags, pokedex_owned=2), g, {"pokedex": 5.0})
        # a big key_item payout with no key-item event -> unexplained
        rv.step(5, "g", 1, "a", _vars(map_num=5, story_flags=flags, pokedex_owned=2), g, {"key_item": 3.0})
        rv.flush()
        verdicts = {e.kind: e.verdict for e in rv.events}
        self.assertEqual(verdicts["map_first_entry"], "MISSED")
        self.assertTrue(rv.events[0].context["scene_active"])
        self.assertEqual(verdicts["story_checkpoint"], "paid")
        self.assertEqual(verdicts["catch"], "paid")
        self.assertEqual([u["source"] for u in rv.unexplained], ["key_item"])
        self.assertEqual(len(closed), len(rv.events))

    def test_flag_set_at_episode_start_is_not_a_checkpoint(self):
        rv = RewardVerifier()
        g = _FakeGame()
        flags = [0] * 256
        flags[1726 // 8] = 1 << (1726 % 8)
        rv.step(1, "g", 1, "a", _vars(story_flags=flags), g, {})
        rv.step(2, "g", 1, "a", _vars(), g, {})
        rv.step(3, "g", 1, "a", _vars(story_flags=flags), g, {})
        rv.flush()
        self.assertNotIn("story_checkpoint", [e.kind for e in rv.events])

    def test_new_map_window_waits_for_scene_end(self):
        rv = RewardVerifier()
        g = _FakeGame()
        rv.step(1, "g", 1, "up", _vars(), g, {})
        rv.step(2, "g", 1, "up", _vars(map_num=5, script_active=True), g, {})
        for step in range(3, 40):
            rv.step(step, "g", 1, "a", _vars(map_num=5, script_active=True), g, {})
        rv.step(40, "g", 1, "a", _vars(map_num=5), g, {"new_map": 5.0})
        rv.flush()
        ev = next(e for e in rv.events if e.kind == "map_first_entry")
        self.assertEqual(ev.verdict, "paid")

    def test_battle_won_and_new_episode(self):
        rv = RewardVerifier()
        g = _FakeGame()
        rv.step(1, "g", 1, "a", _vars(), g, {})
        rv.step(2, "g", 1, "a", _vars(battle_type=1, enemy_hp=14), g, {"battle": 1.0})
        rv.step(3, "g", 1, "a", _vars(battle_type=1, enemy_hp=0), g, {})
        rv.step(4, "g", 1, "a", _vars(battle_type=0), g, {"battle": 2.0})
        rv.flush()
        self.assertEqual([e.kind for e in rv.events], ["battle_start", "battle_won"])
        self.assertTrue(all(e.verdict == "paid" for e in rv.events))
        rv.new_episode()
        rv.step(5, "g", 2, "a", _vars(map_num=5), g, {})
        self.assertEqual(len(rv.events), 2)  # first read of a new episode emits nothing


class TestRewardAudit(unittest.TestCase):
    def test_sums_by_goal_attempt_and_source(self):
        steps = [
            {"goal": "g", "attempt": 1, "reward": 1.0, "sources": {"new_map": 1.0, "delivered_total": 1.0}},
            {"goal": "g", "attempt": 1, "reward": -0.5, "sources": {"frontier_potential": -0.5}},
            {"goal": "g", "attempt": 2, "reward": 0.2, "sources": {"frontier": 0.2}},
        ]
        audit = reward_audit(steps)
        self.assertEqual(audit[("g", 1)]["steps"], 2)
        self.assertAlmostEqual(audit[("g", 1)]["reward"], 0.5)
        self.assertNotIn("delivered_total", audit[("g", 1)]["sources"])
        self.assertAlmostEqual(audit[("g", 2)]["sources"]["frontier"], 0.2)


@unittest.skipUnless(os.path.exists(ROM) and os.path.exists(STATE), "ROM/save-state not available")
class TestGameAndGoals(unittest.TestCase):
    """Scripted walk from the true start: stairs, then Mom's scene, judged by
    the same predicates the LLM is judged by."""

    @classmethod
    def setUpClass(cls):
        from tools.llm_agent.game import Game

        cls.game = Game("configs/stages/open_world.json")

    @classmethod
    def tearDownClass(cls):
        cls.game.close()

    def test_warps_decode_from_rom(self):
        self.assertEqual(self.game.warps.warps(24, 7), [(7, 0, 24, 6)])
        self.assertIn((6, 3, 24, 5), self.game.warps.warps(24, 4))  # Elm's lab door

    def test_ladder_predicates_along_scripted_path(self):
        from tools.llm_agent.goals import GOALS

        g = self.game
        self.assertFalse(GOALS["stairs"].done(g, g.vars()))
        for b in ["right"] * 4 + ["up"] * 3:
            g.press(b)
        self.assertTrue(GOALS["stairs"].done(g, g.vars()))
        self.assertFalse(g.status()["text_open"])
        for b in ["down"] * 3:
            g.press(b)
        self.assertTrue(g.status()["text_open"])  # Mom's scene auto-starts
        for _ in range(80):
            if GOALS["mom"].done(g, g.vars()):
                break
            g.press("a")
        self.assertTrue(GOALS["mom"].done(g, g.vars()))
        # Standing on the doormat: the engine reads the off-map cell below
        # as a wall; the state must say down is the way out.
        for _ in range(8):
            if g.status()["y"] == 7:
                break
            g.press("down")
        for _ in range(8):
            if g.status()["x"] <= 7:
                break
            g.press("left")
        s = g.status()
        self.assertEqual(s["on_exit"]["to"], "NEW_BARK_TOWN", s)
        self.assertEqual(s["on_exit"]["leave"], ["down"])
        self.assertIn("EXIT", s["around"]["down"])
        self.assertTrue(s["local_map"])


if __name__ == "__main__":
    unittest.main()
