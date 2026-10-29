# -*- coding: utf-8 -*-
"""tools/llm_agent — the LLM game-playing harness.

Pure tests (no model server): decision parsing, prompt state formatting and
the reward audit. The emulator-backed checks (warp decode, goal predicates
along a scripted path) skip cleanly without the ROM.
"""
import json
import os
import unittest

from tools.llm_agent.agent import (
    STALL_WARN,
    StallCounter,
    _sig,
    describe,
    format_state,
    press_event,
    press_line,
    progress_facts,
    stall_facts,
    talk_facts,
    talk_note,
)
from tools.llm_agent.game import CHARMAP, decode_menu, looks
from tools.llm_agent.llm import MAX_BUTTONS, parse_decision
from tools.llm_agent.agent import _llm_opts
from tools.llm_agent.report import reward_audit
from tools.llm_agent.agent import Memory
from tools.llm_agent.notebook import MAX_WORDS, Notebook
from tools.llm_agent.verify import RewardVerifier

ROM = "./emu_files/Pokemon - Crystal Version.gbc"
STATE = "./emu_files/states/start.state"
BATTLE_STATE = "./Manual Investigation States/in_wild_battle_route_13.state"
# gemma's run 2026-10-28: New Bark Town, one step below Elm's lab, carrying
# the Mystery Egg (the aide is in the lab, Elm at the back).
LAB_STATE = "./Manual Investigation States/new_bark_below_elms_lab_with_egg.state"


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
        many = ["up"] * (MAX_BUTTONS + 2)
        d, _ = parse_decision(json.dumps({"buttons": ["select"] + many}))
        self.assertEqual(d["buttons"], many[:MAX_BUTTONS])

    def test_thinking_follows_the_screen_mode(self):
        self.assertEqual(_llm_opts("walk", None, True, False), {"thinking": False})
        self.assertEqual(_llm_opts("battle", None, True, False), {"thinking": True})
        self.assertEqual(_llm_opts("menu", None, True, False), {"thinking": True})
        self.assertEqual(_llm_opts("text", None, True, False), {"thinking": True})
        self.assertEqual(_llm_opts("naming", None, True, False), {"thinking": True})

    def test_fallback_thinking_wins_over_the_mode_gate(self):
        # A fallback turn is an escalation: its setting must not collide
        # with the per-mode one (a duplicate ``thinking=`` keyword).
        opts = _llm_opts("walk", "backup-model", False, True)
        self.assertEqual(opts, {"model": "backup-model", "thinking": False})
        self.assertEqual(_llm_opts("walk", "backup-model", True, True)["thinking"], True)

    def test_no_thinking_is_a_ceiling(self):
        # --no-thinking must win even where the mode would ask for thinking.
        self.assertEqual(_llm_opts("battle", None, True, False, False), {"thinking": False})
        self.assertEqual(_llm_opts("menu", None, True, False, False), {"thinking": False})
        self.assertEqual(_llm_opts("battle", None, True, False, True), {"thinking": True})

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
        d, _ = parse_decision(
            '{"buttons": ["a"], "remember": "Mr. Pokemon is north", "recall": " egg "}'
        )
        self.assertEqual(d["remember"], ["Mr. Pokemon is north"])
        self.assertEqual(d["recall"], "egg")
        d, _ = parse_decision('{"buttons": ["a"]}')
        self.assertEqual((d["remember"], d["recall"]), ([], ""))


class TestNotebook(unittest.TestCase):
    def test_add_dedupes_and_search_ranks(self):
        nb = Notebook()
        nb.add(
            ["Mr. Pokemon lives north of Cherrygrove on Route 30"],
            goal="mr_pokemon",
            map_name="CHERRYGROVE_CITY",
            turn=1,
        )
        nb.add(
            ["mr pokemon lives north of cherrygrove on route 30!"], goal="g", map_name="m", turn=2
        )
        nb.add(["Press B to decline the nickname", ""], goal="starter", map_name="ELMS_LAB", turn=3)
        self.assertEqual(len(nb.facts), 2)
        hits = nb.search("where does Mr Pokemon live", turn=4)
        # the restatement replaced the original
        self.assertIn("route 30!", hits[0]["text"])
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

    def test_restatements_replace_the_old_fact(self):
        # ornith's real notebook, run_20261029_014818: one wrong belief, restated
        nb = Notebook()
        for i, text in enumerate(
            [
                "Verified CHERRYGROVE CITY route origin is (27,15): from there UP clears the "
                "L-shaped structure edge then RIGHT twice enters open grass where I can travel west.",
                "Verified CHERRYGROVE CITY route origin is (27,15): from there UP clears L-shaped "
                "structure edge then RIGHT twice enters open grass for westward travel off left map edge.",
                "Verified CHERRYGROVE CITY origin is (27,15): from there UP clears the structure edge "
                "then RIGHT twice enters open grass for westward travel off left map edge.",
            ]
        ):
            nb.add([text], goal="cherrygrove", map_name="ROUTE_29", turn=i)
        self.assertEqual(len(nb.facts), 1)
        self.assertFalse(nb.facts[0]["text"].lower().startswith("verified"))
        self.assertTrue(nb.facts[0]["text"].startswith("CHERRYGROVE CITY origin"))

    def test_a_failed_attempt_leaves_its_facts_unconfirmed(self):
        m = Memory()
        nb = m.notebook
        nb.add(
            ["Elm gave you a Pokemon in his lab"],
            goal="starter",
            map_name="ELMS_LAB",
            turn=1,
            attempt=1,
        )
        nb.add(
            ["The west edge of Route 29 is reached along row 15"],
            goal="cherrygrove",
            map_name="ROUTE_29",
            turn=2,
            attempt=1,
        )
        self.assertEqual(nb.settle("starter", 1, True), 1)
        self.assertEqual(nb.settle("cherrygrove", 1, False), 1)
        self.assertEqual(nb.settle("cherrygrove", 1, False), 0)  # already settled
        text = m.format("Travel west along Route 29 to Cherrygrove City", "ROUTE_29")
        self.assertIn(
            "[UNCONFIRMED: written during an attempt that failed; may be wrong] "
            "The west edge of Route 29",
            text,
        )
        # a fact from a goal that was reached is not replaced by a restatement
        nb.add(["Elm gave you a Pokemon in his lab!"], goal="g", map_name="m", turn=3, attempt=2)
        self.assertEqual([f["status"] for f in nb.facts], ["worked", "unconfirmed"])

    def test_unconfirmed_facts_are_dropped_first(self):
        nb = Notebook()
        nb.add(["a wrong idea about the gym"], goal="g", map_name="m", turn=0, attempt=1)
        nb.settle("g", 1, False)
        # 35 x 14 + 6 + 10 words: one over the cap, so one fact must go
        for i in range(35):
            nb.add([" ".join(f"w{i}x{j}" for j in range(14))], goal="h", map_name="m", turn=i)
        nb.add([" ".join(f"z{j}" for j in range(10))], goal="h", map_name="m", turn=99)
        self.assertEqual(len(nb.facts), 36)
        self.assertFalse(any("wrong idea" in f["text"] for f in nb.facts))
        self.assertTrue(any(f["text"].startswith("w0x0 ") for f in nb.facts))

    def test_memory_shows_recall_once_and_auto_recalls_when_stuck(self):
        m = Memory()
        m.notebook.add(
            ["Elm's lab is the building west of your house"],
            goal="starter",
            map_name="NEW_BARK_TOWN",
            turn=1,
        )
        m.recall_query, m.recall_hits = "lab", m.notebook.search("lab")
        text = m.format("Get your first Pokemon from Elm", "NEW_BARK_TOWN")
        self.assertIn('Results of your recall "lab"', text)
        self.assertNotIn("Results of your recall", m.format("x", "y"))
        stuck = m.format("Get your first Pokemon from Elm", "NEW_BARK_TOWN", stall=6)
        self.assertIn("You are stuck; facts you saved", stuck)


CURSOR = chr(0x25B6)
_TILES = {v: k for k, v in CHARMAP.items() if len(v) == 1}
_TILES["|"] = 0x7C  # the box border


def _tilemap(*rows):
    """A tilemap from (row index, text) pairs."""
    grid = [[0x7F] * 20 for _ in range(18)]
    for idx, line in rows:
        cells = [_TILES.get(ch, 0x7F) for ch in line]
        grid[idx][: len(cells)] = cells
    return grid


def _status(**kw):
    """A minimal status dict, as ``Game.status()`` returns it."""
    s = {
        "mode": "walk",
        "map": "ROUTE_29",
        "map_id": (24, 3),
        "x": 5,
        "y": 5,
        "facing": "up",
        "text_open": False,
        "in_battle": False,
        "party": 1,
        "party_mons": [{"level": 5, "hp": 18, "max_hp": 20}],
        "pokeballs": 5,
        "money": 3000,
        "badges": 0,
        "pokedex_owned": 1,
        "around": {"up": "open", "down": "open", "left": "open", "right": "open"},
        "exits": [],
        "people": [],
        "text": [],
        "menu": False,
        "menu_info": None,
        "naming": None,
    }
    s.update(kw)
    return s


class TestSignature(unittest.TestCase):
    def test_counters_a_press_can_change_are_in_the_signature(self):
        base = _status()
        for field, value in (
            ("pokeballs", 4),
            ("money", 2999),
            ("badges", 1),
            ("pokedex_owned", 2),
            ("party", 2),
            ("naming", "A"),
        ):
            self.assertNotEqual(_sig(base), _sig(_status(**{field: value})), field)
        self.assertNotEqual(_sig(base), _sig(_status(text=["HELLO"])))
        self.assertNotEqual(
            _sig(base), _sig(_status(menu_info={"label": "YES"})), "the menu cursor"
        )

    def test_battle_state_is_in_the_signature(self):
        base = _status(
            mode="battle",
            in_battle=True,
            text_open=True,
            battle={
                "type": "wild",
                "enemy": {"species": "RATTATA", "level": 3, "hp": 8, "max_hp": 14},
                "you": {"species": "CYNDAQUIL", "level": 5, "hp": 18, "max_hp": 20},
                "move_pp": [34, 30, 0, 0],
            },
        )
        other = dict(base, battle=dict(base["battle"], move_pp=[33, 30, 0, 0]))
        self.assertNotEqual(_sig(base), _sig(other))
        other = dict(
            base,
            battle=dict(
                base["battle"], enemy={"species": "RATTATA", "level": 3, "hp": 7, "max_hp": 14}
            ),
        )
        self.assertNotEqual(_sig(base), _sig(other))

    def test_derived_fields_do_not_move_the_signature(self):
        base = _status()
        self.assertEqual(_sig(base), _sig(_status(local_map=["."], people=[{"x": 1}])))
        self.assertEqual(_sig(base), _sig(base))


class TestPressEvent(unittest.TestCase):
    def test_a_press_that_changes_nothing_says_so(self):
        before = _status()
        after = _status()
        self.assertIsNone(press_event(before, after, "a"))
        self.assertIn("NOTHING HAPPENED", press_line(before, after, "a"))

    def test_a_press_that_changes_something_does_not_say_nothing_happened(self):
        before = _status()
        after = _status(text=["HELLO"])
        self.assertEqual(press_event(before, after, "a"), "a: ok")
        self.assertNotIn("NOTHING HAPPENED", press_line(before, after, "a"))

    def test_movement_and_blocking_keep_their_wording(self):
        before = _status()
        after = _status(x=6)
        self.assertEqual(press_event(before, after, "right"), "right: moved you to (6,5)")
        blocked = _status(around={"up": "person"})
        self.assertEqual(press_event(blocked, blocked, "up"), "up: BLOCKED (person), now facing up")

    def test_entering_a_map_and_a_battle_ending_keep_their_wording(self):
        before = _status()
        after = _status(map_id=(24, 4), map="NEW_BARK_TOWN", x=7, y=8)
        self.assertEqual(press_event(before, after, "up"), "up: entered NEW_BARK_TOWN at (7,8)")
        battle = _status(
            mode="battle",
            in_battle=True,
            text_open=True,
            menu_info={"label": "RUN"},
            battle={
                "type": "wild",
                "enemy": {"species": "RATTATA", "level": 3, "hp": 8, "max_hp": 14},
                "you": {"species": "CYNDAQUIL", "level": 5, "hp": 18, "max_hp": 20},
                "move_pp": [34, 30, 0, 0],
            },
        )
        over = _status(
            mode="walk", in_battle=False, text_open=False, menu_info=None, text=["Got away safely!"]
        )
        self.assertEqual(press_event(battle, over, "a"), "a: the battle ended")
        started = _status(
            mode="battle",
            in_battle=True,
            text_open=True,
            menu_info={"label": "FIGHT"},
            battle={
                "type": "wild",
                "enemy": {"species": "RATTATA", "level": 3, "hp": 8, "max_hp": 14},
                "you": {"species": "CYNDAQUIL", "level": 5, "hp": 18, "max_hp": 20},
                "move_pp": [34, 30, 0, 0],
            },
        )
        self.assertEqual(press_event(_status(), started, "a"), "a: the battle started")
        menu = _status(
            mode="menu",
            text_open=True,
            menu=True,
            menu_info={"label": "PACK"},
            text=[CURSOR + "PACK"],
        )
        closed = _status(mode="walk", text_open=False, menu=False, menu_info=None, text=[])
        self.assertEqual(press_event(menu, closed, "b"), "b: closed the menu")

    def test_the_open_menu_names_the_option_under_the_cursor(self):
        s = _status(
            mode="menu",
            text_open=True,
            menu=True,
            text=["YES", "NO"],
            menu_info={"label": "NO", "options": ["YES", "NO"], "cursor": 1},
        )
        self.assertIn("the cursor is on NO", format_state(s))

    def test_a_menu_cursor_move_is_reported(self):
        before = _status(
            mode="menu", text_open=True, text=["\u25b6YES"], menu=True, menu_info={"label": "YES"}
        )
        after = _status(
            mode="menu", text_open=True, text=["NO"], menu=True, menu_info={"label": "NO"}
        )
        self.assertEqual(press_event(before, after, "down"), "down: the cursor is now on NO")
        self.assertNotIn("NOTHING HAPPENED", press_line(before, after, "down"))


class TestDecodeMenu(unittest.TestCase):
    def test_battle_menu_cursor(self):
        rows = _tilemap((14, "| \u25b6FIGHT PKMN |"), (16, "|  PACK  RUN |"))
        mi = decode_menu(rows, in_battle=True)
        self.assertEqual(mi["label"], "FIGHT")
        self.assertEqual(mi["options"], ["FIGHT", "PKMN", "PACK", "RUN"])
        self.assertEqual(mi["grid"]["RUN"], (1, 1))
        self.assertNotIn("rows", mi)

    def test_cursor_on_pkmn_is_not_read_as_the_word_before_it(self):
        rows = _tilemap((14, "| FIGHT\u25b6PKMN |"), (16, "|  PACK  RUN |"))
        self.assertEqual(decode_menu(rows, in_battle=True)["label"], "PKMN")

    def test_cursor_on_run(self):
        rows = _tilemap((14, "|  FIGHT PKMN |"), (16, "| PACK \u25b6RUN |"))
        self.assertEqual(decode_menu(rows, in_battle=True)["label"], "RUN")

    def test_vertical_menu_has_no_grid(self):
        rows = _tilemap((12, "|\u25b6YES   NO |"))
        mi = decode_menu(rows)
        self.assertEqual(mi["label"], "YES")
        self.assertEqual(mi["options"], ["YES", "NO"])
        self.assertEqual(mi["cursor"], 0)
        self.assertIsNone(mi["grid"])

    def test_no_cursor_means_no_menu(self):
        self.assertIsNone(decode_menu(_tilemap((12, "|PLAIN TEXT|"))))


class TestStallCounter(unittest.TestCase):
    def test_counts_a_situation_and_forgets_nothing(self):
        c = StallCounter()
        # the count is how many times the situation has come up, this one
        # included
        self.assertEqual(c.observe(("a",)), 1)
        self.assertEqual(c.observe(("a",)), 2)
        self.assertEqual(c.observe(("b",)), 1)
        self.assertEqual(c.observe(("a",)), 3)

    def test_names_the_button_sequence_used_in_the_situation(self):
        c = StallCounter()
        c.observe(("battle",), ("down", "down", "a"))
        c.observe(("battle",), ("down", "down", "a"))
        c.observe(("battle",), ("b",))
        # the count is how often the sequence was used in the whole attempt
        self.assertEqual(c.top_sequence(("battle",)), (("down", "down", "a"), 2))
        self.assertEqual(c.top_sequence(("other",)), ((), 0))

    def test_a_new_situation_starts_a_new_count(self):
        c = StallCounter()
        for _ in range(5):
            c.observe(("walk", 1, 2))
        self.assertEqual(c.observe(("walk", 3, 4)), 1)

    def test_the_count_does_not_decay_when_the_situation_changes(self):
        # a loop can alternate between two situations, so the count of each
        # keeps rising
        c = StallCounter()
        for _ in range(3):
            c.observe(("menu",), ("down", "down", "a"))
            c.observe(("items",), ("down", "down", "a"))
        self.assertEqual(c.observe(("menu",)), 4)


class TestStallFacts(unittest.TestCase):
    def test_no_facts_below_the_threshold(self):
        c = StallCounter()
        s = _status()
        self.assertEqual(stall_facts(s, STALL_WARN - 1, c, STALL_WARN), [])

    def test_the_warning_names_the_situation_and_the_buttons(self):
        c = StallCounter()
        s = _status(map="ROUTE_29", x=36, y=9)
        for _ in range(6):
            c.observe(_sig(s), ("down", "down", "a"))
        facts = stall_facts(s, 6, c, STALL_WARN)
        self.assertIn("you have been in this exact situation 6 times", facts[0])
        self.assertIn("ROUTE_29 (36,9)", facts[0])
        self.assertIn('You have pressed "down down a" 6 times', facts[1])

    def test_the_sequence_count_is_the_whole_attempt(self):
        c = StallCounter()
        here = _status(map="ROUTE_29", x=36, y=9)
        there = _status(map="ROUTE_29", x=37, y=9)
        for _ in range(4):
            c.observe(_sig(here), ("down", "down", "a"))
            c.observe(_sig(there), ("down", "down", "a"))
        facts = stall_facts(here, 4, c, warn=4)
        self.assertIn('You have pressed "down down a" 8 times', facts[1])


class TestProgressFacts(unittest.TestCase):
    def test_reports_the_steps_the_turns_and_the_elapsed_time(self):
        facts = progress_facts(120, 3000, 1, 45, 90.0)
        self.assertIn("Goal steps used: 120/3000 (attempt 1).", facts[0])
        self.assertIn("Turns used: 45", facts[1])
        self.assertIn("1.5 min", facts[1])


class TestDescribe(unittest.TestCase):
    def test_names_the_battle_and_its_opponent(self):
        s = _status(
            mode="battle",
            in_battle=True,
            text_open=True,
            map="ROUTE_29",
            x=36,
            y=9,
            menu_info={"label": "PACK"},
            battle={
                "type": "wild",
                "enemy": {"species": "SENTRET", "level": 2, "hp": 6, "max_hp": 13},
                "you": {"species": "CYNDAQUIL", "level": 7, "hp": 13, "max_hp": 24},
                "move_pp": [29, 30, 20, 0],
            },
        )
        out = describe(s)
        self.assertIn("ROUTE_29 (36,9)", out)
        self.assertIn("wild battle vs SENTRET Lv2", out)
        self.assertIn("the cursor is on PACK", out)

    def test_names_a_menu_and_a_scene(self):
        self.assertIn("a menu is open", describe(_status(mode="menu", menu=True, text_open=True)))
        self.assertIn("a scene is running", describe(_status(mode="text", text_open=True)))


class TestPeople(unittest.TestCase):
    def test_sprites_are_named_by_how_they_look(self):
        self.assertEqual(looks(0x10), "PROF.ELM")
        self.assertEqual(looks(0x3C), "scientist")
        self.assertTrue(looks(0x54).startswith("Poké Ball"))
        self.assertEqual(looks(0xF3), "person")

    def _ctx(self, said_twice=True):
        key = ("ELMS_LAB", 2, 9, "scientist")
        return {
            "talks": {key: 3},
            "talk_said": {key: ["someone stole a POKéMON."]},
            "talk_same": {key: 3 if said_twice else 1},
        }

    def test_someone_who_repeats_themself_has_nothing_more(self):
        facts = talk_facts(self._ctx(), "ELMS_LAB")
        self.assertIn("scientist at (x=2, y=9) 3 times", facts[0])
        self.assertIn("said the same thing each time", facts[0])
        self.assertIn("it may be someone else", facts[0])
        self.assertNotIn("do what it told you", facts[0])
        self.assertEqual(talk_facts(self._ctx(), "NEW_BARK_TOWN"), [])

    def test_the_people_list_notes_who_you_talked_to(self):
        ctx = self._ctx()
        self.assertIn("same thing each time", talk_note(ctx, "ELMS_LAB", {"x": 2, "y": 9}))
        self.assertEqual(talk_note(ctx, "ELMS_LAB", {"x": 5, "y": 2}), "")
        self.assertEqual(
            talk_note(self._ctx(False), "ELMS_LAB", {"x": 2, "y": 9}), "you talked to it 3 times"
        )

    def _lab(self, **kw):
        return _status(
            map="ELMS_LAB",
            map_id=(24, 5),
            x=4,
            y=11,
            people=[
                {"x": 2, "y": 9, "dx": -2, "dy": -2, "what": "scientist", "loaded": True},
                {"x": 5, "y": 2, "dx": 1, "dy": -9, "what": "PROF.ELM", "loaded": False},
            ],
            on_exit={"to": "NEW_BARK_TOWN", "leave": ["down"]},
            **kw,
        )

    def test_out_of_view_people_are_listed(self):
        text = format_state(self._lab())
        self.assertIn("PROF.ELM at (x=5, y=2): 1 right, 9 up from you (out of view", text)
        self.assertIn("scientist at (x=2, y=9): 2 left, 2 up from you\n", text)

    def test_the_doormat_is_a_fact_not_an_order(self):
        text = format_state(self._lab(arrived_from="NEW_BARK_TOWN"))
        self.assertIn(
            "You are standing on the exit to NEW_BARK_TOWN: pressing DOWN walks "
            "off the edge and leaves this map.",
            text,
        )
        self.assertIn("You have just come in from NEW_BARK_TOWN", text)
        self.assertNotIn("press DOWN to", text)

    def test_routes_to_people_and_the_door_you_stand_on(self):
        text = format_state(
            self._lab(
                routes=[
                    {
                        "to": "NEW_BARK_TOWN",
                        "via": "door at (4,11)",
                        "steps": 0,
                        "first": "",
                        "path": "",
                    },
                    {
                        "to": "PROF.ELM",
                        "via": "at (5,2)",
                        "steps": 8,
                        "first": "up",
                        "path": "up 8",
                        "face": "up",
                        "person": True,
                    },
                    {"to": "nurse", "via": "at (5,1)", "steps": None, "path": "", "person": True},
                ]
            )
        )
        self.assertIn("to NEW_BARK_TOWN (door at (4,11)): you are standing on it", text)
        self.assertIn(
            "to talk to PROF.ELM at (5,2): FIRST MOVE UP; 8 steps: up 8, "
            "then face up and press a",
            text,
        )
        self.assertIn(
            "to talk to the nurse at (5,1): no walking route from here (it may be "
            "behind a counter",
            text,
        )
        self.assertNotIn("FIRST MOVE YOU", text)

    def test_far_people_routes_are_dropped_unless_the_goal_names_them(self):
        far = {
            "to": "teacher",
            "via": "at (15,11)",
            "steps": 83,
            "first": "left",
            "path": "left 83",
            "face": "up",
            "person": True,
        }
        text = format_state(self._lab(routes=[far]))
        self.assertNotIn("to talk to the teacher", text)
        text = format_state(self._lab(routes=[dict(far, goal=True)]))
        self.assertIn("to talk to the teacher at (15,11)", text)


@unittest.skipUnless(
    os.path.exists(ROM) and os.path.exists(LAB_STATE), "ROM/save-state not available"
)
class TestElmsLabPeople(unittest.TestCase):
    """From the door of Elm's lab the engine has loaded only the aide; Elm
    stands at the back, out of view. The state must still name him and give
    the way to him."""

    @classmethod
    def setUpClass(cls):
        from tools.llm_agent.game import Game

        cls.game = Game("configs/stages/open_world.json", state_path=LAB_STATE)
        cls.game.press("up")
        cls.status = cls.game.status()
        cls.routes = cls.game.routes()

    @classmethod
    def tearDownClass(cls):
        cls.game.close()

    def test_elm_is_listed_though_not_loaded(self):
        s = self.status
        self.assertEqual((s["map"], s["x"], s["y"]), ("ELMS_LAB", 4, 11))
        people = {p["what"]: p for p in s["people"]}
        self.assertEqual((people["PROF.ELM"]["x"], people["PROF.ELM"]["y"]), (5, 2))
        self.assertFalse(people["PROF.ELM"]["loaded"])
        self.assertEqual((people["scientist"]["x"], people["scientist"]["y"]), (2, 9))
        self.assertTrue(people["scientist"]["loaded"])
        # Your ball and the one the rival stole are hidden by their event
        # flags; the third is still on the table, and the officer is here.
        balls = [(p["x"], p["y"]) for p in s["people"] if p["what"].startswith("Poké Ball")]
        self.assertEqual(balls, [(8, 3)])
        self.assertEqual((people["police officer"]["x"], people["police officer"]["y"]), (5, 3))

    def test_there_is_a_route_to_elm(self):
        r = next(r for r in self.routes if r.get("person") and r["to"] == "PROF.ELM")
        self.assertIsNotNone(r["steps"])
        self.assertNotEqual(r["first"], "down")


class TestFormatState(unittest.TestCase):
    def test_relative_offsets(self):
        s = {
            "map": "PLAYERS_HOUSE_2F",
            "map_id": (24, 7),
            "x": 3,
            "y": 3,
            "facing": "down",
            "text_open": False,
            "in_battle": False,
            "party": 0,
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
            "map": "ROUTE_29",
            "map_id": (24, 3),
            "x": 5,
            "y": 5,
            "facing": "up",
            "mode": "walk",
            "text_open": False,
            "in_battle": False,
            "party": 1,
            "party_mons": [{"level": 5, "hp": 18, "max_hp": 20}],
            "pokeballs": 5,
            "money": 3000,
            "badges": 0,
            "pokedex_owned": 1,
            "around": {"up": "open", "down": "open", "left": "open", "right": "open"},
            "exits": [],
            "people": [],
            "connections": {
                "west": {"to": "CHERRYGROVE_CITY", "cross": "y=6-9"},
                "north": {"to": "ROUTE_46", "cross": ""},
            },
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
        text = format_state(
            self._route29(
                routes=[
                    {
                        "to": "CHERRYGROVE_CITY",
                        "via": "west edge",
                        "steps": 12,
                        "path": "left 4, up (hop the ledge), left 6, then left off the edge",
                    },
                    {"to": "NEW_BARK_TOWN", "via": "east edge", "steps": None, "path": ""},
                ]
            )
        )
        self.assertIn(
            "to CHERRYGROVE_CITY (west edge): FIRST MOVE LEFT; 12 steps: left 4, up (hop the ledge)",
            text,
        )
        self.assertIn("to NEW_BARK_TOWN (east edge): no walking route from here", text)

    def test_battle_hides_overworld_and_names_enemy(self):
        mon = {"species": "RATTATA", "level": 3, "hp": 8, "max_hp": 14}
        you = {"species": "CYNDAQUIL", "level": 5, "hp": 18, "max_hp": 20}
        text = format_state(
            self._route29(
                mode="battle",
                text_open=True,
                in_battle=True,
                battle={"type": "wild", "enemy": mon, "you": you, "move_pp": [34, 30, 0, 0]},
            )
        )
        self.assertIn("wild BATTLE is running", text)
        self.assertIn("enemy RATTATA Lv3 HP 8/14; your CYNDAQUIL Lv5 HP 18/20", text)
        self.assertNotIn("LOCAL MAP", text)
        self.assertNotIn("Next to you", text)
        self.assertNotIn("CHERRYGROVE_CITY", text)

    def _battle_mi(self, label):
        grid = {"FIGHT": (0, 0), "PKMN": (0, 1), "PACK": (1, 0), "RUN": (1, 1)}
        return {
            "label": label,
            "options": list(grid),
            "cursor": list(grid).index(label),
            "grid": grid,
        }

    def _battle_text(self, label):
        return format_state(
            self._route29(
                mode="battle",
                text_open=True,
                in_battle=True,
                menu_info=self._battle_mi(label),
                battle={
                    "type": "wild",
                    "enemy": {"species": "RATTATA", "level": 3, "hp": 8, "max_hp": 14},
                    "you": {"species": "CYNDAQUIL", "level": 5, "hp": 18, "max_hp": 20},
                    "move_pp": [34, 30, 0, 0],
                },
            )
        )

    def test_battle_menu_names_the_cursor_and_the_options(self):
        text = self._battle_text("PACK")
        self.assertIn("BATTLE MENU: the cursor is on PACK", text)
        self.assertIn("PACK (bottom-left)", text)
        self.assertIn("RUN (bottom-right)", text)
        self.assertIn("to choose RUN press right, then a", text)
        self.assertIn("left/right change column", text)
        self.assertIn("does not wrap", text)

    def test_battle_menu_paths_are_row_first(self):
        # The cursor moves down/up between rows and left/right between
        # columns, and does not wrap, so a path changes the row first.
        self.assertIn("to choose RUN press down, then right, then a", self._battle_text("FIGHT"))
        self.assertIn("to choose PACK press down, then left, then a", self._battle_text("PKMN"))
        self.assertIn("to choose FIGHT press up, then left, then a", self._battle_text("RUN"))
        self.assertIn("to choose PKMN press up, then right, then a", self._battle_text("PACK"))

    def test_battle_menu_is_drawn_as_the_screen_draws_it(self):
        text = self._battle_text("PACK")
        # the cursor's cell is marked, the others are not
        self.assertIn("\u25b6 PACK (bottom-left)", text)
        self.assertNotIn("\u25b6 FIGHT", text)
        self.assertIn("PKMN (top-right)", text)
        self.assertIn("RUN (bottom-right)", text)


class _FakeGame:
    def party(self):
        return [{"level": 5, "hp": 20, "max_hp": 20}]


def _vars(**kw):
    v = {
        "map_bank": 24,
        "map_num": 4,
        "story_flags": [0] * 256,
        "battle_type": 0,
        "enemy_hp": 0,
        "pokedex_owned": 1,
        "pokedex_seen": 1,
        "party_info": (1, 5, 20, 0),
        "key_items_count": 0,
        "script_active": False,
    }
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
        rv.step(
            4,
            "g",
            1,
            "a",
            _vars(map_num=5, story_flags=flags, pokedex_owned=2),
            g,
            {"pokedex": 5.0},
        )
        # a big key_item payout with no key-item event -> unexplained
        rv.step(
            5,
            "g",
            1,
            "a",
            _vars(map_num=5, story_flags=flags, pokedex_owned=2),
            g,
            {"key_item": 3.0},
        )
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
            {
                "goal": "g",
                "attempt": 1,
                "reward": 1.0,
                "sources": {"new_map": 1.0, "delivered_total": 1.0},
            },
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
        self.assertIn("exit to NEW_BARK_TOWN", s["around"]["down"])
        self.assertTrue(s["local_map"])


@unittest.skipUnless(
    os.path.exists(ROM) and os.path.exists(BATTLE_STATE), "ROM/save-state not available"
)
class TestBattleMenu(unittest.TestCase):
    """The battle menu's cursor: a 2x2 grid that does not wrap, decoded
    from the tilemap (the screen text alone does not say where it is)."""

    def setUp(self):
        from tools.llm_agent.game import Game

        self.game = Game("configs/stages/open_world.json", state_path=BATTLE_STATE)

    def tearDown(self):
        self.game.close()

    def test_cursor_moves_in_a_2x2_grid_without_wrapping(self):
        g = self.game
        mi = g.status()["menu_info"]
        self.assertEqual(mi["label"], "FIGHT")
        self.assertEqual(mi["grid"]["RUN"], (1, 1))
        # the battle menu is not flagged as a menu: the battle block is
        self.assertFalse(g.status()["menu"])
        g.press("right")
        self.assertEqual(g.status()["menu_info"]["label"], "PKMN")
        g.press("right")
        self.assertEqual(g.status()["menu_info"]["label"], "PKMN")  # no wrap
        g.press("down")
        self.assertEqual(g.status()["menu_info"]["label"], "RUN")


if __name__ == "__main__":
    unittest.main()
