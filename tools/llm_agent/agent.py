# -*- coding: utf-8 -*-
"""The agent loop: perceive -> prompt -> decide -> execute (with early stop)
-> judge, with a three-tier memory (recent turns, model-edited notes,
harness facts) rebuilt into one fresh request every turn.

Every env step also feeds the reward verifier, the goal video and the live
frame; every goal attempt ends with a markdown write-up."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, deque
from pathlib import Path

from tqdm import tqdm

from tools.llm_agent import docs
from tools.llm_agent.game import _DIRS, BUTTONS, LEGEND, Game
from tools.llm_agent.media import HIGHLIGHTS, Media, compose
from tools.llm_agent.notebook import MAX_WORDS, Notebook
from tools.llm_agent.verify import RewardVerifier

SYSTEM_PROMPT = """\
You are playing Pokémon Crystal on a Game Boy. You control the player character.

HOW TO READ THE SCREEN
- The screen is a grid of 10 columns x 9 rows of cells (faint red lines).
- While you are walking, YOU are always the character in the cell outlined in
  blue (column 5, row 5). The world scrolls around you. In battles, menus and
  the naming keyboard there is no grid or box.
- Coordinates: x grows to the right, y grows DOWNWARD. An offset like
  "2 right, 3 up" is relative to you.

CONTROLS (each button press is one action)
- up / down / left / right: walk one cell that way. One press turns AND walks;
  you never need to turn first. Which way you face only matters for a.
  You cannot walk into walls, furniture or people.
- a: talk to / interact with the person or thing in the cell you are FACING;
  advance a text box; choose the menu option the ▶ arrow points at.
- b: cancel / back out of a menu / say No.
- start: opens the menu (POKéDEX, POKéMON, PACK, ...) while walking; closes
  it again. On the naming keyboard it jumps the cursor to END.

HOW THE GAME WORKS
- To talk to someone: stand in a cell next to them, face them, press a.
- Stairs, doors and exits: walk onto the exit cell. Doormats at the bottom of
  a room lead outside: step onto the mat, then press DOWN once more (walk off
  the bottom of the room). Walking left/right along a mat does nothing.
- Towns and routes join at their edges: keep walking off the edge of the map
  in the direction of the connected area to enter it.
- While a text box, scene or battle is running you cannot walk. Press a to
  advance text. For a Yes/No or the start menu, move the ▶ arrow with
  up/down first if needed, then press a. The BATTLE menu is a 2x2 grid
  (FIGHT PKMN over PACK RUN): there up/down change row and left/right
  change column, so RUN sits to the RIGHT of PACK. The cursor stops at
  the edge of the grid: pressing right in the right-hand column does
  nothing.
- Battles: the menu is FIGHT / PKMN (top), PACK / RUN (bottom). FIGHT then a
  move attacks. To catch a wild Pokémon: weaken it, then PACK, choose a Poké
  Ball, and use it. RUN escapes wild battles (not trainer battles).
- Tall grass hides wild Pokémon. Pokémon Centers (in towns) heal your party
  for free: talk to the nurse at the counter.
- Some scenes start on their own when you walk into a spot — that is normal.
- Say No (move the arrow to NO, or press b) to optional questions that do not
  help your goal, such as "Give a nickname?". If a naming keyboard opens
  anyway, press start then a.

EACH TURN you get the screen, a GAME STATE block read directly from the game's
memory (exact — trust it over your eyes for positions, exits, walls and what
is next to you), a LOCAL MAP of the cells around you, facts from your
NOTEBOOK, your scratchpad notes, and your recent turns with their results.
Reply ONLY by calling the `act` tool. Press up to 4 buttons per turn; use fewer
when you are unsure or when a text box or menu is open.

YOUR NOTEBOOK keeps facts across goals; text boxes scroll away and your recent
turns are forgotten. Use `remember` to save what people and signs tell you
(directions, where someone lives, what to do next) and lessons that worked or
failed. Use `recall` with keywords when you are stuck or need something you
learned earlier; the results are shown on your next turn.

YOUR GOAL: {goal}
"""


# The situation count at which the harness warns the model, and the count at
# which it raises the temperature (and engages the fallback model).
STALL_WARN = 6
STALL_HOT = 12


def _rel(dx, dy):
    parts = []
    if dx:
        parts.append(f"{abs(dx)} {'right' if dx > 0 else 'left'}")
    if dy:
        parts.append(f"{abs(dy)} {'down' if dy > 0 else 'up'}")
    return ", ".join(parts) or "your cell"


def format_state(s: dict) -> str:
    if s.get("battle"):
        busy = f"YES — a {s['battle']['type']} BATTLE is running; use the battle menu"
    elif s["text_open"]:
        busy = "YES — a scene/text/menu is running; you cannot walk (press a to advance)"
    else:
        busy = "no — you can walk"
    lines = [
        f"Location: {s['map']}  position (x={s['x']}, y={s['y']})  facing {s['facing']}",
        "Scene/text/battle running: " + busy,
    ]
    mi = s.get("menu_info") or {}
    if s.get("text"):
        lines.append("TEXT ON SCREEN (exact, read from memory): " + " / ".join(s["text"]))
    if s.get("menu"):
        on = mi.get("label")
        lines.append(
            "A MENU IS OPEN: the ▶ arrow marks the option a will choose (up/down "
            "moves it)"
            + (f"; the cursor is on {on}" if on else "")
            + ". YES and NO answer the literal question on screen — e.g. YES "
            'to "Give a nickname?" opens a keyboard to type a name; NO skips it.'
        )
    if s.get("naming") is not None:
        lines.append(
            f'NAMING KEYBOARD IS OPEN (name so far: "{s["naming"]}"). a types the letter '
            "under the cursor, b deletes one. To finish, press start (jumps the cursor to END) "
            "then a."
        )
    b = s.get("battle")
    if b:
        e, y = b["enemy"], b["you"]
        if not y["max_hp"]:
            # Before both are sent out the battle structs still hold the
            # previous battle (or zeros).
            lines.append(f"Battle ({b['type']}): starting — Pokémon not sent out yet")
        else:
            lines.append(
                f"Battle ({b['type']}): enemy {e['species']} Lv{e['level']} HP "
                f"{e['hp']}/{e['max_hp']}; your {y['species']} Lv{y['level']} HP "
                f"{y['hp']}/{y['max_hp']}, move PP {b['move_pp']}"
            )
        if mi.get("grid"):
            lines.extend(battle_menu_lines(mi))
    mons = s.get("party_mons")
    if mons:
        lines.append(
            "Party: "
            + "; ".join(
                f"#{i + 1} Lv{m['level']} HP {m['hp']}/{m['max_hp']}" for i, m in enumerate(mons)
            )
        )
    else:
        lines.append(f"Pokémon in party: {s['party']}")
    if "pokeballs" in s:
        lines.append(
            f"Poké Balls: {s['pokeballs']}  Money: {s['money']}  Badges: {s['badges']}  "
            f"Pokédex owned: {s['pokedex_owned']}"
        )
    if s.get("on_exit"):
        e = s["on_exit"]
        how = (
            f"press {' or '.join(e['leave']).upper()} to walk off the edge and leave"
            if e["leave"]
            else "it should take you there; if not, step off and back on"
        )
        lines.append(f"YOU ARE STANDING ON THE EXIT to {e['to']}: {how}.")
    if s.get("mode", "walk") != "walk":
        # Off the overworld the neighbours, doors and local map describe a
        # world you cannot move in right now; they only distract.
        return "\n".join(lines)
    if s.get("routes"):
        # First, before the local view: a straight line on the local map is
        # often a dead end (Route 29's ledge strip), the route is not.
        lines.append(
            "WALKING ROUTES from here (exact shortest paths from the map data; "
            "ledges are one-way, people block). To go somewhere, follow its "
            "route, starting with its first move, even if it looks like the wrong way:"
        )
        for r in s["routes"]:
            if r["steps"] is None:
                lines.append(f"  - to {r['to']} ({r['via']}): no walking route from here")
            else:
                first = r.get("first") or r["path"].split(",")[0].split(" ")[0]
                lines.append(
                    f"  - to {r['to']} ({r['via']}): FIRST MOVE {first.upper()}; "
                    f"{r['steps']} steps: {r['path']}"
                )
    lines.append("Next to you: " + ", ".join(f"{d}={what}" for d, what in s["around"].items()))
    if s["exits"]:
        lines.append("Doors/stairs on this map:")
        for e in s["exits"]:
            lines.append(
                f"  - to {e['to']} at (x={e['x']}, y={e['y']}): {_rel(e['dx'], e['dy'])} from you"
            )
    if s.get("connections"):
        lines.append(
            "Map edges: "
            + "; ".join(
                f"{d} edge leads to {c['to']} "
                + (
                    f"(walkable at {c['cross']})"
                    if c["cross"]
                    else "(no walkable crossing: water, a fence, or reached through a gate building)"
                )
                for d, c in s["connections"].items()
            )
        )
    if s["people"]:
        lines.append("People/objects nearby:")
        for p in s["people"]:
            lines.append(
                f"  - {p.get('what', 'person')} at (x={p['x']}, y={p['y']}): "
                f"{_rel(p['dx'], p['dy'])} from you"
            )
    if s.get("local_map"):
        lines.append("LOCAL MAP (up is top; you are @):")
        lines += ["    " + r for r in s["local_map"]]
        lines.append("  legend: " + LEGEND)
    return "\n".join(lines)


def _sig(s: dict) -> tuple:
    """A hashable signature of everything a button press can change.

    Two states with the same signature look the same to the player, so a
    press that leaves the signature unchanged did nothing. The clock is not
    in here (it is not in ``status()`` and it moves on its own), and neither
    are the position-derived fields (connections, local_map, exits, people,
    around, on_exit): they follow the position, which is in the signature,
    and NPCs move on their own.
    """
    b = s.get("battle") or {}

    def mon(m):
        return tuple(m.get(k) for k in ("species", "level", "hp", "max_hp"))

    return (
        s.get("mode"),
        tuple(s.get("map_id") or ()),
        s.get("x"),
        s.get("y"),
        s.get("facing"),
        tuple(s.get("text") or ()),
        (s.get("menu_info") or {}).get("label"),
        s.get("naming"),
        s.get("pokeballs"),
        s.get("money"),
        s.get("badges"),
        s.get("pokedex_owned"),
        s.get("party"),
        tuple((m.get("level"), m.get("hp"), m.get("max_hp")) for m in (s.get("party_mons") or ())),
        b.get("type"),
        mon(b.get("enemy") or {}),
        mon(b.get("you") or {}),
        tuple(b.get("move_pp") or ()),
    )


def press_event(before: dict, after: dict, button: str) -> str | None:
    """What one button press did, as the line the turn history shows, or
    None when the press changed nothing.

    Pure: it reads only the two status dicts.
    """
    busy = bool(before["text_open"] or after["text_open"] or before["in_battle"])
    # A scene walks the player on its own; don't credit the button.
    by = button if button in _DIRS or not busy else "(the scene)"
    if before["map_id"] != after["map_id"]:
        return f"{by}: entered {after['map']} at ({after['x']},{after['y']})"
    if (before["map_id"], before["x"], before["y"]) != (
        after["map_id"],
        after["x"],
        after["y"],
    ):
        return f"{by}: moved you to ({after['x']},{after['y']})"
    if before["in_battle"] != after["in_battle"]:
        # The battle menu goes a press before the battle itself ends, so the
        # transition is read from the flag, not from the menu.
        return f"{button}: the battle {'started' if after['in_battle'] else 'ended'}"
    was = (before.get("menu_info") or {}).get("label")
    now = (after.get("menu_info") or {}).get("label")
    if was and now and was != now:
        return f"{button}: the cursor is now on {now}"
    if button in _DIRS and not busy:
        what = before["around"].get(button, "blocked")
        return f"{button}: BLOCKED ({what}), now facing {after['facing']}"
    if was and not now:
        return f"{button}: closed the menu"
    if _sig(before) != _sig(after):
        return f"{button}: ok"
    return None


def press_line(before: dict, after: dict, button: str) -> str:
    """The turn-history line for one button press. A press that changed
    nothing says so, so the model can tell a dead button from a live one."""
    event = press_event(before, after, button)
    if event is None:
        return f"{button}: NOTHING HAPPENED (the state did not change)"
    return event


def progress_facts(
    steps: int, budget: int, attempt: int, turns: int, elapsed_s: float
) -> list[str]:
    """The HARNESS lines that say where the attempt is."""
    return [
        f"Goal steps used: {steps}/{budget} (attempt {attempt}).",
        f"Turns used: {turns}. Elapsed: {elapsed_s / 60:.1f} min.",
    ]


def stall_facts(s: dict, stall: int, counter: StallCounter, warn: int = STALL_WARN) -> list[str]:
    """The HARNESS lines for a model that keeps coming back to the same
    situation: what the situation is, and the buttons that led there."""
    if stall < warn:
        return []
    out = [
        f"WARNING: you have been in this exact situation {stall} times in this "
        f"attempt ({describe(s)}). Doing the same thing again will give you the "
        "same result - try something clearly different."
    ]
    seq, n = counter.top_sequence(_sig(s))
    if seq:
        out.append(
            f'You have pressed "{" ".join(seq)}" {n} times in this attempt. '
            "That sequence is not working - try something else."
        )
    return out


def describe(s: dict) -> str:
    """A short description of where the model is and what it is looking at."""
    where = f"{s.get('map')} ({s.get('x')},{s.get('y')})"
    b = s.get("battle") or {}
    if b:
        enemy = b.get("enemy") or {}
        out = (
            f"{where}, {b.get('type', 'a')} battle vs "
            f"{enemy.get('species', '?')} Lv{enemy.get('level', '?')}"
        )
    elif s.get("mode") == "menu":
        out = f"{where}, a menu is open"
    elif s.get("mode") == "naming":
        out = f"{where}, the naming keyboard is open"
    elif s.get("mode") == "text":
        out = f"{where}, a scene is running"
    else:
        out = where
    label = (s.get("menu_info") or {}).get("label")
    if label:
        out += f", the cursor is on {label}"
    return out


class StallCounter:
    """How many times each situation has come up in this attempt.

    A situation is the state signature (see ``_sig``): the same place, the
    same screen, the same battle. It is counted in every mode, so a model
    looping inside a battle or a text scene is caught too; the old counter
    only moved while the player could walk, and froze while a scene or a
    battle was running, so it never advanced through a 169-turn battle.

    The count does not decay when a different situation is visited: a loop
    can alternate between two situations (a battle menu and the item
    submenu of the same battle), and a counter that reset on every change
    would never see it.

    The counter also remembers which button sequences were used to get into
    each situation, so the harness can name the one that is not working.
    """

    def __init__(self):
        self.counts: dict = {}
        self.sequences: dict = {}
        self.all_sequences: Counter = Counter()

    def observe(self, sig: tuple, seq: tuple = ()) -> int:
        """Record a turn in this situation. Returns how many times the
        situation has come up, counting this one."""
        self.counts[sig] = self.counts.get(sig, 0) + 1
        if seq:
            seq = tuple(seq)
            self.sequences.setdefault(sig, Counter())[seq] += 1
            self.all_sequences[seq] += 1
        return self.counts[sig]

    def top_sequence(self, sig: tuple) -> tuple:
        """The button sequence that most often led into this situation, and
        how many times it was used in this attempt: ``((), 0)`` if none."""
        counts = self.sequences.get(sig)
        if not counts:
            return (), 0
        seq = counts.most_common(1)[0][0]
        return seq, self.all_sequences.get(seq, 0)


_GRID_POS = {
    (0, 0): "top-left",
    (0, 1): "top-right",
    (1, 0): "bottom-left",
    (1, 1): "bottom-right",
}


def battle_menu_lines(mi: dict) -> list[str]:
    """The state-block lines for the battle menu: where the cursor is, the
    four options laid out as the screen lays them out, and which presses
    reach each option.

    The cursor moves up/down between rows and left/right between columns and
    it does not wrap, so the path to an option changes the row first and the
    column second: a row move keeps the column and a column move keeps the
    row.
    """
    grid = mi.get("grid") or {}
    label = mi.get("label")
    here = grid.get(label)
    out = [
        f"BATTLE MENU: the cursor is on {label}. up/down change row, "
        "left/right change column; the cursor does not wrap, so pressing "
        "right in the right-hand column does nothing."
    ]
    for row in (0, 1):
        cells = []
        for col in (0, 1):
            name = next((n for n, p in grid.items() if p == (row, col)), "")
            mark = "\u25b6" if here == (row, col) else " "
            cells.append(f"{mark} {name} ({_GRID_POS[(row, col)]})")
        out.append("  " + "   ".join(cells))
    if here is not None:
        steps = []
        for name, (row, col) in grid.items():
            if name == label:
                continue
            path = []
            if row > here[0]:
                path += ["down"] * (row - here[0])
            elif row < here[0]:
                path += ["up"] * (here[0] - row)
            if col > here[1]:
                path += ["right"] * (col - here[1])
            elif col < here[1]:
                path += ["left"] * (here[1] - col)
            presses = ", then ".join(path + ["a"]) if path else "a"
            steps.append(f"to choose {name} press {presses}")
        if steps:
            out.append("  " + "; ".join(steps))
    return out


class Memory:
    def __init__(self, recent: int = 8, notebook: Notebook | None = None):
        self.turns = deque(maxlen=recent)
        self.notes = ""
        self.lessons: list[str] = []  # harness-written, survive attempt restarts
        self.notebook = notebook or Notebook()
        self.recall_query = ""
        self.recall_hits: list[dict] = []  # requested last turn, shown once
        self.dialogue = deque(maxlen=14)  # exact text lines read, oldest first

    def read_text(self, lines: list[str]):
        """Log on-screen text lines. Text types out a few letters per step, so
        a line that extends the last logged one replaces it. Menu options
        (the ▶ line and bare YES/NO) are not dialogue."""
        for line in lines:
            if (
                not line
                or "▶" in line
                or line.strip() in ("YES", "NO")
                or any(line == d for d in list(self.dialogue)[-6:])
            ):
                continue
            if self.dialogue and line.startswith(self.dialogue[-1]):
                self.dialogue[-1] = line
            elif self.dialogue and self.dialogue[-1].startswith(line):
                continue
            else:
                self.dialogue.append(line)

    def _notebook_section(
        self, goal_text: str, map_name: str, stall: int, turn: int | None
    ) -> list[str]:
        nb = self.notebook
        if not nb.facts:
            return ["YOUR NOTEBOOK: (empty — use `remember` to save useful facts)"]
        out = [f"YOUR NOTEBOOK ({len(nb.facts)} facts, {nb.words()}/{MAX_WORDS} words)."]
        shown = set()
        if self.recall_query:
            hits = self.recall_hits
            out.append(f'  Results of your recall "{self.recall_query}":')
            out += [f"  - {f['text']}" for f in hits] or ["  - (nothing matched)"]
            shown |= {f["id"] for f in hits}
            self.recall_query, self.recall_hits = "", []
        if stall:
            # Stuck: search with the goal and place so earlier lessons resurface.
            auto = [
                f
                for f in nb.search(f"{goal_text} {map_name}", k=5, turn=turn)
                if f["id"] not in shown
            ]
            if auto:
                out.append("  You are stuck; facts you saved that may help:")
                out += [f"  - {f['text']}" for f in auto]
                shown |= {f["id"] for f in auto}
        relevant = [
            f for f in nb.search(f"{goal_text} {map_name}", k=4, bump=False) if f["id"] not in shown
        ]
        recent = [f for f in nb.facts[-3:] if f["id"] not in shown and f not in relevant]
        if relevant or recent:
            out.append("  Relevant / recent facts:")
            out += [f"  - {f['text']}" for f in relevant + recent]
        return out

    def format(
        self,
        goal_text: str = "",
        map_name: str = "",
        stall: int = 0,
        turn: int | None = None,
        on_screen: list[str] | None = None,
    ) -> str:
        out = []
        if self.lessons:
            out.append(
                "LESSONS FROM EARLIER ATTEMPTS:\n"
                + "\n".join(f"- {lesson}" for lesson in self.lessons)
            )
        out += self._notebook_section(goal_text, map_name, stall, turn)
        shown = set(on_screen or [])
        past = [d for d in self.dialogue if d not in shown]
        if past:
            out.append(
                "TEXT YOU READ EARLIER (exact, oldest first; no longer on screen — "
                "act on it):\n" + "\n".join(f"  {d}" for d in past)
            )
        out.append("YOUR SCRATCHPAD NOTES:\n" + (self.notes.strip() or "(empty)"))
        if self.turns:
            out.append("YOUR RECENT TURNS (oldest first):")
            for t in self.turns:
                out.append(
                    f"  turn {t['turn']} at {t['map']} ({t['x']},{t['y']}): plan was "
                    f'"{t["plan"][:120]}" -> pressed {" ".join(t["buttons"])} -> {t["result"]}'
                )
        return "\n".join(out)


LIVE_HTML = """<!doctype html><meta charset=utf-8><title>LLM agent live</title>
<style>body{background:#111;color:#ddd;font:14px system-ui;margin:16px}
img{image-rendering:pixelated;max-width:100%;height:auto}</style>
<div id=t></div><img id=i src="live.png">
<script>setInterval(()=>{const i=new Image();i.onload=()=>{document.getElementById('i').src=i.src;
document.getElementById('t').textContent=new Date().toLocaleTimeString()};
i.src='live.png?'+Date.now()},500)</script>
"""


class Agent:
    def __init__(
        self,
        game: Game,
        llm,
        recorder,
        *,
        temperature=0.3,
        stall_warn=STALL_WARN,
        stall_hot=STALL_HOT,
        hot_temperature=0.9,
        max_attempts=3,
        two_frames=False,
        fallback_model=None,
        fallback_thinking=True,
        fallback_min_turns=6,
        verbose=False,
        video=True,
        ladder=(),
        autosave_every=900.0,
        max_steps=0,
    ):
        self.game = game
        self.llm = llm
        self.rec = recorder
        self.temperature = temperature
        self.stall_warn = stall_warn
        self.stall_hot = stall_hot
        self.hot_temperature = hot_temperature
        self.max_attempts = max_attempts
        self.two_frames = two_frames
        self.fallback_model = fallback_model
        self.fallback_thinking = fallback_thinking
        self.fallback_min_turns = fallback_min_turns
        self.autosave_every = autosave_every
        self.max_steps = max_steps  # whole-run env-step cap, 0 = none
        self._last_autosave = time.time()
        self.verbose = verbose
        self.ladder = list(ladder)
        self.turn_no = 0
        self.step_no = 0
        self.reached: list[str] = []
        self.run_dir = Path(recorder.dir)
        self.media = Media(self.run_dir) if video else None
        self._clips: list[Path] = []
        self.verifier = RewardVerifier(on_event=self._on_event_closed)
        self._events_fh = open(self.run_dir / "events.jsonl", "a")
        (self.run_dir / "live.html").write_text(LIVE_HTML)
        self.bar = None
        self._caption = []
        self.memory = None
        # Remote control (web.py): control.json is polled once per turn;
        # status.json is rewritten after every turn for the dashboard.
        self._control_path = self.run_dir / "control.json"
        self._saves_done = self._read_control().get("save", 0)
        self.status = {
            "pid": os.getpid(),
            "paused": False,
            "ladder": [g.key for g in self.ladder],
            "last_save": None,
            "max_steps": max_steps,
        }

    # -- logging -------------------------------------------------------------------
    def log(self, msg: str):
        tqdm.write(msg)

    def _on_event_closed(self, ev):
        self._events_fh.write(json.dumps(ev.row(), default=str) + "\n")
        self._events_fh.flush()
        flag = "" if ev.verdict == "paid" else f"  <-- {ev.verdict}"
        if ev.kind not in ("event_flag", "pokedex_seen") or ev.verdict == "MISSED":
            paid = ", ".join(f"{k} {v:+.2f}" for k, v in ev.paid.items()) or "nothing"
            self.log(f"     [event] step {ev.step} {ev.kind}: {ev.detail} | paid {paid}{flag}")

    # -- execution -----------------------------------------------------------------
    def _after_step(self, goal, ctx, b, sample, after):
        g = self.game
        if g.training_end is not None and not g.training_end.get("logged"):
            g.training_end.update(
                logged=True,
                goal=goal.key,
                attempt=ctx["attempt"],
                harness_step=self.step_no,
            )
            self.log(
                f"     [training] the training env would END the episode here: "
                f"cause={g.training_end['cause']} at env step {g.training_end['env_step']}"
            )
            self._events_fh.write(
                json.dumps({"kind": "training_episode_end", **g.training_end}) + "\n"
            )
            self._events_fh.flush()
        if self.memory is not None and after.get("text"):
            self.memory.read_text(after["text"])
        if ctx.get("talk_key") and after.get("text"):
            said = ctx.setdefault("talk_lines", [])
            for line in after["text"]:
                if "▶" in line or line in said:
                    continue
                if said and line.startswith(said[-1]):
                    said[-1] = line
                elif not (said and said[-1].startswith(line)):
                    said.append(line)
        v = g.vars()
        self.rec.step(
            {
                "step": self.step_no,
                "turn": self.turn_no,
                "goal": goal.key,
                "attempt": ctx["attempt"],
                "seeded": g.seeded,
                "button": b,
                "map": after["map"],
                "x": after["x"],
                "y": after["y"],
                "facing": after["facing"],
                "scene": after["text_open"],
                "battle": after["in_battle"],
                "party": after["party"],
                "reward": round(sample["reward"], 4),
                "sources": sample["sources"],
            },
            sample,
        )
        for src, val in sample["sources"].items():
            if src != "delivered_total":
                ctx["sources"][src] = ctx["sources"].get(src, 0.0) + val
        new = self.verifier.step(self.step_no, goal.key, ctx["attempt"], b, v, g, sample["sources"])
        frame = g.frame()
        src_txt = " ".join(
            f"{k}{v:+.1f}"
            for k, v in sample["sources"].items()
            if k != "delivered_total" and abs(v) >= 0.05
        )
        lines = [
            f"[{ctx['idx']}/{len(self.ladder)}] {goal.title}  a{ctx['attempt']}  "
            f"step {ctx['steps']}/{goal.budget}  turn {self.turn_no}",
            f"{after['map']} ({after['x']},{after['y']})  pressed {b}  r {sample['reward']:+.2f} {src_txt}",
            *self._caption,
        ]
        if self.media:
            self.media.frame(frame, lines)
            for ev in new:
                if ev.kind in HIGHLIGHTS:
                    self._clips.append(
                        self.media.clip(f"event_{self.step_no:06d}_{ev.kind}_{goal.key}")
                    )
        try:
            from PIL import Image

            Image.fromarray(compose(frame, lines)).save(self.run_dir / "live.tmp.png", "PNG")
            (self.run_dir / "live.tmp.png").replace(self.run_dir / "live.png")
        except OSError:
            pass
        if self.bar is not None:
            self.bar.update(1)
            self.bar.set_postfix_str(
                f"{after['map']} ({after['x']},{after['y']}) t{self.turn_no}",
                refresh=False,
            )
        return v

    # -- remote control ------------------------------------------------------------
    def _read_control(self) -> dict:
        try:
            return json.loads(self._control_path.read_text())
        except (OSError, ValueError):
            return {}

    def write_status(self, **fields):
        self.status.update(
            fields,
            reached=list(self.reached),
            turn=self.turn_no,
            step=self.step_no,
            updated=time.time(),
        )
        tmp = self.run_dir / "status.tmp.json"
        tmp.write_text(json.dumps(self.status, default=str))
        tmp.replace(self.run_dir / "status.json")

    def poll_control(self, goal, ctx, memory: Memory):
        """Between turns: take a requested save, and block while paused (a
        save requested during the pause is still taken)."""
        while True:
            c = self._read_control()
            if c.get("save", 0) > self._saves_done:
                self._saves_done = c["save"]
                self.autosave(goal, ctx, memory)
            paused = bool(c.get("paused"))
            if paused != self.status["paused"]:
                self.log(f"     [control] {'paused' if paused else 'resumed'}")
                self.write_status(paused=paused)
            if not paused:
                return
            time.sleep(0.5)

    def autosave(self, goal, ctx, memory: "Memory"):
        """Save-state plus everything needed to resume this run from here
        (``--resume <run_dir>`` for the latest, or ``--resume <the .json next
        to the state>`` for this one): goal, notes, notebook, ladder progress."""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        state = self.rec.state(f"autosave_{stamp}", self.game.snapshot())
        meta = {
            "time": stamp,
            "state": str(state),
            "goal": goal.key,
            "attempt": ctx["attempt"],
            "goal_steps": ctx["steps"],
            "turn": self.turn_no,
            "step": self.step_no,
            "reached": list(self.reached),
            "notes": memory.notes,
            "notebook": memory.notebook.export(),
            "seeded": self.game.seeded,
        }
        (self.run_dir / "autosave.json").write_text(json.dumps(meta, indent=2, default=str))
        state.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        self._last_autosave = time.time()
        self.write_status(last_save=stamp)
        self.log(f"     [autosave] {state.name} (goal {goal.key}, step {ctx['steps']})")

    def _execute(self, buttons, goal, ctx):
        """Press buttons in order; stop early when something important
        happens. Returns (pressed, result text)."""
        g = self.game
        pressed, events = [], []
        for i, b in enumerate(buttons):
            nxt = buttons[i + 1] if i + 1 < len(buttons) else None
            before = g.status()
            sample = g.press(b)
            self.step_no += 1
            ctx["steps"] += 1
            after = g.status()
            v = self._after_step(goal, ctx, b, sample, after)
            pressed.append(b)
            auto = getattr(g.env._stepper, "auto_frames", 0) if g.env._stepper else 0
            if auto:
                events.append(
                    f"{b}: the game then played a demonstration by itself ({auto} "
                    "frames) — someone else's actions, not yours; your buttons did "
                    "nothing during it and nothing in it happened to you"
                )
            if (
                b == "a"
                and after["text_open"]
                and not before["text_open"]
                and not after["in_battle"]
            ):
                fx, fy = before["x"], before["y"]
                ddx, ddy = {
                    "up": (0, -1),
                    "down": (0, 1),
                    "left": (-1, 0),
                    "right": (1, 0),
                }.get(before["facing"], (0, 0))
                who = next(
                    (p for p in before["people"] if (p["x"], p["y"]) == (fx + ddx, fy + ddy)),
                    None,
                )
                if who is not None:
                    key = (
                        before["map"],
                        who["x"],
                        who["y"],
                        who["what"].split(" (")[0],
                    )
                    ctx["talks"][key] = ctx["talks"].get(key, 0) + 1
                    ctx["talk_key"] = key
                    ctx["talk_lines"] = []
            if before["text_open"] and not after["text_open"] and ctx.get("talk_key"):
                ctx["talk_said"][ctx["talk_key"]] = ctx.get("talk_lines", [])[-4:]
                ctx["talk_key"] = None
            events.append(press_line(before, after, b))

            stop = None
            if goal.done(g, v):
                stop = "GOAL COMPLETE"
            elif before["map_id"] != after["map_id"]:
                stop = "map changed"
            elif after["in_battle"] != before["in_battle"]:
                stop = "battle started" if after["in_battle"] else "battle ended"
            elif after.get("menu") and not before.get("menu"):
                stop = "a menu/question appeared — read it before answering"
            elif after.get("naming") is not None and before.get("naming") is None:
                stop = "a naming keyboard opened"
            elif after["text_open"] and not before["text_open"]:
                stop = "a scene/text box started"
            elif before["text_open"] and not after["text_open"]:
                stop = "the scene/text ended, you can walk"
            elif after["party"] != before["party"]:
                stop = f"party size is now {after['party']}"
            elif events[-1].startswith(f"{b}: BLOCKED") and nxt != "a":
                # A blocked press still turns you to face the cell, so a queued
                # a is the talk/read/pick-up the turn was planned for.
                stop = "blocked"
            if stop:
                if len(pressed) < len(buttons):
                    events.append(
                        f"(stopped early: {stop}; skipped {' '.join(buttons[len(pressed) :])})"
                    )
                else:
                    events.append(f"({stop})")
                break
            if ctx["steps"] >= goal.budget:
                break
        return pressed, "; ".join(events)

    # -- one goal --------------------------------------------------------------------
    def run_goal(self, goal, memory: Memory):
        g = self.game
        self.memory = memory
        idx = self.ladder.index(goal) + 1 if goal in self.ladder else 0
        start_state = g.snapshot()
        start_path = self.rec.state(f"{idx:02d}_{goal.key}_start", start_state)
        memory.lessons.clear()
        for attempt in range(1, self.max_attempts + 1):
            if attempt > 1:
                g.restart_from(start_state)
                self.verifier.new_episode()
                memory.turns.clear()
                memory.dialogue.clear()
            name = f"{idx:02d}_{goal.key}_a{attempt}"
            ctx = {
                "attempt": attempt,
                "steps": 0,
                "idx": idx,
                "sources": {},
                "talks": {},
                "talk_said": {},
                "talk_key": None,
            }
            memory.recall_query, memory.recall_hits = "", []
            self.rec.begin_demo()
            if self.media:
                self.media.begin_goal(name)
            self._clips = []
            turn_rows = []
            t0 = time.time()
            llm_time, turns, errors, fallback_turns = 0.0, 0, 0, 0
            stall_counter, stall, prev_buttons = StallCounter(), 0, ()
            prev_img = None
            success = goal.done(g, g.vars())
            seeded = g.seeded
            self.log(
                f"\n=== [{idx}/{len(self.ladder)}] goal '{goal.key}' attempt "
                f"{attempt}/{self.max_attempts} ({'seeded' if seeded else 'honest'}) "
                f"budget {goal.budget} steps ==="
            )
            self.bar = tqdm(
                total=goal.budget,
                desc=f"[{idx}/{len(self.ladder)}] {goal.key} a{attempt}",
                unit="step",
                dynamic_ncols=False,
                ncols=110,
                mininterval=1.0,
                leave=True,
            )
            self.write_status(
                goal=goal.key,
                goal_title=goal.title,
                idx=idx,
                attempt=attempt,
                max_attempts=self.max_attempts,
                goal_steps=0,
                budget=goal.budget,
                seeded=seeded,
                sources={},
            )
            while not success and ctx["steps"] < goal.budget and not self.out_of_steps():
                self.poll_control(goal, ctx, memory)
                self.turn_no += 1
                turns += 1
                s = g.status()
                if s["mode"] == "walk":
                    s["routes"] = g.routes()
                # The situation, not the cell: a model looping in a battle or
                # in a scene is stuck too, and the old cell counter never
                # noticed. The buttons that led here are the ones it keeps
                # pressing, so they are the ones to name.
                busy = s["text_open"] or s["in_battle"]
                sig = _sig(s)
                stall = stall_counter.observe(sig, prev_buttons)

                facts = progress_facts(ctx["steps"], goal.budget, attempt, turns, time.time() - t0)
                # The loop only runs while the RAM check fails; say so, so a
                # model that thinks it is done keeps playing.
                detail = goal.check(g, g.vars()) if goal.check else ""
                facts.append(
                    "Goal check (read from game memory): NOT complete yet."
                    + (f" {detail}." if detail else "")
                )
                for (mp, tx, ty, what), n in ctx["talks"].items():
                    if n >= 2 and mp == s["map"]:
                        said = (
                            " ".join(ctx["talk_said"].get((mp, tx, ty, what), []))
                            or "(see text above)"
                        )
                        facts.append(
                            f"You have talked to the {what} at (x={tx}, y={ty}) {n} times this goal. "
                            f'Last time it said: "{said}". Talking again just repeats it — '
                            "do what it told you instead."
                        )
                facts.extend(stall_facts(s, stall, stall_counter, self.stall_warn))
                hot = stall >= self.stall_hot
                temperature = self.hot_temperature if hot else self.temperature
                decision_screen = bool(s.get("menu")) or s.get("naming") is not None
                if self.fallback_model and (hot or decision_screen):
                    # Sticky: each switch makes the server swap models (15-60 s
                    # measured), so once escalated stay for a few turns.
                    ctx["fallback_left"] = max(ctx.get("fallback_left", 0), self.fallback_min_turns)
                use_fallback = bool(self.fallback_model) and (
                    attempt > 1 or ctx.get("fallback_left", 0) > 0
                )
                if ctx.get("fallback_left", 0) > 0:
                    ctx["fallback_left"] -= 1

                user = "\n\n".join(
                    [
                        "GAME STATE (exact, from memory):\n" + format_state(s),
                        memory.format(
                            goal.text,
                            s["map"],
                            stall if stall >= self.stall_warn else 0,
                            self.turn_no,
                            on_screen=s.get("text"),
                        ),
                        "HARNESS:\n" + "\n".join(facts),
                        "The screen image"
                        + (
                            " (previous turn's screen first, then the current one)"
                            if self.two_frames and prev_img
                            else ""
                        )
                        + " is attached. Decide your next buttons.",
                    ]
                )
                system = SYSTEM_PROMPT.format(goal=goal.text)
                img = g.llm_image(grid=s["mode"] == "walk")
                images = [g.png_b64(img)]
                if self.two_frames and prev_img is not None:
                    images.insert(0, g.png_b64(prev_img))
                prev_img = img
                frame = g.frame()

                fb = {"model": self.fallback_model, "thinking": self.fallback_thinking}
                res = self.llm.decide(
                    system, user, images, temperature, **(fb if use_fallback else {})
                )
                if res["decision"] is None and self.fallback_model and not use_fallback:
                    llm_time += res["latency"]
                    res = self.llm.decide(system, user, images, temperature, **fb)
                    use_fallback = True
                fallback_turns += int(use_fallback)
                llm_time += res["latency"]
                d = res["decision"]
                if d is None:
                    errors += 1
                    self.log(f"  [turn {self.turn_no}] model error: {res['error']}")
                    d = {
                        "observation": "",
                        "progress": "",
                        "plan": "(invalid reply)",
                        "buttons": ["a"] if busy else [BUTTONS[turns % 4]],
                    }
                self._caption = [f"plan: {d.get('plan', '')}"]
                pressed, result = self._execute(d["buttons"], goal, ctx)
                prev_buttons = tuple(pressed)
                if d.get("notes"):
                    memory.notes = str(d["notes"])[:1200]
                stored = memory.notebook.add(
                    d.get("remember"),
                    goal=goal.key,
                    map_name=s["map"],
                    turn=self.turn_no,
                )
                for f in stored:
                    self.log(f"     [remember] {f['text']}")
                if d.get("recall"):
                    memory.recall_query = d["recall"][:120]
                    memory.recall_hits = memory.notebook.search(
                        memory.recall_query, k=5, turn=self.turn_no
                    )
                    self.log(
                        f'     [recall] "{memory.recall_query}" -> '
                        f"{len(memory.recall_hits)} facts"
                    )
                memory.turns.append(
                    {
                        "turn": self.turn_no,
                        "map": s["map"],
                        "x": s["x"],
                        "y": s["y"],
                        "plan": d.get("plan", ""),
                        "buttons": pressed,
                        "result": result,
                    }
                )
                row = {
                    "turn": self.turn_no,
                    "goal": goal.key,
                    "attempt": attempt,
                    "seeded": g.seeded,
                    "model": res.get("model"),
                    "temperature": temperature,
                    "stall": stall,
                    "state": s,
                    "user_prompt": user,
                    "decision": d,
                    "raw": res["raw"],
                    "reasoning": res.get("reasoning"),
                    "content": res.get("content"),
                    "error": res["error"],
                    "usage": res["usage"],
                    "latency": round(res["latency"], 2),
                    "pressed": pressed,
                    "result": result,
                    "remembered": [f["text"] for f in stored],
                    "recalled": [f["text"] for f in memory.recall_hits],
                }
                self.rec.turn(row, frame)
                turn_rows.append(row)
                if self.autosave_every and time.time() - self._last_autosave >= self.autosave_every:
                    self.autosave(goal, ctx, memory)
                self.log(
                    f"  [t{self.turn_no} {ctx['steps']:>4}/{goal.budget}] {s['map']} ({s['x']},{s['y']})"
                    f"{' [battle]' if s['in_battle'] else ' [scene]' if s['text_open'] else ''}"
                    f"{' [fallback]' if use_fallback else ''} {res['latency']:.1f}s\n"
                    f"     sees: {d.get('observation', '')[:150]}\n"
                    f"     plan: {d.get('plan', '')[:150]}\n"
                    f"     -> {result[:220]}"
                )
                if self.verbose and res.get("reasoning"):
                    self.log(f"     think: {res['reasoning'][:600]}")
                success = goal.done(g, g.vars())
                after = g.status()
                self.write_status(
                    goal_steps=ctx["steps"],
                    map=after["map"],
                    x=after["x"],
                    y=after["y"],
                    mode=after["mode"],
                    in_battle=after["in_battle"],
                    model=res.get("model"),
                    latency=round(res["latency"], 2),
                    stall=stall,
                    notes=memory.notes,
                    sources={k: round(v, 3) for k, v in ctx["sources"].items()},
                    last={
                        "observation": d.get("observation", ""),
                        "progress": d.get("progress", ""),
                        "plan": d.get("plan", ""),
                        "buttons": pressed,
                        "result": result,
                        "error": res["error"],
                    },
                )

            self.bar.close()
            self.bar = None
            self.verifier.flush()
            seeded_flag = seeded or g.seeded
            self.rec.end_demo(name, success, seeded_flag)
            video = None
            if self.media:
                video = self.media.end_goal(name + ("_SUCCESS" if success else "_FAILED"))
            done_path = (
                self.rec.state(f"{idx:02d}_{goal.key}_done", g.snapshot()) if success else None
            )
            row = {
                "goal": goal.key,
                "attempt": attempt,
                "success": success,
                "seeded": seeded_flag,
                "steps": ctx["steps"],
                "turns": turns,
                "model_errors": errors,
                "fallback_turns": fallback_turns,
                "wall_s": round(time.time() - t0, 1),
                "llm_s": round(llm_time, 1),
                "video": video,
                "clips": list(self._clips),
            }
            row["doc"] = docs.write_goal_doc(
                self.run_dir,
                idx,
                goal,
                row,
                turn_rows,
                ctx["sources"],
                self.verifier,
                video,
                self._clips,
                {"start": start_path, "done": done_path},
                notebook=memory.notebook,
            )
            self.rec.attempt(row)
            if success:
                self.reached.append(goal.key)
            self.write_status(
                last_attempt={k: row[k] for k in ("goal", "attempt", "success", "steps", "turns")}
            )
            docs.write_report(
                self.run_dir,
                self.rec.summary["meta"],
                self.rec.summary["attempts"],
                self.verifier,
                self.ladder,
                self.reached,
            )
            mark = "✔ GOAL HIT" if success else "✘ FAILED"
            self.log(
                f"=== {mark} [{idx}/{len(self.ladder)}] {goal.title}: {ctx['steps']} steps, "
                f"{turns} turns, {(time.time() - t0) / 60:.1f} min | video {video} | doc {row['doc']} ==="
            )
            if success:
                return True
            if self.out_of_steps():
                self.log(f"=== run step cap reached: {self.step_no}/{self.max_steps} env steps ===")
                return False
            st = g.status()
            memory.lessons.append(
                f"Attempt {attempt} at '{goal.key}' ran out of steps; it ended at "
                f"{st['map']} ({st['x']},{st['y']}). Your notes then: "
                f"{memory.notes.strip()[:300] or '(none)'}"
            )
        if goal.optional:
            # Carry on with the ladder from where this goal started.
            g.restart_from(start_state)
            self.verifier.new_episode()
            memory.turns.clear()
        return False

    def out_of_steps(self) -> bool:
        return bool(self.max_steps) and self.step_no >= self.max_steps

    def close(self):
        self.verifier.flush()
        if self.media:
            self.media.close()
        self._events_fh.close()
        docs.write_report(
            self.run_dir,
            self.rec.summary["meta"],
            self.rec.summary["attempts"],
            self.verifier,
            self.ladder,
            self.reached,
        )
