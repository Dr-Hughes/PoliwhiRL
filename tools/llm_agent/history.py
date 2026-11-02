# -*- coding: utf-8 -*-
"""What the model has done, written by the harness from game memory.

Small models cannot be trusted to keep their own history: asked to note what
happened, ornith wrote "Verified CHERRYGROVE origin is (27,15)" eight times
and copied its wrong plan forward from turn to turn. So the history the model
sees is the harness's, in two tiers:

- ``TurnLog``: the recent turns verbatim, one compact line each (where, what
  was pressed, what it did). It is shown in blocks of ``BLOCK`` turns that
  only ever grow at the end (25 to 49 lines), so the start of the prompt stays
  the same from turn to turn and a server that caches prompt prefixes does not
  re-read it.
- ``EpisodeSummary``: everything since the attempt began, compressed per map
  (turns spent, ground covered, exits used, who was talked to and what they
  said, battles) plus the situations the model keeps returning to. It is
  rebuilt every ``BLOCK`` turns for the same reason.

Neither tier contains anything the model wrote; its plans are shown
separately and only for the last few turns.
"""

from __future__ import annotations

import re

BLOCK = 25


def compact_result(result: str) -> str:
    """The turn's result line, shortened: a run of moves becomes the cell it
    ended on, "a: ok" becomes "ok", and the early-stop note keeps its reason
    but not the skipped buttons."""
    out: list[str] = []
    for ev in result.split("; "):
        ev = ev.strip()
        if not ev or ev.startswith("skipped "):
            continue
        m = re.match(r"^\w+: moved you to (\(\d+,\d+\))$", ev)
        if m:
            if out and out[-1].startswith("moved to "):
                out[-1] = f"moved to {m.group(1)}"
            else:
                out.append(f"moved to {m.group(1)}")
            continue
        m = re.match(r"^\(stopped early: ([^;)]*)", ev)
        if m:
            out.append(f"(stopped: {m.group(1)})")
            continue
        if re.match(r"^\w+: ok$", ev):
            ev = "ok"
            if out and out[-1] == "ok":
                continue
        out.append(ev)
    return "; ".join(out)


class TurnLog:
    """Every turn of the attempt, as compact lines."""

    def __init__(self, block: int = BLOCK):
        self.block = block
        self.lines: list[str] = []

    def add(self, turn: int, s: dict, buttons, result: str):
        mode = "" if s.get("mode", "walk") == "walk" else f" [{s['mode']}]"
        self.lines.append(
            f"t{turn} {s['map']} ({s['x']},{s['y']}){mode}: "
            f"{' '.join(buttons)} -> {compact_result(result)}"
        )

    def window(self) -> list[str]:
        """The lines shown: the previous full block and the current one, so
        between ``block`` and ``2 * block - 1`` lines once the log is long."""
        n = len(self.lines)
        start = max(0, (n // self.block - 1) * self.block)
        return self.lines[start:]

    def format(self) -> str:
        shown = self.window()
        if not shown:
            return ""
        skipped = len(self.lines) - len(shown)
        head = (
            "WHAT YOUR LAST TURNS DID (exact, written by the harness, oldest first"
            + (f"; {skipped} earlier turns are summarised above" if skipped else "")
            + "):"
        )
        return "\n".join([head] + [f"  {line}" for line in shown])


class EpisodeSummary:
    """The attempt so far, per map, from game memory."""

    def __init__(self, block: int = BLOCK):
        self.block = block
        self.maps: dict[str, dict] = {}  # in order of first visit
        self.turns = 0
        self._text = ""
        self._built_at = -block

    def _map(self, name: str) -> dict:
        return self.maps.setdefault(
            name,
            {"turns": 0, "cells": set(), "exits": {}, "talks": {}, "battles": 0, "whiteouts": 0},
        )

    def turn(self, s: dict):
        """One turn spent on the map in ``s``."""
        self.turns += 1
        self._map(s["map"])["turns"] += 1

    def step(self, before: dict, after: dict):
        """One button press."""
        m = self._map(before["map"])
        m["cells"].add((before["x"], before["y"]))
        if before["map_id"] != after["map_id"]:
            way = f"({before['x']},{before['y']}) -> {after['map']}"
            m["exits"][way] = m["exits"].get(way, 0) + 1
            self._map(after["map"])["cells"].add((after["x"], after["y"]))
        if after.get("in_battle") and not before.get("in_battle"):
            m["battles"] += 1
        if any("whited out" in line or "blacked out" in line for line in after.get("text") or ()):
            if not any(
                "whited out" in line or "blacked out" in line for line in before.get("text") or ()
            ):
                m["whiteouts"] += 1

    def talk(self, map_name: str, what: str, x: int, y: int, said: list[str]):
        talks = self._map(map_name)["talks"]
        key = (what, x, y)
        n, _ = talks.get(key, (0, []))
        talks[key] = (n + 1, said)

    def text(self, turn: int, loops=()) -> str:
        """The summary, rebuilt every ``block`` turns (``turn`` is the turn
        number within the attempt); between rebuilds the same text."""
        # Compare block indices, not the delta, so the summary is rebuilt on
        # the same boundary TurnLog.window() slides on: one prefix miss per
        # block, not two on different turns.
        if turn // self.block != self._built_at // self.block or not self._text:
            self._built_at = turn
            self._text = self._build(turn, loops)
        return self._text

    def _build(self, turn: int, loops) -> str:
        if not self.maps:
            return ""
        out = [
            f"WHAT YOU HAVE DONE THIS ATTEMPT (from game memory, up to turn {turn} "
            "of this attempt):"
        ]
        for name, m in self.maps.items():
            line = f"- {name}: {m['turns']} turns"
            if m["cells"]:
                xs = [x for x, _ in m["cells"]]
                ys = [y for _, y in m["cells"]]
                line += (
                    f", stood on {len(m['cells'])} different cells "
                    f"(x {min(xs)}-{max(xs)}, y {min(ys)}-{max(ys)})"
                )
            if m["battles"]:
                line += f", {m['battles']} battles"
            if m["whiteouts"]:
                line += f", BLACKED OUT {m['whiteouts']} times"
            out.append(line)
            if m["exits"]:
                out.append(
                    "    left it via: "
                    + "; ".join(
                        f"{way}" + (f" x{n}" if n > 1 else "") for way, n in m["exits"].items()
                    )
                )
            for (what, x, y), (n, said) in m["talks"].items():
                words = " ".join(said)[:160] or "(nothing)"
                out.append(
                    f"    talked to the {what} at ({x},{y})"
                    + (f" {n} times" if n > 1 else "")
                    + f': "{words}"'
                )
        if loops:
            out.append(
                "Situations you keep coming back to (doing the same thing there again "
                "will give the same result):"
            )
            out += [f"  - {where}: {n} times" for where, n in loops]
        return "\n".join(out)

    def one_line(self) -> str:
        """Where the attempt's turns went, for the failed-attempt lesson."""
        parts = []
        for name, m in sorted(self.maps.items(), key=lambda kv: -kv[1]["turns"])[:4]:
            p = f"{name} {m['turns']} turns"
            if m["cells"]:
                xs = [x for x, _ in m["cells"]]
                ys = [y for _, y in m["cells"]]
                p += f" (x {min(xs)}-{max(xs)}, y {min(ys)}-{max(ys)})"
            parts.append(p)
        return "; ".join(parts)


_GENERIC = frozenset("city town route house the prof lab gate s".split())


def _name_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("_", " ").replace(".", " "))
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words}


def goal_names(goal_text: str, to: str) -> bool:
    """Whether the goal text names the route destination ``to`` (a map name
    such as CHERRYGROVE_CITY or ROUTE_29, or a person such as PROF.ELM)."""
    want = _name_tokens(to) - _GENERIC
    if not want:
        return False
    return want <= _name_tokens(goal_text)


def route_named(routes: list[dict], following: str) -> dict | None:
    """The route the model says it is following, or None. Matches the
    destination as listed (CHERRYGROVE_CITY, PROF.ELM), ignoring case and
    punctuation; the nearest such route wins."""
    f = re.sub(r"[^a-z0-9]", "", (following or "").lower())
    if not f or f == "none":
        return None

    def match(r):
        n = re.sub(r"[^a-z0-9]", "", r["to"].lower())
        return n in f or (len(f) >= 3 and f in n)

    hits = [r for r in routes if r.get("steps") is not None and match(r)]
    return min(hits, key=lambda r: r["steps"]) if hits else None
