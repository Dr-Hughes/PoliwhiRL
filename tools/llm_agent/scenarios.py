# -*- coding: utf-8 -*-
"""Scripted regression scenarios for the training env, no LLM needed.

A trajectory is a fixed button list played from a start state (the env is
deterministic; see replay.py). While it plays, every step goes through:

  - the reward verifier (verify.py): a must-pay event that was not paid, or a
    milestone payout with no event, fails the scenario;
  - invariants on the observation and reward state (INVARIANTS).

Probes then branch off the trajectory: the first step a probe's ``when``
predicate holds, a fresh env replays to that step, presses the probe's
buttons and checks the result. That is how "START closes the naming
keyboard" or "down moves the YES/NO cursor" stay true after a change.

    uv run python -m tools.llm_agent.scenarios          # run all, exit 1 on failure

Add a trajectory by cutting a recorded run: ``python -m tools.llm_agent.replay
<run> --buttons --until N > tools/llm_agent/scenarios/<name>.json``.
Trajectories record buttons only, so a framework change that alters the
game's response (frame timing, the action set) can make them drift; a drift
shows up as a failed probe or a changed final position, and the fix is to
re-cut the trajectory from a fresh LLM run.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from PoliwhiRL.environment.gym_env import RAM_FEATURE_INDEX  # noqa: E402
from tools.llm_agent.game import Game  # noqa: E402
from tools.llm_agent.verify import RewardVerifier  # noqa: E402

HERE = Path(__file__).resolve().parent / "scenarios"
CONFIG = "configs/stages/open_world.json"


@dataclass(frozen=True)
class Probe:
    name: str
    when: Callable  # (game, status) -> bool, checked after each step
    press: tuple
    expect: Callable  # (game, status after) -> str | None (failure message)


@dataclass(frozen=True)
class Trajectory:
    name: str
    buttons_file: str
    state: str | None = None  # None = the config's start state
    end_map: str | None = None  # where the recording ended
    end_flag: int | None = None  # a story flag that must be set at the end
    probes: tuple = field(default_factory=tuple)


def _text(s) -> str:
    return " / ".join(s.get("text") or [])


PROBES = (
    Probe(
        "yes_no_cursor_moves_down",
        when=lambda g, s: "▶YES" in _text(s) and " NO" in f" {_text(s)}",
        press=("down",),
        expect=lambda g, s: (
            None if "▶NO" in _text(s) else f"down did not move the cursor to NO: {_text(s)!r}"
        ),
    ),
    # From the nickname question, so it does not depend on the recorded
    # player answering YES: A opens the keyboard, START jumps to END, A closes.
    Probe(
        "nickname_yes_opens_keyboard",
        when=lambda g, s: "▶YES" in _text(s) and "received?" in _text(s),
        press=("a",),
        expect=lambda g, s: (
            None if s["naming"] is not None else f"no naming keyboard after YES: {_text(s)!r}"
        ),
    ),
    Probe(
        "start_then_a_closes_naming_keyboard",
        when=lambda g, s: "▶YES" in _text(s) and "received?" in _text(s),
        press=("a", "start", "a"),
        expect=lambda g, s: (
            None if s["naming"] is None else f"keyboard still open, name {s['naming']!r}"
        ),
    ),
    Probe(
        "start_opens_menu_while_walking",
        when=lambda g, s: s["mode"] == "walk" and s["party"] >= 1,
        press=("start",),
        expect=lambda g, s: (
            None
            if s["mode"] == "menu" and "POKéMON" in _text(s)
            else f"no start menu: mode {s['mode']}, text {_text(s)!r}"
        ),
    ),
    # Settle stepping once ended a step while the game still saw the last
    # button held, so the next press was dropped (START, B left the menu open).
    Probe(
        "start_then_b_closes_menu",
        when=lambda g, s: s["mode"] == "walk" and s["party"] >= 1,
        press=("start", "b"),
        expect=lambda g, s: (
            None if s["mode"] == "walk" else f"menu still open after B: {_text(s)!r}"
        ),
    ),
)

TRAJECTORIES = (
    # settle_run_1 (2026-10-28, settle stepping): bedroom -> Mom -> Elm's lab
    # (starter, nickname NO) -> Route 29, one honest episode.
    Trajectory(
        "new_bark_to_route29", "new_bark_to_route29.json", end_map="ROUTE_29", probes=PROBES
    ),
    # The same run continued (settle_run_1 to step 410, the resume's reset
    # no-op, then settle_run_1_routes) as one honest episode: Cherrygrove,
    # Mr. Pokémon, the rival battle, back to Elm (flag 31).
    Trajectory("bedroom_to_flag31", "bedroom_to_flag31.json", end_flag=31),
)


def _rival_feature_zero(g, s, sample):
    v = float(sample["ram"][RAM_FEATURE_INDEX["rival_cherrygrove"]])
    return None if v == 0.0 else f"rival_cherrygrove observation is {v}"


def _no_scratch_event_flags(g, s, sample):
    bits = sorted(b for b in g.env.reward_calculator._event_flags_fired if b < 8)
    return None if not bits else f"map-reload scratch flags fired as events: {bits}"


# (game, status, sample) -> failure message or None, checked every step.
INVARIANTS = (_rival_feature_zero, _no_scratch_event_flags)


def _buttons(t: Trajectory) -> list[str]:
    return json.loads((HERE / t.buttons_file).read_text())


def _new_game(t: Trajectory) -> Game:
    return Game(CONFIG, state_path=t.state)


def run_trajectory(t: Trajectory) -> list[str]:
    """Play one trajectory with the verifier, invariants and probes; returns
    failure messages (empty = pass)."""
    buttons = _buttons(t)
    g = _new_game(t)
    rv = RewardVerifier()
    failures, fired = [], {}
    pending = list(t.probes)
    try:
        for step, b in enumerate(buttons, start=1):
            sample = g.press(b)
            s = g.status()
            rv.step(step, t.name, 1, b, g.vars(), g, sample["sources"])
            for inv in INVARIANTS:
                msg = inv(g, s, sample)
                if msg and inv.__name__ not in fired:
                    fired[inv.__name__] = step
                    failures.append(f"step {step}: {inv.__name__}: {msg}")
            for p in list(pending):
                if p.when(g, s):
                    fired[p.name] = step
                    pending.remove(p)
        rv.flush()
        end = g.status()["map"]
        if t.end_flag is not None and not g.flag(t.end_flag):
            failures.append(f"trajectory drifted: flag {t.end_flag} not set at the end")
    finally:
        g.close()
    if t.end_map and end != t.end_map:
        failures.append(f"trajectory drifted: ended on {end}, recorded {t.end_map}")
    for ev in rv.events:
        if ev.verdict == "MISSED":
            failures.append(f"step {ev.step}: verifier: {ev.kind} {ev.detail} was not paid")
    for u in rv.unexplained:
        failures.append(f"step {u['step']}: verifier: unexplained {u['source']} {u['value']:+.2f}")
    for p in pending:
        failures.append(f"probe {p.name}: its situation never came up")
    for p in t.probes:
        if p.name in fired:
            failures += [
                f"probe {p.name} (step {fired[p.name]}): {m}"
                for m in [_run_probe(t, buttons[: fired[p.name]], p)]
                if m
            ]
    return failures


def _run_probe(t: Trajectory, prefix: list[str], p: Probe) -> str | None:
    g = _new_game(t)
    try:
        for b in prefix:
            g.press(b)
        for b in p.press:
            g.press(b)
        return p.expect(g, g.status())
    finally:
        g.close()


def run_all() -> dict[str, list[str]]:
    return {t.name: run_trajectory(t) for t in TRAJECTORIES}


def main():
    results = run_all()
    bad = 0
    for name, failures in results.items():
        print(f"{'FAIL' if failures else 'ok  '} {name}")
        for f in failures:
            print(f"     {f}")
        bad += bool(failures)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
