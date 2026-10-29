# -*- coding: utf-8 -*-
"""Drive the training env by hand from a save-state, one command at a time.

    uv run python -m tools.llm_agent.drive STATE [BUTTON ...] [--to MAP] [--out NAME]

Loads STATE, presses each BUTTON (``up``, ``a``, ``left*7`` = seven presses;
``""`` or ``noop`` = the no-op action) and/or walks the shortest route to MAP
(``--to ROUTE_31``: re-plans every press, stops when the screen leaves walk
mode — a battle, a trainer's challenge, a scene — or the map is reached).
Prints one line per press (map, position, mode, on-screen text, battle,
reward sources paid) and the full harness prompt state at the end, and saves
``<dir>/<NAME>.state`` and ``<NAME>.png`` so the next command can continue
from it (pass ``<NAME>`` or the .state path as STATE).

Every invocation starts a fresh episode from STATE, exactly like a Go-Explore
restart: per-episode rewards (new_map, first_trainer_battle, ...) pay again.
Use it to check what a button does, what the policy observes and what the
reward pays — see .claude/skills/live-env-testing/SKILL.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.llm_agent.agent import format_state  # noqa: E402
from tools.llm_agent.game import Game  # noqa: E402

EDGE_MOVE = {"north": "up", "south": "down", "west": "left", "east": "right"}


def _line(g, button, sample) -> str:
    s = g.status()
    text = " / ".join(s["text"])[:60]
    b = s["battle"]
    battle = ""
    if b and b["you"]["max_hp"]:
        e, y = b["enemy"], b["you"]
        battle = (
            f" | {b['type']} {e['species']} Lv{e['level']} {e['hp']}/{e['max_hp']}"
            f" vs {y['species']} {y['hp']}/{y['max_hp']}"
        )
    paid = {k: v for k, v in sample["sources"].items() if k != "delivered_total"}
    return (
        f"{button or 'noop':>5} {s['map']} ({s['x']},{s['y']}) {s['mode']:<6}"
        f"{' | ' + text if text else ''}{battle}{' | ' + str(paid) if paid else ''}"
        f" [{g.env.last_step_frames}f]"
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("state", help="a .state path, or a NAME saved by an earlier call in --dir")
    p.add_argument("buttons", nargs="*")
    p.add_argument("--to", default=None, help="then walk the shortest route to this map")
    p.add_argument("--out", default="last", help="name to save the end state under")
    p.add_argument("--dir", default="Training Outputs/drive", help="where NAME states live")
    p.add_argument("--config", default="configs/stages/open_world.json")
    p.add_argument("--max-steps", type=int, default=300)
    args = p.parse_args()

    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    state = Path(args.state)
    if not state.exists():
        state = d / f"{args.state}.state"
    g = Game(args.config, state_path=str(state))
    presses = []
    for b in args.buttons:
        b, _, n = b.partition("*")
        presses += [("" if b == "noop" else b)] * (int(n) if n else 1)
    for b in presses:
        print(_line(g, b, g.press(b)))
    if args.to:
        for _ in range(args.max_steps):
            route = next((r for r in g.routes() if r["to"] == args.to), None)
            if route is None or route["steps"] is None:
                print(f"  no walking route to {args.to} from here")
                break
            move = route.get("first") or EDGE_MOVE[route["via"].split()[0]]
            print(_line(g, move, g.press(move)))
            s = g.status()
            if s["mode"] != "walk" or s["map"] == args.to:
                break
    s = g.status()
    if s["mode"] == "walk":
        s["routes"] = g.routes()
    print("\n" + format_state(s))
    print(f"party {g.party()}")
    out = d / f"{args.out}.state"
    out.write_bytes(g.snapshot())
    g.llm_image(grid=s["mode"] == "walk").save(d / f"{args.out}.png")
    print(f"saved {out} and {args.out}.png")
    g.close()


if __name__ == "__main__":
    main()
