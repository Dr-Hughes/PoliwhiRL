# -*- coding: utf-8 -*-
"""Replay a recorded run's button presses on the real env, step for step.

The env is deterministic, so a run's ``steps.jsonl`` plus its start states
reproduce it exactly. Use it to re-check a run after a framework change (did
the rewards or observation change?), to reach a state the LLM found without
the LLM, or to cut a button prefix for a scenario (``scenarios.py``).

    uv run python -m tools.llm_agent.replay "<run>"            # determinism check
    uv run python -m tools.llm_agent.replay "<run>" --until 227 --save-state out.state
    uv run python -m tools.llm_agent.replay "<run>" --buttons  # print the button list

Restarts follow the agent: an attempt after the first, and the goal after a
failed attempt, restart a fresh seeded episode from that goal's start state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.llm_agent.game import Game  # noqa: E402


def load_run(run_dir) -> dict:
    run = Path(run_dir)
    summary = json.loads((run / "summary.json").read_text())
    steps = [json.loads(line) for line in (run / "steps.jsonl").open()]
    return {"dir": run, "meta": summary["meta"], "attempts": summary["attempts"], "steps": steps}


def _start_state(run: dict, goal: str) -> bytes:
    hits = sorted((run["dir"] / "states").glob(f"*_{goal}_start.state"))
    if not hits:
        raise FileNotFoundError(f"no start state for goal {goal!r} in {run['dir']}/states")
    return hits[0].read_bytes()


def replay(run_dir, until: int | None = None, game: Game | None = None, on_step=None):
    """Press the run's buttons in order. Returns (game, divergences), where a
    divergence is a step whose (map, x, y) differs from the recording.
    ``on_step(game, row, sample)`` is called after every step."""
    run = load_run(run_dir)
    meta = run["meta"]
    g = game or Game(meta["config"], state_path=meta.get("state"))
    failed = {(a["goal"], a["attempt"]) for a in run["attempts"] if not a["success"]}
    seg = None
    diverged = []
    for row in run["steps"]:
        if until is not None and row["step"] > until:
            break
        key = (row["goal"], row["attempt"])
        if seg is not None and key != seg and (row["attempt"] > 1 or seg in failed):
            g.restart_from(_start_state(run, row["goal"]))
        seg = key
        sample = g.press(row["button"])
        s = g.status()
        if (s["map"], s["x"], s["y"]) != (row["map"], row["x"], row["y"]):
            diverged.append(
                {
                    "step": row["step"],
                    "recorded": (row["map"], row["x"], row["y"]),
                    "replayed": (s["map"], s["x"], s["y"]),
                }
            )
        if on_step:
            on_step(g, row, sample)
    return g, diverged


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run")
    p.add_argument("--until", type=int, default=None, help="stop after this harness step")
    p.add_argument("--save-state", default=None, help="write the final state here")
    p.add_argument("--buttons", action="store_true", help="print the buttons as JSON and exit")
    args = p.parse_args()
    if args.buttons:
        steps = load_run(args.run)["steps"]
        print(
            json.dumps(
                [r["button"] for r in steps if args.until is None or r["step"] <= args.until]
            )
        )
        return
    g, diverged = replay(args.run, until=args.until)
    s = g.status()
    print(
        f"replayed to {s['map']} ({s['x']},{s['y']}); "
        f"{len(diverged)} step(s) diverged from the recording"
    )
    for d in diverged[:10]:
        print(f"  step {d['step']}: recorded {d['recorded']} replayed {d['replayed']}")
    if args.save_state:
        Path(args.save_state).write_bytes(g.snapshot())
        print(f"state -> {args.save_state}")
    g.close()


if __name__ == "__main__":
    main()
