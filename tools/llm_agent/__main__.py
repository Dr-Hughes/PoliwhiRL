# -*- coding: utf-8 -*-
"""CLI: run the goal ladder with a vision LLM on the real training env.

    uv run python -m tools.llm_agent                       # full ladder, honest start
    uv run python -m tools.llm_agent --from violet \\
        --state "Training Outputs/llm_agent/<run>/states/10_violet_start.state"

Watch a run:  tail -f "<run>/live.log"   and   open "<run>/live.html"
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from tools.llm_agent.agent import Agent, Memory  # noqa: E402
from tools.llm_agent.game import Game  # noqa: E402
from tools.llm_agent.goals import GOALS, LADDER  # noqa: E402
from tools.llm_agent.llm import DEFAULT_MODEL, DEFAULT_URL, LLMClient  # noqa: E402
from tools.llm_agent.notebook import Notebook  # noqa: E402
from tools.llm_agent.recorder import Recorder  # noqa: E402


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    # Stop cleanly (report, videos, notebook) on SIGINT and SIGTERM, also when
    # launched from a background shell, which starts us with SIGINT ignored.
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--goals",
        default=None,
        help=f"comma list (default: whole ladder) from {[g.key for g in LADDER]}",
    )
    p.add_argument(
        "--from",
        dest="start_goal",
        default=None,
        help="run the ladder from this goal onwards (use with --state)",
    )
    p.add_argument(
        "--config",
        default="configs/stages/open_world.json",
        help="training config whose env + reward settings to mirror",
    )
    p.add_argument(
        "--state", default=None, help="start from this save-state instead of the config's"
    )
    p.add_argument(
        "--resume",
        default=None,
        metavar="RUN_DIR|SAVE_JSON",
        help="continue a run from its latest autosave, or from one save "
        "(states/autosave_<time>.json): state, goal, notes, notebook; "
        "writes to RUN_DIR_resumed_<time> unless --out is given",
    )
    p.add_argument(
        "--autosave-minutes",
        type=float,
        default=15.0,
        help="save-state + resume info every N minutes (0 = off)",
    )
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--fallback-model",
        default=None,
        help="used on model errors, long stalls and retry attempts",
    )
    p.add_argument("--no-thinking", action="store_true", help="thinking off for --model")
    p.add_argument(
        "--fallback-no-thinking",
        action="store_true",
        help="thinking off for --fallback-model (default on)",
    )
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--until", default=None, help="stop the ladder after this goal")
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="stop the run after this many env steps in total (0 = no cap)",
    )
    p.add_argument(
        "--budget-scale", type=float, default=1.0, help="multiply every goal's step budget"
    )
    p.add_argument("--two-frames", action="store_true", help="also send the previous turn's screen")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--display", action="store_true", help="also open PyBoy's SDL game window")
    p.add_argument(
        "--out", default=None, help="default: Training Outputs/llm_agent/run_<timestamp>"
    )
    p.add_argument("--verbose", action="store_true", help="print model reasoning")
    args = p.parse_args()
    resume = None
    if args.resume:
        import json

        save = Path(args.resume)
        run_dir = save.parent.parent if save.is_file() else save
        resume = json.loads((save if save.is_file() else save / "autosave.json").read_text())
        args.state = resume["state"]
        args.start_goal = args.start_goal or resume["goal"]
        args.out = args.out or f"{str(run_dir).rstrip('/')}_resumed_{time.strftime('%H%M%S')}"

    if args.goals:
        keys = [k.strip() for k in args.goals.split(",") if k.strip()]
    else:
        keys = [g.key for g in LADDER]
    if args.start_goal:
        if args.start_goal not in keys:
            p.error(f"--from {args.start_goal!r} is not in the goal list")
        keys = keys[keys.index(args.start_goal) :]
    if args.until:
        if args.until not in keys:
            p.error(f"--until {args.until!r} is not in the goal list")
        keys = keys[: keys.index(args.until) + 1]
    unknown = [k for k in keys if k not in GOALS]
    if unknown:
        p.error(f"unknown goals {unknown}")
    goals = [GOALS[k] for k in keys]
    if args.budget_scale != 1.0:
        goals = [replace(g, budget=int(g.budget * args.budget_scale)) for g in goals]
    if args.fallback_model == args.model:
        args.fallback_model = None

    out = Path(args.out or f"Training Outputs/llm_agent/run_{time.strftime('%Y%m%d_%H%M%S')}")
    rec = Recorder(out, dict(vars(args)))
    game = Game(args.config, display=args.display, state_path=args.state)
    if args.state:
        game.seeded = True  # not the true start: keep it out of honest data
    llm = LLMClient(args.url, args.model, thinking=not args.no_thinking)
    agent = Agent(
        game,
        llm,
        rec,
        temperature=args.temperature,
        max_attempts=args.max_attempts,
        two_frames=args.two_frames,
        fallback_model=args.fallback_model,
        fallback_thinking=not args.fallback_no_thinking,
        verbose=args.verbose,
        video=not args.no_video,
        ladder=goals,
        autosave_every=args.autosave_minutes * 60,
        max_steps=args.max_steps,
    )
    memory = Memory(notebook=Notebook(out / "notebook.jsonl"))
    if resume:
        memory.notes = resume.get("notes", "")
        memory.notebook.load(resume.get("notebook", []))
        agent.reached = list(resume.get("reached", []))
        print(
            f"Resumed from {args.resume} autosave {resume['time']}: goal {resume['goal']} "
            f"(step {resume['goal_steps']} of attempt {resume['attempt']}), "
            f"{len(memory.notebook.facts)} notebook facts"
        )
    print(
        f"Run dir: {out}\nModel: {args.model} (fallback {args.fallback_model}) @ {args.url}  "
        f"thinking={not args.no_thinking}\nLive view: open \"{out / 'live.html'}\"\n"
        f"Goals: {' -> '.join(keys)}",
        flush=True,
    )
    try:
        for goal in goals:
            if agent.run_goal(goal, memory):
                continue
            if agent.out_of_steps():
                print(f"\nStopping: step cap of {args.max_steps} env steps reached.")
                break
            if goal.optional:
                print(f"\nOptional goal '{goal.key}' not reached; continuing from its start state.")
                continue
            print(f"\nStopping: goal '{goal.key}' not reached in {args.max_attempts} attempts.")
            break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        rec.summary["reached"] = agent.reached
        agent.close()
        memory.notebook.close()
        rec.close()
        game.close()
    print(f"\nReached: {agent.reached or 'none'}\nReport: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
