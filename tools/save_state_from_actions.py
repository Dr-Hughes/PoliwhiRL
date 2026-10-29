# -*- coding: utf-8 -*-
"""Make a save-state from a recorded action prefix.

Replays an ``actions.steps`` file (a first-honest checkpoint recording under
``Runs/checkpoints/<label>/``) through the real env from the configured start
state, then presses B until the player is back in control (no script, no
battle) and saves that frame. Used to build the continuation probe's start
state, e.g.:

    uv run python tools/save_state_from_actions.py \\
        "Training Outputs/open_world_v3/Runs/checkpoints/1213-Beat-Falkner/actions.steps" \\
        emu_files/states/post_falkner.state
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.llm_agent.game import Game  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("actions", help="actions.steps file to replay")
    p.add_argument("out", help="where to write the save-state")
    p.add_argument("--config", default="configs/stages/open_world.json")
    p.add_argument("--settle-max", type=int, default=200, help="max B presses to regain control")
    args = p.parse_args()

    lines = Path(args.actions).read_text().splitlines()
    actions = [int(x) for x in lines if x.strip() and not x.startswith("#")]
    game = Game(args.config)
    env = game.env
    for a in actions:
        env.step(a)
    b = env.actions.index("b")
    for _ in range(args.settle_max):
        v = game.vars()
        if not v["script_active"] and not int(v["battle_type"]):
            break
        env.step(b)
    else:
        sys.exit(
            f"still under a script or battle after {args.settle_max} B presses: {game.where()}"
        )
    Path(args.out).write_bytes(game.snapshot())
    print(f"replayed {len(actions)} actions; saved {args.out} at {game.where()}")


if __name__ == "__main__":
    main()
