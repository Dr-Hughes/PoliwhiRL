"""A/B a game state across models: send the exact prompt the agent would send
N times per model and count the buttons chosen. Use it to tell "the model
can't see/understand this" from "the prompt/state is misleading it" — fix the
prompt, re-run, compare.

    uv run python -m tools.llm_agent.probe --state "<run>/states/03_outside_start.state" \\
        --press right,down,down,down,down,left --goal outside \\
        --models Qwen3.8-Flash-Next-oQ4e-mtp:think,Ornith-1.5-35B-A3B-oQ8e-mtp:nothink
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from tools.llm_agent.agent import SYSTEM_PROMPT, Memory, format_state  # noqa: E402
from tools.llm_agent.game import Game  # noqa: E402
from tools.llm_agent.goals import GOALS  # noqa: E402
from tools.llm_agent.llm import DEFAULT_URL, LLMClient  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", default=None)
    p.add_argument("--press", default="", help="comma list of buttons to press first")
    p.add_argument("--goal", default="stairs", choices=list(GOALS))
    p.add_argument("--models", required=True, help="comma list of model[:think|:nothink]")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--save-image", default=None, help="write the image the models saw here")
    args = p.parse_args()

    g = Game("configs/stages/open_world.json", state_path=args.state)
    for b in [b for b in args.press.split(",") if b]:
        g.press(b.strip())
    s = g.status()
    user = "\n\n".join([
        "GAME STATE (exact, from memory):\n" + format_state(s), Memory().format(),
        "HARNESS:\nGoal steps used: 0/%d (attempt 1)." % GOALS[args.goal].budget,
        "The screen image is attached. Decide your next buttons.",
    ])
    img = g.llm_image(grid=s["mode"] == "walk")
    if args.save_image:
        img.save(args.save_image)
    print(user, "\n")
    system = SYSTEM_PROMPT.format(goal=GOALS[args.goal].text)
    for spec in args.models.split(","):
        model, _, mode = spec.partition(":")
        llm = LLMClient(args.url, model, thinking=mode != "nothink")
        counts = collections.Counter()
        for k in range(args.samples):
            r = llm.decide(system, user, [g.png_b64(img)], args.temperature)
            d = r["decision"] or {}
            counts[" ".join(d.get("buttons") or ["<error>"])] += 1
            print(f"[{model} {mode or 'think'} #{k + 1} {r['latency']:.1f}s] "
                  f"{d.get('buttons')} — {str(d.get('plan') or r['error'])[:200]}")
        print(f"==> {model}: {dict(counts)}\n")
    g.close()


if __name__ == "__main__":
    main()
