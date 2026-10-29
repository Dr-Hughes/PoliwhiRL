"""Summarise an llm_agent run: per-goal outcomes, model cost, and the reward
the training reward function paid along the way, by source.

The reward audit is the point: a competent path through a goal should be paid
mostly by the exploration/milestone sources, and any source that pays for
dithering (or a large negative on the obvious route) shows up here.

    uv run python -m tools.llm_agent.report "Training Outputs/llm_agent/<run>"
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from tools.llm_agent.agent import STALL_WARN


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reward_audit(steps: list[dict]) -> dict:
    """{(goal, attempt): {"steps", "reward", "sources": {src: total}}}."""
    out: dict = {}
    for s in steps:
        key = (s["goal"], s["attempt"])
        row = out.setdefault(key, {"steps": 0, "reward": 0.0, "sources": defaultdict(float)})
        row["steps"] += 1
        row["reward"] += float(s["reward"])
        for src, val in s.get("sources", {}).items():
            if src != "delivered_total":
                row["sources"][src] += float(val)
    return out


def format_report(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    turns = load_jsonl(run_dir / "turns.jsonl")
    steps = load_jsonl(run_dir / "steps.jsonl")
    lines = [f"Run: {run_dir}", f"Reached: {summary.get('reached')}", ""]
    lines.append(f"{'goal':<9} {'att':>3} {'ok':>3} {'kind':<7} {'steps':>5} {'turns':>5} "
                 f"{'errs':>4} {'llm_s':>7} {'s/turn':>6}")
    for a in summary["attempts"]:
        per = a["llm_s"] / a["turns"] if a["turns"] else 0.0
        lines.append(
            f"{a['goal']:<9} {a['attempt']:>3} {'Y' if a['success'] else 'n':>3} "
            f"{'seeded' if a['seeded'] else 'honest':<7} {a['steps']:>5} {a['turns']:>5} "
            f"{a['model_errors']:>4} {a['llm_s']:>7.1f} {per:>6.1f}"
        )
    if turns:
        toks = [t["usage"].get("total_tokens", 0) for t in turns if t.get("usage")]
        if toks:
            lines.append(f"\nTokens/turn: mean {sum(toks) / len(toks):.0f}, max {max(toks)}")
        stalls = sum(1 for t in turns if t.get("stall", 0) >= STALL_WARN)
        lines.append(f"Turns in a repeated situation (stall >= {STALL_WARN}): "
                     f"{stalls}/{len(turns)}")

    lines.append("\nReward audit (training reward function along the LLM's path):")
    for (goal, att), row in reward_audit(steps).items():
        srcs = sorted(row["sources"].items(), key=lambda kv: -abs(kv[1]))
        src_txt = ", ".join(f"{k}={v:+.2f}" for k, v in srcs if abs(v) >= 0.005)
        lines.append(f"  {goal} a{att}: {row['steps']} steps, total {row['reward']:+.2f}  [{src_txt}]")
    return "\n".join(lines)


def observation_audit(run_dir: Path) -> list[str]:
    """Which of the policy's RAM features never changed over the run's demos
    (the exact vectors a policy would have seen), plus NaN / range checks. A
    feature constant over a stretch where the game state it encodes did
    change is a dark or mis-wired observation."""
    import glob
    import re

    import numpy as np

    from PoliwhiRL.environment.gym_env import RAM_FEATURE_KEYS

    files = sorted(glob.glob(str(Path(run_dir) / "demos" / "*.npz")))
    if not files:
        return ["(no demos)"]
    ram = np.concatenate([np.load(f)["ram"] for f in files])
    names = list(RAM_FEATURE_KEYS)
    out = [f"{ram.shape[0]} steps x {ram.shape[1]} features; NaN {int(np.isnan(ram).sum())}; "
           f"range [{np.nanmin(ram):.2f}, {np.nanmax(ram):.2f}]"]
    groups: dict = {}
    for i, name in enumerate(names):
        g = re.sub(r"_?\d+$", "", name)
        d = groups.setdefault(g, {"n": 0, "const": 0, "vals": set()})
        d["n"] += 1
        if np.ptp(ram[:, i]) == 0:
            d["const"] += 1
            d["vals"].add(round(float(ram[0, i]), 3))
    const = [f"{g} ({d['const']}/{d['n']} @ {sorted(d['vals'])[:3]})"
             for g, d in groups.items() if d["const"]]
    out.append("Never changed: " + ", ".join(const))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(format_report(Path(sys.argv[1])))
    print("\nObservation audit (policy RAM vector over the run's demos):")
    print("\n".join(observation_audit(Path(sys.argv[1]))))
