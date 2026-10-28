"""Markdown write-ups: one per goal attempt (goals/NN_key_aK.md) and a run
index (REPORT.md) regenerated after every attempt, so a run can be read and
analysed while it is still going."""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from tools.llm_agent.verify import RULES


def _rel(path, base: Path) -> str:
    return str(Path(path).relative_to(base)) if path else "—"


def _src_table(sources: dict) -> list[str]:
    rows = sorted(((k, v) for k, v in sources.items() if abs(v) >= 0.005), key=lambda kv: -abs(kv[1]))
    if not rows:
        return ["(no reward paid)"]
    out = ["| source | total |", "|---|---:|"]
    out += [f"| {k} | {v:+.2f} |" for k, v in rows]
    return out


def _verify_table(table: list[dict]) -> list[str]:
    if not table:
        return ["(no events detected)"]
    out = ["| event | rule | expected source | events | paid | not paid |", "|---|---|---|---:|---:|---:|"]
    for r in table:
        out.append(f"| {r['kind']} | {r['strength']} | {r['sources']} | {r['events']} | "
                   f"{r['paid']} | {len(r['missed'])} |")
    misses = [ev for r in table for ev in r["missed"] if ev.verdict == "MISSED"]
    if misses:
        out += ["", "**Missed `must` payouts:**", ""]
        out += [f"- step {ev.step}: {ev.kind} — {ev.detail} {ev.context or ''}" for ev in misses[:30]]
    return out


def write_goal_doc(run_dir: Path, idx: int, goal, attempt: dict, turns: list[dict],
                   sources: dict, verifier, video, clips: list, states: dict,
                   notebook=None) -> Path:
    name = f"{idx:02d}_{goal.key}_a{attempt['attempt']}"
    path = run_dir / "goals" / f"{name}.md"
    path.parent.mkdir(exist_ok=True)
    ok = "SUCCESS" if attempt["success"] else "FAILED"
    lines = [
        f"# {idx:02d} · {goal.title} — attempt {attempt['attempt']}: {ok}",
        "",
        f"> {goal.text}",
        "",
        "| | |", "|---|---|",
        f"| outcome | **{ok}** ({'seeded' if attempt['seeded'] else 'honest'}) |",
        f"| env steps | {attempt['steps']} / budget {goal.budget} |",
        f"| model turns | {attempt['turns']} (errors {attempt['model_errors']}, "
        f"fallback turns {attempt.get('fallback_turns', 0)}) |",
        f"| wall time | {attempt['wall_s']:.0f}s (model {attempt['llm_s']:.0f}s) |",
        f"| video | [{_rel(video, run_dir)}](../{_rel(video, run_dir)}) |",
        f"| start state | `{_rel(states.get('start'), run_dir)}` |",
        f"| end state | `{_rel(states.get('done'), run_dir)}` |",
        "",
        "## Reward paid along this attempt (training reward function)",
        "",
        *_src_table(sources),
        "",
        "## Reward verification",
        "",
        *_verify_table(verifier.table(goal.key, attempt["attempt"])),
    ]
    unexplained = [u for u in verifier.unexplained
                   if u["goal"] == goal.key and u["attempt"] == attempt["attempt"]]
    if unexplained:
        lines += ["", "**Unexplained milestone-type payouts** (no matching event in window):", ""]
        lines += [f"- step {u['step']}: {u['source']} {u['value']:+.2f} (button {u['button']})"
                  for u in unexplained[:30]]
    stored = [r for t in turns for r in t.get("remembered") or []]
    recalls = [(t["turn"], t["decision"].get("recall"), t.get("recalled") or [])
               for t in turns if (t.get("decision") or {}).get("recall")]
    if stored or recalls or notebook is not None:
        lines += ["", "## Notebook", ""]
        lines += [f"- remembered: {r}" for r in stored] or ["- (nothing remembered this attempt)"]
        lines += [f"- turn {tn} recall \"{q}\" → {len(hits)} facts" for tn, q, hits in recalls]
        if notebook is not None and notebook.facts:
            lines += ["", f"Notebook at end of attempt ({notebook.words()} words):", ""]
            lines += [f"{i + 1}. {f['text']} _(goal {f['goal']}, {f['map']}, recalled {f['uses']}x)_"
                      for i, f in enumerate(notebook.facts)]
    if clips:
        lines += ["", "## Highlight clips", ""]
        lines += [f"- [{_rel(c, run_dir)}](../{_rel(c, run_dir)})" for c in clips]
    lines += ["", "## Turn log", "",
              "| turn | where | model | sees | plan | pressed → result |", "|---:|---|---|---|---|---|"]
    for t in turns:
        d = t["decision"] or {}
        where = f"{t['state']['map']} ({t['state']['x']},{t['state']['y']})"
        if t["state"].get("text_open"):
            where += " [scene]"
        cell = lambda s: str(s or "").replace("|", "/").replace("\n", " ")[:160]  # noqa: E731
        lines.append(f"| {t['turn']} | {where} | {cell(t.get('model', '')).split('-')[0]} | "
                     f"{cell(d.get('observation'))} | {cell(d.get('plan'))} | {cell(t['result'])} |")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_report(run_dir: Path, meta: dict, attempts: list[dict], verifier, ladder, reached) -> Path:
    path = run_dir / "REPORT.md"
    keys = [g.key for g in ladder]
    lines = [
        f"# LLM agent run — {run_dir.name}",
        "",
        f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}. Model `{meta.get('model')}` "
        f"(fallback `{meta.get('fallback_model') or 'none'}`), thinking "
        f"{'off' if meta.get('no_thinking') else 'on'}.",
        "",
        f"**Goals reached: {len(reached)}/{len(keys)}** — {', '.join(reached) or 'none yet'}",
        "",
        "| # | goal | attempt | outcome | kind | steps | turns | wall | write-up | video |",
        "|---:|---|---:|---|---|---:|---:|---:|---|---|",
    ]
    for a in attempts:
        idx = keys.index(a["goal"]) + 1 if a["goal"] in keys else 0
        lines.append(
            f"| {idx} | {a['goal']} | {a['attempt']} | {'✅' if a['success'] else '❌'} | "
            f"{'seeded' if a['seeded'] else 'honest'} | {a['steps']} | {a['turns']} | "
            f"{a['wall_s'] / 60:.1f}m | [doc]({_rel(a.get('doc'), run_dir)}) | "
            f"[mp4]({_rel(a.get('video'), run_dir)}) |")
    lines += ["", "## Reward verification (whole run)", "", *_verify_table(verifier.table())]
    by_src = defaultdict(float)
    for u in verifier.unexplained:
        by_src[u["source"]] += u["value"]
    if by_src:
        lines += ["", "**Unexplained milestone-type payouts by source:** "
                  + ", ".join(f"{k} {v:+.1f}" for k, v in by_src.items())]
    lines += ["", "### Rules", "",
              "`must`: the source is designed to pay for the event, a miss is a finding. "
              "`may`: pays only under conditions (novelty, caps, decay); the hit rate is "
              "informational. See `tools/llm_agent/verify.py`.", ""]
    lines += [f"- **{k}** ({s}): {', '.join(srcs)} within {w} steps" for k, (s, srcs, w) in RULES.items()]
    lines += ["", "Files: `turns.jsonl` (every model call), `steps.jsonl` (every env step with "
              "per-source reward), `events.jsonl` (verified events), `demos/` (training-format "
              "arrays), `states/` (save-states), `videos/`, `goals/`."]
    path.write_text("\n".join(lines) + "\n")
    return path
