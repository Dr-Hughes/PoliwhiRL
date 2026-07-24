#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-driven Pokémon Crystal debugger — a thin vision-model driver on top of the
*real* PPO training environment.

Design goals:
  * Mirror the PPO training setup exactly. We instantiate the same
    ``PyBoyEnvironment`` used for training and drive it with ``env.step()``, so
    the action mapping, frame timing (90 frames/action, 15-frame button hold)
    and RAM decoding are identical to what the policy sees. No hand-rolled
    PyBoy driving — that previously caused nondeterministic multi-tile moves.
  * Give the vision model the native 160x144 screen and a small, clearly
    described set of controls exposed as a tool call.
  * Keep the prompt lean and concrete so small local models stay grounded.

Usage:
    # Verify controls with NO model: right x5, up x5 should reach downstairs.
    python tools/llm_game_debug.py --verify

    # Drive the game with a vision model.
    python tools/llm_game_debug.py --steps 40 --goal "leave the bedroom" \
        --model google/gemma-4-12b-qat [--display] [--thinking]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from collections import deque
from pathlib import Path

from openai import OpenAI

# Run from the repo root so the project package + configs resolve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import load_default_config, load_user_config, merge_configs  # noqa: E402
from PoliwhiRL.environment.gym_env import PyBoyEnvironment  # noqa: E402

# ---------------------------------------------------------------------------
# LLM config
# ---------------------------------------------------------------------------
LLM_BASE_URL = "http://192.168.0.189:11434/v1"
DEFAULT_MODEL = "google/gemma-4-12b-qat"

# SQLite log of RAM states, shared by --manual (human play) and the LLM driver.
DB_PATH = PROJECT_ROOT / "tools" / "llm_debug.db"

# Buttons the model is allowed to choose. Deliberately a small, meaningful set
# (no wait/start/select — those are in the env's ignored_buttons during
# training anyway) so a small model isn't tempted into no-ops.
CONTROL_ACTIONS = ["up", "down", "left", "right", "a", "b"]

FACING_NAMES = {1: "up", 2: "down", 3: "left", 4: "right"}


# ---------------------------------------------------------------------------
# Environment (mirror of training)
# ---------------------------------------------------------------------------
def build_env(display: bool) -> PyBoyEnvironment:
    """Instantiate the real training env, tweaked only for interactive use."""
    config = merge_configs(
        load_default_config(),
        load_user_config(str(PROJECT_ROOT / "configs" / "inference.json")),
    )
    # Interactive overrides — everything else stays as training uses it.
    config["vision"] = True  # render each tick so we can grab the screen
    config["record"] = False  # don't spew training PNGs
    config["goexplore_enabled"] = False
    config["goexplore_flag_capture"] = False
    config["episode_length"] = 100000

    env = PyBoyEnvironment(config, force_window=display)
    env.reset()
    if display:
        # Real-time so the SDL2 window is watchable (env defaults to 0/unbounded).
        env.pyboy.set_emulation_speed(1)
    return env


def loc(env: PyBoyEnvironment) -> dict:
    """Compact game-state read straight from the env's RAM manager."""
    v = env.ram.get_variables()
    return {
        "bank": v["map_bank"],
        "map": v["map_num"],
        "x": v["X"],
        "y": v["Y"],
        "facing": FACING_NAMES.get(v["player_direction"], "?"),
    }


def screen_b64(env: PyBoyEnvironment) -> str:
    """Native 160x144 RGB frame as a base64 JPEG (what we send to the model)."""
    img = env.pyboy.screen.image.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def do(env: PyBoyEnvironment, action_name: str):
    """Step the env by an action *name* using the env's own action mapping."""
    env.step(env.actions.index(action_name))


# ---------------------------------------------------------------------------
# RAM-state database (shared by --manual and the LLM driver)
# ---------------------------------------------------------------------------
def open_db(path: Path, reset: bool):
    """Open (and optionally wipe) the RAM-state log DB."""
    import sqlite3

    conn = sqlite3.connect(str(path))
    if reset:
        conn.execute("DROP TABLE IF EXISTS ram_log")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS ram_log (
               id       INTEGER PRIMARY KEY AUTOINCREMENT,
               run_ts   REAL,      -- run start time; groups rows by session
               mode     TEXT,      -- 'manual' | 'llm'
               step     INTEGER,   -- step/frame index within the run
               map_bank INTEGER, map INTEGER, x INTEGER, y INTEGER,
               facing   TEXT,
               button   TEXT,      -- action at this state ('' if unknown)
               ram_json TEXT       -- full env_vars snapshot for RAM debugging
           )"""
    )
    conn.commit()
    return conn


def log_state(conn, run_ts, mode, step, env, s, button):
    """Insert one RAM-state row (full env_vars for deep inspection)."""
    ram_json = json.dumps(env.ram.get_variables(), default=str)
    conn.execute(
        "INSERT INTO ram_log (run_ts, mode, step, map_bank, map, x, y, facing, "
        "button, ram_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            run_ts,
            mode,
            step,
            s["bank"],
            s["map"],
            s["x"],
            s["y"],
            s["facing"],
            button,
            ram_json,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an agent playing Pokémon Crystal on a Game Boy. Each turn you see the \
current 160x144 screen and must choose ONE button with the decide_action tool.

Goal: {goal}

Controls (each press advances one turn):
- up / down / left / right: move the player ONE tile that way. If a wall, \
furniture, or person is there, you do NOT move.
- a: talk / interact with the tile you face, confirm, or advance dialogue text.
- b: cancel or close a menu.

Rules:
1. LOOK at the screen first. Fill in `observation` with what is literally \
visible right now (the room, objects, any people, any on-screen text box). \
Report only what you see — do NOT guess from the goal. You may be alone in a room.
2. The player is the small character near the middle of the screen.
3. To leave a room, walk onto a door or a staircase.
4. If the message says a move was BLOCKED, do not repeat that direction — pick \
a different one.
Keep observation and reasoning to one short sentence each.
"""

DECIDE_TOOL = {
    "type": "function",
    "function": {
        "name": "decide_action",
        "description": "Report what is on screen, then press one button.",
        "parameters": {
            "type": "object",
            "properties": {
                "observation": {
                    "type": "string",
                    "description": (
                        "What is literally on the screen right now: room/area, "
                        "visible objects, people, and any text box. Do not guess "
                        "from the goal."
                    ),
                },
                "action": {
                    "type": "string",
                    "enum": CONTROL_ACTIONS,
                    "description": "The button to press this turn.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "One short sentence on why this button.",
                },
            },
            "required": ["observation", "action", "reasoning"],
        },
    },
}


def call_llm(
    client, model, goal, b64, status, nav_hint, history_text, thinking, verbose
):
    """Send screen + minimal state to the model; return the parsed decision."""
    status_line = (
        f"Current location: map {status['map']} (bank {status['bank']})  "
        f"tile ({status['x']}, {status['y']})  facing {status['facing']}."
    )
    parts = [status_line]
    if history_text:
        parts.append(history_text)
    if nav_hint:
        parts.append(nav_hint)
    parts.append("Look at the screen and choose one button.")
    user_text = "\n\n".join(parts)
    system_text = SYSTEM_PROMPT.format(goal=goal)

    if verbose:
        print("  " + "-" * 60)
        print("  [SYSTEM PROMPT]")
        for line in system_text.splitlines():
            print(f"    {line}")
        print("  [USER MESSAGE]")
        for line in user_text.splitlines():
            print(f"    {line}")
        print("    <screen image: 160x144 JPEG attached>")
        print("  " + "-" * 60)

    # Gemma-style thinking control. Off by default for lower latency.
    extra_body = None if thinking else {"thinkingConfig": {"thinkingLevel": "MINIMAL"}}

    resp = client.chat.completions.create(
        model=model,
        extra_body=extra_body,
        temperature=0.0,
        max_tokens=512,
        tools=[DECIDE_TOOL],
        tool_choice="required",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(goal=goal)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            },
        ],
    )
    msg = resp.choices[0].message
    if verbose:
        raw = msg.tool_calls[0].function.arguments if msg.tool_calls else msg.content
        print(f"  [MODEL RESPONSE]\n    {raw}")
        print("  " + "-" * 60)
    if not msg.tool_calls:
        return {"observation": "", "action": "", "reasoning": "(no tool call)"}
    try:
        return json.loads(msg.tool_calls[0].function.arguments)
    except json.JSONDecodeError:
        return {"observation": "", "action": "", "reasoning": "(bad tool args)"}


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_verify(display: bool, outdir: Path):
    """No model. Script right x5, up x5 and confirm we reach the downstairs.

    Ground truth: the bedroom is bank 24 / map 7; the downstairs (1F) is
    bank 24 / map 6. Reaching map 6 means the controls + env stepping work.
    """
    env = build_env(display)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"start: {loc(env)}")
    env.pyboy.screen.image.convert("RGB").save(outdir / "verify_00_start.png")

    reached_downstairs = False
    seq = ["right"] * 5 + ["up"] * 5
    for i, action in enumerate(seq, 1):
        do(env, action)
        s = loc(env)
        if s["map"] == 6 and s["bank"] == 24:
            reached_downstairs = True
        print(
            f"  {i:2d} {action:5s} -> bank={s['bank']} map={s['map']} "
            f"x={s['x']} y={s['y']} facing={s['facing']}"
            + ("   <-- DOWNSTAIRS" if s["map"] == 6 else "")
        )
        env.pyboy.screen.image.convert("RGB").save(
            outdir / f"verify_{i:02d}_{action}.png"
        )
    env.close()

    print()
    if reached_downstairs:
        print("PASS: reached the downstairs (map 6) — controls mirror training.")
    else:
        print("FAIL: never reached map 6. Controls/env stepping are off.")
    print(f"Frames saved to {outdir}")
    return reached_downstairs


def format_history(history) -> str:
    """Render the rolling short-term memory as compact text for the prompt."""
    if not history:
        return ""
    lines = ["Your recent turns (oldest first):"]
    for h in history:
        result = "moved" if h["moved"] else "did NOT move (blocked)"
        lines.append(
            f"  at map {h['map']} ({h['x']},{h['y']}) facing {h['facing']}, "
            f"pressed {h['action']} -> {result}"
        )
    return "\n".join(lines)


def run_llm(
    steps,
    goal,
    model,
    display,
    llm_url,
    thinking,
    outdir,
    history_len,
    verbose,
    conn,
    run_ts,
):
    env = build_env(display)
    outdir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=llm_url, api_key="local", timeout=300.0)

    prev_pos = None
    last_action = ""
    # Rolling short-term memory: the last `history_len` (state, button, result)
    # tuples. Gives the model continuity across turns so it can tell it just
    # changed floors / is looping, instead of deciding statelessly each step.
    history = deque(maxlen=history_len)
    print(
        f"Goal: {goal}\nModel: {model}\nThinking: {'on' if thinking else 'off'}  "
        f"Memory: {history_len} turns\n"
    )

    for i in range(1, steps + 1):
        s = loc(env)
        b64 = screen_b64(env)
        env.pyboy.screen.image.convert("RGB").save(outdir / f"step_{i:03d}.png")

        # Collision feedback: a directional move that didn't change (map, x, y)
        # hit a wall. Small models otherwise loop the same blocked direction.
        cur_pos = (s["bank"], s["map"], s["x"], s["y"])
        moved = prev_pos is not None and cur_pos != prev_pos
        nav_hint = ""
        if (
            last_action in ("up", "down", "left", "right")
            and prev_pos is not None
            and cur_pos == prev_pos
        ):
            nav_hint = (
                f"BLOCKED: your last move '{last_action}' did not change your "
                f"position — something is in the way. Try a DIFFERENT direction."
            )

        # Backfill whether the PREVIOUS turn's action actually moved us (we can
        # only know now that we've read the resulting position).
        if history and last_action:
            history[-1]["moved"] = moved

        print(f"--- Step {i}/{steps} ---")
        print(
            f"  map {s['map']} (bank {s['bank']})  ({s['x']},{s['y']})  facing {s['facing']}"
        )
        if nav_hint:
            print(f"  ⚠ {nav_hint}")

        t0 = time.time()
        d = call_llm(
            client,
            model,
            goal,
            b64,
            s,
            nav_hint,
            format_history(history),
            thinking,
            verbose,
        )
        dt = time.time() - t0

        action = d.get("action", "")
        if action not in CONTROL_ACTIONS:
            print(f"  [WARN] invalid action {action!r}; skipping turn")
            action = ""
        print(f"  Sees: {d.get('observation', '')[:150]}")
        print(
            f"  → {action or '(none)'}  ({dt:.0f}s)  — {d.get('reasoning', '')[:100]}"
        )

        # Record this turn in memory (result filled in next iteration).
        history.append(
            {
                "map": s["map"],
                "x": s["x"],
                "y": s["y"],
                "facing": s["facing"],
                "action": action or "none",
                "moved": False,
            }
        )
        if conn is not None:
            log_state(conn, run_ts, "llm", i, env, s, action or "")
        prev_pos = cur_pos
        last_action = action
        if action:
            do(env, action)

    env.close()
    print(f"\nDone. Frames saved to {outdir}")
    if conn is not None:
        print(f"RAM states logged to {DB_PATH}")


# Single-letter shortcuts the human types each turn in manual mode.
MANUAL_KEYS = {"u": "up", "d": "down", "l": "left", "r": "right", "a": "a", "b": "b"}


def save_replay(actions, run_ts, frames_per_action, outdir) -> Path:
    """Persist the button sequence as JSON so a run can be replayed exactly."""
    path = outdir / f"replay_{int(run_ts)}.json"
    path.write_text(
        json.dumps(
            {
                "created": run_ts,
                "frames_per_action": frames_per_action,
                "actions": actions,
            },
            indent=2,
        )
    )
    return path


def run_manual(conn, run_ts, outdir, frames_override):
    """Turn-based manual driver: one button per turn, stepped through the SAME
    env.step() the policy uses (default 90 frames/action, 15-frame hold), so the
    frame cadence matches training. Every turn we log the RAM state + button and
    append the button to a replay sequence saved on exit.
    """
    env = build_env(display=True)
    if frames_override:
        env.frames_per_action = frames_override
    outdir.mkdir(parents=True, exist_ok=True)

    prompt = "  turn> [u]p [d]own [l]eft [r]ight  [a] [b]  " "(Enter=wait, q=quit): "
    print(
        f"Turn-based manual mode — {env.frames_per_action} frames/turn "
        f"(hold {env.button_hold_frames}), matching training."
    )
    print(f"Logging RAM states + buttons to {DB_PATH}\n")

    actions = []
    turn = 0
    try:
        while True:
            s = loc(env)
            print(
                f"turn {turn}: map {s['map']} (bank {s['bank']})  "
                f"({s['x']},{s['y']})  facing {s['facing']}"
            )
            raw = input(prompt).strip().lower()
            if raw in ("q", "quit"):
                break
            if raw and raw not in MANUAL_KEYS:
                print(f"  ? unknown key {raw!r} — use u/d/l/r/a/b, Enter, or q")
                continue
            action = MANUAL_KEYS.get(raw, "")  # "" => wait (no-op turn)

            # Log the pre-action state + chosen button (consistent with LLM mode),
            # then step with the training cadence.
            turn += 1
            log_state(conn, run_ts, "manual", turn, env, s, action or "")
            actions.append(action)
            do(env, action)
            env.pyboy.screen.image.convert("RGB").save(
                outdir / f"manual_{turn:04d}.png"
            )
    except (KeyboardInterrupt, EOFError):
        print("\n(stopped)")

    env.close()
    replay_path = save_replay(actions, run_ts, env.frames_per_action, outdir)
    print(f"\nDone. {turn} turns logged to {DB_PATH}")
    print(f"Replay sequence ({len(actions)} buttons) saved to {replay_path}")
    print(f"Replay it with:  python {Path(__file__).name} --replay {replay_path}")


def run_replay(replay_path, conn, run_ts, outdir, display):
    """Re-execute a recorded button sequence through env.step(), logging each
    resulting RAM state. Deterministic — same buttons -> same trajectory."""
    data = json.loads(Path(replay_path).read_text())
    actions = data["actions"]
    env = build_env(display)
    if data.get("frames_per_action"):
        env.frames_per_action = data["frames_per_action"]
    outdir.mkdir(parents=True, exist_ok=True)
    print(
        f"Replaying {len(actions)} buttons from {replay_path} "
        f"({env.frames_per_action} frames/turn)\n"
    )

    for i, action in enumerate(actions, 1):
        s = loc(env)
        log_state(conn, run_ts, "replay", i, env, s, action or "")
        print(
            f"  {i:3d} {action or 'wait':5s} -> map {s['map']} "
            f"({s['x']},{s['y']}) facing {s['facing']}"
        )
        if action:
            do(env, action)
    # Log the final resulting state too.
    s = loc(env)
    print(f"  end       -> map {s['map']} ({s['x']},{s['y']}) facing {s['facing']}")
    env.close()
    print(f"\nReplay done. RAM states logged to {DB_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="LLM-driven Pokémon Crystal debugger")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--goal", type=str, default="leave the bedroom and go downstairs")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--llm-url", type=str, default=LLM_BASE_URL)
    p.add_argument("--display", action="store_true", help="Show the PyBoy window")
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Enable model thinking (off by default; sends "
        "thinkingConfig.thinkingLevel=MINIMAL)",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="No model: run right x5, up x5 and check we reach downstairs",
    )
    p.add_argument(
        "--manual",
        action="store_true",
        help="No model: turn-based hand-driving (one button per turn, "
        "training frame cadence), logging RAM + a replay sequence",
    )
    p.add_argument(
        "--replay",
        type=str,
        default=None,
        help="No model: replay a recorded button sequence JSON "
        "(from --manual) through env.step and log the states",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Override frames per turn in manual mode "
        "(default 0 = use the training value, 90)",
    )
    p.add_argument(
        "--history",
        type=int,
        default=6,
        help="Rolling short-term memory: number of past turns "
        "(state + button + result) shown to the model",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print the full system+user prompt and raw model " "response each step",
    )
    p.add_argument(
        "--db", type=str, default=str(DB_PATH), help="SQLite RAM-state log path"
    )
    p.add_argument(
        "--reset-db",
        action="store_true",
        help="Wipe the RAM-state log before this run so it isn't "
        "muddied by earlier testing",
    )
    p.add_argument(
        "--outdir", type=str, default=str(PROJECT_ROOT / "tools" / "llm_debug_frames")
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    if args.verify:
        ok = run_verify(args.display, outdir)
        sys.exit(0 if ok else 1)

    conn = open_db(Path(args.db), args.reset_db)
    run_ts = time.time()
    if args.replay:
        run_replay(args.replay, conn, run_ts, outdir, args.display)
    elif args.manual:
        run_manual(conn, run_ts, outdir, args.frames)
    else:
        run_llm(
            args.steps,
            args.goal,
            args.model,
            args.display,
            args.llm_url,
            args.thinking,
            outdir,
            args.history,
            args.verbose,
            conn,
            run_ts,
        )
    conn.close()


if __name__ == "__main__":
    main()
