#!/usr/bin/env python3
"""
Standalone LLM-driven Pokémon Crystal debugger.

Runs PyBoy, captures screen + RAM each step, sends a vision prompt to an
OpenAI-compatible LLM (Ollama/LM Studio), and logs everything with constrained
labels. Uses tool calling for fast, structured decisions.

Usage:
    python tools/llm_game_debug.py [--steps 10] [--goal "obtain starter pokemon"] \
        [--display] [--model qwen3.6-27b-mlx] [--llm-url http://...]

No PoliwhiRL framework dependencies — only pyboy, openai, and Pillow.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
from openai import OpenAI


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROM_PATH = PROJECT_ROOT / "emu_files" / "Pokemon - Crystal Version.gbc"
STATE_PATH = PROJECT_ROOT / "emu_files" / "states" / "start.state"
DB_PATH = PROJECT_ROOT / "tools" / "llm_debug.db"

# ---------------------------------------------------------------------------
# LLM config
# ---------------------------------------------------------------------------
LLM_BASE_URL = "http://192.168.0.189:11434/v1"
DEFAULT_MODEL = "qwen3.6-27b-mlx"

# ---------------------------------------------------------------------------
# GameBoy action space (matches PoliwhiRL's mapping)
# ---------------------------------------------------------------------------
ACTIONS: list[str] = ["", "a", "b", "left", "right", "up", "down", "start", "select"]

# Frames advanced per LLM decision, and how long a button is held within them.
# A single-frame tap only makes the character *turn*; holding for ~16+ frames is
# required to actually walk a tile, so we hold for a good chunk of the step.
FRAMES_PER_STEP = 60
HOLD_FRAMES = 24

# ---------------------------------------------------------------------------
# Constrained status labels the LLM should pick from for logging
# ---------------------------------------------------------------------------
STATUS_LABELS: list[str] = [
    "title_screen",
    "walking",
    "npc_dialogue",
    "menu_open",
    "battle_wild",
    "battle_trainer",
    "event_sequence",
    "transition",
    "overworld_map",
    "indoor_exploring",
]

# ---------------------------------------------------------------------------
# Key RAM addresses (from RAM.py / RAM_MAPPING.md)
# ---------------------------------------------------------------------------
RAM_ADDRESSES: dict[str, int] = {
    "room_id": 0xD148,
    "map_number": 0xDCB6,
    "overworld_x": 0xDCB8,
    "overworld_y": 0xDCB7,
    "facing_direction": 0xD357,       # 1=up 2=down 3=left 4=right
    "player_state": 0xD95D,           # 0=walking 1=battle 2=cycling 4=surfing
    "battle_type": 0xD22D,            # 0=none 1=wild 2=trainer
    "badges": 0xD857,
    "script_active": 0xD438,          # 255=running
    "ui_state": 0xCF07,               # 0=outdoor 5=indoor 7=text_box
    "map_handler": 0xD43D,            # 128=indoor 165=outdoor
    "money_lo": 0xD84E,
    "money_mid": 0xD84F,
    "money_hi": 0xD850,
}

# Derived flag addresses (story event flags at 0xDA72..0xDB71)
STORY_FLAG_BASE = 0xDA72

# Key story flags for debugging (from _DERIVED_FLAG_TABLE / RAM_MAPPING.md)
KEY_FLAGS: dict[str, int] = {
    "met_profan": 0x16,               # Flag offset from base
    "received_starter": 0x0A,         # Received a starter Pokemon
    "talked_to_prof": 0x05,           # Talked to Professor Oak
    "left_house": 0x1E,               # Left the starting house
}

FACING_NAMES = {1: "up", 2: "down", 3: "left", 4: "right"}
STATE_NAMES = {0: "walking", 1: "battle", 2: "cycling", 4: "surfing"}

# Tool definition
DECIDE_TOOL = {
    "type": "function",
    "function": {
        "name": "decide_action",
        "description": (
            "Decide the next button press based on what you see. "
            "Choose one action, classify the screen status, and give a brief reason."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ACTIONS,
                    "description": (
                        'Button to press. "" means do nothing and wait a frame.'
                    ),
                },
                "status_label": {
                    "type": "string",
                    "enum": STATUS_LABELS,
                    "description": (
                        "What is happening on screen right now? Pick the best match."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation (1-2 sentences) of why you chose this action.",
                },
            },
            "required": ["action", "status_label", "reasoning"],
        },
    },
}


# ===================================================================
# Database helpers
# ===================================================================

def init_db(db_path: Path) -> sqlite3.Connection:
    """Create / open the SQLite log database."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    # Clear previous session data so step_num stays unique
    conn.execute("DELETE FROM steps")
    conn.execute("DELETE FROM session_meta")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS steps (
            step_num      INTEGER PRIMARY KEY,
            timestamp     REAL,
            status_label  TEXT,
            llm_action    TEXT,
            llm_reasoning TEXT,
            screen_b64    TEXT,
            ram_snapshot  TEXT
        );
        CREATE TABLE IF NOT EXISTS session_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()
    return conn


def log_session_meta(conn: sqlite3.Connection, goal: str, model: str):
    for k, v in [("goal", goal), ("model", model)]:
        conn.execute(
            "INSERT OR REPLACE INTO session_meta (key, value) VALUES (?, ?)", (k, v)
        )
    conn.commit()


def log_step(
    conn: sqlite3.Connection,
    step_num: int,
    status_label: str,
    llm_action: str,
    llm_reasoning: str,
    screen_b64: str,
    ram_snapshot: dict,
):
    conn.execute(
        """INSERT INTO steps
           (step_num, timestamp, status_label, llm_action, llm_reasoning, screen_b64, ram_snapshot)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (step_num, time.time(), status_label, llm_action, llm_reasoning, screen_b64, json.dumps(ram_snapshot)),
    )
    conn.commit()


# ===================================================================
# PyBoy helpers
# ===================================================================

def snapshot_ram(pb) -> dict:
    """Read key RAM addresses and return a flat dict."""
    snap: dict = {}
    for name, addr in RAM_ADDRESSES.items():
        snap[name] = pb.memory[addr]

    # Derived convenience fields
    snap["facing"] = FACING_NAMES.get(snap["facing_direction"], "?")
    snap["player_state_name"] = STATE_NAMES.get(snap["player_state"], f"unknown({snap['player_state']})")
    snap["money"] = (snap["money_hi"] << 16) | (snap["money_mid"] << 8) | snap["money_lo"]

    # Story flags
    flag_base = STORY_FLAG_BASE
    for flag_name, offset in KEY_FLAGS.items():
        snap[f"flag_{flag_name}"] = bool(pb.memory[flag_base + offset] & 1)

    return snap


def get_screen_rgb(pb) -> np.ndarray:
    """Return the screen as an RGB numpy array (144x160)."""
    pil_img = pb.screen.image  # PIL RGBA Image
    return np.array(pil_img.convert("RGB"))


def screen_to_base64(rgb: np.ndarray) -> str:
    """Encode an RGB frame as a base64 JPEG string for the LLM."""
    from PIL import Image
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ===================================================================
# LLM interaction (tool-calling)
# ===================================================================

SYSTEM_PROMPT = """\
You are playing Pokémon Crystal on a Game Boy Color emulator. You see the \
current screen and must decide which button to press next using the \
`decide_action` tool.

Your goal: {goal}

Use the RAM summary and screen image to make your decision. Keep reasoning \
brief — 1-2 sentences is enough.
"""


def call_llm(
    client: OpenAI,
    model: str,
    goal: str,
    screen_b64: str,
    ram_snapshot: dict,
    step_history: list[str],
) -> dict:
    """Send screen + context to the LLM via tool calling and return decision."""

    system = SYSTEM_PROMPT.format(goal=goal)

    # Build a compact RAM summary for the prompt
    ram_summary = (
        f"Map: {ram_snapshot.get('map_number', '?')}  "
        f"Pos: ({ram_snapshot.get('overworld_x', '?')}, {ram_snapshot.get('overworld_y', '?')})  "
        f"Facing: {ram_snapshot.get('facing', '?')}  "
        f"State: {ram_snapshot.get('player_state_name', '?')}  "
        f"Script active: {bool(ram_snapshot.get('script_active', 0))}  "
        f"UI: {ram_snapshot.get('ui_state', '?')}  "
        f"Badges: {ram_snapshot.get('badges', '?')}"
    )

    # Key flags summary
    flag_lines = []
    for k, v in ram_snapshot.items():
        if k.startswith("flag_"):
            flag_lines.append(f"{k}={v}")
    flags_text = "  ".join(flag_lines) if flag_lines else ""

    # Recent step history (last 5) for context
    history_text = ""
    if step_history:
        history_text = "\nRecent steps:\n" + "\n".join(step_history[-5:])

    user_msg = (
        f"{ram_summary}\n{flags_text}\n{history_text}\n\n"
        "Look at the screen image and decide your next move using the tool."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_msg},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{screen_b64}",
                            "detail": "low",
                        },
                    },
                ],
            },
        ],
        tools=[DECIDE_TOOL],
        tool_choice="required",  # LM Studio only accepts string values
        temperature=0.0,
        max_tokens=4096,
    )

    msg = response.choices[0].message

    # Extract tool call arguments
    if not msg.tool_calls:
        print(f"  [WARN] No tool call in response. Content: {msg.content[:100]}")
        return {"action": "", "status_label": "transition", "reasoning": "no tool call"}

    tc = msg.tool_calls[0]
    try:
        args = json.loads(tc.function.arguments)
    except json.JSONDecodeError as e:
        print(f"  [WARN] Could not parse tool args: {tc.function.arguments[:100]} ({e})")
        return {"action": "", "status_label": "transition", "reasoning": "parse error"}

    return args


# ===================================================================
# Main loop
# ===================================================================

def run(
    steps: int = 10,
    goal: str = "obtain starter pokemon",
    model: str = DEFAULT_MODEL,
    show_display: bool = False,
    llm_url: str = LLM_BASE_URL,
):
    # -- Setup PyBoy --------------------------------------------------------
    print(f"ROM: {ROM_PATH}")
    print(f"State: {STATE_PATH}")

    # Copy ROM to temp dir so PyBoy doesn't mutate originals
    tmpdir = tempfile.mkdtemp(prefix="pyboy_llm_")
    conn: sqlite3.Connection | None = None

    try:
        tmp_rom = Path(tmpdir) / ROM_PATH.name
        shutil.copy2(ROM_PATH, tmp_rom)

        for ext in [".ram", ".rtc"]:
            src = ROM_PATH.with_suffix(ROM_PATH.suffix + ext)
            if src.exists():
                shutil.copy2(src, Path(tmpdir) / src.name)

        import pyboy
        window_type = "SDL2" if show_display else "null"
        print(f"Display: {'SDL2 (visible)' if show_display else 'headless'}")
        p = pyboy.PyBoy(
            str(tmp_rom),
            window=window_type,
            sound_emulated=False,
        )
        # Real-time (1x) when a window is shown so gameplay is watchable;
        # unbounded (0) when headless so we don't waste wall-clock ticking.
        p.set_emulation_speed(1 if show_display else 0)

        # Load saved state (start of game) — PyBoy wants a file-like object
        if STATE_PATH.exists():
            with open(STATE_PATH, "rb") as f:
                state_bytes = f.read()
            p.load_state(io.BytesIO(state_bytes))
            print(f"Loaded state: {STATE_PATH}")

        # -- Setup DB -------------------------------------------------------
        conn = init_db(DB_PATH)
        log_session_meta(conn, goal, model)

        # -- Setup LLM client -----------------------------------------------
        llm_client = OpenAI(base_url=llm_url, api_key="ollama", timeout=300.0)

        # -- Main loop ------------------------------------------------------
        step_history: list[str] = []

        for i in range(1, steps + 1):
            # Capture screen
            rgb = get_screen_rgb(p)
            b64 = screen_to_base64(rgb)

            # Read RAM
            ram = snapshot_ram(p)

            print(f"\n--- Step {i}/{steps} ---")
            print(
                f"  Map={ram.get('map_number')} "
                f"Pos=({ram.get('overworld_x')},{ram.get('overworld_y')}) "
                f"Facing={ram.get('facing')} "
                f"State={ram.get('player_state_name')} "
                f"Script={bool(ram.get('script_active'))} "
                f"UI={ram.get('ui_state')}"
            )

            # Call LLM
            t0 = time.time()
            decision = call_llm(llm_client, model, goal, b64, ram, step_history)
            elapsed = time.time() - t0

            action = decision.get("action", "")
            status = decision.get("status_label", "unknown")
            reasoning = decision.get("reasoning", "")

            # Validate action
            if action not in ACTIONS:
                print(f"  [WARN] Unknown action '{action}', defaulting to ''")
                action = ""

            print(f"  → Action: {action or '(noop)'}  |  Status: {status}  ({elapsed:.0f}s)")
            print(f"     Reasoning: {reasoning[:120]}")

            # Log to DB
            log_step(conn, i, status, action, reasoning, b64, ram)

            # Add to history
            step_history.append(f"Step {i}: action={action or 'noop'}, status={status}")

            # Execute action in PyBoy. Hold the button for HOLD_FRAMES so the
            # game registers a real step/press (a 1-frame tap only turns you),
            # then release and let the animation finish. Tick one frame at a
            # time with render=True so the SDL2 window animates in real time
            # (tick(N) would only render the final frame). Window events are
            # pumped inside tick() automatically — no manual process_events.
            if action:
                p.button_press(action)
            for f in range(FRAMES_PER_STEP):
                if action and f == HOLD_FRAMES:
                    p.button_release(action)
                p.tick(1, True)
            if action and FRAMES_PER_STEP <= HOLD_FRAMES:
                p.button_release(action)

        print(f"\nDone. {steps} steps logged to {DB_PATH}")
        print(f"Query with: sqlite3 {DB_PATH} 'SELECT step_num, status_label, llm_action FROM steps;'")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if conn is not None:
            conn.close()


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM-driven Pokémon Crystal debugger")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run")
    parser.add_argument("--goal", type=str, default="obtain starter pokemon", help="Goal prompt for the LLM")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--display", action="store_true", help="Show PyBoy SDL2 window")
    parser.add_argument("--llm-url", type=str, default=LLM_BASE_URL, help="LLM API base URL")
    args = parser.parse_args()

    run(steps=args.steps, goal=args.goal, model=args.model, show_display=args.display, llm_url=args.llm_url)


if __name__ == "__main__":
    main()
