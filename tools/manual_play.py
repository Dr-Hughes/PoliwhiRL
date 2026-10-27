#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Play Pokémon Crystal by hand from a save state, recording what you do.

NOT used by training — a standalone tool for producing a human reference
trajectory (e.g. the walk from "Learned To Catch" to Violet City) to compare
against the policy's recorded episodes and find where the agent gets lost.

Opens a PyBoy SDL2 window on ``--state``. Every frame where the held buttons,
map or tile position change is appended to ``trajectory.csv`` in the output
directory. A save state is written on every first entry to a map
(``map_<bank>_<num>.state``) and on exit (``final.state``). Close the window
(or press Esc) to finish; Ctrl-C in the terminal also saves cleanly.

Controls (PyBoy defaults): arrows = D-pad, A = a, S = b, Enter = Start,
Backspace = Select, Space = hold for fast-forward.

Usage (from repo root):

    uv run python tools/manual_play.py --state manual_play/start_post_f66_route29.state
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from pyboy import PyBoy

ROM = "emu_files/Pokemon - Crystal Version.gbc"

# RAM addresses — see PoliwhiRL/environment/RAM.py and docs/reference/ram-mapping.md.
MAP_BANK, MAP_NUM, X, Y = 0xDCB5, 0xDCB6, 0xDCB8, 0xDCB7
BATTLE_TYPE = 0xD22D
EVENT_FLAGS = 0xDA72
JOYPAD_DOWN = 0xFFA4  # hJoypadDown, verified against button_press()
_BUTTON_BITS = (
    (0x01, "a"), (0x02, "b"), (0x04, "select"), (0x08, "start"),
    (0x10, "right"), (0x20, "left"), (0x40, "up"), (0x80, "down"),
)
# Story flags worth tagging in the log (status.md's ladder).
TRACKED_FLAGS = (30, 31, 39, 65, 66)


def _buttons(byte):
    return "+".join(name for bit, name in _BUTTON_BITS if byte & bit)


def _flag(mem, n):
    return (mem[EVENT_FLAGS + n // 8] >> (n % 8)) & 1


def _save_state(pyboy, path):
    with open(path, "wb") as fh:
        pyboy.save_state(fh)


def main():
    ap = argparse.ArgumentParser(description="Record a manual playthrough from a save state.")
    ap.add_argument("--state", required=True, help="save state to start from")
    ap.add_argument("--out-dir", default=None,
                    help="default: manual_play/run_<timestamp>")
    ap.add_argument("--rom", default=ROM)
    ap.add_argument("--scale", type=int, default=4, help="window scale")
    args = ap.parse_args()

    out = Path(args.out_dir or f"manual_play/run_{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)

    pyboy = PyBoy(args.rom, window="SDL2", scale=args.scale, sound_emulated=False)
    with open(args.state, "rb") as fh:
        pyboy.load_state(fh)
    pyboy.set_emulation_speed(1)
    mem = pyboy.memory

    fields = ["frame", "t_sec", "bank", "map", "x", "y", "buttons",
              "battle", *[f"f{n}" for n in TRACKED_FLAGS]]
    log = open(out / "trajectory.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(fields)

    seen_maps, map_entries = set(), []
    last = None
    frame, t0 = 0, time.time()
    try:
        while pyboy.tick():
            frame += 1
            bank, num = mem[MAP_BANK], mem[MAP_NUM]
            row = (bank, num, mem[X], mem[Y], _buttons(mem[JOYPAD_DOWN]),
                   mem[BATTLE_TYPE], *[_flag(mem, n) for n in TRACKED_FLAGS])
            if row != last:
                writer.writerow([frame, round(time.time() - t0, 2), *row])
                last = row
            if (bank, num) not in seen_maps:
                seen_maps.add((bank, num))
                map_entries.append({"frame": frame, "bank": bank, "map": num,
                                    "x": mem[X], "y": mem[Y]})
                _save_state(pyboy, out / f"map_{bank}_{num}.state")
                print(f"[frame {frame}] entered map {bank}:{num} at "
                      f"({mem[X]}, {mem[Y]})")
    except KeyboardInterrupt:
        pass
    finally:
        _save_state(pyboy, out / "final.state")
        log.close()
        (out / "summary.json").write_text(json.dumps({
            "start_state": str(args.state),
            "frames": frame,
            "seconds": round(time.time() - t0, 1),
            "final": {"bank": mem[MAP_BANK], "map": mem[MAP_NUM],
                      "x": mem[X], "y": mem[Y]},
            "map_entries": map_entries,
        }, indent=2))
        pyboy.stop(save=False)  # never touch the ROM's .ram file
        print(f"Saved trajectory + states to {out}/")


if __name__ == "__main__":
    main()
