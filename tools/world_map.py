#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline, post-hoc tile/discovery-order analysis.

NOT used by training — this is a standalone diagnostic tool that
reconstructs two things after a run, purely from data already logged:

1. A per-map walkability/warp/encounter-rate ``WorldMap``, inferred
   empirically from recorded-episode PNG filenames (no PyBoy tile
   decoding — Pokémon Crystal has no semantic tile wrapper, so this
   project builds its own structure from observed movement instead: if a
   directional press changes the player's position, that edge is
   walkable; if it doesn't, it's a wall; a change in ``warp_number``
   tags the tile the player was just standing on as a warp).
2. A flat, chronologically-ordered discovery timeline, loaded from a
   checkpoint's ``episode_data["discovery_log"]`` (see
   ``Rewards._discoveries_this_episode`` / ``VecPPOAgent._commit_episode``)
   — every genuine run-wide first-ever milestone fire (flag/map/pokedex/
   level/key_item), in the order it actually happened.

Recorded PNG filenames encode one step each:
``step_{n}_x_{x}_y_{y}_map_{num}_bank_{bank}_room_{room}_battlestate_{s}
_playerstate_{s}_warp_{n}_btn_{btn}_reward_{r}.png``
(see ``PoliwhiRL/utils/visuals.py::record_step`` and
``PoliwhiRL/environment/gym_env.py::save_step_img_data``). Older
recordings made before the ``warp`` field was added won't have it —
this tool treats a missing warp as "no warp" rather than failing.

Usage (from repo root):

    python tools/world_map.py --run-dir "Training Outputs/03_freeform_to_mrpokemon/Runs" --map 24 3
    python tools/world_map.py --discovery-log "Training Outputs/03_freeform_to_mrpokemon/Checkpoints/info.pth"
"""
import argparse
import re
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path

_FILENAME_RE = re.compile(
    r"step_(?P<step>\d+)"
    r"_x_(?P<x>-?\d+)_y_(?P<y>-?\d+)"
    r"_map_(?P<map>\d+)_bank_(?P<bank>\d+)_room_(?P<room>\d+)"
    r"_battlestate_(?P<battlestate>\w+?)_playerstate_(?P<playerstate>\w+?)"
    r"(?:_warp_(?P<warp>\d+))?"
    r"_btn_(?P<btn>\w*)_reward_(?P<reward>-?[\d.]+)\.png$"
)

_DIR_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


StepRecord = namedtuple(
    "StepRecord",
    ["step", "map_bank", "map_num", "x", "y", "action", "battle_type",
     "player_state", "warp"],
)


def parse_filename(name):
    """Parse one recorded-step filename into a StepRecord, or None if it
    doesn't match (e.g. a non-step file). battle_type/player_state are the
    string labels already baked into the filename (battlestate/playerstate),
    not raw RAM ints — good enough for this tool's purposes (only whether
    a battle is active matters for walkability inference; player_state is
    carried through but not gated on, since it's a movement *mode* — walk/
    bike/skate/surf — not walking-vs-not)."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    warp = m.group("warp")
    return StepRecord(
        step=int(m.group("step")),
        map_bank=int(m.group("bank")),
        map_num=int(m.group("map")),
        x=int(m.group("x")),
        y=int(m.group("y")),
        action=m.group("btn"),
        battle_type=m.group("battlestate"),
        player_state=m.group("playerstate"),
        warp=int(warp) if warp is not None else None,
    )


@dataclass
class Tile:
    discovered: bool = False
    known_wall: bool = False
    walkable: dict = field(default_factory=dict)  # {"up"/"down"/"left"/"right": True/False}
    visit_count: int = 0
    warp: bool = False
    encounter_visits: int = 0
    encounter_hits: int = 0

    @property
    def encounter_rate(self):
        if self.encounter_visits == 0:
            return 0.0
        return self.encounter_hits / self.encounter_visits


class WorldMap:
    """Empirically-built per-tile map, keyed by (map_bank, map_num, x, y).
    Never touches PyBoy tile data — everything here is inferred purely
    from observed player transitions (see module docstring)."""

    def __init__(self):
        self.tiles = {}

    def _tile(self, key):
        return self.tiles.setdefault(key, Tile())

    def update(self, prev, cur):
        """prev may be None (first step of an episode / a fresh map)."""
        cur_key = (cur.map_bank, cur.map_num, cur.x, cur.y)
        tile = self._tile(cur_key)
        tile.discovered = True
        tile.visit_count += 1
        tile.encounter_visits += 1
        if cur.battle_type not in (None, "none"):
            tile.encounter_hits += 1

        if prev is None:
            return
        if prev.map_bank != cur.map_bank or prev.map_num != cur.map_num:
            return  # a map transition isn't a within-map walkability edge

        if prev.warp is not None and cur.warp is not None and prev.warp != cur.warp:
            self._tile((prev.map_bank, prev.map_num, prev.x, prev.y)).warp = True

        # Only infer walkability from a genuine free-walking directional
        # press — a stationary result during a battle isn't a wall bump,
        # it's just not a movement attempt. Mirrors the battle_type gate
        # Rewards uses for the same purpose (see rewards.py's stagnation/
        # blocked-direction accounting) rather than player_state: recorded
        # player_state is one of PLAYER_STATE_LABELS's values ("walk",
        # "bike", "skate", "surf") and never the string "walking", so a
        # player_state-based gate here would silently never fire.
        if prev.battle_type not in (None, "none") or cur.battle_type not in (None, "none"):
            return
        direction = prev.action if prev.action in _DIR_DELTA else None
        if direction is None:
            return

        prev_tile = self._tile((prev.map_bank, prev.map_num, prev.x, prev.y))
        moved = (cur.x, cur.y) != (prev.x, prev.y)
        prev_tile.walkable[direction] = bool(moved)
        if not moved:
            dx, dy = _DIR_DELTA[direction]
            wall_key = (prev.map_bank, prev.map_num, prev.x + dx, prev.y + dy)
            self._tile(wall_key).known_wall = True

    def load_episode(self, steps):
        """Feed one episode's ordered StepRecord list through update()."""
        prev = None
        for cur in steps:
            self.update(prev, cur)
            prev = cur

    def bounds(self, map_bank, map_num):
        keys = [k for k in self.tiles if k[0] == map_bank and k[1] == map_num]
        if not keys:
            return None
        xs = [k[2] for k in keys]
        ys = [k[3] for k in keys]
        return min(xs), max(xs), min(ys), max(ys)

    def render_ascii(self, map_bank, map_num):
        """'.' = confirmed walkable (player has stood here), '#' = known
        wall (bumped into, never stood on), ' ' = undetermined."""
        b = self.bounds(map_bank, map_num)
        if b is None:
            return "(no data for this map)"
        x0, x1, y0, y1 = b
        rows = []
        for y in range(y0, y1 + 1):
            row = []
            for x in range(x0, x1 + 1):
                tile = self.tiles.get((map_bank, map_num, x, y))
                if tile is None:
                    row.append(" ")
                elif tile.discovered:
                    row.append(".")
                elif tile.known_wall:
                    row.append("#")
                else:
                    row.append(" ")
            rows.append("".join(row))
        return "\n".join(rows)

    def local_patch(self, map_bank, map_num, cx, cy, radius=3):
        """Small grid around (cx, cy) — same character scheme as
        render_ascii, with 'P' marking the center. Intended as the shape a
        future observation feature would take if this is ever wired into
        live training; not used there yet."""
        rows = []
        for y in range(cy - radius, cy + radius + 1):
            row = []
            for x in range(cx - radius, cx + radius + 1):
                if (x, y) == (cx, cy):
                    row.append("P")
                    continue
                tile = self.tiles.get((map_bank, map_num, x, y))
                if tile is None:
                    row.append(" ")
                elif tile.discovered:
                    row.append(".")
                elif tile.known_wall:
                    row.append("#")
                else:
                    row.append(" ")
            rows.append("".join(row))
        return "\n".join(rows)

    def maps_seen(self):
        return sorted({(k[0], k[1]) for k in self.tiles})


def load_run_from_pngs(run_dir):
    """Walk a Runs/ directory (one subfolder per recorded episode) and
    build a WorldMap from every episode found. Steps within an episode are
    sorted by step number before feeding — recorded PNGs are written as
    they occur but this guards against any filesystem ordering surprises."""
    world = WorldMap()
    run_dir = Path(run_dir)
    for ep_dir in sorted(p for p in run_dir.rglob("*") if p.is_dir()):
        records = []
        for png in ep_dir.glob("*.png"):
            rec = parse_filename(png.name)
            if rec is not None:
                records.append(rec)
        if not records:
            continue
        records.sort(key=lambda r: r.step)
        world.load_episode(records)
    return world


def load_discovery_log(info_pth_path):
    """Load episode_data["discovery_log"] from a checkpoint's info.pth and
    return it sorted chronologically (episode, then step)."""
    import torch
    info = torch.load(info_pth_path, map_location="cpu", weights_only=False)
    log = info.get("episode_data", {}).get("discovery_log", [])
    return sorted(log, key=lambda r: (r["episode"], r["step"]))


def format_discovery_timeline(discovery_log):
    lines = []
    for rec in discovery_log:
        lines.append(
            f"ep {rec['episode']:>6} (rollout {rec['rollout_idx']:>4}) "
            f"step {rec['step']:>5}  {rec['type']:<14} {rec['key']}"
        )
    return "\n".join(lines)


def _main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", help="Path to a stage's Runs/ directory")
    ap.add_argument("--map", nargs=2, type=int, metavar=("BANK", "NUM"),
                     help="Render this (map_bank, map_num) as ASCII")
    ap.add_argument("--discovery-log", help="Path to a checkpoint's info.pth")
    args = ap.parse_args()

    if args.run_dir and args.map:
        world = load_run_from_pngs(args.run_dir)
        bank, num = args.map
        print(f"Maps with any data: {world.maps_seen()}")
        print()
        print(world.render_ascii(bank, num))
    elif args.run_dir:
        world = load_run_from_pngs(args.run_dir)
        print(f"Maps with any data: {world.maps_seen()}")

    if args.discovery_log:
        log = load_discovery_log(args.discovery_log)
        print(format_discovery_timeline(log))


if __name__ == "__main__":
    _main()
