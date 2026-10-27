# -*- coding: utf-8 -*-
"""Egocentric ground-truth walkability field, decoded from the ROM.

The engine already hands the policy the immediate N/E/S/W collision bytes
(0xC2FA-FD). This generalises that to a ``(2R+1)x(2R+1)`` egocentric window of
the *same* raw collision values, read from the current tileset's collision
table in the ROM. It is fully egocentric and tileset-invariant (a wall reads
the same everywhere), carries no history, and is recomputed every step — so it
adds local "what can I walk onto around me" affordance without teaching a route
or leaking any training-wide/global knowledge.

Decode recipe (see the ``rom-collision-decode`` memory / RAM_MAPPING notes):
- The collision-table pointer for the loaded tileset is resolved by the engine
  into WRAM: bank @ 0xD1DF, address @ 0xD1E0 (little-endian). No fragile static
  Tilesets-struct offset is needed.
- Per cell ``(x, y)`` in RAM tile coords (2x the block dims):
      value = rom[file_off + block_id(x//2, y//2) * 4 + (y%2)*2 + (x%2)]
  where ``block_id`` comes from the ROM-decoded map matrix and ``file_off`` is
  the flat ROM offset of the WRAM-resolved collision pointer.

Validated at 98.5% against live engine collision bytes across a walk; the only
misses are map-boundary tiles where the engine reads its connection border
while the pristine ROM has the map's own edge block (the same border effect
``tools/rom_maps/validate_with_pyboy.py`` documents for the block matrix).
"""
from __future__ import annotations

import sys
from pathlib import Path

# tools/ is a namespace package (no __init__.py); make the repo root importable
# so we can reuse the single source of truth for the ROM map decode.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from collections import deque  # noqa: E402
import math  # noqa: E402

from tools.rom_maps.rom_map_extractor import (  # noqa: E402
    MAP_HEADER_LEN,
    RomMapExtractor,
    rom_offset,
)

# WRAM pointer to the active tileset's collision table (engine-maintained).
W_TILESET_COLLISION_BANK = 0xD1DF
W_TILESET_COLLISION_ADDR = 0xD1E0  # 2 bytes, little-endian

# Off-map / undecodable cells read as this raw collision value. 0xFF is a
# blocked ("wall") value, which is the safe default for an affordance signal —
# beyond the known map is treated as not-walkable.
_OFF_MAP_VALUE = 0xFF


# Path-sense walkability (see RomCollisionField.frontier_path_direction).
# Floor, tall grass and the grass-edge value are walkable; warp tiles (doors,
# stairs, cave mouths) are steppable but not expanded through — the search
# stays on overworld connections. Ledges are one-way hops.
_WALKABLE = frozenset({0x00, 0x14, 0x18})
_LEDGES = {
    0xA0: ((1, 0),),
    0xA1: ((-1, 0),),
    0xA2: ((0, -1),),
    0xA3: ((0, 1),),
    0xA4: ((0, 1), (1, 0)),
    0xA5: ((0, 1), (-1, 0)),
}
_STEPS = (("N", 0, -1), ("S", 0, 1), ("W", -1, 0), ("E", 1, 0))


def _is_warp(value):
    return value is not None and 0x70 <= value <= 0x7F


class RomCollisionField:
    """Builds the egocentric collision-value window for the current map."""

    def __init__(self, rom_path: str):
        self._ex = RomMapExtractor(rom_path)
        self._map_cache: dict[tuple[int, int], object] = {}

    def _decode(self, group: int, number: int):
        key = (group, number)
        gm = self._map_cache.get(key)
        if gm is None:
            gm = self._ex.decode_map(group, number)
            self._map_cache[key] = gm
        return gm

    def _meta(self, group: int, number: int):
        """(GameMap, tileset id, {dir: (map, yoff, xoff)}) for one map; the
        connection structs follow the 12-byte attribute header."""
        key = ("meta", group, number)
        meta = self._map_cache.get(key)
        if meta is None:
            rom = self._ex.rom
            gm = self._decode(group, number)
            hdr = self._ex.group_pointer(group) + (number - 1) * MAP_HEADER_LEN
            attr = rom_offset(rom[hdr], rom[hdr + 3] | (rom[hdr + 4] << 8))
            mask = rom[attr + 11]
            ptr = attr + 12
            conns = {}
            for bit, name in ((8, "N"), (4, "S"), (2, "W"), (1, "E")):
                if mask & bit:
                    c = rom[ptr : ptr + 12]
                    s8 = lambda v: v - 256 if v > 127 else v  # noqa: E731
                    conns[name] = ((c[0], c[1]), s8(c[8]), s8(c[9]))
                    ptr += 12
            meta = (gm, rom[hdr + 1], conns)
            self._map_cache[key] = meta
        return meta

    def _value(self, gm, off, x, y):
        if not (0 <= y < gm.height * 2 and 0 <= x < gm.width * 2):
            return None
        slot = off + gm.blocks[y // 2][x // 2] * 4 + (y % 2) * 2 + (x % 2)
        rom = self._ex.rom
        return rom[slot] if 0 <= slot < len(rom) else None

    def frontier_path_direction(
        self, memory, env_vars, is_visited, max_nodes=4000, lookahead=8
    ):
        """Unit vector (egocentric map axes) along the shortest walkable path
        to the nearest tile NOT yet visited this episode, or None.

        The local frontier sense only sees ~12 tiles; once an episode has
        walked a whole region (after a story flag sends the player back over
        it) it reads (0, 0) everywhere and nothing says where fresh ground is.
        This BFS walks the ROM collision grid across overworld map
        connections (only into maps sharing the current tileset, whose
        collision table is the one loaded in WRAM) and returns the direction
        of the path ``lookahead`` tiles ahead — around walls, not through
        them. It encodes no route, only "where is unvisited ground".

        ``is_visited(bank, num, x, y)`` reports this episode's visit history.
        NPC sprites are not modelled (the CNN sees them)."""
        if env_vars.get("script_active", False):
            return None
        try:
            start_map = (int(env_vars["map_bank"]), int(env_vars["map_num"]))
            meta = self._meta(*start_map)
        except Exception:
            return None
        tileset = meta[1]
        off = self._collision_offset(memory)
        start = (start_map, int(env_vars["X"]), int(env_vars["Y"]))
        # node -> (parent, global x, global y): global coordinates accumulate
        # unit steps, so crossing a connection keeps displacements consistent.
        seen = {start: (None, 0, 0)}
        queue = deque([start])
        goal = None
        while queue and len(seen) < max_nodes:
            node = queue.popleft()
            (mp, x, y) = node
            _p, gx, gy = seen[node]
            if node is not start:
                gm = self._meta(*mp)[0]
                if self._value(gm, off, x, y) in _WALKABLE and not is_visited(
                    mp[0], mp[1], x, y
                ):
                    goal = node
                    break
            gm, _ts, conns = self._meta(*mp)
            for name, dx, dy in _STEPS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < gm.width * 2 and 0 <= ny < gm.height * 2:
                    v = self._value(gm, off, nx, ny)
                    if v in _WALKABLE or _is_warp(v):
                        nxt, step = (mp, nx, ny), 1
                    elif v in _LEDGES and (dx, dy) in _LEDGES[v]:
                        nxt, step = (mp, nx + dx, ny + dy), 2
                        if self._value(gm, off, nxt[1], nxt[2]) not in _WALKABLE:
                            continue
                    else:
                        continue
                else:
                    conn = conns.get(name)
                    if conn is None:
                        continue
                    target, yoff, xoff = conn
                    try:
                        tgm, tts, _c = self._meta(*target)
                    except Exception:
                        continue
                    if tts != tileset:
                        continue
                    if name in ("N", "S"):
                        nxt = (target, x + xoff, yoff)
                    else:
                        nxt = (target, xoff, y + yoff)
                    v = self._value(tgm, off, nxt[1], nxt[2])
                    if not (v in _WALKABLE or _is_warp(v)):
                        continue
                    step = 1
                if nxt not in seen:
                    seen[nxt] = (node, gx + dx * step, gy + dy * step)
                    queue.append(nxt)
        if goal is None:
            return None
        path = [goal]
        while seen[path[-1]][0] is not None:
            path.append(seen[path[-1]][0])
        path.reverse()  # start .. goal
        _p, gx, gy = seen[path[min(lookahead, len(path) - 1)]]
        norm = math.hypot(gx, gy)
        if norm < 1e-8:
            return None
        return [gx / norm, gy / norm]

    def _collision_offset(self, memory) -> int:
        bank = memory[W_TILESET_COLLISION_BANK]
        addr = memory[W_TILESET_COLLISION_ADDR] | (
            memory[W_TILESET_COLLISION_ADDR + 1] << 8
        )
        if addr < 0x4000:
            return addr
        return bank * 0x4000 + (addr - 0x4000)

    def local_field(self, memory, env_vars, radius: int) -> list[float]:
        """Egocentric ``(2R+1)^2`` row-major list of collision values in
        ``[0, 1]`` (raw value / 255), centred on the player.

        Returns all-zeros during scripted overlays or when the map cannot be
        decoded — mirroring the visited-mask/frontier-direction contract, where
        a stale player position has nothing meaningful to report.
        """
        n = (2 * radius + 1) ** 2
        if env_vars.get("script_active", False):
            return [0.0] * n
        try:
            group = int(env_vars["map_bank"])
            number = int(env_vars["map_num"])
            gm = self._decode(group, number)
        except Exception:
            return [0.0] * n

        rom = self._ex.rom
        off = self._collision_offset(memory)
        x0, y0 = int(env_vars["X"]), int(env_vars["Y"])
        blocks = gm.blocks
        out: list[float] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = x0 + dx, y0 + dy
                by, bx = y // 2, x // 2
                if 0 <= by < len(blocks) and 0 <= bx < len(blocks[by]):
                    bid = blocks[by][bx]
                    slot = off + bid * 4 + (y % 2) * 2 + (x % 2)
                    val = rom[slot] if 0 <= slot < len(rom) else _OFF_MAP_VALUE
                else:
                    val = _OFF_MAP_VALUE
                out.append(val / 255.0)
        return out
