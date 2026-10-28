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
import heapq  # noqa: E402
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


# Path-sense walkability (see RomCollisionField.distance_field).
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


# Written into the local field for a cell an NPC sprite is standing on. Not a
# real collision value: distinct from walls (0x07) and off-map (0xFF) so the
# policy can tell "something is standing there" (a blocker that may move or
# vanish after a story flag, or a trainer) from permanent terrain.
NPC_OCCUPIED_VALUE = 0xFE


def _wram(memory, addr):
    """Read WRAM bank 1 for 0xD000-0xDFFF (switchable bank; see
    RAM.get_memory_value). Falls back to a plain index for test doubles."""
    if 0xD000 <= addr < 0xE000:
        try:
            return memory[1, addr]
        except (TypeError, KeyError, IndexError):
            pass
    return memory[addr]


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

    def warps(self, group: int, number: int) -> list:
        """Warp events (doors, stairs, cave mouths) of a map from the ROM
        map-events table (``db y, x, warp_id; map_id group, number``), as
        [(x, y, dest_group, dest_number)]."""
        key = ("warps", group, number)
        out = self._map_cache.get(key)
        if out is None:
            rom = self._ex.rom
            try:
                hdr = self._ex.group_pointer(group) + (number - 1) * MAP_HEADER_LEN
                attr = rom_offset(rom[hdr], rom[hdr + 3] | (rom[hdr + 4] << 8))
                events = rom_offset(rom[attr + 6], rom[attr + 9] | (rom[attr + 10] << 8))
                out = []
                for i in range(rom[events + 2]):
                    y, x, _wid, g, n = rom[events + 3 + 5 * i : events + 8 + 5 * i]
                    out.append((x, y, g, n))
            except Exception:
                out = []
            self._map_cache[key] = out
        return out

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

    def _edges(self, off, tileset, node):
        """Walkable moves out of ``node`` = ((bank, num), x, y): yields
        (next_node, dx, dy, step) where step is 2 for a ledge hop. Crosses
        overworld connections only into maps sharing ``tileset`` (the
        collision table loaded in WRAM belongs to it). Warp tiles (doors,
        stairs) are steppable but the search does not follow the warp."""
        mp, x, y = node
        gm, _ts, conns = self._meta(*mp)
        # Ledges (verified live 2026-10-28, Route 30 gap and Route 29): a
        # ledge cell is walked onto like floor from any side but the one it
        # drops away from (you cannot climb it), and the hop is decided by
        # the cell you STAND on: on a ledge, pressing its direction jumps two
        # cells, over whatever is next (usually a wall). The old model hopped
        # from the cell before the ledge, which closed real paths (the way
        # round Youngster Mikey at the Route 30 gap) and opened none.
        hops = _LEDGES.get(self._value(gm, off, x, y), ())
        for name, dx, dy in _STEPS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < gm.width * 2 and 0 <= ny < gm.height * 2:
                if (dx, dy) in hops:
                    lx, ly = x + 2 * dx, y + 2 * dy
                    lv = self._value(gm, off, lx, ly)
                    if lv in _WALKABLE or _is_warp(lv) or lv in _LEDGES:
                        yield (mp, lx, ly), dx, dy, 2
                    continue
                v = self._value(gm, off, nx, ny)
                if (
                    v in _WALKABLE
                    or _is_warp(v)
                    or (v in _LEDGES and (-dx, -dy) not in _LEDGES[v])
                ):
                    yield (mp, nx, ny), dx, dy, 1
                continue
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
            if v in _WALKABLE or _is_warp(v):
                yield nxt, dx, dy, 1

    def distance_field(self, memory, env_vars, is_target, blocked=(), max_nodes=8000):
        """Path distance (in tiles) from every node of the player's
        walkable region to the nearest node satisfying
        ``is_target(bank, num, x, y)``, or None when unavailable.

        Returns ``(dist, moves)``: ``dist`` maps node -> tiles to the nearest
        target (absent = no target reachable from it); ``moves`` maps node ->
        [(next_node, dx, dy, step)] for descending the field. The region is
        everything reachable from the player across same-tileset overworld
        connections (bounded by ``max_nodes``); ``blocked`` nodes (cells an
        NPC is standing on) are impassable. One-way ledges are respected by
        searching the reversed move graph out from the targets.

        It encodes no route — only "how far is the nearest ground that
        qualifies as frontier", so the same field generalises to any map."""
        if env_vars.get("script_active", False):
            return None
        try:
            start_map = (int(env_vars["map_bank"]), int(env_vars["map_num"]))
            tileset = self._meta(*start_map)[1]
        except Exception:
            return None
        off = self._collision_offset(memory)
        blocked = set(blocked)
        start = (start_map, int(env_vars["X"]), int(env_vars["Y"]))
        moves = {}
        queue = deque([start])
        moves[start] = None
        while queue and len(moves) < max_nodes:
            node = queue.popleft()
            out = [e for e in self._edges(off, tileset, node) if e[0] not in blocked]
            moves[node] = out
            for nxt, _dx, _dy, _step in out:
                if nxt not in moves:
                    moves[nxt] = None
                    queue.append(nxt)
        moves = {n: m for n, m in moves.items() if m is not None}
        reverse = {}
        for node, out in moves.items():
            for nxt, _dx, _dy, step in out:
                reverse.setdefault(nxt, []).append((node, step))
        # Dijkstra over the reversed graph; steps are 1 or 2 tiles.
        dist = {}
        heap = []
        for node in moves:
            if is_target(node[0][0], node[0][1], node[1], node[2]):
                dist[node] = 0
                heap.append((0, node))
        heapq.heapify(heap)
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist.get(node, d):
                continue
            for prev, step in reverse.get(node, ()):
                nd = d + step
                if prev in moves and nd < dist.get(prev, nd + 1):
                    dist[prev] = nd
                    heapq.heappush(heap, (nd, prev))
        return dist, moves

    @staticmethod
    def descent_direction(field, node, lookahead=8):
        """Unit vector (egocentric map axes) following ``field`` downhill
        ``lookahead`` moves from ``node``, or None when the node has no
        reachable target or already stands on one."""
        dist, moves = field
        if dist.get(node) in (None, 0):
            return None
        gx = gy = 0
        for _ in range(lookahead):
            d = dist.get(node)
            if not d:
                break
            best = min(
                (e for e in moves.get(node, ()) if e[0] in dist),
                key=lambda e: dist[e[0]],
                default=None,
            )
            if best is None or dist[best[0]] >= d:
                break
            node, dx, dy, step = best
            gx += dx * step
            gy += dy * step
        norm = math.hypot(gx, gy)
        if norm < 1e-8:
            return None
        return [gx / norm, gy / norm]

    def _collision_offset(self, memory) -> int:
        bank = _wram(memory, W_TILESET_COLLISION_BANK)
        addr = _wram(memory, W_TILESET_COLLISION_ADDR) | (
            _wram(memory, W_TILESET_COLLISION_ADDR + 1) << 8
        )
        if addr < 0x4000:
            return addr
        return bank * 0x4000 + (addr - 0x4000)

    def local_field(
        self, memory, env_vars, radius: int, npc_tiles=()
    ) -> list[float]:
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
        occupied = {(int(x), int(y)) for x, y in npc_tiles}
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
                if (x, y) in occupied:
                    val = NPC_OCCUPIED_VALUE
                out.append(val / 255.0)
        return out
