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
from tools.rom_maps.rom_map_extractor import RomMapExtractor  # noqa: E402

# WRAM pointer to the active tileset's collision table (engine-maintained).
W_TILESET_COLLISION_BANK = 0xD1DF
W_TILESET_COLLISION_ADDR = 0xD1E0  # 2 bytes, little-endian

# Off-map / undecodable cells read as this raw collision value. 0xFF is a
# blocked ("wall") value, which is the safe default for an affordance signal —
# beyond the known map is treated as not-walkable.
_OFF_MAP_VALUE = 0xFF


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
