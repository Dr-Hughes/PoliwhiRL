# -*- coding: utf-8 -*-
"""Static Pokémon Crystal map extractor.

Reads map data **directly from the Crystal ROM** (no emulator rendering, no
screenshots) and exposes every map as a block-ID matrix, exactly as laid out in
the ``pret/pokecrystal`` disassembly. This is the parser half of the system
described in ``rom_maps.md``; ``validate_with_pyboy.py`` loads the same ROM in
PyBoy and checks the parser's output against live WRAM ground truth.

Data chain (all offsets verified against the retail Crystal v1.0 ROM,
md5 9f2922b235a5eeb78d65594e82ef5dde):

    MapGroupPointers (ROM 0x25:4000, file 0x94000)
        -> per-group table of 9-byte map headers  (the `map` macro)
            -> map attributes                      (the `map_attributes` macro)
                -> block data (width*height metatile IDs)

Struct layouts (from pokecrystal, verbatim field order):

  map header (9 bytes, data/maps/maps.asm `map` macro):
      db attr_bank, tileset, environment
      dw attr_ptr            ; -> <name>_MapAttributes, in bank attr_bank
      db landmark, music
      dn phone_service, time_of_day
      db fishing_group

  map attributes (data/maps/attributes.asm `map_attributes` macro):
      db border_block
      db height, width       ; NB: height first, in *blocks*
      db blocks_bank
      dw blocks_ptr          ; -> <name>_Blocks, in bank blocks_bank
      db scripts_bank
      dw scripts_ptr
      dw events_ptr
      db connections         ; bitmask; connection structs follow (unused here)

  block data:
      width*height bytes, each a metatile (block) ID for that map's tileset.

Names, per-group ordering and the authoritative (width, height) for every map
come from the bundled ``map_constants.asm`` (copied from pokecrystal). The ROM
decode is cross-checked against those dimensions for all 388 maps, which is the
primary correctness proof — a wrong anchor or struct offset would desync the
dimensions across the whole ROM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# --- ROM constants (retail Crystal v1.0 / v1.1; map data is identical) ------ #
ROM_MD5_V10 = "9f2922b235a5eeb78d65594e82ef5dde"
BANK_SIZE = 0x4000
NUM_MAP_GROUPS = 26
MAP_HEADER_LEN = 9  # MAP_LENGTH in map_constants
# MapGroupPointers @ 0x25:4000 -> file offset. All MapGroup_* header tables and
# the pointer table live in bank 0x25.
MAP_GROUP_POINTERS_BANK = 0x25
MAP_GROUP_POINTERS_ADDR = 0x4000

_CONSTANTS_FILE = Path(__file__).with_name("map_constants.asm")


def rom_offset(bank: int, addr: int) -> int:
    """Convert a (bank, 16-bit CPU address) pair to a flat ROM file offset.

    Home bank (0x0000-0x3FFF) maps to itself; a switchable-bank address
    (0x4000-0x7FFF) maps into ``bank``.
    """
    if addr < BANK_SIZE:
        return addr
    return bank * BANK_SIZE + (addr - BANK_SIZE)


@dataclass
class MapConst:
    """One ``map_const`` entry: the disassembly's source of truth."""

    group: int
    number: int
    name: str  # e.g. "OLIVINE_POKECENTER_1F"
    width: int
    height: int


@dataclass
class GameMap:
    """A fully decoded map: metadata + the block-ID matrix."""

    group: int
    number: int
    name: str
    width: int
    height: int
    tileset: int
    environment: int
    border_block: int
    landmark: int
    music: int
    connections: int
    blocks_bank: int
    blocks_addr: int
    # blocks[y][x] -> metatile (block) ID; len == height, each row len == width
    blocks: list[list[int]] = field(default_factory=list)
    # Cross-check flag: did the ROM-decoded (w,h) match map_constants?
    dims_match_constants: bool = True

    @property
    def id_pair(self) -> tuple[int, int]:
        return (self.group, self.number)


def parse_map_constants(path: Path = _CONSTANTS_FILE) -> list[MapConst]:
    """Parse the bundled ``map_constants.asm`` into ordered MapConst records.

    Groups are numbered 1..26 in ``newgroup`` order; maps restart at 1 for each
    group in ``map_const`` order. This matches MapGroupPointers indexing.
    """
    text = path.read_text()
    newgroup_re = re.compile(r"^\s*newgroup\s+(\w+)", re.M)
    mapconst_re = re.compile(r"^\s*map_const\s+(\w+)\s*,\s*(\d+)\s*,\s*(\d+)", re.M)

    out: list[MapConst] = []
    group = 0
    number = 0
    # Walk the file line by line to preserve newgroup/map_const interleaving.
    for line in text.splitlines():
        gm = newgroup_re.match(line)
        if gm:
            group += 1
            number = 0
            continue
        mm = mapconst_re.match(line)
        if mm:
            number += 1
            name, w, h = mm.group(1), int(mm.group(2)), int(mm.group(3))
            out.append(MapConst(group, number, name, w, h))
    return out


class RomMapExtractor:
    """Decodes every map's block matrix straight out of the ROM bytes."""

    def __init__(self, rom_path: str | Path):
        self.rom_path = Path(rom_path)
        self.rom = self.rom_path.read_bytes()
        self.constants = parse_map_constants()
        # index: (group, number) -> MapConst
        self._const_by_id = {(c.group, c.number): c for c in self.constants}
        # group -> list of MapConst in map-number order
        self._by_group: dict[int, list[MapConst]] = {}
        for c in self.constants:
            self._by_group.setdefault(c.group, []).append(c)

    # -- low-level reads ---------------------------------------------------- #
    def _u8(self, off: int) -> int:
        return self.rom[off]

    def _u16(self, off: int) -> int:
        return self.rom[off] | (self.rom[off + 1] << 8)

    def group_pointer(self, group: int) -> int:
        """File offset of the map-header table for ``group`` (1-based)."""
        table = rom_offset(MAP_GROUP_POINTERS_BANK, MAP_GROUP_POINTERS_ADDR)
        ptr = self._u16(table + 2 * (group - 1))
        return rom_offset(MAP_GROUP_POINTERS_BANK, ptr)

    # -- map decode --------------------------------------------------------- #
    def decode_map(self, group: int, number: int) -> GameMap:
        """Decode one map (1-based group/number) into a GameMap with blocks."""
        const = self._const_by_id.get((group, number))
        # Map header (9 bytes) at group table + (number-1)*9
        hdr = self.group_pointer(group) + (number - 1) * MAP_HEADER_LEN
        attr_bank = self._u8(hdr + 0)
        tileset = self._u8(hdr + 1)
        environment = self._u8(hdr + 2)
        attr_ptr = self._u16(hdr + 3)
        landmark = self._u8(hdr + 5)
        music = self._u8(hdr + 6)

        # Map attributes
        attr = rom_offset(attr_bank, attr_ptr)
        border_block = self._u8(attr + 0)
        height = self._u8(attr + 1)
        width = self._u8(attr + 2)
        blocks_bank = self._u8(attr + 3)
        blocks_ptr = self._u16(attr + 4)
        # attr+6 scripts_bank, +7 scripts_ptr(2), +9 events_ptr(2)
        connections = self._u8(attr + 11)

        # Block data: width*height metatile IDs, row-major (y outer, x inner).
        blk_off = rom_offset(blocks_bank, blocks_ptr)
        n = width * height
        raw = self.rom[blk_off : blk_off + n]
        blocks = [list(raw[y * width : (y + 1) * width]) for y in range(height)]

        dims_ok = True
        name = const.name if const else f"GROUP_{group}_MAP_{number}"
        if const is not None:
            dims_ok = width == const.width and height == const.height

        return GameMap(
            group=group,
            number=number,
            name=name,
            width=width,
            height=height,
            tileset=tileset,
            environment=environment,
            border_block=border_block,
            landmark=landmark,
            music=music,
            connections=connections,
            blocks_bank=blocks_bank,
            blocks_addr=blocks_ptr,
            blocks=blocks,
            dims_match_constants=dims_ok,
        )

    def decode_all(self) -> list[GameMap]:
        """Decode every map named in map_constants, in (group, number) order."""
        return [self.decode_map(c.group, c.number) for c in self.constants]
