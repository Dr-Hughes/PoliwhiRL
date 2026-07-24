# -*- coding: utf-8 -*-
"""CLI: extract every Pokémon Crystal map from the ROM to CSV.

Reads map block data directly from the ROM (see rom_map_extractor.py) and writes:

  <out>/maps/<group>_<num>_<NAME>.csv   one CSV per map, the block-ID matrix
                                        (height rows x width columns)
  <out>/index.csv                       one row per map with metadata

Usage (from repo root):
  python tools/rom_maps/extract_maps.py
  python tools/rom_maps/extract_maps.py --rom "emu_files/Pokemon - Crystal Version.gbc" --out tools/rom_maps/output
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.rom_maps.rom_map_extractor import GameMap, RomMapExtractor  # noqa: E402

DEFAULT_ROM = "emu_files/Pokemon - Crystal Version.gbc"
DEFAULT_OUT = "tools/rom_maps/output"


def write_map_csv(gm: GameMap, maps_dir: Path) -> Path:
    """Write one map's block-ID matrix as CSV (height rows x width cols)."""
    fname = f"{gm.group:02d}_{gm.number:02d}_{gm.name}.csv"
    path = maps_dir / fname
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(gm.blocks)
    return path


def write_index_csv(maps: list[GameMap], out_dir: Path) -> Path:
    """Write index.csv: one row of metadata per map."""
    path = out_dir / "index.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "group",
                "number",
                "name",
                "width",
                "height",
                "tileset",
                "environment",
                "border_block",
                "landmark",
                "music",
                "connections",
                "blocks_bank",
                "blocks_addr",
                "dims_match_constants",
                "csv_file",
            ]
        )
        for gm in maps:
            writer.writerow(
                [
                    gm.group,
                    gm.number,
                    gm.name,
                    gm.width,
                    gm.height,
                    gm.tileset,
                    gm.environment,
                    gm.border_block,
                    gm.landmark,
                    gm.music,
                    gm.connections,
                    f"0x{gm.blocks_bank:02x}",
                    f"0x{gm.blocks_addr:04x}",
                    gm.dims_match_constants,
                    f"maps/{gm.group:02d}_{gm.number:02d}_{gm.name}.csv",
                ]
            )
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rom", default=DEFAULT_ROM, help="path to the Crystal .gbc ROM")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    args = ap.parse_args()

    ex = RomMapExtractor(args.rom)
    maps = ex.decode_all()

    out_dir = Path(args.out)
    maps_dir = out_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    for gm in maps:
        write_map_csv(gm, maps_dir)
    index_path = write_index_csv(maps, out_dir)

    mismatches = [m for m in maps if not m.dims_match_constants]
    total_blocks = sum(m.width * m.height for m in maps)
    print(f"Extracted {len(maps)} maps -> {maps_dir}/  ({total_blocks} blocks total)")
    print(f"Index: {index_path}")
    print(
        f"Dimension cross-check vs map_constants.asm: "
        f"{len(maps) - len(mismatches)}/{len(maps)} OK"
    )
    if mismatches:
        print("  MISMATCHES:")
        for m in mismatches:
            print(f"    g{m.group} n{m.number} {m.name}: {m.width}x{m.height}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
