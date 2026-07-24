# Pokémon Crystal ROM map extractor

Extracts every map in the Pokémon Crystal ROM as a machine-learning-friendly
**block-ID matrix**, read **directly from the ROM** — no emulator rendering, no
screenshots. The parser is also used by the runtime egocentric collision field.

PyBoy is used as an independent **ground-truth check**: it loads the same ROM,
restores a save state, and confirms the parser's output against the block buffer
the game engine itself decompresses into WRAM.

## Files

| File | Purpose |
|------|---------|
| `rom_map_extractor.py` | Core parser. Reads the ROM byte-for-byte and decodes every map's block matrix + metadata. |
| `extract_maps.py` | CLI. Writes one CSV per map plus an `index.csv`. |
| `validate_with_pyboy.py` | Loads the ROM in PyBoy and checks the parser against live WRAM. |
| `map_constants.asm` | Bundled from `pret/pokecrystal`. Source of truth for map names + dimensions; used to cross-check the ROM decode. |

## Usage

From the repo root:

```bash
# Extract all maps to CSV (writes tools/rom_maps/output/)
python tools/rom_maps/extract_maps.py

# Validate the parser against live PyBoy WRAM
python tools/rom_maps/validate_with_pyboy.py
```

Output layout:

```
tools/rom_maps/output/
    index.csv                         # one row per map (metadata)
    maps/
        24_04_NEW_BARK_TOWN.csv       # <group>_<num>_<NAME>.csv, block-ID matrix
        24_03_ROUTE_29.csv
        ...                           # 388 maps
```

Each map CSV is a plain grid of metatile (block) IDs, `height` rows ×
`width` columns. `index.csv` columns:
`group, number, name, width, height, tileset, environment, border_block,
landmark, music, connections, blocks_bank, blocks_addr, dims_match_constants,
csv_file`.

Programmatic use:

```python
from tools.rom_maps.rom_map_extractor import RomMapExtractor
ex = RomMapExtractor("emu_files/Pokemon - Crystal Version.gbc")
gm = ex.decode_map(24, 4)          # New Bark Town
print(gm.width, gm.height, gm.blocks)   # blocks[y][x] -> block ID
maps = ex.decode_all()             # all 388 maps
```

## How it works

The ROM is Crystal (U/E) v1.0, md5 `9f2922b235a5eeb78d65594e82ef5dde`. Map data
lives in a pointer chain lifted from the `pret/pokecrystal` disassembly:

```
MapGroupPointers (ROM 0x25:4000, file 0x94000)   # 26 little-endian group pointers
    -> per-group table of 9-byte map headers      # data/maps/maps.asm `map` macro
        db attr_bank, tileset, environment
        dw attr_ptr  -------------------------.
        db landmark, music; dn phone,tod; db fishgroup
    -> map attributes  <----------------------'   # data/maps/attributes.asm
        db border_block; db height, width          #   `map_attributes` macro
        db blocks_bank; dw blocks_ptr  ---.
        db script_bank; dw script_ptr; dw events_ptr; db connections
    -> block data  <----------------------'
        width*height bytes, each a metatile (block) ID
```

Names, per-group ordering, and the authoritative `(width, height)` for all 388
maps come from `map_constants.asm`.

## Correctness

The parser is verified three independent ways:

1. **Structural** — the ROM-decoded `(width, height)` for all **388/388** maps
   matches the dimensions in `map_constants.asm`. A wrong table anchor or struct
   offset would desync dimensions across the whole ROM.
2. **Anchor** — the 52-byte `MapGroupPointers` table (26 × 2 bytes) ends exactly
   where `MapGroup_Olivine` begins (`0x4034`), and every group pointer lands in
   the `0x4000–0x7FFF` bank window.
3. **Live WRAM (PyBoy)** — for the save-state map (`PLAYERS_HOUSE_2F`), the
   pristine ROM block matrix aligns into the engine's `wOverworldMapBlocks`
   buffer at stride `width+6` with a 3-block connection border. Blocks match
   byte-for-byte except two cells that the engine overlays at runtime (the
   customizable bedroom **decorations**) — confirming the parser reads the clean,
   pristine map.

## Not yet implemented (documented next steps)

The following are identified but intentionally left out to keep every shipped
output verified. They can be added on the same pointer chain:

- **Exported collision / walkable mask** — the CSV exporter does not yet emit
  collision data. Runtime training already decodes an egocentric ROM collision
  field in `PoliwhiRL/environment/rom_collision.py` and validates it against the
  engine's live four-direction collision bytes.
- **PNG rendering** — needs metatile definitions + 2bpp tile graphics +
  palettes per tileset to rasterize each block matrix.
- **World / warp graph** — the `connections` byte + per-map warp/event tables.
