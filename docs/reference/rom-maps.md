# ROM Map Extraction

> Consolidated from `tools/rom_maps/README.md` and `rom_maps.md`.

This page is the canonical guide to the ROM-map parser: what it decodes out of
the Pokémon Crystal cartridge, what it takes as input, what it writes as output,
how the decode is validated, and what "runtime collision field" means for
training. It replaces the two earlier documents it was merged from — the
tool-local `tools/rom_maps/README.md` (the shipped, working description) and the
root `rom_maps.md` (the original spec and wish-list, largely unimplemented). Each
section below notes which source it came from.

## What the parser decodes

*Source: `tools/rom_maps/README.md`.*

The parser extracts every map in the ROM as a machine-learning-friendly
**block-ID matrix**, read **directly from the ROM** — no emulator rendering, no
screenshots. Each map becomes a plain grid of metatile (block) IDs, `height` rows
× `width` columns, plus metadata. The same parser also backs the runtime
egocentric collision field used in training (see
[Collision: what "runtime collision field" means](#collision-what-runtime-collision-field-means)).

The original spec additionally called for rendered PNGs, `.npy`/`.json` exports,
per-cell feature dicts, and a world/warp graph. None of those are implemented; they
are tracked under [Not implemented](#not-implemented) rather than described as
available.

## Inputs

*Source: `tools/rom_maps/README.md`.*

| Input | Role |
|---|---|
| `emu_files/Pokemon - Crystal Version.gbc` | The ROM itself. Crystal (U/E) v1.0, md5 `9f2922b235a5eeb78d65594e82ef5dde`. |
| `tools/rom_maps/map_constants.asm` | Bundled from `pret/pokecrystal`. Source of truth for map names, per-group ordering, and the authoritative `(width, height)` for all 388 maps; used to cross-check the ROM decode. |

The files in the tool directory:

| File | Purpose |
|---|---|
| `rom_map_extractor.py` | Core parser. Reads the ROM byte-for-byte and decodes every map's block matrix + metadata. |
| `extract_maps.py` | CLI. Writes one CSV per map plus an `index.csv`. |
| `validate_with_pyboy.py` | Loads the ROM in PyBoy and checks the parser against live WRAM. |
| `map_constants.asm` | Bundled from `pret/pokecrystal`; cross-check source for names and dimensions. |

## Outputs

*Source: `tools/rom_maps/README.md`.*

```
tools/rom_maps/output/
    index.csv                         # one row per map (metadata)
    maps/
        24_04_NEW_BARK_TOWN.csv     # <group>_<num>_<NAME>.csv, block-ID matrix
        24_03_ROUTE_29.csv
        ...                         # 388 maps
```

`index.csv` columns: `group, number, name, width, height, tileset, environment,
border_block, landmark, music, connections, blocks_bank, blocks_addr,
dims_match_constants, csv_file`.

The exporter emits **block IDs only** — there is no collision column in these
CSVs, and no PNG or `.npy` output. That is a deliberate scope decision, not an
oversight; see [Collision](#collision-what-runtime-collision-field-means).

## Usage

*Source: `tools/rom_maps/README.md`.*

From the repo root:

```bash
# Extract all maps to CSV (writes tools/rom_maps/output/)
python tools/rom_maps/extract_maps.py

# Validate the parser against live PyBoy WRAM
python tools/rom_maps/validate_with_pyboy.py
```

Programmatic use:

```python
from tools.rom_maps.rom_map_extractor import RomMapExtractor
ex = RomMapExtractor("emu_files/Pokemon - Crystal Version.gbc")
gm = ex.decode_map(24, 4)              # New Bark Town
print(gm.width, gm.height, gm.blocks)  # blocks[y][x] -> block ID
maps = ex.decode_all()                 # all 388 maps
```

## How it works

*Source: `tools/rom_maps/README.md`.*

Map data lives in a pointer chain lifted from the `pret/pokecrystal` disassembly:

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

## The 388-map claim

*Source: `tools/rom_maps/README.md`; independently confirmed on disk.*

The parser decodes **388 maps** — every map defined in `map_constants.asm` across
all 26 map groups. The count is not asserted by the code alone: `index.csv` and
the 388 per-map CSVs under `tools/rom_maps/output/maps/` are both present in the
working tree, and correctness check 1 below verifies `388/388` decoded dimensions
against the assembly constants. Note that this includes `_BETA` maps, which are
unused in the retail game but still occupy an index in their group (see
[`ram-mapping.md`](ram-mapping.md)).

## Validation against live WRAM

*Source: `tools/rom_maps/README.md`.*

The parser is verified three independent ways:

1. **Structural** — the ROM-decoded `(width, height)` for all **388/388** maps
   matches the dimensions in `map_constants.asm`. A wrong table anchor or struct
   offset would desync dimensions across the whole ROM.
2. **Anchor** — the 52-byte `MapGroupPointers` table (26 × 2 bytes) ends exactly
   where `MapGroup_Olivine` begins (`0x4034`), and every group pointer lands in
   the `0x4000–0x7FFF` bank window.
3. **Live WRAM (PyBoy)** — PyBoy is used as an independent **ground-truth
   check**, not as the data source: it loads the same ROM, restores a save state,
   and confirms the parser's output against the block buffer the game engine
   itself decompresses into WRAM. For the save-state map (`PLAYERS_HOUSE_2F`) the
   pristine ROM block matrix aligns into the engine's `wOverworldMapBlocks` buffer
   at stride `width+6` with a 3-block connection border. Blocks match byte-for-byte
   except two cells that the engine overlays at runtime — the customisable bedroom
   **decorations** — confirming the parser reads the clean, pristine map.

## Collision: what "runtime collision field" means

*Source: `tools/rom_maps/README.md` ("Not yet implemented"), clarified.*

**The exporter CSVs do not emit collision data.** There is no collision column in
`index.csv` or in any per-map CSV, and there is no static collision dataset on
disk to train against.

The collision field is instead **decoded at runtime**, in
`PoliwhiRL/environment/rom_collision.py`. On each step it reads the agent's
current map and coordinates, walks the same ROM pointer chain described above for
the *local* neighbourhood around the player, and yields an **egocentric** window
of walk/blocked values scaled to `[0,1]` — "what is solid around me right now",
not a whole-world walkability table. It is validated against the engine's live
four-direction collision bytes (the `C2FA-C2FD` reads in
`PoliwhiRL/environment/RAM.py`), so the decoded field is checked against what the
game itself believes about the tiles the player is touching.

The practical consequence: collision is a *live observation feature*, and there is
no offline collision map to mine for path planning. Anything that needs the whole
world's walkability in one array must wait for the exported mask below.

## Not implemented

*Sources: `tools/rom_maps/README.md` ("Not yet implemented") and the unimplemented
remainder of the root `rom_maps.md` spec. Both were consolidated here; each item
is genuinely absent, not merely undocumented.*

Identified but intentionally left out to keep every shipped output verified. The
first three can be added on the same pointer chain:

- **Exported collision / walkable mask** — the CSV exporter emits no collision
  data; training decodes it at runtime instead (see above).
- **PNG rendering** — needs metatile definitions + 2bpp tile graphics + palettes
  per tileset to rasterise each block matrix.
- **World / warp graph** — the `connections` byte + per-map warp/event tables.
- **Tileset, metatile-definition, tile-graphics and palette decoding** — the
  original spec's pipeline stages after block data; the shipped parser stops at
  block IDs.
- **Per-cell feature export** (`block_id`, `collision`, `terrain_type`, `warp`,
  `grass`, `water`, `ledge`, `elevation`) and `.npy` / `.json` output formats —
  specified in the original root spec, never built. CSV is the only shipped format.
- **NPC spawn, signpost, item and encounter-region extraction** — optional in the
  original spec, not implemented.

## Related

- [`ram-mapping.md`](ram-mapping.md) — the live-RAM half of the same reference:
  addresses, event flags, and what the running game reports.
- [`../subsystems/environment.md`](../subsystems/environment.md) — how the
  runtime collision field and RAM vector are assembled into the observation.
- [`../subsystems/rewards.md`](../subsystems/rewards.md) — how novel territory
  (cells, maps, banks) is paid for.
- [`../subsystems/exploration.md`](../subsystems/exploration.md) — Go-Explore,
  which consumes the frontier this geometry defines.
- [`../reference/manual-states.md`](manual-states.md) — save-state ground truth
  used alongside the PyBoy check.
- [`../../AGENTS.md`](../../AGENTS.md) — codebase map; the tool is described there
  as the one non-post-hoc member of `tools/`.
