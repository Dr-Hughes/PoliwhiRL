# Operations tooling

What this page is for: the standalone diagnostic tools that read persisted run
artefacts — `tools/training_health_report.py` and `tools/world_map.py` in
depth, then the supporting scripts (`png_to_video.py`, `llm_game_debug.py`,
`tools/rom_maps/**`). None of these are imported by training; they are
post-hoc.

**Sources**: `tools/training_health_report.py`, `tools/world_map.py`,
`tools/png_to_video.py`, `tools/llm_game_debug.py`, `tools/rom_maps/README.md`,
`PoliwhiRL/environment/rom_collision.py` (by reference).

## `tools/training_health_report.py`

Offline health check over a checkpoint's `info.pth`; explicitly *not* used by
training, and needs no emulator and no model weights — every check reads data
already logged in `episode_data` / `visit_archive`
(`tools/training_health_report.py:3-11`). It exists so a stalled run can be
triaged without hand-authoring a new script each time
(`training_health_report.py:5-9`).

```bash
# from the repo root
python tools/training_health_report.py "Training Outputs/open_world/Checkpoints/info.pth"
python tools/training_health_report.py <info.pth> \
  --episode-length 8192 --probe-episode-length 8192 --long-probe-episode-length 24576
```

Invocation and options: `training_health_report.py:37-40` (usage),
`:509-527` (argparse). Defaults: entropy band `--band-low 0.6` /
`--band-high 1.2` (`:523-524`), `--coef-floor 1e-3` (`:525`),
`--coef-ceiling 0.1` (`:526`). `--episode-length` may be omitted — it is then
inferred as `max(episode_lengths)` (`:533-535`). Torch is imported lazily so
`--help` stays dependency-light (`:529`), and the load itself uses
`weights_only=False` (`:531`).

### Report sections

Emitted in this order by `main()` (`training_health_report.py:544-552`), after
the header line of episode count, `best_reward` and archive cell/map counts
(`:538-542`):

| Section | Contents | Trigger / failure text | Source |
|---|---|---|---|
| `=== Optimisation ===` | mean base-policy entropy against the band, assisted behaviour entropy, band-occupancy fraction, coef pinned at floor/ceiling, plateau-boost fraction, the four KL families, replay-rejection by quarter, clip fraction, first/latest LR | `WARNING: entropy spends most of its time outside the target band` (`:118-119`); `WARNING: coefficient frequently pinned at the floor` (`:120-121`); `WARNING: replay rejection rate increased over the run` (`:164-169`); `CRITICAL: … epoch(s) breached the hard KL limit and were rolled back` (`:170-174`) | `:84-184` |
| `=== Reward mix ===` | per-channel first-¼ vs last-¼ averages, separately for honest true-start and Go-Explore seeded episodes (`:292-293`), plus milestone/frontier totals and the delivery accounting block | `WARNING: milestone reward isn't decaying relative to frontier` (`:257-265`); `WARNING: delivered reward does not reconcile with source accounting` (`:254-255`); the two `OK:` branches (`:266-282`) | `:187-295` |
| `=== Archive growth ===` | archive size start→end and growth in the final quarter | `WARNING: archive has been flat for the last quarter … exploration plateau` (`:309-314`) | `:298-314` |
| `=== Discovery log ===` | count of all-episode firsts split honest/seeded, most recent honest entry and how many episodes ago, counts by type | `WARNING: no new discovery in a while relative to run length` (`:343-347`) | `:317-347` |
| `=== Time-to-frontier ===` | episodes with ≥1 flag fire, median / P90 / MAX deepest fire step, MAX as a fraction of `episode_length` | `TRIGGER:` once the deepest fire step lands past ~60% of the episode budget (`:376-383`) | `:350-385` |
| `=== Story checkpoints ===` | per-checkpoint honest vs seeded reach counts and last-honest episode (titles via `checkpoint_title`, gated by `is_recordable_checkpoint`), first-honest replay statuses, persisted frontier-pool size | `no durable checkpoints reached` (`:478-480`) | `:457-506` |
| `=== Probe ladder ===` | last three events for regular, assisted-regular, long-horizon and assisted-long probes | `NOTE: rung '<k>' is higher on the long-horizon probe … the short probe's horizon is hiding it` when long > short + 0.15 (`:433-438`); `no long-horizon probe data` (`:439-443`) | `:388-454` |

Read the median and MAX rows of Time-to-frontier as different quantities: the
median is dominated by the shallowest recurring milestone, while the MAX is the
run's actual frontier (`training_health_report.py:363-372`).

## `tools/world_map.py`

Offline, post-hoc tile and discovery-order analysis; also not used by training
(`tools/world_map.py:3-6`). It reconstructs two things purely from logged data
(`world_map.py:8-20`):

1. A per-map walkability/warp/encounter `WorldMap` inferred **empirically from
   observed movement** — a directional press that changes position makes that
   edge walkable, one that doesn't is a wall, and a change in `warp_number`
   tags the tile the player was standing on as a warp. There is no PyBoy tile
   decoding (`world_map.py:8-14`; implementation `:159-200`).
2. A chronological discovery timeline loaded from
   `episode_data["discovery_log"]` — every genuine run-wide first
   (flag/map/pokedex/level/key_item), in order, tagged honest vs seeded
   (`world_map.py:15-20`; provenance tag `:310-318`).

```bash
# from the repo root
python tools/world_map.py --run-dir "Training Outputs/open_world/Runs" --map 24 3
python tools/world_map.py --discovery-log "Training Outputs/open_world/Checkpoints/info.pth"
```

Invocation: `world_map.py:38-41` (usage), `:321-332` (argparse — `--run-dir`
`:323`, `--discovery-log` `:331`). Two recording formats are handled
transparently (`world_map.py:22-36`): `trajectory.csv`, the current default for
periodic full-episode recordings with one row per step
(`step,x,y,map,bank,room,battlestate,playerstate,warp,btn,reward`)
(`world_map.py:23-28`), and per-step PNG filenames whose metadata is encoded in
the name
(`step_N_x_X_y_Y_map_M_bank_B_room_R_battlestate_…_reward_R.png`, regex
`world_map.py:49-56`). Recordings predating the `warp` field are treated as
"no warp" rather than failing (`world_map.py:36`).

ASCII legend: `.` confirmed walkable (player has stood here), `#` known wall
(bumped into, never stood on), ` ` undetermined (`world_map.py:217-219`).

**What a failure looks like:** an all-`#` frontier around a cluster of `.` is
an inferred wall boundary, not decoded geometry — this tool **cannot decode tile
semantics**, it reports movement empirics only (`world_map.py:8-11`). Missing
input is loud, not silent: a `--run-dir` with no CSVs and no matching PNG names
yields an empty map, so cross-check against the ROM export below before
concluding the map is walled in.

## Supporting scripts

| Path | Purpose | Invocation | Source |
|---|---|---|---|
| `tools/png_to_video.py` | Stitch a folder of PNGs into a GIF **and** an MP4, overlaying button labels; requires Pillow and `imageio[ffmpeg]`, and aborts with a message if either is missing | `python tools/png_to_video.py <folder> [-o PREFIX] [--fps 10]` | `png_to_video.py:3,11-21,131-143`; overlay step `:164`; outputs `<prefix>.gif` / `.mp4` (`:160-161`) |
| `tools/llm_game_debug.py` | LLM-driven debugger: a thin vision-model driver on top of the *real* `PyBoyEnvironment`, so action mapping, frame timing (90 frames/action, 15-frame hold) and RAM decoding match training exactly | `python tools/llm_game_debug.py --verify`; `python tools/llm_game_debug.py --steps 40 --goal "leave the bedroom" --model <id> [--display] [--thinking]`; also `--manual` for human play | `llm_game_debug.py:3-24` (docstring incl. usage), env import `:43-44`, flags `:553-570` |
| `tools/llm_debug.db` | SQLite log of RAM states, shared by `--manual` and the LLM driver | artefact, written by the tool | `llm_game_debug.py:52-53` |
| `tools/rom_maps/rom_map_extractor.py` | ROM parser: decodes every map's block-ID matrix + metadata directly from the ROM (all 388 maps); also used by the runtime egocentric collision field | `from tools.rom_maps.rom_map_extractor import RomMapExtractor` | `tools/rom_maps/README.md:1-5,15,51-57` |
| `tools/rom_maps/extract_maps.py` | CLI exporter: one CSV per map plus `index.csv` into `tools/rom_maps/output/` | `python tools/rom_maps/extract_maps.py` | `tools/rom_maps/README.md:16,24-30,35-47` |
| `tools/rom_maps/validate_with_pyboy.py` | Ground-truth check: loads the ROM in PyBoy, restores a state, compares the parser against live WRAM | `python tools/rom_maps/validate_with_pyboy.py` | `tools/rom_maps/README.md:17,28-29,91-96` |
| `tools/rom_maps/map_constants.asm` | Bundled `pret/pokecrystal` map names and authoritative dimensions for all 388 maps | reference data | `tools/rom_maps/README.md:18,78-79,85-87` |

**"Runtime collision field"** means the egocentric collision mask decoded at
run time in `PoliwhiRL/environment/rom_collision.py` (file present;
`tools/rom_maps/README.md:103-106`) and validated against the engine's live
four-direction collision bytes. The exporter CSVs do **not** emit collision data
yet (`tools/rom_maps/README.md:103-104`). PNG rendering and the world/warp
graph are the other documented-not-yet-implemented items
(`tools/rom_maps/README.md:107-109`).

## Known gaps

- `tools/llm_debug_frames/` and `tools/ram_flag_deltas.jsonl` exist as
  artefacts; their producing code paths were not traced in this pass, so their
  exact schema is undocumented here.
- `tools/llm_game_debug.py` hard-codes an LLM base URL and default model
  (`llm_game_debug.py:49-50`); whether that endpoint is reachable, and which
  model is current, is environment-specific and not verified.
- `png_to_video.py` overlays button labels (`png_to_video.py:38-45,164`); the
  exact overlay layout was not rendered or inspected.
- The 388-map count and the three-way correctness argument are quoted from
  `tools/rom_maps/README.md:81-96`; neither the extractor nor the validator was
  executed in this pass.
- `world_map.py` encounter-rate semantics
  (`tools/world_map.py:8-9`) were not compared against battle-state fields.
- The walkability-inference implementation range `tools/world_map.py:159-200` is
  carried over from prior reconnaissance; only the docstring (`:8-14`) and the
  legend (`:217-219`) were re-read in this pass.
