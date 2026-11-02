# Environment & observation pipeline

What this page is for: the contract between the training loop and the PyBoy-backed
environment — what `step()` and `reset()` hand back, exactly what is inside the
observation (screen image plus the RAM feature vector, and how that vector is
composed), the action set and how the ROM-derived collision field is produced,
how episode recordings are written, and what travels over the worker process
boundary in `terminal_info`. Read this when a tensor shape, a feature value, or a
metric that originates in the environment looks wrong; the reward arithmetic itself
lives in [rewards.md](rewards.md) and the seeding machinery in
[exploration.md](exploration.md).

**Sources** — `PoliwhiRL/environment/gym_env.py`, `RAM.py`,
`rom_collision.py`, `vec_env.py`, `PoliwhiRL/utils/visuals.py`, re-read against
the working tree on branch `flag31-breakthrough-and-tuning`. JSON values carried
from the earlier reconnaissance notes (untracked working-tree notes under
`.doc-scratch/`, never committed, therefore not citable from a fresh clone). `AGENTS.md` is the
project map; where prose and
code disagree, the code wins (`AGENTS.md:17`).

## `step()` / `reset()` contract

| item | behaviour | citation |
|---|---|---|
| `step(action)` returns | `(observation, self._fitness, self.done, truncated)` | `PoliwhiRL/environment/gym_env.py:1266` |
| `truncated` | `bool(self.reward_calculator.truncated)` — the env does not decide truncation itself, the reward calculator does | `gym_env.py:1265` |
| `step()` order of operations | buffer pre-checkpoint state → apply action → compute fitness → maybe capture a Go-Explore frontier state / a reverse-curriculum ring → build observation → maybe record | `gym_env.py:1252-1266` |
| reward is computed | inside `_calculate_fitness()`, which reads RAM into `env_vars` and calls `Rewards.calculate_reward(env_vars, self.button)`; a `reward_done` promotes `self.done` | `gym_env.py:1288-1297` |
| `reset()` returns | the observation dict (`self.get_observation()`), after re-loading the state bytes | `gym_env.py:1523` |
| `reset()` closes the recording | any open `trajectory.csv` handle is closed and cleared | `gym_env.py:1488-1490` |
| `reset()` per-episode reset | fresh `Rewards` (handed the seed provenance via `reward_config["episode_seeded"]`), zeroed capture sets, `steps = 0`, `episode += 1`, `render` forced true | `gym_env.py:1491-1518` |
| domain balancer | constructed once and handed to each fresh `Rewards`; `DomainRewardBalancer.scale(...)` is called per domain inside the reward calculation, and the balancer's state is broadcast/read over the worker pipe | `gym_env.py:1494-1498`, `rewards.py:1151-1155`, `vec_env.py:314-317` |
| startup ordering | fitness is computed **before** the observation is returned, so the goal-target fields reflect the no-op startup step | `gym_env.py:1519-1523` |

## Observation structure

`get_observation()` returns a two-key dict (`gym_env.py:1299-1339`):

| key | dtype / shape | source | citation |
|---|---|---|---|
| `"image"` | `uint8 (C, H, W)` — the rendered screen, always | `get_screen_image()` | `gym_env.py:1306`, `:1549-1583` |
| `"ram"` | `float32 (RAM_OBS_DIM,)` | `_build_ram_vector(...)` | `gym_env.py:1315-1338`, shape via `ram_observation_shape()` `gym_env.py:1272-1273` |

There is **no alternative image path**. The `vision` config key that used to
select between the rendered screen and a tile-ID grid read from
`pyboy.game_area()` is gone from every config and from the code:
`get_game_area()` is deleted, `self.render` is unconditionally `True` (set in
`__init__`, `gym_env.py:645`, and re-asserted on every `reset()`, `:1518`), and
`output_shape()` simply returns `get_screen_image().shape` (`:1268-1270`).

### The resize filter is BOX, and that is load-bearing

`get_screen_image` downsamples by `scaling_factor` with `Image.BOX`
(`gym_env.py:1556-1572`). The filter is hardcoded, **not** a config key, and has
a single source of truth: the module constants `OBS_RESIZE_FILTER` and
`OBS_RESIZE_FILTER_NAME = "box"` (`gym_env.py:604-615`). `OBS_RESIZE_FILTER` is
what the resize call uses; `OBS_RESIZE_FILTER_NAME` is what gets recorded in
`info.pth`'s `run_config` (`vec_ppo_agent.py:3077`) and stamped into every SIL
demo `.npz` as `obs_resize_filter`, so a future filter change cannot silently
leave stale demos behind. `tools/preflight.py` fails if a demo's stamp is absent
or differs from the env's filter.

Why BOX rather than the `Image.LANCZOS` it replaced — measured at
`scaling_factor` 0.5, where one Game Boy tile is 4×4 scaled px and 160→80 /
144→72 are both exact halvings:

| filter | cross-tile leakage |
|---|---|
| `Image.BOX` | **zero** — an exact 2×2 area average, so every output pixel lies inside exactly one tile |
| `Image.LANCZOS` | 0.75 tile on all four sides, at 5.490% of peak |

Tile isolation is precisely what `TileGridCNN`'s patchify stem assumes (one
output vector per tile, `kernel == stride == 4`); LANCZOS let a tile's pixels
reach into its neighbours' patch vectors. See
[models.md](models.md#image-branch-tilegridcnn--image-only).

**This changed every observation.** Pre-change checkpoints, demos and recordings
are not comparable with anything produced after it. The eight SIL demo `.npz`
files under `Training Outputs/llm_demos/` were regenerated under BOX — only their
`images` array changed, while `ram`, `actions`, `rewards` and `target_flag` are
bit-identical and the 1060 total steps are unchanged — and they now carry the
`obs_resize_filter` stamp. The `rc/*.state` reverse-curriculum rings are PyBoy
save states replayed live, so they were never stale.

`use_grayscale` and `scaling_factor` are the remaining shape knobs, both read
inside `get_screen_image` (`gym_env.py:1553-1554`). No pixel scaling is applied
at the env boundary: the array leaves as `uint8` 0–255 (`:1583`) and the
division by 255 happens once inside the encoder (`GameBoy.py:112`).

`TileGridCNN` consumes the image only and has no `ram` input; only
`PPOTransformer` fuses the RAM vector
(`PoliwhiRL/models/PPO/PPOTransformer.py:205-212`) — see
[models.md](models.md) rather than re-deriving it here.

## RAM vector composition

`RAM_FEATURE_KEYS` is the append-only contract; `RAM_OBS_DIM = len(RAM_FEATURE_KEYS)`
(`gym_env.py:304-305`), with named positions exported as `RAM_FEATURE_INDEX` so
downstream code never hard-codes offsets (`gym_env.py:311-316`). The vector is
built from one `ram.get_variables()` read per step (`gym_env.py:1308`) and
assembled by `_build_ram_vector` (`gym_env.py:389`).

Base block (`_BASE_RAM_FEATURE_KEYS`, `gym_env.py:66-186`) then the appended
egocentric blocks, in this exact order:

| block | keys | count | citation |
|---|---|---|---|
| position / identity | `x`, `y`, facing one-hot (`facing_up/down/left/right`), `map_num`, `map_bank`, `room`, `warp_number` | 8 | `gym_env.py:66-79` |
| party / economy / dex | `party_size`, `party_level`, `party_hp`, `party_exp`, `money`, `pokedex_seen`, `pokedex_owned` | 7 | `gym_env.py:80-86` |
| raw collision cross | `collision_down/up/left/right` | 4 | `gym_env.py:87-90` |
| exploration + progress counters | `explored_tile_count`, `maps_visited_this_episode`, `n_pokedex_goals_completed`, `n_level_goals_completed`, `n_flag_goals_completed` | 5 | `gym_env.py:91-101` |
| battle / badges / player state one-hots | `battle_none/wild/trainer`, `johto_badges_count`, `player_state_walking/battle/cycling/surfing` | 8 | `gym_env.py:102-118` |
| items / time / audio | `key_items_count`, `game_hour`, `bgm_id` | 3 | `gym_env.py:119-121` |
| enemy context | `enemy_hp` (log1p), `enemy_hp_ratio` | 2 | `gym_env.py:122-130` |
| move PP | `player_pp1..4` (normalised `/64`) | 4 | `gym_env.py:131-137` |
| script / UI state one-hots | `script_active`, 5 `ui_state_*`, 5 `map_handler_*` | 11 | `gym_env.py:138-157` |
| map-goal rung | `n_map_goals_completed` | 1 | `gym_env.py:158-166` |
| recent-map history | 6 × `(recent_map_bank_i, recent_map_num_i)`, oldest first, zero-padded | 12 | `gym_env.py:167-185` |
| **visited mask** | `visited_local_0..24` | `(2·2+1)²` = 25 | `gym_env.py:187-200` (`VISITED_MASK_RADIUS = 2` `:196`) |
| **frontier direction** | `frontier_dir_x`, `frontier_dir_y`, `frontier_local_saturation` | 3 | `gym_env.py:201-212` |
| **ROM collision field** | `collision_local_0..24` | `(2·2+1)²` = 25 | `gym_env.py:213-225` (`COLLISION_FIELD_RADIUS = 2` `:221`) |
| **stagnation clock** | `stagnation_clock`, appended last in the base block | 1 | `gym_env.py:226-237` |
| derived story flags | `_DERIVED_FLAG_KEYS`, appended after the raw base features | `len(DERIVED_FLAG_TABLE)` | `gym_env.py:238-241`, table `:64` |

Notes that matter when reading a value:

- `script_active` / `ui_state_text_box` / the battle one-hots are binary
  indicators (one-hot slots in the RAM vector).
- The visited mask and frontier direction are **per-episode and egocentric** by
  design; they replaced run-wide global-visit-count features (`gym_env.py:187-195`).
- `stagnation_clock` is the only feature that de-aliases "just arrived here"
  (≈0) from "stuck here for hundreds of steps" (≈1) (`gym_env.py:226-235`;
  helper `Rewards.stagnation_fraction()`, see [rewards.md](rewards.md)).
- The builder validates its own contract: if the assembled base vector's length
  is not `len(_BASE_RAM_FEATURE_KEYS)` it raises rather than silently
  mis-aligning (`gym_env.py:497-500`); absent mask / direction / field inputs are
  zero-filled at the right width (`gym_env.py:471-493`).

## Address groups (`RAM.py`)

Counts per group, pivotal addresses only — the full table lives in [`docs/reference/ram-mapping.md`](../reference/ram-mapping.md)
(`AGENTS.md:40-43`).

| group | count | pivotal addresses | citation |
|---|---|---|---|
| identity / position | 9 | X `0xDCB8`, Y `0xDCB7`, map_num `0xDCB6`, map_bank `0xDCB5`, room `0xD148`, warp `0xDCB4`, facing `0xD4DE`, player_state `0xD95D`, received `0xCF60` | `RAM.py:10-24`, `:54`, `:59` |
| story flags | 256-byte window = 2048 bits | `0xDA72`–`0xDB71` (`story_flags_size` computed) | `RAM.py:26-32` |
| battle | 8 | `battle_type 0xD22D`, enemy HP `0xD216`/`0xD217`, enemy max HP `0xD218`/`0xD219`, PP `0xC634`–`0xC637` | `RAM.py:57`, `:64-80` |
| party | base + slot stride | party base `0xDCDF`, party size `0xDCD7`, stride `0x30` per slot | `RAM.py:16-17`, `:201-203` |
| items / economy / badges | 4 | badges `0xD857`, key items `0xD8BC`, money `0xD84E`, coins `0xD855/56` | `RAM.py:58`, `:60`, `:15`, `:105-106` |
| pokédex | 2 windows | owned `0xDE99`–`0xDEB8`, seen `0xDEB9`–`0xDED8` (popcount per byte) | `RAM.py:18-19`, `:217-235` |
| audio / time | 4 | `bgm_id 0xC2A9`, `game_hour 0xD4B7`, day-of-week `0xD4B6`, minute `0xD4B8` | `RAM.py:61-62`, `:95-96` |
| WRAM bounds / tile buffer | 3 | WRAM `0xC000`–`0xDFFF`, screen tiles `0xC4A0`–`0xC607` | `RAM.py:20-21`, `:34-36` |
| evaluator-only debug windows | 8 windows | `audio_c2a0`, `joypad_c2cd`, `battle_d200`, `player_obj_d4d0`, `map_d145`, `map_event_dc40`, `text_cf00`, `script_d430` — explicitly "candidate regions to investigate", not stable contracts | `RAM.py:121-156` |
| extended read-only probes | 17 | legendary capture markers, roaming legendary positions, item-pocket/repel/blue-card counters (debug evaluator only) | `RAM.py:82-119` |

### Two documented fixes worth knowing before you trust a byte

- **Facing.** `player_direction` was `0xD357`, which is always 0 in Crystal, so
  the facing one-hot in the observation was permanently all-zeros — a dead
  feature. It is now `0xD4DE`, verified empirically to track facing on both turns
  and walks (`RAM.py:44-54`). The raw byte encodes direction as
  `facing << 2` (0=down, 4=up, 8=left, 12=right) and is decoded into the project
  convention (`RAM.py:44-48`).
- **Enemy HP endianness.** `get_enemy_hp()` / `get_enemy_max_hp()` read
  **big-endian**: high byte at the lower address (`0xD216`/`0xD218`), low byte at
  `0xD217`/`0xD219`, verified against a saved in-battle state
  (`RAM.py:265-282`). The comment at `RAM.py:64-68` still says "little-endian (low
  byte at the lower address)" for the same fields — that comment is stale and
  contradicts the read immediately below it. Treat it as a known code-comment
  defect; the code is right (`AGENTS.md:17`).
- Party reads clamp `num_pokemon` to 6 because a mid-write read at battle init can
  return garbage (e.g. 122), which would otherwise walk past the party data and
  drive a 24-bit exp read into the reward clip in one step (`RAM.py:189-197`).

## Stepping

One env step is one decision, not a fixed number of frames. With
`step_mode: "settle"` (the default) `_handle_action` presses the button for 15
frames and then emulates until the game is waiting for input again — a text
prompt, a menu, or the player standing in the overworld — plus 2 frames,
capped at `settle_frame_cap` (1200; a Poké Ball catch takes ~760). It also
waits until the game has seen the button released and is polling input
steadily, so the next press is never dropped. Readiness comes from PyBoy hooks on
pret pokecrystal 1.0 routines, so it only arms for that ROM (SHA1 guard) and
otherwise falls back to `step_mode: "fixed"` (`frames_per_action` = 90 frames
per step). While the game plays itself (`wInputType` = auto input, e.g. the Dude's
catching demo) a step keeps running, so a scripted demo is one step. See `settle.py` for the exact rules and measurements. A walk is
~20 frames, a text box one step, a wild battle ~10 steps.

Every step also writes `game_options_byte` (default 0xD1: instant text,
battle scene off, battle style SET — no "change POKéMON?" prompt) to `wOptions`, since the agent can reach the OPTION menu.
The frames the last step took are in the RAM vector as `last_step_frames`.

## Observation groups added 2026-10-28

These sit after the run-frontier distance in the RAM vector:

| Group | Contents | Source |
|---|---|---|
| `last_step_frames` | log-scaled frames the last step took | `gym_env` |
| exit sense | path direction and distance to the nearest door or crossable edge into a map not yet entered this episode, from the ROM collision graph; cached until a map is entered or blockers change | `_exit_sense` |
| menu state | ▶ cursor visible / row / column, YES-NO open, naming keyboard open, from the tilemap | `RAM.get_menu_state` |
| battle identity | enemy and own active mon types (multi-hot), enemy level, own live HP from `wBattleMon`; zeros outside battle | `RAM.get_battle_identity` |

The collision graph (`rom_collision._edges`, used by the run-frontier field,
path sense and exit sense) models ledges as the engine does. A ledge cell is
walkable from every side except its drop side, and standing on it and
pressing its direction hops two cells.

## Actions

There is no action mask: all 8 actions (`gym_env.ACTIONS`:
`"", a, b, left, right, up, down, start`) are always available to the policy.
The no-op `""` is in the default `ignored_buttons`
(`configs/default_configs/episode_settings.json`), so it waits for the game's
next input prompt without a press. SELECT was dropped on 2026-10-28 (it was a
second no-op). A RAM-derived mask was removed
on 2026-10-28: it read `0xCF07 == 7` as "dialog" and blocked the D-pad, but the
game uses the same value for YES/NO boxes, the start menu and other selectors,
so the policy could not move a menu cursor.

## ROM collision field

`RomCollisionField.local_field(memory, env_vars, radius)` returns an egocentric
`(2R+1)²` row-major list of walkability values in `[0, 1]` (raw collision byte /
255), centred on the player (`rom_collision.py:72-74`); the observation feeds it
`COLLISION_FIELD_RADIUS` (`gym_env.py:1331-1336`).

| aspect | behaviour | citation |
|---|---|---|
| source of truth | the ROM tileset collision table, located through the WRAM pointer pair — bank `0xD1DF`, address `0xD1E0` (little-endian) | `rom_collision.py:14`, `:39`, bank-relative offset arithmetic `:70` |
| per-cell decode | `slot = off + bid * 4 + (y % 2) * 2 + (x % 2)`, where `bid` is the block at `(y // 2, x // 2)` | `rom_collision.py:95-101` |
| scripted overlays | all zeros when `script_active`, mirroring the visited-mask / frontier-direction contract (a stale position has nothing meaningful to report) | `rom_collision.py:76-82` |
| undecodable map / out-of-range | zero-filled, or `_OFF_MAP_VALUE` for off-map cells | `rom_collision.py:83-88`, `:102-104` |
| accuracy | validated at **98.5%** agreement with the live engine's collision bytes across a walk; the misses are map-boundary / connection-border tiles | `rom_collision.py:21-24` |
| semantics | a per-step affordance signal, tileset-invariant and carrying **no history** | `gym_env.py:216-228` |

## Recordings and the trajectory CSV

`enable_record(folder, use_episode_number=True, record_format="images")`
(`gym_env.py:1278-1286`):

| format | behaviour | citation |
|---|---|---|
| `"images"` | the only format that calls `enable_render()` — `"log"` never reads pixels, so pyboy's `tick` render flag is skipped entirely | `gym_env.py:1283-1286` |
| `"log"` | opens `trajectory.csv` lazily on the first recorded step and appends one row per step | `gym_env.py:1625-1631` |
| handle lifecycle | the file handle lives on the env (`_record_log_fh`, initialised `gym_env.py:656`) and is closed on `reset()` (`:1488-1490`) and on `close()` (`:1530-1535`) | `gym_env.py:1488-1490`, `:1530-1535` |
| row writer | `open_trajectory_log(episode_id, fldr, outdir)` / `write_trajectory_row(fh, steps, button, fitness, location)` | `gym_env.py:15-19`, `:1625-1631` |
| CSV header | `step,x,y,map,bank,room,battlestate,playerstate,warp,btn,reward` (per the fact sheet, in `PoliwhiRL/utils/visuals.py:8-20`) | not re-checked this pass — see Known gaps |

## Worker IPC and `terminal_info`

`VecPyBoyEnv` runs each environment in its own process and speaks a small
command/reply protocol over a `multiprocessing.Pipe` — `"step"`, `"reset"`,
`"set_state_path"` and friends, each replying `("ok", payload)`
(`PoliwhiRL/environment/vec_env.py:227`, `:228`).

On a step that ends an episode, the worker builds `terminal_info` **before** the
auto-reset replaces the observation (`vec_env.py:132-139`); it is `None` on any
step where the episode is still running (`vec_env.py:132`).

| key | meaning | citation |
|---|---|---|
| `n_flag`, `n_pokedex`, `n_map`, `n_target` | goal-rung counts and the stage's target | `vec_env.py:140-143` |
| `goal_success` | authoritative "did this episode hit the stage milestone" | `vec_env.py:144-149` |
| `flag_fires` | count of derived-table story-flag fires, *not* the stage's configured-goal count (which is 0 in freeform stages and hid all story progress — observed 2026-07-20) | `vec_env.py:150-159` |
| `unique_cells`, `unique_maps` | whole-episode coverage, from `_ever_visited_cells_this_episode` (the novelty epoch is cleared after flags and would undercount) | `vec_env.py:160-164` |
| `archive_size` | canonical visit-archive size | `vec_env.py:165` |
| `reward_breakdown` | the episode reward breakdown dict — see [rewards.md](rewards.md) | `vec_env.py:166` |
| `truncated` | **bootstrap flag**: `True` only for a `budget` truncation; stagnation terminals report `False` even though `env.step`'s `truncated` is `True` | `vec_env.py:167-176` |
| `truncation_cause` | the raw reason string, for logging and analysis | `vec_env.py:177` |
| `final_stagnation_clock` | how close the episode got to the stagnation limit, 0–1 | `vec_env.py:178-187` |
| `visited_cells`, `visited_maps` | genuinely-visited this episode, merged into the agent's canonical archive (the env's archive is a read-only replica refreshed by `set_visit_archive` broadcasts) | `vec_env.py:188-193` |
| `goal_fire_steps`, `flag_fire_steps` | step at which each goal rung / each derived flag fired — the time-to-rung series for goal-less stages | `vec_env.py:194-202` |
| `discoveries` | run-wide first-ever milestone fires this episode, for the discovery-order graph | `vec_env.py:203-207` |
| `milestone_state` | this episode's milestone ledger, merged run-wide by the agent | `vec_env.py:208-212` |
| `frontier_captures` | Go-Explore save-states captured this episode — see [exploration.md](exploration.md) | `vec_env.py:213-216` |
| `terminal_observation` | attached **only** on a budget truncation, so PPO bootstraps the real continuation state and never the fresh episode's | `vec_env.py:218-225` |

The bootstrap consequence of that split is the agent's job: only `"budget"` seeds
the GAE bootstrap; `stagnation` and `battle_stagnation` are zero-bootstrap
terminals (`vec_env.py:167-176` → `PoliwhiRL/agents/PPO/vec_ppo_agent.py:737-745`,
cause decided at `PoliwhiRL/environment/rewards.py:835-881`).

## Cross-links

- [../architecture.md](../architecture.md) — how the env, agent and models fit together.
- [rewards.md](rewards.md) — the reward arithmetic called at `gym_env.py:1093`.
- [exploration.md](exploration.md) — frontier captures, the archive, seeding.
- [models.md](models.md) — which model consumes `image` and which fuses `ram`.
- [training.md](training.md) — rollout collection over `VecPyBoyEnv`.
- [../configuration.md](../configuration.md) — `record`, mask switches, RAM-encoder shape.
- [../operations/run-triage.md](../operations/run-triage.md) — reading the artefacts this page writes.

## Known gaps

- `RAM_OBS_DIM = 222` — re-verified this session by importing it
  (`python -c "from PoliwhiRL.environment.gym_env import RAM_OBS_DIM;
  print(RAM_OBS_DIM)"`): 173 base keys plus 49 derived flags from
  `_DERIVED_FLAG_TABLE` (`checkpoints.py`), concatenated and counted at
  `gym_env.py:304-305`. An earlier revision of this bullet said 170 and
  itemised the base block by hand; the vector has grown since and a
  hand-listed subgroup count is the first thing to go stale. This is the
  number stated on [`models.md`](models.md); keep the two pages in step.
- **The base-key table above is stale in its line citations.** The block ranges
  were taken against an older `gym_env.py` and the feature list has grown since
  (pokeball count, party resources, run-frontier distance, `last_step_frames`,
  exit sense, menu state and battle identity all sit after it — see
  "Observation groups added 2026-10-28"). Read `_BASE_RAM_FEATURE_KEYS` and its
  `_POKEBALL_COUNT_KEYS` / `_PARTY_RESOURCE_KEYS` / `_RUN_FRONTIER_KEYS` /
  `_STEP_FRAMES_KEYS` / `_EXIT_KEYS` / `_MENU_KEYS` / `_BATTLE_ID_KEYS`
  extensions (`gym_env.py:71-300`) for the current order rather than the table.
- `_DERIVED_FLAG_TABLE` contents and the provenance of each derived flag were not
  audited; `tests/test_event_flags.py` is cited as the validator
  (`gym_env.py:60-64`). Note flag 26 is transient and must not be used as a goal
  terminal (`gym_env.py:60-63`).
- The trajectory CSV header string (`step,x,y,map,bank,room,battlestate,playerstate,warp,btn,reward`)
  is carried from the reconnaissance fact sheet at
  `PoliwhiRL/utils/visuals.py:8-20`; the exact line numbers and header literal
  were not re-checked in this pass.
- The non-terminal branch of the worker loop (what is sent when `done` is `False`)
  was not read; "`terminal_info` is `None` unless `done`" rests on the fact that
  the dict is constructed inside `if done:` (`vec_env.py:132-217`).
- The full command set of the worker pipe was not enumerated — only `step`
  (`vec_env.py:227`) and `set_state_path` (`vec_env.py:228`) were seen.
- `PoliwhiRL/models/CNN/GameBoy.py` **is** the definition site again:
  `TileGridCNN` lives there (`:36-115`) and `PPOTransformer.py:5` imports it. The
  inline `GameBoyCNN` that used to sit inside `PPOTransformer.py` is deleted, as
  is `GameBoyOptimizedCNN`; [models.md](models.md) owns the current layout.
- Staleness of the `little-endian` comment at `RAM.py:64-68` is documented above as
  a comment defect; no fix was attempted (documentation-only pass).
