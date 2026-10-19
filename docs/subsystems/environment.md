# Environment & observation pipeline

What this page is for: the contract between the training loop and the PyBoy-backed
environment — what `step()` and `reset()` hand back, exactly what is inside the
observation (screen image plus the RAM feature vector, and how that vector is
composed), how the action mask and the ROM-derived collision field are produced,
how episode recordings are written, and what travels over the worker process
boundary in `terminal_info`. Read this when a tensor shape, a feature value, or a
metric that originates in the environment looks wrong; the reward arithmetic itself
lives in [rewards.md](rewards.md) and the seeding machinery in
[exploration.md](exploration.md).

**Sources** — `PoliwhiRL/environment/gym_env.py`, `RAM.py`, `action_mask.py`,
`rom_collision.py`, `vec_env.py`, `PoliwhiRL/utils/visuals.py`, re-read against
the working tree on branch `flag31-breakthrough-and-tuning`. JSON values carried
from the earlier reconnaissance notes (untracked working-tree notes under
`.doc-scratch/`, never committed, therefore not citable from a fresh clone). `AGENTS.md` is the
project map; where prose and
code disagree, the code wins (`AGENTS.md:17`).

## `step()` / `reset()` contract

| item | behaviour | citation |
|---|---|---|
| `step(action)` returns | `(observation, self._fitness, self.done, truncated)` | `PoliwhiRL/environment/gym_env.py:1050` |
| `truncated` | `bool(self.reward_calculator.truncated)` — the env does not decide truncation itself, the reward calculator does | `gym_env.py:1049` |
| `step()` order of operations | buffer pre-checkpoint state → apply action → compute fitness → maybe capture a Go-Explore frontier state → build observation → maybe record | `gym_env.py:1036-1050` |
| reward is computed | inside `_calculate_fitness()`, which reads RAM into `env_vars` and calls `Rewards.calculate_reward(env_vars, self.button)`; a `reward_done` promotes `self.done` | `gym_env.py:1090-1097` |
| `reset()` returns | the observation dict (`self.get_observation()`), after re-loading the state bytes | `gym_env.py:1179` |
| `reset()` closes the recording | any open `trajectory.csv` handle is closed and cleared | `gym_env.py:1143-1145` |
| `reset()` per-episode reset | fresh `Rewards` (handed the seed provenance via `reward_config["episode_seeded"]`), zeroed capture sets, `steps = 0`, `episode += 1` | `gym_env.py:1154-1173` |
| domain balancer is fed first | `domain_balancer.note_episode_end(...)` runs on the *outgoing* calculator, before it is replaced | `gym_env.py:1147-1153` |
| startup ordering | fitness is computed **before** the observation is returned, so the goal-target fields reflect the no-op startup step | `gym_env.py:1175-1178` |

## Observation structure

`get_observation()` returns a two-key dict (`gym_env.py:1099-1136`):

| key | dtype / shape | source | citation |
|---|---|---|---|
| `"image"` | `(C,H,W)` RGB float when `vision: true` | `get_screen_image()` | `gym_env.py:1106-1110` |
| `"image"` (tile path) | `(1,18,20)` `uint8` tile-ID grid | `pyboy.game_area()[:18, :20]`, then `np.newaxis` for the channel axis | `gym_env.py:1061-1075` |
| `"ram"` | `float32 (RAM_OBS_DIM,)` | `_build_ram_vector(...)` | `gym_env.py:1115-1135`, shape via `ram_observation_shape()` `gym_env.py:1058-1059` |

The tile grid is deliberately channel-first: `GameBoyCNN` reads the channel count
off `input_shape[0]`, and a bare `(18, 20)` tuple would be misread as 18
channel-less images and raise a `Conv2d` shape error at model construction
(`gym_env.py:1062-1068`). No pixel scaling is applied on either path — the raw
value feeds conv1 and the `GroupNorm` normalises the activation scale (`gym_env.py:1069-1072`).

`GameBoyCNN` consumes the image only and has no `ram` input; only
`PPOTransformer` fuses the RAM vector (`PoliwhiRL/models/PPO/PPOTransformer.py:22`,
`:153`) — see [models.md](models.md) rather than re-deriving it here.

## RAM vector composition

`RAM_FEATURE_KEYS` is the append-only contract; `RAM_OBS_DIM = len(RAM_FEATURE_KEYS)`
(`gym_env.py:241-242`), with named positions exported as `RAM_FEATURE_INDEX` so
downstream code never hard-codes offsets (`gym_env.py:245-253`). The vector is
built from one `ram.get_variables()` read per step (`gym_env.py:1112`) and
assembled by `_build_ram_vector` (`gym_env.py:318`).

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
  indicators, read at `>= 0.5` downstream (`action_mask.py:99-105`).
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

## Action masking

`compute_action_mask(ram_last_step, allow_menus_walking=False)` returns a
`(B, ACTION_SIZE)` tensor, `1.0` allowed / `0.0` masked (`action_mask.py:78-96`).

| rule | detail | citation |
|---|---|---|
| driver signals | `script_active` (RAM byte `0xD438`, one-hot slot), `ui_state_text_box` (`0xCF07 == 7`), battle one-hots (wild / trainer, i.e. `battle_type` 1 or 2) | `action_mask.py:62-67`, `:99-105`; addresses `RAM.py:57`, `:146-149` |
| action ids | `ACTION_SIZE = 9`; `NOOP, A, B, LEFT, RIGHT, UP, DOWN, START, SELECT = range(9)` | `action_mask.py:72-73` |
| **battle-row special case** | `dialog = script_active & text_box & (~in_battle)` — directionals are blocked in overworld dialog but stay **allowed** in battle, because the battle menu (including RUN) is D-pad driven; blocking them makes a losing fight inescapable | `action_mask.py:106-110`, rationale `:64-65` |
| start / select while walking | blocked unless the stage opts in via `allow_menus_walking` | `action_mask.py:42-46`, `:87-90` |
| no stored buffer | the mask is deterministic in the RAM observation, so it is rebuilt at PPO-update time from the stored `ram_states`; callers must pass the last timestep only | `action_mask.py:48-52`, `:83-86` |
| import-time binding | feature indices are resolved once at import so a changed feature set fails loudly rather than silently mis-computing | `action_mask.py:60-61` |

## ROM collision field

`RomCollisionField.local_field(memory, env_vars, radius)` returns an egocentric
`(2R+1)²` row-major list of walkability values in `[0, 1]` (raw collision byte /
255), centred on the player (`rom_collision.py:72-74`); the observation feeds it
`COLLISION_FIELD_RADIUS` (`gym_env.py:1131-1133`).

| aspect | behaviour | citation |
|---|---|---|
| source of truth | the ROM tileset collision table, located through the WRAM pointer pair — bank `0xD1DF`, address `0xD1E0` (little-endian) | `rom_collision.py:14`, `:39`, bank-relative offset arithmetic `:70` |
| per-cell decode | `slot = off + bid * 4 + (y % 2) * 2 + (x % 2)`, where `bid` is the block at `(y // 2, x // 2)` | `rom_collision.py:95-101` |
| scripted overlays | all zeros when `script_active`, mirroring the visited-mask / frontier-direction contract (a stale position has nothing meaningful to report) | `rom_collision.py:76-82` |
| undecodable map / out-of-range | zero-filled, or `_OFF_MAP_VALUE` for off-map cells | `rom_collision.py:83-88`, `:102-104` |
| accuracy | validated at **98.5%** agreement with the live engine's collision bytes across a walk; the misses are map-boundary / connection-border tiles | `rom_collision.py:21-24` |
| semantics | a per-step affordance signal, tileset-invariant and carrying **no history** | `gym_env.py:213-220` |

## Recordings and the trajectory CSV

`enable_record(folder, use_episode_number=True, record_format="images")`
(`gym_env.py:1080-1088`):

| format | behaviour | citation |
|---|---|---|
| `"images"` | the only format that calls `enable_render()` — `"log"` never reads pixels, so pyboy's `tick` render flag is skipped entirely | `gym_env.py:1085-1088` |
| `"log"` | opens `trajectory.csv` lazily on the first recorded step and appends one row per step | `gym_env.py:1273-1277` |
| handle lifecycle | the file handle lives on the env (`_record_log_fh`, initialised `gym_env.py:550-551`) and is closed on `reset()` (`:1143-1145`) and on `close()` (`:1186-1191`) | `gym_env.py:1143-1145`, `:1186-1191` |
| row writer | `open_trajectory_log(episode_id, fldr, outdir)` / `write_trajectory_row(fh, steps, button, fitness, location)` | `gym_env.py:17-18`, `:1275-1277` |
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
- [../configuration.md](../configuration.md) — `vision`, `record`, mask switches.
- [../operations/run-triage.md](../operations/run-triage.md) — reading the artefacts this page writes.

## Known gaps

- `RAM_OBS_DIM = 170` — evaluated from source this pass. Base block is 121 keys
  (67-key literal `gym_env.py:66-186`, plus the 25 `visited_local_*` `:197-200`,
  3 `frontier_dir_*` `:207-212`, 25 `collision_local_*` `:222-225` and 1
  `stagnation_clock` `:236-237` appended blocks) and derived flags add
  `len(_DERIVED_FLAG_TABLE)` = 49 (`checkpoints.py:11-60`, appended and counted at
  `gym_env.py:240-242`). This is the number stated on
  [`models.md`](models.md); keep the two pages in step if the table grows.
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
- `PoliwhiRL/models/CNN/GameBoy.py` was **not** the definition site in this
  working tree: `GameBoyCNN` resolves to
  `PoliwhiRL/models/PPO/PPOTransformer.py:22`. The fact sheet's
  `models/CNN/GameBoy.py:34-48` citation is therefore unconfirmed (that file may
  no longer exist); [models.md](models.md) owns the current layout.
- Staleness of the `little-endian` comment at `RAM.py:64-68` is documented above as
  a comment defect; no fix was attempted (documentation-only pass).
