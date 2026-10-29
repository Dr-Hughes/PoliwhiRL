# Training-framework bugs found by the LLM-agent runs

Issues in the RL training framework (environment, rewards, observation,
action space) that the LLM-agent runs surfaced. The LLM agent only plays the
game to exercise the framework; problems with the agent itself are not
tracked here. Tick an item once the training framework has addressed it.
Held for a separate session.

## Fixed (2026-10-28, uncommitted)

- [x] **`new_map` never pays for a map whose first entry starts a scene.**
  `Rewards._new_map_bonus` (`PoliwhiRL/environment/rewards.py`, the
  `if not env_vars.get("script_active")` branch) skips the payout while a
  script runs but still adds the map to `explored_maps`, so there is no second
  chance that episode. Elm's lab is the clear case: Elm's scene starts on the
  entry step, so the lab never pays `new_map` in any episode. Seen as
  `map_first_entry: ELMS_LAB | paid nothing <-- MISSED` in `violet_run_1`
  (step 86), `violet_run_2` (step 91) and `violet_run_3` (step 103);
  `steps.jsonl` shows `scene: true`, `sources: {}` on the entry step. Any map
  with an on-entry scene is affected.
  *Fix:* payout tracked in a separate `_new_map_paid` set and deferred to the
  first unscripted step on that map; the episode's first map stays waived.
  Replaying run 3: ELMS_LAB now pays 5.0 at step 190 when Elm's scene ends.

- [x] **START is not in the action set, so the nickname keyboard is a trap.**
  `ignored_buttons` drops START, so a policy that answers YES to "Give a
  nickname?" (starter, every catch) can only leave the naming keyboard by
  moving to END or by pressing A until the name is full (the cursor then
  jumps to END). The LLM agent lost ~30 turns there in `violet_run_3`
  (turns 186–215). Check policy rollouts stuck in `ELMS_LAB` or after a catch
  for this before blaming exploration; decide whether to allow START or rely
  on the A-fill escape.
  *Fix:* START removed from `ignored_buttons`; the action mask allows it only
  in a non-battle menu/keyboard overlay (script on, no text box). Verified on
  the run-3 keyboard: START jumps to END, A closes it. The harness offers
  `start` and applies the same mask (`Game.allowed`).

- [x] **`rival_cherrygrove` observation feature reads 1 almost always.** Flag
  1726 (`EVENT_RIVAL_CHERRYGROVE_CITY`) is a sprite-visibility flag: set while
  the rival is hidden, i.e. from game start and again after the fight. It is
  already excluded from rewards/checkpoints (`NON_RECORDABLE_FLAGS`,
  `event_novelty_exclude_flags`), but it is still in the RAM vector, where it
  was 1.0 on every step of `violet_run_4` from the bedroom on (observation
  audit, `report.py`). It carries "rival not currently visible", not "met the
  rival". Decide whether to drop it from the observation or invert/replace it
  with a real post-fight flag.
  *Fix:* no event flag marks the fight (1726 was set in 600/600 sampled
  Go-Explore snapshots, with and without the egg), so the slot is held at 0
  (`OBSERVATION_ZEROED_FLAGS`); vector layout and checkpoints unchanged.

- [x] **The action mask blocked the D-pad in YES/NO boxes and the start
  menu.** It read `0xCF07 == 7` as "dialog", but the game uses 7 for YES/NO,
  the start menu and Mom's day picker too, so the policy could answer NO only
  with B and could never navigate the start menu (harness audit 2026-10-28).
  *Fix:* the action mask is removed; every action is always available and
  select is a no-op via `ignored_buttons`. Scenario probes check down moves
  the YES/NO cursor and START opens the menu.

- [x] **Event flags 0-7 paid `event` novelty.** They are
  `EVENT_TEMPORARY_UNTIL_MAP_RELOAD_1..8`, cleared on every map load; walking
  up the Players-house stairs paid `event +10`, and the same fires fed
  flag-state Go-Explore capture. *Fix:* always excluded in `Rewards`; a
  scenario invariant fails if one fires.

- [x] **A pressed step was 75 frames, a no-op 90, and text printed over
  many steps.** `pyboy.button(delay=15)` queues the release without ticking;
  dialogue cost ~60-80 A presses (Mom, Elm) with most frames half printed.
  *Fix:* settle stepping (`PoliwhiRL/environment/settle.py`): a step runs
  until the game waits for input, and `wOptions` is forced to instant text /
  battle scene off. The fixed mode (`step_mode: "fixed"`) is now a uniform 90
  frames. An adversarial check of the first version found early returns in
  menus (dropped next press, half-drawn frames); fixed by requiring the
  release to be seen and two steady polls.

- [x] **The Dude's catching demo looked like the player's own catch.** On
  "show you how to catch? YES" the game plays a whole wild battle and catch
  on automatic input (`wInputType` 0xC2C7 = 0xFF, ~1,600 frames); the policy
  spent ~16 inert steps in it and the LLM believed it had caught the Rattata
  and stopped (settle_run_1_routes turns 313-328; Pokédex owned stayed 1).
  *Fix:* settle stepping treats automatic input as not ready, so the demo is
  one step (`auto_frames` reports it); the harness says a demo was played and
  shows the goal's RAM check every turn.

- [x] **Ledges were modelled the wrong way round in the collision graph.**
  `rom_collision._edges` hopped from the cell before a ledge, over it; the
  engine hops from the ledge cell itself (verified live: stepping onto the
  Route 30 ledge at (4,24) from the side, then down hops to (4,26)), and a
  ledge cell is walkable from every side but its drop side. The old model
  closed the real path round Youngster Mikey at the Route 30 gap (the v2
  run's bottleneck) for the run-frontier field, path sense, exit sense and
  harness routes. *Fix:* new ledge model in `_edges`.

- [x] **"Will KRIS change POKéMON?" traps an A-masher.** In every multi-mon
  trainer battle, after an enemy faints, YES (the default) opens a party
  screen that answers A with "is already out" forever; only B on "Which
  PKMN?" escapes (Route 30: Mikey and Bug Catcher Don). *Fix:* `wOptions`
  forced to SET battle style (0xD1), which drops the prompt.

- [x] **Opening START froze the stagnation clock.** Only free-walking steps
  counted, and the start menu sets the script byte, so without the mask a
  policy could idle in menus for a whole episode (plan review). *Fix:* every
  non-battle step counts; a novel cell resets only while walking.

- [x] **A failed checkpoint load half-loaded the weights and wiped the run's
  snapshots** (plan review). *Fix:* `info.pth` records `ram_dim`; any load
  error other than a missing checkpoint is raised.

- [x] **The whiteout penalty almost never paid** (fixed 2026-10-29). `_check_whiteout`
  (`PoliwhiRL/environment/rewards.py`) moved its HP baseline without
  checking while `script_active` is set, and a battle is a script, so a
  party that faints in battle gets to HP 0 unchecked; the warp home then
  heals it. All 3 whiteouts in the LLM runs went unpaid
  (`run_20261029_014818` steps 459 and 1301, `run_20261029_001432` step
  571). The heal on arrival also pays `resource_potential` about +12, which
  only refunds the HP lost on the way down. So a whiteout costs nothing
  but time, and in a seeded episode the warp pays `new_map` +5 and
  `frontier` for the house, an instance of the empty-visit-history item
  below. In training, honest episodes pay about −3 to −4 of whiteout per
  episode, so a few transitions land on non-script frames. *Fix:* party HP 0
  arms the whiteout under any script, and the warp (HP back on another
  map) pays it; the warp's heal earns no `resource_potential` refund
  (`Rewards._check_whiteout`). Shipped as step 1 of
  `docs/next-stage-training.md`; watch flag 31.

- [x] **The policy had no menu, cursor or battle-identity features.** What
  the LLM reads as text (which menu is open, where the cursor is, enemy
  species/level, the active mon's live HP from wBattleMon) is at best a few
  blurred pixels at 80x72. Candidate RAM features, once addresses are
  verified.
  *Fixed by v3 (found stale 2026-10-29):* `menu_cursor_*`, `enemy_type_*`,
  own active-mon types, `enemy_level` and `battle_mon_hp_frac` are in the
  RAM vector (`gym_env.py` `RAM_FEATURE_KEYS`).

## Fixed (2026-10-29, uncommitted)

- [x] **A held egg counted as a fainted Pokémon, so the harness nagged for
  an unhealable heal.** An egg slot in `wPartySpecies` (0xDCD8, value
  0xFD = EGGSPECIES) holds a placeholder party_struct — hatch species
  Togepi, level byte 5, HP 0, max HP 19, zeroed moves — and nothing read
  the species byte. The harness saw a fainted mon: in
  `post_gym_01_resumed_20261029_162703` (state `01_togepi_egg_done`,
  Quilava Lv16 46/46 + Togepi egg, species bytes `9C FD FF`) `Game.party()`
  rendered "Party: … #2 Lv5 HP 0/19", `status()["fainted"]` was 1, and
  `heal_facts` injected "YOUR POKéMON NEED HEALING (1 of your Pokémon
  fainted) … Your main goal waits until you have healed" every turn; the
  nurse cannot heal eggs, so the nag never cleared (the game's own menu
  shows "EGG" with no HP bar — `Training Outputs/drive/post_gym/egg_party3.png`).
  On the training side the same slot summed into `party_hp_frac`
  (`RAM.get_party_resources`), capping it at 46/65 = 0.71 in the RAM
  vector while the egg is held and dragging the `heals` gate with it.
  *Fix:* both readers now skip slots whose species byte is 0xFD —
  `get_party_resources` excludes the egg from the hp/max_hp/pp/max_pp sums
  and the able count (tuple shape and vector layout unchanged);
  `Game.party()` returns `egg: bool` per slot, `fainted` excludes eggs, and
  the Party line renders "#2 Egg". Evidence probe:
  `Training Outputs/drive/post_gym/probe_egg_fix.py` (egg state now gives
  resources (1.0, 1.0, 1.0, 1.0, 1/6), `fainted` 0, `heal_reasons` []; the
  no-egg control `emu_files/states/post_falkner.state` reads identical to
  its pre-fix values).

## Open

- [ ] **Seeded episodes start with an empty visit history.** Coverage, the
  visited mask and the exit sense treat already-done maps as new in a
  seeded episode (Go-Explore, reverse curriculum, LLM rings).

- [ ] **`trajectory.csv` `playerstate` reads `walk` on every step, including
  thousands of battle steps.** In all five recorded episodes that reached
  Violet Gym (metrics idx 405/321/460/363/159), the recorder's player-state
  column decoded `walk` while `battlestate` was `wild`/`trainer` (ep_449:
  3,969 such steps), so whiteouts and menus can only be inferred from
  PC-after-battle patterns, never observed, in post-hoc forensics. The
  recorder column, not the env, is suspect — the same runs' live harness
  `mode` reads battle correctly. Fix: decode the recorder column from the
  same bytes `Game.status()` uses; add a scenario invariant that a
  `battlestate != none` step never records `playerstate == walk`. Found by
  the step-1 trajectory pass (`Training Outputs/drive/post_gym/step1/`).

- [ ] **Two recorded episode dirs match no metrics row.** `ep_369` and
  `ep_381` CSVs have no corresponding entry in `training_metrics.json` at
  any length (pre-gym episodes; recorder/metrics bookkeeping gap). Found by
  the same pass.

