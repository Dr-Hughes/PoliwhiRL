# Manual investigation states

> [!IMPORTANT]
> **The live save states are not here.** The `.state` files and this table live
> at the repository root under `Manual Investigation States/`, which is
> **gitignored — do not move them.** `Scripts/verify_flags_from_states.py`
> resolves that directory at run time (`Scripts/verify_flags_from_states.py:29`,
> `STATES_DIR = ROOT / "Manual Investigation States"`) and loads each state in
> PyBoy against the ROM at `emu_files/Pokemon - Crystal Version.gbc`
> (`Scripts/verify_flags_from_states.py:28`). Moving the directory breaks that
> script. This page is a readable mirror of
> `Manual Investigation States/state_info.md`, copied verbatim below.

What this page is for: a browsable index of the manually captured save states
that anchor RAM-signal verification, copied from the source-of-truth table in
`Manual Investigation States/state_info.md` so it can be linked from the docs
tree without moving any artefact.

**Sources**: `Manual Investigation States/state_info.md:1-34`,
`Scripts/verify_flags_from_states.py:1-29`.

---

# Save States

Manually captured save states from a single Pokémon Crystal playthrough. The order below reflects the sequence they were saved in during that session.

| # | File | Description |
|---|------|-------------|
| 1 | `bedroom.state` | Player in bedroom, Newbark Town |
| 2 | `downstairs_pre_speaking_to_mom.state` | Downstairs, before talking to mom |
| 3 | `downstairs_during_speaking_to_mom.state` | Mid-dialogue with mom |
| 4 | `downstairs_post_speaking_to_mom.state` | After mom cutscene |
| 5 | `outside_newbark_town.state` | Outside Newbark Town |
| 6 | `pokelab_mid_first_prof_cutscene_dialog.state` | Prof. Elm's lab, mid-first cutscene dialogue |
| 7 | `pokelab_choose_starter.state` | Prof. Elm's lab, choosing starter Pokémon |
| 8 | `pokelab_cynd_selected_prof_cutscene_dialog.state` | Cyndaquil selected, Prof. Elm cutscene dialogue |
| 9 | `pokelab_cynd_post_selection_and_cutscene.state` | Post-starter selection and cutscene |
| 10 | `pokelab_received_potion.state` | Received potion from Prof. Elm |
| 11 | `outside_newbark_town_post_starter_selection.state` | Outside Newbark Town after starter selection |
| 12 | `outside_route_13.state` | Outside Route 13 |
| 13 | `in_wild_battle_route_13.state` | In wild battle on Route 13 |
| 14 | `outside_route_13_post_wild_battle_win.state` | Route 13 after winning wild battle |
| 15 | `inside_route_13_building.state` | Inside building on Route 13 |
| 16 | `outside_route_46.state` | Outside Route 46 |
| 17 | `outside_cherrygrove_city.state` | Outside Cherrygrove City |
| 18 | `inside_cherrygrove_city_pokecentre.state` | Inside Cherrygrove City Pokécenter |
| 19 | `outside_route_30.state` | Outside Route 30 |
| 20 | `outside_route_30_post_battle_learnt_smokescreen.state` | Route 30 after battle, learned Smokescreen |
| 21 | `inside_mr_pokemon_house_in_cutscene_dialog.state` | Mr. Pokémon's house, in cutscene dialogue |
| 22 | `inside_mr_pokemon_house_in_cutscene_dialog_post_received_mystery_egg.state` | After receiving Mystery Egg |
| 23 | `inside_mr_pokemon_house_in_cutscene_dialog_post_received_mystery_egg_and_pokedex.state` | After receiving Mystery Egg and Pokédex |
| 24 | `inside_mr_pokemon_house_post_cutscene__post_received_mystery_egg_and_pokedex.state` | Post-cutscene, after receiving egg and Pokédex |
| 25 | `outside_route_30_post_mr_pokemon_house_in_pokegearcall_with_prof_elm.state` | Route 30, mid-Pokégear call with Prof. Elm |
| 26 | `outside_route_30_post_mr_pokemon_house_post_pokegearcall_with_prof_elm.state` | Route 30, after Pokégear call with Prof. Elm |

All states belong to the same continuous playthrough. The ordering is important for understanding game state progression (flags, items, party composition, etc.) across the session.

---

## What the states are used for

`Scripts/verify_flags_from_states.py` checks the early-game RAM signals against
these states, because the set only covers the opening of the game (bedroom →
starter → Mr. Pokémon's egg → route 30/46)
(`Scripts/verify_flags_from_states.py:7-9`). The signals it verifies are
(`Scripts/verify_flags_from_states.py:11-15`):

| Signal | Meaning |
|---|---|
| `pokedex_owned` | the stage-2 goal (`owned >= 1`) |
| `party species` | should be Cyndaquil once owned |
| `got_starter` | flag 26 (`EVENT_GOT_A_POKEMON_FROM_ELM`) |
| `got_cyndaquil` | flag 27 (`EVENT_GOT_CYNDAQUIL_FROM_ELM`) |
| `got_mystery_egg` | flag 30 (`EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON`) |

It deliberately does **not** touch gym-leader, legendary or HM flags: none of
them are reachable from these states, so there is nothing to verify
(`Scripts/verify_flags_from_states.py:17-18`).

Run it from the project root:

```bash
python Scripts/verify_flags_from_states.py
```

(`Scripts/verify_flags_from_states.py:4-5`; it exits with `pyboy not installed.`
if PyBoy is absent, `Scripts/verify_flags_from_states.py:22-25`.)

The source table lists 26 rows (`Manual Investigation States/state_info.md:7-32`)
and the directory on disk contains 26 `.state` files plus `state_info.md`
itself — the table and the directory agree.

## Known gaps

- The copied table is a snapshot; if the live `state_info.md` is edited, this
  page goes stale. Nothing automated checks the two stay in sync.
- No flag/`(map_bank, map_num)` values are recorded per state here — only the
  five signals the verifier checks
  (`Scripts/verify_flags_from_states.py:11-15`). The full mapping is the job
  of `docs/reference/ram-mapping.md`.
- Whether the training start state (`emu_files/states/start.state`,
  `configs/default_configs/rom_settings.json:3`) is identical to one of these
  26 states was not checked.
- The directory on disk contains exactly 26 `.state` files plus `state_info.md`
  as of this pass, so the copied table is complete; because the directory is
  gitignored, any future state added there will not appear in version control
  and this page will silently drift.
