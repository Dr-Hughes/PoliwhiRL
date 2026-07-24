# Pokemon Crystal RAM Mapping for PoliwhiRL

Comprehensive reference for Pokemon Crystal WRAM addresses relevant to tracking game progress, story state, and environment conditions for RL training.

---

## map_bank vs map_num vs warp_number

| Address | Name | What It Is | Range | Most Useful For | Source |
|---|---|---|---|---|---|
| `DCB4` | `wWarpNumber` | Which warp point within the current map | 0–N per map | Low — only useful for sub-location disambiguation (e.g., which door you entered a building through) | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#dcb4) |
| `DCB5` | `wMapGroup` ("map_bank") | Map group/region ID | 1–26 | High — identifies the region. Group 24 = New Bark area, 10 = Violet, 11 = Goldenrod, 3 = Dungeons, etc. | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#dcb5) |
| `DCB6` | `wCurMap` ("map_num") | Map number within its group | 1–N per group | High — combined with map_bank gives unique map ID | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#dcb6) |

**Unique location = `(map_bank, map_num)`**. For example, `(24, 4)` = New Bark Town, `(24, 5)` = Elm's Lab, `(3, 49)` = Rocket Hideout B1F. `warp_number` is rarely useful for goals — it changes when you step through a door but doesn't tell you *where* you are.

### Complete Map Library — `(map_bank, map_num)` → Map

Source of truth: [`pret/pokecrystal` → `constants/map_constants.asm`](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm).

The disassembly defines each map with a `newgroup` (the **group** = `map_bank` at `DCB5`) and a `map_const`
index that **restarts at 1 for every group** (the **number** = `map_num` at `DCB6`). A location is therefore
only unique as the pair `(group, number)`. The names below are the pokecrystal map constants verbatim
(strip the `MAP_` prefix). `_BETA` maps are unused/leftover in the retail game but still occupy their index.

> **Verified against live RAM reads:** group 24 produced `3 = ROUTE_29`, `4 = NEW_BARK_TOWN`,
> `7 = PLAYERS_HOUSE_2F` (the player's bedroom upstairs) — matching this table exactly.

> **Caution for the observation vector:** `map_num` is **categorical, not ordinal**, and per-group counts run
> up to 91 (group 3). Dividing it by 255 (see *Currently Tracked RAM Features* below) compresses every map into
> a near-zero scalar and implies a false ordering. Prefer a one-hot / embedding over the `(group, number)` pair.

**Group 1 — OLIVINE:** 1 OLIVINE_POKECENTER_1F · 2 OLIVINE_GYM · 3 OLIVINE_TIMS_HOUSE · 4 OLIVINE_HOUSE_BETA · 5 OLIVINE_PUNISHMENT_SPEECH_HOUSE · 6 OLIVINE_GOOD_ROD_HOUSE · 7 OLIVINE_CAFE · 8 OLIVINE_MART · 9 ROUTE_38_ECRUTEAK_GATE · 10 ROUTE_39_BARN · 11 ROUTE_39_FARMHOUSE · 12 ROUTE_38 · 13 ROUTE_39 · 14 OLIVINE_CITY

**Group 2 — MAHOGANY:** 1 MAHOGANY_RED_GYARADOS_SPEECH_HOUSE · 2 MAHOGANY_GYM · 3 MAHOGANY_POKECENTER_1F · 4 ROUTE_42_ECRUTEAK_GATE · 5 ROUTE_42 · 6 ROUTE_44 · 7 MAHOGANY_TOWN

**Group 3 — DUNGEONS:** 1 SPROUT_TOWER_1F · 2 SPROUT_TOWER_2F · 3 SPROUT_TOWER_3F · 4 TIN_TOWER_1F · 5 TIN_TOWER_2F · 6 TIN_TOWER_3F · 7 TIN_TOWER_4F · 8 TIN_TOWER_5F · 9 TIN_TOWER_6F · 10 TIN_TOWER_7F · 11 TIN_TOWER_8F · 12 TIN_TOWER_9F · 13 BURNED_TOWER_1F · 14 BURNED_TOWER_B1F · 15 NATIONAL_PARK · 16 NATIONAL_PARK_BUG_CONTEST · 17 RADIO_TOWER_1F · 18 RADIO_TOWER_2F · 19 RADIO_TOWER_3F · 20 RADIO_TOWER_4F · 21 RADIO_TOWER_5F · 22 RUINS_OF_ALPH_OUTSIDE · 23 RUINS_OF_ALPH_HO_OH_CHAMBER · 24 RUINS_OF_ALPH_KABUTO_CHAMBER · 25 RUINS_OF_ALPH_OMANYTE_CHAMBER · 26 RUINS_OF_ALPH_AERODACTYL_CHAMBER · 27 RUINS_OF_ALPH_INNER_CHAMBER · 28 RUINS_OF_ALPH_RESEARCH_CENTER · 29 RUINS_OF_ALPH_HO_OH_ITEM_ROOM · 30 RUINS_OF_ALPH_KABUTO_ITEM_ROOM · 31 RUINS_OF_ALPH_OMANYTE_ITEM_ROOM · 32 RUINS_OF_ALPH_AERODACTYL_ITEM_ROOM · 33 RUINS_OF_ALPH_HO_OH_WORD_ROOM · 34 RUINS_OF_ALPH_KABUTO_WORD_ROOM · 35 RUINS_OF_ALPH_OMANYTE_WORD_ROOM · 36 RUINS_OF_ALPH_AERODACTYL_WORD_ROOM · 37 UNION_CAVE_1F · 38 UNION_CAVE_B1F · 39 UNION_CAVE_B2F · 40 SLOWPOKE_WELL_B1F · 41 SLOWPOKE_WELL_B2F · 42 OLIVINE_LIGHTHOUSE_1F · 43 OLIVINE_LIGHTHOUSE_2F · 44 OLIVINE_LIGHTHOUSE_3F · 45 OLIVINE_LIGHTHOUSE_4F · 46 OLIVINE_LIGHTHOUSE_5F · 47 OLIVINE_LIGHTHOUSE_6F · 48 MAHOGANY_MART_1F · 49 TEAM_ROCKET_BASE_B1F · 50 TEAM_ROCKET_BASE_B2F · 51 TEAM_ROCKET_BASE_B3F · 52 ILEX_FOREST · 53 GOLDENROD_UNDERGROUND · 54 GOLDENROD_UNDERGROUND_SWITCH_ROOM_ENTRANCES · 55 GOLDENROD_DEPT_STORE_B1F · 56 GOLDENROD_UNDERGROUND_WAREHOUSE · 57 MOUNT_MORTAR_1F_OUTSIDE · 58 MOUNT_MORTAR_1F_INSIDE · 59 MOUNT_MORTAR_2F_INSIDE · 60 MOUNT_MORTAR_B1F · 61 ICE_PATH_1F · 62 ICE_PATH_B1F · 63 ICE_PATH_B2F_MAHOGANY_SIDE · 64 ICE_PATH_B2F_BLACKTHORN_SIDE · 65 ICE_PATH_B3F · 66 WHIRL_ISLAND_NW · 67 WHIRL_ISLAND_NE · 68 WHIRL_ISLAND_SW · 69 WHIRL_ISLAND_CAVE · 70 WHIRL_ISLAND_SE · 71 WHIRL_ISLAND_B1F · 72 WHIRL_ISLAND_B2F · 73 WHIRL_ISLAND_LUGIA_CHAMBER · 74 SILVER_CAVE_ROOM_1 · 75 SILVER_CAVE_ROOM_2 · 76 SILVER_CAVE_ROOM_3 · 77 SILVER_CAVE_ITEM_ROOMS · 78 DARK_CAVE_VIOLET_ENTRANCE · 79 DARK_CAVE_BLACKTHORN_ENTRANCE · 80 DRAGONS_DEN_1F · 81 DRAGONS_DEN_B1F · 82 DRAGON_SHRINE · 83 TOHJO_FALLS · 84 DIGLETTS_CAVE · 85 MOUNT_MOON · 86 UNDERGROUND_PATH · 87 ROCK_TUNNEL_1F · 88 ROCK_TUNNEL_B1F · 89 SAFARI_ZONE_FUCHSIA_GATE_BETA · 90 SAFARI_ZONE_BETA · 91 VICTORY_ROAD

**Group 4 — ECRUTEAK:** 1 ECRUTEAK_TIN_TOWER_ENTRANCE · 2 WISE_TRIOS_ROOM · 3 ECRUTEAK_POKECENTER_1F · 4 ECRUTEAK_LUGIA_SPEECH_HOUSE · 5 DANCE_THEATER · 6 ECRUTEAK_MART · 7 ECRUTEAK_GYM · 8 ECRUTEAK_ITEMFINDER_HOUSE · 9 ECRUTEAK_CITY

**Group 5 — BLACKTHORN:** 1 BLACKTHORN_GYM_1F · 2 BLACKTHORN_GYM_2F · 3 BLACKTHORN_DRAGON_SPEECH_HOUSE · 4 BLACKTHORN_EMYS_HOUSE · 5 BLACKTHORN_MART · 6 BLACKTHORN_POKECENTER_1F · 7 MOVE_DELETERS_HOUSE · 8 ROUTE_45 · 9 ROUTE_46 · 10 BLACKTHORN_CITY

**Group 6 — CINNABAR:** 1 CINNABAR_POKECENTER_1F · 2 CINNABAR_POKECENTER_2F_BETA · 3 ROUTE_19_FUCHSIA_GATE · 4 SEAFOAM_GYM · 5 ROUTE_19 · 6 ROUTE_20 · 7 ROUTE_21 · 8 CINNABAR_ISLAND

**Group 7 — CERULEAN:** 1 CERULEAN_GYM_BADGE_SPEECH_HOUSE · 2 CERULEAN_POLICE_STATION · 3 CERULEAN_TRADE_SPEECH_HOUSE · 4 CERULEAN_POKECENTER_1F · 5 CERULEAN_POKECENTER_2F_BETA · 6 CERULEAN_GYM · 7 CERULEAN_MART · 8 ROUTE_10_POKECENTER_1F · 9 ROUTE_10_POKECENTER_2F_BETA · 10 POWER_PLANT · 11 BILLS_HOUSE · 12 ROUTE_4 · 13 ROUTE_9 · 14 ROUTE_10_NORTH · 15 ROUTE_24 · 16 ROUTE_25 · 17 CERULEAN_CITY

**Group 8 — AZALEA:** 1 AZALEA_POKECENTER_1F · 2 CHARCOAL_KILN · 3 AZALEA_MART · 4 KURTS_HOUSE · 5 AZALEA_GYM · 6 ROUTE_33 · 7 AZALEA_TOWN

**Group 9 — LAKE_OF_RAGE:** 1 LAKE_OF_RAGE_HIDDEN_POWER_HOUSE · 2 LAKE_OF_RAGE_MAGIKARP_HOUSE · 3 ROUTE_43_MAHOGANY_GATE · 4 ROUTE_43_GATE · 5 ROUTE_43 · 6 LAKE_OF_RAGE

**Group 10 — VIOLET:** 1 ROUTE_32 · 2 ROUTE_35 · 3 ROUTE_36 · 4 ROUTE_37 · 5 VIOLET_CITY · 6 VIOLET_MART · 7 VIOLET_GYM · 8 EARLS_POKEMON_ACADEMY · 9 VIOLET_NICKNAME_SPEECH_HOUSE · 10 VIOLET_POKECENTER_1F · 11 VIOLET_KYLES_HOUSE · 12 ROUTE_32_RUINS_OF_ALPH_GATE · 13 ROUTE_32_POKECENTER_1F · 14 ROUTE_35_GOLDENROD_GATE · 15 ROUTE_35_NATIONAL_PARK_GATE · 16 ROUTE_36_RUINS_OF_ALPH_GATE · 17 ROUTE_36_NATIONAL_PARK_GATE

**Group 11 — GOLDENROD:** 1 ROUTE_34 · 2 GOLDENROD_CITY · 3 GOLDENROD_GYM · 4 GOLDENROD_BIKE_SHOP · 5 GOLDENROD_HAPPINESS_RATER · 6 BILLS_FAMILYS_HOUSE · 7 GOLDENROD_MAGNET_TRAIN_STATION · 8 GOLDENROD_FLOWER_SHOP · 9 GOLDENROD_PP_SPEECH_HOUSE · 10 GOLDENROD_NAME_RATER · 11 GOLDENROD_DEPT_STORE_1F · 12 GOLDENROD_DEPT_STORE_2F · 13 GOLDENROD_DEPT_STORE_3F · 14 GOLDENROD_DEPT_STORE_4F · 15 GOLDENROD_DEPT_STORE_5F · 16 GOLDENROD_DEPT_STORE_6F · 17 GOLDENROD_DEPT_STORE_ELEVATOR · 18 GOLDENROD_DEPT_STORE_ROOF · 19 GOLDENROD_GAME_CORNER · 20 GOLDENROD_POKECENTER_1F · 21 POKECOM_CENTER_ADMIN_OFFICE_MOBILE · 22 ILEX_FOREST_AZALEA_GATE · 23 ROUTE_34_ILEX_FOREST_GATE · 24 DAY_CARE

**Group 12 — VERMILION:** 1 ROUTE_6 · 2 ROUTE_11 · 3 VERMILION_CITY · 4 VERMILION_FISHING_SPEECH_HOUSE · 5 VERMILION_POKECENTER_1F · 6 VERMILION_POKECENTER_2F_BETA · 7 POKEMON_FAN_CLUB · 8 VERMILION_MAGNET_TRAIN_SPEECH_HOUSE · 9 VERMILION_MART · 10 VERMILION_DIGLETTS_CAVE_SPEECH_HOUSE · 11 VERMILION_GYM · 12 ROUTE_6_SAFFRON_GATE · 13 ROUTE_6_UNDERGROUND_PATH_ENTRANCE

**Group 13 — PALLET:** 1 ROUTE_1 · 2 PALLET_TOWN · 3 REDS_HOUSE_1F · 4 REDS_HOUSE_2F · 5 BLUES_HOUSE · 6 OAKS_LAB

**Group 14 — PEWTER:** 1 ROUTE_3 · 2 PEWTER_CITY · 3 PEWTER_NIDORAN_SPEECH_HOUSE · 4 PEWTER_GYM · 5 PEWTER_MART · 6 PEWTER_POKECENTER_1F · 7 PEWTER_POKECENTER_2F_BETA · 8 PEWTER_SNOOZE_SPEECH_HOUSE

**Group 15 — FAST_SHIP:** 1 OLIVINE_PORT · 2 VERMILION_PORT · 3 FAST_SHIP_1F · 4 FAST_SHIP_CABINS_NNW_NNE_NE · 5 FAST_SHIP_CABINS_SW_SSW_NW · 6 FAST_SHIP_CABINS_SE_SSE_CAPTAINS_CABIN · 7 FAST_SHIP_B1F · 8 OLIVINE_PORT_PASSAGE · 9 VERMILION_PORT_PASSAGE · 10 MOUNT_MOON_SQUARE · 11 MOUNT_MOON_GIFT_SHOP · 12 TIN_TOWER_ROOF

**Group 16 — INDIGO:** 1 ROUTE_23 · 2 INDIGO_PLATEAU_POKECENTER_1F · 3 WILLS_ROOM · 4 KOGAS_ROOM · 5 BRUNOS_ROOM · 6 KARENS_ROOM · 7 LANCES_ROOM · 8 HALL_OF_FAME

**Group 17 — FUCHSIA:** 1 ROUTE_13 · 2 ROUTE_14 · 3 ROUTE_15 · 4 ROUTE_18 · 5 FUCHSIA_CITY · 6 FUCHSIA_MART · 7 SAFARI_ZONE_MAIN_OFFICE · 8 FUCHSIA_GYM · 9 BILLS_OLDER_SISTERS_HOUSE · 10 FUCHSIA_POKECENTER_1F · 11 FUCHSIA_POKECENTER_2F_BETA · 12 SAFARI_ZONE_WARDENS_HOME · 13 ROUTE_15_FUCHSIA_GATE

**Group 18 — LAVENDER:** 1 ROUTE_8 · 2 ROUTE_12 · 3 ROUTE_10_SOUTH · 4 LAVENDER_TOWN · 5 LAVENDER_POKECENTER_1F · 6 LAVENDER_POKECENTER_2F_BETA · 7 MR_FUJIS_HOUSE · 8 LAVENDER_SPEECH_HOUSE · 9 LAVENDER_NAME_RATER · 10 LAVENDER_MART · 11 SOUL_HOUSE · 12 LAV_RADIO_TOWER_1F · 13 ROUTE_8_SAFFRON_GATE · 14 ROUTE_12_SUPER_ROD_HOUSE

**Group 19 — SILVER:** 1 ROUTE_28 · 2 SILVER_CAVE_OUTSIDE · 3 SILVER_CAVE_POKECENTER_1F · 4 ROUTE_28_STEEL_WING_HOUSE

**Group 20 — CABLE_CLUB:** 1 POKECENTER_2F · 2 TRADE_CENTER · 3 COLOSSEUM · 4 TIME_CAPSULE · 5 MOBILE_TRADE_ROOM · 6 MOBILE_BATTLE_ROOM

**Group 21 — CELADON:** 1 ROUTE_7 · 2 ROUTE_16 · 3 ROUTE_17 · 4 CELADON_CITY · 5 CELADON_DEPT_STORE_1F · 6 CELADON_DEPT_STORE_2F · 7 CELADON_DEPT_STORE_3F · 8 CELADON_DEPT_STORE_4F · 9 CELADON_DEPT_STORE_5F · 10 CELADON_DEPT_STORE_6F · 11 CELADON_DEPT_STORE_ELEVATOR · 12 CELADON_MANSION_1F · 13 CELADON_MANSION_2F · 14 CELADON_MANSION_3F · 15 CELADON_MANSION_ROOF · 16 CELADON_MANSION_ROOF_HOUSE · 17 CELADON_POKECENTER_1F · 18 CELADON_POKECENTER_2F_BETA · 19 CELADON_GAME_CORNER · 20 CELADON_GAME_CORNER_PRIZE_ROOM · 21 CELADON_GYM · 22 CELADON_CAFE · 23 ROUTE_16_FUCHSIA_SPEECH_HOUSE · 24 ROUTE_16_GATE · 25 ROUTE_7_SAFFRON_GATE · 26 ROUTE_17_ROUTE_18_GATE

**Group 22 — CIANWOOD:** 1 ROUTE_40 · 2 ROUTE_41 · 3 CIANWOOD_CITY · 4 MANIAS_HOUSE · 5 CIANWOOD_GYM · 6 CIANWOOD_POKECENTER_1F · 7 CIANWOOD_PHARMACY · 8 CIANWOOD_PHOTO_STUDIO · 9 CIANWOOD_LUGIA_SPEECH_HOUSE · 10 POKE_SEERS_HOUSE · 11 BATTLE_TOWER_1F · 12 BATTLE_TOWER_BATTLE_ROOM · 13 BATTLE_TOWER_ELEVATOR · 14 BATTLE_TOWER_HALLWAY · 15 ROUTE_40_BATTLE_TOWER_GATE · 16 BATTLE_TOWER_OUTSIDE

**Group 23 — VIRIDIAN:** 1 ROUTE_2 · 2 ROUTE_22 · 3 VIRIDIAN_CITY · 4 VIRIDIAN_GYM · 5 VIRIDIAN_NICKNAME_SPEECH_HOUSE · 6 TRAINER_HOUSE_1F · 7 TRAINER_HOUSE_B1F · 8 VIRIDIAN_MART · 9 VIRIDIAN_POKECENTER_1F · 10 VIRIDIAN_POKECENTER_2F_BETA · 11 ROUTE_2_NUGGET_HOUSE · 12 ROUTE_2_GATE · 13 VICTORY_ROAD_GATE

**Group 24 — NEW_BARK:** 1 ROUTE_26 · 2 ROUTE_27 · 3 ROUTE_29 · 4 NEW_BARK_TOWN · 5 ELMS_LAB · 6 PLAYERS_HOUSE_1F · 7 PLAYERS_HOUSE_2F · 8 PLAYERS_NEIGHBORS_HOUSE · 9 ELMS_HOUSE · 10 ROUTE_26_HEAL_HOUSE · 11 DAY_OF_WEEK_SIBLINGS_HOUSE · 12 ROUTE_27_SANDSTORM_HOUSE · 13 ROUTE_29_ROUTE_46_GATE

**Group 25 — SAFFRON:** 1 ROUTE_5 · 2 SAFFRON_CITY · 3 FIGHTING_DOJO · 4 SAFFRON_GYM · 5 SAFFRON_MART · 6 SAFFRON_POKECENTER_1F · 7 SAFFRON_POKECENTER_2F_BETA · 8 MR_PSYCHICS_HOUSE · 9 SAFFRON_MAGNET_TRAIN_STATION · 10 SILPH_CO_1F · 11 COPYCATS_HOUSE_1F · 12 COPYCATS_HOUSE_2F · 13 ROUTE_5_UNDERGROUND_PATH_ENTRANCE · 14 ROUTE_5_SAFFRON_GATE · 15 ROUTE_5_CLEANSE_TAG_HOUSE

**Group 26 — CHERRYGROVE:** 1 ROUTE_30 · 2 ROUTE_31 · 3 CHERRYGROVE_CITY · 4 CHERRYGROVE_MART · 5 CHERRYGROVE_POKECENTER_1F · 6 CHERRYGROVE_GYM_SPEECH_HOUSE · 7 GUIDE_GENTS_HOUSE · 8 CHERRYGROVE_EVOLUTION_SPEECH_HOUSE · 9 ROUTE_30_BERRY_HOUSE · 10 MR_POKEMONS_HOUSE · 11 ROUTE_31_VIOLET_GATE

> **Note on group naming:** the group label (OLIVINE, DUNGEONS, NEW_BARK, …) is the pokecrystal `newgroup`
> name, *not* a guarantee that every map in it is in that town — groups bundle a city with its surrounding
> routes, gates, and houses, and group 3 (DUNGEONS) and group 15 (FAST_SHIP) are pure grab-bags. Always
> resolve a location by the full `(group, number)` pair, never the group name alone.

---

## Currently Tracked RAM Features

The policy receives a fixed-order vector built only by
[`_build_ram_vector`](./PoliwhiRL/environment/gym_env.py). The exact append-only
feature contract is `RAM_FEATURE_KEYS` in that module; this table groups the
current inputs rather than duplicating every vector index.

| Address/source | Features and encoding | Source |
|---|---|---|
| `DCB4-DCB8`, `D148`, `D4DE` | Warp, map pair, coordinates, room and facing one-hot | [RAM.py](./PoliwhiRL/environment/RAM.py) |
| `D84E-D850`, `DCD7`, `DCDF+` | Money; party size, level, HP and log-scaled EXP | [RAM.py](./PoliwhiRL/environment/RAM.py) |
| `DE99-DED8` | Pokédex owned/seen popcounts | [RAM.py](./PoliwhiRL/environment/RAM.py) |
| `C2FA-C2FD` | Live four-direction collision bytes | [RAM.py](./PoliwhiRL/environment/RAM.py) |
| `D22D`, `D857`, `D95D` | Battle-type one-hot, Johto badge popcount `/8`, player-state one-hot | [RAM.py](./PoliwhiRL/environment/RAM.py) |
| `D8BC`, `D4B7`, `C2A9` | Key-item count, game hour and BGM ID | [RAM.py](./PoliwhiRL/environment/RAM.py) |
| `D216-D219`, `C634-C637` | Enemy HP/ratio and active Pokémon's four move-PP values | [RAM.py](./PoliwhiRL/environment/RAM.py) |
| `D438`, `CF07`, `D43D` | Script/UI/map-handler state as binary and one-hot features | [RAM.py](./PoliwhiRL/environment/RAM.py) |
| `DA72-DB71` | Raw event bytes are read internally; only curated durable/progress bits are appended to the policy vector | [checkpoints.py](./PoliwhiRL/checkpoints.py) |
| Reward/archive state | Goal counters, explored count, recent maps, egocentric visited mask, frontier direction and stagnation clock | [gym_env.py](./PoliwhiRL/environment/gym_env.py) |
| ROM map decoder | Egocentric collision field scaled to `[0,1]` | [rom_collision.py](./PoliwhiRL/environment/rom_collision.py) |
| `C4A0-C607` | Screen tilemap used when tile observations are selected | [RAM.py](./PoliwhiRL/environment/RAM.py) |

---

## Event Flags — Story Progress Tracking

The 256-byte region at `0xDA72–0xDB71` contains 2048 individual bit flags. Byte offset = `flag_index / 8`, bit = `flag_index % 8`.

> **All flag numbers below verified on 2026-06-03** against
> `pret/pokecrystal/constants/event_flags.asm` (vendored as
> `PoliwhiRL/environment/event_flags.asm`, identical to upstream).
> Previous versions had ~40 wrong indices due to ignored `const_skip` and
> `const_next N` directives in the ASM — every flag after the first jump
> (gym leaders, rivals, legendaries, HM07) was at a far higher index than
> a naive sequential count gives. `DERIVED_FLAG_TABLE` in
> `PoliwhiRL/checkpoints.py` is the authoritative in-code source and
> matches these numbers exactly (enforced by `tests/test_event_flags.py`).

### Early Game Gates (catch-first-Pokémon milestone chain)

| Flag # | Byte | Bit | Constant Name | Meaning |
|---|---|---|---|---|
| 26 | 3 | 2 | `EVENT_GOT_A_POKEMON_FROM_ELM` | Any starter received from Elm (fires after 2nd Elm talk, per `maps/ElmsLab.asm:274`) |
| 27 | 3 | 3 | `EVENT_GOT_CYNDAQUIL_FROM_ELM` | Picked Cyndaquil |
| 28 | 3 | 4 | `EVENT_GOT_TOTODILE_FROM_ELM` | Picked Totodile |
| 29 | 3 | 5 | `EVENT_GOT_CHIKORITA_FROM_ELM` | Picked Chikorita |
| 30 | 3 | 6 | `EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON` | Egg received at Mr. Pokémon's house (Route 30) |
| 31 | 3 | 7 | `EVENT_GAVE_MYSTERY_EGG_TO_ELM` | Egg returned to Elm (implies surviving Route 29 rival battle) |
| 39 | 4 | 7 | `EVENT_GOT_BERRY_FROM_ROUTE_30_HOUSE` | Route 30 navigation proxy |
| 44 | 5 | 4 | `EVENT_REFUSED_TO_TAKE_EGG_FROM_ELMS_AIDE` | Declined the Togepi egg first time |
| 45 | 5 | 5 | `EVENT_GOT_TOGEPI_EGG_FROM_ELMS_AIDE` | Received Pokéballs + Togepi egg as aide hand-off |
| 65 | 8 | 1 | `EVENT_DUDE_TALKED_TO_YOU` | Catching tutorial NPC engaged |
| 66 | 8 | 2 | `EVENT_LEARNED_TO_CATCH_POKEMON` | Tutorial complete — implies player has Pokéballs |
| 91 | 11 | 3 | `EVENT_GOT_BICYCLE` | Has Bicycle |
| 123 | 15 | 3 | `EVENT_RELEASED_THE_BEASTS` | Roaming Raikou/Entei active |

### HM & Key Item Acquisition

| Flag # | Byte | Bit | Constant Name | Meaning |
|---|---|---|---|---|
| 16 | 2 | 0 | `EVENT_GOT_HM01_CUT` | Can cut trees |
| 17 | 2 | 1 | `EVENT_GOT_HM02_FLY` | Can fly (post-E4) |
| 18 | 2 | 2 | `EVENT_GOT_HM03_SURF` | Can surf on water |
| 19 | 2 | 3 | `EVENT_GOT_HM04_STRENGTH` | Can push boulders |
| 20 | 2 | 4 | `EVENT_GOT_HM05_FLASH` | Can light dark caves |
| 21 | 2 | 5 | `EVENT_GOT_HM06_WHIRLPOOL` | Can navigate whirlpools |
| 1672 | 209 | 0 | `EVENT_GOT_HM07_WATERFALL` | Can climb waterfalls (Kanto) |
| 23 | 2 | 7 | `EVENT_GOT_OLD_ROD` | Has Old Rod |
| 24 | 3 | 0 | `EVENT_GOT_GOOD_ROD` | Has Good Rod |
| 25 | 3 | 1 | `EVENT_GOT_SUPER_ROD` | Has Super Rod |

### Johto Story Milestones

| Flag # | Byte | Bit | Constant Name | Meaning |
|---|---|---|---|---|
| 33 | 4 | 1 | `EVENT_CLEARED_RADIO_TOWER` | Rocket cleared from Radio Tower |
| 34 | 4 | 2 | `EVENT_CLEARED_ROCKET_HIDEOUT` | Goldenrod Rocket Hideout cleared |
| 40 | 5 | 0 | `EVENT_MADE_WHITNEY_CRY` | Defeated Whitney |
| 41 | 5 | 1 | `EVENT_HERDED_FARFETCHD` | Farfetch'd herded in Ilex Forest |
| 42 | 5 | 2 | `EVENT_FOUGHT_SUDOWOODO` | Sudowoodo defeated on Route 36 |
| 43 | 5 | 3 | `EVENT_CLEARED_SLOWPOKE_WELL` | Team Rocket cleared from Slowpoke Well |
| 32 | 4 | 0 | `EVENT_JASMINE_RETURNED_TO_GYM` | Jasmine returned from Lighthouse |
| 1726 | 215 | 6 | `EVENT_RIVAL_CHERRYGROVE_CITY` | Route 29 rival fight spawn (sprite-vis) |
| 793 | 99 | 1 | `EVENT_BEAT_RIVAL_IN_MT_MOON` | Kanto rival defeated |

### Gym Leader Defeat Flags

| Flag # | Byte | Bit | Constant Name | Gym |
|---|---|---|---|---|
| 1213 | 151 | 5 | `EVENT_BEAT_FALKNER` | Gym 1 — Violet City (Flying) |
| 1214 | 151 | 6 | `EVENT_BEAT_BUGSY` | Gym 2 — Azalea Town (Bug) |
| 1215 | 151 | 7 | `EVENT_BEAT_WHITNEY` | Gym 3 — Goldenrod City (Normal) |
| 1216 | 152 | 0 | `EVENT_BEAT_MORTY` | Gym 4 — Ecruteak City (Ghost) |
| 1217 | 152 | 1 | `EVENT_BEAT_JASMINE` | Gym 5 — Olivine City (Steel) |
| 1218 | 152 | 2 | `EVENT_BEAT_CHUCK` | Gym 6 — Cianwood City (Fighting) |
| 1219 | 152 | 3 | `EVENT_BEAT_PRYCE` | Gym 7 — Mahogany Town (Ice) |
| 1220 | 152 | 4 | `EVENT_BEAT_CLAIR` | Gym 8 — Blackthorn City (Dragon) |
| 1221 | 152 | 5 | `EVENT_BEAT_BROCK` | Kanto Gym 1 — Pewter City (Rock) |
| 1222 | 152 | 6 | `EVENT_BEAT_MISTY` | Kanto Gym 2 — Cerulean City (Water) |
| 1224 | 153 | 0 | `EVENT_BEAT_ERIKA` | Kanto Gym 4 — Celadon City (Grass) |
| 1225 | 153 | 1 | `EVENT_BEAT_JANINE` | Kanto Gym 5 — Fuchsia City (Poison) |
| 1226 | 153 | 2 | `EVENT_BEAT_SABRINA` | Kanto Gym 6 — Saffron City (Psychic) |
| 1227 | 153 | 3 | `EVENT_BEAT_BLAINE` | Kanto Gym 7 — Cinnabar Island (Fire) |
| 1228 | 153 | 4 | `EVENT_BEAT_BLUE` | Kanto Gym 8 — Viridian City (Normal) |

### Champion & Legendaries

| Flag # | Byte | Bit | Constant Name | Meaning |
|---|---|---|---|---|
| 1468 | 183 | 4 | `EVENT_BEAT_CHAMPION_LANCE` | Defeated Champion Lance |
| 791 | 98 | 7 | `EVENT_FOUGHT_HO_OH` | Encountered Ho-Oh at Tin Tower |
| 792 | 99 | 0 | `EVENT_FOUGHT_LUGIA` | Encountered Lugia at Whirl Islands |

> Note: there are no individual `EVENT_BEAT_*` flags for the Elite Four
> members (Will/Koga/Bruno/Karen). Champion Lance is the only flag for
> the E4 → Champion sequence. NPC-blocking flag indices in the
> following section were not re-verified in the 2026-05-29 rebuild —
> use `pret/pokecrystal/constants/event_flags.asm` as the canonical
> source if relying on them.

### NPC Blocking Flags (sprite visibility = path blocked)

When the flag is **SET**, the sprite is **hidden** (path is open). When **CLEAR**, the sprite is **visible** (path is blocked).

| Flag # | Byte Offset | Bit | Constant Name | Blocks | Source |
|---|---|---|---|---|---|
| 1813 | byte 226 | 5 | `EVENT_ROUTE_30_YOUNGSTER_JOEY` | Joey blocks Route 30 | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1745 | byte 218 | 1 | `EVENT_RADIO_TOWER_BLACKBELT_BLOCKS_STAIRS` | Blackbelt blocks Radio Tower stairs | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1763 | byte 220 | 3 | `EVENT_BLACKTHORN_CITY_SUPER_NERD_BLOCKS_GYM` | Super Nerd blocks Blackthorn Gym | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1878 | byte 234 | 6 | `EVENT_MAHOGANY_TOWN_POKEFAN_M_BLOCKS_EAST` | Pokefan blocks east exit of Mahogany | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1879 | byte 234 | 7 | `EVENT_MAHOGANY_TOWN_POKEFAN_M_BLOCKS_GYM` | Pokefan blocks Mahogany Gym | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1741 | byte 217 | 5 | `EVENT_GOLDENROD_CITY_ROCKET_TAKEOVER` | Rocket grunts appear in Goldenrod | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1889 | byte 236 | 1 | `EVENT_TEAM_ROCKET_DISBANDED` | All Rocket NPCs gone (post-E4) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1756 | byte 219 | 4 | `EVENT_ROUTE_43_GATE_ROCKETS` | Rocket grunts block Route 43 gate | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1868 | byte 233 | 4 | `EVENT_BLACKTHORN_CITY_GRAMPS_BLOCKS_DRAGONS_DEN` | Gramps blocks Dragon's Den entrance | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1872 | byte 234 | 0 | `EVENT_FOUGHT_SNORLAX` | Snorlax moved from route | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1743 | byte 217 | 7 | `EVENT_GOLDENROD_CITY_CIVILIANS` | Civilians reappear in Goldenrod (post-Rocket) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1744 | byte 218 | 0 | `EVENT_RADIO_TOWER_CIVILIANS_AFTER` | Civilians reappear in Radio Tower | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1765 | byte 220 | 5 | `EVENT_DAY_CARE_MAN_IN_DAY_CARE` | Day Care Man present | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1769 | byte 221 | 1 | `EVENT_ILEX_FOREST_FARFETCHD` | Farfetch'd present in Ilex Forest | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1770 | byte 221 | 2 | `EVENT_ROUTE_34_ILEX_FOREST_GATE_TEACHER_BEHIND_COUNTER` | Teacher at Ilex Forest gate (before Farfetch'd) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1771 | byte 221 | 3 | `EVENT_ROUTE_34_ILEX_FOREST_GATE_LASS` | Lass blocks Ilex Forest gate | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1772 | byte 221 | 4 | `EVENT_ROUTE_34_ILEX_FOREST_GATE_TEACHER_IN_WALKWAY` | Teacher moves to walkway (after Farfetch'd) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |

### Story Phase Flags (high-level game state)

| Flag / Engine Flag | Constant Name | Meaning | Source |
|---|---|---|---|
| Engine flag 14 | `ENGINE_ROCKET_SIGNAL_ON_CH20` | Rocket signal received (triggers Radio Tower event) | [engine_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/engine_flags.asm) |
| Engine flag 22 | `ENGINE_ROCKETS_IN_RADIO_TOWER` | Rockets currently in Radio Tower | [engine_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/engine_flags.asm) |
| Engine flag 26 | `ENGINE_ROCKETS_IN_MAHOGANY` | Rockets currently in Mahogany Mart | [engine_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/engine_flags.asm) |
| Engine flag 16 | `ENGINE_HALL_OF_FAME` | Beat Elite Four, in Kanto phase | [engine_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/engine_flags.asm) |
| 800+ range | `EVENT_GOT_NUGGET_FROM_GUY` through `EVENT_RESTORED_POWER_TO_KANTO` | Kanto power restoration quest | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1700+ range | `EVENT_ROUTE_24_ROCKET` through `EVENT_TELEPORT_GUY` | Kanto NPC states | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |

### Crystal-Exclusive Events

| Flag # | Byte Offset | Bit | Constant Name | Meaning | Source |
|---|---|---|---|---|---|
| 821 | byte 102 | 5 | `EVENT_FOUGHT_SUICUNE` | Fought Suicune (Crystal-exclusive legendary) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 822 | byte 102 | 6 | `EVENT_GOT_RAINBOW_WING` | Obtained Rainbow Wing (triggers Ho-Oh) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1967 | byte 245 | 7 | `EVENT_SAW_SUICUNE_ON_ROUTE_42` | Suicune appeared on Route 42 | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1968 | byte 246 | 0 | `EVENT_SAW_SUICUNE_ON_ROUTE_36` | Suicune appeared on Route 36 | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1970 | byte 246 | 2 | `EVENT_TIN_TOWER_1F_SUICUNE` | Suicune at Tin Tower 1F | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1971 | byte 246 | 3 | `EVENT_TIN_TOWER_1F_ENTEI` | Entei at Tin Tower 1F | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1972 | byte 246 | 4 | `EVENT_TIN_TOWER_1F_RAIKOU` | Raikou at Tin Tower 1F | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 819 | byte 102 | 3 | `EVENT_FOUGHT_EUSINE` | Fought Eusine (Crystal-exclusive NPC) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 820 | byte 102 | 4 | `EVENT_KOJI_ALLOWS_YOU_PASSAGE_TO_TIN_TOWER` | Koji lets you pass to Tin Tower roof | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 145–148 | byte 18–19 | — | `EVENT_WALL_OPENED_IN_*_CHAMBER` | Ruins of Alph chamber walls opened | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 192 | byte 24 | 0 | `EVENT_FOREST_IS_RESTLESS` | Ilex Forest is restless (Celebi event) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |

### Legendary Capture Flags (Sarial's research)

| Address | Constant / Meaning | Source |
|---|---|---|
| `DAD4` | Set when Ho-Oh is captured/defeated | [GCL Forum - Sarial](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |
| `DAD5` | Set when Lugia is captured/defeated | [GCL Forum - Sarial](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |
| `DB51` | Set when Sudowoodo is captured/defeated | [GCL Forum - Sarial](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |
| `DB5C` | Set when Red Gyarados is captured/defeated | [GCL Forum - Sarial](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |
| `DB60` | Set when Snorlax is captured/defeated | [GCL Forum - Sarial](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |

### Roaming Pokemon Locations (Randil's research)

| Address | Meaning | Source |
|---|---|---|
| `DFD1` | Roaming Raikou map bank | [GCL Forum - Randil](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |
| `DFD2` | Roaming Raikou map number | [GCL Forum - Randil](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |
| `DFD8` | Roaming Entei map bank | [GCL Forum - Randil](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |
| `DFD9` | Roaming Entei map number | [GCL Forum - Randil](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |

---

## Additional Single-Byte RAM Reference

This is a wider research inventory, not a list of uniformly untracked fields.
The current vector already includes `D857`, `D95D`, `D4B7`, `D22D`, `C2A9` and
`D8BC`; the other entries remain debug probes or candidates. The authoritative
current contract is the table above and `_build_ram_vector`.

### Game Progress & Badges

| Address | Feature | Size | Scaling | Why It's Useful | Source |
|---|---|---|---|---|---|
| `D857` | **Johto badges** (bitmask: bit 0=Zephyr, 1=Hive, 2=Plain, 3=Fog, 4=Mineral, 5=Storm, 6=Glacier, 7=Rising) | 1 byte | popcount /8 (tracked) | **Highest impact** — compact ordinal progress through Johto storyline. Popcount = number of gyms beaten. | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d857), [ram_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/ram_constants.asm) |
| `D858` | **Kanto badges** (bitmask: bit 0=Boulder, 1=Cascade, 2=Thunder, 3=Rainbow, 4=Soul, 5=Marsh, 6=Volcano, 7=Earth) | 1 byte | /255 | Post-game progress through Kanto gyms | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d858), [ram_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/ram_constants.asm) |
| `D84C` | Unowndex status (`93` = unlocked) | 1 byte | /255 | Ruins of Alph puzzle completion | [GCL Forum - Sarial](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |
| `D84D` | Bug Catching Contest (`93`=not done, `97`=active) | 1 byte | /255 | Side quest tracking | [GCL Forum - Sarial](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |

### Player State & Movement

| Address | Feature | Scaling | Why It's Useful | Source |
|---|---|---|---|---|
| `D95D` | **Player state** (0=walk, 1=battle, 2=cycling, 4=surfing/diving) | one-hot (tracked) | Indicates mobility/context and what actions make sense | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d95d), [ram_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/ram_constants.asm) |
| `D4B6` | Day of week (0=Sunday–6=Saturday) | /7 | Time-gated daily events | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d4b6), [ram_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/ram_constants.asm) |
| `D4B7` | **Game hour** (0–23) | /255 (tracked) | Time-gated events (morning/day/night Pokémon, certain NPCs) | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d4b7) |
| `D4B8` | Game minute | /60 | Fine-grained time | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d4b8) |
| `D4C4-D4C5` | **Play time hours** (16-bit LE) | /10000 | Total progress proxy — correlates with story advancement | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d4c4) |

### Battle Context

| Address | Feature | Scaling | Why It's Useful | Source |
|---|---|---|---|---|
| `D22D` | **Battle type** (0=none, 1=wild, 2=trainer) | one-hot (tracked) | **High value** — detect if in battle vs. exploring. Fundamentally changes what actions make sense. | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d22d) |
| `D230` | Wild battle type (07=shiny/can't escape, 08=Headbutt, etc.) | /255 | Encounter type detection | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d230) |
| `D233` | Enemy trainer type | /255 | Gym leader / Elite Four detection | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d233) |
| `C2A9` | **Currently playing BGM** | /255 (tracked) | Context signal: battle music, city music, cave music, gym music. Changes every map/load. | [GCL Forum - Sarial](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |

### Inventory Signals

| Address | Feature | Scaling | Why It's Useful | Source |
|---|---|---|---|---|
| `D892` | Number of items (Item Pocket) | /25 | Inventory fullness | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d892) |
| `D8BC` | **Number of key items** (Key Pocket) | /25 (tracked) | Key item count as story progress proxy | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d8bc) |
| `D8D7` | Number of Poke Balls | /25 | Ball inventory for catching | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d8d7) |
| `D855-D856` | **Coins** (16-bit LE) | /10000 | Goldenrod Game Corner progress | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d855) |
| `DCA1` | Repel steps left | /255 | In grass / actively exploring | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#dca1) |
| `DC4B` | Blue Card points | /255 | Game Corner progress | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#dc4b) |

### Script & Overworld State

| Address | Feature | Scaling | Why It's Useful | Source |
|---|---|---|---|---|
| `wScriptVar` | Script variable (temporary value used by map scripts) | /255 | Holds values set by `setval`/`addval` script commands | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `wMapStatus` | Map loading state (0=start, 1=enter, 2=handle, 3=done) | /3 | Detect if map is still loading (transitions) | [ram_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/ram_constants.asm) |
| `wScriptFlags` | Script execution state (bit 2=running, bit 3=deferred) | /255 | Detect if a cutscene/script is playing | [ram_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/ram_constants.asm) |
| `wWalkingDirection` | Player movement direction (-1=standing, 0=down, 1=up, 2=left, 3=right) | /3 | Player orientation | [ram_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/ram_constants.asm) |
| `wPlayerState` | Player mobility mode (related to `D95D`) | /255 | Bike/surf/skate state | [ram_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/ram_constants.asm) |

---

### Empirically Verified Script / UI State Bytes

Addresses below were cross-checked against ground-truth gameplay labels from a 278-step reward-evaluator playthrough (New Bark Town → Elm's lab → starter selection → nickname → exit to Route 29). Documented behaviour is what was *observed in this run*, not a symbol attribution from pokecrystal — these likely correspond to the symbol-named entries above (`wScriptFlags`, `wMapStatus`, etc.) but the mapping has not been confirmed against `wram.asm`. Use these for empirical detectors; treat the symbol-named table above as the eventual canonical source.

#### `0xD438` — scripted-overlay / control-lock flag

| Value | State observed |
|---|---|
| `0` | Player has free walking control (overworld, indoor, route walking) |
| `255` (`0xFF`) | A script controls the screen — covers cutscenes, dialogue boxes the player is locked into (even if technically holding the controller), menus, and the nickname keyboard |

**Verification:** Flipped to 255 on the exact frame each cutscene started (steps 11, 75, 153, 168, 185, 240) and back to 0 on the exact frame walking control was restored (steps 58, 147, 254). The "I accidentally locked myself in dialog at steps 220-239" case correctly stays at 255 — confirming the byte encodes *script-state*, not *button-input availability*. Single-frame `0xD438 = 0` flickers (at 219, 228, 236, 240, 253) are script-boundary frames: one script ends and the next starts on the following frame.

**Usefulness for the policy:** This is the single most useful undocumented byte found so far. A policy with `0xD438` in its observation can learn *"don't try to navigate when this byte is 255 — just press A"* in a handful of episodes, which is exactly the dialogue-mashing skill the agent currently has to infer from screen pixels.

#### `0xCF07` — UI rendering / text-box state

| Value | State observed |
|---|---|
| `5` | Indoor overworld walking baseline |
| `0` | Outdoor overworld walking OR a UI overlay is being rendered (menu, ball selection, nickname keyboard) — context-disambiguated by `0xD438` |
| `7` | Text box is currently displayed |
| `1` | Transient / between-frame transitions (observed at step 58 leaving the house) |

**Verification:** Toggled `5 ↔ 7` cleanly at every inter-dialogue-box transition during long cutscenes (e.g. steps 112-113, 138-139 inside the Elm speech). Dropped to `0` for the exact 17-step nickname-keyboard window (168-184) and the ball-selection menu (step 153). Note: bytes `0xCF08`–`0xCF20` move in lock-step with `0xCF07` and look like a contiguous text-box fill region; `0xCF07` alone is enough as a feature.

**Useful combinations:**
- `0xD438 == 255 AND 0xCF07 == 7` → cutscene currently showing a dialogue box (press A to advance)
- `0xD438 == 255 AND 0xCF07 == 0` → menu / keyboard / ball selection (navigation, not A-mashing)

#### `0xD43D` — map handler / script bank context

| Value | State observed |
|---|---|
| `128` | Inside the player's house (overworld, indoor walking) |
| `30` | A script is currently active (overlaps with `0xD438 == 255`; persists during script cleanup frames 254-258 after `0xD438` has cleared) |
| `165` | Outdoor overworld walking (New Bark exterior + Route 29) |
| `0` | Single-frame map-load transition (observed at step 77 during the warp-to-lab) |

**Verification:** Took `128` exclusively while in the player's house, `165` exclusively outside, and `30` during every scripted period in the playthrough. The "fade / regaining control" window at steps 254-258 keeps `0xD43D == 30` after `0xD438` has dropped to 0 — i.e. this byte reflects map/script-bank context that lingers slightly beyond the script-active flag.

**Usefulness:** Cheaper proxy for "indoor vs outdoor" than re-checking `map_bank`/`map_num` against the map-group table.

#### `0xD143` — sub-map / room context byte (provisional)

| Value | State observed |
|---|---|
| `0` | Throughout the entire pre-starter sequence (steps 1-164) |
| `5` | From step 165 (frame the starter was added to party / Elm script repositioned the player) onwards through the rest of the run |

**Caveat:** The flip coincides with both *the starter joining the party* and *a scripted repositioning inside the lab* on the same frame. We have not isolated which event the byte reflects. Treat as "definitely changes around step 165, semantics not fully verified" until a second probe pins it down.

#### `0xC2A9` — BGM ID (observed values)

The address itself is already documented (`bgm_id`); these are concrete observed values from this run that pin down specific BGM IDs to specific contexts:

| Value | Context |
|---|---|
| `0` | New Bark Town outdoor (steps 1-74, before warp to lab) |
| `50` | Elm's lab (steps 75-258) |
| `60` | Route 29 (steps 259-271) |
| `52` | Later Route 29 sub-section (steps 272+, possibly entering tall grass / a different map cell) |

Useful as a cheap location fingerprint independent of `(map_bank, map_num)`.

---

### Implemented Script/UI Features

`0xD438`, `0xCF07` and `0xD43D` are now part of the policy vector as binary and
one-hot features. They give the policy direct access to the distinction between
walking, cutscenes, text boxes, menus and map-handler transitions. Keep the
empirical value tables above as the verification record; use
`_build_ram_vector` for the current encoding.

---

## Key Item IDs (for Key Pocket scanning at `D8BD–D8D6`)

Scanning the Key Pocket for these item IDs tells you which story-critical items the player has acquired.

| Item ID | Item Name | Story Significance | Source |
|---|---|---|---|
| `07` | Bicycle | Movement upgrade (Route 30, from Mom) | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `36` | Coin Case | Goldenrod Underground / Game Corner access | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `39` | Exp. Share | Party training | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `44` | S.S. Ticket | Fast ship to Kanto (from Elm, post-E4) | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `73` | GS Ball | Celebi trigger (post-game, from Kurt) | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `7F` | Card Key | Radio Tower 5F (from Blackbelt) | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `80` | Machine Part | Restores power to Kanto (from Cerulean Gym Rocket) | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `85` | Basement Key | Rocket Hideout B3F (from Radio Tower) | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `86` | Pass | Olivine Gym access (from Captain on Route 37) | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `67` | Slowpoke Tail | Olivine Gym quest (from Slowpoke Well) | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `45` | Mystery Egg | From Mr. Pokemon, leads to Togepi | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `46` | Clear Bell | Revives Legendary Beast in Burned Tower | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `47` | Silver Wing | Given by Oak, triggers Ho-Oh event | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `B2` | Rainbow Wing | Crystal-exclusive, triggers Ho-Oh at Tin Tower roof | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `F3` | HM01 Cut | Cut trees | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `F4` | HM02 Fly | Fly between registered cities (post-E4) | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `F5` | HM03 Surf | Surf on water | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `F6` | HM04 Strength | Push boulders | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `F7` | HM05 Flash | Light dark caves | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `F8` | HM06 Whirlpool | Navigate whirlpools in Whirl Islands | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |
| `F9` | HM07 Waterfall | Climb waterfalls to Silver Cave (Kanto) | [GCL RAM map - Items](https://gbdev.io/pokemem/?p=crystal#d8bd) |

---

## Engine Flags (Persistent State)

Engine flags are a separate flag system from event flags, stored in a dedicated region. They persist across map loads and are not part of the `DA72-DB71` event flag block.

| Engine Flag # | Constant Name | Meaning | Source |
|---|---|---|---|
| 5–9 | `ENGINE_ZEPHYRBADGE` through `ENGINE_RISINGBADGE` | Johto badge flags (mirrors `D857`) | [engine_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/engine_flags.asm) |
| 10–17 | `ENGINE_BOULDERBADGE` through `ENGINE_EARTHBADGE` | Kanto badge flags (mirrors `D858`) | [engine_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/engine_flags.asm) |
| 14 | `ENGINE_ROCKET_SIGNAL_ON_CH20` | Rocket signal received | [engine_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/engine_flags.asm) |
| 16 | `ENGINE_HALL_OF_FAME` | In Hall of Fame / Kanto phase | [engine_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/engine_flags.asm) |
| 22 | `ENGINE_ROCKETS_IN_RADIO_TOWER` | Rockets occupying Radio Tower | [engine_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/engine_flags.asm) |
| 26 | `ENGINE_ROCKETS_IN_MAHOGANY` | Rockets occupying Mahogany Mart | [engine_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/engine_flags.asm) |
| 27 | `ENGINE_REACHED_GOLDENROD` | Player has reached Goldenrod City | [engine_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/engine_flags.asm) |
| 28 | `ENGINE_STRENGTH_ACTIVE` | Strength is currently active (pushing boulders) | [engine_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/engine_flags.asm) |
| 30 | `ENGINE_DOWNHILL` | Downhill mode on Route 32 | [engine_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/engine_flags.asm) |

---

## Scripting System Reference

Crystal's map event scripts use these commands to check and modify game state. Understanding them helps interpret what the game checks before allowing passage:

| Command | Opcode | What It Does | Source |
|---|---|---|---|
| `checkevent` | `$31` | Checks an event flag | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `setevent` | `$33` | Sets an event flag | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `clearevent` | `$32` | Clears an event flag | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `checkflag` | `$34` | Checks an engine flag | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `setflag` | `$36` | Sets an engine flag | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `clearflag` | `$35` | Clears an engine flag | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `checkitem` | `$21` | Checks if player has an item | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `checkpoke` | `$2C` | Checks if player has a Pokemon | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `checktime` | `$2B` | Checks time of day | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `xycompare` | `$39` | Compares player XY coordinates | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `warp` | `$3C` | Warps player to a map | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `disappear` | `$6E` | Hides an NPC sprite (sets event flag) | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `appear` | `$6F` | Shows an NPC sprite (clears event flag) | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `moveobject` | `$72` | Moves an NPC to coordinates | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `changemapblocks` | `$79` | Changes map tile layout (e.g., removes barriers) | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `wildon` / `wildoff` | `$37` / `$38` | Enables/disables wild encounters | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `readvar` | `$1C` | Reads a game variable into script var | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |
| `readmem` | `$19` | Reads a RAM address into script var | [event_commands.html](https://pret.github.io/pokecrystal/event_commands.html) |

---

## How Path Blocking Works

The game gates progress through three mechanisms:

### 1. NPC Sprites Blocking Path
An NPC stands in the way. The map script checks an event flag before allowing the player to pass. When the flag is set, `disappear` or `moveobject` removes the blocking sprite.

**Examples:**
- Lorelei blocks Route 29 until `EVENT_GOT_A_POKEMON_FROM_ELM` is set
- A Lass blocks Ilex Forest gate until you herd Farfetch'd
- Rocket grunts block Route 43 until `EVENT_CLEARED_ROCKET_HIDEOUT`
- A Super Nerd blocks Blackthorn Gym until Clair's storyline progresses

### 2. Map Block Changes
The `changemapblocks` / `changeblock` script commands modify the tile layout at specific coordinates. A walkable tile becomes a wall, or vice versa.

**Examples:**
- Ruins of Alph chamber walls open when puzzles are solved (`EVENT_WALL_OPENED_IN_*_CHAMBER`)
- Goldenrod Underground Switch Room entrances change layout
- Goldenrod Dept Store B1F has 3 different layouts controlled by event flags
- Rocket Hideout doors open sequentially as you disable security cameras

### 3. HM-Dependent Terrain
Certain tiles (water, tall grass with trees, dark caves, boulders, whirlpools, waterfalls) require specific HMs to traverse. The game checks the `wStatusFlags` byte and the `wPlayerState` byte to determine if the player can use the field move.

**Examples:**
- Water tiles require Surf (HM03)
- Trees require Cut (HM01)
- Dark caves (Union Cave B2F, Dark Cave, Rock Tunnel) require Flash (HM05)
- Boulders in Ice Path, Dark Cave, and Blackthorn Gym require Strength (HM04)
- Whirlpools in Whirl Islands require Whirlpool (HM06)
- Waterfall on Route 28 requires Waterfall (HM07)

---

## Recommendations for PoliwhiRL

### Priority 1: Add to RAM vector (high signal-to-noise)

| Feature | Address | Rationale |
|---|---|---|
| Johto badges | `D857` | Single byte = ordinal story progress. Popcount gives gym count. Each bit corresponds to an HM unlock. |
| Battle type | `D22D` | 0=exploring, 1=wild battle, 2=trainer battle. Prevents the policy from pressing random buttons during battles. |
| Player state | `D95D` | Encodes movement mode (walk/bike/surf/skate). Implicitly tells you which HMs you've acquired. |

### Priority 2: Derived features (computed from existing event flags)

You already read `DA72-DB71`. Rather than adding more raw bytes, consider computing derived features in `_build_ram_vector`:

| Derived Feature | Source Flags | Encoding |
|---|---|---|
| `gym_count` | `EVENT_BEAT_FALKNER` through `EVENT_BEAT_CLAIR` (flags 1213–1220) | popcount / 8 |
| `has_cut` | `EVENT_GOT_HM01_CUT` (flag 16) | 0 or 1 |
| `has_surf` | `EVENT_GOT_HM03_SURF` (flag 18) | 0 or 1 |
| `has_strength` | `EVENT_GOT_HM04_STRENGTH` (flag 19) | 0 or 1 |
| `has_flash` | `EVENT_GOT_HM05_FLASH` (flag 20) | 0 or 1 |
| `has_whirlpool` | `EVENT_GOT_HM06_WHIRLPOOL` (flag 21) | 0 or 1 |
| `rocket_cleared_hideout` | `EVENT_CLEARED_ROCKET_HIDEOUT` (flag 34) | 0 or 1 |
| `rocket_cleared_radio` | `EVENT_CLEARED_RADIO_TOWER` (flag 33) | 0 or 1 |
| `has_starter` | `EVENT_GOT_A_POKEMON_FROM_ELM` (flag 26) | 0 or 1 |
| `beat_elite_four` | `EVENT_BEAT_ELITE_FOUR` (flag 68) | 0 or 1 |

### Priority 3: Location disambiguation

Change goal matching from `[x, y, map_num]` to `[x, y, map_bank, map_num]` to avoid collisions between maps with the same number in different groups (e.g., map #5 in group 24 = Elm's Lab, map #5 in group 8 = Azalea Gym).
