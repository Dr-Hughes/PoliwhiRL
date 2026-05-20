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

### Map Group Reference

| Group | Name | Example Maps | Source |
|---|---|---|---|
| 1 | Olivine | Olivine City, Route 38, Route 39 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 2 | Mahogany | Mahogany Town, Route 42, Route 44 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 3 | Dungeons | Sprout Tower, Tin Tower, Burned Tower, Union Cave, Slowpoke Well, Rocket Hideout, Ilex Forest, Goldenrod Underground, Mt. Mortar, Ice Path, Whirl Islands, Silver Cave, Dark Cave, Victory Road, Mt. Moon, Rock Tunnel, Diglett's Cave, Tohjo Falls | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 4 | Ecruteak | Ecruteak City, Tin Tower Entrance, Dance Theater, Wise Trio's Room | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 5 | Blackthorn | Blackthorn City, Gym, Route 45, Route 46 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 6 | Cinnabar | Cinnabar Island, Seafoam Gym, Route 19–21 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 7 | Cerulean | Cerulean City, Power Plant, Bill's House, Route 4/9/10/24/25 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 8 | Azalea | Azalea Town, Kurt's House, Charcoal Kiln, Route 33 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 9 | Lake of Rage | Lake of Rage, Route 43 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 10 | Violet | Violet City, Route 32–37, Earl's Academy | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 11 | Goldenrod | Goldenrod City, Gym, Game Corner, Dept Store, Underground, Day Care, Route 34 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 12 | Vermilion | Vermilion City, Fan Club, Route 6/11 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 13 | Pallet | Pallet Town, Red's House, Oak's Lab, Route 1 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 14 | Pewter | Pewter City, Route 3 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 15 | Fast Ship | Olivine Port, Vermilion Port, S.S. Aqua, Mt. Moon Square, Tin Tower Roof | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 16 | Indigo | Indigo Plateau, Hall of Fame, Elite Four rooms, Route 23 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 17 | Fuchsia | Fuchsia City, Gym, Safari Zone, Route 13–18 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 18 | Lavender | Lavender Town, Route 8/10/12 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 19 | Silver | Silver Cave, Route 28 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 20 | Cable Club | Trade Center, Colosseum, Time Capsule | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 21 | Celadon | Celadon City, Dept Store, Mansion, Game Corner, Gym, Route 7/16/17 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 22 | Cianwood | Cianwood City, Gym, Battle Tower, Route 40/41 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 23 | Viridian | Viridian City, Gym, Trainer House, Route 2/22 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 24 | New Bark | New Bark Town, Elm's Lab, Player's House, Route 26/27/29 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 25 | Saffron | Saffron City, Gym, Route 5 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |
| 26 | Cherrygrove | Cherrygrove City, Route 30/31 | [map_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/map_constants.asm) |

---

## Currently Tracked RAM Features

| Address | Feature | Scaling in Code | Source |
|---|---|---|---|
| `DCB8` | Player X coordinate | /255 | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `DCB7` | Player Y coordinate | /255 | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `DCB6` | Map number (`wCurMap`) | /255 | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `DCB5` | Map bank (`wMapGroup`) | /255 | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `DCB4` | Warp number | /255 | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `D148` | Room player is in | /255 | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `D84E-D850` | Player money (24-bit LE) | /1,000,000 | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `DCD7` | Number of Pokemon in party | — | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `DCDF+` | Party Pokemon data (species, level, HP, EXP) | level/100, hp/1000, log1p(exp)/20 | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `DEB9-DED8` | Pokedex seen (256 bit flags) | popcount /251 | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `DE99-DEB8` | Pokedex owned (256 bit flags) | popcount /251 | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `C2FA` | Collision data (down) | /255 | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `C2FB` | Collision data (up) | /255 | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `C2FC` | Collision data (left) | /255 | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `C2FD` | Collision data (right) | /255 | [RAM.py](../PoliwhiRL/environment/RAM.py) |
| `DA72-DB71` | Event flags (256 bytes, 2048 bitflags) | raw bytes /255 | [gym_env.py](../PoliwhiRL/environment/gym_env.py) |
| `C4A0-C607` | Screen tilemap (18×20 tiles) | — | [RAM.py](../PoliwhiRL/environment/RAM.py) |

---

## Event Flags — Story Progress Tracking

The 256-byte region at `0xDA72–0xDB71` contains 2048 individual bit flags. Byte offset = `flag_index / 8`, bit = `flag_index % 8`.

### Early Game Gates (the "can I leave?" flags)

| Flag # | Byte Offset | Bit | Constant Name | Meaning | Unblocks | Source |
|---|---|---|---|---|---|---|
| 25 | byte 3 | 1 | `EVENT_GOT_A_POKEMON_FROM_ELM` | Received any starter from Elm | Route 29 (Lorelei moves) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 26 | byte 3 | 2 | `EVENT_GOT_CYNDAQUIL_FROM_ELM` | Got Cyndaquil specifically | — | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 27 | byte 3 | 3 | `EVENT_GOT_TOTODILE_FROM_ELM` | Got Totodile specifically | — | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 28 | byte 3 | 4 | `EVENT_GOT_CHIKORITA_FROM_ELM` | Got Chikorita specifically | — | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 38 | byte 4 | 6 | `EVENT_CLEARED_SLOWPOKE_WELL` | Team Rocket cleared from Slowpoke Well | Azalea Gym (Bugsy returns) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 41 | byte 5 | 1 | `EVENT_MADE_WHITNEY_CRY` | Defeated Whitney | Goldenrod Gym accessible | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 43 | byte 5 | 3 | `EVENT_HERDED_FARFETCHD` | Caught Farfetch'd in Ilex Forest | Headbutt TM, Route 32 west | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 44 | byte 5 | 4 | `EVENT_FOUGHT_SUDOWOODO` | Defeated Sudowoodo on Route 36 | Route 36 to Ecruteak open | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 52 | byte 6 | 4 | `EVENT_CLEARED_RADIO_TOWER` | Rocket cleared from Radio Tower | Radio Tower functional, Gym 4 route open | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 53 | byte 6 | 5 | `EVENT_CLEARED_ROCKET_HIDEOUT` | Rocket Hideout in Goldenrod cleared | Mahogany Town accessible, Route 43 | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 40 | byte 5 | 0 | `EVENT_JASMINE_RETURNED_TO_GYM` | Jasmine returned from Lighthouse | Olivine Gym (Gym 6) challengeable | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 95 | byte 11 | 7 | `EVENT_BEAT_ELITE_FOUR` | Defeated all Elite Four members | Access to Kanto region | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 46 | byte 5 | 6 | `EVENT_DECIDED_TO_HELP_LANCE` | Agreed to help Lance at Lake of Rage | Lance appears at Rocket Hideout | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 54 | byte 6 | 6 | `EVENT_GOT_SS_TICKET_FROM_ELM` | Received SS Ticket | Can board S.S. Aqua to Kanto | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 120 | byte 15 | 0 | `EVENT_RELEASED_THE_BEASTS` | Released Raikou/Entei from Burned Tower | Roaming legendaries active | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |

### HM & Key Item Acquisition (movement ability gates)

| Flag # | Byte Offset | Bit | Constant Name | Meaning | Source |
|---|---|---|---|---|---|
| 16 | byte 2 | 0 | `EVENT_GOT_HM01_CUT` | Can cut trees | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 17 | byte 2 | 1 | `EVENT_GOT_HM02_FLY` | Can fly (post-E4) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 18 | byte 2 | 2 | `EVENT_GOT_HM03_SURF` | Can surf on water | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 19 | byte 2 | 3 | `EVENT_GOT_HM04_STRENGTH` | Can push boulders | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 20 | byte 2 | 4 | `EVENT_GOT_HM05_FLASH` | Can light dark caves | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 21 | byte 2 | 5 | `EVENT_GOT_HM06_WHIRLPOOL` | Can navigate whirlpools | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 134 | byte 16 | 6 | `EVENT_GOT_HM07_WATERFALL` | Can climb waterfalls (Kanto) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 86 | byte 10 | 6 | `EVENT_GOT_BICYCLE` | Has Bicycle | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 22 | byte 2 | 6 | `EVENT_GOT_OLD_ROD` | Has Old Rod | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 23 | byte 2 | 7 | `EVENT_GOT_GOOD_ROD` | Has Good Rod | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 24 | byte 3 | 0 | `EVENT_GOT_SUPER_ROD` | Has Super Rod | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |

### Gym Leader Defeat Flags

| Flag # | Byte Offset | Bit | Constant Name | Gym | Source |
|---|---|---|---|---|---|
| 1030 | byte 128 | 6 | `EVENT_BEAT_FALKNER` | Gym 1 — Violet City (Flying) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1031 | byte 128 | 7 | `EVENT_BEAT_BUGSY` | Gym 2 — Azalea Town (Bug) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1032 | byte 129 | 0 | `EVENT_BEAT_WHITNEY` | Gym 3 — Goldenrod City (Normal) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1033 | byte 129 | 1 | `EVENT_BEAT_MORTY` | Gym 4 — Ecruteak City (Ghost) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1034 | byte 129 | 2 | `EVENT_BEAT_JASMINE` | Gym 5 — Olivine City (Steel) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1035 | byte 129 | 3 | `EVENT_BEAT_CHUCK` | Gym 6 — Cianwood City (Fighting) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1036 | byte 129 | 4 | `EVENT_BEAT_PRYCE` | Gym 7 — Mahogany Town (Ice) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1037 | byte 129 | 5 | `EVENT_BEAT_CLAIR` | Gym 8 — Blackthorn City (Dragon) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1038 | byte 129 | 6 | `EVENT_BEAT_BROCK` | Kanto Gym 1 — Pewter City (Rock) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1039 | byte 129 | 7 | `EVENT_BEAT_MISTY` | Kanto Gym 2 — Cerulean City (Water) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1040 | byte 130 | 0 | `EVENT_BEAT_LTSURGE` | Kanto Gym 3 — Vermilion City (Electric) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1041 | byte 130 | 1 | `EVENT_BEAT_ERIKA` | Kanto Gym 4 — Celadon City (Grass) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1042 | byte 130 | 2 | `EVENT_BEAT_JANINE` | Kanto Gym 5 — Fuchsia City (Poison) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1043 | byte 130 | 3 | `EVENT_BEAT_SABRINA` | Kanto Gym 6 — Saffron City (Psychic) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1044 | byte 130 | 4 | `EVENT_BEAT_BLAINE` | Kanto Gym 7 — Cinnabar Island (Fire) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1045 | byte 130 | 5 | `EVENT_BEAT_BLUE` | Kanto Gym 8 — Viridian City (Normal) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |

### Elite Four & Champion

| Flag # | Byte Offset | Bit | Constant Name | Meaning | Source |
|---|---|---|---|---|---|
| 1046 | byte 130 | 6 | `EVENT_BEAT_ELITE_4_WILL` | Defeated Will (Psychic) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1047 | byte 130 | 7 | `EVENT_BEAT_ELITE_4_KOGA` | Defeated Koga (Poison) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1048 | byte 131 | 0 | `EVENT_BEAT_ELITE_4_BRUNO` | Defeated Bruno (Fighting) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1049 | byte 131 | 1 | `EVENT_BEAT_ELITE_4_KAREN` | Defeated Karen (Dark) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1050 | byte 131 | 2 | `EVENT_BEAT_CHAMPION_LANCE` | Defeated Champion Lance | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |

### NPC Blocking Flags (sprite visibility = path blocked)

When the flag is **SET**, the sprite is **hidden** (path is open). When **CLEAR**, the sprite is **visible** (path is blocked).

| Flag # | Byte Offset | Bit | Constant Name | Blocks | Source |
|---|---|---|---|---|---|
| 1619 | byte 202 | 3 | `EVENT_ROUTE_30_YOUNGSTER_JOEY` | Joey blocks Route 30 | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1587 | byte 198 | 3 | `EVENT_RADIO_TOWER_BLACKBELT_BLOCKS_STAIRS` | Blackbelt blocks Radio Tower stairs | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1646 | byte 205 | 6 | `EVENT_BLACKTHORN_CITY_SUPER_NERD_BLOCKS_GYM` | Super Nerd blocks Blackthorn Gym | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1631 | byte 203 | 7 | `EVENT_MAHOGANY_TOWN_POKEFAN_M_BLOCKS_EAST` | Pokefan blocks east exit of Mahogany | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1632 | byte 204 | 0 | `EVENT_MAHOGANY_TOWN_POKEFAN_M_BLOCKS_GYM` | Pokefan blocks Mahogany Gym | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1567 | byte 195 | 15 | `EVENT_GOLDENROD_CITY_ROCKET_TAKEOVER` | Rocket grunts appear in Goldenrod | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1575 | byte 196 | 7 | `EVENT_TEAM_ROCKET_DISBANDED` | All Rocket NPCs gone (post-E4) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1625 | byte 203 | 1 | `EVENT_ROUTE_43_GATE_ROCKETS` | Rocket grunts block Route 43 gate | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1655 | byte 206 | 7 | `EVENT_BLACKTHORN_CITY_GRAMPS_BLOCKS_DRAGONS_DEN` | Gramps blocks Dragon's Den entrance | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1661 | byte 207 | 5 | `EVENT_FOUGHT_SNORLAX` | Snorlax moved from route | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1568 | byte 196 | 0 | `EVENT_GOLDENROD_CITY_CIVILIANS` | Civilians reappear in Goldenrod (post-Rocket) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1569 | byte 196 | 1 | `EVENT_RADIO_TOWER_CIVILIANS_AFTER` | Civilians reappear in Radio Tower | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1623 | byte 202 | 15 | `EVENT_ROUTE_43_GATE_ROCKETS` | Rockets at Route 43 gate | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1597 | byte 199 | 5 | `EVENT_DAY_CARE_MAN_IN_DAY_CARE` | Day Care Man present | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1609 | byte 201 | 1 | `EVENT_ILEX_FOREST_FARFETCHD` | Farfetch'd present in Ilex Forest | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1610 | byte 201 | 2 | `EVENT_ROUTE_34_ILEX_FOREST_GATE_TEACHER_BEHIND_COUNTER` | Teacher at Ilex Forest gate (before Farfetch'd) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1611 | byte 201 | 3 | `EVENT_ROUTE_34_ILEX_FOREST_GATE_LASS` | Lass blocks Ilex Forest gate | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1612 | byte 201 | 4 | `EVENT_ROUTE_34_ILEX_FOREST_GATE_TEACHER_IN_WALKWAY` | Teacher moves to walkway (after Farfetch'd) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |

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
| 1540 | byte 192 | 4 | `EVENT_FOUGHT_SUICUNE` | Fought Suicune (Crystal-exclusive legendary) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1542 | byte 192 | 6 | `EVENT_GOT_RAINBOW_WING` | Obtained Rainbow Wing (triggers Ho-Oh) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1469 | byte 183 | 5 | `EVENT_SAW_SUICUNE_AT_CIANDWOOD_CITY` | Suicune appeared at Cianwood | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1470 | byte 183 | 6 | `EVENT_SAW_SUICUNE_ON_ROUTE_42` | Suicune appeared on Route 42 | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1471 | byte 183 | 7 | `EVENT_SAW_SUICUNE_ON_ROUTE_36` | Suicune appeared on Route 36 | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1473 | byte 184 | 1 | `EVENT_TIN_TOWER_1F_SUICUNE` | Suicune at Tin Tower 1F | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1474 | byte 184 | 2 | `EVENT_TIN_TOWER_1F_ENTEI` | Entei at Tin Tower 1F | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1475 | byte 184 | 3 | `EVENT_TIN_TOWER_1F_RAIKOU` | Raikou at Tin Tower 1F | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1539 | byte 192 | 3 | `EVENT_FOUGHT_EUSINE` | Fought Eusine (Crystal-exclusive NPC) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 1541 | byte 192 | 5 | `EVENT_KOJI_ALLOWS_YOU_PASSAGE_TO_TIN_TOWER` | Koji lets you pass to Tin Tower roof | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 145–148 | byte 18–19 | — | `EVENT_WALL_OPENED_IN_*_CHAMBER` | Ruins of Alph chamber walls opened | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |
| 140 | byte 17 | 4 | `EVENT_FOREST_IS_RESTLESS` | Ilex Forest is restless (Celebi event) | [event_flags.asm](https://github.com/pret/pokecrystal/blob/master/constants/event_flags.asm) |

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

## Additional Single-Byte RAM Features (Not Yet Tracked)

### Game Progress & Badges

| Address | Feature | Size | Scaling | Why It's Useful | Source |
|---|---|---|---|---|---|
| `D857` | **Johto badges** (bitmask: bit 0=Zephyr, 1=Hive, 2=Plain, 3=Fog, 4=Mineral, 5=Storm, 6=Glacier, 7=Rising) | 1 byte | /255 | **Highest impact** — compact ordinal progress through Johto storyline. Popcount = number of gyms beaten. | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d857), [ram_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/ram_constants.asm) |
| `D858` | **Kanto badges** (bitmask: bit 0=Boulder, 1=Cascade, 2=Thunder, 3=Rainbow, 4=Soul, 5=Marsh, 6=Volcano, 7=Earth) | 1 byte | /255 | Post-game progress through Kanto gyms | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d858), [ram_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/ram_constants.asm) |
| `D84C` | Unowndex status (`93` = unlocked) | 1 byte | /255 | Ruins of Alph puzzle completion | [GCL Forum - Sarial](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |
| `D84D` | Bug Catching Contest (`93`=not done, `97`=active) | 1 byte | /255 | Side quest tracking | [GCL Forum - Sarial](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |

### Player State & Movement

| Address | Feature | Scaling | Why It's Useful | Source |
|---|---|---|---|---|
| `D95D` | **Player state** (0=walk, 1=bike, 2=skate, 4=surf) | /255 | Indicates movement HMs acquired (surf = has HM03, bike = has Bicycle) | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d95d), [ram_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/ram_constants.asm) |
| `D4B6` | Day of week (0=Sunday–6=Saturday) | /7 | Time-gated daily events | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d4b6), [ram_constants.asm](https://github.com/pret/pokecrystal/blob/master/constants/ram_constants.asm) |
| `D4B7` | **Game hour** (0–23) | /255 | Time-gated events (morning/day/night Pokemon, certain NPCs) | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d4b7) |
| `D4B8` | Game minute | /60 | Fine-grained time | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d4b8) |
| `D4C4-D4C5` | **Play time hours** (16-bit LE) | /10000 | Total progress proxy — correlates with story advancement | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d4c4) |

### Battle Context

| Address | Feature | Scaling | Why It's Useful | Source |
|---|---|---|---|---|
| `D22D` | **Battle type** (0=none, 1=wild, 2=trainer) | /255 | **High value** — detect if in battle vs. exploring. Fundamentally changes what actions make sense. | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d22d) |
| `D230` | Wild battle type (07=shiny/can't escape, 08=Headbutt, etc.) | /255 | Encounter type detection | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d230) |
| `D233` | Enemy trainer type | /255 | Gym leader / Elite Four detection | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d233) |
| `C2A9` | **Currently playing BGM** | /255 | Context signal: battle music, city music, cave music, gym music. Changes every map/load. | [GCL Forum - Sarial](https://web.archive.org/web/20200901000000*/forum.glitchcity.info thread 125) |

### Inventory Signals

| Address | Feature | Scaling | Why It's Useful | Source |
|---|---|---|---|---|
| `D892` | Number of items (Item Pocket) | /25 | Inventory fullness | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d892) |
| `D8BC` | **Number of key items** (Key Pocket) | /25 | Key item count as story progress proxy | [GCL RAM map](https://gbdev.io/pokemem/?p=crystal#d8bc) |
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
| `gym_count` | `EVENT_BEAT_FALKNER` through `EVENT_BEAT_CLAIR` (flags 1030–1037) | popcount / 8 |
| `has_cut` | `EVENT_GOT_HM01_CUT` (flag 16) | 0 or 1 |
| `has_surf` | `EVENT_GOT_HM03_SURF` (flag 18) | 0 or 1 |
| `has_strength` | `EVENT_GOT_HM04_STRENGTH` (flag 19) | 0 or 1 |
| `has_flash` | `EVENT_GOT_HM05_FLASH` (flag 20) | 0 or 1 |
| `has_whirlpool` | `EVENT_GOT_HM06_WHIRLPOOL` (flag 21) | 0 or 1 |
| `rocket_cleared_hideout` | `EVENT_CLEARED_ROCKET_HIDEOUT` (flag 53) | 0 or 1 |
| `rocket_cleared_radio` | `EVENT_CLEARED_RADIO_TOWER` (flag 52) | 0 or 1 |
| `has_starter` | `EVENT_GOT_A_POKEMON_FROM_ELM` (flag 25) | 0 or 1 |
| `beat_elite_four` | `EVENT_BEAT_ELITE_FOUR` (flag 95) | 0 or 1 |

### Priority 3: Location disambiguation

Change goal matching from `[x, y, map_num]` to `[x, y, map_bank, map_num]` to avoid collisions between maps with the same number in different groups (e.g., map #5 in group 24 = Elm's Lab, map #5 in group 8 = Azalea Gym).
