# -*- coding: utf-8 -*-
"""The game side of the LLM agent: the real training env plus a readable view
of its state.

Every button press goes through ``PyBoyEnvironment.step`` with the env's own
action indices, so frame timing, the reward function and the observation are
exactly what PPO sees — the recorded (image, ram, action, reward) tuples are
training-format by construction.
"""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

from main import load_default_config, load_user_config, merge_configs
from PoliwhiRL.environment.gym_env import PyBoyEnvironment
from tools.rom_maps.rom_map_extractor import parse_map_constants

REPO = Path(__file__).resolve().parents[2]

# Buttons the agent may press: the policy's action set (gym_env.ACTIONS)
# without the no-op.
BUTTONS = ("up", "down", "left", "right", "a", "b", "start")
_DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
FACING = {1: "up", 2: "down", 3: "left", 4: "right"}

# On-screen tilemap (wTilemap, 20x18 tiles). Text boxes and menus are framed
# by the border tiles below; what is inside them is text in the game charmap.
TILEMAP = 0xC4A0
_BOX_SIDE = 0x7C
# The cursor tile, drawn immediately left of the option it points at.
_CURSOR_TILE = 0xED
# The battle menu is a fixed 2x2 grid: FIGHT PKMN over PACK RUN. The
# cursor moves up/down between rows and left/right between columns,
# and it does not wrap: right from the right-hand column does nothing.
BATTLE_MENU_GRID = {"FIGHT": (0, 0), "PKMN": (0, 1), "PACK": (1, 0), "RUN": (1, 1)}


def _charmap() -> dict:
    cm = {
        0x7F: " ",
        0xE0: "'",
        0xE3: "-",
        0xE6: "?",
        0xE7: "!",
        0xE8: ".",
        0xE9: "&",
        0xEA: "é",
        0xED: "▶",
        0xEE: "▼",
        0xEF: "♂",
        0xF0: "¥",
        0xF1: "×",
        0xF3: "/",
        0xF4: ",",
        0xF5: "♀",
        0x9A: "(",
        0x9B: ")",
        0x9C: ":",
        0x9D: ";",
        0x9E: "[",
        0x9F: "]",
        0xD0: "'d",
        0xD1: "'l",
        0xD2: "'m",
        0xD3: "'r",
        0xD4: "'s",
        0xD5: "'t",
        0xD6: "'v",
        0xE1: "PK",
        0xE2: "MN",
        0x54: "POKé",
        0x56: "……",
        0x75: "…",
        0xBA: "é",
    }
    cm.update({0x80 + i: chr(ord("A") + i) for i in range(26)})
    cm.update({0xA0 + i: chr(ord("a") + i) for i in range(26)})
    cm.update({0xF6 + i: str(i) for i in range(10)})
    return cm


CHARMAP = _charmap()
# Items: the sprite marks the object as something to pick up, not a person.
SPRITE_NAMES = {0x54: "Poké Ball (item: face it and press a)"}
# What every other object looks like (pokecrystal constants/sprite_constants.asm).
# Verified live: 0x0C Mom, 0x10 Elm, 0x29 the New Bark teacher, 0x2E the
# neighbour in the player's house, 0x3A the New Bark fisher, 0x3C Elm's aide,
# 0x40 Mr. Pokémon, 0x43 the officer in Elm's lab. Without these every NPC was
# "person", and a model looking for Elm took his aide for him and talked to
# her for an hour.
SPRITE_LOOKS = {
    0x01: "boy (like you)",
    0x03: "kid with a Game Boy",
    0x04: "red-haired boy",
    0x05: "PROF.OAK",
    0x06: "RED",
    0x07: "BLUE",
    0x08: "BILL",
    0x09: "elder",
    0x0A: "JANINE",
    0x0B: "KURT",
    0x0C: "MOM",
    0x0D: "BLAINE",
    0x0E: "RED's MOM",
    0x0F: "DAISY",
    0x10: "PROF.ELM",
    0x11: "WILL",
    0x12: "FALKNER",
    0x13: "WHITNEY",
    0x14: "BUGSY",
    0x15: "MORTY",
    0x16: "CHUCK",
    0x17: "JASMINE",
    0x18: "PRYCE",
    0x19: "CLAIR",
    0x1A: "BROCK",
    0x1B: "KAREN",
    0x1C: "BRUNO",
    0x1D: "MISTY",
    0x1E: "LANCE",
    0x1F: "LT.SURGE",
    0x20: "ERIKA",
    0x21: "KOGA",
    0x22: "SABRINA",
    0x23: "cooltrainer (man)",
    0x24: "cooltrainer (woman)",
    0x25: "bug catcher",
    0x26: "twin girl",
    0x27: "youngster",
    0x28: "lass",
    0x29: "teacher",
    0x2A: "beauty",
    0x2B: "super nerd",
    0x2C: "rocker",
    0x2D: "pokéfan (man)",
    0x2E: "pokéfan (woman)",
    0x2F: "old man",
    0x30: "old woman",
    0x31: "swimmer (man)",
    0x32: "swimmer (woman)",
    0x33: "big SNORLAX",
    0x35: "TEAM ROCKET grunt",
    0x36: "TEAM ROCKET grunt (woman)",
    0x37: "nurse",
    0x38: "receptionist",
    0x39: "clerk",
    0x3A: "fisherman",
    0x3B: "fishing guru",
    0x3C: "scientist",
    0x3D: "kimono girl",
    0x3E: "sage",
    0x40: "gentleman",
    0x41: "black belt",
    0x42: "receptionist",
    0x43: "police officer",
    0x44: "man",
    0x45: "SLOWPOKE",
    0x46: "captain",
    0x47: "big LAPRAS",
    0x48: "gym guide",
    0x49: "sailor",
    0x4A: "biker",
    0x4B: "pharmacist",
    0x4C: "Pokémon",
    0x4D: "Pokémon (fairy)",
    0x4E: "Pokémon (bird)",
    0x4F: "Pokémon (dragon)",
    0x50: "big ONIX",
    0x52: "tree-like Pokémon",
    0x55: "POKéDEX",
    0x56: "paper",
    0x59: "rock",
    0x5A: "boulder",
    0x5D: "fruit tree",
    0x5E: "gold trophy",
    0x5F: "silver trophy",
    0x60: "girl (like you)",
}
OBJECT_STRUCTS, OBJECT_STRUCT_LEN = 0xD4D6, 0x28
# wMapObjects: every object the map places, loaded or not (map_object
# struct, 16 bytes: object-struct index or 0xFF when not loaded, sprite,
# y+4, x+4, ..., event flag at +0x0C, little-endian, 0xFFFF = always shown).
# Entry 0 is the player. Verified live in Elm's lab: Elm (not loaded from the
# door) at its home cell, the aide loaded in object struct 1.
MAP_OBJECTS, MAP_OBJECT_LEN, NUM_MAP_OBJECTS = 0xD71E, 0x10, 16


def looks(sprite: int) -> str:
    """What an object looks like, from its sprite."""
    return SPRITE_NAMES.get(sprite) or SPRITE_LOOKS.get(sprite, "person")


PARTY_COUNT = 0xDCD7
PARTY_BASE = 0xDCDF  # party_struct x6, 0x30 bytes each
# wPartySpecies (0xDCD8..0xDCDE): one species byte per party slot, 0xFF
# terminator. An egg in the party reads EGG_SPECIES here, while its
# party_struct holds the hatch species (Togepi) — the only way to tell an
# egg from a hurt Pokémon (bugs_found.md, 2026-10-29: the nurse cannot
# heal an egg, so counting it as fainted nagged the model forever).
PARTY_SPECIES = 0xDCD8
EGG_SPECIES = 0xFD
# battle_struct (pokecrystal macros/ram.asm): species +0, level +0x0D, HP +0x10,
# max HP +0x12. wBattleMon holds the active mon's live HP; the party struct is
# only synced after the turn, so it lags a step behind the HP bar.
BATTLE_MON, ENEMY_MON = 0xC62C, 0xD206
BATTLE_TYPES = {0: "none", 1: "wild", 2: "trainer"}

# Local map legend (raw ROM collision values -> one character). Values from
# pokecrystal constants/collision_constants.asm; anything unlisted that is not
# walkable reads as a wall.
LEGEND = (
    "@ you  . floor  , tall grass (wild Pokemon)  # wall/obstacle  ~ water  "
    "E exit/door/stairs  P person  o item (Poke Ball)  > < ^ v ledge: standing on it and pressing "
    "that direction hops you over the next cell (one-way, cannot climb back)  "
    "+ next area (walk off this edge)"
)
_LEDGE_CHARS = {0xA0: ">", 0xA1: "<", 0xA2: "^", 0xA3: "v", 0xA4: "v", 0xA5: "v"}
# Screen geometry: 160x144 = 10x9 metatile-sized (16px) cells. When the player
# stands still the camera is tile-aligned and the player occupies the cell at
# column 4, row 4 (verified against live frames, see _player_cell_check).
CELL = 16
PLAYER_CELL = (4, 4)


@lru_cache(maxsize=1)
def species_names(rom_path: str) -> dict:
    """{species id: name} from the ROM's PokemonNames table (251 x 10 bytes,
    0x50-padded), found by searching for BULBASAUR in the game charmap."""
    rom = Path(rom_path).read_bytes()
    enc = {v: k for k, v in CHARMAP.items() if len(v) == 1}
    start = rom.find(bytes(enc[c] for c in "BULBASAUR") + b"\x50")
    if start < 0:
        return {}
    names = {}
    for i in range(251):
        raw = rom[start + 10 * i : start + 10 * i + 10].split(b"\x50")[0]
        names[i + 1] = "".join(CHARMAP.get(b, "?") for b in raw)
    return names


@lru_cache(maxsize=1)
def map_names() -> dict:
    return {(m.group, m.number): m.name for m in parse_map_constants()}


def map_name(bank: int, num: int) -> str:
    return map_names().get((bank, num), f"MAP_{bank}_{num}")


def _row_words(row: list[int]) -> list[tuple[str, int]]:
    """The words in a tilemap row, as (text, first column). A space, the
    box border and the cursor tile all end a word: the cursor is drawn
    immediately left of the option it points at."""
    out = []
    c = 0
    while c < len(row):
        if row[c] in (0x7F, _BOX_SIDE, _CURSOR_TILE):
            c += 1
            continue
        start = c
        text = []
        while c < len(row) and row[c] not in (0x7F, _BOX_SIDE, _CURSOR_TILE):
            text.append(CHARMAP.get(row[c], ""))
            c += 1
        word = "".join(text).strip()
        if word:
            out.append((word, start))
    return out


def decode_menu(rows: list[list[int]], in_battle: bool = False) -> dict | None:
    """Decode the open menu from the raw tilemap rows (18 rows of 20 tiles).

    Returns None when no menu is open (no cursor on screen), otherwise
    ``label`` the option under the cursor, ``options`` the options the
    cursor can reach: the four battle-menu labels when ``in_battle`` and
    the label is one of them, otherwise the words on the cursor's row;
    ``cursor`` which of them the cursor is on; and ``grid`` the 2x2 battle
    menu (None for the vertical menus).

    The cursor is read from the tilemap, not from ``screen_text()``, which
    is lossy: it strips the box, joins the boxes of a row with three
    spaces and drops the spans it judges to be pictures.
    """
    cur = None
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            if t == _CURSOR_TILE:
                cur = (r, c)
                break
        if cur is not None:
            break
    if cur is None:
        return None
    r, c = cur
    words = _row_words(rows[r])
    label, cursor = "", 0
    for i, (word, start) in enumerate(words):
        if start >= c + 1:
            label, cursor = word, i
            break
    if not label:
        return None
    row_options = [w for w, _ in words]
    if in_battle and label in BATTLE_MENU_GRID:
        # The battle menu: a fixed 2x2 grid, whatever the cursor is on.
        options = list(BATTLE_MENU_GRID)
        return {
            "label": label,
            "options": options,
            "cursor": options.index(label),
            "grid": dict(BATTLE_MENU_GRID),
        }
    # Any other menu: the words on the cursor's row. Its other options sit
    # on other rows, and the screen text already shows them.
    return {"label": label, "options": row_options, "cursor": cursor, "grid": None}


def _ranges(axis: str, vals: list[int]) -> str:
    """[3, 4, 5, 9] -> "x=3-5, 9"."""
    if not vals:
        return ""
    vals = sorted(vals)
    runs, start = [], vals[0]
    for a, b in zip(vals, vals[1:] + [None]):
        if b != a + 1:
            runs.append(str(start) if start == a else f"{start}-{a}")
            start = b
    return f"{axis}=" + ", ".join(runs)


def build_config(config_path: str, overrides: dict | None = None) -> dict:
    config = merge_configs(load_default_config(), load_user_config(config_path))
    config.update(
        vision=True,
        record=False,
        goexplore_enabled=False,
        goexplore_flag_capture=False,
        reverse_curriculum_enabled=False,
    )
    # episode_length and the stagnation watchdogs keep their training values so
    # the observation (stagnation_clock, ...) matches what a policy sees. The
    # harness ignores `done` and records where training would have ended the
    # episode (Game.training_end).
    config.update(overrides or {})
    return config


class Game:
    def __init__(
        self, config_path: str, display: bool = False, state_path: str | None = None
    ):
        overrides = {"state_path": state_path} if state_path else None
        self.config = build_config(config_path, overrides)
        self.env = PyBoyEnvironment(self.config, force_window=display)
        self.warps = self.env._collision_field  # .warps(bank, num)
        self.obs = self.env.get_observation()
        self._breakdown = dict(self.env.reward_calculator.get_episode_breakdown())
        self.seeded = False
        # Where the training env would have ended this episode (first time
        # done/truncated fired), or None. Reset with each new episode.
        self.training_end = None

    # -- state -------------------------------------------------------------
    def vars(self) -> dict:
        return self.env.ram.get_variables()

    def wram(self, addr: int) -> int:
        return self.env.ram.get_memory_value(addr)

    def flag(self, n: int, v: dict | None = None) -> bool:
        flags = (v or self.vars())["story_flags"]
        return bool((int(flags[n // 8]) >> (n % 8)) & 1)

    def party_count(self) -> int:
        return min(self.wram(PARTY_COUNT), 6)

    def where(self, v: dict | None = None) -> tuple[int, int, int, int]:
        v = v or self.vars()
        return int(v["map_bank"]), int(v["map_num"]), int(v["X"]), int(v["Y"])

    def snapshot(self) -> bytes:
        buf = io.BytesIO()
        self.env.pyboy.save_state(buf)
        return buf.getvalue()

    def restart_from(self, state: bytes):
        """Start a fresh (seeded) episode from a snapshot: new reward
        calculator, same as a Go-Explore restart in training."""
        self.env.state_bytes_content = state
        self.env._next_episode_seeded = True
        self.obs = self.env.reset()
        self._breakdown = dict(self.env.reward_calculator.get_episode_breakdown())
        self.seeded = True
        self.training_end = None

    # -- acting ------------------------------------------------------------
    def press(self, button: str) -> dict:
        """One env step. Returns the training tuple plus this step's reward
        broken down by source."""
        action = self.env.actions.index(button)
        prev_obs = self.obs
        obs, reward, done, truncated = self.env.step(action)
        self.obs = obs
        bd = dict(self.env.reward_calculator.get_episode_breakdown())
        delta = {
            k: round(float(bd.get(k, 0.0)) - float(self._breakdown.get(k, 0.0)), 4)
            for k in bd
        }
        self._breakdown = bd
        rc = self.env.reward_calculator
        if (done or truncated) and self.training_end is None:
            self.training_end = {
                "env_step": int(rc.steps),
                "cause": getattr(rc, "truncation_cause", None)
                or ("goal" if done and not truncated else "unknown"),
            }
        return {
            "image": prev_obs["image"],
            "ram": prev_obs["ram"],
            "action": action,
            "reward": float(reward),
            "sources": {k: v for k, v in delta.items() if v},
            # What the training env reports for this step, for debugging the
            # env and the reward from the harness's logs.
            "env": {
                "env_step": int(getattr(rc, "steps", 0)),
                "done": bool(done),
                "truncated": bool(truncated),
                "cause": (
                    getattr(rc, "truncation_cause", None)
                    if (done or truncated)
                    else None
                ),
                "frames": int(getattr(self.env, "last_step_frames", 0) or 0),
                "episode_reward": round(float(bd.get("delivered_total", 0.0)), 4),
            },
        }

    # -- perception --------------------------------------------------------
    def frame(self) -> Image.Image:
        return self.env.pyboy.screen.image.convert("RGB")

    def llm_image(self, scale: int = 1, grid: bool = True) -> Image.Image:
        """The screen at ``scale`` (1 = the native 160x144) with a faint cell
        grid and the player's cell boxed, so the model can count cells to
        things it sees. Pass grid=False off the overworld (battle, keyboard):
        there the box marks nothing. Native is the default: a vision model's
        tokens scale with pixels, and the upscale bought nothing, because the
        model reads an 8x8 sprite at either size."""
        img = self.frame().resize((160 * scale, 144 * scale), Image.NEAREST)
        if grid:
            d = ImageDraw.Draw(img, "RGBA")
            c = CELL * scale
            for i in range(1, 10):
                d.line([(i * c, 0), (i * c, img.height)], fill=(255, 0, 0, 40), width=1)
            for j in range(1, 9):
                d.line([(0, j * c), (img.width, j * c)], fill=(255, 0, 0, 40), width=1)
            px, py = PLAYER_CELL
            d.rectangle(
                [px * c, py * c, (px + 1) * c - 1, (py + 1) * c - 1],
                outline=(0, 120, 255, 220),
                width=max(1, scale),
            )
        return img

    @staticmethod
    def png_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # -- screen text ------------------------------------------------------------
    def tilemap(self) -> list[list[int]]:
        """The 18x20 on-screen tilemap, row by row."""
        return [[self.wram(TILEMAP + r * 20 + c) for c in range(20)] for r in range(18)]

    def menu_info(self, in_battle: bool = False) -> dict | None:
        """The open menu decoded from the tilemap, or None when no menu
        is open. See ``decode_menu``."""
        return decode_menu(self.tilemap(), in_battle=in_battle)

    def screen_text(self) -> list[str]:
        """Exact text inside on-screen text boxes and menus, top to bottom,
        decoded from the tilemap (not OCR)."""
        lines = []
        for r in range(18):
            row = [self.wram(TILEMAP + r * 20 + c) for c in range(20)]
            sides = [c for c, t in enumerate(row) if t == _BOX_SIDE]
            if len(sides) < 2:
                continue
            spans = list(zip(sides[:-1], sides[1:]))
            parts = []
            for a, b in spans:
                inner = [t for t in row[a + 1 : b] if t != 0x7F]
                steps = {q - p for p, q in zip(inner, inner[1:])}
                if len(inner) >= 4 and len(steps) == 1 and steps.pop() >= 2:
                    continue  # a picture (consecutive graphics tiles), not text
                text = "".join(CHARMAP.get(t, "") for t in row[a + 1 : b]).strip()
                if sum(ch.isalnum() for ch in text) >= 1 or "▶" in text:
                    parts.append(text)
            if parts:
                lines.append("   ".join(parts))
        return lines

    def naming_screen(self) -> str | None:
        """The name typed so far if the naming keyboard is open, else None.
        Detected by its bottom row (lower / DEL / END); the name is the row
        holding the typed letters under the 'NICKNAME?'/'NAME?' prompt."""
        rows = [
            "".join(
                CHARMAP.get(self.wram(TILEMAP + r * 20 + c), " ") for c in range(20)
            )
            for r in range(18)
        ]
        if not any("DEL" in r and "END" in r for r in rows):
            return None
        for i, r in enumerate(rows):
            if "NAME" in r and i + 2 < len(rows):
                return rows[i + 2].strip()
        return ""

    def objects(self) -> list[tuple[int, int, int]]:
        """(x, y, sprite) for each loaded map object."""
        out = []
        for i in range(1, 13):
            base = OBJECT_STRUCTS + OBJECT_STRUCT_LEN * i
            sprite = self.wram(base)
            if sprite:
                out.append(
                    (self.wram(base + 0x10) - 4, self.wram(base + 0x11) - 4, sprite)
                )
        return out

    def map_objects(self, v: dict | None = None) -> list[dict]:
        """Every object on this map that is not hidden by its event flag.

        The engine only loads an object into an object struct while it is
        near the camera, so ``objects()`` alone hides most of a room: from
        the door of Elm's lab only the aide is loaded, and Elm is not. Loaded
        objects are reported where they stand now; the rest where the map
        places them (``loaded`` False)."""
        flags = (v or self.vars())["story_flags"]
        out = []
        for i in range(1, NUM_MAP_OBJECTS):
            base = MAP_OBJECTS + MAP_OBJECT_LEN * i
            sprite = self.wram(base + 1)
            if not sprite:
                continue
            struct = self.wram(base)
            if 1 <= struct < 13:
                sb = OBJECT_STRUCTS + OBJECT_STRUCT_LEN * struct
                if self.wram(sb):
                    out.append(
                        {
                            "x": self.wram(sb + 0x10) - 4,
                            "y": self.wram(sb + 0x11) - 4,
                            "sprite": self.wram(sb),
                            "loaded": True,
                        }
                    )
                    continue
            flag = self.wram(base + 0x0C) | (self.wram(base + 0x0D) << 8)
            if (
                flag != 0xFFFF
                and flag // 8 < len(flags)
                and (int(flags[flag // 8]) >> (flag % 8)) & 1
            ):
                continue  # hidden: picked up, gone, or not here yet
            out.append(
                {
                    "x": self.wram(base + 3) - 4,
                    "y": self.wram(base + 2) - 4,
                    "sprite": sprite,
                    "loaded": False,
                }
            )
        return out

    # -- local map --------------------------------------------------------------
    def _collision(self):
        return self.env._collision_field

    def _crossings(self, bank: int, num: int) -> dict:
        """{"west": ((bank, num) of the next map, [(x, y) edge cells you can
        walk across]), ...} for each connected map edge."""
        cf = self._collision()
        gm, ts, conns = cf._meta(bank, num)
        off = cf._collision_offset(self.env.pyboy.memory)
        w, h = gm.width * 2, gm.height * 2
        names = {"N": "north", "S": "south", "W": "west", "E": "east"}
        out = {}
        for d, c in conns.items():
            if d in ("N", "S"):
                cells = [(x, 0 if d == "N" else h - 1) for x in range(w)]
            else:
                cells = [(0 if d == "W" else w - 1, y) for y in range(h)]
            ok = [
                (x, y)
                for x, y in cells
                if any(
                    mp == c[0]
                    for (mp, _nx, _ny), *_ in cf._edges(off, ts, ((bank, num), x, y))
                )
            ]
            out[names[d]] = (c[0], ok)
        return out

    def routes(self, v: dict | None = None, max_people: int = 6) -> list[dict]:
        """Shortest walking route from the player to each door/stairs, each
        crossable map edge, and the cell next to each of the nearest
        ``max_people`` people/items on this map, from the ROM collision data
        (ledges one-way, people block). [{"to", "via", "steps", "path",
        "first"}], plus "face" (the direction to face on arrival) for a
        person; steps None = no walking route from here. Overworld only."""
        v = v or self.vars()
        if v["script_active"] or int(v["battle_type"]):
            return []
        bank, num, x0, y0 = self.where(v)
        cf = self._collision()
        mem = self.env.pyboy.memory
        here = (bank, num)
        people = self.map_objects(v)
        blocked = {(here, p["x"], p["y"]) for p in people}
        # (to, via, target cells, None | edge name | person cell)
        targets: list[tuple] = [
            (map_name(g, n), f"door at ({wx},{wy})", {(wx, wy)}, None)
            for wx, wy, g, n in self.warps.warps(bank, num)
        ]
        try:
            for d, (to, cells) in self._crossings(bank, num).items():
                if cells:
                    targets.append((map_name(*to), f"{d} edge", set(cells), d))
        except Exception:
            pass
        near = sorted(people, key=lambda p: abs(p["x"] - x0) + abs(p["y"] - y0))[
            :max_people
        ]
        for p in near:
            what = looks(p["sprite"]).split(" (item")[0]
            cells = {(p["x"] + dx, p["y"] + dy) for dx, dy in _DIRS.values()}
            targets.append(
                (what, f"at ({p['x']},{p['y']})", cells, (p["x"], p["y"], p["sprite"]))
            )
        out = []
        for to, via, cells, edge in targets:
            field = cf.distance_field(
                mem,
                v,
                lambda b, n, x, y, c=cells: (b, n) == here and (x, y) in c,
                blocked=blocked,
            )
            start = (here, x0, y0)
            across = False
            if isinstance(edge, tuple) and (field is None or start not in field[0]):
                # Nobody can stand next to a nurse or a clerk: they are
                # talked to across the counter, from two cells away.
                px, py = edge[0], edge[1]
                far = {(px + 2 * dx, py + 2 * dy) for dx, dy in _DIRS.values()}
                field = cf.distance_field(
                    mem,
                    v,
                    lambda b, n, x, y, c=far: (b, n) == here and (x, y) in c,
                    blocked=blocked,
                )
                across = True
            if field is None or start not in field[0]:
                row = {"to": to, "via": via, "steps": None, "path": ""}
                if isinstance(edge, tuple):
                    row.update(person=True, item=edge[2] in SPRITE_NAMES)
                out.append(row)
                continue
            dist, moves = field
            node, legs = start, []
            while dist.get(node):
                nxt = min(
                    (e for e in moves.get(node, ()) if e[0] in dist),
                    key=lambda e: dist[e[0]],
                )
                node, dx, dy, step = nxt
                name = {
                    (1, 0): "right",
                    (-1, 0): "left",
                    (0, 1): "down",
                    (0, -1): "up",
                }[(dx, dy)]
                if step == 2:
                    legs.append([name + " (hop the ledge)", 1])
                elif legs and legs[-1][0] == name:
                    legs[-1][1] += 1
                else:
                    legs.append([name, 1])
            path = ", ".join(f"{n} {k}" if "hop" not in n else n for n, k in legs)
            first = legs[0][0].split(" ")[0] if legs else ""
            row = {
                "to": to,
                "via": via,
                "steps": dist[start],
                "first": first,
                "path": path,
            }
            if isinstance(edge, tuple):
                # A person: the route ends next to them; facing them is one
                # more press (it turns you without moving), then a.
                _, ex, ey = node
                px, py, sprite = edge
                ddx, ddy = px - ex, py - ey
                row["face"] = {
                    (1, 0): "right",
                    (-1, 0): "left",
                    (0, 1): "down",
                    (0, -1): "up",
                }[((ddx > 0) - (ddx < 0), (ddy > 0) - (ddy < 0))]
                row["person"] = True
                row["across"] = across
                row["item"] = sprite in SPRITE_NAMES
            elif edge:
                off_edge = dict(west="left", east="right", north="up", south="down")[
                    edge
                ]
                row["path"] = (
                    f"{path}, then {off_edge} off the edge"
                    if path
                    else f"{off_edge} off the edge"
                )
                row["first"] = first or off_edge
            out.append(row)
        return out

    def connections(self, bank: int, num: int) -> dict:
        """{"north": {"to": map name, "cross": "x=10-11"}, ...} for each
        connected map edge. ``cross`` lists the edge cells you can actually
        walk across (both sides walkable); "" means none on foot (water, a
        fence), so the edge is a connection on paper only."""
        cf = self._collision()
        try:
            gm, ts, conns = cf._meta(bank, num)
            off = cf._collision_offset(self.env.pyboy.memory)
        except Exception:
            return {}
        w, h = gm.width * 2, gm.height * 2
        names = {"N": "north", "S": "south", "W": "west", "E": "east"}
        out = {}
        for d, c in conns.items():
            if d in ("N", "S"):
                axis, cells = "x", [(x, 0 if d == "N" else h - 1) for x in range(w)]
            else:
                axis, cells = "y", [(0 if d == "W" else w - 1, y) for y in range(h)]
            ok = []
            for x, y in cells:
                for (mp, _nx, _ny), *_ in cf._edges(off, ts, ((bank, num), x, y)):
                    if mp == c[0]:
                        ok.append(x if axis == "x" else y)
                        break
            out[names[d]] = {"to": map_name(*c[0]), "cross": _ranges(axis, ok)}
        return out

    def local_map(self, v: dict, rx: int = 4, ry: int = 4) -> list[str]:
        """ASCII walkability around the player from the ROM collision decode,
        with exits, people and ledges marked. Rows are top (up) to bottom."""
        cf = self._collision()
        bank, num, x0, y0 = self.where(v)
        try:
            gm = cf._decode(bank, num)
            _gm, _ts, conns = cf._meta(bank, num)
            off = cf._collision_offset(self.env.pyboy.memory)
        except Exception:
            return []
        warps = {(wx, wy) for wx, wy, *_ in self.warps.warps(bank, num)}
        people = {
            (p["x"], p["y"]): ("o" if p["sprite"] in SPRITE_NAMES else "P")
            for p in self.map_objects(v)
        }
        w, h = gm.width * 2, gm.height * 2
        rows = []
        for dy in range(-ry, ry + 1):
            row = []
            for dx in range(-rx, rx + 1):
                x, y = x0 + dx, y0 + dy
                if (dx, dy) == (0, 0):
                    ch = "@"
                elif not (0 <= x < w and 0 <= y < h):
                    side = "W" if x < 0 else "E" if x >= w else "N" if y < 0 else "S"
                    ch = "+" if side in conns else "#"
                elif (x, y) in people:
                    ch = people[(x, y)]
                elif (x, y) in warps:
                    ch = "E"
                else:
                    val = cf._value(gm, off, x, y)
                    if val == 0x00:
                        ch = "."
                    elif val in (0x14, 0x18):
                        ch = ","
                    elif val is not None and 0x70 <= val <= 0x7F:
                        ch = "E"
                    elif val in _LEDGE_CHARS:
                        ch = _LEDGE_CHARS[val]
                    elif val is not None and 0x20 <= val <= 0x2F:
                        ch = "~"
                    else:
                        ch = "#"
                row.append(ch)
            rows.append(" ".join(row))
        return rows

    # -- party / battle -----------------------------------------------------------
    def _be(self, addr: int, n: int) -> int:
        out = 0
        for i in range(n):
            out = (out << 8) | self.wram(addr + i)
        return out

    def party(self) -> list[dict]:
        out = []
        for i in range(self.party_count()):
            base = PARTY_BASE + 0x30 * i
            out.append(
                {
                    "level": self.wram(base + 0x1F),
                    "hp": self._be(base + 0x22, 2),
                    "max_hp": self._be(base + 0x24, 2),
                    "egg": self.wram(PARTY_SPECIES + i) == EGG_SPECIES,
                }
            )
        return out

    def battle_mon(self, base: int) -> dict:
        species = self.wram(base)
        return {
            "species": species_names(self.config["rom_path"]).get(
                species, f"#{species}"
            ),
            "level": self.wram(base + 0x0D),
            "hp": self._be(base + 0x10, 2),
            "max_hp": self._be(base + 0x12, 2),
        }

    def _neighbour(self, v, dx, dy) -> str:
        x, y = int(v["X"]) + dx, int(v["Y"]) + dy
        for ox, oy, sprite in self.objects():
            if (ox, oy) == (x, y):
                return looks(sprite).split(" (item")[0]
        bank, num = int(v["map_bank"]), int(v["map_num"])
        if any((wx, wy) == (x, y) for wx, wy, *_ in self.warps.warps(bank, num)):
            return "exit"
        key = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}[
            (dx, dy)
        ]
        val = int(v[f"collision_{key}"])
        if val in (0x00, 0x14, 0x18):
            return "open"
        if 0x70 <= val <= 0x7F:
            return "exit"
        if 0xA0 <= val <= 0xA5:
            return "ledge"
        return "blocked"

    def _on_exit(self, v, bank, num, x, y):
        """The warp the player is standing on, and which directions use it
        (the map edges it touches), or None."""
        for wx, wy, g, n in self.warps.warps(bank, num):
            if (wx, wy) != (x, y):
                continue
            try:
                gm = self._collision()._decode(bank, num)
                w, h = gm.width * 2, gm.height * 2
            except Exception:
                return {"to": map_name(g, n), "leave": []}
            leave = [
                d
                for d, edge in (
                    ("down", y == h - 1),
                    ("up", y == 0),
                    ("left", x == 0),
                    ("right", x == w - 1),
                )
                if edge
            ]
            return {"to": map_name(g, n), "leave": leave}
        return None

    def status(self) -> dict:
        """The structured facts shown to the model each turn."""
        v = self.vars()
        bank, num, x, y = self.where(v)
        exits = []
        for wx, wy, g, n in self.warps.warps(bank, num):
            exits.append(
                {"x": wx, "y": wy, "to": map_name(g, n), "dx": wx - x, "dy": wy - y}
            )
        # Every object on the map, not only those near the camera: from the
        # door of Elm's lab the old 11x9 window held only the aide, so the
        # model took her for Elm.
        people = sorted(
            (
                {
                    "x": p["x"],
                    "y": p["y"],
                    "dx": p["x"] - x,
                    "dy": p["y"] - y,
                    "what": looks(p["sprite"]),
                    "loaded": p["loaded"],
                }
                for p in self.map_objects(v)
            ),
            key=lambda p: abs(p["dx"]) + abs(p["dy"]),
        )
        around = {d: self._neighbour(v, dx, dy) for d, (dx, dy) in _DIRS.items()}
        on_exit = self._on_exit(v, bank, num, x, y)
        if on_exit:
            # Doormat-style exits sit on the map edge and fire when you walk
            # off it; the engine reads that off-map cell as a wall, which
            # otherwise tells the model the way out is blocked. Stated as a
            # fact, not an order: "press down to leave" sent gemma straight
            # back out of Elm's lab on 38 of 49 arrivals.
            for d in on_exit["leave"]:
                around[d] = f"exit to {on_exit['to']} ({d} leaves this map)"
        battle = None
        if int(v["battle_type"]):
            battle = {
                "type": BATTLE_TYPES.get(int(v["battle_type"]), str(v["battle_type"])),
                "enemy": self.battle_mon(ENEMY_MON),
                "you": self.battle_mon(BATTLE_MON),
                "move_pp": list(v["player_move_pp"]),
            }
        text = self.screen_text()
        naming = self.naming_screen()
        mons = self.party()
        try:
            # The same read the policy's observation uses (party structs and
            # the ROM move table); read-only.
            hp_frac, pp_frac = self.env.ram.get_party_resources()[:2]
        except Exception:
            hp_frac = pp_frac = None
        menu = any("▶" in line for line in text) and not battle
        # The cursor is decoded from the tilemap, not from the text: the
        # text is lossy, and in a battle the menu flag is off on purpose.
        menu_info = self.menu_info(in_battle=bool(battle))
        # What the screen is for, which decides what the prompt shows: the
        # overworld facts (neighbours, doors, local map) and the player box
        # only mean something while walking.
        if battle:
            mode = "battle"
        elif naming is not None:
            mode = "naming"
        elif menu:
            mode = "menu"
        elif v["script_active"] or text:
            # A trainer's challenge ("You're a POKéMON trainer, right?") is
            # on screen with the script byte still 0.
            mode = "text"
        else:
            mode = "walk"
        return {
            "mode": mode,
            "battle": battle,
            "text": text,
            "menu": menu,
            "menu_info": menu_info,
            "naming": naming,
            "party_mons": mons,
            "party_hp_frac": hp_frac,
            "party_pp_frac": pp_frac,
            # An egg reads as HP 0/19 in its struct but is not hurt and no
            # nurse can heal it; only real Pokémon count as fainted.
            "fainted": sum(
                1 for m in mons if not m["egg"] and m["max_hp"] and m["hp"] == 0
            ),
            "pokeballs": int(v["pokeball_count"]),
            "money": int(v["money"]),
            "badges": bin(int(v["johto_badges"])).count("1"),
            "pokedex_owned": int(v["pokedex_owned"]),
            "connections": self.connections(bank, num),
            "local_map": self.local_map(v),
            "map": map_name(bank, num),
            "map_id": (bank, num),
            "x": x,
            "y": y,
            "facing": FACING.get(int(v["player_direction"]), "?"),
            # script_byte == 255 for the whole of a scene/text/yes-no prompt
            # and 0 once the player can walk again (ui_byte is 5 in the
            # overworld, so it is not a text-box signal).
            "text_open": bool(v["script_active"]),
            "in_battle": int(v["battle_type"]) != 0,
            "party": self.party_count(),
            "exits": exits,
            "people": people,
            "around": around,
            "on_exit": on_exit,
        }

    def close(self):
        self.env.close()
