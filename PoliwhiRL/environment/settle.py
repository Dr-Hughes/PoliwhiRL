# -*- coding: utf-8 -*-
"""Event-driven ("settle") stepping: one env step = press, then run the
emulator until the game is waiting for input again.

A fixed frame-skip (90 frames, 75 after a press — ``pyboy.button(delay=)``
queues the release without ticking) left most of every step idle (a one-tile
walk is ready after ~20 frames) yet still split one text box over two steps
and a battle turn over 3-8, and could capture a stairs warp mid-fade. Here a
step ends the first frame one of the game's own input-wait routines runs:

- text prompt / menus: ``JoyWaitAorB``, ``JoyTextDelay``, ``GetMenuJoypad``,
  ``GetScrollingMenuJoypad`` (a menu that reprints its description every frame
  also calls ``PrintLetterDelay``, so these win over it);
- other screens that poll ``GetJoypad`` directly (Pokémon summary, ...), but
  not within ``_LETTER_GUARD`` frames of a ``PrintLetterDelay`` call: at
  MID/SLOW text speed it polls the joypad between letters;
- the overworld: ``DoPlayerMovement`` with the player standing
  (``wWalkingDirection`` 0xFF, ``wPlayerStepFlags`` 0) — it also runs at the
  tile boundary of a continuing walk.

Two more conditions stop a step ending early (verification 2026-10-28: on
1,000 sampled states the plain rules above dropped the next press in 17% of
menu and 7% of text-box states and left 30-40% of menu frames half drawn):

- ``hJoyDown`` must be 0: the game has polled since the release, else a
  repeat of the same button has no press edge and is ignored (start menu,
  options: B, B or START, START);
- two ready frames at most ``_POLL_GAP`` apart with the tilemap and the
  background palettes unchanged between them: a single poll during screen
  setup (the Pack polls once before drawing its item list, the Pokédex after
  a white fade) is not the input loop, and a fade (battle exit to white) can
  poll twice over a fixed tilemap while the palettes still change. The gap is
  5 because the Mart quantity screen polls every 5th frame.

While the game plays itself (``wInputType`` = AUTO_INPUT: the Dude's
catching demo on Route 29 feeds scripted joypad input for ~1,600 frames) the
player's presses do nothing, so it is never ready and those frames do not
count towards the cap (bounded at ``10 * cap``): the demo is one step, not ~16
inert ones. ``auto_frames`` says how much of the last step was such a demo.

Then ``tail`` more frames so the last scroll renders, capped at ``cap``
frames (a successful Poké Ball catch takes ~760). A step is never shorter
than ``min_frames`` (hold + 1): a 15-frame press turns and walks exactly one
tile (hold <= 9 only turns, >= 18 walks two; the overworld polls input every
second frame). In menus (``hInMenu`` set) the hold is ``menu_hold`` instead:
menus auto-repeat a held button after 15 frames, so a 15-frame hold
occasionally switched Pack pockets twice.

The hooks are code addresses of pret pokecrystal 1.0 (``pokecrystal.sym``),
so the stepper only arms for that ROM (``ROM_SHA1``); any other ROM falls back
to the fixed frame-skip. Measured 2026-10-28 (frame-skip review): dialogue
needs one step per text box (Elm 64 -> 40 steps), a wild battle ~11 steps
instead of ~26, walking stays one tile per step at ~20 frames instead of 75.
Most of the dialogue saving is the instant-text options byte (fixed stepping
with it: Elm 41 steps); settle stepping saves frames there (~2.8x).
"""

import hashlib

# sha1 of Pokemon - Crystal Version (UE) (V1.0), pret pokecrystal 1.0.
ROM_SHA1 = "f4cd194bdee0d04ca4eac29e09b8e4e9d818c133"

# (bank, address) from pokecrystal.sym.
_GET_JOYPAD = (0x00, 0x0984)
_JOY_WAIT_A_OR_B = (0x00, 0x0A36)
_JOY_TEXT_DELAY = (0x00, 0x0A57)
_GET_MENU_JOYPAD = (0x00, 0x1BDD)
_GET_SCROLLING_MENU_JOYPAD = (0x00, 0x1F1A)
_PRINT_LETTER_DELAY = (0x00, 0x313D)
_DO_PLAYER_MOVEMENT = (0x20, 0x4000)
_HANDLE_MAP_TIME_AND_JOYPAD = (0x25, 0x67C1)
W_WALKING_DIRECTION = 0xD043  # WRAM bank 1
W_PLAYER_STEP_FLAGS = 0xD150  # WRAM bank 1
H_JOY_DOWN = 0xFFA8
H_IN_MENU = 0xFFAA
W_TILEMAP, TILEMAP_LEN = 0xC4A0, 20 * 18
W_BG_PALS2 = 0xD080  # WRAM bank 5, 8 palettes x 8 bytes
W_INPUT_TYPE, AUTO_INPUT = 0xC2C7, 0xFF
# Options byte: bits 0-2 text delay, bit 4 = no per-letter delay (instant
# text: PrintLetterDelay returns at once), bit 6 = battle style SET, bit 7 =
# battle scene off. SET drops the "Will <player> change POKéMON?" prompt
# after each enemy faint, whose YES (the default) opens a party screen that
# loops on "is already out" for as long as A is pressed (2026-10-28, both
# Route 30 trainer battles); only B on that screen escapes.
W_OPTIONS = 0xCFCC
FAST_OPTIONS = 0xD1

# Hook-hit bits.
_JOY, _WAIT, _LETTER, _MOVE, _MAPJOY = 1, 2, 4, 8, 16
_LETTER_GUARD = 6
_POLL_GAP = 5


def rom_sha1(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


class SettleStepper:
    def __init__(self, pyboy, hold=15, menu_hold=6, cap=1200, tail=2, min_frames=None):
        self.pyboy = pyboy
        self.hold = int(hold)
        self.menu_hold = int(menu_hold)
        self.cap = int(cap)
        self.tail = int(tail)
        if self.tail < 1:
            raise ValueError("settle tail must be >= 1: only the last tail frame is rendered")
        self.min_frames = self.hold + 1 if min_frames is None else int(min_frames)
        self.capped_steps = 0
        self.auto_frames = 0
        self.episode_frames = 0
        self.episode_auto_frames = 0
        self._hits = 0
        self._frame = 0
        self._last_letter = -100
        for addr, bit in (
            (_GET_JOYPAD, _JOY),
            (_JOY_WAIT_A_OR_B, _WAIT),
            (_JOY_TEXT_DELAY, _WAIT),
            (_GET_MENU_JOYPAD, _WAIT),
            (_GET_SCROLLING_MENU_JOYPAD, _WAIT),
            (_PRINT_LETTER_DELAY, _LETTER),
            (_DO_PLAYER_MOVEMENT, _MOVE),
            (_HANDLE_MAP_TIME_AND_JOYPAD, _MAPJOY),
        ):
            pyboy.hook_register(addr[0], addr[1], self._hit, bit)

    def new_episode(self):
        """Forget the previous episode's letter timing and per-episode
        counts: a reset must not change the first step's ready check
        (deterministic replays and recordings)."""
        self._frame = 0
        self._last_letter = -100
        self.capped_steps = 0
        self.episode_frames = 0
        self.episode_auto_frames = 0

    def _hit(self, bit):
        self._hits |= bit

    def _tick(self, render):
        self._hits = 0
        self.pyboy.tick(1, render)
        self._frame += 1
        if self._hits & _LETTER:
            self._last_letter = self._frame
        return self._hits

    def _ready(self, h):
        if h & _WAIT:
            return True
        if h & _LETTER:
            return False
        if h & _JOY and not h & _MAPJOY and self._frame - self._last_letter > _LETTER_GUARD:
            return True
        mem = self.pyboy.memory
        return bool(
            h & _MOVE and mem[1, W_WALKING_DIRECTION] == 0xFF and mem[1, W_PLAYER_STEP_FLAGS] == 0
        )

    def step(self, button, render):
        """Press ``button`` (falsy = no press) and run until the game waits
        for input. Returns the number of frames emulated. Only the last frame
        is rendered."""
        mem = self.pyboy.memory
        if button:
            hold = self.menu_hold if mem[H_IN_MENU] else self.hold
            self.pyboy.button(button, delay=hold)
        n = 0
        self.auto_frames = 0
        last_ready, last_screen = None, None
        while True:
            h = self._tick(False)
            n += 1
            if mem[W_INPUT_TYPE] == AUTO_INPUT:
                self.auto_frames += 1
                last_ready = None
                if n < 10 * self.cap:
                    continue
            if n >= self.min_frames and self._ready(h) and mem[H_JOY_DOWN] == 0:
                screen = (
                    mem[W_TILEMAP : W_TILEMAP + TILEMAP_LEN],
                    mem[5, W_BG_PALS2 : W_BG_PALS2 + 64],
                )
                if last_ready is not None and n - last_ready <= _POLL_GAP and screen == last_screen:
                    break
                last_ready, last_screen = n, screen
            if n - self.auto_frames >= self.cap:
                self.capped_steps += 1
                break
        for i in range(self.tail):
            self._tick(render and i == self.tail - 1)
            n += 1
        self.episode_frames += n
        self.episode_auto_frames += self.auto_frames
        return n
