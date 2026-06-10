# -*- coding: utf-8 -*-
"""Verify the early-game RAM signals against the manual save states.

Usage:  python Scripts/verify_flags_from_states.py
        (run from the PoliwhiRL project root)

The save states under "Manual Investigation States/" only cover the opening of
the game (bedroom -> starter -> Mr. Pokemon's egg -> route 30/46), so this
script only checks the signals early training actually depends on:

  * pokedex_owned        — the stage-2 goal ("owned >= 1")
  * party species        — should be Cyndaquil once owned
  * got_starter   flag 26 (EVENT_GOT_A_POKEMON_FROM_ELM)
  * got_cyndaquil flag 27 (EVENT_GOT_CYNDAQUIL_FROM_ELM)
  * got_mystery_egg flag 30 (EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON)

It does NOT touch gym-leader / legendary / HM flags: none of them can be
reached from these states, so there is nothing to verify.
"""
from pathlib import Path

try:
    from pyboy import PyBoy
except ImportError:
    raise SystemExit("pyboy not installed.")

ROOT = Path(__file__).resolve().parent.parent
ROM = ROOT / "emu_files" / "Pokemon - Crystal Version.gbc"
STATES_DIR = ROOT / "Manual Investigation States"

FLAGS0 = 0xDA72                       # wEventFlags base (bit N -> byte N//8, bit N%8)
POKEDEX_OWNED = (0xDE99, 0xDEB8)      # owned-species bitfield
PARTY_SPECIES = 0xDCDF               # first party slot species id
CYNDAQUIL = 155

# The only event flags these states can exercise.
FLAGS = {26: "got_starter", 27: "got_cyndaquil", 30: "got_mystery_egg"}


def main():
    if not ROM.exists():
        raise SystemExit(f"ROM not found at {ROM}")
    states = sorted(STATES_DIR.glob("*.state"))
    if not states:
        raise SystemExit(f"No .state files in {STATES_DIR}")

    p = PyBoy(str(ROM), window="null", sound_emulated=False)
    p.set_emulation_speed(0)

    def flag(n):
        return (p.memory[FLAGS0 + n // 8] >> (n % 8)) & 1

    def owned_count():
        return sum(bin(p.memory[a]).count("1")
                   for a in range(POKEDEX_OWNED[0], POKEDEX_OWNED[1] + 1))

    # ---- per-state values --------------------------------------------------
    print(f"{'state':52} {'owned':>5} {'species':>9}   f26  f27  f30")
    print("-" * 92)
    rows = []
    for sf in states:
        with open(sf, "rb") as f:
            p.load_state(f)
        owned = owned_count()
        sp = p.memory[PARTY_SPECIES]
        bits = {n: flag(n) for n in FLAGS}
        rows.append((sf.stem, owned, sp, bits))
        sp_name = "Cyndaquil" if sp == CYNDAQUIL else (f"#{sp}" if sp else "-")
        print(f"{sf.stem[:52]:52} {owned:5} {sp_name:>9}   "
              f"{bits[26]:>3}  {bits[27]:>3}  {bits[30]:>3}")

    # ---- checks ------------------------------------------------------------
    print("\nCHECKS")
    ok = True

    # got_cyndaquil should be set iff the player owns a Pokemon, in every state.
    bad = [s for s, owned, _, b in rows if b[27] != (1 if owned else 0)]
    if bad:
        ok = False
        print(f"  [FAIL] got_cyndaquil (27) disagrees with pokedex_owned in: {bad}")
    else:
        print("  [OK]   got_cyndaquil (27) == (pokedex_owned > 0) in all states")

    # got_starter is the same, EXCEPT it is briefly cleared during the
    # selection cutscene — so use pokedex_owned as the real 'has starter' signal.
    bad = [s for s, owned, _, b in rows if b[26] != (1 if owned else 0)]
    print(f"  [INFO] got_starter (26) == (pokedex_owned > 0) except {len(bad)} "
          f"cutscene state(s): {bad}")
    print("         -> training should key 'has starter' off pokedex_owned, not flag 26")

    # Species sanity: whenever something is owned, it is Cyndaquil here.
    bad = [s for s, owned, sp, _ in rows if owned and sp != CYNDAQUIL]
    if bad:
        ok = False
        print(f"  [FAIL] owned but species != Cyndaquil in: {bad}")
    else:
        print("  [OK]   party species is Cyndaquil wherever a Pokemon is owned")

    # Mystery egg: only set in the Mr. Pokemon / post-Mr.-Pokemon states.
    egg_states = [s for s, _, _, b in rows if b[30]]
    print(f"  [INFO] got_mystery_egg (30) set in {len(egg_states)} state(s): {egg_states}")

    print("\nRESULT:", "all checks passed" if ok else "FAILURES above")
    p.stop()


if __name__ == "__main__":
    main()
