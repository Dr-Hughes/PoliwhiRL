# -*- coding: utf-8 -*-
"""Decide empirically which event-flag bit = "got a starter from Elm".
Replays a starter-getting trajectory and prints, at the END (persistent
state), every bit in the 0xDA72 region bytes 2-7 plus the candidate flag
numbers, so we can see which bit is set-and-stays-set."""
from pyboy import PyBoy

ROM = "./emu_files/Pokemon - Crystal Version.gbc"
STATE = "./emu_files/states/start.state"
ACTIONS = "./Training Outputs/first_steps/Checkpoints/best/actions.steps"
A = ["", "a", "b", "left", "right", "up", "down", "start", "select"]
FPA, HOLD = 90, 15
FLAGS0 = 0xDA72
OWNED = (0xDE99, 0xDEB8)
PARTY_SPECIES = 0xDCDF  # first party mon species


def trajs(path):
    blocks, cur = [], None
    for line in open(path):
        s = line.strip()
        if s.startswith("# trajectory"):
            if cur is not None: blocks.append(cur)
            cur = []
        elif cur is not None and s and not s.startswith("#"):
            try: cur.append(int(s))
            except ValueError: pass
    if cur: blocks.append(cur)
    return blocks


def owned(p):
    return sum(bin(p.memory[a]).count("1") for a in range(OWNED[0], OWNED[1] + 1))


def fbit(p, n):
    return (p.memory[FLAGS0 + n // 8] >> (n % 8)) & 1


def step(p, a):
    b = A[a]; fr = FPA
    if b: p.button(b, delay=HOLD); fr -= HOLD
    p.tick(fr, False)


def main():
    p = PyBoy(ROM, window="null", sound_emulated=False)
    p.set_emulation_speed(0)
    blocks = trajs(ACTIONS)
    for ti in (1, 3, 5):
        with open(STATE, "rb") as f: p.load_state(f)
        # track which candidate bits were EVER set vs final
        ever = {n: 0 for n in range(20, 36)}
        for a in blocks[ti]:
            step(p, a)
            for n in ever:
                if fbit(p, n): ever[n] = 1
        final = {n: fbit(p, n) for n in range(20, 36)}
        species = p.memory[PARTY_SPECIES]
        print(f"\n=== traj {ti}: owned={owned(p)} starter_species={species} ===")
        print("flag : ever / final  (bytes 2-4 region, flags 20..35)")
        for n in range(20, 36):
            mark = ""
            if ever[n] and final[n]: mark = "  <- persistent"
            elif ever[n] and not final[n]: mark = "  <- TRANSIENT (set then cleared)"
            print(f"  {n:3} : {ever[n]} / {final[n]}{mark}")
    p.stop()


if __name__ == "__main__":
    main()
