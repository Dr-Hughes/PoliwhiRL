import pygame
from pyboy import PyBoy

pyboy = PyBoy("emu_files/Pokemon - Crystal Version.gbc")

pygame.init()

save_slot = 0

while pyboy.tick():
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_o:
                filename = f"save_{save_slot}.state"

                with open(filename, "wb") as f:
                    pyboy.save_state(f)

                print(f"Saved state to {filename}")
                save_slot += 1

pyboy.stop()