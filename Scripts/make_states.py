from pyboy import PyBoy
pyboy = PyBoy('emu_files/Pokemon - Crystal Version.gbc')
while pyboy.tick():
    pass

with open("save.state", "wb") as stateFile:
    pyboy.save_state(stateFile)
pyboy.stop()