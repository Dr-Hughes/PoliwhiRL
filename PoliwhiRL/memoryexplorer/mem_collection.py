# -*- coding: utf-8 -*-
from PoliwhiRL.environment import PyBoyEnvironment as Env
import numpy as np
import sqlite3
import io
import os
from tqdm import tqdm
import hashlib
import sdl2
import sdl2.ext


def get_sdl_action():
    action_map = ["", "a", "b", "left", "right", "up", "down", "start", "select"]
    events = sdl2.ext.get_events()
    for event in events:
        if event.type == sdl2.SDL_KEYDOWN:
            if event.key.keysym.sym == sdl2.SDLK_a:
                return action_map.index("a")
            elif event.key.keysym.sym == sdl2.SDLK_b:
                return action_map.index("b")
            elif event.key.keysym.sym == sdl2.SDLK_LEFT:
                return action_map.index("left")
            elif event.key.keysym.sym == sdl2.SDLK_RIGHT:
                return action_map.index("right")
            elif event.key.keysym.sym == sdl2.SDLK_UP:
                return action_map.index("up")
            elif event.key.keysym.sym == sdl2.SDLK_DOWN:
                return action_map.index("down")
            elif event.key.keysym.sym == sdl2.SDLK_RETURN:
                return action_map.index("start")
            elif event.key.keysym.sym == sdl2.SDLK_BACKSPACE:
                return action_map.index("select")
            elif event.key.keysym.sym == sdl2.SDLK_q:
                return -1  # Quit signal
    return 0  # No relevant key pressed


def memory_collector(config):
    """
    This function is used to collect memory data from the environment.
    """

    # Check if the state file exists
    if os.path.exists("emu_files/states/exploration.state"):
        state_path = "emu_files/states/exploration.state"
        config["state_path"] = state_path
        print("found previous explore state... using it")

    manual_control = config.get("manual_control", True)
    if manual_control:
        env = Env(config, force_window=True)
    else:
        env = Env(config)
    img = env.reset()

    # Connect to the SQLite database
    conn = sqlite3.connect(config.get("explore_db_loc", "memory_data.db"))
    cursor = conn.cursor()

    # Create a table to store the image data and associated information
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_data (
            id INTEGER PRIMARY KEY,
            image BLOB,
            money INTEGER,
            location TEXT,
            X INTEGER,
            Y INTEGER,
            party_info TEXT,
            pkdex_seen TEXT,
            pkdex_owned TEXT,
            map_num_loc TEXT,
            mem_view BLOB,
            ram_view BLOB,
            warp_number INTEGER,
            map_bank INTEGER,
            is_manual BOOLEAN,
            action INTEGER,
            manual_run_id INTEGER
        )
        """
    )

    # Create a set to store image hashes (only used for automatic mode)
    image_hashes = set()

    # Get a new manual_run_id if in manual mode
    if manual_control:
        cursor.execute("SELECT MAX(manual_run_id) FROM memory_data")
        max_run_id = cursor.fetchone()[0]
        manual_run_id = (max_run_id or 0) + 1
        print("Press keys to control the game. Press 'q' to quit.")
    else:
        manual_run_id = None

    for _ in tqdm(range(config.get("episode_length", 1000))):

        if manual_control:
            action = get_sdl_action()
            while action == 0:
                action = get_sdl_action()
            if action == -1:  # Empty string in the action map, use this as quit signal
                print("Quitting...")
                break
        else:
            action = np.random.randint(1, 7)

        env.handle_action(action)
        mem_view = env.get_game_area()
        ram_vars = env.get_RAM_variables()
        img = env.pyboy.screen.image
        ram_view = env.ram.export_wram()

        # Convert the image to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        # Calculate image hash
        img_hash = hashlib.md5(img_bytes).hexdigest()

        # Insert the image if it's unique (automatic mode) or always (manual mode)
        if manual_control or img_hash not in image_hashes:
            if not manual_control:
                image_hashes.add(img_hash)

            # Serialize mem_view and ram_view to binary
            mem_view_binary = io.BytesIO()
            np.save(mem_view_binary, mem_view, allow_pickle=False)
            mem_view_binary = mem_view_binary.getvalue()

            ram_view_binary = io.BytesIO()
            np.save(ram_view_binary, ram_view, allow_pickle=False)
            ram_view_binary = ram_view_binary.getvalue()

            # Insert the image data and ram_vars into the database
            cursor.execute(
                """
                INSERT INTO memory_data (
                    image, money, location, X, Y, party_info,
                    pkdex_seen, pkdex_owned, map_num_loc, mem_view, ram_view,
                    warp_number, map_bank, is_manual, action, manual_run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    img_bytes,
                    ram_vars["money"],
                    ram_vars["location"],
                    ram_vars["X"],
                    ram_vars["Y"],
                    str(ram_vars["party_info"]),
                    str(ram_vars["pkdex_seen"]),
                    str(ram_vars["pkdex_owned"]),
                    str(ram_vars["map_num_loc"]),
                    mem_view_binary,
                    ram_view_binary,
                    ram_vars["warp_number"],
                    ram_vars["map_bank"],
                    manual_control,
                    action,
                    manual_run_id,
                ),
            )
            conn.commit()

    # Close the database connection
    conn.close()
    env.save_state("emu_files/states/", "exploration.state")
