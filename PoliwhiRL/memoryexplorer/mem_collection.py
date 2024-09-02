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

def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
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
        episode_id INTEGER
    )
    """)
    return conn, cursor

def get_next_episode_id(cursor):
    cursor.execute("SELECT MAX(episode_id) FROM memory_data")
    max_episode_id = cursor.fetchone()[0]
    return (max_episode_id or 0) + 1

def run_episode(env, conn, cursor, episode_id, is_manual, config):
    
    for _ in tqdm(range(config.get("episode_length", 1000)), desc=f"Episode {episode_id}"):
        if is_manual:
            action = get_sdl_action()
            while action == 0:
                action = get_sdl_action()
            if action == -1:
                print("Quitting...")
                return False
        else:
            action = np.random.randint(1, 7)

        env.handle_action(action)
        mem_view = env.get_game_area()
        ram_vars = env.get_RAM_variables()
        img = env.pyboy.screen.image
        ram_view = env.ram.export_wram()

        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()



        mem_view_binary = io.BytesIO()
        np.save(mem_view_binary, mem_view, allow_pickle=False)
        mem_view_binary = mem_view_binary.getvalue()

        ram_view_binary = io.BytesIO()
        np.save(ram_view_binary, ram_view, allow_pickle=False)
        ram_view_binary = ram_view_binary.getvalue()

        cursor.execute("""
        INSERT INTO memory_data (
            image, money, location, X, Y, party_info,
            pkdex_seen, pkdex_owned, map_num_loc, mem_view, ram_view,
            warp_number, map_bank, is_manual, action, episode_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            img_bytes, ram_vars["money"], ram_vars["location"],
            ram_vars["X"], ram_vars["Y"], str(ram_vars["party_info"]),
            str(ram_vars["pkdex_seen"]), str(ram_vars["pkdex_owned"]),
            str(ram_vars["map_num_loc"]), mem_view_binary, ram_view_binary,
            ram_vars["warp_number"], ram_vars["map_bank"], is_manual, action, episode_id
        ))
        conn.commit()

    return True

def memory_collector(config):
    # if os.path.exists("emu_files/states/exploration.state"):
    #     config["state_path"] = "emu_files/states/exploration.state"
    #     print("Found previous explore state... using it")

    conn, cursor = setup_database(config.get("explore_db_loc", "memory_data.db"))

    num_episodes = config.get("num_episodes", 1)
    manual_control = config.get("manual_control", True)

    env = Env(config, force_window=manual_control)
    env.reset()

    next_episode_id = get_next_episode_id(cursor)

    if manual_control:
        print("Press keys to control the game. Press 'q' to quit.")
        run_episode(env, conn, cursor, next_episode_id, True, config)
    else:
        for _ in range(num_episodes):
            if not run_episode(env, conn, cursor, next_episode_id, False, config):
                break
            next_episode_id += 1

    conn.close()
    env.save_state("emu_files/states/", "exploration.state")
