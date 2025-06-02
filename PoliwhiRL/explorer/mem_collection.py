# -*- coding: utf-8 -*-
from PoliwhiRL.environment import PyBoyEnvironment as Env
import numpy as np
import sqlite3
import io
import os
from tqdm import tqdm
import sdl2
import sdl2.ext
import pyboy.plugins.window_sdl2 as window_sdl2


# Monkey patch the event pump to avoid the double reading of events
def dummy_event_pump(events):
    return events


def apply_monkey_patch():
    window_sdl2.sdl2_event_pump = dummy_event_pump


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
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS memory_data (
        id INTEGER PRIMARY KEY,
        image BLOB,
        money INTEGER,
        location INTEGER,
        X INTEGER,
        Y INTEGER,
        party_total_level INTEGER,
        party_total_hp INTEGER,
        party_total_exp INTEGER,
        pokedex_seen INTEGER,
        pokedex_owned INTEGER,
        map_num INTEGER,
        screen_tiles BLOB,
        wram BLOB,
        warp_number INTEGER,
        map_bank INTEGER,
        is_manual BOOLEAN,
        action INTEGER,
        episode_id INTEGER
        )
    """
    )
    return conn, cursor


def get_next_episode_id(cursor):
    cursor.execute("SELECT MAX(episode_id) FROM memory_data")
    max_episode_id = cursor.fetchone()[0]
    return (max_episode_id or 0) + 1


def insert_buffer_to_db(cursor, buffer):
    cursor.executemany(
        """
    INSERT INTO memory_data (
        image, money, location, X, Y, party_total_level, party_total_hp, party_total_exp,
        pokedex_seen, pokedex_owned, map_num, screen_tiles, wram,
        warp_number, map_bank, is_manual, action, episode_id
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        buffer,
    )


def run_episode(env, conn, cursor, episode_id, is_manual, config):
    buffer = []
    actions = np.random.choice(
        [1, 2, 3, 4, 5, 6],
        size=config["episode_length"],
        p=[0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
    )
    for step in tqdm(range(config["episode_length"]), desc=f"Episode {episode_id}"):
        if is_manual:
            action = get_sdl_action()
            while action == 0:
                action = get_sdl_action()
            if action == -1:
                print("Quitting...")
                break
        else:
            action = actions[step]

        env._handle_action(action)

        screen_tiles = env.ram.get_screen_tiles()
        ram_vars = env.ram.get_variables()
        if is_manual:
            print("#" * 20)
            for key, value in ram_vars.items():
                print(f"{key}: {value}")
            print("#" * 20)
        img = env.pyboy.screen.image
        wram = env.ram.export_wram()

        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        screen_tiles_binary = io.BytesIO()
        np.save(screen_tiles_binary, screen_tiles, allow_pickle=False)
        screen_tiles_binary = screen_tiles_binary.getvalue()

        wram_binary = io.BytesIO()
        np.save(wram_binary, wram, allow_pickle=False)
        wram_binary = wram_binary.getvalue()

        buffer.append(
            (
                img_bytes,
                ram_vars["money"],
                ram_vars["room"],
                ram_vars["X"],
                ram_vars["Y"],
                ram_vars["party_info"][0],
                ram_vars["party_info"][1],
                ram_vars["party_info"][2],
                ram_vars["pokedex_seen"],
                ram_vars["pokedex_owned"],
                ram_vars["map_num"],
                screen_tiles_binary,
                wram_binary,
                ram_vars["warp_number"],
                ram_vars["map_bank"],
                is_manual,
                action,
                episode_id,
            )
        )

        if len(buffer) >= 1000:
            insert_buffer_to_db(cursor, buffer)
            conn.commit()
            buffer.clear()

    # Insert any remaining entries in the buffer
    if buffer:
        insert_buffer_to_db(cursor, buffer)
        conn.commit()

    return True


def memory_collector(config):
    conn, cursor = setup_database(config["explore_db_loc"])

    num_episodes = config["num_episodes"]
    manual_control = config["manual_control"]

    env = Env(config, force_window=manual_control)
    try:
        env.reset()

        next_episode_id = get_next_episode_id(cursor)

        if manual_control:
            print("Press keys to control the game. Press 'q' to quit.")
            apply_monkey_patch()
            run_episode(env, conn, cursor, next_episode_id, True, config)
        else:
            for _ in range(num_episodes):
                if not run_episode(env, conn, cursor, next_episode_id, False, config):
                    break
                next_episode_id += 1
    finally:
        # Ensure environment is properly closed
        env.close()
        conn.close()
