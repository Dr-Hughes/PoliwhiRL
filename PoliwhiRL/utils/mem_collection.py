# -*- coding: utf-8 -*-
from PoliwhiRL.environment import PyBoyEnvironment as Env
import numpy as np
import sqlite3
import io
import os
from PIL import Image
from tqdm import tqdm
import hashlib


def memory_collector(config):
    """
    This function is used to collect unique memory data from the environment.
    """
    env = Env(config)
    img, _ = env.reset()

    # Connect to the SQLite database
    conn = sqlite3.connect("memory_data.db")
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
    mem_view TEXT,
    mem_view_bg TEXT,
    mem_view_wnd TEXT,
    warp_number INTEGER,
    map_bank INTEGER
    )
    """
    )

    # Create a set to store image hashes
    image_hashes = set()

    for _ in tqdm(range(config.get("episode_length"))):
        action = np.random.randint(1, env.action_space.n)

        env.handle_action(action)
        mem_view_bg = env.get_pyboy_bg()
        mem_view_wnd = env.get_pyboy_wnd()
        mem_view = env.get_game_area()
        ram_vars = env.get_RAM_variables()
        img = env.get_screen_image()

        # Convert the image to bytes
        img_bytes = io.BytesIO()
        Image.fromarray(img).save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        # Calculate image hash
        img_hash = hashlib.md5(img_bytes).hexdigest()

        # Only insert the image if it's unique
        if img_hash not in image_hashes:
            image_hashes.add(img_hash)

            # Convert the mem_view matrix to a string
            mem_view_bg = str(mem_view_bg)
            mem_view_wnd = str(mem_view_wnd)
            mem_view = str(mem_view.tolist())

            # Insert the image data and ram_vars into the database
            cursor.execute(
                """
            INSERT INTO memory_data (
            image, money, location, X, Y, party_info,
            pkdex_seen, pkdex_owned, map_num_loc, mem_view, mem_view_bg,
            mem_view_wnd,
            warp_number, map_bank
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    mem_view,
                    mem_view_bg,
                    mem_view_wnd,
                    ram_vars["warp_number"],
                    ram_vars["map_bank"],
                ),
            )
            conn.commit()

    # Close the database connection
    conn.close()
    extract_images_by_map()


def extract_images_by_map(db_path="memory_data.db", output_dir="extracted_images"):
    """
    Extract images from the database and save them into folders based on map_num_loc and uniqueness.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the main output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all unique map_num_loc values
    cursor.execute("SELECT DISTINCT map_num_loc FROM memory_data")
    map_locations = cursor.fetchall()

    for map_loc in tqdm(map_locations, desc="Extracting images"):
        map_loc = map_loc[0]  # Extract string from tuple

        # Create folders for this map_num_loc
        map_dir = os.path.join(output_dir, f"map_{map_loc}")
        os.makedirs(map_dir, exist_ok=True)
        # Get all images for this map_num_loc
        cursor.execute(
            "SELECT id, image FROM memory_data WHERE map_num_loc = ?", (map_loc,)
        )
        images = cursor.fetchall()

        for img_id, img_data in images:
            # Convert BLOB to image
            image = Image.open(io.BytesIO(img_data))

            image_path = os.path.join(map_dir, f"image_{img_id}.png")

            image.save(image_path)

    # Close the database connection
    conn.close()
