# -*- coding: utf-8 -*-
import sqlite3
import os
import argparse
from PIL import Image
import io
from tqdm import tqdm
import hashlib


def get_image_hash(image):
    """Calculate a hash for the image data."""
    return hashlib.md5(image.tobytes()).hexdigest()


def save_image_with_check(image, file_path):
    """Save the image, handling duplicates by adding a repetition number if necessary."""
    base_path, ext = os.path.splitext(file_path)
    counter = 0
    current_path = file_path
    while os.path.exists(current_path):
        existing_image = Image.open(current_path)
        if get_image_hash(existing_image) == get_image_hash(image):
            # Image is identical, no need to save
            return
        counter += 1
        current_path = f"{base_path}_rep{counter}{ext}"
    image.save(current_path)


def extract_images(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get total number of rows for progress bar
    cursor.execute("SELECT COUNT(*) FROM memory_data")
    total_rows = cursor.fetchone()[0]

    # Fetch data in chunks
    chunk_size = 1000
    offset = 0
    pbar = tqdm(total=total_rows, desc="Extracting images")

    while True:
        cursor.execute(
            """
            SELECT episode_id, map_bank, map_num, X, Y, location, image
            FROM memory_data
            LIMIT ? OFFSET ?
        """,
            (chunk_size, offset),
        )

        rows = cursor.fetchall()
        if not rows:
            break

        for row in rows:
            episode_id, map_bank, map_num, X, Y, location, image_data = row
            # Create folder structure
            folder_path = os.path.join(
                "extracted_images",
                f"episode_{episode_id}",
                f"map_bank_{map_bank}",
                f"map_num_{map_num}",
            )
            os.makedirs(folder_path, exist_ok=True)

            # Create filename
            filename = f"{X}_{Y}_{location}.png"
            file_path = os.path.join(folder_path, filename)

            # Open image
            image = Image.open(io.BytesIO(image_data))

            # Save image with duplicate checking
            save_image_with_check(image, file_path)

            pbar.update(1)

        offset += chunk_size

    pbar.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Extract images from the database")
    parser.add_argument("db_path", help="Path to the SQLite database")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"Error: Database file not found at {args.db_path}")
        return

    extract_images(args.db_path)
    print("Image extraction complete!")


if __name__ == "__main__":
    main()
