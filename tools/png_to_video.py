#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stitch a folder of PNGs into a GIF and an MP4."""

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Aborted: Pillow is required. Install with: pip install Pillow")

try:
    import imageio_ffmpeg  # noqa: F401
except ImportError:
    sys.exit(
        "Aborted: imageio[ffmpeg] is required. Install with: pip install 'imageio[ffmpeg]'"
    )

import imageio


def sorted_pngs(folder: Path) -> list[str]:
    """Return PNG paths sorted numerically by the last integer in each filename."""

    def sort_key(name: str) -> int:
        parts = Path(name).stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
        return 0

    return sorted([str(p) for p in folder.glob("*.png")], key=sort_key)


BUTTON_LABELS = {
    "a": "A",
    "b": "B",
    "start": "START",
    "select": "SELECT",
    "up": "^",
    "down": "v",
    "left": "<-",
    "right": "->",
}


def extract_button(filepath: str) -> str | None:
    """Extract the button name from a filename like step_1_x_6_y_1_map_7_bank_24_room_7_btn_a_reward_-0.5.png"""
    stem = Path(filepath).stem
    m = re.search(r"btn_(a|b|start|select|up|down|left|right)", stem)
    if m:
        raw = m.group(1)
        return BUTTON_LABELS.get(raw, raw.upper())
    return None


def overlay_button(text: str, img: Image.Image, padding: int = 4) -> Image.Image:
    """Draw text in top-right corner with black background."""
    draw = ImageDraw.Draw(img)

    # Pick a font that fits within the frame width
    max_box_w = img.width - 2 * padding
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    for size in range(48, 6, -1):
        try:
            font = ImageFont.truetype(font_path, size)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0] + 2 * padding
            if tw <= max_box_w:
                break
        except (OSError, IOError):
            font = ImageFont.load_default()
            break

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0] + 2 * padding
    th = bbox[3] - bbox[1] + 2 * padding

    x = img.width - tw
    y = 0
    draw.rectangle([x, y, x + tw, y + th], fill=(0, 0, 0))
    draw.text((x + padding, y + padding), text, fill=(255, 255, 255), font=font)
    return img


def add_overlays(pngs: list[str]) -> tuple[list[str], tempfile.TemporaryDirectory]:
    """Overlay button labels on each frame, return new paths + temp dir handle."""
    tmp = tempfile.TemporaryDirectory(prefix="png_overlay_")
    result: list[str] = []
    for i, path in enumerate(pngs):
        btn = extract_button(path) or "?"
        img = Image.open(path)
        if "RGBA" in str(img.mode):
            # Composite onto RGB for clean overlay (avoids alpha issues)
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = overlay_button(btn, bg)
        else:
            img = overlay_button(btn, img.convert("RGB"))
        out = Path(tmp.name) / f"{i:06d}.png"
        img.save(out, "PNG")
        result.append(str(out))
    return result, tmp


def make_gif(pngs: list[str], output: Path, fps: int):
    duration_ms = 1000 // fps
    frames = [Image.open(p) for p in pngs]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"  GIF → {output} ({len(frames)} frames)")


def make_mp4(pngs: list[str], output: Path, fps: int):
    writer = imageio.get_writer(output, fps=fps, codec="libx264")
    for p in pngs:
        writer.append_data(imageio.v3.imread(p))
    writer.close()
    print(f"  MP4 → {output} ({len(pngs)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Stitch PNGs into a GIF and MP4.")
    parser.add_argument("folder", help="Directory containing the PNGs")
    parser.add_argument(
        "-o",
        "--output_prefix",
        default=None,
        help="Output prefix (default: first PNG's stem)",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second (default: 10)"
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Not a directory: {folder}")

    pngs = sorted_pngs(folder)
    if not pngs:
        sys.exit(f"No PNGs found in {folder}")

    if args.output_prefix:
        prefix = args.output_prefix
        out_dir = Path.cwd()
    else:
        prefix = folder.name.replace(" ", "_")
        out_dir = folder

    gif_path = out_dir / f"{prefix}.gif"
    mp4_path = out_dir / f"{prefix}.mp4"

    print(f"Processing {len(pngs)} frames at {args.fps} fps…")
    overlayed, tmp_dir = add_overlays(pngs)
    try:
        make_gif(overlayed, gif_path, args.fps)
        make_mp4(overlayed, mp4_path, args.fps)
    finally:
        tmp_dir.cleanup()
    print("Done.")


if __name__ == "__main__":
    main()
