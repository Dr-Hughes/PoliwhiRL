"""MP4s: one per goal attempt (every env step, captioned with the goal, the
model's plan and the step's reward), plus short clips around highlight events
(catches, whiteouts, battle wins, story checkpoints)."""
from __future__ import annotations

import textwrap
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCALE = 3
W, H = 160 * SCALE, 144 * SCALE
PANEL = 112
FPS = 6
CLIP_BEFORE, CLIP_AFTER = 90, 24
HIGHLIGHTS = {"catch", "whiteout", "battle_won", "story_checkpoint"}

try:
    _FONT = ImageFont.load_default(size=13)
except TypeError:  # Pillow < 10.1
    _FONT = ImageFont.load_default()


def compose(frame: Image.Image, lines: list[str]) -> np.ndarray:
    canvas = Image.new("RGB", (W, H + PANEL), (16, 16, 24))
    canvas.paste(frame.resize((W, H), Image.NEAREST), (0, 0))
    d = ImageDraw.Draw(canvas)
    y = H + 4
    for line in lines:
        for chunk in textwrap.wrap(line, 64)[:2]:
            if y > H + PANEL - 14:
                break
            d.text((6, y), chunk, fill=(230, 230, 230), font=_FONT)
            y += 15
    return np.asarray(canvas)


def _writer(path: Path):
    return imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=7,
                              pixelformat="yuv420p", macro_block_size=16)


class Media:
    def __init__(self, out_dir: Path):
        self.dir = Path(out_dir) / "videos"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._goal = None
        self._goal_tmp = None
        self._buffer = deque(maxlen=CLIP_BEFORE)
        self._clips = []  # [writer, frames_left, path]

    def begin_goal(self, name: str):
        self._goal_tmp = self.dir / f"{name}.partial.mp4"
        self._goal = _writer(self._goal_tmp)

    def frame(self, frame: Image.Image, lines: list[str]):
        arr = compose(frame, lines)
        if self._goal is not None:
            self._goal.append_data(arr)
        self._buffer.append(arr)
        for clip in self._clips:
            clip[0].append_data(arr)
            clip[1] -= 1
        for clip in [c for c in self._clips if c[1] <= 0]:
            clip[0].close()
            self._clips.remove(clip)

    def clip(self, name: str) -> Path:
        """Start a highlight clip: the buffered lead-in plus CLIP_AFTER more steps."""
        path = self.dir / f"{name}.mp4"
        w = _writer(path)
        for arr in self._buffer:
            w.append_data(arr)
        self._clips.append([w, CLIP_AFTER, path])
        return path

    def end_goal(self, name: str) -> Path | None:
        if self._goal is None:
            return None
        self._goal.close()
        self._goal = None
        path = self.dir / f"{name}.mp4"
        self._goal_tmp.rename(path)
        return path

    def close(self):
        for clip in self._clips:
            clip[0].close()
        self._clips = []
        if self._goal is not None:
            self._goal.close()
            self._goal = None
