# -*- coding: utf-8 -*-
"""Everything a run writes, in one directory:

  turns.jsonl        one row per LLM call: prompt, decision, raw output,
                     reasoning, latency, tokens, what actually executed
  steps.jsonl        one row per env step: position, scene/battle state, the
                     reward and its per-source breakdown
  frames/            the raw 160x144 frame the model saw on each turn
  states/            PyBoy save-states at each goal attempt's start and success
  demos/             per-attempt training-format arrays (image, ram, action,
                     reward), tagged source=llm_demo and seeded/honest
  summary.json       per-goal attempt outcomes
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Recorder:
    def __init__(self, out_dir: Path, meta: dict):
        self.dir = Path(out_dir)
        for sub in ("frames", "states", "demos"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self._turns = open(self.dir / "turns.jsonl", "a")
        self._steps = open(self.dir / "steps.jsonl", "a")
        self.summary = {"meta": meta, "attempts": []}
        self._demo = None

    def turn(self, row: dict, frame):
        frame.save(self.dir / "frames" / f"turn_{row['turn']:05d}.png")
        self._turns.write(json.dumps(row, default=str) + "\n")
        self._turns.flush()

    def step(self, row: dict, sample: dict):
        self._steps.write(json.dumps(row, default=str) + "\n")
        self._steps.flush()
        if self._demo is not None:
            for k in ("image", "ram", "action", "reward"):
                self._demo[k].append(sample[k])

    def state(self, name: str, data: bytes) -> Path:
        path = self.dir / "states" / f"{name}.state"
        path.write_bytes(data)
        return path

    def begin_demo(self):
        self._demo = {"image": [], "ram": [], "action": [], "reward": []}

    def end_demo(self, name: str, success: bool, seeded: bool):
        demo, self._demo = self._demo, None
        if not demo or not demo["action"]:
            return None
        path = self.dir / "demos" / f"{name}.npz"
        np.savez_compressed(
            path,
            images=np.stack(demo["image"]).astype(np.uint8),
            ram=np.stack(demo["ram"]).astype(np.float32),
            actions=np.asarray(demo["action"], dtype=np.int64),
            rewards=np.asarray(demo["reward"], dtype=np.float32),
            success=success,
            seeded=seeded,
            source="llm_demo",
        )
        return path

    def attempt(self, row: dict):
        self.summary["attempts"].append(row)
        self.write_summary()

    def write_summary(self):
        (self.dir / "summary.json").write_text(json.dumps(self.summary, indent=2, default=str))

    def close(self):
        self.write_summary()
        self._turns.close()
        self._steps.close()
