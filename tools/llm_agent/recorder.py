# -*- coding: utf-8 -*-
"""Everything a run writes, in one directory:

  turns.jsonl        one row per LLM call: prompt, decision, raw output,
                     reasoning, latency, tokens, what actually executed
  steps.jsonl        one row per env step: position, scene/battle state, the
                     reward and its per-source breakdown
  frames/            the raw 160x144 frame the model saw on each turn
  states/            PyBoy save-states at each goal attempt's start and success,
                     plus autosaves (with resume info)
  states/turns/      a save-state every few turns (turn rows name theirs), so
                     any moment is a ``drive.py`` load plus a few replayed
                     buttons from steps.jsonl away
  states/events/     a save-state at each notable game event (first map
                     entry, story checkpoint, whiteout, catch, ...) and where
                     training would have ended the episode
  demos/             per-attempt training-format arrays (image, ram, action,
                     reward, plus the env's done/truncated flags), tagged
                     source=llm_demo and seeded/honest; an attempt cut short
                     by a crash or Ctrl-C is saved as ``*_partial.npz``
  events.jsonl       verified game events and training-episode ends
  health.jsonl       every few turns: speed, latency, errors, memory, disk
  run_info.json      git commit and branch, uncommitted diff (git.diff), the
                     resolved env config, versions, host, command line
  agent.log          the agent's console lines, timestamped
  crash.json         the traceback and where the run was, if it crashed
  summary.json       per-goal attempt outcomes
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


class Recorder:
    def __init__(self, out_dir: Path, meta: dict):
        self.dir = Path(out_dir)
        for sub in ("frames", "states", "states/turns", "states/events", "demos"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self._turns = open(self.dir / "turns.jsonl", "a")
        self._steps = open(self.dir / "steps.jsonl", "a")
        self._health = open(self.dir / "health.jsonl", "a")
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
            env = sample.get("env") or {}
            self._demo["done"].append(bool(env.get("done")))
            self._demo["truncated"].append(bool(env.get("truncated")))

    def state(self, name: str, data: bytes) -> Path:
        path = self.dir / "states" / f"{name}.state"
        path.write_bytes(data)
        return path

    def begin_demo(self):
        self._demo = {
            "image": [],
            "ram": [],
            "action": [],
            "reward": [],
            "done": [],
            "truncated": [],
        }

    def end_demo(self, name: str, success: bool, seeded: bool, partial: bool = False):
        """Write the attempt's training-format arrays. ``partial``: the
        attempt was cut short (crash, Ctrl-C); saved as ``<name>_partial``
        so a demo loader can tell it from a finished attempt."""
        demo, self._demo = self._demo, None
        if not demo or not demo["action"]:
            return None
        path = self.dir / "demos" / f"{name}{'_partial' if partial else ''}.npz"
        np.savez_compressed(
            path,
            images=np.stack(demo["image"]).astype(np.uint8),
            ram=np.stack(demo["ram"]).astype(np.float32),
            actions=np.asarray(demo["action"], dtype=np.int64),
            rewards=np.asarray(demo["reward"], dtype=np.float32),
            dones=np.asarray(demo["done"], dtype=bool),
            truncateds=np.asarray(demo["truncated"], dtype=bool),
            success=success,
            seeded=seeded,
            partial=partial,
            source="llm_demo",
        )
        return path

    def health(self, row: dict):
        self._health.write(json.dumps(row, default=str) + "\n")
        self._health.flush()

    def run_info(self, config: dict, argv: list[str]):
        """Where this run came from: enough to rebuild the exact code and
        env config it ran with."""
        repo = Path(__file__).resolve().parents[2]

        def git(*args):
            try:
                return subprocess.run(
                    ["git", *args], cwd=repo, capture_output=True, text=True, timeout=30
                ).stdout
            except (OSError, subprocess.SubprocessError):
                return ""

        diff = git("diff", "HEAD")
        # New files are not in `git diff HEAD`; add the untracked Python.
        for f in git("ls-files", "--others", "--exclude-standard").splitlines():
            if f.endswith(".py"):
                diff += git("diff", "--no-index", "--", "/dev/null", f)
        (self.dir / "git.diff").write_text(diff)
        versions = {"python": sys.version.split()[0], "platform": platform.platform()}
        for mod in ("pyboy", "numpy", "torch"):
            try:
                versions[mod] = __import__(mod).__version__
            except Exception:
                pass
        info = {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": socket.gethostname(),
            "argv": argv,
            "git": {
                "commit": git("rev-parse", "HEAD").strip(),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
                "status": git("status", "--short").splitlines(),
                "diff_file": "git.diff" if diff else None,
            },
            "versions": versions,
            "config": config,
        }
        (self.dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))

    def attempt(self, row: dict):
        self.summary["attempts"].append(row)
        self.write_summary()

    def write_summary(self):
        (self.dir / "summary.json").write_text(json.dumps(self.summary, indent=2, default=str))

    def close(self):
        self.write_summary()
        self._turns.close()
        self._steps.close()
        self._health.close()
