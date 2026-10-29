# -*- coding: utf-8 -*-
"""What an LLM-harness run leaves on disk: enough to recover any moment of
a long run, and everything it had if it crashes. Emulator-backed, with a
stub model in place of the LLM server."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROM = "./emu_files/Pokemon - Crystal Version.gbc"


class StubLLM:
    """Walks back and forth; raises on turn ``crash_at`` if set."""

    def __init__(self, crash_at=None):
        self.calls = 0
        self.crash_at = crash_at

    def decide(self, system, user, images, temperature, **kw):
        self.calls += 1
        if self.crash_at and self.calls >= self.crash_at:
            raise RuntimeError("stub crash")
        button = "left" if self.calls % 2 else "right"
        return {
            "decision": {
                "observation": "",
                "progress": "",
                "plan": "pace",
                "following": "none",
                "buttons": [button],
            },
            "raw": "{}",
            "reasoning": None,
            "content": None,
            "usage": {"prompt_tokens": 100},
            "latency": 0.01,
            "error": None,
            "model": "stub",
            "retries": [],
            "finish_reason": "tool_calls",
        }


@unittest.skipUnless(os.path.exists(ROM), "ROM not available")
class TestRunLogging(unittest.TestCase):
    def _run(self, llm, budget):
        from tools.llm_agent.agent import Agent, Memory
        from tools.llm_agent.game import Game
        from tools.llm_agent.goals import Goal
        from tools.llm_agent.recorder import Recorder

        out = Path(tempfile.mkdtemp())
        rec = Recorder(out, {"test": True})
        game = Game("configs/stages/open_world.json")
        rec.run_info(game.config, ["test"])
        goal = Goal("pace", "Pace", "Walk about.", lambda g, v: False, budget)
        agent = Agent(game, llm, rec, video=False, ladder=[goal], max_attempts=1, autosave_every=0)
        memory = Memory()
        try:
            agent.run_goal(goal, memory)
        except RuntimeError as e:
            agent.emergency_save("crash", e)
        finally:
            agent.close()
            rec.close()
            game.close()
        return out

    def test_a_run_leaves_states_health_and_env_fields(self):
        out = self._run(StubLLM(), budget=30)
        turns = [json.loads(line) for line in (out / "turns.jsonl").open()]
        steps = [json.loads(line) for line in (out / "steps.jsonl").open()]
        self.assertEqual(len(turns), 30)
        # a state every 10 turns, named by its turn row
        self.assertEqual(
            sorted(p.name for p in (out / "states" / "turns").iterdir()),
            ["turn_00010.state", "turn_00020.state", "turn_00030.state"],
        )
        self.assertTrue(turns[9]["state_file"].endswith("turn_00010.state"))
        self.assertEqual(turns[0]["steps"], [1, 1])
        for k in ("env_step", "done", "truncated", "frames", "episode_reward", "mode", "time"):
            self.assertIn(k, steps[-1])
        health = [json.loads(line) for line in (out / "health.jsonl").open()]
        self.assertEqual(health[0]["turn"], 25)
        info = json.loads((out / "run_info.json").read_text())
        self.assertTrue(info["git"]["commit"])
        self.assertIn("rom_path", info["config"])
        demo = np.load(next((out / "demos").glob("*.npz")))
        self.assertEqual(len(demo["dones"]), len(demo["actions"]))
        self.assertFalse(bool(demo["partial"]))

    def test_a_goal_already_done_passes_in_zero_turns(self):
        # the return-trip steps count as done once a later one is: the
        # ladder must step over them cleanly
        from tools.llm_agent.agent import Agent, Memory
        from tools.llm_agent.game import Game
        from tools.llm_agent.goals import Goal
        from tools.llm_agent.recorder import Recorder

        out = Path(tempfile.mkdtemp())
        rec = Recorder(out, {"test": True})
        game = Game("configs/stages/open_world.json")
        goal = Goal("already", "Already", "Done.", lambda g, v: True, 100)
        agent = Agent(game, StubLLM(), rec, video=False, ladder=[goal], autosave_every=0)
        try:
            self.assertTrue(agent.run_goal(goal, Memory()))
        finally:
            agent.close()
            rec.close()
            game.close()
        self.assertEqual(rec.summary["attempts"][0]["turns"], 0)
        self.assertEqual(list((out / "demos").glob("*.npz")), [])

    def test_a_crash_keeps_the_state_the_demo_and_the_traceback(self):
        out = self._run(StubLLM(crash_at=8), budget=100)
        crash = json.loads((out / "crash.json").read_text())
        self.assertIn("stub crash", crash["traceback"])
        self.assertEqual(crash["goal"], "pace")
        self.assertTrue(Path(crash["state"]).exists())
        demo = np.load(next((out / "demos").glob("*_partial.npz")))
        self.assertEqual(len(demo["actions"]), 7)  # 7 turns of one press each
        self.assertTrue(bool(demo["partial"]))
        self.assertTrue((out / "autosave.json").exists())
        events = [json.loads(line) for line in (out / "events.jsonl").open()]
        self.assertTrue(any(e["kind"] == "run_crash" for e in events))


if __name__ == "__main__":
    unittest.main()
