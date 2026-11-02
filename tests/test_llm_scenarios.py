# -*- coding: utf-8 -*-
"""Emulator regression scenarios (tools/llm_agent/scenarios.py): a recorded
button trajectory replayed through the real env with the reward verifier,
observation invariants and button probes. Skips without the ROM."""

import os
import unittest

ROM = "./emu_files/Pokemon - Crystal Version.gbc"
STATE = "./emu_files/states/start.state"


@unittest.skipUnless(os.path.exists(ROM) and os.path.exists(STATE), "ROM/start state not present")
class TestScenarios(unittest.TestCase):
    def test_all_trajectories_pass(self):
        from tools.llm_agent.scenarios import run_all

        for name, failures in run_all().items():
            with self.subTest(trajectory=name):
                self.assertEqual(failures, [], "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
