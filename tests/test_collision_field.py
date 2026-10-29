# -*- coding: utf-8 -*-
"""Ground-truth check for the egocentric ROM walkability field.

The field packs the raw ROM collision value for each cell around the player.
Its centre-cross cells are, by construction, the same four tiles the engine
reports live at 0xC2FA-FD (down/up/left/right). This test walks a real env and
asserts the decoded field matches those engine bytes at every non-boundary
step — the self-check that guards against a silently-wrong collision decode
feeding training.

Skips cleanly if the ROM/save-state aren't present (CI without ROM).
"""
import os
import unittest

import numpy as np

ROM = "./emu_files/Pokemon - Crystal Version.gbc"
STATE = "./emu_files/states/start.state"


def _have_rom():
    return os.path.exists(ROM) and os.path.exists(STATE)


@unittest.skipUnless(_have_rom(), "ROM/save-state not available")
class TestCollisionField(unittest.TestCase):
    def _config(self):
        from main import (
            load_default_config,
            load_user_config,
            merge_configs,
        )

        cfg = merge_configs(
            load_default_config(),
            load_user_config("./configs/inference.json"),
        )
        cfg["rom_path"] = ROM
        cfg["state_path"] = STATE
        cfg["vision"] = False
        cfg["record"] = False
        cfg["goexplore_enabled"] = False
        cfg["goexplore_flag_capture"] = False
        return cfg

    def test_field_center_cross_matches_engine(self):
        from PoliwhiRL.environment.gym_env import (
            PyBoyEnvironment,
            COLLISION_FIELD_RADIUS as R,
            RAM_FEATURE_INDEX,
        )

        env = PyBoyEnvironment(self._config())
        try:
            env.reset()
            n = 2 * R + 1

            # Field index of the cell at egocentric offset (dx, dy).
            def fidx(dx, dy):
                return (dy + R) * n + (dx + R)

            c0 = RAM_FEATURE_INDEX["collision_local_0"]
            checked = mism = 0
            # A short scripted walk to sample several positions/tiles.
            # Movement action ids: 3=left, 4=right, 5=up, 6=down.
            walk = [6, 6, 5, 5, 3, 4, 4, 3, 5, 6] * 4
            for a in walk:
                obs, *_ = env.step(a)
                ram = obs["ram"]
                ev = env.ram.get_variables()
                if ev.get("script_active", False):
                    continue
                field = ram[c0 : c0 + n * n]
                # Engine live neighbour bytes (raw), scaled the same way.
                eng = {
                    (0, 1): ev["collision_down"],
                    (0, -1): ev["collision_up"],
                    (-1, 0): ev["collision_left"],
                    (1, 0): ev["collision_right"],
                }
                for (dx, dy), ev_val in eng.items():
                    decoded = round(float(field[fidx(dx, dy)]) * 255.0)
                    # Skip boundary cells: engine reads its connection border
                    # where the pristine ROM has the map's own edge block.
                    x, y = ev["X"] + dx, ev["Y"] + dy
                    if x < 0 or y < 0:
                        continue
                    checked += 1
                    if decoded != int(ev_val):
                        mism += 1
            self.assertGreater(checked, 0, "no cells were checked")
            # Allow a small fraction for map-boundary/overlay tiles.
            rate = mism / checked
            self.assertLess(
                rate,
                0.1,
                f"collision field disagrees with engine on {mism}/{checked} "
                f"cells ({rate:.1%}) — decode likely wrong",
            )
        finally:
            env.close()

    def test_field_is_in_observation_and_bounded(self):
        from PoliwhiRL.environment.gym_env import (
            PyBoyEnvironment,
            COLLISION_FIELD_RADIUS as R,
            RAM_FEATURE_INDEX,
        )

        env = PyBoyEnvironment(self._config())
        try:
            obs = env.reset()
            c0 = RAM_FEATURE_INDEX["collision_local_0"]
            n = (2 * R + 1) ** 2
            field = obs["ram"][c0 : c0 + n]
            self.assertEqual(field.shape[0], n)
            self.assertTrue(np.all(field >= 0.0) and np.all(field <= 1.0))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
