# -*- coding: utf-8 -*-
import unittest
from PoliwhiRL.environment.gym_env import PyBoyEnvironment
from main import load_default_config


class TestRAMManagement(unittest.TestCase):
    def setUp(self):
        self.config = load_default_config()
        self.env = PyBoyEnvironment(self.config)
        self.ram = self.env.ram

    def tearDown(self):
        self.env.close()

    def test_get_current_location(self):
        location = self.ram.get_current_location()
        self.assertIsInstance(location, int)
        self.assertGreaterEqual(location, 0)

    def test_get_XY(self):
        x, y = self.ram.get_XY()
        self.assertIsInstance(x, int)
        self.assertIsInstance(y, int)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)

    def test_get_player_money(self):
        money = self.ram.get_player_money()
        self.assertIsInstance(money, int)
        self.assertGreaterEqual(money, 0)

    def test_get_party_info(self):
        total_level, total_hp, total_exp = self.ram.get_party_info()
        self.assertIsInstance(total_level, int)
        self.assertIsInstance(total_hp, int)
        self.assertIsInstance(total_exp, int)
        self.assertGreaterEqual(total_level, 0)
        self.assertGreaterEqual(total_hp, 0)
        self.assertGreaterEqual(total_exp, 0)

    def test_get_pkdex_seen(self):
        seen = self.ram.get_pkdex_seen()
        self.assertIsInstance(seen, int)
        self.assertGreaterEqual(seen, 0)

    def test_get_pkdex_owned(self):
        owned = self.ram.get_pkdex_owned()
        self.assertIsInstance(owned, int)
        self.assertGreaterEqual(owned, 0)

    def test_get_map_num_loc(self):
        map_num = self.ram.get_map_num_loc()
        self.assertIsInstance(map_num, int)
        self.assertGreaterEqual(map_num, 0)

    def test_update_variables(self):
        self.ram.update_variables()
        self.assertIsInstance(self.ram.money, int)
        self.assertIsInstance(self.ram.location, int)
        self.assertIsInstance(self.ram.X, int)
        self.assertIsInstance(self.ram.Y, int)
        self.assertIsInstance(self.ram.party_info, tuple)
        self.assertIsInstance(self.ram.pkdex_seen, int)
        self.assertIsInstance(self.ram.pkdex_owned, int)
        self.assertIsInstance(self.ram.map_num_loc, int)
        self.assertIsInstance(self.ram.warp_number, int)
        self.assertIsInstance(self.ram.map_bank, int)

    def test_get_variables(self):
        variables = self.ram.get_variables()
        self.assertIsInstance(variables, dict)
        self.assertIn("money", variables)
        self.assertIn("location", variables)
        self.assertIn("X", variables)
        self.assertIn("Y", variables)
        self.assertIn("party_info", variables)
        self.assertIn("pkdex_seen", variables)
        self.assertIn("pkdex_owned", variables)
        self.assertIn("map_num_loc", variables)
        self.assertIn("warp_number", variables)
        self.assertIn("map_bank", variables)


if __name__ == "__main__":
    unittest.main()
