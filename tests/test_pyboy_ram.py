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
        num_pokemon, total_level, total_hp, total_exp = self.ram.get_party_info()
        self.assertIsInstance(total_level, int)
        self.assertIsInstance(total_hp, int)
        self.assertIsInstance(total_exp, int)
        self.assertGreaterEqual(total_level, 0)
        self.assertGreaterEqual(total_hp, 0)
        self.assertGreaterEqual(total_exp, 0)

    def test_get_pokedex_seen(self):
        seen = self.ram.get_pokedex_seen()
        self.assertIsInstance(seen, int)
        self.assertGreaterEqual(seen, 0)

    def test_get_pokedex_owned(self):
        owned = self.ram.get_pokedex_owned()
        self.assertIsInstance(owned, int)
        self.assertGreaterEqual(owned, 0)

    def test_get_map_num(self):
        map_num = self.ram.get_map_num()
        self.assertIsInstance(map_num, int)
        self.assertGreaterEqual(map_num, 0)

    def test_get_variables(self):
        variables = self.ram.get_variables()
        self.assertIsInstance(variables, dict)
        self.assertIn("money", variables)
        self.assertIn("room", variables)
        self.assertIn("X", variables)
        self.assertIn("Y", variables)
        self.assertIn("party_info", variables)
        self.assertIn("pokedex_seen", variables)
        self.assertIn("pokedex_owned", variables)
        self.assertIn("map_num", variables)
        self.assertIn("warp_number", variables)
        self.assertIn("map_bank", variables)

    def test_new_raw_features(self):
        variables = self.ram.get_variables()
        for key in ["battle_type", "johto_badges", "player_state",
                     "key_items_count", "game_hour", "bgm_id"]:
            self.assertIn(key, variables)
            self.assertIsInstance(variables[key], int)
            self.assertGreaterEqual(variables[key], 0)

    def test_battle_type_range(self):
        bt = self.ram.get_battle_type()
        self.assertIn(bt, [0, 1, 2], f"battle_type should be 0/1/2, got {bt}")

    def test_johto_badges_range(self):
        badges = self.ram.get_johto_badges()
        self.assertGreaterEqual(badges, 0)
        self.assertLessEqual(badges, 255)

    def test_player_state_range(self):
        ps = self.ram.get_player_state()
        self.assertGreaterEqual(ps, 0)
        self.assertLessEqual(ps, 255)


if __name__ == "__main__":
    unittest.main()
