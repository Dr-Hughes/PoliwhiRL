"""Party-struct decoding, NPC-aware run-frontier field and the two
potential-based shaping terms (2026-10-28)."""
import types
import unittest

from PoliwhiRL.environment.RAM import RAMManagement
from PoliwhiRL.environment.rewards import Rewards
from PoliwhiRL.environment.rom_collision import RomCollisionField
from tests.test_step_penalty import _base_config, _env_vars

PARTY = 0xDCDF


def _ram_with_party(mons, move_table):
    """mons: list of dicts with hp, max_hp, exp, moves [(id, cur_pp, ups)]."""
    mem = {a: 0 for a in range(0xC000, 0xE000)}
    mem[0xDCD7] = len(mons)
    for i, mon in enumerate(mons):
        b = PARTY + 0x30 * i
        mem[b + 0x1F] = mon.get("level", 5)
        mem[b + 0x22], mem[b + 0x23] = mon["hp"] >> 8, mon["hp"] & 0xFF
        mem[b + 0x24], mem[b + 0x25] = mon["max_hp"] >> 8, mon["max_hp"] & 0xFF
        exp = mon.get("exp", 0)
        mem[b + 0x08], mem[b + 0x09], mem[b + 0x0A] = exp >> 16, (exp >> 8) & 0xFF, exp & 0xFF
        for j, (move, cur, ups) in enumerate(mon["moves"]):
            mem[b + 0x02 + j] = move
            mem[b + 0x17 + j] = (ups << 6) | cur
    ram = RAMManagement(types.SimpleNamespace(memory=mem))
    ram.move_table = move_table
    return ram


MOVES = {33: (35, 35), 45: (0, 40), 52: (40, 25)}  # tackle, growl, ember


class TestPartyStruct(unittest.TestCase):
    def test_hp_and_exp_are_big_endian(self):
        ram = _ram_with_party(
            [{"hp": 300, "max_hp": 310, "exp": 70000, "moves": [(33, 35, 0)]}], MOVES
        )
        self.assertEqual(ram.get_party_info(), (1, 5, 300, 70000))

    def test_resources_track_depletion_and_pp_ups(self):
        ram = _ram_with_party(
            [
                # Tackle 0/35, Growl 40/40, Ember 10/(25 + 5*1 PP-up = 30)
                {"hp": 17, "max_hp": 34, "moves": [(33, 0, 0), (45, 40, 0), (52, 10, 1)]},
                {"hp": 0, "max_hp": 20, "moves": [(33, 35, 0)]},
            ],
            MOVES,
        )
        hp, pp, lead_hp, lead_atk, able = ram.get_party_resources()
        self.assertAlmostEqual(hp, 17 / 54)
        self.assertAlmostEqual(pp, (0 + 40 + 10 + 35) / (35 + 40 + 30 + 35))
        self.assertAlmostEqual(lead_hp, 0.5)
        self.assertAlmostEqual(lead_atk, 10 / 65)
        self.assertAlmostEqual(able, 1 / 6)  # fainted second mon is not able

    def test_resources_zero_without_move_table(self):
        ram = _ram_with_party([{"hp": 1, "max_hp": 1, "moves": []}], None)
        self.assertEqual(ram.get_party_resources(), (0.0,) * 5)


class _GridField(RomCollisionField):
    """RomCollisionField over a hand-drawn single map ('.' floor, '#' wall)."""

    def __init__(self, rows):
        self._rows = rows
        self._gm = types.SimpleNamespace(width=len(rows[0]) // 2, height=len(rows) // 2)

    def _meta(self, group, number):
        return self._gm, 0, {}

    def _collision_offset(self, memory):
        return 0

    def _value(self, gm, off, x, y):
        if not (0 <= y < len(self._rows) and 0 <= x < len(self._rows[0])):
            return None
        return 0x00 if self._rows[y][x] == "." else 0x07


class TestDistanceField(unittest.TestCase):
    ROWS = [
        "....",
        "#.##",
        "....",
        "....",
    ]

    def _field(self, blocked=()):
        f = _GridField(self.ROWS)
        ev = {"map_bank": 1, "map_num": 1, "X": 0, "Y": 3}
        target = lambda b, n, x, y: (x, y) == (0, 0)  # noqa: E731
        return f, f.distance_field(None, ev, target, blocked=blocked)

    def test_distance_goes_around_walls(self):
        f, field = self._field()
        self.assertEqual(field[0][((1, 1), 0, 3)], 5)  # via the one-wide gap at x=1
        start = ((1, 1), 1, 3)
        self.assertEqual(field[0][start], 4)
        self.assertEqual(f.descent_direction(field, start, lookahead=2), [0.0, -1.0])

    def test_npc_on_the_gap_makes_frontier_unreachable(self):
        _f, field = self._field(blocked={((1, 1), 1, 1)})
        self.assertNotIn(((1, 1), 0, 3), field[0])


class TestPotentialShaping(unittest.TestCase):
    def _rw(self, **cfg):
        c = _base_config()
        c.update(**cfg)
        return Rewards(c)

    def test_frontier_potential_telescopes_and_holds_when_unavailable(self):
        rw = self._rw(run_frontier_potential_coef=10.0, run_frontier_potential_horizon=100)
        paid = [
            rw._frontier_potential_shaping({"run_frontier_distance": d})
            for d in (100, 50, None, 50, 100, 0)
        ]
        self.assertEqual(paid[0], 0.0)  # first observation only sets a baseline
        self.assertAlmostEqual(paid[1], 5.0)
        self.assertAlmostEqual(paid[2], 0.0)  # held, not dropped to zero
        self.assertAlmostEqual(sum(paid), 10.0)  # = phi(d=0) - phi(d=100)

    def test_heal_loop_cannot_be_farmed(self):
        rw = self._rw(resource_potential_coef=20.0)
        seq = [(1.0, 1.0), (0.2, 0.3), (1.0, 1.0), (0.2, 0.3), (1.0, 1.0)]
        paid = [
            rw._resource_potential_shaping({"party_resources": (h, p, 0, 0, 0)})
            for h, p in seq
        ]
        self.assertAlmostEqual(paid[2], 20.0 * 0.5 * (0.8 + 0.7))  # a heal pays
        self.assertAlmostEqual(sum(paid), 0.0)  # deplete/heal cycles net zero

    def test_terms_reach_breakdown_and_total(self):
        rw = self._rw(resource_potential_coef=20.0)
        ev = _env_vars()
        ev["party_resources"] = (1.0, 1.0, 1.0, 1.0, 1 / 6)
        rw.calculate_reward(ev, "")
        ev = dict(ev, party_resources=(0.5, 1.0, 0.5, 1.0, 1 / 6))
        rw.calculate_reward(ev, "")
        self.assertAlmostEqual(rw.get_episode_breakdown()["resource_potential"], -5.0)


if __name__ == "__main__":
    unittest.main()
