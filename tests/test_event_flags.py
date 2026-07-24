# -*- coding: utf-8 -*-
"""Lock the derived story-flag numbers to pret/pokecrystal.

`_DERIVED_FLAG_TABLE` in gym_env.py maps a feature name to an absolute bit
index into wEventFlags (0xDA72). Those indices MUST match
constants/event_flags.asm exactly — a previous hand-maintained table was
off by varying amounts because it didn't account for `const_skip`
directives (e.g. got_starter listed as 25 when it is really 26,
learned_to_catch listed as 63 when it is really 66). This test parses the
vendored asm and fails if any number disagrees with the asm index for its
documented EVENT_ name, so the table can't silently drift again.

The vendored asm (`PoliwhiRL/environment/event_flags.asm`) was confirmed
against the live ROM: after picking Cyndaquil, bits 26 (got_starter) and
27 (got_cyndaquil) are the persistently-set event bits.
"""
import os
import re
import unittest

from PoliwhiRL.environment.gym_env import _DERIVED_FLAG_TABLE
from PoliwhiRL.checkpoints import checkpoint_title, is_recordable_checkpoint

ASM = os.path.join(
    os.path.dirname(__file__), "..", "PoliwhiRL", "environment", "event_flags.asm"
)

# Canonical feature -> EVENT_ name mapping. The single source of truth for
# what each derived-flag feature is *supposed* to track.
FEATURE_EVENT = {
    "got_starter": "EVENT_GOT_A_POKEMON_FROM_ELM",
    "got_cyndaquil": "EVENT_GOT_CYNDAQUIL_FROM_ELM",
    "got_totodile": "EVENT_GOT_TOTODILE_FROM_ELM",
    "got_chikorita": "EVENT_GOT_CHIKORITA_FROM_ELM",
    "got_mystery_egg": "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON",
    "gave_mystery_egg": "EVENT_GAVE_MYSTERY_EGG_TO_ELM",
    "got_berry_route_30": "EVENT_GOT_BERRY_FROM_ROUTE_30_HOUSE",
    "got_togepi_egg": "EVENT_GOT_TOGEPI_EGG_FROM_ELMS_AIDE",
    "dude_talked": "EVENT_DUDE_TALKED_TO_YOU",
    "learned_to_catch": "EVENT_LEARNED_TO_CATCH_POKEMON",
    "has_cut": "EVENT_GOT_HM01_CUT",
    "has_fly": "EVENT_GOT_HM02_FLY",
    "has_surf": "EVENT_GOT_HM03_SURF",
    "has_strength": "EVENT_GOT_HM04_STRENGTH",
    "has_flash": "EVENT_GOT_HM05_FLASH",
    "has_whirlpool": "EVENT_GOT_HM06_WHIRLPOOL",
    "has_waterfall": "EVENT_GOT_HM07_WATERFALL",
    "has_old_rod": "EVENT_GOT_OLD_ROD",
    "has_good_rod": "EVENT_GOT_GOOD_ROD",
    "has_super_rod": "EVENT_GOT_SUPER_ROD",
    "cleared_radio_tower": "EVENT_CLEARED_RADIO_TOWER",
    "cleared_rocket_hideout": "EVENT_CLEARED_ROCKET_HIDEOUT",
    "made_whitney_cry": "EVENT_MADE_WHITNEY_CRY",
    "herded_farfetchd": "EVENT_HERDED_FARFETCHD",
    "fought_sudowoodo": "EVENT_FOUGHT_SUDOWOODO",
    "cleared_slowpoke_well": "EVENT_CLEARED_SLOWPOKE_WELL",
    "got_bicycle": "EVENT_GOT_BICYCLE",
    "released_the_beasts": "EVENT_RELEASED_THE_BEASTS",
    "rival_cherrygrove": "EVENT_RIVAL_CHERRYGROVE_CITY",
    "beat_rival_mt_moon": "EVENT_BEAT_RIVAL_IN_MT_MOON",
    "beat_falkner": "EVENT_BEAT_FALKNER",
    "beat_bugsy": "EVENT_BEAT_BUGSY",
    "beat_whitney": "EVENT_BEAT_WHITNEY",
    "beat_morty": "EVENT_BEAT_MORTY",
    "beat_jasmine": "EVENT_BEAT_JASMINE",
    "beat_chuck": "EVENT_BEAT_CHUCK",
    "beat_pryce": "EVENT_BEAT_PRYCE",
    "beat_clair": "EVENT_BEAT_CLAIR",
    "beat_brock": "EVENT_BEAT_BROCK",
    "beat_misty": "EVENT_BEAT_MISTY",
    "beat_erika": "EVENT_BEAT_ERIKA",
    "beat_janine": "EVENT_BEAT_JANINE",
    "beat_sabrina": "EVENT_BEAT_SABRINA",
    "beat_blaine": "EVENT_BEAT_BLAINE",
    "beat_blue": "EVENT_BEAT_BLUE",
    "beat_champion_lance": "EVENT_BEAT_CHAMPION_LANCE",
    "fought_ho_oh": "EVENT_FOUGHT_HO_OH",
    "fought_lugia": "EVENT_FOUGHT_LUGIA",
    "talked_to_mom": "EVENT_PLAYERS_HOUSE_MOM_1",
}


def _parse_asm():
    """name -> absolute wEventFlags index.

    Honours every directive that moves the constant counter:
      const_def [N]   -> reset counter to N (default 0)
      const_skip [N]  -> advance counter by N (default 1), no name
      const_next N    -> jump counter to absolute N (used to leave large
                         gaps for unused ranges; ignoring this is what made
                         every late-game flag number in the table too low)
      const NAME      -> record NAME at the counter, then advance by 1
    """
    name2idx, idx = {}, None
    with open(ASM) as f:
        for line in f:
            s = line.split(";")[0].strip()
            if not s:
                continue
            m = re.match(r"const_def(?:\s+(-?\d+))?$", s)
            if m:
                idx = int(m.group(1)) if m.group(1) else 0
                continue
            if idx is None:
                continue
            m = re.match(r"const_next\s+(\d+)$", s)
            if m:
                idx = int(m.group(1))
                continue
            m = re.match(r"const_skip(?:\s+(\d+))?$", s)
            if m:
                idx += int(m.group(1)) if m.group(1) else 1
                continue
            m = re.match(r"const\s+([A-Z0-9_]+)$", s)
            if m:
                name2idx[m.group(1)] = idx
                idx += 1
    return name2idx


class TestEventFlagTable(unittest.TestCase):
    def setUp(self):
        self.assertTrue(os.path.isfile(ASM), "vendored event_flags.asm missing")
        self.name2idx = _parse_asm()

    def test_every_table_number_matches_asm(self):
        for num, feature in _DERIVED_FLAG_TABLE:
            self.assertIn(
                feature, FEATURE_EVENT, f"{feature} missing from FEATURE_EVENT map"
            )
            event = FEATURE_EVENT[feature]
            self.assertIn(event, self.name2idx, f"{event} not found in event_flags.asm")
            self.assertEqual(
                num,
                self.name2idx[event],
                f"{feature}: table has {num}, asm says {self.name2idx[event]} "
                f"for {event}",
            )

    def test_curriculum_flag_goals_match_asm(self):
        # The flag numbers used by the curriculum stage configs.
        self.assertEqual(self.name2idx["EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON"], 30)
        self.assertEqual(self.name2idx["EVENT_GAVE_MYSTERY_EGG_TO_ELM"], 31)
        self.assertEqual(self.name2idx["EVENT_LEARNED_TO_CATCH_POKEMON"], 66)
        self.assertEqual(self.name2idx["EVENT_PLAYERS_HOUSE_MOM_1"], 1735)

    def test_recordable_checkpoint_titles_cover_stable_derived_flags(self):
        titles = []
        for flag_num, _feature in _DERIVED_FLAG_TABLE:
            if is_recordable_checkpoint(flag_num):
                title = checkpoint_title(flag_num)
                self.assertTrue(title)
                self.assertNotIn("/", title)
                titles.append(title)
        self.assertEqual(len(titles), len(set(titles)))


if __name__ == "__main__":
    unittest.main()
