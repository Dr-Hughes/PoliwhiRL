# -*- coding: utf-8 -*-
"""Human-readable metadata for story checkpoints tracked from event flags.

The ordered flag table remains the observation-vector contract.  Recording and
reporting deliberately exclude transient/non-monotonic flags: they are useful
as observations, but do not represent durable game progress.
"""

import unicodedata

DERIVED_FLAG_TABLE = (
    (26, "got_starter"),
    (27, "got_cyndaquil"),
    (28, "got_totodile"),
    (29, "got_chikorita"),
    (30, "got_mystery_egg"),
    (31, "gave_mystery_egg"),
    (39, "got_berry_route_30"),
    (45, "got_togepi_egg"),
    (65, "dude_talked"),
    (66, "learned_to_catch"),
    (16, "has_cut"),
    (17, "has_fly"),
    (18, "has_surf"),
    (19, "has_strength"),
    (20, "has_flash"),
    (21, "has_whirlpool"),
    (1672, "has_waterfall"),
    (23, "has_old_rod"),
    (24, "has_good_rod"),
    (25, "has_super_rod"),
    (33, "cleared_radio_tower"),
    (34, "cleared_rocket_hideout"),
    (40, "made_whitney_cry"),
    (41, "herded_farfetchd"),
    (42, "fought_sudowoodo"),
    (43, "cleared_slowpoke_well"),
    (91, "got_bicycle"),
    (123, "released_the_beasts"),
    (1726, "rival_cherrygrove"),
    (793, "beat_rival_mt_moon"),
    (1213, "beat_falkner"),
    (1214, "beat_bugsy"),
    (1215, "beat_whitney"),
    (1216, "beat_morty"),
    (1217, "beat_jasmine"),
    (1218, "beat_chuck"),
    (1219, "beat_pryce"),
    (1220, "beat_clair"),
    (1221, "beat_brock"),
    (1222, "beat_misty"),
    (1224, "beat_erika"),
    (1225, "beat_janine"),
    (1226, "beat_sabrina"),
    (1227, "beat_blaine"),
    (1228, "beat_blue"),
    (1468, "beat_champion_lance"),
    (791, "fought_ho_oh"),
    (792, "fought_lugia"),
    (1735, "talked_to_mom"),
)

_FEATURE_BY_FLAG = dict(DERIVED_FLAG_TABLE)

# These bits are observation features, not durable checkpoints.
NON_RECORDABLE_FLAGS = frozenset({26, 1726})

_TITLE_OVERRIDES = {
    30: "Got Mystery Egg From Mr. Pokémon",
    31: "Gave Mystery Egg To Elm",
    39: "Got Berry From Route 30 House",
    45: "Got Togepi Egg From Elm's Aide",
    65: "Talked To Catching Tutorial Dude",
    66: "Learned To Catch Pokémon",
    1735: "Talked To Mom",
}


def is_recordable_checkpoint(flag_num):
    """Whether ``flag_num`` is a stable, meaningful story checkpoint."""
    flag_num = int(flag_num)
    return flag_num in _FEATURE_BY_FLAG and flag_num not in NON_RECORDABLE_FLAGS


def checkpoint_title(flag_num):
    """Return a stable human-readable title for a tracked checkpoint flag."""
    flag_num = int(flag_num)
    if flag_num in _TITLE_OVERRIDES:
        return _TITLE_OVERRIDES[flag_num]
    feature = _FEATURE_BY_FLAG.get(flag_num, f"Flag {flag_num}")
    return feature.replace("_", " ").title()


def checkpoint_slug(flag_num):
    """Filesystem-safe, sortable checkpoint folder name."""
    flag_num = int(flag_num)
    title = unicodedata.normalize("NFKD", checkpoint_title(flag_num))
    title = title.encode("ascii", "ignore").decode("ascii")
    safe = "".join(c if c.isalnum() else "-" for c in title)
    safe = "-".join(part for part in safe.split("-") if part)
    return f"{flag_num:04d}-{safe}"
