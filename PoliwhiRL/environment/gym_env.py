# -*- coding: utf-8 -*-
import io
import json
import math
import os
import pickle
import shutil
import tempfile
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from . import RAM
from PoliwhiRL.utils.visuals import record_step
from .rewards import Rewards, is_ram_state_valid
from .visit_archive import VisitArchive
from pyboy import PyBoy


# Stable ordering for the RAM observation vector. Treat this as a contract:
# changing the order or removing an entry will invalidate trained models.
# New features should be appended to the end. The model's RAM encoder reads
# its input dim from RAM_OBS_DIM at startup, so additions here are
# automatically picked up by the model.
# Raw 256-byte story flags replaced by curated _DERIVED_FLAG_TABLE (~70 bits).
STORY_FLAGS_NUM_BYTES = 256  # Still read from RAM for bit extraction

# Derived flag table: extracted from story-flag bytes (0xDA72–0xDB71).
# Each entry is (flag_number, "feature_name"). flag_number -> byte = flag_number // 8, bit = flag_number % 8.
# When the bit is SET (1), the flag is true.
#
# All flag numbers verified against pret/pokecrystal/constants/event_flags.asm.
# Flag numbers are absolute indices into wEventFlags (0xDA72), taken
# directly from pret/pokecrystal/constants/event_flags.asm — VENDORED in
# this directory as `event_flags.asm` and parsed by the regression test
# `tests/test_event_flags.py` (which fails if any number below disagrees
# with the asm for its documented EVENT_ name).
#
# Two classes of error have bitten this table before:
#   1. `const_skip` (unused slots) — got_starter is 26, not 25.
#   2. `const_next N` jumps — the asm leaves large gaps between flag groups
#      (const_next 200, const_next 600, ...). Every flag AFTER the first jump
#      (gym leaders, rivals, legendaries, HM07 waterfall) lives at a far
#      higher index than a naive sequential count gives. This is what made
#      ~21 late-game entries off by +345/+512/+628; e.g. beat_falkner is 1213
#      not 701, rival_cherrygrove is 1726 not 1098, waterfall is 1672 not 1044.
# Both the table AND the test parser must honour const_next, or the test
# silently reproduces the same wrong numbers and passes anyway.
# Confirmed against live ROM save states (Scripts/verify_flags_from_states.py):
# after picking Cyndaquil bits 26/27 are the persistently-set ones, and at the
# post-Cherrygrove-rival state flag 1726 is SET while the old 1098 is clear.
# NOTE: EVENT_GOT_A_POKEMON_FROM_ELM (26) is *transient* — the script sets
# it then clears it — so don't use it as a goal terminal; use a persistent
# state signal (pokedex_owned) instead. See tests/test_event_flags.py for
# the feature -> EVENT_ name mapping used to validate these numbers.
_DERIVED_FLAG_TABLE = [
    # ----- Starter sequence -----
    (26, "got_starter"),                 # EVENT_GOT_A_POKEMON_FROM_ELM
    (27, "got_cyndaquil"),               # EVENT_GOT_CYNDAQUIL_FROM_ELM
    (28, "got_totodile"),                # EVENT_GOT_TOTODILE_FROM_ELM
    (29, "got_chikorita"),               # EVENT_GOT_CHIKORITA_FROM_ELM
    # ----- Mystery-egg quest -----
    (30, "got_mystery_egg"),             # EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON
    (31, "gave_mystery_egg"),            # EVENT_GAVE_MYSTERY_EGG_TO_ELM
    (39, "got_berry_route_30"),          # EVENT_GOT_BERRY_FROM_ROUTE_30_HOUSE
    (45, "got_togepi_egg"),              # EVENT_GOT_TOGEPI_EGG_FROM_ELMS_AIDE
    # ----- Catch tutorial (implies player has pokeballs) -----
    (65, "dude_talked"),                 # EVENT_DUDE_TALKED_TO_YOU
    (66, "learned_to_catch"),            # EVENT_LEARNED_TO_CATCH_POKEMON
    # ----- HMs -----
    (16, "has_cut"),                     # EVENT_GOT_HM01_CUT
    (17, "has_fly"),                     # EVENT_GOT_HM02_FLY
    (18, "has_surf"),                    # EVENT_GOT_HM03_SURF
    (19, "has_strength"),                # EVENT_GOT_HM04_STRENGTH
    (20, "has_flash"),                   # EVENT_GOT_HM05_FLASH
    (21, "has_whirlpool"),               # EVENT_GOT_HM06_WHIRLPOOL
    (1672, "has_waterfall"),             # EVENT_GOT_HM07_WATERFALL
    # ----- Rods -----
    (23, "has_old_rod"),                 # EVENT_GOT_OLD_ROD
    (24, "has_good_rod"),                # EVENT_GOT_GOOD_ROD
    (25, "has_super_rod"),               # EVENT_GOT_SUPER_ROD
    # ----- Johto story milestones -----
    (33, "cleared_radio_tower"),         # EVENT_CLEARED_RADIO_TOWER
    (34, "cleared_rocket_hideout"),      # EVENT_CLEARED_ROCKET_HIDEOUT
    (40, "made_whitney_cry"),            # EVENT_MADE_WHITNEY_CRY
    (41, "herded_farfetchd"),            # EVENT_HERDED_FARFETCHD
    (42, "fought_sudowoodo"),            # EVENT_FOUGHT_SUDOWOODO
    (43, "cleared_slowpoke_well"),       # EVENT_CLEARED_SLOWPOKE_WELL
    (91, "got_bicycle"),                 # EVENT_GOT_BICYCLE
    (123, "released_the_beasts"),        # EVENT_RELEASED_THE_BEASTS
    # ----- Rival battles -----
    # NOTE: EVENT_RIVAL_CHERRYGROVE_CITY (1726) is a *sprite-visibility* flag
    # (asm "Sprite visibility flags" section: set => sprite hidden), NOT a
    # persistent "beat the rival" milestone. It is SET at game start, cleared
    # when the rival appears, then set again afterwards — so it is low-info /
    # non-monotonic. Kept for observation-vector stability; the model should
    # learn to ignore it. There is no dedicated early-rival victory flag to use
    # instead (EVENT_BEAT_RIVAL_IN_MT_MOON below is the only true rival flag).
    # Flagged automatically by Scripts/verify_flags_from_states.py.
    (1726, "rival_cherrygrove"),         # EVENT_RIVAL_CHERRYGROVE_CITY (sprite-vis)
    (793, "beat_rival_mt_moon"),         # EVENT_BEAT_RIVAL_IN_MT_MOON
    # ----- Johto gym leaders -----
    (1213, "beat_falkner"),              # EVENT_BEAT_FALKNER
    (1214, "beat_bugsy"),                # EVENT_BEAT_BUGSY
    (1215, "beat_whitney"),              # EVENT_BEAT_WHITNEY
    (1216, "beat_morty"),                # EVENT_BEAT_MORTY
    (1217, "beat_jasmine"),              # EVENT_BEAT_JASMINE
    (1218, "beat_chuck"),                # EVENT_BEAT_CHUCK
    (1219, "beat_pryce"),                # EVENT_BEAT_PRYCE
    (1220, "beat_clair"),                # EVENT_BEAT_CLAIR
    # ----- Kanto gym leaders -----
    (1221, "beat_brock"),                # EVENT_BEAT_BROCK
    (1222, "beat_misty"),                # EVENT_BEAT_MISTY
    (1224, "beat_erika"),                # EVENT_BEAT_ERIKA
    (1225, "beat_janine"),               # EVENT_BEAT_JANINE
    (1226, "beat_sabrina"),              # EVENT_BEAT_SABRINA
    (1227, "beat_blaine"),               # EVENT_BEAT_BLAINE
    (1228, "beat_blue"),                 # EVENT_BEAT_BLUE
    # ----- Champion / legendaries -----
    (1468, "beat_champion_lance"),       # EVENT_BEAT_CHAMPION_LANCE
    (791, "fought_ho_oh"),               # EVENT_FOUGHT_HO_OH
    (792, "fought_lugia"),               # EVENT_FOUGHT_LUGIA
    # ----- Leaving the house (appended — see append-only contract above) -----
    (1735, "talked_to_mom"),             # EVENT_PLAYERS_HOUSE_MOM_1
]

_BASE_RAM_FEATURE_KEYS = (
    "x",
    "y",
    # Facing direction one-hot: up=1, down=2, left=3, right=4.
    # Critical for navigation — identical (x,y) with different facing is
    # a distinct game state. Unseen values get all-zeros (safe fallback).
    "facing_up",
    "facing_down",
    "facing_left",
    "facing_right",
    "map_num",
    "map_bank",
    "room",
    "warp_number",
    "party_size",
    "party_level",
    "party_hp",
    "party_exp",
    "money",
    "pokedex_seen",
    "pokedex_owned",
    "collision_down",
    "collision_up",
    "collision_left",
    "collision_right",
    # Per-episode exploration summary.
    "explored_tile_count",
    # Per-episode count of unique (map_bank, map_num) visited. Surfaces
    # the map-novelty potential to the policy directly.
    "maps_visited_this_episode",
    # Progress counters. Same numeric value means the same thing across
    # stages — the policy can carry "I've already triggered N flag fires
    # this episode" forward across the replay boundary.
    "n_pokedex_goals_completed",
    "n_level_goals_completed",
    "n_flag_goals_completed",
    # Priority 1 raw features. battle_type replaced by one-hot so the model
    # can distinguish states (0→0, 1→0.0039, 2→0.0078 under /255 was
    # indistinguishable noise). Three slots: none, wild, trainer.
    "battle_none",
    "battle_wild",
    "battle_trainer",
    # Badges: popcount of the 8-bit bitmask (one bit per Johto gym badge),
    # normalised by /8 so the feature stays in [0, 1]. The raw bitmask was
    # previously /255 which gave ~0 for any badge count < 4.
    "johto_badges_count",
    # Player state one-hot: replaces /255 scaling which was indistinguishable
    # noise (0→0, 1→0.0039, 2→0.0078). Known values:
    #   0 = walking / idle, 1 = in battle, 2 = cycling, 4 = surfing/diving
    "player_state_walking",
    "player_state_battle",
    "player_state_cycling",
    "player_state_surfing",
    "key_items_count",
    "game_hour",
    "bgm_id",
    # Battle context: the policy needs to see when an enemy is close to
    # KO so it can choose Catch vs another attack. log1p-scaled because
    # HP ranges 0 to a few hundred in the early game.
    "enemy_hp",
    # Ratio of current enemy HP to max (clamped [0,1]). Gives the model
    # direct "is this enemy near KO?" signal without having to learn it
    # from two separate features. Outside battle both are stale — guarded
    # by the battle_type one-hot.
    "enemy_hp_ratio",
    # Player's in-battle Pokemon move PP (C634-C637). Normalised by /64 so
    # the feature stays in [0, 1]. Outside battle the values are stale — the
    # battle_none/wild/trainer one-hot tells the model not to trust them.
    "player_pp1",
    "player_pp2",
    "player_pp3",
    "player_pp4",
    # Verified script / UI state bytes (see RAM_MAPPING.md). All three are
    # encoded as small one-hots so the policy doesn't have to learn the
    # discrete-value semantics from a /255 scaled float. Buckets cover
    # every value seen in the 278-step gold-run playthrough plus an
    # ``other`` catch-all for unseen values.
    #
    # 0xD438 — scripted-overlay flag (binary; 1 = script active / locked).
    "script_active",
    # 0xCF07 — UI rendering / text-box state.
    "ui_state_walking_indoor",   # =5
    "ui_state_walking_outdoor",  # =0
    "ui_state_text_box",          # =7
    "ui_state_transition",        # =1
    "ui_state_other",
    # 0xD43D — map handler / script bank context.
    "map_handler_indoor",         # =128
    "map_handler_outdoor",        # =165
    "map_handler_script_active",  # =30
    "map_handler_transition",     # =0
    "map_handler_other",
    # Map-goal progress (appended last per the append-only contract; it
    # belongs logically with the other n_*_goals counters above). Map goals
    # are the backbone of the navigation curriculum, and with replay
    # seeding an episode can start with several already complete — the
    # policy must be able to see "which rung of the ladder am I on" the
    # same way it sees pokedex/flag progress, and the agent's
    # goals_at_start snapshot must count them or goals_made is inflated
    # for seeded episodes.
    "n_map_goals_completed",
    # Ordered recent-map history: the last 6 unique maps entered during
    # the training portion of this episode (bank, num pairs, oldest first).
    # Slots are zero-padded when fewer maps have been visited, so the
    # vector position is stable. Gives the transformer explicit episode-
    # level map memory without requiring a context window that spans the
    # full episode length. ram_recent_maps_n in config controls the count;
    # the feature count here (12 = 6 * 2) must equal 2 * ram_recent_maps_n.
    "recent_map_bank_0", "recent_map_num_0",
    "recent_map_bank_1", "recent_map_num_1",
    "recent_map_bank_2", "recent_map_num_2",
    "recent_map_bank_3", "recent_map_num_3",
    "recent_map_bank_4", "recent_map_num_4",
    "recent_map_bank_5", "recent_map_num_5",
    # Explicit exploration-frontier features (appended last per the
    # append-only contract). These mirror exactly what the frontier-novelty
    # reward pays for, so the POLICY can directly perceive its own
    # exploration state instead of only inferring it indirectly through the
    # value function. See Rewards.last_cell_novel_flag / steps_since_novel_cell.
    "cell_novel_this_episode",
    "steps_since_novel_cell",
)
# Derived flags are appended after raw base features. Raw 256-byte story-flag
# bytes have been removed in favour of the curated _DERIVED_FLAG_TABLE.
_DERIVED_FLAG_KEYS = tuple(name for _, name in _DERIVED_FLAG_TABLE)
RAM_FEATURE_KEYS = _BASE_RAM_FEATURE_KEYS + _DERIVED_FLAG_KEYS
RAM_OBS_DIM = len(RAM_FEATURE_KEYS)
_BASE_RAM_LEN = len(_BASE_RAM_FEATURE_KEYS)

# Named-index helpers so downstream code (vec agent, eval tools) can read
# specific scalars out of the RAM vector without hard-coding integer
# positions that would drift as features are appended.
RAM_FEATURE_INDEX = {name: i for i, name in enumerate(RAM_FEATURE_KEYS)}
N_POK_GOALS_RAM_IDX = RAM_FEATURE_INDEX["n_pokedex_goals_completed"]
N_LVL_GOALS_RAM_IDX = RAM_FEATURE_INDEX["n_level_goals_completed"]
N_FLAG_GOALS_RAM_IDX = RAM_FEATURE_INDEX["n_flag_goals_completed"]
N_MAP_GOALS_RAM_IDX = RAM_FEATURE_INDEX["n_map_goals_completed"]


BATTLE_STATE_LABELS = {0: "none", 1: "wild", 2: "trainer"}
PLAYER_STATE_LABELS = {0: "walk", 1: "bike", 2: "skate", 4: "surf"}


def _safe_state_label(state_path):
    """Filename-safe identifier for a save-state path.

    Strips the directory and trailing ``.state`` suffix, then replaces any
    underscores so the result is one token in the underscore-separated
    PNG filename layout (keeps ``png_to_video.py``'s split-by-underscore
    sort logic intact).
    """
    if not state_path:
        return "none"
    base = os.path.basename(str(state_path))
    if base.endswith(".state"):
        base = base[: -len(".state")]
    return base.replace("_", "-") or "none"


def _battle_state_label(value):
    """Map a raw battle_type RAM value to a short filename-safe label."""
    return BATTLE_STATE_LABELS.get(int(value), f"unknown_{value}")


def _player_state_label(value):
    """Map a raw player_state RAM value to a short filename-safe label."""
    return PLAYER_STATE_LABELS.get(int(value), f"unknown_{value}")


def _extract_derived_flags(story_flags):
    """Extract individual bits from story-flag bytes as binary features.

    Parameters
    ----------
    story_flags : ndarray of shape (256,) uint8
        Raw story-flag bytes from RAM 0xDA72–0xDB71.

    Returns
    -------
    dict mapping feature name -> 0.0 or 1.0
    """
    result = {}
    for flag_num, name in _DERIVED_FLAG_TABLE:
        byte_idx = flag_num // 8
        bit_idx = flag_num % 8
        bit = (story_flags[byte_idx] >> bit_idx) & 1
        result[name] = float(bit)
    return result


def _one_hot_bucket(value, buckets):
    """Return a one-hot list of len(buckets)+1. Last slot is "other"."""
    out = [0.0] * (len(buckets) + 1)
    for i, b in enumerate(buckets):
        if value == b:
            out[i] = 1.0
            return out
    out[-1] = 1.0
    return out


def _build_ram_vector(
    env_vars,
    explored_tile_count,
    maps_visited_this_episode,
    n_pokedex_goals_completed,
    n_level_goals_completed,
    n_flag_goals_completed,
    n_map_goals_completed,
    script_state_bytes,
    recent_maps=None,
    cell_novel_this_episode=0.0,
    steps_since_novel_cell=0,
):
    """Pack RAM + exploration + progress scalars into a fixed-order
    ~[0, 1]-scaled float32 vector. Single source of truth — env, tests,
    any eval tool should construct the vector via this function.

    Parameters
    ----------
    env_vars : dict
        Output of RAM.RAMManagement.get_variables().
    explored_tile_count : int
        len(Rewards.explored_tiles) at this step.
    maps_visited_this_episode : int
        Number of unique (map_bank, map_num) visited this episode.
    n_pokedex_goals_completed : int
        Rewards.pokedex_goals_completed.
    n_level_goals_completed : int
        Rewards.level_goals_completed.
    n_flag_goals_completed : int
        Rewards.flag_goals_completed.
    n_map_goals_completed : int
        Rewards.n_map_goals_completed() (map-reach + maps-visited goals).
    script_state_bytes : tuple
        (d438, cf07, d43d) raw byte values from the empirically-verified
        script / UI state addresses (see RAM_MAPPING.md). Encoded into
        one-hot features so the policy doesn't have to discover discrete
        semantics from a scaled float.
    recent_maps : list of (int, int) or None
        Last N (map_bank, map_num) pairs entered during the training portion
        of the episode, oldest first. Padded with (0, 0) at the front.
        Defaults to all-zeros when None.
    """
    party_size, party_level, party_hp, party_exp = env_vars["party_info"]
    d438, cf07, d43d = script_state_bytes
    _facing = int(env_vars.get("player_direction", 0))
    # Pre-compute enemy HP ratio (can't use inline assignment in list literal).
    _enemy_hp = max(0, int(env_vars.get("enemy_hp", 0)))
    _enemy_max = max(1, int(env_vars.get("enemy_max_hp", 1)))
    _enemy_hp_ratio = min(_enemy_hp / _enemy_max, 1.0)
    base_scalars = [
        min(env_vars["X"], 32) / 32.0,
        min(env_vars["Y"], 32) / 32.0,
        # Facing direction one-hot (1=up, 2=down, 3=left, 4=right).
        # Unseen values get all-zeros.
        1.0 if _facing == 1 else 0.0,   # up
        1.0 if _facing == 2 else 0.0,   # down
        1.0 if _facing == 3 else 0.0,   # left
        1.0 if _facing == 4 else 0.0,   # right
        env_vars["map_num"] / 255.0,
        env_vars["map_bank"] / 255.0,
        env_vars["room"] / 255.0,
        env_vars["warp_number"] / 255.0,
        party_size / 6.0,
        party_level / 100.0,
        party_hp / 1000.0,
        math.log1p(max(0, party_exp)) / 20.0,
        env_vars["money"] / 1_000_000.0,
        env_vars["pokedex_seen"] / 251.0,
        env_vars["pokedex_owned"] / 251.0,
        env_vars["collision_down"] / 255.0,
        env_vars["collision_up"] / 255.0,
        env_vars["collision_left"] / 255.0,
        env_vars["collision_right"] / 255.0,
        math.log1p(max(0, explored_tile_count)) / 6.0,
        # log1p-scaled so it stays in a comparable range to other features.
        # Without scaling this grows unbounded (50+ maps) while the rest of
        # the vector is [0, 1]. log1p(50) / 4 ≈ 0.83, good range.
        math.log1p(max(0, maps_visited_this_episode)) / 4.0,
        # Goal counters: log1p-scaled so they stay bounded in [0, ~0.7]
        # even when counters grow large (e.g. 251 pokedex fires).
        # log1p(251) / 6 ≈ 0.93, good range.
        math.log1p(max(0, n_pokedex_goals_completed)) / 6.0,
        math.log1p(max(0, n_level_goals_completed)) / 6.0,
        math.log1p(max(0, n_flag_goals_completed)) / 6.0,
        # battle_type one-hot: replaces /255 scaling which produced
        # indistinguishable near-zero values (0→0, 1→0.0039, 2→0.0078).
        # The model needs to distinguish "not in battle" from "in wild
        # battle" vs "in trainer battle".
        1.0 if int(env_vars["battle_type"]) == 0 else 0.0,
        1.0 if int(env_vars["battle_type"]) == 1 else 0.0,
        1.0 if int(env_vars["battle_type"]) == 2 else 0.0,
        # Badges: popcount of the 8-bit bitmask (one bit per Johto gym badge),
        # normalised by /8 so the feature stays in [0, 1].
        bin(int(env_vars["johto_badges"])).count("1") / 8.0,
        # Player state one-hot: known values 0=walking, 1=battle, 2=cycling,
        # 4=surfing/diving. Unseen values get all-zeros.
        1.0 if int(env_vars["player_state"]) == 0 else 0.0,
        1.0 if int(env_vars["player_state"]) == 1 else 0.0,
        1.0 if int(env_vars["player_state"]) == 2 else 0.0,
        1.0 if int(env_vars["player_state"]) == 4 else 0.0,
        env_vars["key_items_count"] / 25.0,
        env_vars["game_hour"] / 255.0,
        env_vars["bgm_id"] / 255.0,
        math.log1p(max(0, int(env_vars.get("enemy_hp", 0)))) / 6.0,
        # Enemy HP ratio: current/max clamped [0,1]. Direct "is this enemy
        # near KO?" signal. Guarded by battle_type one-hot outside battle.
        _enemy_hp_ratio,
        # Player's in-battle Pokemon move PP (C634-C637). /64 normalisation
        # assumes max PP per move ≈ 64. Outside battle these are stale but
        # the battle one-hot tells the model to ignore them.
        min(int(env_vars.get("player_move_pp", (0, 0, 0, 0))[0]), 64) / 64.0,
        min(int(env_vars.get("player_move_pp", (0, 0, 0, 0))[1]), 64) / 64.0,
        min(int(env_vars.get("player_move_pp", (0, 0, 0, 0))[2]), 64) / 64.0,
        min(int(env_vars.get("player_move_pp", (0, 0, 0, 0))[3]), 64) / 64.0,
        # 0xD438 == 255 ⇒ script active (binary feature).
        1.0 if int(d438) == 255 else 0.0,
    ]
    # 0xCF07 one-hot over observed values.
    base_scalars.extend(_one_hot_bucket(int(cf07), [5, 0, 7, 1]))
    # 0xD43D one-hot.
    base_scalars.extend(_one_hot_bucket(int(d43d), [128, 165, 30, 0]))
    # Map-goal progress counter — appended last (append-only contract);
    # same log1p/6 scaling as the other goal counters.
    base_scalars.append(math.log1p(max(0, n_map_goals_completed)) / 6.0)
    # Recent-map history: 6 * (bank/255, num/255) pairs, oldest first.
    # Gives the policy explicit within-episode trajectory context that
    # TransformerXL memory can't reliably retain over 500+ step episodes.
    _rm = recent_maps if recent_maps is not None else [(0, 0)] * 6
    for _bank, _num in _rm:
        base_scalars.append(float(_bank) / 255.0)
        base_scalars.append(float(_num) / 255.0)

    # Exploration-frontier features: is the current cell new this episode,
    # and how long since one last was. log1p-scaled so a long dry spell
    # doesn't dominate the ~[0,1] vector.
    base_scalars.append(float(cell_novel_this_episode))
    base_scalars.append(math.log1p(max(0, int(steps_since_novel_cell))) / 6.0)

    base = np.array(base_scalars, dtype=np.float32)
    if base.size != len(_BASE_RAM_FEATURE_KEYS):
        raise AssertionError(
            f"RAM-vector base size mismatch: built {base.size}, expected "
            f"{len(_BASE_RAM_FEATURE_KEYS)} (RAM_FEATURE_KEYS contract)."
        )
    story_flags_raw = np.asarray(env_vars["story_flags"], dtype=np.uint8)
    if story_flags_raw.size != STORY_FLAGS_NUM_BYTES:
        raise ValueError(
            f"Expected {STORY_FLAGS_NUM_BYTES} story flag bytes, got {story_flags_raw.size}"
        )
    derived = np.array(
        list(_extract_derived_flags(story_flags_raw).values()),
        dtype=np.float32,
    )
    return np.concatenate([base, derived])


class PyBoyEnvironment(gym.Env):
    def __init__(self, config, force_window=False):
        super().__init__()
        self.config = config
        self._is_closed = False

        self.frames_per_action = 90
        self.button_hold_frames = 15
        self._fitness = 0
        self.steps = 0
        self.done = False
        self.episode = -2
        self._last_env_vars = None
        # Phase-4 frontier archive — persistent across env.reset() so cell
        # visit counts accumulate over the whole training run. Each
        # Rewards instance (recreated on every reset) shares a reference.
        self.visit_archive = VisitArchive()
        self.button = 0
        self.actions = ["", "a", "b", "left", "right", "up", "down", "start", "select"]
        self.ignored_buttons = config["ignored_buttons"]
        self.action_space = spaces.Discrete(len(self.actions))
        self.render = config["vision"]
        self.record = False
        self.use_episode_number = True
        self.record_folder = None
        self.current_max_steps = config["episode_length"]

        files_to_copy = [config["rom_path"], config["state_path"]]
        files_to_copy.extend(
            [
                file
                for file in config["extra_files"]
                if os.path.isfile(file) and os.path.getsize(file) > 0
            ]
        )

        self.check_files_exist(files_to_copy)

        self.paths = list(files_to_copy)
        self.state_path = self.paths[1]

        with open(self.state_path, "rb") as state_file:
            state_content = state_file.read()
        self.state_bytes_content = state_content

        # Copy ROM (and any sidecars) into a per-instance temp dir so PyBoy's
        # .ram/.rtc writes don't mutate the canonical files in emu_files/.
        self._tmpdir = tempfile.TemporaryDirectory(prefix="poliwhirl_emu_")
        rom_dst = os.path.join(self._tmpdir.name, os.path.basename(self.paths[0]))
        shutil.copy(self.paths[0], rom_dst)
        for extra in files_to_copy[2:]:
            shutil.copy(extra, os.path.join(self._tmpdir.name, os.path.basename(extra)))
        self.paths[0] = rom_dst

        try:
            self.pyboy = PyBoy(
                self.paths[0],
                window="null" if not force_window else "SDL2",
                sound_emulated=False,
            )
        except Exception:
            self._tmpdir.cleanup()
            raise
        self.pyboy.rtc_lock_experimental(True)
        self.pyboy.set_emulation_speed(0)
        self.ram = RAM.RAMManagement(self.pyboy)
        self.reset()

    def get_state_bytes(self):
        return io.BytesIO(self.state_bytes_content)

    def set_state_path(self, path):
        """Swap in a new save-state. Takes effect on the next reset() so
        the current episode finishes normally. Used by VecPyBoyEnv for
        per-episode state cycling across a pool of save-states.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"State file not found: {path}")
        with open(path, "rb") as f:
            self.state_bytes_content = f.read()
        self.state_path = path

    def replay_actions(self, actions):
        """Walk the env forward by replaying a sequence of actions.

        Used by eval/debug/exploration tooling (model_evaluator.py,
        debug_evaluator.py, random_walker.py) to warm-start a run from a
        captured action trajectory before observing/recording — NOT used
        by the training path (the vec worker never calls this; training
        always starts each stage from the configured save-state). The
        actions are executed without storing transitions; rewards accrued
        during replay are not counted toward anything.

        Progress state preserved across the replay boundary:

            - PyBoy memory is at the post-replay position.
            - Rewards.N_goals and pokedex_seen/owned reflect what was
              triggered during replay (so the RAM-vector progress counters
              are continuous).
            - Rewards.explored_tiles is preserved so re-walking
              replay-visited tiles doesn't pay a fresh exploration bonus.
            - The set of (map_bank, map_num) the replay touched is seeded
              into Rewards.explored_maps after start_new_episode, so the
              new_map bonus can't pay for maps the replay already explored.

        Flags already set at the post-replay state do NOT fire — only
        fresh 0→1 transitions during the training portion count.
        """
        if not actions:
            return self.get_observation()
        replay_maps = set()
        replay_cells = set()
        for a in actions:
            self._handle_action(int(a))
            self._calculate_fitness()
            env_vars = self.ram.get_variables()
            map_key = (int(env_vars["map_bank"]), int(env_vars["map_num"]))
            replay_maps.add(map_key)
            # Collect quantised cells so the (per-episode) frontier novelty
            # bonus doesn't fire on step 1 for cells the replay already
            # visited.
            replay_cells.add(self.visit_archive.cell_key(
                env_vars["map_bank"], env_vars["map_num"],
                env_vars["X"], env_vars["Y"],
            ))
        # Clear per-episode counters, then seed explored_maps with the
        # maps the replay walked through so the new_map bonus only pays
        # for maps the training segment genuinely discovers. Also seed
        # frontier cells so step 1 reward is clean.
        self.reward_calculator.start_new_episode()
        self.reward_calculator.seed_explored_maps(replay_maps)
        for cell in replay_cells:
            self.reward_calculator._novel_cells_this_episode.add(cell)
        # Seed the GoalsManager's map tracker so maps_visited goal credits
        # replay progress and is consistent with new_map_reward.
        self.reward_calculator.goals.seed_seen_maps(replay_maps)
        self.steps = 0
        self._fitness = 0
        self.done = False
        return self.get_observation()

    def check_files_exist(self, files):
        for file in files:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"File {file} not found.")

    def enable_render(self):
        self.render = True

    def _handle_action(self, action):
        frames = self.frames_per_action
        self.button = self.actions[action]
        if self.button not in self.ignored_buttons:
            self.pyboy.button(self.button, delay=self.button_hold_frames)
            frames -= self.button_hold_frames
        self.pyboy.tick(frames, self.render)
        self.steps += 1

    def step(self, action):
        self._handle_action(action)
        self._calculate_fitness()
        observation = self.get_observation()

        if self.record:
            self.save_step_img_data(
                self.record_folder, outdir=self.config["record_path"]
            )

        truncated = bool(self.reward_calculator.truncated)
        return observation, self._fitness, self.done, truncated

    def output_shape(self):
        """Image observation shape (C, H, W)."""
        if not self.config["vision"]:
            return self.get_game_area().shape
        return self.get_screen_image().shape

    def ram_observation_shape(self):
        return (RAM_OBS_DIM,)

    def get_game_area(self):
        return self.pyboy.game_area()[:18, :20].astype(np.uint8)

    def get_screen_size(self):
        return self.get_screen_image().shape

    def enable_record(self, folder, use_episode_number=True):
        self.use_episode_number = use_episode_number
        self.record = True
        self.record_folder = folder
        self.enable_render()

    def _calculate_fitness(self):
        env_vars = self.ram.get_variables()
        self._last_env_vars = env_vars
        self._fitness, reward_done = self.reward_calculator.calculate_reward(
            env_vars, self.button
        )
        if reward_done:
            self.done = True

    def get_observation(self):
        """Multi-modal observation dict {"image": ndarray, "ram": ndarray}.

        The image preserves the original screen / tilemap output for the
        CNN. The RAM vector packs position, party state, goal target, and
        exploration summary for the policy to condition on directly.
        """
        image = (
            self.get_game_area()
            if not self.config["vision"]
            else self.get_screen_image()
        )
        rc = self.reward_calculator
        env_vars = self.ram.get_variables()
        # Verified script / UI state bytes come from env_vars now (see
        # RAM_MAPPING.md). One read per step rather than four.
        ram = _build_ram_vector(
            env_vars,
            rc.explored_tile_count(),
            len(rc.goals._maps_seen_this_episode),
            rc.n_pokedex_goals_completed(),
            rc.n_level_goals_completed(),
            rc.n_flag_goals_completed(),
            rc.n_map_goals_completed(),
            (env_vars["script_byte"], env_vars["ui_byte"], env_vars["map_handler_byte"]),
            rc.recent_maps_visited(),
            cell_novel_this_episode=rc.last_cell_novel_flag(),
            steps_since_novel_cell=rc.steps_since_novel_cell(),
        )
        return {"image": image, "ram": ram}

    def reset(self):
        self.button = 0
        self.done = False
        self.record = False
        self.record_folder = None
        self.pyboy.load_state(self.get_state_bytes())
        self.reward_calculator = Rewards(self.config, visit_archive=self.visit_archive)
        self._fitness = 0
        self._handle_action(0)
        self.steps = 0
        self.episode += 1
        self.render = self.config["vision"]
        # Compute fitness BEFORE building the observation so the goal-target
        # field in the RAM vector reflects any goal advance that fired on the
        # no-op startup step.
        self._calculate_fitness()
        return self.get_observation()

    def close(self):
        if self._is_closed:
            return
        self._is_closed = True

        try:
            if hasattr(self, "pyboy"):
                self.pyboy.stop()
        except Exception as e:
            print(f"Error stopping PyBoy: {e}")

        try:
            if hasattr(self, "_tmpdir"):
                self._tmpdir.cleanup()
        except Exception as e:
            print(f"Error cleaning emu temp dir: {e}")

    def get_screen_image(self, no_resize=False):
        pil_image = self.pyboy.screen.image
        numpy_image = np.array(pil_image)[:, :, :3]

        use_grayscale = self.config["use_grayscale"]
        scaling_factor = self.config["scaling_factor"]

        if scaling_factor != 1.0 and not no_resize:
            new_width = int(numpy_image.shape[1] * scaling_factor)
            new_height = int(numpy_image.shape[0] * scaling_factor)
            numpy_image = cv2.resize(
                numpy_image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        if use_grayscale:
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
            numpy_image = np.expand_dims(numpy_image, axis=-1)

        if use_grayscale:
            numpy_image = numpy_image.transpose(2, 0, 1)
        else:
            numpy_image = numpy_image.transpose(2, 0, 1)

        return numpy_image.astype(np.uint8)

    def get_pyboy_bg(self):
        return np.array(self.pyboy.tilemap_background[:18, :20])

    def get_pyboy_wnd(self):
        return np.array(self.pyboy.tilemap_window[:18, :20])

    def get_location_data(self):
        variables = self.ram.get_variables()
        return {
            "x": variables["X"],
            "y": variables["Y"],
            "map_num": variables["map_num"],
            "room": variables["room"],
        }

    def save_step_img_data(self, fldr, outdir="./Training Outputs/Runs"):
        # Tag each saved PNG with the full game-engine location so the user
        # can manually verify goal-match conditions from the filename alone
        # without re-running the env. Also tag the save-state identifier so
        # multi-start runs can be split by starting state during review.
        variables = self.ram.get_variables()
        # Skip recording on transitional / junk RAM snapshots. Writing those
        # frames pollutes the recorded run with filenames like
        # "battlestate_unknown_122 x_0 y_0 map_0" that don't reflect a real
        # game state and make the recording harder to review.
        if not is_ram_state_valid(variables):
            return
        location = {
            "x": int(variables["X"]),
            "y": int(variables["Y"]),
            "map": int(variables["map_num"]),
            "bank": int(variables["map_bank"]),
            "room": int(variables["room"]),
            "battlestate": _battle_state_label(variables["battle_type"]),
            "playerstate": _player_state_label(variables["player_state"]),
        }
        record_step(
            self.episode if self.use_episode_number else -1,
            self.steps,
            self.pyboy.screen.image,
            self.button,
            self._fitness,
            fldr,
            outdir,
            location=location,
        )

    def _debug_init_tracking(self):
        """Lazy-init the per-run debug accumulators. Called by
        save_debug_step_img_data before the first write. Safe to call
        repeatedly — only the first call has effect."""
        if getattr(self, "_debug_prev_windows", "__sentinel__") != "__sentinel__":
            return
        self._debug_prev_windows = None
        # change_counts: {window_name: {offset_int: count_int}}
        self._debug_change_counts = {}
        # values_seen: {window_name: {offset_int: set(int)}}
        self._debug_values_seen = {}
        # first_change_step: {window_name: {offset_int: int}}
        self._debug_first_change = {}
        self._debug_frames_written = 0

    def save_debug_step_img_data(self, fldr, outdir="./Training Outputs/Runs"):
        """Debug variant of save_step_img_data.

        Differences vs. save_step_img_data:
        - Records every frame, including ones where is_ram_state_valid()
          would reject (the *point* of the debug evaluator is to see those
          transitional states).
        - Appends a curated, high-signal subset of the extended RAM probes
          to the PNG filename so menu / battle / story state is visible at
          a glance.
        - Writes a sidecar `<png-filename>.json` next to the PNG containing
          the full extended-RAM dict plus byte windows. Use this for
          bit-level diffing across the scripted action sequence.
        - Tracks which byte-window offsets change frame-to-frame and
          accumulates per-address change counts so the user can scan a
          long rollout via the run_summary.json written by
          finalize_debug_run() rather than 2000 sidecar diffs by hand.
        """
        self._debug_init_tracking()
        variables = self.ram.get_variables()
        extended = self.ram.get_extended_variables()
        byte_windows = self.ram.get_debug_byte_windows()
        party_size, party_level, party_hp, party_exp = variables["party_info"]

        # PNG filename stays compact — matches save_step_img_data so paths
        # don't blow past the 255-byte limit on macOS / ext4. All extended
        # values land in the JSON sidecar instead.
        location = {
            "x": int(variables["X"]),
            "y": int(variables["Y"]),
            "map": int(variables["map_num"]),
            "bank": int(variables["map_bank"]),
            "room": int(variables["room"]),
            "battlestate": _battle_state_label(variables["battle_type"]),
            "playerstate": _player_state_label(variables["player_state"]),
        }

        record_step(
            self.episode if self.use_episode_number else -1,
            self.steps,
            self.pyboy.screen.image,
            self.button,
            self._fitness,
            fldr,
            outdir,
            location=location,
        )

        # Sidecar JSON with everything. record_step builds its own save_dir
        # (out_dir/phase[/episode_id]) — mirror that here so the .json
        # lands next to the .png.
        save_dir = os.path.join(outdir or "Results", fldr)
        if self.use_episode_number and self.episode != -1:
            save_dir = os.path.join(save_dir, str(self.episode))

        sidecar = {
            "step": int(self.steps),
            "episode": int(self.episode if self.use_episode_number else -1),
            "button": self.button,
            "reward": float(self._fitness),
            "ram_state_valid": bool(is_ram_state_valid(variables)),
            # Mirror the base ram getter (story_flags is omitted — it's a
            # 256-byte ndarray that bloats the JSON; the curated derived
            # flags are already in extended via the event-flag layer).
            "base": {
                "X": int(variables["X"]),
                "Y": int(variables["Y"]),
                "map_num": int(variables["map_num"]),
                "map_bank": int(variables["map_bank"]),
                "warp_number": int(variables["warp_number"]),
                "room": int(variables["room"]),
                "money": int(variables["money"]),
                "pokedex_seen": int(variables["pokedex_seen"]),
                "pokedex_owned": int(variables["pokedex_owned"]),
                "collision_down": int(variables["collision_down"]),
                "collision_up": int(variables["collision_up"]),
                "collision_left": int(variables["collision_left"]),
                "collision_right": int(variables["collision_right"]),
                "battle_type": int(variables["battle_type"]),
                "johto_badges": int(variables["johto_badges"]),
                "player_state": int(variables["player_state"]),
                "key_items_count": int(variables["key_items_count"]),
                "game_hour": int(variables["game_hour"]),
                "bgm_id": int(variables["bgm_id"]),
                "enemy_hp": int(variables["enemy_hp"]),
                "enemy_max_hp": int(variables["enemy_max_hp"]),
                "party_size": int(party_size),
                "party_total_level": int(party_level),
                "party_total_hp": int(party_hp),
                "party_total_exp": int(party_exp),
            },
            "extended": {k: int(v) for k, v in extended.items()},
            "byte_windows_hex": {
                name: " ".join(f"{b:02x}" for b in bs)
                for name, bs in byte_windows.items()
            },
        }

        # Compute per-frame change deltas vs. the previous frame's
        # byte windows. The first frame seeds the baseline — every later
        # frame compares against the frame immediately before it. This is
        # what makes a 2000-step rollout actionable: 99% of frames will
        # have an empty `byte_windows_changed`, and the user can grep
        # sidecars for the non-empty ones to find script firings.
        changed = {}
        if self._debug_prev_windows is not None:
            for name, bs in byte_windows.items():
                prev = self._debug_prev_windows.get(name)
                if prev is None or len(prev) != len(bs):
                    continue
                window_start = self.ram.debug_byte_windows[name][0]
                deltas = []
                for offset, (a, b) in enumerate(zip(prev, bs)):
                    if a == b:
                        continue
                    deltas.append({
                        "offset": offset,
                        "addr": f"0x{window_start + offset:04X}",
                        "from": int(a),
                        "to": int(b),
                    })
                    # Update accumulators
                    cc = self._debug_change_counts.setdefault(name, {})
                    cc[offset] = cc.get(offset, 0) + 1
                    vs = self._debug_values_seen.setdefault(name, {})
                    vs.setdefault(offset, set()).add(int(a))
                    vs[offset].add(int(b))
                    fc = self._debug_first_change.setdefault(name, {})
                    if offset not in fc:
                        fc[offset] = int(self.steps)
                if deltas:
                    changed[name] = deltas
        sidecar["byte_windows_changed"] = changed
        self._debug_prev_windows = byte_windows
        self._debug_frames_written += 1

        # Reuse the same filename stem record_step constructed.
        loc_chunk = ""
        for key in ("x", "y", "map", "bank", "room"):
            if key in location:
                loc_chunk += f"_{key}_{int(location[key])}"
        for key, val in location.items():
            if key not in ("x", "y", "map", "bank", "room"):
                loc_chunk += f"_{key}_{val}"
        json_name = (
            f"step_{self.steps}{loc_chunk}_btn_{self.button}"
            f"_reward_{np.around(self._fitness, 4)}.json"
        )
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, json_name), "w") as f:
            json.dump(sidecar, f, indent=2)

    def finalize_debug_run(self, fldr, outdir="./Training Outputs/Runs"):
        """Emit run_summary.json aggregating which byte-window offsets
        changed during this debug run. Call once at the end of a debug
        rollout. Cheap to call even if no debug frames were written —
        in that case the summary is empty.

        The summary's `addresses` array is sorted by `changes` descending
        so the user immediately sees the most-active offsets. Each entry
        has the absolute address (hex), the window it lives in, the
        change count, the step it first changed at, and every distinct
        value the byte took during the run. Use it as the entry point
        for investigating a long rollout — drill into specific frames
        only after identifying interesting addresses here.
        """
        # Lazy-init guards against finalize-before-any-frame.
        self._debug_init_tracking()

        save_dir = os.path.join(outdir or "Results", fldr)
        if self.use_episode_number and self.episode != -1:
            save_dir = os.path.join(save_dir, str(self.episode))
        os.makedirs(save_dir, exist_ok=True)

        # Flatten into one sorted address table for the headline view.
        addresses = []
        for name, offsets in self._debug_change_counts.items():
            window_start = self.ram.debug_byte_windows[name][0]
            for offset, count in offsets.items():
                values = sorted(self._debug_values_seen[name][offset])
                addresses.append({
                    "addr": f"0x{window_start + offset:04X}",
                    "window": name,
                    "offset": offset,
                    "changes": int(count),
                    "first_change_step": int(self._debug_first_change[name][offset]),
                    "values_seen": values,
                    "n_distinct_values": len(values),
                })
        addresses.sort(key=lambda e: (-e["changes"], e["addr"]))

        summary = {
            "frames_written": int(self._debug_frames_written),
            "n_changed_addresses": len(addresses),
            "byte_window_ranges": {
                name: [f"0x{r[0]:04X}", f"0x{r[1]:04X}"]
                for name, r in self.ram.debug_byte_windows.items()
            },
            # Headline table: every address that flipped at least once.
            "addresses": addresses,
        }
        out_path = os.path.join(save_dir, "run_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        return out_path

    def save_state(self, save_path, save_name):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path + "/" + save_name, "wb") as stateFile:
            self.pyboy.save_state(stateFile)

    def save_gym_state(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        emulator_state_buffer = io.BytesIO()
        self.pyboy.save_state(emulator_state_buffer)
        emulator_state_bytes = emulator_state_buffer.getvalue()

        gym_state = {
            "steps": self.steps,
            "episode": self.episode,
            "button": self.button,
            "_fitness": self._fitness,
            "done": self.done,
            "render": self.render,
            "reward_calculator": self.reward_calculator,
        }

        with open(save_path, "wb") as f:
            pickle.dump(
                {"emulator_state": emulator_state_bytes, "gym_state": gym_state}, f
            )

    def load_gym_state(self, load_path, updated_steps=None, updated_n_goals=None):
        try:
            with open(load_path, "rb") as f:
                combined_state = pickle.load(f)
        except FileNotFoundError:
            print("Could not find file at path:", load_path)
            print("Returning to initial state.")
            return self.reset()

        emulator_state_bytes = combined_state["emulator_state"]
        state_bytes_io = io.BytesIO(emulator_state_bytes)
        self.pyboy.load_state(state_bytes_io)
        self.state_bytes_content = emulator_state_bytes

        gym_state = combined_state["gym_state"]
        self.steps = gym_state["steps"]
        self.episode = gym_state["episode"]
        self.button = gym_state["button"]
        self._fitness = gym_state["_fitness"]
        self.done = gym_state["done"]
        self.render = gym_state["render"]
        self.reward_calculator = gym_state["reward_calculator"]

        if updated_steps:
            # New design has no goal-count target; just refresh the step
            # budget if the caller wanted a different one. ``updated_n_goals``
            # is accepted for callsite compatibility but ignored.
            self.reward_calculator.max_steps = int(updated_steps)
            self.done = False

        return self.get_observation()
