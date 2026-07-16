# -*- coding: utf-8 -*-
import numpy as np


class RAMManagement:
    def __init__(self, pyboy):
        self.pyboy = pyboy

        # Memory locations
        self.room_player_is_in = 0xD148
        self.map_number = 0xDCB6
        self.overworld_X = 0xDCB8
        self.overworld_Y = 0xDCB7
        self.received = 0xCF60
        self.player_money = 0xD84E
        self.party_base = 0xDCDF
        self.num_pokemon_in_party = 0xDCD7
        self.pokedex_seen = (0xDEB9, 0xDED8)
        self.pokedex_owned = (0xDE99, 0xDEB8)
        self.screen_tile_start = 0xC4A0
        self.screen_tile_end = 0xC607

        self.warp_number = 0xDCB4
        self.map_bank = 0xDCB5

        # Story / event flags region. 256 bytes × 8 bits = 2048 individual
        # story flags. The 8-byte GameShark write quirk is unrelated to our
        # read-only use — we just need to surface these bits to the policy
        # so it can condition on game progress.
        self.story_flags_start = 0xDA72
        self.story_flags_end = 0xDB71
        self.story_flags_size = self.story_flags_end - self.story_flags_start + 1  # 256

        self.wram_start = 0xC000
        self.wram_end = 0xDFFF
        self.wram_size = self.wram_end - self.wram_start + 1

        # collision data
        self.collision_down = 0xC2FA
        self.collision_up = 0xC2FB
        self.collision_left = 0xC2FC
        self.collision_right = 0xC2FD

        # Player facing direction: 1=up, 2=down, 3=left, 4=right.
        # Used for navigation — the policy needs to know which way it's
        # facing at a given (x, y) coordinate.
        self.player_direction = 0xD357

        # Priority 1 raw features
        self.battle_type = 0xD22D
        self.johto_badges = 0xD857
        self.player_state = 0xD95D
        self.key_items_count = 0xD8BC
        self.game_hour = 0xD4B7
        self.bgm_id = 0xC2A9

        # Enemy mon stats during a battle. wEnemyMon starts at 0xD206 with
        # the battle_struct layout (species, item, moves(4), DVs(2), PP(4),
        # happiness, level, status(2), HP(2), MaxHP(2), ...). HP is stored
        # little-endian (low byte at the lower address). Outside of battles
        # the values are stale — guard any consumer with battle_type != 0.
        self.enemy_hp_high = 0xD216
        self.enemy_hp_low = 0xD217
        self.enemy_max_hp_high = 0xD218
        self.enemy_max_hp_low = 0xD219

        # Player's in-battle Pokemon move PP (C634-C637, one byte per move).
        # These are the current PP values for the active battle Pokemon's four
        # moves. Read during battles; stale outside battle context.
        self.player_pp1 = 0xC634
        self.player_pp2 = 0xC635
        self.player_pp3 = 0xC636
        self.player_pp4 = 0xC637

        # --------------------------------------------------------------- #
        # Extended probes — addresses documented in RAM_MAPPING.md but not
        # currently surfaced in get_variables(). Read-only. Added for the
        # debug evaluator (model=debug_eval) so we can see which RAM
        # regions change in response to a scripted action sequence (e.g.
        # noop×3 → START → noop×3 → B for menu-state surfacing). All hex
        # values come straight from RAM_MAPPING.md.
        # --------------------------------------------------------------- #
        # Game progress & badges
        self.kanto_badges_addr = 0xD858
        self.unowndex_status_addr = 0xD84C
        self.bug_catching_contest_addr = 0xD84D
        # Time & play clock
        self.day_of_week_addr = 0xD4B6
        self.game_minute_addr = 0xD4B8
        self.play_time_hours_lo_addr = 0xD4C4
        self.play_time_hours_hi_addr = 0xD4C5
        # Battle context (in addition to battle_type at D22D)
        self.wild_battle_type_addr = 0xD230
        self.enemy_trainer_type_addr = 0xD233
        # Inventory signals
        self.item_pocket_count_addr = 0xD892
        self.pokeball_count_addr = 0xD8D7
        self.coins_lo_addr = 0xD855
        self.coins_hi_addr = 0xD856
        self.repel_steps_addr = 0xDCA1
        self.blue_card_points_addr = 0xDC4B
        # Legendary / unique encounter capture markers (Sarial)
        self.hooh_captured_addr = 0xDAD4
        self.lugia_captured_addr = 0xDAD5
        self.sudowoodo_captured_addr = 0xDB51
        self.red_gyarados_captured_addr = 0xDB5C
        self.snorlax_captured_addr = 0xDB60
        # Roaming legendary positions (Randil)
        self.raikou_map_bank_addr = 0xDFD1
        self.raikou_map_num_addr = 0xDFD2
        self.entei_map_bank_addr = 0xDFD8
        self.entei_map_num_addr = 0xDFD9

        # Speculative byte windows for debug-mode dumps. These regions are
        # known to contain text-engine / menu / script state in pokecrystal
        # but the exact symbol↔address mapping is not pinned in our
        # RAM_MAPPING.md. The debug evaluator dumps each window so the user
        # can diff bytes across a menu-open scripted sequence and identify
        # which offsets carry the signal. Treat as "candidate regions to
        # investigate", not stable contracts. Keep them small to keep the
        # JSON sidecar files readable.
        self.debug_byte_windows = {
            # Audio engine — BGM ID at 0xC2A9 lives here; nearby bytes hold
            # channel state and flip when menu / battle sound plays.
            "audio_c2a0": (0xC2A0, 0xC2BF),
            # Joypad mirror + input filter state. Bytes here change on
            # every button press / release and on menu joypad-mask flips.
            "joypad_c2cd": (0xC2CD, 0xC2D5),
            # Battle struct prefix — species, item, moves. Already partly
            # exposed via enemy_hp, but the whole prefix reveals when a
            # battle init is mid-write vs. settled.
            "battle_d200": (0xD200, 0xD220),
            # Player object struct (sprite slot 0). Direction, walking
            # frame, step counter — flips on movement and on faced-NPC
            # interactions.
            "player_obj_d4d0": (0xD4D0, 0xD4F0),
            # Map handler region. wMapStatus is somewhere here in
            # pokecrystal; nearby bytes hold map-load state machine vars.
            "map_d145": (0xD140, 0xD160),
            # Map event / connection bytes — change on map transitions.
            "map_event_dc40": (0xDC40, 0xDC50),
            # Text engine — wTextBoxFrame, cursor state, joypad filter.
            # Strongly signalled on menu open/close in earlier probe.
            "text_cf00": (0xCF00, 0xCF20),
            # Script execution state — wScriptFlags and wScriptBank/Pos
            # cluster around 0xD43x in pokecrystal. Empirically the most
            # informative window for "is a script running" detection.
            "script_d430": (0xD430, 0xD460),
        }

    def get_memory_value(self, address):
        return self.pyboy.memory[address]

    def set_memory_value(self, address, value):
        self.pyboy.memory[address] = value

    def get_XY(self):
        x_coord = self.get_memory_value(self.overworld_X)
        y_coord = self.get_memory_value(self.overworld_Y)
        return x_coord, y_coord

    def get_player_money(self):
        money_bytes = [self.get_memory_value(self.player_money + i) for i in range(3)]
        money = self.bytes_to_int(money_bytes[::-1])
        return money

    def bytes_to_int(self, byte_list):
        return int.from_bytes(byte_list, byteorder="little")

    def read_little_endian(self, start, end):
        """Read bytes from ``start`` to ``end`` (inclusive) in address order.

        The caller multiplies by [1, 256, 65536, ...] to assemble the
        little-endian integer (low byte at lowest address).
        """
        raw_bytes = []
        for i in range(start, end + 1):
            byte = self.get_memory_value(i)
            raw_bytes.append(byte)
        return raw_bytes

    def get_party_info(self):
        # Clamp num_pokemon to the real max party size (6). Without this,
        # a transitional / mid-write read at battle init can return a
        # garbage byte (e.g. 122), making the loop walk far past the
        # party data into other parts of WRAM. Each junk "Pokemon" slot
        # contributes a 24-bit exp read up to 16M; the resulting Δexp
        # then drives party_exp_reward through the reward clip in one
        # step. See AGENTS.md "Garbage-RAM guards."
        num_pokemon = min(self.get_memory_value(self.num_pokemon_in_party), 6)
        total_level = 0
        total_hp = 0
        total_exp = 0
        for i in range(num_pokemon):
            base_address = self.party_base + 0x30 * i
            level = self.get_memory_value(base_address + 0x1F)
            hp = np.sum(
                self.read_little_endian(base_address + 0x22, base_address + 0x23)
                * np.array([1, 256])
            )
            exp = np.sum(
                self.read_little_endian(base_address + 0x08, base_address + 0x0A)
                * np.array([1, 256, 65536])
            )
            total_level += level
            total_hp += hp
            total_exp += exp
        return int(num_pokemon), int(total_level), int(total_hp), int(total_exp)

    def get_pokedex_seen(self):
        start_address, end_address = self.pokedex_seen
        total_seen = 0
        for address in range(start_address, end_address + 1):
            byte_value = self.get_memory_value(address)
            while byte_value:
                total_seen += byte_value & 1
                byte_value >>= 1
        return total_seen

    def get_pokedex_owned(self):
        start_address, end_address = self.pokedex_owned
        total_owned = 0
        for address in range(start_address, end_address + 1):
            byte_value = self.get_memory_value(address)
            while byte_value:
                total_owned += byte_value & 1
                byte_value >>= 1
        return total_owned

    def get_map_num(self):
        return self.get_memory_value(self.map_number)

    def get_battle_type(self):
        return self.get_memory_value(self.battle_type)

    def get_johto_badges(self):
        return self.get_memory_value(self.johto_badges)

    def get_player_direction(self):
        return int(self.get_memory_value(self.player_direction))

    def get_player_state(self):
        return self.get_memory_value(self.player_state)

    def get_key_items_count(self):
        return self.get_memory_value(self.key_items_count)

    def get_game_hour(self):
        return self.get_memory_value(self.game_hour)

    def get_bgm_id(self):
        return self.get_memory_value(self.bgm_id)

    def get_enemy_hp(self):
        # pokecrystal stores the enemy mon's HP BIG-endian in wEnemyMon:
        # high byte at the lower address (0xD216), low byte at 0xD217.
        # Verified empirically against Manual Investigation States/
        # in_wild_battle_route_13.state — a level-2 Sentret with 14 HP reads
        # 0xD216=0, 0xD217=14, and 0xD217 counts down 14->8->3->0 as it takes
        # damage. The previous read (treating 0xD216 as the low byte) returned
        # 256x the true value for any enemy with HP < 256 (i.e. all early-game
        # encounters), and went non-monotonic once the high byte was in use.
        high = self.get_memory_value(self.enemy_hp_high)  # 0xD216 — high byte
        low = self.get_memory_value(self.enemy_hp_low)    # 0xD217 — low byte
        return (high << 8) | low

    def get_enemy_max_hp(self):
        # Same big-endian layout as get_enemy_hp (0xD218 high, 0xD219 low).
        high = self.get_memory_value(self.enemy_max_hp_high)  # 0xD218 — high byte
        low = self.get_memory_value(self.enemy_max_hp_low)    # 0xD219 — low byte
        return (high << 8) | low

    def get_player_move_pp(self):
        """Return current PP for the player's in-battle Pokemon's four moves.

        Returns
        -------
        tuple of 4 ints, each in [0, max_pp_for_move]. Normalised downstream
        by dividing by 64 (typical max PP for a move) so the feature stays
        in [0, 1]. Outside battle the values are stale — guard with
        battle_type != 0.
        """
        return (
            self.get_memory_value(self.player_pp1),
            self.get_memory_value(self.player_pp2),
            self.get_memory_value(self.player_pp3),
            self.get_memory_value(self.player_pp4),
        )

    def get_story_flags(self):
        """Return the 256-byte story-flag region as a uint8 ndarray.

        These bytes are bitfields — each byte holds 8 individual flags —
        but we expose them as raw bytes for the policy to learn the
        relevant bit patterns from. Read-only; we never write to this region.
        """
        out = np.zeros(self.story_flags_size, dtype=np.uint8)
        for i in range(self.story_flags_size):
            out[i] = self.get_memory_value(self.story_flags_start + i)
        return out

    def export_wram(self):
        """
        Export the entire Work RAM (WRAM) as a numpy array.
        This includes both WRAM Bank 0 (C000-CFFF) and WRAM Bank 1 (D000-DFFF).
        """
        wram_data = np.zeros(self.wram_size, dtype=np.uint8)

        for i in range(self.wram_size):
            wram_data[i] = self.get_memory_value(self.wram_start + i)

        return wram_data

    def get_screen_tiles(self):
        # This is basically the static background ...

        # The screen is 20 tiles wide and 18 tiles high
        screen_width = 20
        screen_height = 18

        # Create a 2D numpy array to store the tile values
        screen_tiles = np.zeros((screen_height, screen_width), dtype=np.uint8)

        # Read the tile values from memory and populate the array
        for i in range(screen_height):
            for j in range(screen_width):
                mem_address = self.screen_tile_start + i * screen_width + j
                screen_tiles[i, j] = self.get_memory_value(mem_address)

        return screen_tiles

    def get_variables(self):
        x, y = self.get_XY()
        story_flags = self.get_story_flags()
        # Verified script / UI state bytes (see RAM_MAPPING.md). Surfaced
        # here as raw ints so consumers (rewards.py for novelty gating,
        # gym_env.py for the one-hot observation features) don't have to
        # re-read pyboy memory.
        script_byte = int(self.get_memory_value(0xD438))
        ui_byte = int(self.get_memory_value(0xCF07))
        map_handler_byte = int(self.get_memory_value(0xD43D))
        return {
            "money": self.get_player_money(),
            "X": x,
            "Y": y,
            "party_info": self.get_party_info(),
            "pokedex_seen": self.get_pokedex_seen(),
            "pokedex_owned": self.get_pokedex_owned(),
            "map_num": self.get_map_num(),
            "warp_number": self.get_memory_value(self.warp_number),
            "map_bank": self.get_memory_value(self.map_bank),
            "room": self.get_memory_value(self.room_player_is_in),
            "collision_down": self.get_memory_value(self.collision_down),
            "collision_up": self.get_memory_value(self.collision_up),
            "collision_left": self.get_memory_value(self.collision_left),
            "collision_right": self.get_memory_value(self.collision_right),
            "story_flags": story_flags,
            "battle_type": self.get_battle_type(),
            "player_direction": self.get_player_direction(),
            "johto_badges": self.get_johto_badges(),
            "player_state": self.get_player_state(),
            "key_items_count": self.get_key_items_count(),
            "game_hour": self.get_game_hour(),
            "bgm_id": self.get_bgm_id(),
            "enemy_hp": self.get_enemy_hp(),
            "enemy_max_hp": self.get_enemy_max_hp(),
            "player_move_pp": self.get_player_move_pp(),
            "script_byte": script_byte,
            "ui_byte": ui_byte,
            "map_handler_byte": map_handler_byte,
            "script_active": script_byte == 255,
        }

    # ------------------------------------------------------------------ #
    # Debug-only probes. NOT part of get_variables() — the live training
    # observation deliberately does not include these so the model input
    # contract (RAM_OBS_DIM) is stable. Consumed by the debug evaluator.
    # ------------------------------------------------------------------ #

    def _read_u8(self, addr):
        return int(self.get_memory_value(addr))

    def _read_u16_le(self, lo_addr, hi_addr):
        return self._read_u8(lo_addr) | (self._read_u8(hi_addr) << 8)

    def _popcount(self, byte_value):
        n = 0
        while byte_value:
            n += byte_value & 1
            byte_value >>= 1
        return n

    def get_extended_variables(self):
        """Return every RAM probe currently documented but not part of
        get_variables(). Used only by the debug evaluator. Values are
        intentionally raw integers (no scaling) — the consumer formats them
        for filename / JSON output. Stable key names so downstream parsers
        of the debug PNGs / .json sidecars can rely on the schema.
        """
        kanto_badges_raw = self._read_u8(self.kanto_badges_addr)
        johto_badges_raw = self._read_u8(self.johto_badges)
        return {
            # Progress
            "johto_badges_raw": johto_badges_raw,
            "johto_badges_count": self._popcount(johto_badges_raw),
            "kanto_badges_raw": kanto_badges_raw,
            "kanto_badges_count": self._popcount(kanto_badges_raw),
            "unowndex_status": self._read_u8(self.unowndex_status_addr),
            "bug_catching_contest": self._read_u8(self.bug_catching_contest_addr),
            # Time / clock
            "day_of_week": self._read_u8(self.day_of_week_addr),
            "game_minute": self._read_u8(self.game_minute_addr),
            "play_time_hours": self._read_u16_le(
                self.play_time_hours_lo_addr, self.play_time_hours_hi_addr
            ),
            # Battle context
            "wild_battle_type": self._read_u8(self.wild_battle_type_addr),
            "enemy_trainer_type": self._read_u8(self.enemy_trainer_type_addr),
            # Inventory
            "item_pocket_count": self._read_u8(self.item_pocket_count_addr),
            "pokeball_count": self._read_u8(self.pokeball_count_addr),
            "coins": self._read_u16_le(self.coins_lo_addr, self.coins_hi_addr),
            "repel_steps": self._read_u8(self.repel_steps_addr),
            "blue_card_points": self._read_u8(self.blue_card_points_addr),
            # Legendary / unique capture markers
            "hooh_captured": self._read_u8(self.hooh_captured_addr),
            "lugia_captured": self._read_u8(self.lugia_captured_addr),
            "sudowoodo_captured": self._read_u8(self.sudowoodo_captured_addr),
            "red_gyarados_captured": self._read_u8(self.red_gyarados_captured_addr),
            "snorlax_captured": self._read_u8(self.snorlax_captured_addr),
            # Roaming Pokémon
            "raikou_map_bank": self._read_u8(self.raikou_map_bank_addr),
            "raikou_map_num": self._read_u8(self.raikou_map_num_addr),
            "entei_map_bank": self._read_u8(self.entei_map_bank_addr),
            "entei_map_num": self._read_u8(self.entei_map_num_addr),
        }

    def get_debug_byte_windows(self):
        """Return raw bytes for each speculative debug byte window as a
        dict of {window_name: list[int]}. Cheap to call (≤ ~80 bytes total
        across the four windows) and consumed by the JSON sidecar so the
        user can spot which bytes flip across a scripted menu sequence
        without having to know the exact pokecrystal address up front.
        """
        out = {}
        for name, (start, end) in self.debug_byte_windows.items():
            out[name] = [self._read_u8(a) for a in range(start, end + 1)]
        return out
