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

        self.wram_start = 0xC000
        self.wram_end = 0xDFFF
        self.wram_size = self.wram_end - self.wram_start + 1


    def get_memory_value(self, address):
        return self.pyboy.memory[address]
    
    def set_memory_value(self, address, value):
        self.pyboy.memory[address] = value

    def get_current_location(self):
        loc = self.get_memory_value(self.room_player_is_in)
        if loc == 7:  # starting zone is also 7 if you leave and come back
            loc = 0
        return loc

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
        raw_bytes = []
        for i in range(end, start - 1, -1):
            byte = self.get_memory_value(i)
            raw_bytes.append(byte)
        return raw_bytes

    def get_party_info(self):
        num_pokemon = self.get_memory_value(self.num_pokemon_in_party)
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
        return int(total_level), int(total_hp), int(total_exp)

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
        return {
            "money": self.get_player_money(),
            "location": self.get_current_location(),
            "X": x,
            "Y": y,
            "party_info": self.get_party_info(),
            "pokedex_seen": self.get_pokedex_seen(),
            "pokedex_owned": self.get_pokedex_owned(),
            "map_num": self.get_map_num(),
            "warp_number": self.get_memory_value(self.warp_number),
            "map_bank": self.get_memory_value(self.map_bank),
        }
