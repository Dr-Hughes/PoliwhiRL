# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict


class ExplorationMemory:
    def __init__(self, max_size=100):
        """
        Initialize exploration memory bank to track visited locations

        Args:
            max_size: Maximum number of recent locations to store
        """
        self.max_size = max_size
        self.memory = []  # List of (x, y, map_num, room, visits) tuples
        self.location_visits = defaultdict(int)  # Track visit counts for each location

    def add_location(self, x, y, map_num, room):
        """
        Add a location to the memory bank

        Args:
            x: X coordinate
            y: Y coordinate
            map_num: Map number
            room: Room number
        """
        # Create a unique key for this location
        location_key = (x, y, map_num, room)

        # Increment visit count
        self.location_visits[location_key] += 1
        visits = self.location_visits[location_key]

        # Check if this location is already in memory
        for i, (mem_x, mem_y, mem_map, mem_room, _) in enumerate(self.memory):
            if (mem_x, mem_y, mem_map, mem_room) == location_key:
                # Update existing entry with new visit count
                self.memory[i] = (x, y, map_num, room, visits)
                # Move to end of list (most recent)
                self.memory.append(self.memory.pop(i))
                return

        # Add new location to memory
        self.memory.append((x, y, map_num, room, visits))

        # Remove oldest entry if exceeding max size
        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def get_memory_tensor(self):
        """
        Convert memory to a tensor for model input

        Returns:
            numpy array of shape (max_size, 5) containing (x, y, map_num, room, visits)
            for each location, padded with zeros if fewer than max_size entries
        """
        # Create a zero-filled array of shape (max_size, 5)
        tensor = np.zeros((self.max_size, 5), dtype=np.float32)

        # Fill with actual memory data
        for i, (x, y, map_num, room, visits) in enumerate(self.memory):
            if i >= self.max_size:
                break
            tensor[i] = [x, y, map_num, room, visits]

        return tensor

    def reset(self):
        """Reset the memory bank"""
        self.memory = []
        self.location_visits = defaultdict(int)
