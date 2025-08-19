# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
import hashlib


class ExplorationMemory:
    def __init__(self, max_size=100, history_length=5, use_memory=True):
        """
        Initialize exploration memory bank to track visited screens

        Args:
            max_size: Maximum number of recent screen hashes to store
            history_length: Number of past screen hashes to store for each hash
            use_memory: Whether to use the memory functionality (can be toggled off)
        """
        self.max_size = max_size
        self.history_length = history_length
        self.use_memory = use_memory
        self.memory = []  # List of (hash, history, visits) tuples
        self.hash_visits = defaultdict(int)  # Track visit counts for each hash
        self.recent_hashes = []  # Store recent hashes for history tracking

    def add_screen(self, screen_array):
        """
        Add a screen to the memory bank

        Args:
            screen_array: The screen array to hash and store
        """
        if not self.use_memory:
            return

        # Compute hash of the screen array
        screen_hash = self._compute_hash(screen_array)

        # Update recent hashes list
        self.recent_hashes.append(screen_hash)
        if len(self.recent_hashes) > self.history_length:
            self.recent_hashes.pop(0)

        # Get current history (last N hashes excluding current one)
        history = tuple(self.recent_hashes[:-1])

        # Increment visit count
        self.hash_visits[screen_hash] += 1
        visits = self.hash_visits[screen_hash]

        # Check if this hash is already in memory
        for i, (mem_hash, mem_history, _) in enumerate(self.memory):
            if mem_hash == screen_hash:
                # Update existing entry with new visit count and history
                self.memory[i] = (screen_hash, history, visits)
                # Move to end of list (most recent)
                self.memory.append(self.memory.pop(i))
                return

        # Add new hash to memory
        self.memory.append((screen_hash, history, visits))

        # Remove oldest entry if exceeding max size
        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def _compute_hash(self, screen_array):
        """
        Compute a hash of the screen array

        Args:
            screen_array: The screen array to hash

        Returns:
            A hash string representing the screen
        """
        # Convert array to bytes and hash it
        array_bytes = screen_array.tobytes()
        return hashlib.md5(array_bytes).hexdigest()

    def get_memory_tensor(self):
        """
        Convert memory to a tensor for model input

        Returns:
            numpy array containing hash visit counts and history information,
            padded with zeros if fewer than max_size entries
        """
        if not self.use_memory or not self.memory:
            # Return empty tensor if memory is disabled or empty
            return np.zeros((self.max_size, 1 + self.history_length), dtype=np.float32)

        # Create a zero-filled array of shape (max_size, 1 + history_length)
        # First column is visit count, remaining columns are binary indicators of history
        tensor = np.zeros((self.max_size, 1 + self.history_length), dtype=np.float32)

        # Fill with actual memory data
        for i, (_, history, visits) in enumerate(self.memory):
            if i >= self.max_size:
                break

            # Set visit count
            tensor[i, 0] = visits

            # Set history indicators (1 if hash was in history, 0 otherwise)
            for j, past_hash in enumerate(history):
                if j < self.history_length:
                    tensor[i, j + 1] = 1.0

        return tensor

    def reset(self):
        """Reset the memory bank"""
        self.memory = []
        self.hash_visits = defaultdict(int)
        self.recent_hashes = []
