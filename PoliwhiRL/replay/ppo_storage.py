# -*- coding: utf-8 -*-
import numpy as np
import torch
import sqlite3
import os
import json
import zlib


class PPOMemory:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.update_frequency = config["ppo_update_frequency"]
        self.sequence_length = config["sequence_length"]
        self.input_shape = config["input_shape"]
        self.db_path = config["db_path"]
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.memory_id = 0
        self.episode_length = 0
        self.reset()

    def reset(self, config=None):
        if self.episode_length > 0:
            self.save_to_database()
        if config is not None:
            self.__init__(config)
        self.states = np.zeros(
            (self.update_frequency,) + self.input_shape, dtype=np.uint8
        )
        self.actions = np.zeros(self.update_frequency, dtype=np.uint8)
        self.rewards = np.zeros(self.update_frequency, dtype=np.int8)
        self.dones = np.zeros(self.update_frequency, dtype=np.bool_)
        self.log_probs = np.zeros(self.update_frequency, dtype=np.float16)
        self.last_next_state = None
        self.episode_length = 0

    def store_transition(self, state, next_state, action, reward, done, log_prob):
        idx = self.episode_length
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        self.last_next_state = next_state
        self.episode_length += 1

    def get_all_data(self):
        if self.episode_length < self.sequence_length:
            return None
        num_sequences = self.episode_length - self.sequence_length + 1
        sequences = np.array(
            [self.states[i : i + self.sequence_length] for i in range(num_sequences)]
        )
        next_sequences = np.array(
            [
                self.states[i + 1 : i + self.sequence_length + 1]
                for i in range(num_sequences - 1)
            ]
            + [
                np.concatenate(
                    [self.states[-self.sequence_length + 1 :], [self.last_next_state]]
                )
            ]
        )
        return {
            "states": torch.FloatTensor(sequences).to(self.device),
            "next_states": torch.FloatTensor(next_sequences).to(self.device),
            "actions": torch.LongTensor(
                self.actions[self.sequence_length - 1 : self.episode_length]
            ).to(self.device),
            "rewards": torch.FloatTensor(
                self.rewards[self.sequence_length - 1 : self.episode_length]
            ).to(self.device),
            "dones": torch.BoolTensor(
                self.dones[self.sequence_length - 1 : self.episode_length]
            ).to(self.device),
            "old_log_probs": torch.FloatTensor(
                self.log_probs[self.sequence_length - 1 : self.episode_length]
            ).to(self.device),
        }

    def __len__(self):
        return self.episode_length

    @staticmethod
    def compress_data(data):
        return zlib.compress(data.tobytes())

    @staticmethod
    def decompress_data(compressed_data, dtype, shape):
        return np.frombuffer(zlib.decompress(compressed_data), dtype=dtype).reshape(
            shape
        )

    def save_to_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """CREATE TABLE IF NOT EXISTS memory
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           states BLOB,
                           actions BLOB,
                           rewards BLOB,
                           dones BLOB,
                           log_probs BLOB,
                           last_next_state BLOB,
                           episode_length INTEGER,
                           input_shape TEXT,
                           sequence_length INTEGER)"""
        )

        states_binary = self.compress_data(self.states[: self.episode_length])
        actions_binary = self.compress_data(self.actions[: self.episode_length])
        rewards_binary = self.compress_data(self.rewards[: self.episode_length])
        dones_binary = self.compress_data(self.dones[: self.episode_length])
        log_probs_binary = self.compress_data(self.log_probs[: self.episode_length])
        last_next_state_binary = (
            self.compress_data(self.last_next_state)
            if self.last_next_state is not None
            else None
        )

        cursor.execute(
            """INSERT INTO memory
                          (states, actions, rewards, dones, log_probs, last_next_state, episode_length, input_shape, sequence_length)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                states_binary,
                actions_binary,
                rewards_binary,
                dones_binary,
                log_probs_binary,
                last_next_state_binary,
                self.episode_length,
                json.dumps(self.input_shape),
                self.sequence_length,
            ),
        )

        conn.commit()
        self.memory_id = cursor.lastrowid
        conn.close()

    @staticmethod
    def load_from_database(config, memory_id):
        db_path = config["db_path"]
        device = torch.device(config["device"])
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM memory WHERE id = ?", (memory_id,))
        row = cursor.fetchone()

        if row is None:
            conn.close()
            return None

        input_shape = tuple(json.loads(row[8]))
        sequence_length = row[9]
        episode_length = row[7]

        states = PPOMemory.decompress_data(
            row[1], np.uint8, (episode_length,) + input_shape
        ).copy()
        actions = PPOMemory.decompress_data(row[2], np.uint8, (episode_length,)).copy()
        rewards = PPOMemory.decompress_data(row[3], np.int8, (episode_length,)).copy()
        dones = PPOMemory.decompress_data(row[4], np.bool_, (episode_length,)).copy()
        log_probs = PPOMemory.decompress_data(
            row[5], np.float16, (episode_length,)
        ).copy()
        last_next_state = (
            PPOMemory.decompress_data(row[6], np.uint8, input_shape).copy()
            if row[6] is not None
            else None
        )

        conn.close()

        if episode_length < sequence_length:
            return None

        num_sequences = episode_length - sequence_length + 1
        sequences = np.array(
            [states[i : i + sequence_length] for i in range(num_sequences)]
        )
        next_sequences = np.array(
            [states[i + 1 : i + sequence_length + 1] for i in range(num_sequences - 1)]
            + [np.concatenate([states[-sequence_length + 1 :], [last_next_state]])]
        )

        return {
            "states": torch.FloatTensor(sequences).to(device),
            "next_states": torch.FloatTensor(next_sequences).to(device),
            "actions": torch.LongTensor(
                actions[sequence_length - 1 : episode_length]
            ).to(device),
            "rewards": torch.FloatTensor(
                rewards[sequence_length - 1 : episode_length]
            ).to(device),
            "dones": torch.BoolTensor(dones[sequence_length - 1 : episode_length]).to(
                device
            ),
            "old_log_probs": torch.FloatTensor(
                log_probs[sequence_length - 1 : episode_length]
            ).to(device),
        }

    @staticmethod
    def get_memory_ids(config):
        try:
            db_path = config["db_path"]
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM memory")
            ids = [row[0] for row in cursor.fetchall()]

            conn.close()
            return ids
        except sqlite3.OperationalError:
            return []
