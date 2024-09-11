# -*- coding: utf-8 -*-
import os
import sqlite3
import numpy as np
import torch
import pickle


class SequenceStorage:
    def __init__(
        self,
        db_path,
        capacity,
        sequence_length,
        device,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001,
        remove_old_sequences=True,
    ):
        self.db_path = db_path
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.episode_buffer = []
        self.conn = None
        self.cursor = None
        if remove_old_sequences and os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.connect()
        self.setup_database()
        self.device = device

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

    def setup_database(self):
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS sequences
        (sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
         priority REAL,
         data BLOB)
        """
        )
        self.conn.commit()

    def add(self, state, action, reward, next_state, done):
        self.episode_buffer.append((state, action, reward, next_state, done))

        if done or len(self.episode_buffer) >= self.sequence_length:
            sequences_to_add = []
            while len(self.episode_buffer) >= self.sequence_length:
                sequence = self.episode_buffer[: self.sequence_length]
                sequences_to_add.append(sequence)
                self.episode_buffer = self.episode_buffer[1:]

            self.add_sequences(sequences_to_add)

            if done:
                self.episode_buffer = []

    def add_sequences(self, sequences):
        if not self.conn:
            self.connect()

        try:
            self.conn.execute("BEGIN TRANSACTION")

            # Get max priority
            self.cursor.execute("SELECT MAX(priority) FROM sequences")
            max_priority = self.cursor.fetchone()[0]
            priority = max_priority if max_priority is not None else 1.0

            # Prepare batch insert
            insert_data = [
                (priority, sqlite3.Binary(pickle.dumps(seq))) for seq in sequences
            ]
            self.cursor.executemany(
                "INSERT INTO sequences (priority, data) VALUES (?, ?)", insert_data
            )

            # Remove old sequences if capacity is exceeded
            self.cursor.execute("SELECT COUNT(*) FROM sequences")
            count = self.cursor.fetchone()[0]
            if count > self.capacity:
                self.cursor.execute(
                    "DELETE FROM sequences WHERE sequence_id IN (SELECT sequence_id FROM sequences ORDER BY priority LIMIT ?)",
                    (count - self.capacity,),
                )

            self.conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while adding sequences: {e}")
            self.conn.rollback()
        finally:
            self.close()

    def sample(self, batch_size):
        if not self.conn:
            self.connect()

        try:
            self.conn.execute("BEGIN TRANSACTION")

            self.cursor.execute("SELECT SUM(priority) FROM sequences")
            total_priority = self.cursor.fetchone()[0]
            if total_priority is None:
                return None

            self.cursor.execute("SELECT sequence_id, priority, data FROM sequences")
            sequences = self.cursor.fetchall()
            probabilities = np.array([seq[1] for seq in sequences]) / total_priority
            indices = np.random.choice(
                len(sequences), batch_size, p=probabilities, replace=False
            )

            batch = [pickle.loads(sequences[i][2]) for i in indices]
            states, actions, rewards, next_states, dones = zip(
                *[zip(*seq) for seq in batch]
            )

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions))
            rewards = torch.FloatTensor(np.array(rewards))
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.BoolTensor(np.array(dones))

            weights = (len(sequences) * probabilities[indices]) ** -self.beta
            weights /= weights.max()
            weights = torch.FloatTensor(weights)

            self.beta = min(1.0, self.beta + self.beta_increment)

            sequence_ids = [sequences[i][0] for i in indices]

            self.conn.commit()
            return (
                states.to(self.device),
                actions.to(self.device),
                rewards.to(self.device),
                next_states.to(self.device),
                dones.to(self.device),
                sequence_ids,
                weights.to(self.device),
            )
        except sqlite3.Error as e:
            print(f"An error occurred while sampling: {e}")
            self.conn.rollback()
            return None
        finally:
            self.close()

    def update_priorities(self, sequence_ids, errors):
        if not self.conn:
            self.connect()

        try:
            self.conn.execute("BEGIN TRANSACTION")

            for sequence_id, error in zip(sequence_ids, errors):
                priority = (error + 1e-5) ** self.alpha
                self.cursor.execute(
                    "UPDATE sequences SET priority = ? WHERE sequence_id = ?",
                    (priority, sequence_id),
                )

            self.conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while updating priorities: {e}")
            self.conn.rollback()
        finally:
            self.close()

    def __len__(self):
        if not self.conn:
            self.connect()
        try:
            self.cursor.execute("SELECT COUNT(*) FROM sequences")
            return self.cursor.fetchone()[0]
        finally:
            self.close()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def get_max_priority_sequences(self):
        if not self.conn:
            self.connect()

        try:
            self.conn.execute("BEGIN TRANSACTION")

            # Get the maximum priority
            self.cursor.execute("SELECT MAX(priority) FROM sequences")
            max_priority = self.cursor.fetchone()[0]

            if max_priority is None:
                return None

            # Select all sequences with the maximum priority
            self.cursor.execute(
                "SELECT sequence_id, data FROM sequences WHERE priority = ?",
                (max_priority,),
            )
            sequences = self.cursor.fetchall()

            if not sequences:
                return None

            batch = [pickle.loads(seq[1]) for seq in sequences]
            states, actions, rewards, next_states, dones = zip(
                *[zip(*seq) for seq in batch]
            )

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions))
            rewards = torch.FloatTensor(np.array(rewards))
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.BoolTensor(np.array(dones))

            sequence_ids = [seq[0] for seq in sequences]

            self.conn.commit()
            return (
                states.to(self.device),
                actions.to(self.device),
                rewards.to(self.device),
                next_states.to(self.device),
                dones.to(self.device),
                sequence_ids,
            )
        except sqlite3.Error as e:
            print(f"An error occurred while getting max priority sequences: {e}")
            self.conn.rollback()
            return None
        finally:
            self.close()
