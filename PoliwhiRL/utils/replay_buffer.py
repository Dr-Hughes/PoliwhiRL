# -*- coding: utf-8 -*-
import sqlite3
import numpy as np
import torch
from collections import defaultdict
import pickle


class PrioritizedReplayBuffer:
    def __init__(
        self, db_path, capacity=1000000, alpha=0.6, beta=0.4, beta_increment=0.001
    ):
        self.db_path = db_path
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.conn = None
        self.cursor = None
        self.connect()
        self.setup_database()
        self.episode_buffer = []

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.conn.execute("PRAGMA journal_mode=WAL")  # Use Write-Ahead Logging
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Reduce synchronous writes

    def setup_database(self):
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS episodes
        (episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
         priority REAL,
         length INTEGER)
        """
        )
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS experiences
        (experience_id INTEGER PRIMARY KEY AUTOINCREMENT,
         episode_id INTEGER,
         state BLOB,
         action INTEGER,
         reward REAL,
         next_state BLOB,
         done INTEGER,
         FOREIGN KEY(episode_id) REFERENCES episodes(episode_id))
        """
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_episode_id ON experiences(episode_id)"
        )
        self.conn.commit()

    def add(self, state, action, reward, next_state, done):
        self.episode_buffer.append((state, action, reward, next_state, done))

        if done:
            self.add_episode(self.episode_buffer)
            self.episode_buffer = []

    def add_episode(self, episode):
        if not self.conn:
            self.connect()

        try:
            self.conn.execute("BEGIN TRANSACTION")

            priority = 1.0  # Start with max priority for new episodes
            self.cursor.execute(
                "INSERT INTO episodes (priority, length) VALUES (?, ?)",
                (priority, len(episode)),
            )
            episode_id = self.cursor.lastrowid

            experiences = [
                (
                    episode_id,
                    self.numpy_to_blob(state),
                    int(action),
                    float(reward),
                    self.numpy_to_blob(next_state),
                    int(done),
                )
                for state, action, reward, next_state, done in episode
            ]

            self.cursor.executemany(
                """
                INSERT INTO experiences (episode_id, state, action, reward, next_state, done)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                experiences,
            )

            # Remove old episodes if capacity is exceeded
            self.cursor.execute("SELECT COUNT(*) FROM episodes")
            count = self.cursor.fetchone()[0]
            if count > self.capacity:
                self.cursor.execute(
                    """
                DELETE FROM experiences WHERE episode_id IN
                (SELECT episode_id FROM episodes ORDER BY priority LIMIT ?)
                """,
                    (count - self.capacity,),
                )
                self.cursor.execute(
                    """
                DELETE FROM episodes WHERE episode_id IN
                (SELECT episode_id FROM episodes ORDER BY priority LIMIT ?)
                """,
                    (count - self.capacity,),
                )

            self.conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while adding episode: {e}")
            self.conn.rollback()
        finally:
            self.close()

    def sample(self, num_episodes, sequences_per_episode, sequence_length):
        if not self.conn:
            self.connect()

        try:
            self.conn.execute("BEGIN TRANSACTION")

            self.cursor.execute("SELECT SUM(priority) FROM episodes")
            total_priority = self.cursor.fetchone()[0]
            if total_priority is None:
                return None  # No episodes available

            # Sample episodes based on priority
            self.cursor.execute("SELECT episode_id, priority, length FROM episodes")
            episodes = self.cursor.fetchall()
            probabilities = np.array([ep[1] for ep in episodes]) / total_priority
            chosen_indices = np.random.choice(
                len(episodes), num_episodes, p=probabilities, replace=False
            )

            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_next_states,
                batch_dones,
            ) = ([], [], [], [], [])
            episode_ids = []

            for idx in chosen_indices:
                episode = episodes[idx]
                episode_id, _, episode_length = episode
                if episode_length >= sequence_length:
                    # Always include start and end sequences
                    start_sequences = [(0, min(sequence_length, episode_length))]
                    end_sequences = [(max(0, episode_length - sequence_length), episode_length)]
                    
                    # Calculate remaining sequences to sample
                    remaining_sequences = sequences_per_episode - 2
                    if remaining_sequences > 0:
                        for _ in range(remaining_sequences):
                            start = np.random.randint(0, episode_length - sequence_length + 1)
                            start_sequences.append((start, start + sequence_length))
                    
                    all_sequences = start_sequences + end_sequences
                    
                    for start, end in all_sequences:
                        self.cursor.execute(
                            """
                            SELECT state, action, reward, next_state, done
                            FROM experiences
                            WHERE episode_id = ?
                            LIMIT ? OFFSET ?
                            """,
                            (episode_id, end - start, start),
                        )

                        sequence = self.cursor.fetchall()
                        states, actions, rewards, next_states, dones = zip(*sequence)

                        batch_states.append([self.blob_to_numpy(s) for s in states])
                        batch_actions.append(list(actions))
                        batch_rewards.append(list(rewards))
                        batch_next_states.append([self.blob_to_numpy(ns) for ns in next_states])
                        batch_dones.append([bool(d) for d in dones])
                        episode_ids.append(episode_id)

            # Calculate importance sampling weights
            weights = (len(episodes) * probabilities[chosen_indices]) ** -self.beta
            weights = np.repeat(weights, len(all_sequences))
            weights /= weights.max()

            self.beta = min(1.0, self.beta + self.beta_increment)

            self.conn.commit()
            return (
                torch.FloatTensor(np.array(batch_states)),
                torch.LongTensor(np.array(batch_actions)),
                torch.FloatTensor(np.array(batch_rewards)),
                torch.FloatTensor(np.array(batch_next_states)),
                torch.BoolTensor(np.array(batch_dones)),
                episode_ids,
                torch.FloatTensor(weights),
            )
        except sqlite3.Error as e:
            print(f"An error occurred while sampling: {e}")
            self.conn.rollback()
            return None
        finally:
            self.close()

    def update_priorities(self, episode_ids, errors):
        if not self.conn:
            self.connect()

        try:
            self.conn.execute("BEGIN TRANSACTION")

            episode_errors = defaultdict(list)
            for episode_id, error in zip(episode_ids, errors):
                episode_errors[episode_id].append(error)

            updates = []
            for episode_id, errors_list in episode_errors.items():
                avg_error = np.mean(errors_list)
                priority = (avg_error + 1e-5) ** self.alpha
                updates.append((priority, episode_id))

            self.cursor.executemany(
                "UPDATE episodes SET priority = ? WHERE episode_id = ?", updates
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
            self.cursor.execute("SELECT COUNT(*) FROM episodes")
            return self.cursor.fetchone()[0]
        finally:
            self.close()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    @staticmethod
    def numpy_to_blob(arr):
        return sqlite3.Binary(pickle.dumps(arr, protocol=4))

    @staticmethod
    def blob_to_numpy(blob):
        return pickle.loads(blob)
