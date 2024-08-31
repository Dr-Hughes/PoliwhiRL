# -*- coding: utf-8 -*-
import sqlite3
import numpy as np
import torch
from collections import defaultdict
import io


class PrioritizedReplayBuffer:
    def __init__(
        self, db_path, capacity=1000000, alpha=0.6, beta=0.4, beta_increment=0.001
    ):
        self.db_path = db_path
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.setup_database()
        self.episode_buffer = []

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
        self.conn.commit()

    def add(self, state, action, reward, next_state, done):
        self.episode_buffer.append((state, action, reward, next_state, done))

        if done:
            self.add_episode(self.episode_buffer)
            self.episode_buffer = []

    def add_episode(self, episode):
        priority = 1.0  # Start with max priority for new episodes
        self.cursor.execute(
            "INSERT INTO episodes (priority, length) VALUES (?, ?)",
            (priority, len(episode)),
        )
        episode_id = self.cursor.lastrowid

        for state, action, reward, next_state, done in episode:
            state_blob = self.numpy_to_blob(state)
            next_state_blob = self.numpy_to_blob(next_state)
            self.cursor.execute(
                """
            INSERT INTO experiences (episode_id, state, action, reward, next_state, done)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    episode_id,
                    state_blob,
                    int(action),
                    float(reward),
                    next_state_blob,
                    int(done),
                ),
            )

        self.conn.commit()

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

    def sample(self, num_episodes, sequences_per_episode, sequence_length):
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

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = (
            [],
            [],
            [],
            [],
            [],
        )
        episode_ids = []

        for idx in chosen_indices:
            episode = episodes[idx]
            episode_id, _, episode_length = episode

            if episode_length >= sequence_length:
                for _ in range(sequences_per_episode):
                    start = np.random.randint(0, episode_length - sequence_length + 1)

                    self.cursor.execute(
                        """
                    SELECT state, action, reward, next_state, done
                    FROM experiences
                    WHERE episode_id = ?
                    LIMIT ? OFFSET ?
                    """,
                        (episode_id, sequence_length, start),
                    )

                    sequence = self.cursor.fetchall()
                    states, actions, rewards, next_states, dones = zip(*sequence)

                    batch_states.append([self.blob_to_numpy(s) for s in states])
                    batch_actions.append(list(actions))
                    batch_rewards.append(list(rewards))
                    batch_next_states.append(
                        [self.blob_to_numpy(ns) for ns in next_states]
                    )
                    batch_dones.append([bool(d) for d in dones])
                    episode_ids.append(episode_id)

        # Calculate importance sampling weights
        weights = (len(episodes) * probabilities[chosen_indices]) ** -self.beta
        weights = np.repeat(weights, sequences_per_episode)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            torch.FloatTensor(np.array(batch_states)),
            torch.LongTensor(np.array(batch_actions)),
            torch.FloatTensor(np.array(batch_rewards)),
            torch.FloatTensor(np.array(batch_next_states)),
            torch.BoolTensor(np.array(batch_dones)),
            episode_ids,
            torch.FloatTensor(weights),
        )

    def update_priorities(self, episode_ids, errors):
        # Use a defaultdict to collect errors for each episode
        episode_errors = defaultdict(list)

        # Collect all errors for each episode
        for episode_id, error in zip(episode_ids, errors):
            episode_errors[episode_id].append(error)

        # Calculate average error for each episode and update priorities
        for episode_id, errors_list in episode_errors.items():
            avg_error = np.mean(errors_list)
            priority = (avg_error + 1e-5) ** self.alpha
            self.cursor.execute(
                "UPDATE episodes SET priority = ? WHERE episode_id = ?",
                (priority, episode_id),
            )

        self.conn.commit()

    def __len__(self):
        self.cursor.execute("SELECT COUNT(*) FROM episodes")
        return self.cursor.fetchone()[0]

    def close(self):
        self.conn.close()

    @staticmethod
    def numpy_to_blob(arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    @staticmethod
    def blob_to_numpy(blob):
        out = io.BytesIO(blob)
        out.seek(0)
        return np.load(out)
