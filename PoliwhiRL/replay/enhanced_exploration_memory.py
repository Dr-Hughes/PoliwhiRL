# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict, deque
import hashlib
import pickle
import os


class EnhancedExplorationMemory:
    """Enhanced exploration memory that tracks state transitions, action outcomes, and waypoints"""

    def __init__(
        self, max_size=100, history_length=5, use_memory=True, action_space_size=9
    ):
        """
        Initialize enhanced exploration memory

        Args:
            max_size: Maximum number of recent screen hashes to store
            history_length: Number of past screen hashes to store
            use_memory: Whether to use the memory functionality
            action_space_size: Number of possible actions
        """
        self.max_size = max_size
        self.history_length = history_length
        self.use_memory = use_memory
        self.action_space_size = action_space_size

        # Original exploration memory components
        self.memory = []
        self.hash_visits = defaultdict(int)
        self.recent_hashes = []

        # Enhanced components for transition learning
        self.state_transitions = defaultdict(
            dict
        )  # state_hash -> {action: next_state_hash}
        self.action_outcomes = defaultdict(list)  # action -> [reward_history]
        self.state_rewards = defaultdict(list)  # state_hash -> [reward_history]
        self.state_coordinates = {}  # state_hash -> (x, y, map_id)

        # Waypoint discovery
        self.waypoints = {}  # state_hash -> waypoint_info
        self.waypoint_connections = defaultdict(
            list
        )  # waypoint -> [connected_waypoints]

        # Action sequence learning
        self.action_sequences = defaultdict(int)  # action_sequence -> frequency
        self.successful_sequences = []  # [(sequence, total_reward, frequency)]
        self.current_sequence = deque(maxlen=10)  # Current action sequence
        self.sequence_start_state = None

        # Statistics
        self.total_transitions = 0
        self.total_episodes = 0

    def add_screen(self, screen_array):
        """Add a screen to the memory (backward compatibility)"""
        if not self.use_memory:
            return

        screen_hash = self._compute_hash(screen_array)
        self._add_state(screen_hash, screen_array)

    def _add_state(self, state_hash, screen_array=None):
        """Internal method to add a state"""
        # Update recent hashes
        self.recent_hashes.append(state_hash)
        if len(self.recent_hashes) > self.history_length:
            self.recent_hashes.pop(0)

        # Get current history
        history = tuple(self.recent_hashes[:-1])

        # Increment visit count
        self.hash_visits[state_hash] += 1
        visits = self.hash_visits[state_hash]

        # Update memory (same as original)
        for i, (mem_hash, mem_history, _) in enumerate(self.memory):
            if mem_hash == state_hash:
                self.memory[i] = (state_hash, history, visits)
                self.memory.append(self.memory.pop(i))
                return

        self.memory.append((state_hash, history, visits))
        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def add_transition(self, state, action, next_state, reward, coordinates=None):
        """
        Add a state transition with action and reward information

        Args:
            state: Current state (screen array or hash)
            action: Action taken (integer)
            next_state: Next state (screen array or hash)
            reward: Reward received
            coordinates: Optional (x, y, map_id) tuple
        """
        if not self.use_memory:
            return

        # Get state hashes
        if isinstance(state, str):
            state_hash = state
        else:
            state_hash = self._compute_hash(state)
            self._add_state(state_hash, state)

        if isinstance(next_state, str):
            next_state_hash = next_state
        else:
            next_state_hash = self._compute_hash(next_state)
            self._add_state(next_state_hash, next_state)

        # Record transition
        self.state_transitions[state_hash][action] = next_state_hash
        self.total_transitions += 1

        # Record action outcome
        self.action_outcomes[action].append(reward)
        if len(self.action_outcomes[action]) > 1000:  # Keep last 1000 outcomes
            self.action_outcomes[action].pop(0)

        # Record state reward
        self.state_rewards[state_hash].append(reward)
        if len(self.state_rewards[state_hash]) > 100:  # Keep last 100 rewards
            self.state_rewards[state_hash].pop(0)

        # Store coordinates if provided
        if coordinates is not None:
            self.state_coordinates[state_hash] = coordinates

        # Update action sequence tracking
        self.current_sequence.append(action)
        if len(self.current_sequence) >= 3:  # Minimum sequence length
            seq = tuple(self.current_sequence)
            self.action_sequences[seq] += 1

        # Check if this should be a waypoint
        self._update_waypoints(state_hash, reward)

    def _update_waypoints(self, state_hash, reward):
        """Update waypoint information based on new observations"""
        visits = self.hash_visits[state_hash]
        avg_reward = (
            np.mean(self.state_rewards[state_hash])
            if self.state_rewards[state_hash]
            else 0
        )

        # Waypoint criteria:
        # 1. High reward states
        # 2. Frequently visited states (bottlenecks)
        # 3. States with high connectivity (many transitions)
        connectivity = len(self.state_transitions.get(state_hash, {}))

        # Calculate waypoint score
        reward_score = max(0, avg_reward) * 10
        visit_score = min(visits, 50) * 0.1  # Cap at 5.0
        connectivity_score = connectivity * 0.5

        waypoint_score = reward_score + visit_score + connectivity_score

        # Threshold for being considered a waypoint
        if waypoint_score > 2.0:
            self.waypoints[state_hash] = {
                "score": waypoint_score,
                "visits": visits,
                "avg_reward": avg_reward,
                "connectivity": connectivity,
                "coordinates": self.state_coordinates.get(state_hash),
            }

    def get_macro_actions(self, min_frequency=5, min_reward_improvement=0.1):
        """
        Get discovered macro actions (useful action sequences)

        Args:
            min_frequency: Minimum times sequence must be observed
            min_reward_improvement: Minimum average reward improvement

        Returns:
            List of (sequence, score) tuples
        """
        macro_actions = []

        for sequence, frequency in self.action_sequences.items():
            if frequency >= min_frequency and len(sequence) >= 3:
                # Calculate average reward for this sequence
                # This is simplified - in practice you'd track sequence outcomes
                sequence_score = frequency * 0.1

                if sequence_score > min_reward_improvement:
                    macro_actions.append((sequence, sequence_score))

        # Sort by score
        macro_actions.sort(key=lambda x: x[1], reverse=True)
        return macro_actions[:10]  # Return top 10

    def get_waypoints(self, top_k=20):
        """
        Get discovered waypoints

        Args:
            top_k: Number of top waypoints to return

        Returns:
            List of (state_hash, waypoint_info) tuples
        """
        sorted_waypoints = sorted(
            self.waypoints.items(), key=lambda x: x[1]["score"], reverse=True
        )
        return sorted_waypoints[:top_k]

    def get_action_effectiveness(self):
        """Get effectiveness statistics for each action"""
        effectiveness = {}

        for action in range(self.action_space_size):
            outcomes = self.action_outcomes.get(action, [])
            if outcomes:
                effectiveness[action] = {
                    "mean_reward": np.mean(outcomes),
                    "std_reward": np.std(outcomes),
                    "usage_count": len(outcomes),
                    "success_rate": sum(1 for r in outcomes if r > 0) / len(outcomes),
                }
            else:
                effectiveness[action] = {
                    "mean_reward": 0.0,
                    "std_reward": 0.0,
                    "usage_count": 0,
                    "success_rate": 0.0,
                }

        return effectiveness

    def get_state_connectivity(self, state_hash):
        """Get information about state connectivity"""
        if state_hash not in self.state_transitions:
            return {}

        transitions = self.state_transitions[state_hash]
        return {
            "outgoing_actions": list(transitions.keys()),
            "reachable_states": list(transitions.values()),
            "connectivity_score": len(transitions),
        }

    def find_path_between_waypoints(self, start_waypoint, end_waypoint):
        """Simple pathfinding between waypoints using learned transitions"""
        if start_waypoint not in self.waypoints or end_waypoint not in self.waypoints:
            return None

        # Simple BFS to find path
        from collections import deque

        queue = deque([(start_waypoint, [])])
        visited = {start_waypoint}

        while queue:
            current_state, path = queue.popleft()

            if current_state == end_waypoint:
                return path

            if len(path) > 20:  # Prevent infinite loops
                continue

            # Explore reachable states
            for action, next_state in self.state_transitions.get(
                current_state, {}
            ).items():
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [action]))

        return None  # No path found

    def get_memory_tensor(self, enhanced_features=False):
        """Enhanced memory tensor with optional additional features"""
        if enhanced_features:
            # Enhanced tensor: [visits, history..., avg_reward, connectivity, is_waypoint]
            if not self.use_memory or not self.memory:
                return np.zeros(
                    (self.max_size, 1 + self.history_length + 3), dtype=np.float32
                )

            tensor = np.zeros(
                (self.max_size, 1 + self.history_length + 3), dtype=np.float32
            )

            for i, (state_hash, history, visits) in enumerate(self.memory):
                if i >= self.max_size:
                    break

                # Original features
                tensor[i, 0] = visits
                for j, past_hash in enumerate(history):
                    if j < self.history_length:
                        tensor[i, j + 1] = 1.0

                # Enhanced features
                avg_reward = (
                    np.mean(self.state_rewards[state_hash])
                    if self.state_rewards[state_hash]
                    else 0
                )
                connectivity = len(self.state_transitions.get(state_hash, {}))
                is_waypoint = 1.0 if state_hash in self.waypoints else 0.0

                tensor[i, 1 + self.history_length] = avg_reward
                tensor[i, 1 + self.history_length + 1] = connectivity
                tensor[i, 1 + self.history_length + 2] = is_waypoint

            return tensor
        else:
            # Backward compatible: original format [visits, history...]
            if not self.use_memory or not self.memory:
                return np.zeros(
                    (self.max_size, 1 + self.history_length), dtype=np.float32
                )

            tensor = np.zeros(
                (self.max_size, 1 + self.history_length), dtype=np.float32
            )

            for i, (state_hash, history, visits) in enumerate(self.memory):
                if i >= self.max_size:
                    break

                # Original features only
                tensor[i, 0] = visits
                for j, past_hash in enumerate(history):
                    if j < self.history_length:
                        tensor[i, j + 1] = 1.0

            return tensor

    def get_exploration_bonus(self, state_hash):
        """Calculate exploration bonus for a state"""
        visits = self.hash_visits.get(state_hash, 0)

        # Decreasing bonus with more visits
        if visits == 0:
            return 1.0  # High bonus for new states
        elif visits < 5:
            return 0.5  # Medium bonus for rarely visited
        elif visits < 20:
            return 0.1  # Small bonus for occasionally visited
        else:
            return 0.0  # No bonus for frequently visited

    def save_to_file(self, filepath):
        """Save memory to file"""
        data = {
            "memory": self.memory,
            "hash_visits": dict(self.hash_visits),
            "state_transitions": dict(self.state_transitions),
            "action_outcomes": dict(self.action_outcomes),
            "state_rewards": dict(self.state_rewards),
            "state_coordinates": self.state_coordinates,
            "waypoints": self.waypoints,
            "action_sequences": dict(self.action_sequences),
            "total_transitions": self.total_transitions,
            "total_episodes": self.total_episodes,
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_from_file(self, filepath):
        """Load memory from file"""
        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            self.memory = data.get("memory", [])
            self.hash_visits = defaultdict(int, data.get("hash_visits", {}))
            self.state_transitions = defaultdict(
                dict, data.get("state_transitions", {})
            )
            self.action_outcomes = defaultdict(list, data.get("action_outcomes", {}))
            self.state_rewards = defaultdict(list, data.get("state_rewards", {}))
            self.state_coordinates = data.get("state_coordinates", {})
            self.waypoints = data.get("waypoints", {})
            self.action_sequences = defaultdict(int, data.get("action_sequences", {}))
            self.total_transitions = data.get("total_transitions", 0)
            self.total_episodes = data.get("total_episodes", 0)

            return True
        except Exception as e:
            print(f"Error loading exploration memory: {e}")
            return False

    def _compute_hash(self, screen_array):
        """Compute hash of screen array"""
        array_bytes = screen_array.tobytes()
        return hashlib.md5(array_bytes).hexdigest()

    def reset(self):
        """Reset memory but keep learned patterns"""
        # Reset episode-specific data but keep learned knowledge
        self.memory = []
        self.recent_hashes = []
        self.current_sequence = deque(maxlen=10)
        self.sequence_start_state = None
        self.total_episodes += 1

    def get_statistics(self):
        """Get comprehensive statistics about the exploration memory"""
        return {
            "total_unique_states": len(self.hash_visits),
            "total_transitions": self.total_transitions,
            "total_episodes": self.total_episodes,
            "waypoints_discovered": len(self.waypoints),
            "macro_actions_discovered": len(
                [s for s, f in self.action_sequences.items() if f >= 5]
            ),
            "most_visited_state_visits": (
                max(self.hash_visits.values()) if self.hash_visits else 0
            ),
            "avg_state_visits": (
                np.mean(list(self.hash_visits.values())) if self.hash_visits else 0
            ),
            "action_usage": {
                action: len(outcomes)
                for action, outcomes in self.action_outcomes.items()
            },
        }
