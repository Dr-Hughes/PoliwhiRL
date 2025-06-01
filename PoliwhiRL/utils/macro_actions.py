# -*- coding: utf-8 -*-
"""Macro action learning system for discovering useful action sequences"""
import numpy as np
from collections import defaultdict, deque
import torch
import torch.nn as nn


class MacroActionLearner:
    """Learns and executes useful action sequences (macro actions)"""

    def __init__(self, action_space_size=9, max_sequence_length=8, min_frequency=5):
        """
        Initialize macro action learner

        Args:
            action_space_size: Number of primitive actions
            max_sequence_length: Maximum length of action sequences to consider
            min_frequency: Minimum frequency for a sequence to be considered a macro
        """
        self.action_space_size = action_space_size
        self.max_sequence_length = max_sequence_length
        self.min_frequency = min_frequency

        # Action sequence tracking
        self.sequence_outcomes = defaultdict(list)  # sequence -> [total_rewards]
        self.sequence_counts = defaultdict(int)  # sequence -> frequency
        self.sequence_contexts = defaultdict(list)  # sequence -> [state_contexts]

        # Current episode tracking
        self.current_episode_actions = []
        self.current_episode_rewards = []
        self.current_episode_states = []

        # Discovered macro actions
        self.macro_actions = {}  # sequence -> macro_info
        self.macro_action_ids = {}  # sequence -> unique_id
        self.next_macro_id = action_space_size  # Start IDs after primitive actions

        # Execution state
        self.executing_macro = None
        self.macro_step = 0

    def add_step(self, action, reward, state_hash=None):
        """Add a step from the current episode"""
        self.current_episode_actions.append(action)
        self.current_episode_rewards.append(reward)
        if state_hash:
            self.current_episode_states.append(state_hash)

    def end_episode(self):
        """Process the completed episode to discover sequences"""
        if len(self.current_episode_actions) < 3:
            self._reset_episode()
            return

        # Look for successful subsequences
        self._analyze_episode_sequences()
        self._reset_episode()

    def _analyze_episode_sequences(self):
        """Analyze the episode for useful action sequences"""
        actions = self.current_episode_actions
        rewards = self.current_episode_rewards

        # Calculate cumulative rewards for sequence evaluation
        cumulative_rewards = np.cumsum(rewards)

        # Extract sequences of different lengths
        for seq_len in range(3, min(self.max_sequence_length + 1, len(actions) + 1)):
            for start_idx in range(len(actions) - seq_len + 1):
                end_idx = start_idx + seq_len

                sequence = tuple(actions[start_idx:end_idx])

                # Calculate reward for this sequence
                if start_idx == 0:
                    sequence_reward = cumulative_rewards[end_idx - 1]
                else:
                    sequence_reward = (
                        cumulative_rewards[end_idx - 1]
                        - cumulative_rewards[start_idx - 1]
                    )

                # Only consider sequences with positive reward
                if sequence_reward > 0.1:
                    self.sequence_outcomes[sequence].append(sequence_reward)
                    self.sequence_counts[sequence] += 1

                    # Store context if available
                    if self.current_episode_states and start_idx < len(
                        self.current_episode_states
                    ):
                        context = self.current_episode_states[start_idx]
                        self.sequence_contexts[sequence].append(context)

        # Update macro actions
        self._update_macro_actions()

    def _update_macro_actions(self):
        """Update the set of discovered macro actions"""
        for sequence, count in self.sequence_counts.items():
            if count >= self.min_frequency and sequence not in self.macro_actions:
                outcomes = self.sequence_outcomes[sequence]

                if len(outcomes) >= self.min_frequency:
                    avg_reward = np.mean(outcomes)
                    std_reward = np.std(outcomes)
                    success_rate = len([r for r in outcomes if r > 0]) / len(outcomes)

                    # Quality criteria for macro actions
                    if avg_reward > 0.2 and success_rate > 0.7:
                        macro_id = self.next_macro_id
                        self.next_macro_id += 1

                        self.macro_actions[sequence] = {
                            "id": macro_id,
                            "avg_reward": avg_reward,
                            "std_reward": std_reward,
                            "success_rate": success_rate,
                            "frequency": count,
                            "length": len(sequence),
                        }
                        self.macro_action_ids[sequence] = macro_id

                        print(
                            f"Discovered macro action {macro_id}: {sequence} "
                            f"(reward: {avg_reward:.3f}, success: {success_rate:.3f})"
                        )

    def get_available_actions(self, include_macros=True):
        """Get list of available actions (primitive + macro)"""
        actions = list(range(self.action_space_size))

        if include_macros:
            actions.extend(self.macro_action_ids.values())

        return actions

    def is_macro_action(self, action_id):
        """Check if an action ID corresponds to a macro action"""
        return action_id >= self.action_space_size

    def get_macro_sequence(self, action_id):
        """Get the action sequence for a macro action ID"""
        for sequence, macro_id in self.macro_action_ids.items():
            if macro_id == action_id:
                return sequence
        return None

    def start_macro_execution(self, action_id):
        """Start executing a macro action"""
        sequence = self.get_macro_sequence(action_id)
        if sequence:
            self.executing_macro = sequence
            self.macro_step = 0
            return True
        return False

    def get_next_primitive_action(self):
        """Get the next primitive action in the current macro"""
        if self.executing_macro is None:
            return None

        if self.macro_step >= len(self.executing_macro):
            # Macro completed
            self.executing_macro = None
            self.macro_step = 0
            return None

        action = self.executing_macro[self.macro_step]
        self.macro_step += 1
        return action

    def is_executing_macro(self):
        """Check if currently executing a macro action"""
        return self.executing_macro is not None

    def abort_macro(self):
        """Abort current macro execution"""
        self.executing_macro = None
        self.macro_step = 0

    def get_macro_statistics(self):
        """Get statistics about discovered macro actions"""
        if not self.macro_actions:
            return {}

        stats = {
            "num_macros": len(self.macro_actions),
            "total_sequences_analyzed": len(self.sequence_counts),
            "avg_macro_length": np.mean(
                [info["length"] for info in self.macro_actions.values()]
            ),
            "avg_macro_reward": np.mean(
                [info["avg_reward"] for info in self.macro_actions.values()]
            ),
            "best_macro": max(
                self.macro_actions.items(), key=lambda x: x[1]["avg_reward"]
            ),
        }

        return stats

    def get_top_macros(self, n=5):
        """Get top N macro actions by average reward"""
        sorted_macros = sorted(
            self.macro_actions.items(), key=lambda x: x[1]["avg_reward"], reverse=True
        )
        return sorted_macros[:n]

    def _reset_episode(self):
        """Reset episode tracking"""
        self.current_episode_actions = []
        self.current_episode_rewards = []
        self.current_episode_states = []


class MacroActionPolicy(nn.Module):
    """Neural network policy that can output both primitive and macro actions"""

    def __init__(self, input_dim, primitive_actions, macro_learner):
        """
        Initialize macro action policy

        Args:
            input_dim: Input feature dimension
            primitive_actions: Number of primitive actions
            macro_learner: MacroActionLearner instance
        """
        super().__init__()
        self.primitive_actions = primitive_actions
        self.macro_learner = macro_learner

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
        )

        # Separate heads for primitive vs macro action selection
        self.primitive_head = nn.Linear(128, primitive_actions)
        self.macro_head = nn.Linear(128, 1)  # Probability of using macro action

        # Macro action selector (when macro mode is chosen)
        max_macros = 50  # Maximum number of macro actions to support
        self.macro_selector = nn.Linear(128, max_macros)

    def forward(self, x):
        """
        Forward pass

        Returns:
            (primitive_logits, macro_prob, macro_logits)
        """
        features = self.feature_extractor(x)

        primitive_logits = self.primitive_head(features)
        macro_prob = torch.sigmoid(self.macro_head(features))
        macro_logits = self.macro_selector(features)

        return primitive_logits, macro_prob, macro_logits

    def select_action(self, x, exploration=True):
        """
        Select an action (primitive or macro)

        Args:
            x: Input features
            exploration: Whether to use exploration

        Returns:
            (action_id, is_macro, log_prob)
        """
        primitive_logits, macro_prob, macro_logits = self.forward(x)

        # Decide whether to use macro action
        use_macro = torch.bernoulli(macro_prob).item() > 0.5

        available_macros = list(self.macro_learner.macro_action_ids.values())

        if use_macro and available_macros:
            # Select macro action
            # Mask unavailable macro slots
            macro_mask = torch.full_like(macro_logits, float("-inf"))
            for i, macro_id in enumerate(available_macros):
                if i < len(macro_logits[0]):
                    macro_mask[0, i] = 0

            masked_macro_logits = macro_logits + macro_mask

            if exploration:
                macro_dist = torch.distributions.Categorical(logits=masked_macro_logits)
                macro_idx = macro_dist.sample()
                log_prob = macro_dist.log_prob(macro_idx)
            else:
                macro_idx = torch.argmax(masked_macro_logits, dim=-1)
                log_prob = torch.log(
                    torch.softmax(masked_macro_logits, dim=-1)[0, macro_idx]
                )

            if macro_idx.item() < len(available_macros):
                action_id = available_macros[macro_idx.item()]
                return action_id, True, log_prob

        # Fall back to primitive action
        if exploration:
            primitive_dist = torch.distributions.Categorical(logits=primitive_logits)
            action_id = primitive_dist.sample()
            log_prob = primitive_dist.log_prob(action_id)
        else:
            action_id = torch.argmax(primitive_logits, dim=-1)
            log_prob = torch.log(torch.softmax(primitive_logits, dim=-1)[0, action_id])

        return action_id.item(), False, log_prob
