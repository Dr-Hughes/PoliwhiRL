# -*- coding: utf-8 -*-
import unittest
import numpy as np
import torch

from PoliwhiRL.replay import PPOMemory


def make_config(update_frequency=10, sequence_length=4, input_shape=(3, 8, 8)):
    return {
        "device": "cpu",
        "ppo_update_frequency": update_frequency,
        "sequence_length": sequence_length,
        "input_shape": input_shape,
    }


class TestPPOMemory(unittest.TestCase):
    def setUp(self):
        self.input_shape = (3, 8, 8)
        self.update_frequency = 10
        self.sequence_length = 4
        self.config = make_config(
            update_frequency=self.update_frequency,
            sequence_length=self.sequence_length,
            input_shape=self.input_shape,
        )
        self.memory = PPOMemory(self.config)

    def _store_n(self, n, num_layers=2, mem_len=5, d_model=16, with_mems=True):
        for i in range(n):
            state = np.full(self.input_shape, i, dtype=np.uint8)
            next_state = np.full(self.input_shape, i + 1, dtype=np.uint8)
            mems = (
                [torch.full((1, mem_len, d_model), float(i)) for _ in range(num_layers)]
                if with_mems
                else None
            )
            self.memory.store_transition(
                state=state,
                next_state=next_state,
                action=i % 9,
                reward=float(i),
                done=(i == n - 1),
                log_prob=-0.1 * i,
                mems=mems,
            )

    def test_initial_state(self):
        self.assertEqual(len(self.memory), 0)
        self.assertIsNone(self.memory.mems)
        self.assertIsNone(self.memory.last_next_state)

    def test_store_transition_increments_length(self):
        self._store_n(3, with_mems=False)
        self.assertEqual(len(self.memory), 3)

    def test_get_all_data_returns_none_below_sequence_length(self):
        self._store_n(self.sequence_length - 1, with_mems=False)
        self.assertIsNone(self.memory.get_all_data())

    def test_get_all_data_shapes(self):
        n = 6
        num_layers = 2
        mem_len = 5
        d_model = 16
        self._store_n(n, num_layers=num_layers, mem_len=mem_len, d_model=d_model)

        data = self.memory.get_all_data()
        self.assertIsNotNone(data)

        expected_seqs = n - self.sequence_length + 1
        self.assertEqual(
            data["states"].shape,
            (expected_seqs, self.sequence_length, *self.input_shape),
        )
        self.assertEqual(
            data["next_states"].shape,
            (expected_seqs, self.sequence_length, *self.input_shape),
        )
        self.assertEqual(data["actions"].shape, (expected_seqs,))
        self.assertEqual(data["rewards"].shape, (expected_seqs,))
        self.assertEqual(data["dones"].shape, (expected_seqs,))
        self.assertEqual(data["old_log_probs"].shape, (expected_seqs,))

        self.assertIn("mems", data)
        self.assertEqual(len(data["mems"]), num_layers)
        for layer_mem in data["mems"]:
            self.assertEqual(layer_mem.shape, (expected_seqs, mem_len, d_model))

    def test_get_all_data_action_alignment(self):
        """Actions/rewards align with the last frame of each window."""
        n = 6
        self._store_n(n, with_mems=False)
        data = self.memory.get_all_data()

        # First action returned corresponds to step (sequence_length - 1).
        expected_first_action = (self.sequence_length - 1) % 9
        self.assertEqual(data["actions"][0].item(), expected_first_action)
        # Last action returned corresponds to step (n - 1).
        self.assertEqual(data["actions"][-1].item(), (n - 1) % 9)
        # Rewards stored as floats matching the step index.
        self.assertAlmostEqual(
            data["rewards"][0].item(), float(self.sequence_length - 1)
        )
        self.assertAlmostEqual(data["rewards"][-1].item(), float(n - 1))

    def test_get_all_data_next_states_boundary(self):
        """The final next-state window ends at last_next_state."""
        # Size update_frequency to match the rollout so negative slicing into
        # the buffer hits only filled entries. (When episode_length <
        # update_frequency the tail of the buffer is zeros — see the known
        # edge-case caveat on PPOMemory.get_all_data.)
        n = self.update_frequency
        self.memory = PPOMemory(
            make_config(
                update_frequency=n,
                sequence_length=self.sequence_length,
                input_shape=self.input_shape,
            )
        )
        self._store_n(n, with_mems=False)
        data = self.memory.get_all_data()

        # last frame of final next_state window should equal last_next_state
        # (the next_state stored on the final transition: filled with n).
        last_window = data["next_states"][-1].numpy()
        self.assertTrue(np.all(last_window[-1] == n))

        # The frame just before it is the previous state (filled with n-1).
        self.assertTrue(np.all(last_window[-2] == n - 1))

    def test_get_all_data_mems_alignment(self):
        """mems[t] corresponds to the mems captured at the action's timestep."""
        n = 6
        num_layers = 2
        self._store_n(n, num_layers=num_layers)
        data = self.memory.get_all_data()

        # Mems were filled with float(i) at step i. Aligned mems start at
        # step (sequence_length - 1).
        first_mem_value = data["mems"][0][0, 0, 0].item()
        self.assertAlmostEqual(first_mem_value, float(self.sequence_length - 1))
        last_mem_value = data["mems"][0][-1, 0, 0].item()
        self.assertAlmostEqual(last_mem_value, float(n - 1))

    def test_reset_clears_all_state(self):
        self._store_n(3)
        self.memory.reset()
        self.assertEqual(len(self.memory), 0)
        self.assertIsNone(self.memory.mems)
        self.assertIsNone(self.memory.last_next_state)
        # Buffers are reallocated but zeroed.
        self.assertTrue(np.all(self.memory.states == 0))
        self.assertTrue(np.all(self.memory.actions == 0))
        self.assertTrue(np.all(self.memory.rewards == 0))
        self.assertTrue(np.all(self.memory.dones == 0))
        self.assertTrue(np.all(self.memory.log_probs == 0))

    def test_get_all_data_without_mems(self):
        self._store_n(self.sequence_length, with_mems=False)
        data = self.memory.get_all_data()
        self.assertIsNotNone(data)
        self.assertNotIn("mems", data)

    def test_exact_sequence_length_returns_single_window(self):
        self._store_n(self.sequence_length, with_mems=False)
        data = self.memory.get_all_data()
        self.assertEqual(data["states"].shape[0], 1)
        self.assertEqual(data["actions"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
