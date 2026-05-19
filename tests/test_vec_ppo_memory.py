# -*- coding: utf-8 -*-
"""Unit tests for VecPPOMemory.

No emulator needed — these test the buffer's shape/slicing logic against
synthetic data.
"""
import unittest
import numpy as np
import torch

from PoliwhiRL.replay import VecPPOMemory


def _make_config(rollout_length=10, sequence_length=4, input_shape=(3, 8, 8)):
    return {
        "device": "cpu",
        "ppo_update_frequency": rollout_length,
        "sequence_length": sequence_length,
        "input_shape": input_shape,
    }


def _fake_mems(num_layers=2, num_envs=2, mem_len=5, d_model=6):
    return [torch.zeros(num_envs, mem_len, d_model) for _ in range(num_layers)]


class TestVecPPOMemory(unittest.TestCase):
    def test_store_and_get_data_shapes(self):
        T, seq, in_shape, N = 10, 4, (3, 4, 4), 2
        cfg = _make_config(T, seq, in_shape)
        mem = VecPPOMemory(cfg, num_envs=N)
        rng = np.random.default_rng(0)

        for _ in range(T):
            states = rng.integers(0, 255, size=(N,) + in_shape, dtype=np.uint8)
            next_states = rng.integers(0, 255, size=(N,) + in_shape, dtype=np.uint8)
            mem.store_step(
                states=states,
                next_states=next_states,
                actions=rng.integers(0, 9, size=N),
                rewards=rng.standard_normal(N).astype(np.float32),
                dones=rng.choice([True, False], size=N),
                log_probs=rng.standard_normal(N).astype(np.float32),
                mems=_fake_mems(num_envs=N),
            )

        data = mem.get_data()
        W = T - seq + 1
        self.assertEqual(data["states"].shape, (W, N, seq) + in_shape)
        self.assertEqual(data["next_states"].shape, (W, N, seq) + in_shape)
        self.assertEqual(data["actions"].shape, (W, N))
        self.assertEqual(data["rewards"].shape, (W, N))
        self.assertEqual(data["dones"].shape, (W, N))
        self.assertEqual(data["old_log_probs"].shape, (W, N))
        self.assertEqual(data["last_next_obs"].shape, (N,) + in_shape)
        self.assertEqual(len(data["mems"]), 2)
        for m in data["mems"]:
            self.assertEqual(m.shape, (W, N, 5, 6))

    def test_get_data_none_when_underfilled(self):
        cfg = _make_config(rollout_length=10, sequence_length=4)
        mem = VecPPOMemory(cfg, num_envs=2)
        # Fewer than sequence_length+1 steps -> not enough data.
        for _ in range(3):
            mem.store_step(
                states=np.zeros((2, 3, 8, 8), dtype=np.uint8),
                next_states=np.zeros((2, 3, 8, 8), dtype=np.uint8),
                actions=np.zeros(2),
                rewards=np.zeros(2),
                dones=np.zeros(2, dtype=bool),
                log_probs=np.zeros(2),
                mems=_fake_mems(num_envs=2),
            )
        self.assertIsNone(mem.get_data())

    def test_window_alignment_against_raw_buffers(self):
        """Window w's action/reward/log_prob should equal buffer[w + seq - 1]."""
        T, seq, in_shape, N = 8, 3, (3, 4, 4), 2
        cfg = _make_config(T, seq, in_shape)
        mem = VecPPOMemory(cfg, num_envs=N)
        rng = np.random.default_rng(42)

        all_actions, all_rewards, all_log_probs = [], [], []
        for _ in range(T):
            actions = rng.integers(0, 9, size=N).astype(np.int64)
            rewards = rng.standard_normal(N).astype(np.float32)
            log_probs = rng.standard_normal(N).astype(np.float32)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_log_probs.append(log_probs)
            mem.store_step(
                states=rng.integers(0, 255, size=(N,) + in_shape, dtype=np.uint8),
                next_states=rng.integers(0, 255, size=(N,) + in_shape, dtype=np.uint8),
                actions=actions,
                rewards=rewards,
                dones=np.zeros(N, dtype=bool),
                log_probs=log_probs,
                mems=_fake_mems(num_envs=N),
            )

        data = mem.get_data()
        W = T - seq + 1
        for w in range(W):
            raw_idx = w + seq - 1
            self.assertTrue(
                torch.equal(
                    data["actions"][w],
                    torch.from_numpy(all_actions[raw_idx]),
                )
            )
            self.assertTrue(
                np.allclose(
                    data["rewards"][w].numpy(), all_rewards[raw_idx], atol=1e-6
                )
            )
            self.assertTrue(
                np.allclose(
                    data["old_log_probs"][w].numpy(),
                    all_log_probs[raw_idx],
                    atol=1e-6,
                )
            )

    def test_last_next_state_window_uses_last_next_obs(self):
        T, seq, in_shape, N = 6, 3, (3, 4, 4), 2
        cfg = _make_config(T, seq, in_shape)
        mem = VecPPOMemory(cfg, num_envs=N)

        rng = np.random.default_rng(7)
        last_real_next = None
        for t in range(T):
            states = rng.integers(0, 255, size=(N,) + in_shape, dtype=np.uint8)
            next_states = rng.integers(0, 255, size=(N,) + in_shape, dtype=np.uint8)
            last_real_next = next_states
            mem.store_step(
                states=states,
                next_states=next_states,
                actions=np.zeros(N, dtype=np.int64),
                rewards=np.zeros(N, dtype=np.float32),
                dones=np.zeros(N, dtype=bool),
                log_probs=np.zeros(N, dtype=np.float32),
                mems=_fake_mems(num_envs=N),
            )

        data = mem.get_data()
        W = T - seq + 1
        # The last frame of the last window's next_states should be last_next_obs
        # (i.e., the post-step observation after the final stored transition).
        last_window_tail = data["next_states"][W - 1, :, -1].numpy().astype(np.uint8)
        self.assertTrue(np.array_equal(last_window_tail, last_real_next))

    def test_reset_clears_state(self):
        cfg = _make_config(rollout_length=4, sequence_length=2)
        mem = VecPPOMemory(cfg, num_envs=2)
        for _ in range(3):
            mem.store_step(
                states=np.ones((2, 3, 8, 8), dtype=np.uint8),
                next_states=np.ones((2, 3, 8, 8), dtype=np.uint8),
                actions=np.zeros(2),
                rewards=np.ones(2),
                dones=np.zeros(2, dtype=bool),
                log_probs=np.zeros(2),
                mems=_fake_mems(num_envs=2),
            )
        self.assertEqual(len(mem), 3)
        mem.reset()
        self.assertEqual(len(mem), 0)
        self.assertFalse(mem.is_full())
        self.assertIsNone(mem.last_next_obs)


if __name__ == "__main__":
    unittest.main()
