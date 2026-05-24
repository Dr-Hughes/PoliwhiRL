# -*- coding: utf-8 -*-
"""Unit tests for VecPPOMemory.

No emulator needed — these test the buffer's shape/slicing logic against
synthetic data.
"""
import unittest
import numpy as np
import torch

from PoliwhiRL.replay import VecPPOMemory


RAM_DIM = 4


def _make_config(rollout_length=10, sequence_length=4, input_shape=(3, 8, 8)):
    return {
        "device": "cpu",
        "ppo_update_frequency": rollout_length,
        "sequence_length": sequence_length,
        "input_shape": input_shape,
        "ram_obs_dim": RAM_DIM,
    }


def _fake_mems(num_layers=2, num_envs=2, mem_len=5, d_model=6):
    return [torch.zeros(num_envs, mem_len, d_model) for _ in range(num_layers)]


def _zeros_ram(num_envs):
    return np.zeros((num_envs, RAM_DIM), dtype=np.float32)


def _store_step(mem, **overrides):
    """Helper that fills required store_step args with sane zero defaults so
    each test only specifies the fields it cares about."""
    N = mem.num_envs
    in_shape = mem.input_shape
    defaults = dict(
        states=np.zeros((N,) + in_shape, dtype=np.uint8),
        ram_states=_zeros_ram(N),
        next_states=np.zeros((N,) + in_shape, dtype=np.uint8),
        next_ram_states=_zeros_ram(N),
        actions=np.zeros(N, dtype=np.int64),
        rewards=np.zeros(N, dtype=np.float32),
        dones=np.zeros(N, dtype=bool),
        log_probs=np.zeros(N, dtype=np.float32),
        mems=_fake_mems(num_envs=N),
    )
    defaults.update(overrides)
    mem.store_step(**defaults)


class TestVecPPOMemory(unittest.TestCase):
    def test_store_and_get_data_shapes(self):
        T, seq, in_shape, N = 10, 4, (3, 4, 4), 2
        cfg = _make_config(T, seq, in_shape)
        mem = VecPPOMemory(cfg, num_envs=N)
        rng = np.random.default_rng(0)

        for _ in range(T):
            _store_step(
                mem,
                states=rng.integers(0, 255, size=(N,) + in_shape, dtype=np.uint8),
                next_states=rng.integers(0, 255, size=(N,) + in_shape, dtype=np.uint8),
                ram_states=rng.standard_normal((N, RAM_DIM)).astype(np.float32),
                next_ram_states=rng.standard_normal((N, RAM_DIM)).astype(np.float32),
                actions=rng.integers(0, 9, size=N),
                rewards=rng.standard_normal(N).astype(np.float32),
                dones=rng.choice([True, False], size=N),
                log_probs=rng.standard_normal(N).astype(np.float32),
            )

        data = mem.get_data()
        W = T - seq + 1
        self.assertEqual(data["states"].shape, (W, N, seq) + in_shape)
        self.assertEqual(data["ram_states"].shape, (W, N, seq, RAM_DIM))
        self.assertEqual(data["next_states"].shape, (W, N, seq) + in_shape)
        self.assertEqual(data["next_ram_states"].shape, (W, N, seq, RAM_DIM))
        self.assertEqual(data["actions"].shape, (W, N))
        self.assertEqual(data["rewards"].shape, (W, N))
        self.assertEqual(data["dones"].shape, (W, N))
        self.assertEqual(data["old_log_probs"].shape, (W, N))
        self.assertEqual(data["last_next_obs"].shape, (N,) + in_shape)
        self.assertEqual(data["last_next_ram"].shape, (N, RAM_DIM))
        self.assertEqual(len(data["mems"]), 2)
        for m in data["mems"]:
            self.assertEqual(m.shape, (W, N, 5, 6))

    def test_get_data_none_when_underfilled(self):
        cfg = _make_config(rollout_length=10, sequence_length=4)
        mem = VecPPOMemory(cfg, num_envs=2)
        # Fewer than sequence_length+1 steps -> not enough data.
        for _ in range(3):
            _store_step(mem)
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
            _store_step(
                mem,
                states=rng.integers(0, 255, size=(N,) + in_shape, dtype=np.uint8),
                next_states=rng.integers(0, 255, size=(N,) + in_shape, dtype=np.uint8),
                actions=actions,
                rewards=rewards,
                log_probs=log_probs,
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
                np.allclose(data["rewards"][w].numpy(), all_rewards[raw_idx], atol=1e-6)
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
        last_real_next_ram = None
        for t in range(T):
            next_states = rng.integers(0, 255, size=(N,) + in_shape, dtype=np.uint8)
            next_ram = rng.standard_normal((N, RAM_DIM)).astype(np.float32)
            last_real_next = next_states
            last_real_next_ram = next_ram
            _store_step(
                mem,
                states=rng.integers(0, 255, size=(N,) + in_shape, dtype=np.uint8),
                next_states=next_states,
                ram_states=rng.standard_normal((N, RAM_DIM)).astype(np.float32),
                next_ram_states=next_ram,
            )

        data = mem.get_data()
        W = T - seq + 1
        last_window_tail = data["next_states"][W - 1, :, -1].numpy().astype(np.uint8)
        self.assertTrue(np.array_equal(last_window_tail, last_real_next))
        last_ram_tail = data["next_ram_states"][W - 1, :, -1].numpy()
        self.assertTrue(np.allclose(last_ram_tail, last_real_next_ram, atol=1e-6))

    def test_reset_clears_state(self):
        cfg = _make_config(rollout_length=4, sequence_length=2)
        mem = VecPPOMemory(cfg, num_envs=2)
        for _ in range(3):
            _store_step(mem, states=np.ones((2, 3, 8, 8), dtype=np.uint8))
        self.assertEqual(len(mem), 3)
        mem.reset()
        self.assertEqual(len(mem), 0)
        self.assertFalse(mem.is_full())
        self.assertIsNone(mem.last_next_obs)
        self.assertIsNone(mem.last_next_ram)


if __name__ == "__main__":
    unittest.main()
