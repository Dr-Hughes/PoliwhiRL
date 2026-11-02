# -*- coding: utf-8 -*-
"""SIL demonstration pool (vec_ppo_agent._load_sil_demos / _sil_train):
demo legs are cut into sequence_length segments with raw discounted returns
(scaled by the live reward scaler when drawn), get burnt-in memories, are
mixed into SIL batches at most at sil_demo_fraction (never demo-only), and a
leg is retired once honest play is competent at its target checkpoint."""

import tempfile
import unittest
from collections import deque
from pathlib import Path

import numpy as np
import torch

from PoliwhiRL.agents.PPO.vec_ppo_agent import VecPPOAgent
from PoliwhiRL.environment.gym_env import RAM_OBS_DIM


class _Model:
    def __init__(self):
        self.batches = []
        self.calls = 0

    def init_mems(self, n):
        return [torch.zeros(n, 4, 8)]

    def actor_critic(self, states, ram, mems):
        self.calls += 1
        return None, None, [m + 1.0 for m in mems]

    def sil_update(self, batch):
        self.batches.append(batch)
        return {"sil_loss": 0.0}


class _Scaler:
    def scale_factor(self):
        return 0.5


def _agent(files, buffer=(), fraction=0.5, batch=2, competence=None):
    agent = VecPPOAgent.__new__(VecPPOAgent)
    agent.sequence_length = 4
    agent.gamma = 0.5
    agent.config = {}
    agent.total_rollouts = 0
    agent.model = _Model()
    agent.reward_scaler = _Scaler()
    agent.sil_batch_size = batch
    agent.sil_updates_per_rollout = 1
    agent.sil_demo_fraction = fraction
    agent.rc_exit_rate = 0.8
    agent.checkpoint_competence = competence or {}
    agent.device = torch.device("cpu")
    agent._sil_buffer = deque(buffer)
    agent._sil_demos = agent._load_sil_demos(files)
    return agent


def _demo(path, ram_dim=RAM_OBS_DIM, T=9, target=30):
    np.savez(
        path,
        images=np.zeros((T, 1, 2, 2), np.uint8),
        ram=np.zeros((T, ram_dim), np.float32),
        actions=np.arange(T),
        rewards=np.r_[np.zeros(T - 1), 8.0].astype(np.float32),
        ram_dim=ram_dim,
        target_flag=target,
    )


def _honest():
    return {
        "states": torch.zeros(4, 1, 2, 2, dtype=torch.uint8),
        "ram_states": torch.zeros(4, RAM_OBS_DIM),
        "actions": torch.zeros(4, dtype=torch.long),
        "returns": torch.full((4,), 7.0),
        "dones": torch.zeros(4, dtype=torch.bool),
        "mems": [torch.zeros(4, 8, dtype=torch.float16)],
    }


class TestSilDemos(unittest.TestCase):
    def test_segments_returns_and_skip(self):
        with tempfile.TemporaryDirectory() as d:
            _demo(Path(d) / "a.npz")
            _demo(Path(d) / "old.npz", ram_dim=RAM_OBS_DIM - 1)
            agent = _agent([str(Path(d) / "*.npz")])
        self.assertEqual(len(agent._sil_demos), 2)  # 9 steps -> two full segments
        # reward 8 at t=8, gamma 0.5: return at t=0 is 8 * 0.5**8
        self.assertAlmostEqual(float(agent._sil_demos[0]["raw_returns"][0]), 8 * 0.5**8, places=6)

    def test_batch_mixes_demos_scales_returns_and_burns_in_memory(self):
        with tempfile.TemporaryDirectory() as d:
            _demo(Path(d) / "a.npz")
            agent = _agent([str(Path(d) / "a.npz")], buffer=[_honest(), _honest()])
            agent._sil_train()
        batch = agent.model.batches[0]
        returns = batch["returns"]
        self.assertIn(7.0, returns[:, 0].tolist())  # one honest item
        demo_row = 0 if returns[0, 0] != 7.0 else 1
        self.assertTrue(torch.all(returns[demo_row] <= 4.0))  # scaled by 0.5
        # The second segment's memory is the policy's state after the first.
        self.assertEqual(float(agent._sil_demos[1]["mems"][0].max()), 1.0)

    def test_no_demo_only_batches(self):
        with tempfile.TemporaryDirectory() as d:
            _demo(Path(d) / "a.npz")
            agent = _agent([str(Path(d) / "a.npz")], fraction=0.5)
            agent._sil_train()
        self.assertEqual(agent.model.batches, [])

    def test_leg_retired_when_honest_play_is_competent(self):
        with tempfile.TemporaryDirectory() as d:
            _demo(Path(d) / "a.npz", target=30)
            agent = _agent([str(Path(d) / "a.npz")], competence={30: 0.9})
        self.assertEqual(agent._active_sil_demos(), [])
        agent.checkpoint_competence = {30: 0.5}
        self.assertEqual(len(agent._active_sil_demos()), 2)


if __name__ == "__main__":
    unittest.main()
