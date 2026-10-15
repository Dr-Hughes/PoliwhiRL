# -*- coding: utf-8 -*-
"""Tests for the PPO minibatch helper. Uses a stub model so no emulator
or real network is needed."""
import unittest
import torch

from PoliwhiRL.agents.PPO._minibatch import run_ppo_epochs


class _StubModel:
    """Records every (batch_size, episode) call to update() and returns a
    canned (loss, approx_kl) tuple sequence so we can assert the loop's
    minibatching + KL early-stop behaviour."""

    def __init__(self, kl_sequence=None):
        self.calls = []
        self._kls = list(kl_sequence) if kl_sequence is not None else None
        self._kl_idx = 0
        self.actor_critic = torch.nn.Linear(1, 1, bias=False)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=0.1)
        self.config = {"ppo_hard_kl": 0.06, "ppo_lr_hard_floor": 0.01}
        self.last_update_diag = {}

    def update(self, mb, episode):
        self.calls.append(
            {
                "batch_size": mb["states"].size(0),
                "episode": episode,
                "tensor_keys": sorted(mb.keys()),
                "mem_layer_count": len(mb.get("mems", [])),
            }
        )
        self.optimizer.zero_grad()
        loss = self.actor_critic.weight.sum()
        loss.backward()
        self.optimizer.step()
        self.last_update_diag = {"actor_loss": 1.0}
        return 1.0, 0.0

    def exact_policy_kls(self, data):
        if self._kls is None:
            return 0.0, 0.0
        kl = self._kls[min(self._kl_idx, len(self._kls) - 1)]
        self._kl_idx += 1
        return kl, kl

    def effective_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def halve_lr_multiplier(self):
        lr = max(self.effective_lr() * 0.5, self.config["ppo_lr_hard_floor"])
        self.optimizer.param_groups[0]["lr"] = lr
        return lr


def _make_data(batch_size, seq_len=4, num_layers=2, mem_len=3, d_model=5):
    return {
        "states": torch.zeros(batch_size, seq_len, 3, 8, 8),
        "actions": torch.zeros(batch_size, dtype=torch.long),
        "old_log_probs": torch.zeros(batch_size),
        "returns": torch.zeros(batch_size),
        "advantages": torch.zeros(batch_size),
        "old_values": torch.zeros(batch_size),
        "mems": [torch.zeros(batch_size, mem_len, d_model) for _ in range(num_layers)],
    }


class TestRunPpoEpochs(unittest.TestCase):
    def test_full_batch_when_minibatch_none(self):
        data = _make_data(64)
        model = _StubModel()
        total_loss, epochs_run = run_ppo_epochs(
            model=model,
            data=data,
            step=0,
            epochs=3,
            minibatch_size=None,
            target_kl=None,
        )
        self.assertEqual(epochs_run, 3)
        self.assertEqual(len(model.calls), 3)
        for c in model.calls:
            self.assertEqual(c["batch_size"], 64)
        self.assertEqual(total_loss, 3.0)  # 3 epochs * (1.0 / 1 minibatch)

    def test_minibatching_splits_correctly(self):
        data = _make_data(100)
        model = _StubModel()
        run_ppo_epochs(
            model=model,
            data=data,
            step=0,
            epochs=2,
            minibatch_size=32,
            target_kl=None,
        )
        # 100 / 32 -> ceil = 4 minibatches per epoch * 2 epochs = 8 calls
        self.assertEqual(len(model.calls), 8)
        # Each minibatch should be <= 32, and the sum across one epoch == 100
        sizes_epoch1 = [c["batch_size"] for c in model.calls[:4]]
        sizes_epoch2 = [c["batch_size"] for c in model.calls[4:]]
        self.assertEqual(sum(sizes_epoch1), 100)
        self.assertEqual(sum(sizes_epoch2), 100)
        for s in sizes_epoch1 + sizes_epoch2:
            self.assertLessEqual(s, 32)
            self.assertGreater(s, 0)

    def test_kl_early_stop_aborts_remaining_epochs(self):
        data = _make_data(32)
        # Exact full-data post-update KLs stay low for two accepted epochs;
        # the third exceeds the soft target but remains below the hard limit.
        model = _StubModel(kl_sequence=[0.001, 0.001, 0.05])
        total_loss, epochs_run = run_ppo_epochs(
            model=model,
            data=data,
            step=0,
            epochs=10,
            minibatch_size=None,
            target_kl=0.01,
        )
        self.assertEqual(epochs_run, 3)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(model.last_epoch_diag["accepted_epochs"], 3)
        self.assertAlmostEqual(model.last_epoch_diag["approx_kl"], 0.05)

    def test_hard_kl_rolls_back_model_and_optimizer_and_halves_lr(self):
        model = _StubModel(kl_sequence=[0.2])
        before_weight = model.actor_critic.weight.detach().clone()
        before_state = model.optimizer.state_dict()
        total_loss, epochs_run = run_ppo_epochs(
            model, _make_data(8), 0, 3, None, 0.02
        )
        self.assertEqual((total_loss, epochs_run), (0.0, 0))
        self.assertTrue(torch.equal(model.actor_critic.weight, before_weight))
        self.assertEqual(len(before_state["state"]), len(model.optimizer.state_dict()["state"]))
        self.assertAlmostEqual(model.effective_lr(), 0.05)
        self.assertTrue(model.last_epoch_diag["kl_rejected"])

    def test_lr_halving_stops_at_floor(self):
        model = _StubModel(kl_sequence=[0.2])
        model.optimizer.param_groups[0]["lr"] = 0.015
        run_ppo_epochs(model, _make_data(8), 0, 1, None, 0.02)
        self.assertAlmostEqual(model.effective_lr(), 0.01)

    def test_nonfinite_kl_rejects_epoch(self):
        model = _StubModel(kl_sequence=[float("nan")])
        before_weight = model.actor_critic.weight.detach().clone()
        _, epochs_run = run_ppo_epochs(model, _make_data(8), 0, 1, None, 0.02)
        self.assertEqual(epochs_run, 0)
        self.assertTrue(torch.equal(model.actor_critic.weight, before_weight))
        self.assertTrue(model.last_epoch_diag["kl_rejected"])

    def test_rare_action_log_noise_does_not_reject_matching_policies(self):
        class ReplayProbe(_StubModel):
            def replay_policy_diagnostics(self, data):
                return {
                    "base_replay_kl": 2e-7,
                    "behaviour_replay_kl": 2e-7,
                    # A low-probability action can have a large log-space
                    # delta despite a negligible full-policy difference.
                    "action_log_prob_abs_max": 0.05,
                    "action_log_prob_abs_mean": 1e-4,
                }

        model = ReplayProbe()
        _loss, epochs_run = run_ppo_epochs(model, _make_data(8), 0, 1, None, 0.02)

        self.assertEqual(epochs_run, 1)
        self.assertFalse(model.last_epoch_diag["replay_rejected"])
        self.assertAlmostEqual(model.last_epoch_diag["action_log_prob_abs_max"], 0.05)

    def test_stage_tolerance_accepts_observed_mps_replay_noise(self):
        class ReplayProbe(_StubModel):
            def __init__(self):
                super().__init__()
                self.config["ppo_replay_tolerance"] = 1e-5

            def replay_policy_diagnostics(self, data):
                return {
                    # The corrected sequential/segmented replay has shown
                    # ~3e-6 categorical KL on MPS despite equivalent policy
                    # behaviour. This must not suppress all PPO updates.
                    "base_replay_kl": 3e-6,
                    "behaviour_replay_kl": 3e-6,
                    "action_log_prob_abs_max": 0.04,
                    "action_log_prob_abs_mean": 2e-4,
                }

        model = ReplayProbe()
        _loss, epochs_run = run_ppo_epochs(model, _make_data(8), 0, 1, None, 0.02)

        self.assertEqual(epochs_run, 1)
        self.assertFalse(model.last_epoch_diag["replay_rejected"])

    def test_minibatch_partitions_are_disjoint(self):
        # Use a permutation-marker trick: stamp each row with its index in
        # the `actions` field and confirm minibatches cover the full set
        # disjointly across one epoch.
        N = 50
        data = _make_data(N)
        data["actions"] = torch.arange(N, dtype=torch.long)

        seen = []

        class Probe(_StubModel):
            def update(self, mb, episode):
                seen.extend(mb["actions"].tolist())
                return super().update(mb, episode)

        run_ppo_epochs(
            model=Probe(),
            data=data,
            step=0,
            epochs=1,
            minibatch_size=17,
            target_kl=None,
        )
        self.assertEqual(sorted(seen), list(range(N)))

    def test_mems_list_sliced_per_minibatch(self):
        data = _make_data(64, num_layers=3, mem_len=4, d_model=6)
        model = _StubModel()
        run_ppo_epochs(
            model=model,
            data=data,
            step=0,
            epochs=1,
            minibatch_size=16,
            target_kl=None,
        )
        # 64/16 = 4 minibatches, each carrying all 3 mems layers
        self.assertEqual(len(model.calls), 4)
        for c in model.calls:
            self.assertEqual(c["mem_layer_count"], 3)


if __name__ == "__main__":
    unittest.main()
