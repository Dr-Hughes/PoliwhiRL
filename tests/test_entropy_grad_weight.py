# -*- coding: utf-8 -*-
"""The entropy bonus's gradient weight must equal ppo_entropy_coef.

Regression test for the 2026-10-23 review finding F1: _update_networks clips
the policy+value gradient and the entropy gradient against INDEPENDENT norm
budgets. When the clip was applied to coef*entropy, any raw entropy-gradient
norm above ppo_max_grad_norm was rescaled to exactly the budget, so the
configured coefficient no longer set the regulariser's weight — the clip did —
and the entropy term acted as a co-equal objective pushing the policy toward
uniform. The fix clips the UNIT-coefficient entropy gradient and multiplies by
the coefficient afterwards.

The test zeroes the policy-gradient and value-loss contributions (zero
advantages, zero value-loss coefficient) so the only surviving gradient is the
entropy term's; doubling the coefficient must then double the total gradient
norm. Under the pre-fix behaviour both runs would clip to the same budget and
the ratio would be 1.0.
"""

import math
import unittest

import torch

from PoliwhiRL.models.PPO.ppo_model_implementation import PPOModel


def _config(entropy_coef):
    return {
        "device": "cpu",
        "ram_obs_dim": 171,
        "ppo_learning_rate": 1e-4,
        "ppo_gamma": 0.99,
        "ppo_epsilon": 0.2,
        "ppo_value_loss_coef": 0.0,  # no critic gradient in this test
        "ppo_entropy_coef": entropy_coef,
        "ppo_entropy_coef_decay": 0.0,
        "ppo_entropy_coef_min": entropy_coef,
        "ppo_entropy_anneal_enabled": False,  # keep the coefficient constant
        "ppo_clip_value_loss": True,
        "ppo_max_grad_norm": 0.2,
        "ppo_gae_lambda": 0.95,
        "ppo_adam_eps": 1e-5,
        "ppo_lr_schedule": "constant",
        "num_rollouts": 10,
        "d_ram": 16,
        "d_model": 16,
        "n_heads": 2,
        "num_layers": 1,
        "mem_len": 4,
        "ppo_dropout": 0.0,
    }


def _data():
    # Non-degenerate inputs: at an exactly-uniform policy the entropy
    # gradient w.r.t. the logits is analytically zero, which would vacuate
    # the test. Random (seeded, hence identical across both runs) inputs give
    # a non-uniform policy with a non-zero entropy gradient.
    b, steps = 2, 2
    return {
        "states": torch.randn(b, steps, 3, 8, 8),
        "ram_states": torch.randn(b, steps, 171),
        "actions": torch.zeros(b, steps, dtype=torch.long),
        "old_log_probs": torch.zeros(b, steps),
        "old_values": torch.zeros(b, steps),
        "returns": torch.zeros(b, steps),
        "advantages": torch.zeros(b, steps),  # no policy-gradient contribution
        "dones": torch.zeros(b, steps, dtype=torch.bool),
    }


# The nine diagnostic keys added to PPOModel.last_update_diag: three critic
# metrics from _compute_ppo_losses, the two pre-clip totals and the four
# per-branch norms from _update_networks.
_DIAG_KEYS = (
    "explained_variance",
    "value_error_rmse",
    "return_std",
    "pv_grad_norm",
    "ent_grad_norm",
    "grad_norm_cnn",
    "grad_norm_ram",
    "grad_norm_trunk",
    "grad_norm_heads",
)


def _diag_config():
    # _config() zeroes ppo_value_loss_coef, which with _data()'s zero
    # advantages makes actor_loss AND critic_loss identically zero: the
    # policy/value backward would then produce all-zero gradients and every
    # grad_norm_* would read 0.0 whether or not the branch prefixes matched.
    # A live critic term is what makes the group norms a real assertion.
    return dict(_config(0.02), ppo_value_loss_coef=0.5)


def _diag_data():
    # Same shapes as _data(), but with non-degenerate advantages and returns.
    # Var(returns) == 0 is exactly the degenerate case explained_variance
    # reports as nan by design, so _data()'s zero returns cannot be reused here.
    g = torch.Generator().manual_seed(0)
    b, steps = 2, 2
    return {
        "states": torch.randn(b, steps, 3, 8, 8, generator=g),
        "ram_states": torch.randn(b, steps, 171, generator=g),
        "actions": torch.zeros(b, steps, dtype=torch.long),
        "old_log_probs": torch.zeros(b, steps),
        "old_values": torch.zeros(b, steps),
        "returns": torch.randn(b, steps, generator=g),
        "advantages": torch.randn(b, steps, generator=g),
        "dones": torch.zeros(b, steps, dtype=torch.bool),
    }


def _entropy_grad_norm(entropy_coef):
    torch.manual_seed(0)
    model = PPOModel((3, 8, 8), 9, _config(entropy_coef))
    model.update(_data(), step=0)
    params = [p for p in model.actor_critic.parameters() if p.grad is not None]
    return float(torch.norm(torch.stack([p.grad.norm() for p in params])))


class TestEntropyGradientWeight(unittest.TestCase):
    def test_doubling_the_coefficient_doubles_the_entropy_gradient(self):
        low = _entropy_grad_norm(0.02)
        high = _entropy_grad_norm(0.04)
        self.assertGreater(low, 0.0)
        self.assertAlmostEqual(high / low, 2.0, places=3)


class TestUpdateDiagnostics(unittest.TestCase):
    """The value-function and gradient-norm diagnostics added to
    ``last_update_diag`` must all survive a normal ``update()`` as finite
    floats, and the per-branch gradient norms must actually be attributed to
    their branch rather than silently reading 0.0 from a wrong name prefix."""

    def test_value_and_grad_norm_diagnostics_present_and_finite(self):
        torch.manual_seed(0)
        model = PPOModel((3, 8, 8), 9, _diag_config())
        model.update(_diag_data(), step=0)
        diag = model.last_update_diag
        for key in _DIAG_KEYS:
            self.assertIn(key, diag)
            self.assertTrue(math.isfinite(diag[key]), f"{key}={diag[key]!r}")
        # Non-degenerate (seeded random) image input puts a genuinely non-zero
        # gradient on the CNN branch, so a mis-grouped prefix — which would
        # report the empty-group value 0.0 — cannot pass silently. All four
        # groups are pinned, not just the CNN's: a wrong prefix on
        # ram_encoder./fuse, transformer_blocks. or fc_actor/fc_critic would
        # likewise read the empty-group 0.0, which is the exact failure this
        # test exists to catch.
        self.assertGreater(diag["grad_norm_cnn"], 0.0)
        self.assertGreater(diag["grad_norm_ram"], 0.0)
        self.assertGreater(diag["grad_norm_trunk"], 0.0)
        self.assertGreater(diag["grad_norm_heads"], 0.0)


if __name__ == "__main__":
    unittest.main()
