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


if __name__ == "__main__":
    unittest.main()
