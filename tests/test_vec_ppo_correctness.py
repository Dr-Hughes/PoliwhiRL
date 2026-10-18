# -*- coding: utf-8 -*-
"""Focused regressions for vector PPO distribution, memory, and GAE semantics."""
import tempfile
import types
import unittest

import numpy as np
import torch

from PoliwhiRL.agents.PPO.vec_ppo_agent import VecPPOAgent
from PoliwhiRL.environment.gym_env import (
    RAM_FEATURE_INDEX,
    RAM_OBS_DIM,
)
from PoliwhiRL.models.PPO import PPOModel
from PoliwhiRL.models.PPO.PPOTransformer import PPOTransformer
from PoliwhiRL.replay import VecPPOMemory


def _model_config():
    return {
        "device": "cpu",
        "ram_obs_dim": RAM_OBS_DIM,
        "d_ram": 8,
        "d_model": 16,
        "n_heads": 2,
        "num_layers": 1,
        "mem_len": 4,
        "ppo_dropout": 0.0,
        "ppo_learning_rate": 3e-4,
        "ppo_gamma": 0.99,
        "ppo_epsilon": 0.2,
        "ppo_value_loss_coef": 0.5,
        "ppo_entropy_coef": 0.01,
        "ppo_entropy_coef_decay": 1.0,
        "ppo_entropy_coef_min": 0.01,
        "ppo_entropy_anneal_enabled": False,
        "ppo_clip_value_loss": True,
        "ppo_scheduler_t_max": 1,
        "ppo_lr_schedule": "constant",
        "action_mask_enabled": True,
        # T = 1 + 1.0*clock, so clock==1 gives T=2 (sqrt reshaping) — clean
        # numbers for test assertions (see test_temperature_endpoints_and_mask).
        "stuck_action_temperature": 1.0,
        # Obsolete settings must not alter the behaviour distribution.
    }


class TestBehaviourPolicyInvariant(unittest.TestCase):
    def test_collected_log_prob_is_policy_log_prob_at_high_stagnation(self):
        class FakeModel:
            @staticmethod
            def _action_mask_for(ram):
                mask = torch.zeros(ram.shape[0], ram.shape[1], 9)
                mask[..., :2] = 1.0
                return mask

            @staticmethod
            def actor_critic(states, ram, mems, action_mask=None):
                probs = torch.tensor(
                    [[[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
                    dtype=torch.float32,
                ).expand(states.shape[0], states.shape[1], -1)
                values = torch.zeros(states.shape[0], states.shape[1], 1)
                return probs, values, [m + 1.0 for m in mems]

            @staticmethod
            def _normalise_base_policy(probs, action_mask=None):
                if action_mask is not None:
                    probs = probs * action_mask
                return probs / probs.sum(dim=-1, keepdim=True)

            @staticmethod
            def _behaviour_policy(probs, ram, action_mask=None):
                valid = action_mask[:, -1]
                uniform = valid / valid.sum(dim=-1, keepdim=True)
                return 0.5 * probs + 0.5 * uniform

        class FakeMemory:
            stored = None

            def store_step(self, **kwargs):
                self.stored = kwargs

        class FakeScaler:
            @staticmethod
            def observe(rewards, dones):
                return None

        class FakeEnv:
            @staticmethod
            def step(actions):
                obs = {
                    "image": np.zeros((1, 1, 8, 8), dtype=np.uint8),
                    "ram": np.zeros((1, RAM_OBS_DIM), dtype=np.float32),
                }
                return obs, np.zeros(1, dtype=np.float32), np.zeros(1, dtype=bool), [None]

        agent = VecPPOAgent.__new__(VecPPOAgent)
        agent.rollout_length = 1
        agent.num_envs = 1
        agent.device = torch.device("cpu")
        agent.model = FakeModel()
        agent.memory = FakeMemory()
        agent.reward_scaler = FakeScaler()
        agent.episode_data = {"buttons_pressed": []}
        agent._episode_action_traces = [[]]
        agent._rollout_entropy_sum = 0.0
        agent._rollout_behaviour_entropy_sum = 0.0
        agent._rollout_entropy_n = 0
        state_seq = np.zeros((1, 1, 1, 8, 8), dtype=np.uint8)
        ram_seq = np.zeros((1, 1, RAM_OBS_DIM), dtype=np.float32)
        ram_seq[0, 0, RAM_FEATURE_INDEX["stagnation_clock"]] = 1.0
        mems = [torch.zeros(1, 4, 3)]
        agent._collect_rollout(
            FakeEnv(),
            state_seq,
            ram_seq,
            mems,
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.int64),
        )

        action = int(agent.memory.stored["actions"][0])
        behaviour = 0.5 * np.array([0.9, 0.1]) + 0.5 * np.array([0.5, 0.5])
        expected_log_prob = np.log(behaviour[action] + 1e-10)
        self.assertAlmostEqual(
            float(agent.memory.stored["log_probs"][0]), expected_log_prob, places=6
        )
        expected_base_entropy = -sum(np.array([0.9, 0.1]) * np.log([0.9, 0.1]))
        expected_behaviour_entropy = -sum(behaviour * np.log(behaviour))
        self.assertAlmostEqual(agent._rollout_entropy_sum, expected_base_entropy, places=6)
        self.assertAlmostEqual(
            agent._rollout_behaviour_entropy_sum, expected_behaviour_entropy, places=6
        )
        self.assertTrue(np.allclose(agent.memory.stored["base_probs"][0, :2], [0.9, 0.1]))
        self.assertTrue(np.allclose(agent.memory.stored["behaviour_probs"][0, :2], behaviour))

    def test_unchanged_high_stagnation_policy_has_unit_ratio_and_zero_kl(self):
        torch.manual_seed(4)
        model = PPOModel((1, 8, 8), 9, _model_config())
        model.actor_critic.eval()
        states = torch.randn(2, 3, 1, 8, 8)
        ram = torch.zeros(2, 3, RAM_OBS_DIM)
        ram[:, :, RAM_FEATURE_INDEX["stagnation_clock"]] = 1.0
        mems = model.init_mems(batch_size=2)

        with torch.no_grad():
            mask = model._action_mask_for(ram)
            base_probs, values, _ = model.actor_critic(states, ram, mems, action_mask=mask)
            base_probs = model._normalise_base_policy(base_probs, mask)
            probs = model._behaviour_policy(base_probs, ram, mask)
            actions = probs.argmax(dim=-1)
            old_log_probs = torch.log(probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1) + 1e-10)
            replay_base_probs, _, _ = model.actor_critic(
                states, ram, mems, action_mask=model._action_mask_for(ram)
            )
            replay_base_probs = model._normalise_base_policy(replay_base_probs, mask)
            replay_probs = model._behaviour_policy(replay_base_probs, ram, mask)
            new_log_probs = torch.log(
                replay_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1) + 1e-10
            )

        ratio = torch.exp(new_log_probs - old_log_probs)
        self.assertTrue(torch.equal(ratio, torch.ones_like(ratio)))
        data = {
            "states": states,
            "ram_states": ram,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "returns": values.squeeze(-1),
            "advantages": torch.zeros_like(old_log_probs),
            "old_values": values.squeeze(-1),
            "mems": mems,
            "old_base_probs": base_probs,
            "old_behaviour_probs": probs,
        }
        _, _, _, approx_kl = model._compute_ppo_losses(data, step=0)
        self.assertAlmostEqual(approx_kl, 0.0, places=8)
        base_kl, behaviour_kl = model.exact_policy_kls(data)
        self.assertAlmostEqual(base_kl, 0.0, places=8)
        self.assertAlmostEqual(behaviour_kl, 0.0, places=8)

    def test_temperature_reshape_endpoints_and_mask(self):
        model = PPOModel((1, 8, 8), 9, _model_config())
        probs = torch.tensor([[[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        ram = torch.zeros(1, 1, RAM_OBS_DIM)

        unchanged = model._behaviour_policy(probs, ram, mask)
        self.assertTrue(torch.allclose(unchanged, probs))

        # Fixture's stuck_action_temperature=1.0 -> T=2 at clock=1, i.e.
        # sqrt reshaping: sqrt(0.9):sqrt(0.1) == 3:1 -> normalises to 0.75/0.25.
        ram[..., RAM_FEATURE_INDEX["stagnation_clock"]] = 1.0
        reshaped = model._behaviour_policy(probs, ram, mask)
        expected = torch.tensor([[[0.75, 0.25, 0, 0, 0, 0, 0, 0, 0]]])
        self.assertTrue(torch.allclose(reshaped, expected, atol=1e-6))
        self.assertEqual(float(reshaped[..., 2:].sum()), 0.0)
        self.assertAlmostEqual(float(reshaped.sum()), 1.0, places=6)
        # Softened toward uniform but still ordered like the base policy.
        self.assertGreater(float(reshaped[..., 0]), float(reshaped[..., 1]))
        self.assertLess(float(reshaped[..., 0]), float(probs[..., 0]))

        reshaped_2d = model._behaviour_policy(probs[:, 0], ram, mask)
        self.assertTrue(torch.allclose(reshaped_2d, expected[:, 0], atol=1e-6))

    def test_onset_exponent_defaults_to_linear_ramp(self):
        """Omitting stuck_temperature_onset_exponent must reproduce the
        original linear clock->temperature ramp exactly (backward compat)."""
        cfg = _model_config()
        self.assertNotIn("stuck_temperature_onset_exponent", cfg)
        model = PPOModel((1, 8, 8), 9, cfg)
        self.assertAlmostEqual(model.stuck_temperature_onset_exponent, 1.0)

    def test_onset_exponent_front_loads_exploration_pressure(self):
        """A sub-linear exponent (<1) must reach a GIVEN temperature at a
        LOWER stagnation clock than the default linear ramp — the whole
        point of front-loading exploration pressure earlier in the
        stagnation window."""
        cfg = _model_config()
        cfg["stuck_action_temperature"] = 9.0
        linear_cfg = dict(cfg, stuck_temperature_onset_exponent=1.0)
        sublinear_cfg = dict(cfg, stuck_temperature_onset_exponent=0.5)

        probs = torch.tensor([[[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        ram = torch.zeros(1, 1, RAM_OBS_DIM)
        ram[..., RAM_FEATURE_INDEX["stagnation_clock"]] = 0.25

        linear_model = PPOModel((1, 8, 8), 9, linear_cfg)
        sublinear_model = PPOModel((1, 8, 8), 9, sublinear_cfg)
        linear_reshaped = linear_model._behaviour_policy(probs, ram, mask)
        sublinear_reshaped = sublinear_model._behaviour_policy(probs, ram, mask)

        # Sub-linear reaches a higher effective temperature at the same
        # (partial) clock value, so its distribution is pushed further
        # toward uniform than the linear ramp's.
        self.assertLess(float(sublinear_reshaped[..., 0]), float(linear_reshaped[..., 0]))

    def test_get_action_and_compute_log_prob_use_same_behaviour_policy(self):
        torch.manual_seed(9)
        model = PPOModel((1, 8, 8), 9, _model_config())
        state = np.zeros((1, 1, 8, 8), dtype=np.float32)
        ram = np.zeros((1, RAM_OBS_DIM), dtype=np.float32)
        ram[0, RAM_FEATURE_INDEX["stagnation_clock"]] = 1.0

        action, sampled_log_prob, _ = model.get_action(state, ram, model.init_mems(batch_size=1))
        recomputed_log_prob = model.compute_log_prob(
            state, ram, action, model.init_mems(batch_size=1)
        )

        self.assertAlmostEqual(sampled_log_prob, recomputed_log_prob, places=7)

        base_action, base_log_prob, _ = model.get_action(
            state, ram, model.init_mems(batch_size=1), assisted=False
        )
        recomputed_base_log_prob = model.compute_log_prob(
            state,
            ram,
            base_action,
            model.init_mems(batch_size=1),
            assisted=False,
        )
        self.assertAlmostEqual(base_log_prob, recomputed_base_log_prob, places=7)

    def test_paired_probe_restores_training_rng_and_reuses_probe_seed(self):
        agent = VecPPOAgent.__new__(VecPPOAgent)
        agent.device = torch.device("cpu")
        agent.total_rollouts = 400
        draws = []

        def fake_pass(env, *_args, **_kwargs):
            torch.manual_seed(agent.total_rollouts)
            draws.append(float(torch.rand(())))
            return env

        agent._run_probe_pass = fake_pass
        torch.manual_seed(123)
        before = torch.random.get_rng_state()
        expected = float(torch.rand(()))
        torch.random.set_rng_state(before)

        returned = agent._run_probe_pair(
            env="probe-env",
            probe_config={},
            episode_length=1,
            n_episodes=1,
            base_keys=("a", "b", "c", "d", "e"),
            assisted_keys=("f", "g", "h", "i", "j"),
            label="test",
        )

        self.assertEqual(returned, "probe-env")
        self.assertEqual(draws[0], draws[1])
        self.assertEqual(float(torch.rand(())), expected)

    def test_replay_log_diagnostic_matches_collection_for_tiny_probability(self):
        base = torch.tensor([[[0.99999999, 0.00000001]]], dtype=torch.float32)
        action = torch.tensor([[1]])
        collected_log_prob = torch.log(base[..., 1] + 1e-10)
        data = {
            "actions": action,
            "old_log_probs": collected_log_prob,
            "old_base_probs": base,
            "old_behaviour_probs": base,
        }
        stub = types.SimpleNamespace(
            policy_distributions=lambda _data: (base, base),
            exact_categorical_kl=PPOModel.exact_categorical_kl,
        )

        diagnostics = PPOModel.replay_policy_diagnostics(stub, data)

        self.assertLess(diagnostics["action_log_prob_abs_max"], 1e-7)
        self.assertLess(diagnostics["action_log_prob_abs_mean"], 1e-7)

    def test_lr_multiplier_survives_scheduler_and_checkpoint(self):
        model = PPOModel((1, 8, 8), 9, _model_config())
        self.assertAlmostEqual(model.halve_lr_multiplier(), 1.5e-4)
        model.step_scheduler()
        self.assertAlmostEqual(model.effective_lr(), 1.5e-4)
        for _ in range(5):
            model.halve_lr_multiplier()
        self.assertAlmostEqual(model.effective_lr(), 1e-5)

        with tempfile.TemporaryDirectory() as path:
            model.save(path)
            restored = PPOModel((1, 8, 8), 9, _model_config())
            restored.load(path)
            self.assertAlmostEqual(restored.effective_lr(), 1e-5)
            restored.step_scheduler()
            self.assertAlmostEqual(restored.effective_lr(), 1e-5)


class TestTransformerTailMemory(unittest.TestCase):
    def test_segmented_replay_matches_sequential_collection_with_age_memory(self):
        """Replay must reset only the env rows that end inside a segment."""
        torch.manual_seed(17)
        config = _model_config()
        config.update({"action_mask_enabled": False, "mem_len": 4})
        model = PPOModel((1, 8, 8), 9, config)
        model.actor_critic.eval()
        # A nonzero age embedding exposes the former chunk-relative-age bug.
        with torch.no_grad():
            model.actor_critic.transformer_blocks[0].age_emb.normal_(std=0.1)
        storage = VecPPOMemory(
            {
                "device": "cpu",
                "ppo_update_frequency": 6,
                "sequence_length": 3,
                "input_shape": (1, 8, 8),
                "ram_obs_dim": RAM_OBS_DIM,
                "action_size": 9,
            },
            num_envs=2,
        )
        images = torch.randint(0, 256, (7, 2, 1, 8, 8), dtype=torch.uint8)
        ram = torch.randn(7, 2, RAM_OBS_DIM)
        mems = model.init_mems(2)
        with torch.no_grad():
            for t in range(6):
                incoming = [layer.clone() for layer in mems]
                state = images[t].float().unsqueeze(1)
                ram_state = ram[t].unsqueeze(1)
                mask = model._action_mask_for(ram_state)
                base_seq, _, mems = model.actor_critic(state, ram_state, mems, mask)
                base_seq = model._normalise_base_policy(base_seq, None)
                behaviour_seq = model._behaviour_policy(base_seq, ram_state, None)
                base = base_seq[:, 0]
                behaviour = behaviour_seq[:, 0]
                action = behaviour.argmax(dim=-1)
                log_prob = torch.log(behaviour.gather(1, action[:, None]).squeeze(1))
                # Natural and truncated boundaries both reset recurrent history.
                # Both are strictly inside a segment and occur in different envs.
                done = np.array([t == 1, t == 4])
                truncated = np.array([False, t == 4])
                for layer in mems:
                    layer[torch.from_numpy(done)] = 0
                storage.store_step(
                    states=images[t].numpy(),
                    ram_states=ram[t].numpy(),
                    next_states=images[t + 1].numpy(),
                    next_ram_states=ram[t + 1].numpy(),
                    actions=action.numpy(),
                    rewards=np.zeros(2, dtype=np.float32),
                    dones=done,
                    truncated=truncated,
                    log_probs=log_prob.numpy(),
                    base_probs=base.numpy(),
                    behaviour_probs=behaviour.numpy(),
                    mems=incoming,
                    outgoing_mems=mems,
                )

        data = storage.get_data()
        flat = {
            "states": data["states"].reshape(4, 3, 1, 8, 8),
            "ram_states": data["ram_states"].reshape(4, 3, RAM_OBS_DIM),
            "actions": data["actions"].reshape(4, 3),
            "dones": data["dones"].reshape(4, 3),
            "old_log_probs": data["old_log_probs"].reshape(4, 3),
            "old_base_probs": data["old_base_probs"].reshape(4, 3, 9),
            "old_behaviour_probs": data["old_behaviour_probs"].reshape(4, 3, 9),
            "mems": [m.reshape(4, 4, 16) for m in data["mems"]],
        }
        diagnostic = model.replay_policy_diagnostics(flat)
        self.assertLess(diagnostic["base_replay_kl"], 1e-7)
        self.assertLess(diagnostic["behaviour_replay_kl"], 1e-7)
        self.assertLess(diagnostic["action_log_prob_abs_max"], 1e-6)

        replay_base, _ = model.policy_distributions(flat)
        fresh_mems = model.init_mems(1)
        fresh_base, _, _ = model.actor_critic(
            images[2, 0:1].float().unsqueeze(1),
            ram[2, 0:1].unsqueeze(1),
            fresh_mems,
        )
        self.assertTrue(torch.allclose(replay_base[0, 2], fresh_base[0, 0], atol=1e-7))

    def test_tail_evaluation_matches_sequential_inference(self):
        torch.manual_seed(7)
        model = PPOTransformer(
            input_shape=(1, 8, 8),
            action_size=9,
            ram_dim=5,
            d_model=16,
            d_ram=8,
            n_heads=2,
            num_layers=1,
            mem_len=4,
            dropout=0.0,
        )
        model.eval()
        images = torch.randint(0, 256, (1, 4, 1, 8, 8), dtype=torch.uint8).float()
        ram = torch.randn(1, 4, 5)
        storage = VecPPOMemory(
            {
                "device": "cpu",
                "ppo_update_frequency": 3,
                "sequence_length": 3,
                "input_shape": (1, 8, 8),
                "ram_obs_dim": 5,
            },
            num_envs=1,
        )
        mems = model.init_mems(1, torch.device("cpu"))

        with torch.no_grad():
            for t in range(3):
                incoming_mems = mems
                _, _, mems = model(images[:, t : t + 1], ram[:, t : t + 1], mems)
                storage.store_step(
                    states=images[:, t].numpy().astype(np.uint8),
                    ram_states=ram[:, t].numpy(),
                    next_states=images[:, t + 1].numpy().astype(np.uint8),
                    next_ram_states=ram[:, t + 1].numpy(),
                    actions=np.zeros(1, dtype=np.int64),
                    rewards=np.zeros(1, dtype=np.float32),
                    dones=np.zeros(1, dtype=bool),
                    log_probs=np.zeros(1, dtype=np.float32),
                    base_probs=np.full((1, 9), 1.0 / 9, dtype=np.float32),
                    behaviour_probs=np.full((1, 9), 1.0 / 9, dtype=np.float32),
                    mems=incoming_mems,
                    outgoing_mems=mems,
                )

            data = storage.get_data()
            tail_probs, tail_value, _ = model(
                data["last_next_obs"].unsqueeze(1),
                data["last_next_ram"].unsqueeze(1),
                data["tail_mems"],
            )

            reference_mems = model.init_mems(1, torch.device("cpu"))
            for t in range(4):
                ref_probs, ref_value, reference_mems = model(
                    images[:, t : t + 1], ram[:, t : t + 1], reference_mems
                )

        self.assertTrue(torch.allclose(tail_probs, ref_probs, atol=1e-6))
        self.assertTrue(torch.allclose(tail_value, ref_value, atol=1e-6))


class TestTruncationGAE(unittest.TestCase):
    @staticmethod
    def _agent():
        agent = VecPPOAgent.__new__(VecPPOAgent)
        agent.gamma = 0.9
        agent.gae_lambda = 1.0
        agent.use_gae = True
        return agent

    def test_multiple_internal_truncations_use_terminal_not_reset_values(self):
        rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        values = torch.tensor([[0.0], [100.0], [200.0], [300.0], [400.0]])
        dones = torch.tensor([[False], [True], [False], [True], [False]])
        truncated = torch.tensor([[False], [True], [False], [True], [False]])
        terminal_values = torch.tensor([[0.0], [10.0], [0.0], [20.0], [0.0]])

        returns, _ = self._agent()._per_env_gae(
            rewards,
            values,
            dones,
            tail_values=torch.tensor([30.0]),
            truncated=truncated,
            terminal_bootstrap_values=terminal_values,
        )
        self.assertAlmostEqual(returns[1, 0].item(), 2.0 + 0.9 * 10.0, places=5)
        self.assertAlmostEqual(returns[3, 0].item(), 4.0 + 0.9 * 20.0, places=5)

        changed_reset_values = values.clone()
        changed_reset_values[2, 0] = -9999.0
        changed_reset_values[4, 0] = 9999.0
        changed, _ = self._agent()._per_env_gae(
            rewards,
            changed_reset_values,
            dones,
            tail_values=torch.tensor([30.0]),
            truncated=truncated,
            terminal_bootstrap_values=terminal_values,
        )
        self.assertEqual(changed[1, 0].item(), returns[1, 0].item())
        self.assertEqual(changed[3, 0].item(), returns[3, 0].item())

    def test_non_budget_terminal_does_not_bootstrap(self):
        returns, _ = self._agent()._per_env_gae(
            rewards=torch.tensor([[2.0]]),
            values=torch.tensor([[5.0]]),
            dones=torch.tensor([[True]]),
            tail_values=torch.tensor([100.0]),
            truncated=torch.tensor([[False]]),
            terminal_bootstrap_values=torch.tensor([[50.0]]),
        )
        self.assertAlmostEqual(returns[0, 0].item(), 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
