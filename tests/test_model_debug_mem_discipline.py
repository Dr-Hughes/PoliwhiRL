# -*- coding: utf-8 -*-
"""T4c mem discipline for debug_evaluator's model-debug loop (no emulator).

The eval loop must forward a SINGLE frame per step and carry the
transformer mems across steps, zeroing them when the episode ends — the
same discipline as model_evaluator.run_evaluator. The pre-fix loop kept a
growing (L, H, W, C) window AND passed the mems, double-counting the last
L-1 frames (v6 audit: KL drift mean 0.090, p95 0.50, 22% of steps >0.1
nats vs the fixed loop; v6_verify_20261101/ws2_results.json).

The loop is Env-coupled (get_observation / _handle_action / recording),
but not irreducibly: it runs against a duck-typed fake env, so this test
drives the REAL _run_model_debug with a tiny real PPOModel (CPU, no
PyBoy) whose actor_critic is wrapped to record every forward. It asserts:
- the image T-dim is exactly 1 at every forward (no window);
- the mems passed at step k+1 are the very list returned by the forward
  at step k (carried, non-None from step 2);
- mems are zeroed after the episode ends and the loop stops stepping.

Distinct from the PyBoy-backed tests/test_debug_eval.py.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import PoliwhiRL.evaluator.debug_evaluator as debug_evaluator
from PoliwhiRL.environment.gym_env import RAM_OBS_DIM
from PoliwhiRL.models.PPO import PPOModel

_H = 8
_W = 8
_A = 9
_N_STEPS = 3


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
    }


class _RecordingAC:
    """Wraps the real actor_critic and records each forward's shapes."""

    def __init__(self, real, recorder, events):
        self._real = real
        self._recorder = recorder
        self._events = events

    def __call__(self, image, ram, mems=None):
        out = self._real(image, ram, mems)
        self._events.append("fwd")
        self._recorder.append(
            {
                "image_shape": tuple(image.shape),
                "ram_shape": tuple(ram.shape),
                "mems": mems,
                "new_mems": out[2],
            }
        )
        return out

    def eval(self):
        return self._real.eval()

    def load_state_dict(self, state_dict):
        return self._real.load_state_dict(state_dict)

    def init_mems(self, batch_size, device):
        return self._real.init_mems(batch_size, device)


class _FakeRC:
    N_goals = 0
    flag_goals_completed = 0


class _FakeEnv:
    """Duck-typed stand-in for PyBoyEnvironment in _run_model_debug."""

    def __init__(self, config, done_step=None):
        self._done_step = done_step
        self.steps = 0
        self._fitness = 0.0
        self.reward_calculator = _FakeRC()

    def output_shape(self):
        return (1, _H, _W)

    def ram_observation_shape(self):
        return (RAM_OBS_DIM,)

    @property
    def action_space(self):
        import types

        return types.SimpleNamespace(n=_A)

    def close(self):
        pass

    def reset(self):
        pass

    def enable_record(self, folder, use_episode_number=False):
        pass

    def save_debug_step_img_data(self, folder, outdir):
        pass

    def finalize_debug_run(self, folder, outdir):
        return "fake_summary.json"

    def _handle_action(self, action):
        self.steps += 1

    def _calculate_fitness(self):
        self._fitness = 0.0

    @property
    def done(self):
        return self._done_step is not None and self.steps >= self._done_step

    def get_observation(self):
        image = np.zeros((_H, _W, 1), dtype=np.float32)
        image[0, 0, 0] = 1.0
        ram = np.zeros(RAM_OBS_DIM, dtype=np.float32)
        return {"image": image, "ram": ram}


class TestModelDebugMemDiscipline(unittest.TestCase):
    def setUp(self):
        self._recorder = []
        self._events = []
        self._real_ppo_model = debug_evaluator.PPOModel
        self._real_env = debug_evaluator.Env
        self._real_zeros_like = torch.zeros_like

    def tearDown(self):
        debug_evaluator.PPOModel = self._real_ppo_model
        debug_evaluator.Env = self._real_env
        torch.zeros_like = self._real_zeros_like

    def _make_checkpoint(self, directory):
        """A real actor_critic.pth for the tiny model the fake builds."""
        model = PPOModel((1, _H, _W), _A, _model_config())
        torch.save(model.actor_critic.state_dict(), directory / "actor_critic.pth")

    def _run_loop(self, checkpoint_dir, done_step=None):
        def fake_ppo_model_factory(input_shape, action_size, config):
            model = self._real_ppo_model(input_shape, action_size, config)
            model.actor_critic = _RecordingAC(model.actor_critic, self._recorder, self._events)
            return model

        def fake_env_factory(config):
            return _FakeEnv(config, done_step=done_step)

        def zeros_like_spy(*args, **kwargs):
            self._events.append("zl")
            return self._real_zeros_like(*args, **kwargs)

        debug_evaluator.PPOModel = fake_ppo_model_factory
        debug_evaluator.Env = fake_env_factory
        torch.zeros_like = zeros_like_spy

        config = _model_config()
        config["load_checkpoint"] = str(checkpoint_dir)
        config["episode_length"] = _N_STEPS
        debug_evaluator._run_model_debug(config)
        return self._recorder

    def test_single_frame_and_carried_mems(self):
        """Every forward gets T==1; mems returned by step k are the very
        list fed into step k+1."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            self._make_checkpoint(Path(tmp))
            rec = self._run_loop(Path(tmp))

        self.assertEqual(len(rec), _N_STEPS)
        for entry in rec:
            # (B, T, H, W, C): the T-dim must be a single frame, never the
            # L-frame window the pre-fix loop built.
            self.assertEqual(entry["image_shape"], (1, 1, _H, _W, 1))
            self.assertEqual(entry["ram_shape"], (1, 1, RAM_OBS_DIM))
        for k in range(1, len(rec)):
            self.assertIsNotNone(rec[k]["mems"])
            self.assertIs(rec[k]["mems"], rec[k - 1]["new_mems"])
        # No done, so the final action of the run is a forward, not the
        # zeroing pass.
        self.assertEqual(self._events[-1], "fwd")

    def test_mems_zeroed_after_done(self):
        """When the episode ends, mems are zeroed and the loop stops."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            self._make_checkpoint(Path(tmp))
            rec = self._run_loop(Path(tmp), done_step=2)

        # Two forwards, then the done break — no third step.
        self.assertEqual(len(rec), 2)
        # The run's final action is the zeroing pass (zeros_like after the
        # last forward), so the episode-end mem reset did happen.
        self.assertEqual(self._events[-1], "zl")
        self.assertEqual(self._events.count("fwd"), 2)


if __name__ == "__main__":
    unittest.main()
