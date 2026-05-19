# -*- coding: utf-8 -*-
"""Numerically-stable running mean/variance for reward normalization.

Welford's online algorithm with parallel-batch merging. Tracks raw rewards
so PPO can divide by sqrt(var) before GAE — the critic naturally settles
into predicting normalized returns, keeping its target range stable across
curriculum stages that introduce much larger reward magnitudes.

Only `var` is used for scaling; we do not subtract the mean (zero reward
should stay zero — the step penalty and goal reward have distinct
semantic meanings around zero).
"""
import numpy as np


class RunningMeanStd:
    __slots__ = ("mean", "var", "count")

    def __init__(self, epsilon=1e-4):
        # Start var=1 so early-training scaling is a no-op until enough
        # samples have accumulated to make the estimate meaningful.
        self.mean = 0.0
        self.var = 1.0
        # `epsilon` doubles as the initial pseudo-count, preventing
        # division blow-up if the first batch arrives with var=0.
        self.count = float(epsilon)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = float(x.size)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta * delta) * (self.count * batch_count / total)

        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    @property
    def std(self):
        return float(np.sqrt(self.var))

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean = float(state["mean"])
        self.var = float(state["var"])
        self.count = float(state["count"])


class RewardScaler:
    """Tracks running variance of per-env discounted returns, exposes a
    scale factor for normalising rewards before GAE. The critic learns to
    predict normalised returns, keeping its target range stable across
    curriculum stages with very different reward magnitudes.

    Designed to be agnostic to single-env vs vec mode — pass num_envs=1
    for single-env and observe() scalar inputs.
    """

    def __init__(self, gamma, num_envs=1, epsilon=1e-4):
        self.gamma = float(gamma)
        self.num_envs = int(num_envs)
        self.rms = RunningMeanStd(epsilon=epsilon)
        self.running_returns = np.zeros(self.num_envs, dtype=np.float64)

    def observe(self, rewards, dones):
        rewards = np.atleast_1d(np.asarray(rewards, dtype=np.float64)).reshape(-1)
        dones = np.atleast_1d(np.asarray(dones, dtype=bool)).reshape(-1)
        self.running_returns = self.gamma * self.running_returns + rewards
        self.rms.update(self.running_returns)
        self.running_returns[dones] = 0.0

    def scale_factor(self):
        return 1.0 / max(self.rms.std, 1e-8)

    def state_dict(self):
        return {
            "rms": self.rms.state_dict(),
            "running_returns": self.running_returns.tolist(),
        }

    def load_state_dict(self, state):
        self.rms.load_state_dict(state["rms"])
        rr = np.asarray(state.get("running_returns", []), dtype=np.float64)
        n = min(len(rr), self.num_envs)
        if n > 0:
            self.running_returns[:n] = rr[:n]
