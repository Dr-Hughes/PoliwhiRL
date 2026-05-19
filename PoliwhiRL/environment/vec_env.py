# -*- coding: utf-8 -*-
"""Process-based vectorised wrapper around PyBoyEnvironment.

Each worker process owns one PyBoyEnvironment instance with its own
per-instance temp directory (handled by the env itself). Workers auto-reset
on done so the main loop can collect fixed-length rollouts across N envs
without having to track episode boundaries inside the wrapper.

Uses the 'spawn' multiprocessing context for portability (macOS default,
also safer than 'fork' for libraries that initialise SDL/threads on import).
"""
import multiprocessing as mp
import traceback
import numpy as np

from PoliwhiRL.environment.gym_env import PyBoyEnvironment


def _worker(remote, config, env_idx):
    """Subprocess entry. Owns a single env and serves command messages.

    Protocol:
      ("init", None)         -> ("init_ok", (output_shape, action_size)) | ("error", tb)
      ("reset", None)        -> ("ok", obs)
      ("step", action)       -> ("ok", (obs, reward, done))   # auto-resets on done
      ("enable_record", (folder, use_ep_num))
                             -> ("ok", None)
      ("close", None)        -> ("ok", None)  and worker exits
    """
    env = None
    try:
        env = PyBoyEnvironment(config)
        remote.send(("init_ok", (env.output_shape(), env.action_space.n)))

        while True:
            cmd, payload = remote.recv()
            if cmd == "reset":
                obs = env.reset()
                remote.send(("ok", obs))
            elif cmd == "step":
                obs, reward, done, _ = env.step(int(payload))
                if done:
                    obs = env.reset()
                remote.send(("ok", (obs, float(reward), bool(done))))
            elif cmd == "enable_record":
                folder, use_ep_num = payload
                env.enable_record(folder, use_ep_num)
                remote.send(("ok", None))
            elif cmd == "close":
                remote.send(("ok", None))
                break
            else:
                remote.send(("error", f"unknown cmd {cmd!r}"))
    except Exception:
        try:
            remote.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            remote.close()
        except Exception:
            pass


class VecPyBoyEnv:
    """Vectorised PyBoy env with auto-reset.

    Use like:
        vec = VecPyBoyEnv(config, num_envs=4)
        obs = vec.reset()                    # (N, *obs_shape)
        obs, rew, done = vec.step(actions)   # actions shape (N,)
        vec.close()
    """

    def __init__(self, config, num_envs):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = num_envs
        self.config = config
        self._closed = False

        ctx = mp.get_context("spawn")
        self._remotes = []
        self._workers = []

        for i in range(num_envs):
            parent_remote, child_remote = ctx.Pipe()
            worker = ctx.Process(
                target=_worker, args=(child_remote, config, i), daemon=True
            )
            worker.start()
            child_remote.close()
            self._remotes.append(parent_remote)
            self._workers.append(worker)

        shapes = []
        action_sizes = []
        for remote in self._remotes:
            tag, payload = remote.recv()
            if tag != "init_ok":
                self._hard_terminate()
                raise RuntimeError(f"Vec env worker init failed:\n{payload}")
            shape, asize = payload
            shapes.append(shape)
            action_sizes.append(asize)

        if len(set(action_sizes)) != 1:
            self._hard_terminate()
            raise RuntimeError(f"Workers disagree on action size: {action_sizes}")
        if len(set(tuple(s) for s in shapes)) != 1:
            self._hard_terminate()
            raise RuntimeError(f"Workers disagree on output shape: {shapes}")

        self._output_shape = shapes[0]
        self._action_size = action_sizes[0]

    def output_shape(self):
        return self._output_shape

    @property
    def action_size(self):
        return self._action_size

    def reset(self):
        for remote in self._remotes:
            remote.send(("reset", None))
        return np.stack([self._recv_ok(remote) for remote in self._remotes])

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError(
                f"step expected {self.num_envs} actions, got {len(actions)}"
            )
        for remote, action in zip(self._remotes, actions):
            remote.send(("step", int(action)))
        obs_list, rew_list, done_list = [], [], []
        for remote in self._remotes:
            obs, reward, done = self._recv_ok(remote)
            obs_list.append(obs)
            rew_list.append(reward)
            done_list.append(done)
        return (
            np.stack(obs_list),
            np.asarray(rew_list, dtype=np.float32),
            np.asarray(done_list, dtype=bool),
        )

    def enable_record(self, folder, use_episode_number=True, env_idx=0):
        """Turn on per-step image recording for one env (usually #0 for low cost)."""
        if not (0 <= env_idx < self.num_envs):
            raise IndexError(env_idx)
        self._remotes[env_idx].send(
            ("enable_record", (folder, use_episode_number))
        )
        self._recv_ok(self._remotes[env_idx])

    def close(self):
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for remote in self._remotes:
            try:
                remote.recv()
            except (EOFError, OSError):
                pass
        for worker in self._workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
        for remote in self._remotes:
            try:
                remote.close()
            except OSError:
                pass

    def __del__(self):
        # Belt-and-braces; explicit close() is still preferred.
        try:
            self.close()
        except Exception:
            pass

    def _recv_ok(self, remote):
        tag, payload = remote.recv()
        if tag == "error":
            self._hard_terminate()
            raise RuntimeError(f"Vec env worker raised:\n{payload}")
        if tag != "ok":
            self._hard_terminate()
            raise RuntimeError(f"Unexpected vec env reply: {tag}")
        return payload

    def _hard_terminate(self):
        for worker in self._workers:
            try:
                worker.terminate()
            except Exception:
                pass
        for worker in self._workers:
            try:
                worker.join(timeout=2)
            except Exception:
                pass
