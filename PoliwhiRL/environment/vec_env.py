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
import os
import traceback
import numpy as np

from PoliwhiRL.environment.gym_env import PyBoyEnvironment


def _load_actions_file(path):
    """Read a .steps file (one integer action per line, optional blank lines
    and #-prefixed comments tolerated). Returns a list[int] of actions.
    Missing files are tolerated with a warning — replay just becomes a no-op.
    """
    if not os.path.isfile(path):
        print(f"[VecPyBoyEnv] action_replay file not found, skipping: {path}")
        return []
    actions = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            actions.append(int(stripped))
    return actions


def _worker(remote, config, env_idx):
    """Subprocess entry. Owns a single env and serves command messages.

    Protocol:
      ("init", None)               -> ("init_ok", (image_shape, ram_dim, action_size)) | ("error", tb)
      ("reset", None)              -> ("ok", obs_dict)
      ("step", action)             -> ("ok", (obs_dict, reward, done))   # auto-resets on done
      ("set_state_path", path)     -> ("ok", None)
                                       Takes effect on the next reset (including auto-reset on done).
      ("set_replay_actions", list) -> ("ok", None)
                                       Sets the replay action sequence applied after every reset.
                                       Pass None or [] to disable replay.
      ("enable_record", (folder, use_ep_num))
                                   -> ("ok", None)
      ("close", None)              -> ("ok", None) and worker exits

    Replay invariant: after env.reset() (whether explicit or auto on done),
    the worker walks the env through `replay_actions` so the Rewards object
    is positioned at the right curriculum point. The training agent never
    sees the replay transitions — it only sees the post-replay observation
    as the "first step" of its episode.
    """
    env = None
    replay_actions = []
    try:
        env = PyBoyEnvironment(config)
        remote.send(
            (
                "init_ok",
                (env.output_shape(), env.ram_observation_shape()[0], env.action_space.n),
            )
        )

        def do_reset():
            env.reset()
            if replay_actions:
                env.replay_actions(replay_actions)
            return env.get_observation()

        while True:
            cmd, payload = remote.recv()
            if cmd == "reset":
                obs = do_reset()
                remote.send(("ok", obs))
            elif cmd == "step":
                obs, reward, done, _ = env.step(int(payload))
                if done:
                    obs = do_reset()
                remote.send(("ok", (obs, float(reward), bool(done))))
            elif cmd == "set_state_path":
                env.set_state_path(payload)
                remote.send(("ok", None))
            elif cmd == "set_replay_actions":
                replay_actions = list(payload) if payload else []
                remote.send(("ok", None))
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
    """Vectorised PyBoy env with auto-reset and dict observations.

    Use like:
        vec = VecPyBoyEnv(config, num_envs=4)
        obs = vec.reset()                       # {"image": (N, C, H, W), "ram": (N, D)}
        obs, rew, done = vec.step(actions)      # actions shape (N,)
        vec.close()

    Multi-state pool: when config supplies `state_paths` (list of save-state
    file paths), workers are assigned states round-robin at init. Subsequent
    `set_env_state(env_idx, path)` calls cycle a worker's state on its next
    reset. `state_indices` is exposed so the agent can tag per-episode
    metrics with the state each env is currently running.
    """

    def __init__(self, config, num_envs):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = num_envs
        self.config = config
        self._closed = False

        # Resolve the state pool. A single `state_path` (legacy) is still
        # supported and treated as a one-element pool.
        state_paths = config.get("state_paths")
        if not state_paths:
            state_paths = [config["state_path"]]
        self.state_paths = list(state_paths)

        # Round-robin: worker i starts with state_paths[i % len(pool)].
        self.state_indices = [i % len(self.state_paths) for i in range(num_envs)]

        # Action-replay pool. Each entry is a path to a .steps file (one
        # integer per line). Empty -> no replay (workers behave as before).
        # The replay is applied AFTER each env.reset(), warming Rewards
        # forward through the recorded action sequence so the training
        # episode starts at a curriculum-aligned position.
        self.action_replay_paths = list(config.get("action_replay_paths") or [])
        self.replay_action_sequences = [
            _load_actions_file(p) for p in self.action_replay_paths
        ]
        # If no replay pool was configured, replay_indices stays empty and
        # replay assignment is a no-op. Otherwise round-robin assign.
        self.replay_indices = (
            [i % len(self.action_replay_paths) for i in range(num_envs)]
            if self.action_replay_paths
            else [-1] * num_envs
        )

        ctx = mp.get_context("spawn")
        self._remotes = []
        self._workers = []

        for i in range(num_envs):
            parent_remote, child_remote = ctx.Pipe()
            # Each worker boots with its assigned state. We pass a config
            # *copy* with state_path overridden so the worker's env is
            # initialised against the right state file from the start.
            worker_config = dict(config)
            worker_config["state_path"] = self.state_paths[self.state_indices[i]]
            worker = ctx.Process(
                target=_worker, args=(child_remote, worker_config, i), daemon=True
            )
            worker.start()
            child_remote.close()
            self._remotes.append(parent_remote)
            self._workers.append(worker)

        image_shapes = []
        ram_dims = []
        action_sizes = []
        for remote in self._remotes:
            tag, payload = remote.recv()
            if tag != "init_ok":
                self._hard_terminate()
                raise RuntimeError(f"Vec env worker init failed:\n{payload}")
            img_shape, ram_dim, asize = payload
            image_shapes.append(img_shape)
            ram_dims.append(ram_dim)
            action_sizes.append(asize)

        if len(set(action_sizes)) != 1:
            self._hard_terminate()
            raise RuntimeError(f"Workers disagree on action size: {action_sizes}")
        if len(set(tuple(s) for s in image_shapes)) != 1:
            self._hard_terminate()
            raise RuntimeError(f"Workers disagree on image shape: {image_shapes}")
        if len(set(ram_dims)) != 1:
            self._hard_terminate()
            raise RuntimeError(f"Workers disagree on RAM dim: {ram_dims}")

        self._output_shape = image_shapes[0]
        self._ram_dim = ram_dims[0]
        self._action_size = action_sizes[0]

        # Push initial replay assignments to workers. Skipped silently when
        # no replay pool was configured (replay_indices is all -1 in that case).
        if self.action_replay_paths:
            for env_idx in range(self.num_envs):
                actions = self.replay_action_sequences[self.replay_indices[env_idx]]
                self._remotes[env_idx].send(("set_replay_actions", actions))
            for env_idx in range(self.num_envs):
                self._recv_ok(self._remotes[env_idx])

    def output_shape(self):
        return self._output_shape

    def ram_observation_shape(self):
        return (self._ram_dim,)

    @property
    def action_size(self):
        return self._action_size

    def reset(self):
        for remote in self._remotes:
            remote.send(("reset", None))
        obs_list = [self._recv_ok(remote) for remote in self._remotes]
        return self._stack_obs(obs_list)

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
            self._stack_obs(obs_list),
            np.asarray(rew_list, dtype=np.float32),
            np.asarray(done_list, dtype=bool),
        )

    def set_env_state(self, env_idx, state_path):
        """Tell a worker to load a different save-state on its next reset.

        The current episode finishes normally; on its next auto-reset the
        worker swaps in the new state. The agent should also update
        `state_indices[env_idx]` (use set_env_state_index for atomicity).
        """
        if not (0 <= env_idx < self.num_envs):
            raise IndexError(env_idx)
        self._remotes[env_idx].send(("set_state_path", state_path))
        self._recv_ok(self._remotes[env_idx])

    def set_env_state_index(self, env_idx, state_idx):
        """Cycle env `env_idx` to state_paths[state_idx] on its next reset."""
        if not (0 <= state_idx < len(self.state_paths)):
            raise IndexError(state_idx)
        self.set_env_state(env_idx, self.state_paths[state_idx])
        self.state_indices[env_idx] = state_idx

    def set_env_replay_index(self, env_idx, replay_idx):
        """Switch env `env_idx` to replay_action_sequences[replay_idx]. The
        new replay applies starting from the next reset. Mirrors
        set_env_state_index for the action-replay pool."""
        if not self.action_replay_paths:
            raise RuntimeError("No action_replay_paths configured for this VecPyBoyEnv")
        if not (0 <= replay_idx < len(self.action_replay_paths)):
            raise IndexError(replay_idx)
        if not (0 <= env_idx < self.num_envs):
            raise IndexError(env_idx)
        actions = self.replay_action_sequences[replay_idx]
        self._remotes[env_idx].send(("set_replay_actions", actions))
        self._recv_ok(self._remotes[env_idx])
        self.replay_indices[env_idx] = replay_idx

    @staticmethod
    def _stack_obs(obs_list):
        """Stack a list of per-env dict observations into batched dict."""
        return {
            "image": np.stack([o["image"] for o in obs_list]),
            "ram": np.stack([o["ram"] for o in obs_list]),
        }

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
