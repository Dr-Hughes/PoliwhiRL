# -*- coding: utf-8 -*-
"""Process-based vectorised wrapper around PyBoyEnvironment.

Each worker process owns one PyBoyEnvironment instance with its own
per-instance temp directory (handled by the env itself). Workers auto-reset
on done so the main loop can collect fixed-length rollouts across N envs
without having to track episode boundaries inside the wrapper.

Uses the 'spawn' multiprocessing context for portability (macOS default,
also safer than 'fork' for libraries that initialise SDL/threads on import).
"""
import glob
import multiprocessing as mp
import os
import random
import traceback
import numpy as np

from PoliwhiRL.environment.gym_env import (
    PyBoyEnvironment,
    N_LOC_GOALS_RAM_IDX,
    N_POK_GOALS_RAM_IDX,
)


_TRAJECTORY_MARKER = "# trajectory"


def _load_actions_file(path):
    """Read a `.steps` file and return a list of trajectories.

    Supports two formats in the same file:
      (a) Single-trajectory: just one int per line, optional `#` comments
          and blank lines. The whole file becomes one trajectory.
      (b) Multi-trajectory: trajectory blocks delimited by lines beginning
          with `# trajectory` (case-sensitive). Each block contributes one
          trajectory to the returned list.

    Missing files are tolerated with a warning — replay just becomes a
    no-op (returns []).
    """
    if not os.path.isfile(path):
        print(f"[VecPyBoyEnv] action_replay file not found, skipping: {path}")
        return []

    trajectories = []
    current = None  # None until we see content or a marker

    def _flush():
        nonlocal current
        if current is not None and len(current) > 0:
            trajectories.append(current)
        current = None

    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(_TRAJECTORY_MARKER):
                _flush()
                current = []
                continue
            if not stripped or stripped.startswith("#"):
                continue
            if current is None:
                current = []
            current.append(int(stripped))
    _flush()
    return trajectories


def write_actions_file(path, trajectories, metadata=None):
    """Write a list of action trajectories to a `.steps` file using the
    multi-trajectory format. `metadata` (optional) is a list parallel to
    `trajectories`; each entry is rendered as a `# key=value` line after
    the trajectory header.
    """
    trajectories = [list(t) for t in trajectories if t]
    if not trajectories:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for i, traj in enumerate(trajectories):
            f.write(f"{_TRAJECTORY_MARKER} {i}\n")
            if metadata is not None and i < len(metadata) and metadata[i]:
                for k, v in metadata[i].items():
                    f.write(f"# {k}={v}\n")
            for a in traj:
                f.write(f"{int(a)}\n")


def _worker(remote, config, env_idx):
    """Subprocess entry. Owns a single env and serves command messages.

    Protocol:
      ("init", None)               -> ("init_ok", (image_shape, ram_dim, action_size)) | ("error", tb)
      ("reset", None)              -> ("ok", obs_dict)
      ("step", action)             -> ("ok", (obs_dict, reward, done, terminal_info))   # auto-resets on done
                                       `terminal_info` is None when done is False; on done it is
                                       (n_location_goals_completed, n_pokedex_goals_completed,
                                        N_goals_target) captured just before auto-reset clobbers
                                       the env. Needed by the agent because the returned obs_dict
                                       is the post-reset observation (the terminal obs is lost).
      ("set_state_path", path)     -> ("ok", None)
                                       Takes effect on the next reset (including auto-reset on done).
      ("set_replay_pool", list_of_lists)
                                   -> ("ok", None)
                                       Replaces the worker's replay-trajectory pool. Pass [] to disable
                                       replay entirely. On every (auto-)reset the worker samples one
                                       trajectory uniformly from the pool and a cutoff k uniformly in
                                       [0, len(trajectory)], then replays trajectory[:k].
      ("enable_record", (folder, use_ep_num))
                                   -> ("ok", None)
      ("close", None)              -> ("ok", None) and worker exits

    Replay invariant: after env.reset() (whether explicit or auto on done),
    the worker walks the env through a randomly-sampled prefix of a
    randomly-sampled trajectory from `replay_pool`. The Rewards object
    advances naturally through goals; the training agent never sees the
    replay transitions — it only sees the post-replay observation as the
    "first step" of its episode.
    """
    env = None
    replay_pool = []  # list[list[int]]
    # Per-worker RNG seeded from env_idx so each worker explores a
    # different sequence of (trajectory, cutoff) samples even with the
    # same pool, but training across runs stays reproducible.
    rng = random.Random(0xC0FFEE ^ env_idx)
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
            if replay_pool:
                traj = replay_pool[rng.randrange(len(replay_pool))]
                # Uniform-cutoff prefix replay. Endpoint inclusive so the
                # full trajectory is one of the possible draws.
                k = rng.randint(0, len(traj))
                if k > 0:
                    env.replay_actions(traj[:k])
            return env.get_observation()

        while True:
            cmd, payload = remote.recv()
            if cmd == "reset":
                obs = do_reset()
                remote.send(("ok", obs))
            elif cmd == "step":
                obs, reward, done, _ = env.step(int(payload))
                terminal_info = None
                if done:
                    # Snapshot terminal progress before the auto-reset
                    # replaces obs with the post-reset observation.
                    rc = env.reward_calculator
                    terminal_info = (
                        int(rc.n_location_goals_completed()),
                        int(rc.n_pokedex_goals_completed()),
                        int(rc.N_goals_target),
                    )
                    obs = do_reset()
                remote.send(("ok", (obs, float(reward), bool(done), terminal_info)))
            elif cmd == "set_state_path":
                env.set_state_path(payload)
                remote.send(("ok", None))
            elif cmd == "set_replay_pool":
                replay_pool = [list(t) for t in payload] if payload else []
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


def _load_replay_pool(paths):
    """Resolve `action_replay_paths` to a flat list of trajectories.

    Each path is glob-expanded; each resolved file is parsed into one or
    more trajectories (see `_load_actions_file`); the trajectories from
    all files are concatenated. Workers sample uniformly from this pool.
    Returns (expanded_paths, trajectories).
    """
    expanded = []
    for p in paths or []:
        matched = glob.glob(p)
        if matched:
            expanded.extend(sorted(matched))
        else:
            expanded.append(p)  # keep so load can warn

    trajectories = []
    for path in expanded:
        trajectories.extend(_load_actions_file(path))
    return expanded, trajectories


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

    Action-replay pool: workers each get the SAME trajectory pool
    (concatenated trajectories from all `action_replay_paths` files). On
    each reset, every worker independently samples (trajectory, cutoff)
    uniformly. There is no per-env cycling state — diversity comes from
    the random sampling itself.
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

        # Action-replay pool. Trajectories from every configured `.steps`
        # file are flattened into a single pool; each worker samples
        # uniformly from this pool on each (auto-)reset and applies a
        # uniformly-sampled prefix. No round-robin / cycling — randomness
        # drives diversity instead.
        self.action_replay_paths, self.replay_trajectories = _load_replay_pool(
            config.get("action_replay_paths") or []
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

        # Push the replay pool to workers. Skipped silently when no pool
        # was configured (replay_trajectories is empty, replay is a no-op).
        if self.replay_trajectories:
            self._broadcast_replay_pool()

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
        """Step all envs in lock-step.

        Returns
        -------
        obs : dict of stacked arrays
        rewards : (N,) float32
        dones : (N,) bool
        terminal_infos : list[Optional[tuple]] length N
            For envs that finished an episode on this step, contains
            (n_location_goals_completed, n_pokedex_goals_completed,
             N_goals_target) captured at the terminal observation. None
            for envs that did not finish. The agent reads these to log
            curriculum progress; without them the post-reset obs would
            be the only thing visible after a done.
        """
        if len(actions) != self.num_envs:
            raise ValueError(
                f"step expected {self.num_envs} actions, got {len(actions)}"
            )
        for remote, action in zip(self._remotes, actions):
            remote.send(("step", int(action)))
        obs_list, rew_list, done_list, terminal_infos = [], [], [], []
        for remote in self._remotes:
            obs, reward, done, terminal_info = self._recv_ok(remote)
            obs_list.append(obs)
            rew_list.append(reward)
            done_list.append(done)
            terminal_infos.append(terminal_info)
        return (
            self._stack_obs(obs_list),
            np.asarray(rew_list, dtype=np.float32),
            np.asarray(done_list, dtype=bool),
            terminal_infos,
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

    def set_replay_pool(self, trajectories):
        """Replace the in-memory replay pool and broadcast it to every
        worker. Takes effect on each worker's next reset (the running
        episode is unaffected). Pass [] to disable replay entirely.
        """
        self.replay_trajectories = [list(t) for t in trajectories if t]
        self._broadcast_replay_pool()

    def _broadcast_replay_pool(self):
        for remote in self._remotes:
            remote.send(("set_replay_pool", self.replay_trajectories))
        for remote in self._remotes:
            self._recv_ok(remote)

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
