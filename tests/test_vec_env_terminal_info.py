# -*- coding: utf-8 -*-
"""Boot-free tests for the end-of-episode location plumbing.

Covers the three places a "where did this episode end?" answer has to survive:

1. ``vec_env._worker``'s ``terminal_info`` — the four ``final_*`` values must
   come from the PRE-reset state. Reading the post-reset row would record where
   the NEXT episode starts, which for a Go-Explore-seeded worker is a frontier
   snapshot deep in already-solved territory, so a stagnation histogram would
   report restart points instead of stall points.
2. ``VecPPOAgent._commit_episode`` — the two list-valued episode series must
   stay index-aligned with ``episode_truncation_cause`` / ``episode_seeded``,
   including across a resume, where ``load_model``'s allowlist silently drops
   any series missing from its ``fresh`` dict.
3. ``VecPPOAgent._record_rollout_diagnostics`` — ``rollout_accepted_epochs``
   must record NaN ("unmeasured"), never 0.0, when the key is absent.

No emulator, no ROM, no PyBoy boot: ``_worker`` is driven in-process against a
stub env that reproduces the two facts the change depends on (``step`` leaves
the ending step's raw RAM variables in ``_last_env_vars``; ``reset`` overwrites
that same dict), and the agent is built with ``__new__`` + ``reset_tracking``
the way tests/test_checkpoint_progress.py does.

The one thing genuinely unreachable without the emulator is whether the real
``RAM.get_variables()`` keys ``map_bank`` / ``map_num`` / ``X`` / ``Y`` are the
live ones at the terminal step. Those key names are not asserted here; they are
the same four the env already reads at gym_env.py:856-857 for the visit
archive, and tests/test_vec_env.py exercises the booted path end to end.
"""

import ast
import math
import pathlib
import types
from unittest import mock

from PoliwhiRL.agents.PPO import vec_ppo_agent as agent_module
from PoliwhiRL.agents.PPO.vec_ppo_agent import VecPPOAgent
from PoliwhiRL.environment import vec_env


# --------------------------------------------------------------------------
# Worker stubs
# --------------------------------------------------------------------------


class _FakeRewards:
    """Minimal ``Rewards`` stand-in: everything terminal_info reads off it."""

    def __init__(self, truncation_cause="stagnation"):
        self.truncation_cause = truncation_cause
        self.corrupt = False
        self.flag_fire_step_log = [[30, 12]]
        self.goal_fire_steps = [4]
        self._cells_to_record = {(3, 42, 7, 9)}
        self._maps_to_record = {(3, 42)}
        self._ever_visited_cells_this_episode = {(3, 42, 7, 9)}
        self.goals = types.SimpleNamespace(
            _maps_seen_this_episode={(3, 42)},
            all_goal_thresholds_met=lambda: False,
        )

    def n_flag_goals_completed(self):
        return 1

    def n_pokedex_goals_completed(self):
        return 0

    def n_map_goals_completed(self):
        return 0

    def get_episode_breakdown(self):
        return {"frontier": 0.5}

    def stagnation_fraction(self):
        return 0.75

    def get_discoveries(self):
        return []

    def get_milestone_state(self):
        return {
            "flags_fired": [30],
            "pokedex_seen_max": 0,
            "pokedex_owned_max": 0,
            "level_max": 5,
            "key_items_max": 0,
            "milestone_fires": [],
        }

    def get_episode_event_counts(self):
        return {}


class _FakeEnv:
    """``PyBoyEnvironment`` stand-in with a controllable end location."""

    #: raw RAM variables for the step that terminates the episode
    end_location = {"map_bank": 3, "map_num": 42, "X": 7, "Y": 9}
    #: raw RAM variables the auto-reset leaves behind — deliberately different
    restart_location = {"map_bank": 0, "map_num": 1, "X": 5, "Y": 6}
    #: False -> step() leaves _last_env_vars alone (no RAM read at all)
    sets_location_on_step = True
    truncation_cause = "stagnation"

    def __init__(self, config):
        self.config = config
        self.resets = 0
        self.steps = 0
        self.closed = False
        self._last_env_vars = None
        self.action_space = types.SimpleNamespace(n=8)
        self.visit_archive = types.SimpleNamespace(n_cells_seen=lambda: 0)
        self._stepper = types.SimpleNamespace(
            episode_frames=0, capped_steps=0, episode_auto_frames=0
        )
        self.reward_calculator = _FakeRewards(self.truncation_cause)

    def output_shape(self):
        return (3, 36, 40)

    def ram_observation_shape(self):
        return (128,)

    def reset(self):
        self.resets += 1
        # The real reset() ends with _calculate_fitness(), which overwrites
        # _last_env_vars with the new episode's state.
        self._last_env_vars = dict(self.restart_location)
        return {"image": "post-reset-image", "ram": "post-reset-ram"}

    def step(self, action):
        self.steps += 1
        if self.sets_location_on_step:
            self._last_env_vars = dict(self.end_location)
        return {"image": "image", "ram": "ram"}, 1.0, True, False

    def get_frontier_captures(self):
        return []

    def close(self):
        self.closed = True


class _PartialLocationEnv(_FakeEnv):
    """map_bank absent from the RAM read, Y unreadable, X genuinely 0."""

    end_location = {"map_num": 42, "X": 0, "Y": None}


class _OriginLocationEnv(_FakeEnv):
    """Bank 0 / cell (0, 0) is a real coordinate, not a missing one."""

    end_location = {"map_bank": 0, "map_num": 0, "X": 0, "Y": 0}


class _NoRamReadEnv(_FakeEnv):
    sets_location_on_step = False


class _FakeRemote:
    def __init__(self, commands):
        self._commands = list(commands)
        self.sent = []
        self.closed = False

    def recv(self):
        return self._commands.pop(0)

    def send(self, message):
        self.sent.append(message)

    def close(self):
        self.closed = True


def _run_worker(env_cls, commands=(("step", 0), ("close", None))):
    """Drive vec_env._worker in-process against ``env_cls``."""
    created = {}

    def factory(config):
        env = env_cls(config)
        created["env"] = env
        return env

    remote = _FakeRemote(commands)
    with mock.patch.object(vec_env, "PyBoyEnvironment", factory):
        vec_env._worker(remote, {"n_goals_target": 0, "episode_length": 8}, 0)
    assert created["env"].closed is True
    assert remote.closed is True
    return remote, created["env"]


def _step_result(remote):
    assert remote.sent[0][0] == "init_ok", remote.sent[0]
    kind, payload = remote.sent[1]
    # _worker turns any exception into ("error", traceback) — surface it.
    assert kind == "ok", payload
    return payload


_LOCATION_KEYS = ("final_map_bank", "final_map_num", "final_x", "final_y")


# --------------------------------------------------------------------------
# 1. terminal_info carries the PRE-reset location
# --------------------------------------------------------------------------


def test_terminal_info_records_the_pre_reset_location():
    remote, env = _run_worker(_FakeEnv)
    _obs, reward, done, info = _step_result(remote)

    assert done is True
    assert reward == 1.0

    # The auto-reset really ran, and really did clobber the env's raw RAM
    # variables with the next episode's location — so the values asserted
    # below can only have been read from the episode that was ending.
    assert env.resets == 1
    assert env._last_env_vars == _FakeEnv.restart_location

    assert info["final_map_bank"] == 3
    assert info["final_map_num"] == 42
    assert info["final_x"] == 7
    assert info["final_y"] == 9
    # Not the post-reset row (where a seeded episode would restart).
    assert (info["final_map_bank"], info["final_map_num"]) != (0, 1)
    assert (info["final_x"], info["final_y"]) != (5, 6)


def test_terminal_info_location_does_not_disturb_the_other_keys():
    remote, _env = _run_worker(_FakeEnv)
    _obs, _reward, done, info = _step_result(remote)

    assert done is True
    assert info["truncation_cause"] == "stagnation"
    # Only a BUDGET truncation bootstraps the value of the continuation.
    assert info["truncated"] is False
    assert "terminal_observation" not in info
    assert info["flag_fires"] == 1
    assert info["unique_cells"] == 1
    assert info["unique_maps"] == 1
    assert info["final_stagnation_clock"] == 0.75
    # visited_cells / visited_maps stay sorted sets: chronology and the final
    # entry are exactly what they cannot give back, which is why the four
    # scalar location values exist.
    assert info["visited_cells"] == [(3, 42, 7, 9)]
    assert info["visited_maps"] == [(3, 42)]


def test_unreadable_location_components_stay_none_not_zero():
    remote, _env = _run_worker(_PartialLocationEnv)
    _obs, _reward, done, info = _step_result(remote)

    assert done is True
    assert info["final_map_bank"] is None  # key absent from the RAM read
    assert info["final_y"] is None  # key present, value unreadable
    assert info["final_map_num"] == 42
    # 0 is a REAL coordinate and must not be folded into None.
    assert info["final_x"] == 0


def test_origin_location_survives_as_zeros():
    remote, _env = _run_worker(_OriginLocationEnv)
    _obs, _reward, done, info = _step_result(remote)

    assert done is True
    assert [info[key] for key in _LOCATION_KEYS] == [0, 0, 0, 0]


def test_location_is_all_none_when_the_env_never_read_ram():
    remote, env = _run_worker(_NoRamReadEnv)
    _obs, _reward, done, info = _step_result(remote)

    assert done is True
    assert [info[key] for key in _LOCATION_KEYS] == [None, None, None, None]
    # The reset still happened; it just could not rescue a location that was
    # never read during the episode.
    assert env._last_env_vars == _FakeEnv.restart_location


# --------------------------------------------------------------------------
# 2. agent-side episode series
# --------------------------------------------------------------------------


def _tracking_agent():
    """Agent shell built the way tests/test_checkpoint_progress.py builds one."""
    agent = VecPPOAgent.__new__(VecPPOAgent)
    agent.config = {"best_success_window": 5}
    agent.episode = 0
    agent.stage_start_episode = 0
    agent.total_rollouts = 0
    agent.rollout_idx = 0
    agent.model = types.SimpleNamespace(_get_entropy_coef=lambda _idx: 0.01)
    agent.env_state_indices = [0]
    agent.reset_tracking()
    return agent


def test_final_location_series_stay_aligned_with_truncation_cause():
    agent = _tracking_agent()
    ends = [
        dict(
            truncation_cause="stagnation",
            final_map_bank=3,
            final_map_num=42,
            final_x=7,
            final_y=9,
            seeded=False,
        ),
        dict(
            truncation_cause="budget",
            final_map_bank=0,
            final_map_num=0,
            final_x=0,
            final_y=0,
            seeded=True,
        ),
        dict(
            truncation_cause=None,
            final_map_bank=None,
            final_map_num=5,
            final_x=None,
            final_y=2,
            seeded=False,
        ),
        # No location at all: an old worker, or one whose RAM read failed.
        dict(truncation_cause="battle_stagnation", seeded=True),
    ]
    for end in ends:
        agent._commit_episode(
            env_idx=0,
            reward_sum=1.0,
            length=100,
            goals_total=0,
            n_goals_target=0,
            **end,
        )

    data = agent.episode_data
    n = len(data["episode_truncation_cause"])
    assert n == len(ends)
    assert len(data["episode_final_map"]) == n
    assert len(data["episode_final_cell"]) == n
    assert len(data["episode_seeded"]) == n
    assert len(data["episode_rewards"]) == n

    assert data["episode_truncation_cause"] == [
        "stagnation",
        "budget",
        None,
        "battle_stagnation",
    ]
    assert data["episode_seeded"] == [False, True, False, True]
    assert data["episode_final_map"] == [[3, 42], [0, 0], [None, 5], [None, None]]
    assert data["episode_final_cell"] == [[7, 9], [0, 0], [None, 2], [None, None]]


def _dict_literal_keys(func_name, target):
    """String keys of the dict literal assigned to ``target`` in ``func_name``."""
    tree = ast.parse(pathlib.Path(agent_module.__file__).read_text(encoding="utf-8"))
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == func_name
    )
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        assigned = [t.id for t in node.targets if isinstance(t, ast.Name)]
        assigned += [t.attr for t in node.targets if isinstance(t, ast.Attribute)]
        if target in assigned:
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError(f"no dict literal assigned to {target!r} in {func_name}()")


def test_new_series_are_registered_in_both_episode_data_init_dicts():
    """load_model's allowlist silently DISCARDS any loaded key that is not in
    its ``fresh`` dict, so a series registered only in ``reset_tracking`` would
    survive a from-scratch start and vanish on every resume and curriculum
    stage transition — taking the index alignment with
    ``episode_truncation_cause`` / ``episode_seeded`` with it."""
    fresh = _dict_literal_keys("load_model", "fresh")
    tracking = _dict_literal_keys("reset_tracking", "episode_data")
    for key in (
        "episode_final_map",
        "episode_final_cell",
        "rollout_accepted_epochs",
        # the series the new ones must stay aligned with
        "episode_truncation_cause",
        "episode_seeded",
    ):
        assert key in tracking, f"{key} missing from reset_tracking's episode_data"
        assert key in fresh, f"{key} missing from load_model's allowlist"


def test_worker_terminal_info_feeds_the_episode_series_unchanged():
    """End-to-end value flow: a real terminal_info from _worker (with an
    unreadable component and a genuine 0) lands in the two series as-is."""
    remote, _env = _run_worker(_PartialLocationEnv)
    _obs, _reward, done, info = _step_result(remote)
    assert done is True

    agent = _tracking_agent()
    agent._commit_episode(
        env_idx=0,
        reward_sum=1.0,
        length=100,
        goals_total=0,
        n_goals_target=0,
        truncation_cause=info.get("truncation_cause"),
        final_map_bank=info.get("final_map_bank"),
        final_map_num=info.get("final_map_num"),
        final_x=info.get("final_x"),
        final_y=info.get("final_y"),
    )

    data = agent.episode_data
    assert data["episode_truncation_cause"] == ["stagnation"]
    assert data["episode_final_map"] == [[None, 42]]
    assert data["episode_final_cell"] == [[0, None]]


def _commit_episode_call_site():
    tree = ast.parse(pathlib.Path(agent_module.__file__).read_text(encoding="utf-8"))
    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_commit_episode"
    ]
    assert len(sites) == 1, f"expected one _commit_episode call site, got {len(sites)}"
    return sites[0]


def test_terminal_block_passes_the_location_through_uncoerced():
    """`info.get(...)` and nothing wrapped around it. An int() / `or 0` at the
    call site would turn an unreadable component into map bank 0 or cell
    (0, 0) — both real coordinates — and silently poison the stall histogram,
    which no _commit_episode-level test can see."""
    keywords = {kw.arg: kw.value for kw in _commit_episode_call_site().keywords}
    for arg in _LOCATION_KEYS:
        value = keywords.get(arg)
        assert isinstance(value, ast.Call), f"{arg}: not a plain call"
        assert isinstance(value.func, ast.Attribute), f"{arg}: not attribute access"
        assert value.func.attr == "get", f"{arg}: not .get()"
        assert isinstance(value.func.value, ast.Name), f"{arg}: not read off a name"
        assert value.func.value.id == "info", f"{arg}: not read off terminal_info"
        assert not value.keywords, f"{arg}: .get() carries a default"
        assert len(value.args) == 1, f"{arg}: .get() takes one argument"
        assert value.args[0].value == arg, f"{arg}: reads the wrong terminal_info key"


# --------------------------------------------------------------------------
# 3. rollout_accepted_epochs
# --------------------------------------------------------------------------


def _diag_agent(diag):
    agent = _tracking_agent()
    agent.model = types.SimpleNamespace(
        _get_entropy_coef=lambda _idx: 0.01,
        last_epoch_diag=diag,
        optimizer=None,
    )
    agent._rollout_entropy_n = 1
    agent._rollout_entropy_sum = 0.5
    agent._rollout_behaviour_entropy_sum = 0.4
    return agent


def test_accepted_epochs_is_nan_not_zero_when_the_diag_lacks_it():
    # No accepted_epochs key: the update never produced an epoch tally.
    agent = _diag_agent({"approx_kl": 0.01})
    agent._record_rollout_diagnostics(duration_s=0.5)

    series = agent.episode_data["rollout_accepted_epochs"]
    assert len(series) == 1
    assert math.isnan(series[-1])
    assert not series[-1] == 0.0  # 0.0 would read as "stopped after 0 epochs"


def test_accepted_epochs_records_the_completed_epoch_count():
    agent = _diag_agent({"accepted_epochs": 3, "epochs_run": 3})
    agent._record_rollout_diagnostics()

    assert agent.episode_data["rollout_accepted_epochs"] == [3.0]


def test_zero_epochs_from_a_replay_rejection_stays_zero_and_stays_flagged():
    """_minibatch.py's collection/replay early return sets accepted_epochs=0
    with replay_rejected=True — no gradient was applied at all. A first-check
    KL early-stop also lands on 0, so the 0 is only interpretable next to the
    already-persisted rejection flags."""
    agent = _diag_agent({"accepted_epochs": 0, "replay_rejected": True, "kl_rejected": False})
    agent._record_rollout_diagnostics()

    data = agent.episode_data
    assert data["rollout_accepted_epochs"] == [0.0]
    assert data["rollout_replay_rejected"] == [True]
    assert data["rollout_kl_rejected"] == [False]


def test_one_entry_per_rollout_for_the_new_series():
    agent = _diag_agent({"accepted_epochs": 2})
    agent._commit_episode(
        env_idx=0,
        reward_sum=1.0,
        length=100,
        goals_total=0,
        n_goals_target=0,
        truncation_cause="stagnation",
        final_map_bank=3,
        final_map_num=42,
        final_x=7,
        final_y=9,
    )
    agent._record_rollout_diagnostics()
    agent._commit_episode(
        env_idx=0,
        reward_sum=2.0,
        length=50,
        goals_total=0,
        n_goals_target=0,
        truncation_cause="budget",
        final_map_bank=1,
        final_map_num=2,
        final_x=3,
        final_y=4,
    )
    agent._record_rollout_diagnostics()

    data = agent.episode_data
    # Rollout series: one entry per _record_rollout_diagnostics call.
    assert len(data["rollout_accepted_epochs"]) == 2
    assert len(data["rollout_grad_norm_heads"]) == 2
    # Episode series: one entry per terminal, still aligned.
    assert len(data["episode_final_map"]) == 2
    assert len(data["episode_truncation_cause"]) == 2
    assert data["episode_final_map"][1] == [1, 2]
    assert data["episode_final_cell"][1] == [3, 4]
