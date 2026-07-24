# -*- coding: utf-8 -*-
"""Go-Explore frontier seeding: pool curation, rarity-biased sampling,
probe-worker isolation, and worker-side capture gating.

These exercise the pure orchestration/gating logic via lightweight stubs, so
no ROM or PyBoy instance is needed. The re-pay-suppression invariant (a
frontier-seeded reset pays only for territory BEYOND the seed) is a property
of the existing reset() no-op baseline step and is covered by the reward tests;
it needs no special Go-Explore code.
"""

import types
import numpy as np
import pytest

from PoliwhiRL.agents.PPO.vec_ppo_agent import VecPPOAgent
from PoliwhiRL.environment.gym_env import PyBoyEnvironment


def _agent_stub(**over):
    a = types.SimpleNamespace()
    a._frontier_index = {}
    a.frontier_pool = []
    a.goexplore_pool_size = 4
    a.goexplore_granularity = over.pop("goexplore_granularity", "map")
    a.goexplore_probe_workers = 2
    a.goexplore_seed_fraction = 1.0
    a.env_is_seeded = [False] * 6
    a.env_is_seeded_pending = [False] * 6
    a._true_start_path = "/tmp/true_start.state"
    # Live rarity source: rank/sample use the CURRENT archive count, not the
    # capture-time count. Tests populate a._live keyed by region or cell.
    a._live = {}
    a.visit_archive = types.SimpleNamespace(
        map_count=lambda b, n: a._live.get((int(b), int(n)), 0),
        count=lambda b, n, x, y: a._live.get((int(b), int(n), int(x), int(y)), 0),
        cell_key=lambda b, n, x, y: (int(b), int(n), int(x), int(y)),
    )
    for name in ("_pick_frontier_snapshot", "_live_count", "_frontier_key"):
        setattr(a, name, types.MethodType(getattr(VecPPOAgent, name), a))
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _cap(bank, num, count, path=None):
    return {
        "path": path or f"/tmp/b{bank}_m{num}_c{count}.state",
        "map_count": count,
        "bank": bank,
        "num": num,
        "x": 1,
        "y": 2,
    }


# ---------------------------------------------------------------- pool curation


def test_ingest_keeps_one_per_region():
    a = _agent_stub()
    VecPPOAgent._ingest_frontier_captures(a, [_cap(5, 9, 10), _cap(5, 9, 3)])
    assert set(a._frontier_index.keys()) == {(5, 9)}
    assert len(a.frontier_pool) == 1


def test_pool_capped_to_live_rarest_regions():
    a = _agent_stub(goexplore_pool_size=2)
    # stale capture-time counts are irrelevant; LIVE counts decide the pool
    a._live = {(5, 9): 30, (6, 1): 2, (7, 2): 90, (8, 3): 15}
    VecPPOAgent._ingest_frontier_captures(
        a, [_cap(5, 9, 0), _cap(6, 1, 0), _cap(7, 2, 0), _cap(8, 3, 0)]
    )
    # all four regions retained in the index, pool keeps only the 2 live-rarest
    assert len(a._frontier_index) == 4
    assert len(a.frontier_pool) == 2
    kept = sorted((c["bank"], c["num"]) for c in a.frontier_pool)
    assert kept == [(6, 1), (8, 3)]  # live counts 2 and 15


def test_pick_biases_toward_live_rarer_region():
    a = _agent_stub()
    a._live = {(5, 9): 1, (6, 1): 999}
    VecPPOAgent._ingest_frontier_captures(a, [_cap(5, 9, 0), _cap(6, 1, 0)])
    np.random.seed(0)
    picks = [VecPPOAgent._pick_frontier_snapshot(a)["num"] for _ in range(400)]
    rare = sum(1 for p in picks if p == 9)  # live-count-1 region
    assert rare > 300  # overwhelmingly the rare one under 1/(1+live_count)


def test_stale_common_region_sinks_out_of_pool():
    # A region captured when rare (count 0) but now common must lose to a
    # genuinely-rare region under live ranking.
    a = _agent_stub(goexplore_pool_size=1)
    a._live = {(24, 4): 7000, (5, 9): 3}
    VecPPOAgent._ingest_frontier_captures(a, [_cap(24, 4, 0), _cap(5, 9, 0)])
    assert len(a.frontier_pool) == 1
    assert (a.frontier_pool[0]["bank"], a.frontier_pool[0]["num"]) == (5, 9)


def test_pick_empty_pool_returns_none():
    a = _agent_stub()
    assert VecPPOAgent._pick_frontier_snapshot(a) is None


# ------------------------------------------------------------ probe isolation


def test_goexplore_cycle_never_seeds_probe_workers():
    a = _agent_stub()
    VecPPOAgent._ingest_frontier_captures(a, [_cap(5, 9, 2)])
    calls = []
    vec = types.SimpleNamespace(
        set_env_state=lambda idx, path: calls.append((idx, path))
    )
    # env 0 and 1 are probe workers -> must never be seeded/touched
    for idx in (0, 1):
        VecPPOAgent._goexplore_cycle(a, vec, idx)
        assert a.env_is_seeded_pending[idx] is False
    assert calls == []


def test_goexplore_cycle_seeds_non_probe_worker():
    a = _agent_stub(goexplore_seed_fraction=1.0)
    VecPPOAgent._ingest_frontier_captures(a, [_cap(5, 9, 2)])
    calls = []
    vec = types.SimpleNamespace(
        set_env_state=lambda idx, path: calls.append((idx, path))
    )
    np.random.seed(0)
    VecPPOAgent._goexplore_cycle(a, vec, 3)
    assert a.env_is_seeded_pending[3] is True
    assert len(calls) == 1 and calls[0][0] == 3
    assert calls[0][1].endswith("b5_m9_c2.state")


def test_goexplore_cycle_returns_seeded_worker_to_true_start():
    a = _agent_stub(goexplore_seed_fraction=0.0)  # never seed this cycle
    a.env_is_seeded[3] = True  # but it was seeded last ep
    calls = []
    vec = types.SimpleNamespace(
        set_env_state=lambda idx, path: calls.append((idx, path))
    )
    VecPPOAgent._goexplore_cycle(a, vec, 3)
    assert a.env_is_seeded_pending[3] is False
    assert calls == [(3, "/tmp/true_start.state")]


# ------------------------------------------------------- worker-side capture


class _FakePyBoy:
    def __init__(self):
        self.saves = 0

    def save_state(self, f):
        self.saves += 1
        f.write(b"STATE")


class _FakeArchive:
    def __init__(self, counts, cell_counts=None):
        self._counts = counts
        self._cell_counts = cell_counts or {}

    def map_count(self, bank, num):
        return self._counts.get((bank, num), 0)

    def count(self, bank, num, x, y):
        return self._cell_counts.get((bank, num, x, y), 0)

    def cell_key(self, bank, num, x, y):
        return (bank, num, x, y)


def _env_stub(tmpdir, counts, granularity="map", cell_counts=None):
    e = types.SimpleNamespace()
    e._goexplore_enabled = True
    e._goexplore_granularity = granularity
    e._goexplore_map_count_max = 100
    e._goexplore_cell_count_max = 3
    e._goexplore_max_captures_per_ep = 3
    e._goexplore_snapshot_dir = str(tmpdir)
    e._goexplore_flag_capture = False
    e._goexplore_flag_count_max = 3
    e._goexplore_catch_capture = False
    e._goexplore_catch_party_min = 2
    e._frontier_capture_seq = 0
    e._frontier_captures = []
    e._captured_maps_this_episode = set()
    e._captured_cells_this_episode = set()
    e._captured_flags_this_episode = set()
    e._captured_party_milestones_this_episode = set()
    e._pending_flag_capture = None
    e._frontier_capture_count_this_episode = 0
    e.pyboy = _FakePyBoy()
    e.visit_archive = _FakeArchive(counts, cell_counts or {})
    e._save_frontier_state = types.MethodType(PyBoyEnvironment._save_frontier_state, e)
    return e


def _vars(bank, num, script=False, party_size=1):
    # A valid, non-scripted overworld frame.
    return {
        "map_bank": bank,
        "map_num": num,
        "X": 5,
        "Y": 6,
        "script_active": script,
        "party_info": (party_size, 5, 20, 100),
        "battle_type": 0,
        "ui_byte": 0,
        "map_handler_byte": 0,
    }


def test_catch_capture_fires_on_party_growth_and_dedups(tmp_path, monkeypatch):
    import PoliwhiRL.environment.gym_env as ge

    monkeypatch.setattr(ge, "is_ram_state_valid", lambda v: True)
    e = _env_stub(
        tmp_path,
        counts={(24, 3): 5000},
        granularity="cell",
        cell_counts={(24, 3, 5, 6): 9000},
    )
    e._goexplore_catch_capture = True
    # Starter only (party 1) -> below the >=2 threshold, no capture.
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 3, party_size=1))
    assert e._frontier_captures == []
    # First catch (party 2) -> one "party" capture carrying the party size.
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 3, party_size=2))
    assert len(e._frontier_captures) == 1
    cap = e._frontier_captures[0]
    assert cap["kind"] == "party" and cap["party_size"] == 2
    # Same milestone again -> deduped (one snapshot per party size per episode).
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 3, party_size=2))
    assert len(e._frontier_captures) == 1


def test_catch_capture_off_by_default(tmp_path, monkeypatch):
    import PoliwhiRL.environment.gym_env as ge

    monkeypatch.setattr(ge, "is_ram_state_valid", lambda v: True)
    e = _env_stub(
        tmp_path,
        counts={(24, 3): 5000},
        granularity="cell",
        cell_counts={(24, 3, 5, 6): 9000},
    )
    # _goexplore_catch_capture defaults False in the stub -> never captures.
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 3, party_size=3))
    assert all(c.get("kind") != "party" for c in e._frontier_captures)


def test_capture_fires_only_for_rare_map(tmp_path, monkeypatch):
    import PoliwhiRL.environment.gym_env as ge

    monkeypatch.setattr(ge, "is_ram_state_valid", lambda v: True)
    e = _env_stub(tmp_path, counts={(24, 4): 5000, (5, 9): 2})
    # common map -> no capture
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 4))
    assert e._frontier_captures == []
    # rare map -> one capture, file written
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(5, 9))
    assert len(e._frontier_captures) == 1
    assert e._frontier_captures[0]["bank"] == 5
    assert e.pyboy.saves == 1
    written = list(tmp_path.iterdir())
    assert len(written) == 1 and written[0].read_bytes() == b"STATE"


def test_capture_dedups_per_map_per_episode(tmp_path, monkeypatch):
    import PoliwhiRL.environment.gym_env as ge

    monkeypatch.setattr(ge, "is_ram_state_valid", lambda v: True)
    e = _env_stub(tmp_path, counts={(5, 9): 2})
    for _ in range(5):
        PyBoyEnvironment._maybe_capture_frontier(e, _vars(5, 9))
    assert len(e._frontier_captures) == 1  # only the first entry this episode


def test_capture_skips_scripted_and_invalid_frames(tmp_path, monkeypatch):
    import PoliwhiRL.environment.gym_env as ge

    monkeypatch.setattr(ge, "is_ram_state_valid", lambda v: True)
    e = _env_stub(tmp_path, counts={(5, 9): 2})
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(5, 9, script=True))
    assert e._frontier_captures == []
    # invalid RAM frame also rejected
    monkeypatch.setattr(ge, "is_ram_state_valid", lambda v: False)
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(5, 9))
    assert e._frontier_captures == []


# ------------------------------------------------------- cell-granularity mode


def test_cell_mode_pool_keyed_by_cell():
    a = _agent_stub(goexplore_granularity="cell", goexplore_pool_size=8)
    # two distinct cells in the same map are DISTINCT frontier locations
    caps = [
        {"path": "/tmp/a", "map_count": 0, "bank": 5, "num": 9, "x": 10, "y": 20},
        {"path": "/tmp/b", "map_count": 0, "bank": 5, "num": 9, "x": 40, "y": 60},
    ]
    VecPPOAgent._ingest_frontier_captures(a, caps)
    assert len(a._frontier_index) == 2
    assert len(a.frontier_pool) == 2


def test_cell_mode_ranks_by_live_cell_count():
    a = _agent_stub(goexplore_granularity="cell", goexplore_pool_size=1)
    a._live = {(5, 9, 10, 20): 900, (5, 9, 40, 60): 2}  # keyed by cell
    caps = [
        {"path": "/tmp/a", "map_count": 0, "bank": 5, "num": 9, "x": 10, "y": 20},
        {"path": "/tmp/b", "map_count": 0, "bank": 5, "num": 9, "x": 40, "y": 60},
    ]
    VecPPOAgent._ingest_frontier_captures(a, caps)
    # the live-rarer cell (count 2, the frontier edge) wins the single pool slot
    assert len(a.frontier_pool) == 1
    assert (a.frontier_pool[0]["x"], a.frontier_pool[0]["y"]) == (40, 60)


def test_cell_capture_fires_for_rare_cell_and_caps_per_episode(tmp_path, monkeypatch):
    import PoliwhiRL.environment.gym_env as ge

    monkeypatch.setattr(ge, "is_ram_state_valid", lambda v: True)
    # map is common but we're in cell mode: rarity judged per-cell (all count 0)
    e = _env_stub(tmp_path, counts={(24, 4): 9000}, granularity="cell")
    for i in range(6):
        PyBoyEnvironment._maybe_capture_frontier(
            e, {"map_bank": 24, "map_num": 4, "X": i, "Y": 0, "script_active": False}
        )
    # capped at max_captures_per_episode (3), one per distinct cell
    assert len(e._frontier_captures) == 3


def test_cell_capture_dedups_and_respects_cell_threshold(tmp_path, monkeypatch):
    import PoliwhiRL.environment.gym_env as ge

    monkeypatch.setattr(ge, "is_ram_state_valid", lambda v: True)
    e = _env_stub(
        tmp_path,
        counts={},
        granularity="cell",
        cell_counts={(24, 4, 5, 6): 99},  # this cell is well-trodden
    )
    # same cell twice -> at most one capture; and it's over threshold (3) anyway
    for _ in range(3):
        PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 4))
    assert e._frontier_captures == []  # count 99 > cell_count_max 3


# ---------------------------------------------------- flag-state Go-Explore


def _flag_cap(bit, count, path=None):
    return {
        "path": path or f"/tmp/flag{bit}_c{count}.state",
        "map_count": count,
        "bank": 24,
        "num": 3,
        "x": 1,
        "y": 2,
        "kind": "flag",
        "flag": bit,
    }


def _agent_stub_with_flags(**over):
    a = _agent_stub(**over)
    # Rarity source for flag captures: run-wide event-flag fire count.
    a._flag_live = {}
    a.visit_archive.event_flag_fire_count = lambda b: a._flag_live.get(int(b), 0)
    return a


def test_flag_cap_keyed_by_bit():
    a = _agent_stub_with_flags()
    assert VecPPOAgent._frontier_key(a, _flag_cap(31, 0)) == ("flag", 31)


def test_flag_cap_ranked_by_fire_count():
    a = _agent_stub_with_flags(goexplore_pool_size=1)
    a._flag_live = {31: 0, 65: 40}  # flag 31 is rare, 65 well-worn
    VecPPOAgent._ingest_frontier_captures(a, [_flag_cap(31, 0), _flag_cap(65, 0)])
    assert len(a._frontier_index) == 2
    assert [c["flag"] for c in a.frontier_pool] == [31]  # rarest flag kept


def test_flag_and_cell_caps_coexist_in_one_pool():
    a = _agent_stub_with_flags(goexplore_granularity="cell", goexplore_pool_size=4)
    a._live = {(24, 3, 1, 2): 0}  # a fresh cell
    a._flag_live = {31: 0}  # a fresh flag
    cell_cap = {
        "path": "/tmp/c.state",
        "map_count": 0,
        "bank": 24,
        "num": 3,
        "x": 1,
        "y": 2,
        "kind": "cell",
    }
    VecPPOAgent._ingest_frontier_captures(a, [cell_cap, _flag_cap(31, 0)])
    keys = set(a._frontier_index.keys())
    assert ("flag", 31) in keys
    assert (24, 3, 1, 2) in keys


def test_flag_capture_snapshots_rare_flag(tmp_path, monkeypatch):
    import PoliwhiRL.environment.gym_env as ge

    monkeypatch.setattr(ge, "is_ram_state_valid", lambda v: True)
    e = _env_stub(
        tmp_path, counts={}, granularity="cell", cell_counts={(24, 3, 5, 6): 9000}
    )  # cell well-trodden
    e._goexplore_flag_capture = True
    # cell is well-trodden so no cell capture; only a flag capture should fire
    e.reward_calculator = types.SimpleNamespace(rare_event_fires=lambda cmax: [(31, 0)])
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 3))
    assert len(e._frontier_captures) == 1
    cap = e._frontier_captures[0]
    assert cap["kind"] == "flag" and cap["flag"] == 31
    assert 31 in e._captured_flags_this_episode


def test_flag_capture_latches_through_script_then_snapshots(tmp_path, monkeypatch):
    import PoliwhiRL.environment.gym_env as ge

    monkeypatch.setattr(ge, "is_ram_state_valid", lambda v: True)
    e = _env_stub(
        tmp_path, counts={}, granularity="cell", cell_counts={(24, 3, 5, 6): 9000}
    )  # cell well-trodden
    e._goexplore_flag_capture = True
    # Flag fires DURING a scripted frame -> latched, no snapshot yet.
    e.reward_calculator = types.SimpleNamespace(rare_event_fires=lambda cmax: [(31, 0)])
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 3, script=True))
    assert e._frontier_captures == []
    assert e._pending_flag_capture == (31, 0)
    # Next clean frame, no new fire -> the latched capture is taken.
    e.reward_calculator = types.SimpleNamespace(rare_event_fires=lambda cmax: [])
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 3, script=False))
    assert len(e._frontier_captures) == 1
    assert e._frontier_captures[0]["flag"] == 31


def test_empty_pool_recovery_rearms_common_checkpoint_once(tmp_path, monkeypatch):
    import PoliwhiRL.environment.gym_env as ge

    monkeypatch.setattr(ge, "is_ram_state_valid", lambda v: True)
    e = _env_stub(
        tmp_path, counts={}, granularity="cell", cell_counts={(24, 3, 5, 6): 9000}
    )
    e._goexplore_flag_capture = True
    e._goexplore_rearm_checkpoint_capture = True
    e._rearmed_checkpoint_flags = set()
    e.reward_calculator = types.SimpleNamespace(
        rare_event_fires=lambda _count_max: [],
        event_fires=lambda: [(31, 99)],
    )

    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 3))
    assert len(e._frontier_captures) == 1
    assert e._frontier_captures[0]["flag"] == 31
    assert e._rearmed_checkpoint_flags == {31}

    e._captured_flags_this_episode.clear()
    e._frontier_capture_count_this_episode = 0
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 3))
    assert len(e._frontier_captures) == 1
    assert e._pending_flag_capture is None


def test_map_capture_respects_shared_episode_budget(tmp_path, monkeypatch):
    import PoliwhiRL.environment.gym_env as ge

    monkeypatch.setattr(ge, "is_ram_state_valid", lambda v: True)
    e = _env_stub(tmp_path, counts={}, granularity="map")
    e._goexplore_max_captures_per_ep = 1

    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 3))
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 4))

    assert len(e._frontier_captures) == 1
    assert e._frontier_capture_count_this_episode == 1


def test_scripted_flag_does_not_overwrite_pending_capture(tmp_path, monkeypatch):
    import PoliwhiRL.environment.gym_env as ge

    monkeypatch.setattr(ge, "is_ram_state_valid", lambda v: True)
    e = _env_stub(tmp_path, counts={}, granularity="cell")
    e._goexplore_flag_capture = True
    fires = [[(30, 0)], [(31, 0)]]
    e.reward_calculator = types.SimpleNamespace(
        rare_event_fires=lambda _count_max: fires.pop(0)
    )

    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 3, script=True))
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 3, script=True))

    assert e._pending_flag_capture == (30, 0)


def test_flag_capture_off_by_default(tmp_path, monkeypatch):
    import PoliwhiRL.environment.gym_env as ge

    monkeypatch.setattr(ge, "is_ram_state_valid", lambda v: True)
    e = _env_stub(
        tmp_path, counts={}, granularity="cell", cell_counts={(24, 3, 5, 6): 9000}
    )
    # flag capture disabled -> reward_calculator.rare_event_fires never consulted
    e.reward_calculator = types.SimpleNamespace(
        rare_event_fires=lambda cmax: (_ for _ in ()).throw(
            AssertionError("should not be called")
        )
    )
    PyBoyEnvironment._maybe_capture_frontier(e, _vars(24, 3))
    assert e._frontier_captures == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
