# -*- coding: utf-8 -*-
import os
import types

from PoliwhiRL.agents.PPO.vec_ppo_agent import VecPPOAgent
from PoliwhiRL.checkpoints import (
    checkpoint_slug,
    checkpoint_title,
    is_recordable_checkpoint,
)
from PoliwhiRL.environment.vec_env import _load_actions_file


def _recording_agent(tmp_path):
    agent = VecPPOAgent.__new__(VecPPOAgent)
    agent.checkpoint_recording_enabled = True
    agent.checkpoint_recordings = {}
    agent.config = {"record_path": str(tmp_path)}
    agent.episode = 12
    agent.rollout_idx = 34
    agent.total_rollouts = 34
    return agent


def test_checkpoint_titles_are_human_readable_and_safe():
    assert checkpoint_title(30) == "Got Mystery Egg From Mr. Pokémon"
    assert checkpoint_title(31) == "Gave Mystery Egg To Elm"
    assert checkpoint_slug(30) == "0030-Got-Mystery-Egg-From-Mr-Pokemon"
    assert is_recordable_checkpoint(30)
    assert not is_recordable_checkpoint(26)  # transient script bit
    assert not is_recordable_checkpoint(1726)  # non-monotonic sprite bit


def test_first_honest_checkpoint_saves_action_prefix_once(tmp_path):
    agent = _recording_agent(tmp_path)
    jobs = agent._claim_checkpoint_recordings(
        env_idx=2,
        info={"flag_fire_steps": [[30, 4]]},
        actions=[4, 5, 6, 7, 8],
        seeded=False,
        state_index=0,
    )

    assert len(jobs) == 1
    entry = agent.checkpoint_recordings[30]
    assert entry["title"] == "Got Mystery Egg From Mr. Pokémon"
    assert entry["fire_step"] == 4
    assert entry["action_count"] == 3
    assert os.path.isfile(entry["actions_path"])
    assert _load_actions_file(entry["actions_path"]) == [[4, 5, 6]]

    duplicate = agent._claim_checkpoint_recordings(
        env_idx=3,
        info={"flag_fire_steps": [[30, 2]]},
        actions=[1, 2],
        seeded=False,
        state_index=0,
    )
    assert duplicate == []
    assert _load_actions_file(entry["actions_path"]) == [[4, 5, 6]]


def test_seeded_or_noncanonical_checkpoint_is_not_recorded(tmp_path):
    agent = _recording_agent(tmp_path)
    info = {"flag_fire_steps": [[31, 2]]}

    assert (
        agent._claim_checkpoint_recordings(0, info, [1, 2], seeded=True, state_index=0)
        == []
    )
    assert (
        agent._claim_checkpoint_recordings(0, info, [1, 2], seeded=False, state_index=1)
        == []
    )
    assert agent.checkpoint_recordings == {}


def test_progress_bar_tolerates_legacy_missing_parallel_metrics():
    agent = VecPPOAgent.__new__(VecPPOAgent)
    agent.episode = 5
    agent.episode_data = {
        "episode_rewards": [1.0, 2.0],
        "episode_seeded": [],
        "episode_unique_maps": [],
        "episode_unique_cells": [],
        "episode_flag_fires": [],
        "rollout_policy_entropy": [],
        "rollout_approx_kl": [],
        "rollout_clip_fraction": [],
    }
    agent.visit_archive = types.SimpleNamespace(n_cells_seen=lambda: 0)
    agent.frontier_pool = []
    agent.probe_enabled = False
    captured = {}
    pbar = types.SimpleNamespace(set_postfix=lambda value: captured.update(value))

    agent._update_progress_bar(pbar)

    assert captured["h_r"] == "1.5"
    assert captured["h_maps"] == "n/a"


def test_parent_deduplicates_simultaneous_worker_discoveries():
    agent = VecPPOAgent.__new__(VecPPOAgent)
    agent.config = {"best_success_window": 5}
    agent.episode = 0
    agent.stage_start_episode = 0
    agent.total_rollouts = 1
    agent.rollout_idx = 0
    agent.model = types.SimpleNamespace(_get_entropy_coef=lambda _idx: 0.01)
    agent.env_state_indices = [0, 0]
    agent.reset_tracking()
    discovery = [{"type": "flag", "key": 30, "step": 100}]

    for env_idx in (0, 1):
        agent._commit_episode(
            env_idx=env_idx,
            reward_sum=1.0,
            length=100,
            goals_total=0,
            n_goals_target=0,
            discoveries=discovery,
            seeded=False,
        )

    assert len(agent.episode_data["discovery_log"]) == 1
