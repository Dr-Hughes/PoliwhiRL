# -*- coding: utf-8 -*-
"""2026-10-25 reward-economy rebuild: catch classification, species-keyed
catch reward, competence-gated checkpoints, corruption guard, NPC-talk
novelty, progress-reset stagnation, LR recovery, self-imitation and the
agent-side competence / reverse-curriculum bookkeeping."""

import os
import tempfile
import types
import unittest
from collections import deque

import numpy as np
import torch

from PoliwhiRL.agents.PPO.vec_ppo_agent import VecPPOAgent
from PoliwhiRL.checkpoints import DERIVED_FLAG_TABLE
from PoliwhiRL.environment.rewards import Rewards, is_ram_state_valid
from PoliwhiRL.environment.visit_archive import VisitArchive, map_rung_id
from PoliwhiRL.models.PPO.ppo_model_implementation import PPOModel


def _zero_flags():
    return np.zeros(256, dtype=np.uint8)


def _with_flags(flags, *flag_nums):
    out = flags.copy()
    for f in flag_nums:
        out[f // 8] |= 1 << (f % 8)
    return out


def _cfg(**over):
    cfg = {
        "episode_length": 1000,
        "new_map_reward": 0,
        "new_bank_reward": 0,
        "frontier_novelty_bonus": 0,
        "whiteout_penalty": 0,
        "step_penalty": 0.0,
        "flag_progress_reward": 0,
        "map_goal_reward": 0,
        "pokedex_owned_reward": 30,
        "pokedex_first_sight_reward": 0,
        "level_up_reward": 0,
        "battle_engagement_reward": 0.0,
        "battle_win_reward": 0.0,
        "reward_round_dp": None,
        "reward_clip": 1000,
        "goals": [],
    }
    cfg.update(over)
    return cfg


def _ev(
    x=4,
    y=3,
    map_bank=24,
    map_num=7,
    battle_type=0,
    party_size=1,
    owned=1,
    flags=None,
    script_active=False,
    key_items_count=0,
    direction=2,
):
    return {
        "X": x,
        "Y": y,
        "map_num": map_num,
        "map_bank": map_bank,
        "pokedex_seen": owned,
        "pokedex_owned": owned,
        "story_flags": flags if flags is not None else _zero_flags(),
        "battle_type": battle_type,
        "player_state": 0,
        "key_items_count": key_items_count,
        "enemy_hp": 10,
        "enemy_max_hp": 10,
        "party_info": (party_size, 5 * party_size, 20, 0),
        "script_active": script_active,
        "player_direction": direction,
    }


class TestCatchClassification(unittest.TestCase):
    def _catch(self, rw, owned_rises_before_party):
        rw.calculate_reward(_ev(owned=1), "")
        rw.calculate_reward(_ev(battle_type=1, owned=1), "")
        if owned_rises_before_party:
            # Crystal order: owned bit set (dex screen), THEN party add.
            rw.calculate_reward(_ev(battle_type=1, owned=2), "")
            r, _ = rw.calculate_reward(_ev(battle_type=1, owned=2, party_size=2), "")
        else:
            r, _ = rw.calculate_reward(_ev(battle_type=1, owned=1, party_size=2), "")
            r2, _ = rw.calculate_reward(_ev(battle_type=1, owned=2, party_size=2), "")
            r += r2
        return float(r)

    def test_owned_bit_before_party_add_is_a_new_species(self):
        rw = Rewards(_cfg(species_catch_reward_enabled=True))
        r = self._catch(rw, owned_rises_before_party=True)
        counts = rw.get_episode_event_counts()
        self.assertEqual(counts["new_species_catches"], 1)
        self.assertEqual(counts["duplicate_catches"], 0)
        self.assertAlmostEqual(r, 30.0)

    def test_owned_bit_after_party_add_is_still_a_new_species(self):
        rw = Rewards(_cfg(species_catch_reward_enabled=True))
        r = self._catch(rw, owned_rises_before_party=False)
        self.assertEqual(rw.get_episode_event_counts()["new_species_catches"], 1)
        self.assertAlmostEqual(r, 30.0)

    def test_duplicate_catch_is_not_new(self):
        rw = Rewards(_cfg(species_catch_reward_enabled=True))
        rw.calculate_reward(_ev(owned=2), "")
        rw.calculate_reward(_ev(battle_type=1, owned=2), "")
        rw.calculate_reward(_ev(battle_type=1, owned=2, party_size=2), "")
        for _ in range(10):
            rw.calculate_reward(_ev(owned=2, party_size=2), "")
        counts = rw.get_episode_event_counts()
        self.assertEqual(counts["catches"], 1)
        self.assertEqual(counts["new_species_catches"], 0)
        self.assertEqual(counts["duplicate_catches"], 1)

    def test_evolution_owned_rise_pays_nothing(self):
        rw = Rewards(_cfg(species_catch_reward_enabled=True))
        rw.calculate_reward(_ev(owned=1), "")
        r, _ = rw.calculate_reward(_ev(owned=2), "")  # Quilava, no battle
        self.assertEqual(float(r), 0.0)
        self.assertEqual(rw.get_episode_event_counts()["new_species_catches"], 0)

    def test_species_reward_floors_after_depletion(self):
        archive = VisitArchive()
        archive.merge_milestones(
            [], 0, 0, 0, 0, milestone_fires=[("species_catch", 1)] * 1
        )
        for _ in range(399):
            archive.merge_milestones([], 0, 0, 0, 0, milestone_fires=[("species_catch", 1)])
        rw = Rewards(_cfg(species_catch_reward_enabled=True), visit_archive=archive)
        r = self._catch(rw, owned_rises_before_party=True)
        self.assertAlmostEqual(r, 30.0 * 0.34, places=4)


class TestCompetenceGating(unittest.TestCase):
    def _fire_31(self, rate, seeded=False):
        archive = VisitArchive()
        archive.set_checkpoint_competence({31: rate})
        rw = Rewards(
            _cfg(
                checkpoint_competence_gating=True,
                checkpoint_progress_reward=40,
                episode_seeded=seeded,
            ),
            visit_archive=archive,
        )
        rw.calculate_reward(_ev(), "")
        r, _ = rw.calculate_reward(_ev(flags=_with_flags(_zero_flags(), 31)), "")
        return float(r)

    def test_full_price_until_target(self):
        self.assertAlmostEqual(self._fire_31(0.17), 40.0)
        self.assertAlmostEqual(self._fire_31(0.6), 40.0)

    def test_recedes_to_floor(self):
        self.assertAlmostEqual(self._fire_31(0.8), 20.0)
        self.assertAlmostEqual(self._fire_31(1.0), 4.0)

    def test_seeded_reach_uses_honest_competence(self):
        self.assertAlmostEqual(self._fire_31(0.17, seeded=True), 40.0)

    def test_alternative_starter_is_priced_on_the_group(self):
        # Cyndaquil (27) habitual: the rare Chikorita (29) must not pay full.
        archive = VisitArchive()
        archive.set_checkpoint_competence({27: 1.0})
        rw = Rewards(
            _cfg(checkpoint_competence_gating=True, checkpoint_progress_reward=40),
            visit_archive=archive,
        )
        rw.calculate_reward(_ev(), "")
        r, _ = rw.calculate_reward(_ev(flags=_with_flags(_zero_flags(), 29)), "")
        self.assertAlmostEqual(float(r), 4.0)


class TestCorruptionGuard(unittest.TestCase):
    def test_garbage_reads_are_invalid(self):
        self.assertFalse(is_ram_state_valid(_ev(map_bank=15, map_num=228)))
        self.assertFalse(is_ram_state_valid(_ev(key_items_count=255)))
        self.assertTrue(is_ram_state_valid(_ev()))

    def test_flag_burst_ends_episode_unpaid(self):
        rw = Rewards(_cfg(flag_progress_reward=20, checkpoint_progress_reward=40))
        rw.calculate_reward(_ev(), "")
        burst = [f for f, _ in DERIVED_FLAG_TABLE][:10]
        r, done = rw.calculate_reward(_ev(flags=_with_flags(_zero_flags(), *burst)), "")
        self.assertEqual(float(r), 0.0)
        self.assertTrue(done)
        self.assertEqual(rw.truncation_cause, "corrupt")
        self.assertEqual(rw.flag_fire_step_log, [])

    def test_invalid_streak_ends_episode(self):
        rw = Rewards(_cfg(corrupt_invalid_streak=5))
        rw.calculate_reward(_ev(), "")
        done = False
        for _ in range(5):
            _, done = rw.calculate_reward(_ev(key_items_count=255), "")
        self.assertTrue(done)
        self.assertEqual(rw.truncation_cause, "corrupt")


class TestNpcTalk(unittest.TestCase):
    def test_a_press_starting_a_script_pays_once_per_context(self):
        rw = Rewards(_cfg(npc_talk_reward=2.0))
        rw.calculate_reward(_ev(), "")
        r1, _ = rw.calculate_reward(_ev(script_active=True), "a")
        rw.calculate_reward(_ev(), "")
        r2, _ = rw.calculate_reward(_ev(script_active=True), "a")
        self.assertAlmostEqual(float(r1), 2.0)
        self.assertEqual(float(r2), 0.0)
        self.assertEqual(rw.get_episode_event_counts()["npc_talks"], 2)

    def test_same_object_from_two_sides_pays_once(self):
        # Facing down from (4,3) and up from (4,5) both target tile (4,4).
        rw = Rewards(_cfg(npc_talk_reward=2.0))
        rw.calculate_reward(_ev(x=4, y=3, direction=2), "")
        r1, _ = rw.calculate_reward(_ev(script_active=True), "a")
        rw.calculate_reward(_ev(x=4, y=5, direction=1), "")
        r2, _ = rw.calculate_reward(_ev(x=4, y=5, script_active=True), "a")
        rw.calculate_reward(_ev(x=4, y=5, direction=3), "")
        r3, _ = rw.calculate_reward(_ev(x=4, y=5, script_active=True), "a")
        self.assertAlmostEqual(float(r1), 2.0)
        self.assertEqual(float(r2), 0.0)
        self.assertAlmostEqual(float(r3), 2.0)

    def test_script_without_a_press_does_not_pay(self):
        rw = Rewards(_cfg(npc_talk_reward=2.0))
        rw.calculate_reward(_ev(), "")
        r, _ = rw.calculate_reward(_ev(script_active=True), "up")
        self.assertEqual(float(r), 0.0)


class TestStagnationResetOnProgress(unittest.TestCase):
    def test_story_flag_resets_counter(self):
        rw = Rewards(
            _cfg(
                stagnation_resets_on_progress=True,
                flag_progress_reward=20,
                stagnation_truncation_steps=1000,
            )
        )
        for _ in range(50):
            rw.calculate_reward(_ev(), "")
        self.assertGreater(rw._stagnation_steps, 0)
        flag = next(f for f, _ in DERIVED_FLAG_TABLE if f not in (26, 27, 28, 29, 30, 31))
        rw.calculate_reward(_ev(flags=_with_flags(_zero_flags(), flag)), "")
        self.assertEqual(rw._stagnation_steps, 0)


def _model_config(**over):
    cfg = {
        "device": "cpu",
        "ram_obs_dim": 171,
        "ppo_learning_rate": 1e-4,
        "ppo_gamma": 0.99,
        "ppo_epsilon": 0.2,
        "ppo_value_loss_coef": 0.5,
        "ppo_entropy_coef": 0.01,
        "ppo_entropy_coef_decay": 0.0,
        "ppo_entropy_coef_min": 0.01,
        "ppo_entropy_anneal_enabled": False,
        "ppo_clip_value_loss": True,
        "ppo_gae_lambda": 0.95,
        "ppo_adam_eps": 1e-5,
        "ppo_dropout": 0.0,
        "ppo_max_grad_norm": 0.2,
        "ppo_lr_schedule": "constant",
        "num_rollouts": 10,
        "d_ram": 16,
        "d_model": 16,
        "n_heads": 2,
        "num_layers": 1,
        "mem_len": 4,
    }
    cfg.update(over)
    return cfg


class TestModelAdditions(unittest.TestCase):
    def test_lr_recovers_after_halving(self):
        model = PPOModel((3, 8, 8), 9, _model_config(ppo_lr_recovery_factor=2.0))
        model.halve_lr_multiplier()
        model.halve_lr_multiplier()
        self.assertAlmostEqual(model.effective_lr(), 2.5e-5)
        model.recover_lr_multiplier()
        self.assertAlmostEqual(model.effective_lr(), 5e-5)
        for _ in range(5):
            model.recover_lr_multiplier()
        self.assertAlmostEqual(model.effective_lr(), 1e-4)

    def test_sil_raises_logprob_of_stored_action(self):
        torch.manual_seed(0)
        model = PPOModel((3, 8, 8), 9, _model_config(sil_coef=1.0))
        b, l = 4, 3
        batch = {
            "states": torch.rand(b, l, 3, 8, 8) * 255,
            "ram_states": torch.rand(b, l, 171),
            "actions": torch.full((b, l), 4, dtype=torch.long),
            "returns": torch.full((b, l), 5.0),
            "dones": torch.zeros(b, l, dtype=torch.bool),
            "mems": [torch.zeros(b, 4, 16)],
        }

        def logp():
            with torch.no_grad():
                probs, _, _ = model.actor_critic(
                    batch["states"], batch["ram_states"], batch["mems"]
                )
            return float(torch.log(probs[..., 4]).mean())

        before = logp()
        diag = None
        for _ in range(20):
            diag = model.sil_update(batch)
        self.assertIsNotNone(diag)
        self.assertGreater(logp(), before)

    def test_reset_value_head(self):
        model = PPOModel((3, 8, 8), 9, _model_config())
        with torch.no_grad():
            model.actor_critic.fc_critic.bias.fill_(7.0)
        model.reset_value_head()
        self.assertEqual(float(model.actor_critic.fc_critic.bias.abs().sum()), 0.0)


def _agent_stub(**over):
    a = types.SimpleNamespace()
    a.competence_window = 200
    a.competence_min_episodes = 4
    a._honest_checkpoint_history = deque(maxlen=200)
    a._honest_fire_steps = {}
    a.checkpoint_competence = {}
    a.rc_enabled = True
    a.rc_target_rate = 0.5
    a.rc_exit_rate = 0.5
    a.rc_max_targets = 2
    a.rc_frontier_enabled = False
    a.rc_frontier_max_rate = 0.05
    a.rc_frontier_min_seeded = 3
    a.rc_max_frontier_targets = 1
    a.rc_frontier_targets = []
    a._honest_map_history = deque(maxlen=200)
    a.rc_trajectories_per_flag = 2
    a.rc_outcome_window = 4
    a.rc_promote_rate = 0.6
    a.rc_demote_rate = 0.2
    a.rc_targets = []
    a.rc_pool = {}
    a.rc_level = {}
    a.rc_outcomes = {}
    a.rc_stats = {}
    a.episode = 0
    a.visit_archive = VisitArchive()
    a._archive_dirty = False
    for name in (
        "_record_honest_checkpoints",
        "_update_competence",
        "_rc_available_targets",
        "_rc_max_rank",
        "_pick_rc_snapshot",
        "_ingest_rc_captures",
        "_record_rc_outcome",
        "_frontier_rungs",
        "_filter_rc_captures",
        "_honest_map_rates",
        "_map_rung_origin_ok",
        "_filter_map_rung_origins",
        "_prune_rc_pool_origins",
    ):
        setattr(a, name, types.MethodType(getattr(VecPPOAgent, name), a))
    for k, v in over.items():
        setattr(a, k, v)
    return a


class TestAgentCurriculum(unittest.TestCase):
    def test_targets_are_weak_rungs_beyond_the_reliable_horizon(self):
        a = _agent_stub()
        # 27 (starter) always at ~3500; 29 (alternative starter) rare but EARLY;
        # 30 reliable at ~4400; 31 rare at ~12000; 65 rare at ~13000.
        for i in range(10):
            fired = [[27, 3500], [30, 4400]]
            if i == 0:
                fired.append([29, 3600])
            if i < 2:
                fired += [[31, 12000], [65, 13000]]
            a._record_honest_checkpoints(fired)
        a._update_competence()
        self.assertEqual(a.rc_targets, [31, 65])
        self.assertEqual(a.visit_archive.rc_targets(), {31, 65})
        self.assertAlmostEqual(a.visit_archive.checkpoint_competence(31), 0.2)
        self.assertTrue(a._archive_dirty)

    def test_alternative_starters_are_one_rung(self):
        a = _agent_stub()
        for i in range(10):
            a._record_honest_checkpoints([[27 if i % 2 else 29, 3500], [30, 4400]])
        # A window recorded before grouping is canonicalised too.
        a._honest_checkpoint_history.append({29, 30})
        a._update_competence()
        self.assertEqual(a.checkpoint_competence[27], 1.0)
        self.assertNotIn(29, a.checkpoint_competence)
        self.assertEqual(a.rc_targets, [])

    def test_hysteresis_keeps_a_target_until_the_exit_rate(self):
        a = _agent_stub(rc_exit_rate=0.8)
        a.rc_targets = [31]
        for i in range(10):
            fired = [[30, 4400]]
            if i < 6:
                fired.append([31, 12000])
            a._record_honest_checkpoints(fired)
        a._update_competence()
        # 60%: above the 50% entry rate, below the 80% exit rate -> kept.
        self.assertEqual(a.rc_targets, [31])
        a.rc_targets = []
        a._update_competence()
        self.assertEqual(a.rc_targets, [])

    def test_frontier_rungs_are_seeded_only_maps_and_flags_by_seeded_reach(self):
        a = _agent_stub(rc_frontier_enabled=True, rc_max_frontier_targets=2, rc_max_targets=3)
        va = a.visit_archive
        for _ in range(8):  # main route: entered by every seeded episode
            va.merge_visits([], [(26, 2, True)])
        for _ in range(5):
            va.merge_visits([], [(10, 5, True)])
        for _ in range(4):  # a side room: honest play enters it often
            va.merge_visits([], [(24, 6, True)])
        for i in range(10):
            maps = [(24, 4, False)] + ([(24, 6, False)] if i < 5 else [])
            a._record_honest_checkpoints([[30, 4400]], maps)
        a._update_competence()
        self.assertEqual(
            a.rc_targets, [map_rung_id(26, 2), map_rung_id(10, 5)]
        )
        self.assertEqual(a.visit_archive.rc_targets(), set(a.rc_targets))
        # Map rungs never enter the flag rates the weak-rung logic reads, but
        # the workers' price table carries their honest entry rate.
        self.assertNotIn(map_rung_id(26, 2), a.checkpoint_competence)
        self.assertEqual(a.visit_archive.checkpoint_competence(map_rung_id(26, 2)), 0.0)
        self.assertIn(map_rung_id(26, 2), a.map_rung_competence)

    def test_map_rung_rings_keep_only_approach_states_and_rerank(self):
        a = _agent_stub()
        rung = map_rung_id(26, 2)  # Route 31
        for i in range(10):
            maps = {map_rung_id(26, 1)}  # Route 30: every honest episode
            if i == 0:
                maps |= {rung, map_rung_id(26, 11)}
            a._honest_map_history.append(maps)
        with tempfile.TemporaryDirectory() as d:
            def cap(rank, bank, num):
                p = os.path.join(d, f"k{rank}.state")
                open(p, "wb").write(b"x")
                return {"kind": "rc", "flag": rung, "rc_rank": rank, "path": p,
                        "bank": bank, "num": num}
            # Newest first: inside the rung, the gate beyond it (as rare as
            # the rung), a cave never entered honestly, then Route 30 twice.
            caps = [cap(1, 26, 2), cap(2, 26, 11), cap(3, 3, 78),
                    cap(4, 26, 1), cap(5, 26, 1)]
            kept = a._filter_rc_captures(caps, seeded=False, job=None)
            self.assertEqual([(c["num"], c["rc_rank"]) for c in kept], [(1, 1), (1, 2)])
            for c in caps[:3]:
                self.assertFalse(os.path.exists(c["path"]))

            # Stored trajectories are pruned the same way.
            a.rc_pool = {rung: [
                {"episode": 0, "caps": {1: cap(1, 10, 7), 2: cap(2, 26, 1)}},
                {"episode": 1, "caps": {1: cap(1, 10, 8)}},
            ]}
            a.rc_level = {rung: 2}
            a._prune_rc_pool_origins()
            self.assertEqual(len(a.rc_pool[rung]), 1)
            self.assertEqual(list(a.rc_pool[rung][0]["caps"]), [1])
            self.assertEqual(a.rc_level[rung], 1)

    def test_seeded_rings_kept_only_for_frontier_rungs(self):
        a = _agent_stub()
        a.rc_frontier_targets = [map_rung_id(26, 2)]
        for _ in range(4):
            a._honest_map_history.append({map_rung_id(26, 1)})
        with tempfile.TemporaryDirectory() as d:
            def cap(flag):
                p = os.path.join(d, f"{flag}.state")
                open(p, "wb").write(b"x")
                return {
                    "kind": "rc", "flag": flag, "rc_rank": 1, "path": p,
                    "bank": 26, "num": 1,
                }
            caps = [cap(31), cap(map_rung_id(26, 2))]
            kept = a._filter_rc_captures(caps, seeded=True, job=None)
            self.assertEqual([c["flag"] for c in kept], [map_rung_id(26, 2)])
            self.assertFalse(os.path.exists(caps[0]["path"]))
            # An RC job's own rung is dropped even from an honest episode.
            kept = a._filter_rc_captures([cap(31)], seeded=False, job=(31, 2))
            self.assertEqual(kept, [])

    def test_rc_pool_evicts_and_deletes_oldest(self):
        a = _agent_stub()
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for t in range(3):
                caps = []
                for rank in (1, 2):
                    p = os.path.join(d, f"t{t}_k{rank}.state")
                    open(p, "wb").write(b"x")
                    paths.append(p)
                    caps.append({"kind": "rc", "flag": 31, "rc_rank": rank, "path": p})
                a._ingest_rc_captures(caps)
            self.assertEqual(len(a.rc_pool[31]), 2)
            self.assertFalse(os.path.exists(paths[0]))
            self.assertTrue(os.path.exists(paths[-1]))

    def test_rank_widens_on_success_and_narrows_on_failure(self):
        a = _agent_stub()
        a.rc_targets = [31]
        a.rc_pool = {
            31: [{"episode": 0, "caps": {r: {"path": f"/x{r}"} for r in (1, 2, 3)}}]
        }
        a.rc_level = {31: 1}
        _cap, flag, rank = a._pick_rc_snapshot()
        self.assertEqual((flag, rank), (31, 1))
        for _ in range(4):
            a._record_rc_outcome(31, 1, True)
        self.assertEqual(a.rc_level[31], 2)
        for _ in range(4):
            a._record_rc_outcome(31, 2, False)
        self.assertEqual(a.rc_level[31], 1)


class TestFrontierRungEntryReward(unittest.TestCase):
    def _enter(self, target=True, rate=0.0, seeded=False):
        archive = VisitArchive()
        rung = map_rung_id(26, 2)
        archive.set_checkpoint_competence({rung: rate})
        archive.set_rc_targets([rung] if target else [31])
        rw = Rewards(
            _cfg(
                checkpoint_competence_gating=True,
                checkpoint_progress_reward=40,
                frontier_rung_entry_reward=40,
                episode_seeded=seeded,
            ),
            visit_archive=archive,
        )
        rw.calculate_reward(_ev(map_bank=26, map_num=1), "")
        r, _ = rw.calculate_reward(_ev(map_bank=26, map_num=2), "")
        bd = rw.get_episode_breakdown()
        # Leave and re-enter: paid once per episode.
        rw.calculate_reward(_ev(map_bank=26, map_num=1, x=5), "")
        r2, _ = rw.calculate_reward(_ev(map_bank=26, map_num=2, x=5), "")
        return float(r), float(r2), bd["map_rung"]

    def test_targeted_map_pays_once_at_checkpoint_price(self):
        r, r2, booked = self._enter()
        self.assertAlmostEqual(r, 40.0)
        self.assertAlmostEqual(booked, 40.0)
        self.assertAlmostEqual(r2, 0.0)

    def test_untargeted_map_pays_nothing(self):
        self.assertAlmostEqual(self._enter(target=False)[0], 0.0)

    def test_price_recedes_with_honest_rate(self):
        self.assertAlmostEqual(self._enter(rate=0.8)[0], 20.0)


class TestMapRungCapture(unittest.TestCase):
    def _env(self, targets, start_map=(26, 1)):
        from PoliwhiRL.environment.gym_env import PyBoyEnvironment

        e = types.SimpleNamespace()
        e._rc_enabled = True
        e._goexplore_snapshot_dir = "/unused"
        e._rc_states = deque([{"state_bytes": b"s", "step": 0, "env_vars": {}}])
        e._rc_captured_flags_this_episode = set()
        e._rc_maps_seen_this_episode = set()
        e.reward_calculator = types.SimpleNamespace(checkpoint_fires=lambda: [])
        e.visit_archive = VisitArchive()
        e.visit_archive.set_rc_targets(targets)
        e.saved = []
        e._save_frontier_state = lambda *a, **k: e.saved.append(k["flag"])
        e.steps = 1
        e._last_env_vars = _ev(map_bank=start_map[0], map_num=start_map[1])
        e.capture = types.MethodType(
            PyBoyEnvironment._maybe_capture_reverse_curriculum, e
        )
        return e

    def test_first_entry_into_a_targeted_map_captures_once(self):
        route31 = map_rung_id(26, 2)
        e = self._env([route31])
        e.capture()  # start map at step 1: never a fire
        e.steps = 600
        e._last_env_vars = _ev(map_bank=26, map_num=2)
        e.capture()
        e.capture()
        self.assertEqual(e.saved, [route31])

    def test_starting_inside_the_target_map_is_not_a_fire(self):
        route31 = map_rung_id(26, 2)
        e = self._env([route31], start_map=(26, 2))
        e.capture()
        e.steps = 600
        e.capture()
        self.assertEqual(e.saved, [])


class TestFrontierPathFallback(unittest.TestCase):
    def _env(self, local, field_node_dist):
        from PoliwhiRL.environment.gym_env import PyBoyEnvironment

        e = types.SimpleNamespace()
        e._frontier_path_fallback = True
        e.calls = 0
        start = ((26, 1), 5, 5)
        north = ((26, 1), 5, 4)
        field = (
            {start: field_node_dist, north: field_node_dist - 1 if field_node_dist else 0},
            {start: [(north, 0, -1, 1)], north: []},
        )

        def run_frontier(rc, ev):
            e.calls += 1
            return field[0].get(start), field, start

        e._run_frontier = run_frontier
        rc = types.SimpleNamespace(frontier_direction=lambda ev: list(local))
        return e, rc, types.MethodType(PyBoyEnvironment._frontier_dir, e)

    def test_local_gradient_wins(self):
        e, rc, fn = self._env([1.0, 0.0, 0.3], 3)
        self.assertEqual(fn(rc, _ev()), [1.0, 0.0, 0.3])
        self.assertEqual(e.calls, 0)

    def test_dead_local_sense_descends_field_and_keeps_saturation(self):
        e, rc, fn = self._env([0.0, 0.0, 1.0], 3)
        self.assertEqual(fn(rc, _ev()), [0.0, -1.0, 1.0])

    def test_on_frontier_keeps_local_reading(self):
        e, rc, fn = self._env([0.0, 0.0, 1.0], 0)
        self.assertEqual(fn(rc, _ev()), [0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
