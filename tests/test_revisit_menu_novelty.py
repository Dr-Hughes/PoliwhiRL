# -*- coding: utf-8 -*-
"""Story-flag revisit re-reward and menu/UI-state novelty.

Pinned behaviours:

- Tile coverage pays full for a genuinely-new-this-episode cell and 0 for a
  cell already paid this reward epoch.
- A story-flag flip re-opens coverage: a cell already visited this episode
  then re-pays at ``revisit_novelty_scale`` (discounted), while a
  genuinely-new cell still pays full. Strict ordering new > revisit > 0.
- The re-open does NOT corrupt the observation history: local_visited_mask /
  frontier_direction read the true (monotonic) episode history, so a re-opened
  cell still reads as visited.
- Menu/UI novelty pays once per distinct (battle_type, ui_byte,
  map_handler_byte) context per epoch, revisit-discounted the same way, and is
  a no-op when the bonus is 0.
"""
import numpy as np

from PoliwhiRL.environment.rewards import Rewards


def _cfg(**overrides):
    cfg = {
        "episode_length": 1000,
        "frontier_novelty_bonus": 1.0,
        "menu_novelty_bonus": 0.5,
        "revisit_novelty_scale": 0.5,
        "flag_reopens_novelty": True,
        "step_penalty": 0.0,
        "reward_clip": 50,
        "frontier_lifelong_decay": False,
    }
    cfg.update(overrides)
    return cfg


def _vars(x, y, ui=0, bt=0):
    return {
        "map_bank": 24,
        "map_num": 3,
        "X": x,
        "Y": y,
        "script_active": False,
        "battle_type": bt,
        "ui_byte": ui,
        "map_handler_byte": 0,
    }


def test_revisit_pays_discounted_after_flag_reopen():
    rw = Rewards(_cfg())
    assert rw._frontier_novelty_bonus(_vars(5, 5)) == 1.0  # new
    assert rw._frontier_novelty_bonus(_vars(5, 5)) == 0.0  # paid this epoch
    assert rw._frontier_novelty_bonus(_vars(6, 5)) == 1.0  # new
    # Simulate a story-flag re-open (what _global_flag_progress_bonus does).
    rw._novel_cells_this_episode.clear()
    assert rw._frontier_novelty_bonus(_vars(5, 5)) == 0.5  # revisit -> half
    assert rw._frontier_novelty_bonus(_vars(5, 5)) == 0.0  # paid again
    assert rw._frontier_novelty_bonus(_vars(9, 9)) == 1.0  # truly new -> full


def test_revisit_scale_configurable():
    rw = Rewards(_cfg(revisit_novelty_scale=0.25))
    rw._frontier_novelty_bonus(_vars(5, 5))
    rw._novel_cells_this_episode.clear()
    assert rw._frontier_novelty_bonus(_vars(5, 5)) == 0.25


def test_no_reopen_when_disabled_is_pure_per_episode():
    rw = Rewards(_cfg(flag_reopens_novelty=False))
    assert rw._frontier_novelty_bonus(_vars(5, 5)) == 1.0
    # Without the flag re-open the paid gate is never cleared, so a revisit
    # stays at 0 (identical to the old flat per-episode coverage).
    assert rw._frontier_novelty_bonus(_vars(5, 5)) == 0.0


def test_visited_mask_stays_true_across_reopen():
    rw = Rewards(_cfg())
    rw._frontier_novelty_bonus(_vars(5, 5))
    rw._novel_cells_this_episode.clear()  # flag re-open
    mask = rw.local_visited_mask(_vars(5, 5))
    # Centre of the egocentric window is the current cell — still "visited".
    assert mask[len(mask) // 2] == 1.0


def test_menu_novelty_new_repeat_and_revisit():
    rw = Rewards(_cfg())
    assert rw._menu_novelty_bonus(_vars(5, 5, ui=7, bt=1)) == 0.5  # new ctx
    assert rw._menu_novelty_bonus(_vars(5, 5, ui=7, bt=1)) == 0.0  # paid
    assert rw._menu_novelty_bonus(_vars(5, 5, ui=1, bt=1)) == 0.5  # new ctx
    rw._novel_menu_states_this_episode.clear()  # reopen
    assert rw._menu_novelty_bonus(_vars(5, 5, ui=7, bt=1)) == 0.25  # revisit


def test_menu_novelty_off_by_default():
    rw = Rewards(_cfg(menu_novelty_bonus=0.0))
    assert rw._menu_novelty_bonus(_vars(5, 5, ui=7, bt=1)) == 0.0


def test_party_growth_registers_milestone_for_seeding():
    rw = Rewards(_cfg())
    # First call seeds the party baseline (no milestone).
    rw._global_level_bonus(1, 5)
    # Growth to 2 registers a ("party_size", 2) milestone (no reward paid).
    reward = rw._global_level_bonus(2, 10)
    assert reward == 0.0
    assert ("party_size", 2) in rw._milestone_fires_pending
    # A shrink (whiteout / release) registers nothing new.
    rw._global_level_bonus(1, 5)
    assert ("party_size", 1) not in rw._milestone_fires_pending
