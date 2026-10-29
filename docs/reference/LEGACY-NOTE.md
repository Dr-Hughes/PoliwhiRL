# Legacy note: `Reward Documentation/rewards.md`

What this page is for: to record, in one short place, that
`Reward Documentation/rewards.md` describes a **superseded** reward scheme and is
kept only as historical context, so nobody implements against it by accident.

**Sources**: `Reward Documentation/rewards.md:1-30`,
`configs/default_configs/reward_settings.json:1-32`,
`docs/subsystems/rewards.md`.

## Superseded: the coordinate/location scheme

`Reward Documentation/rewards.md` opens with goals expressed as integer tuples —
`[[9, 1, 6, 50]]`, `[[8, 4, 6, 50], [9, 4, 6, 50]]`, and so on
(`Reward Documentation/rewards.md:3-11`) — and defines them as
`x, y, map_location, reward`, where "reward is the amount of reward given
(though this isn't currently used and all rewards are equal)"
(`Reward Documentation/rewards.md:13-19`). Its body then walks through named
rooms ("6 The Player's Downstairs House",
`Reward Documentation/rewards.md:22-30`).

That model no longer describes how this repository rewards progress:

- Rewards are configured as **named channels with magnitudes**, not as
  coordinate tuples: `flag_progress_reward`, `checkpoint_progress_reward`,
  `seeded_checkpoint_progress_reward`, `honest_checkpoint_reward_floor`,
  `map_goal_reward`, `pokedex_*_reward`, `key_item_pickup_reward`,
  `battle_*`, `first_trainer_battle_*`, `level_up_reward`,
  `frontier_novelty_bonus`, `new_map_reward`, `new_bank_reward`,
  `whiteout_penalty`, `step_penalty`, `reward_clip`
  (`configs/default_configs/reward_settings.json:2-31`).
- Goals are **declarative typed objects**, not positions:
  `{"type": "flag", "flag_num": …}`,
  `{"type": "map", "map_bank": …, "map_num": …}`,
  `{"type": "pokedex", "kind": "owned"|"seen", "threshold": …}`
  (`configs/stages/open_world.json:83-96`, `configs/inference.json:15-19`,
  `configs/evaluate_reward_system.json:13-17`).
- Progress is identified by **event-flag number** and by the full
  `(map_bank, map_num)` pair — never by a bare `map_location` integer or a
  guessed name (`.opencode/skills/training-run-analysis/SKILL.md:140-141`).

## Current account

The authoritative description of the live reward system is
**[docs/subsystems/rewards.md](../subsystems/rewards.md)**. Read that for
channel semantics, decay and honest/seeded scaling; read
[operations/run-triage](../operations/run-triage.md) for how to tell whether
those channels are behaving.

## What to do with the legacy file

Nothing. `Reward Documentation/` — `rewards.md` and its `Image Examples/`
folder — is **retained in place** as legacy context and must not be deleted,
moved or renamed as part of this documentation pass. It remains useful for one
thing: understanding how the earliest reward experiments were framed, and why
the scheme moved off hand-picked coordinates onto flags and typed goals.

## Known gaps

- The exact commit that abandoned the coordinate scheme was not identified; the
  supersession here is inferred from the fact that no shipped config expresses
  goals as `[x, y, map_location, reward]` tuples.
- `Reward Documentation/rewards.md` was read only through line 30 of its 98;
  later sections may contain further superseded detail (or, possibly, still-live
  rationale) not summarised here.
- Whether `Reward Documentation/Image Examples/` is referenced by any code was
  not checked.
- `docs/reference/reward-locations-legacy.md` (the RAM-location legacy page) is
  written by a separate pass and did not exist when this note was authored.
