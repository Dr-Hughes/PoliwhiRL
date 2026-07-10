# Next steps — generalized reward + freeform exploration test

## Goal

Get the agent through the full quest (leave house → mom → starter → Route 29 →
Cherrygrove → Mr. Pokémon's house → egg → back to Elm, handling the rival
fight) **without hand-authoring a map/flag target for every stage** — because
that doesn't scale past this quest, and risks the model only ever following a
golden path instead of learning to explore.

Stages 1 (leave house) and 2 (get starter) are done and verified: stage 1 hit
65% success in 15 rollouts, stage 2 hit 61% in 327 rollouts, both confirmed
via the actual reward breakdown in the saved checkpoints, not just the
training log. Checkpoints: `Training Outputs/01_leave_house/Checkpoints/best`,
`Training Outputs/02_get_starter/Checkpoints/best`.

## Issues diagnosed

1. **Reward was golden-path by construction.** `flag_progress_reward` /
   `map_goal_reward` / pokedex / level only paid for whatever a stage's
   `goals` list happened to name. Every new stage needed a human to look up
   a flag number and add it to a config — nothing else the game offers
   (badges, HMs, other species, items) was ever reward-eligible.
2. **Prior freeplay collapsed** (observed before this session): walked into
   corners, lost previously-learned skills. Likely cause: zero milestone
   reward in that design + a run-wide persistent novelty archive that
   saturates and flattens the whole reward landscape (documented in
   AGENTS.md §10a) + an abrupt reward-distribution shift on stage transition
   with no mechanism to notice regression on old skills.

## What changed (implemented, not yet run)

1. **Flag/pokedex/level/key-item reward is now always-on**, not
   stage-gated. `Rewards._global_flag_progress_bonus` pays for ANY 0→1
   transition among the ~50 curated flags in
   `gym_env._DERIVED_FLAG_TABLE` (badges, HMs, story milestones, egg
   quest, etc.) regardless of what a stage's `goals` list names. Pokedex/
   level/key-item deltas are similarly unconditional now
   (`_global_pokedex_bonus` / `_global_level_bonus` / `_global_key_item_bonus`).
   A stage's `goals` list still exists, but only drives that stage's
   termination/success-rate metric — it no longer gates reward.
   `map`-type goals are the deliberate exception: reaching a specific
   coordinate has no general substitute, so `map_goal_reward` still only
   pays for an explicitly configured target.
   - Added flag 1735 (`talked_to_mom` / `EVENT_PLAYERS_HOUSE_MOM_1`) to the
     curated table properly. **This bumps `RAM_OBS_DIM` by one — stage 1
     and 2's saved checkpoints are now dimensionally incompatible and must
     be regenerated before anything downstream can load them.** Re-running
     `configs/stages/first.json` then `second.json` is fast (~15 min total
     based on the last run).
2. **Regression probe** (`VecPPOAgent._run_probe`, config keys
   `probe_enabled`/`probe_frequency`/`probe_episodes`/`probe_episode_length`/
   `probe_goals`/`probe_label`): every `probe_frequency` rollouts, runs
   `probe_episodes` short episodes from the true starting save-state with
   current weights, checks success against a **fixed earlier skill**,
   independent of whatever the current stage's own (possibly goal-less)
   objective is. This is the only thing that can catch catastrophic
   forgetting — the stage's own metrics only see its own goal.
   Logged to `episode_data["probe_success_rate"]` /
   `["probe_rollout_idx"]` (also shown live in the progress bar as
   `probe_sr`) — not yet in a plotted PNG, just raw series in `info.pth` /
   `training_metrics.json`. Runs single-threaded on the main process, so it
   adds real wall-clock overhead — keep `probe_frequency`/`probe_episodes`
   modest.
3. **`configs/stages/third_freeform.json`**: loads stage 2's checkpoint,
   **empty `goals` list** (no hand-picked map/flag target for Cherrygrove/
   Route 30/Mr. Pokémon's house — relies entirely on the now-general
   flag/pokedex/level reward + per-episode novelty + capped battle reward
   to find its own way there). Probe configured to re-check stage 2's
   `pokedex_owned >= 1` skill every 20 rollouts. `battle_reward_episode_cap`
   lowered to 15 (bias traversal over fighting, matching the existing
   stages 4/5/6 convention). Episode length 3072, 1500-rollout budget.
   The original directed `third.json`/`fourth.json`/`fifth.json`/`sixth.json`
   are untouched, kept as a comparison baseline — this is an experiment,
   not a replacement.

## What to run next (not yet started — needs explicit go-ahead)

1. Re-run `configs/stages/first.json` then `configs/stages/second.json`
   (RAM_OBS_DIM changed, see above).
2. Run `configs/stages/third_freeform.json`.

## What to examine

- `episode_unique_maps` trend — should climb/plateau at a real number, not
  collapse to 1-2 (corner-walking signature).
- `rollout_policy_entropy` trend — collapsing toward ~0 is the degenerate-
  repetition signature; pinned near max means it's not learning anything.
- `episode_reward_sources` — do `flag`/`pokedex` entries appear for
  milestones this stage never configured as its own goal (direct proof the
  general mechanism is doing work unprompted)? Does `frontier`/`new_map`
  stay > 0 across the whole run (still discovering things)?
- `probe_success_rate` — does it stay near stage 2's ~61%, or does it drop
  (catastrophic forgetting of "get the starter" while exploring further)?
- Does it ever reach Cherrygrove (26,3) / Route 30 (26,1) / Mr. Pokémon's
  house (26,10) at all, per `episode_unique_maps`/recorded runs — without
  ever being told to?

## Expected outcome / what would falsify this approach

Success: unique-maps keeps growing, entropy stays moderate (not collapsed),
probe success rate holds roughly steady or grows, and flag/new-map rewards
keep firing well past what stage 2 covered — ideally reaching Cherrygrove or
beyond on its own.

Failure mode to watch for specifically: if `probe_success_rate` degrades over
the run, that's catastrophic forgetting recurring even with the improved
reward design — next lever would be to reduce how sharply the reward
distribution shifts between stages, not to add more hand-picked goals back.

If it explores geography fine but never touches menus/items/battles
meaningfully: the next lever is novelty keyed on (map, script/UI-state)
combinations, not just position — frontier novelty is purely geometric right
now and has no notion of "first time I've seen this menu."
