# Future Ideas

Deferred experiments and architectural directions, kept here so they aren't lost.

## Save-state checkpointing + backward-expanding start distribution

**Why.** The current training base is *from-scratch + milestone-ladder*: every
episode runs from the true game start, and a ladder of milestone rewards
(leave-house → starter → route29 → cherrygrove → Mr-Pokemon) provides the
gradient breadcrumbs. This deletes the fragile action-replay seed chain and the
catastrophic-forgetting problem entirely, at the cost of sample efficiency
(every episode pays the full traversal). It is the right base while the route is
short (~10% of the game).

It will **not** scale indefinitely. As goals get deeper, the route a from-scratch
agent must traverse to reach fresh signal grows, episodes get long and expensive,
and with γ=0.999 a goal N steps away is discounted to ~0.999^N at the start
(~0.13 at N=2000, ~4e-5 at N=10000) — eventually exploration cannot bridge the
gap to a new milestone and the curriculum stalls. The milestone breadcrumbs
mitigate this (the agent climbs rung by rung) but only while consecutive rungs
stay inside the effective horizon.

**The escape hatch (preferred over re-introducing action-replay).** When a
milestone gap proves unbridgeable from scratch:

1. **Save-state checkpointing.** Snapshot the *emulator state* at the last
   cleared milestone and reset a fraction of episodes to that snapshot. Unlike
   action-replay this is **O(1) per reset** regardless of route depth, has no
   trajectory-length growth, and no single-point-of-failure seed chain.
2. **Backward-expanding start distribution** (Salimans & Chen, *Learning
   Montezuma's Revenge from a Single Demonstration*, 2018). Start episodes near
   the frontier milestone, then sweep the start point backward toward the true
   game start as each segment is mastered. This keeps sparse deep goals reachable
   *and* keeps earlier segments getting gradient (controls forgetting) *and*
   converges to from-scratch capability — the best of both.

**CRITICAL implementation note.** A PyBoy save-state must be paired with a
**matching snapshot/restore of the `Rewards` object** (`PoliwhiRL/environment/
rewards.py`). The reward calculator holds process-lifetime / per-episode state
that must be consistent with the emulator state, or a restored episode will
mis-reward:
- `visit_archive` (global cell + map ledgers),
- `pokedex_seen` / `pokedex_owned`, `_key_items_count`,
- the map first-discovery ledger and `GoalsManager` per-episode trackers
  (`_map_fired`, `_map_initial`, `_maps_seen_this_episode`, `_flag_progress`,
  pokedex/level/xp counters),
- battle / level trackers, per-episode novelty sets and the new
  `_intrinsic_reward_paid` / `_battle_reward_paid` cap accumulators.

Design the save-state + reward-state as **one atomic checkpoint** (write both,
restore both together). A save-state that restores the emulator but not the
reward object will, for example, re-pay first-discovery bonuses for maps already
seen, or fire a map goal whose `_map_initial` snapshot no longer matches.

## Other candidates (lower priority)
- **Rehearsal episodes** if the from-scratch eval (`configs/eval_endtoend.json`)
  shows per-milestone decay: mix in a fraction of episodes that deliberately
  exercise earlier segments. Only pay this cost once the eval justifies it.
- **Event-flag progress as a first-class signal.** `flag_progress_reward` is
  currently OFF by default (WIP). Story flags are a far more robust "this beat
  happened" signal than map-reach goals for the full game; wire + test the
  detection (`event_flags.asm`, `verify_flags_from_states.py`) and re-introduce
  the reward deliberately, sized into the capped economy.
- **Per-stage tuning of the reward economy.** `intrinsic_reward_episode_cap`,
  `intrinsic_reward_weight`, `scaler_min_std_int`, `step_penalty` and per-stage
  `episode_length` are economy knobs; as stages are added, the per-stream
  variance shifts and a single global setting will drift — expect to tune per
  stage and watch `reward_sources` + the end-to-end eval.
