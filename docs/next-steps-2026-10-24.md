# Next steps — catch, gym win, PokéCenter heal (2026-10-24)

Standalone forward plan from the 2026-10-23/24 review passes. The full review
is `detailed_feedback.md` at the repo root; the project's standing backlog and
dated decision history live in `docs/roadmap.md` (this file deliberately does
not edit either). Every number below was re-derived from
`Training Outputs/open_world/Results/metrics/training_metrics.json` at
2026-10-24 08:44 (940 episodes / 3,428 rollouts / 12 h process time) or from
`Training Outputs/open_world/Checkpoints/info.pth`; re-derive before acting,
the run is live.

## 1. Where the run is

The unblock set shipped on 2026-10-23 (entropy-gradient weight, lr 1e-4 +
constant schedule + no per-restart cosine rebuild, `/255` pixels,
entropy-offset restart continuity, balancer activity-multiplier deleted,
`event` split into its own domain, Go-Explore captures 3 → 6) and the fresh
run on top of it is the best this project has produced:

| Signal | Value | Read |
|---|---|---|
| Honest coverage | maps 6.1 → 18.5, cells 106 → 467; archive 1,121 cells; pool 110 | best ever |
| Story | first honest fires past probe rung 4 in project history: flag 30 ep 120; flags 31/65/66 ep 304 | the old curriculum gate is open |
| Catches / trainer wins | 45 catches; 249 episodes with a trainer win | skills exist but are mis-measured (finding 1) |
| Probe ladder | rungs 5–6 at 1.0 (rollouts 2000/2800), 0.0 at 2400/3200 | threshold artefact of an 8,192-step probe budget, not instability (finding 4) |
| Entropy | 1.34 → 0.74 nats (min 0.385), coef fixed 0.02 | sharpening, not collapse; behaviour-policy entropy unexported |
| Optimiser | `approx_kl` mean 0.0214, clip ≤ 0.126 | F2 acceptance band met |
| LR | **5e-5** — halved once at rollout 285 by the exact-KL hard limit, one-way ratchet | finding 3 |
| Balancer | `domain_scale_adjustment` = 68 % of delivered honest reward, monotonic 11 → 600 per 100-ep bucket | finding 2 — top reward-side risk |

## 2. Findings, load-bearing order

1. **Catch classification is same-step, so `new_species_catches` is blind.**
   `rewards.py:1527-1531` compares a `party_size` delta against a
   `pokedex_owned` popcount delta in the *same sampled action*; one action is
   15 held frames of 90 (`gym_env.py:532-533`), so dex-and-party deltas in
   different sampled actions book a genuine new-species catch as a duplicate.
   Live: 45 catches, `new_species_catches` = 0, yet 80 episodes ended with
   `pokedex_owned` ≥ 3. This mislabels the counter, the
   `Caught First Wild Pokemon` threshold checkpoint and `event_counts.png`.
   It does **not** mean new species pay nothing — they pay 30 + 8 through the
   untouched pokedex path. This is a code cause behind the roadmap's
   long-open "zero new-species catches" item; the reachability-vs-choice half
   of that item is still open and is settled by its deferred instrumentation
   (catch events × per-episode `unique_maps`).
2. **Balancer amplification is the majority of delivered reward and growing.**
   Per-fire normalisation turns "many small fires" into "many unit fires", so
   delivered reward tracks fire *count*: `frontier` 37.6 → 160.5 and `battle`
   24.5 → 62.9 raw while every sparse story channel halved (`flag` 4.4 → 0.8,
   `pokedex` 16.4 → 5.9, `checkpoint_honest` 26.3 → 15.1). Any new reward price
   added now gets renormalised away before it teaches. This is
   `detailed_feedback.md` F5 measured worse than reported.
3. **The LR ratchet.** `_minibatch.py:145-150` halves a multiplier on exact-KL
   breach; nothing restores it and it persists across restarts
   (`ppo_state.pth`). 1e-4 has been 5e-5 for ~11 h; the trigger value
   (`base_post_kl`) is computed but never exported, so the breach is
   unobservable in the metrics JSON.
4. **The probe ladder's ruler is too short.** `probe_episode_length` 8,192 vs
   recorded honest first-fire costs of 11,052–22,496 actions
   (`checkpoint_recordings` fire steps): rungs 5–9 are unmeasurable and the
   1.0/0.0 flips are threshold behaviour. Exported `probe_goal_rates` is the
   *unassisted* base-policy pass while training data comes from the
   temperature-assisted behaviour policy.
5. **The live process depends on an uncommitted `break → continue`** at
   `vec_ppo_agent.py:720-728`. HEAD still breaks the training loop on a
   hard-KL rejection (a 19:48 process died from it; the live 20:49 process
   survived rollout 285 only because of the working-tree fix). Any restart
   path that checks out HEAD reintroduces a run-killing bug.
6. **`pokeball_count` byte semantics unverified.** `gym_env.py:506` scales /99;
   `docs/reference/ram-mapping.md:327` documents /25; the address sits in the
   pocket-*entry*-count family. The "scarce resource" feature may be reading an
   entry count at ≤ 0.12 of its scale.

## 3. Ordered path to the three skills

Do 1–2 before pricing anything (finding 2). Each step: guard, then the metric
that proves it landed.

1. **Clamp the balancer scale factor** — e.g. `[0.25, 4.0]` in
   `DomainRewardBalancer.scale` (`domain_reward_balancer.py`), or exempt the
   sparse domains from the running-mean divisor. Guard: caps and per-channel
   depletion untouched. Proof: `domain_scale_adjustment` share of
   `delivered_total` < 20 % while `flag` + `checkpoint_honest` raw means stop
   falling.
2. **Commit the verified working tree** (finding 5). Includes the unblock set,
   the `continue` fix and the test additions; the run currently depends on
   uncommitted state.
3. **Catch — fix the measurement, keep the price at zero.** Windowed
   new/duplicate classification (dex rose within N actions of the party delta,
   or diff the dex at battle end) at `rewards.py:1527-1531`; then run the
   deferred catch × `unique_maps` instrumentation. **Keep
   `duplicate_catch_reward: 0` and `catch_reward_episode_cap: 0`**: commit
   `10ddf8d` zeroed them deliberately (balls are scarce and non-renewing
   mid-episode; paying duplicates invites burning stock on junk) and that
   rationale holds until a money/Pokémart restock incentive exists. New-species
   catches already pay through the pokedex path — what was broken was the
   counter and the recording, not the price. Proof: `new_species_catches`
   nonzero, a `Caught First Wild Pokemon` recording, and `event_counts.png`'s
   new-vs-duplicate split stops being all-duplicate.
4. **Heal — potential-based shaping only.** Φ = `party_hp / 1000`, reward
   γΦ′ − Φ as a new `heal` source. The roadmap's 2026-10-23 design review
   rejected a heal reward; that rejection was of a *deficit-gated flat bonus*
   and correctly identified its damage→heal farm. PBRS closes that farm by
   construction: re-healing at full HP pays exactly 0, and spending HP to
   collect the bonus nets ≤ 0 over the cycle, so no new dedup is needed.
   Proof: nonzero `heal` source with `whiteout` still ~0 and no rise in
   re-entry counts of maps 10:10 / 26:5 in the visit archive.
5. **Gym — seed from the bank-10 pool; it is already there.** 41 of 110
   frontier-pool entries are bank 10, including six cells *inside* VIOLET_GYM
   (10:7 at (4,15), (4,1), (3,1), (1,1), (1,2), (1,4), one a flag capture) and
   the PokéCenter (10:10), so no new capture rule is needed; the rarity-weighted
   selector already favours them. Nudge `goexplore_seed_fraction` 0.4 → 0.5
   only if bank-10 draws prove too rare. Flag 1213 already pays through the
   durable-checkpoint path (`PoliwhiRL/checkpoints.py:42`); the binding
   constraint is budget inside the gym, not signal. Proof: assisted-probe
   "Reached Violet City" > 0, then a 1213 fire.
6. **Macro-actions (deferred F6) after 3–5 land.** The gym *walk* fits an
   episode (flag 31 fired at action 11,052 of 24,576); the gym *fight* —
   multi-thousand-action approach plus a turn-based battle under γ=0.999's
   ~1,000-action credit horizon — is what hold-until-blocked macro-actions
   unlock. Budget the knock-on work: `actions.steps` replays, the PNG recorder
   and the checkpoint-recording tooling all encode the old action semantics.
7. **Verify the `pokeball_count` byte** against a savestate with a known ball
   count before any ball-budgeting reward work builds on it (finding 6).

Alongside, cheap and independent: `probe_episode_length` 8,192 → 24,576 and
export `assisted_probe_goal_rates`, `rollout_kl_rejected`, `base_post_kl`,
`behaviour_post_kl` into the metrics JSON (findings 3–4); raise `ppo_hard_kl`
to ~0.1 or add multiplier recovery (finding 3).

## 4. Explicitly not recommended

- **Enabling `allow_menus_walking`.** In-battle BAG is already D-pad-reachable
  (45 catches prove it); un-masking START/SELECT mid-run changes the mask PPO
  re-derives from stored RAM, breaking the importance-ratio invariant on stored
  rollouts.
- **Re-pricing duplicate catches** before a restock incentive exists (step 3).
- **A directed Violet/gym stage** before bank-10 seeding is tried (step 5).
- **A flat or deficit-gated heal bonus** — the 2026-10-23 rejection stands for
  those; only the potential-based form (step 4) is proposed.
