# Next stage: past Falkner (open_world_v3 sticking points)

Written 2026-10-29 against `open_world_v3` (branch `open-world-v3-first-gym`,
training code unchanged since `789cdba`), while the run was live. It lists
what stops the current best run from getting past gym 1, the evidence for
each point, and one recommended fix per point, in the order to try them.
Run status is in [`status.md`](status.md); the plan this run launched from is
[`next-from-scratch-plan.md`](next-from-scratch-plan.md). Where this page and
the code disagree, the code is right.

**How the evidence was gathered.** Counts come from
`Checkpoints/info.pth` at episode 330 (225 honest episodes) and
`tools/training_health_report.py`. Step-level claims come from replaying three
first-honest-hit recordings (`Runs/checkpoints/<label>/actions.steps`)
through the real env (`tools.llm_agent.game.Game` with
`configs/stages/open_world.json`). Each replay reproduced its episode: the
Falkner prefix ends with flag 1213 set and 1 badge. The three recordings are
1213-Beat-Falkner (episode 213, 16,254 steps), 0020-Has-Flash (episode 157,
40,198 steps; also the 7-owned episode) and 0045-Got-Togepi-Egg (episode 234,
24,171 steps; it beat Falkner at step 18,264). The repo has no tool that
replays `actions.steps` (`tools/llm_agent/replay.py` replays LLM runs only).
Adding one is part of step 0 below. Map/position runs come from
`Runs/N_goals_0/ep_*/trajectory.csv`. The Route 32 block was checked live with
`tools.llm_agent.drive` from a state cut out of the Falkner replay.

## Where the run is

| Signal | Value (honest episodes) |
|---|---|
| Flags, last 200-episode window | 30 92%, 39 74%, 31 58%, 65 57%, 66 48%, 20 6%, 1213 3% |
| Last 25-episode bucket (after the entropy-coef cut at update 2400) | 31 92%, 66 76%, unique maps 32.4, trainer wins 16.3 per episode |
| Median fire step | 30: ~4.9k, 31: ~13.2k, 66: ~13.4k, 20: ~22k, 1213: ~27k |
| Longest legs (median steps) | 39→31 7,000; 66→20 7,571; 27→30 4,271 |
| How episodes end (last 200) | stagnation 170, budget 27, battle stagnation 3 |
| After Falkner | 6 episodes; median 5,867 steps left, then 5 of 6 end on stagnation |
| Archive | 1,451 cells, 42 maps; 1 new cell in the last quarter; last honest discovery 96 episodes ago |
| Reward mix, last quarter | coverage 53.5%, story 28.3%, battle 14.9%, catching 1.7%, level 1.0% |
| Optimisation | entropy 1.43 nats (13% of rollouts in band); KL ~0.02; clip ~0.07 |

No honest episode has both Flash and Falkner. At 6% and 3% you would expect
fewer than one such episode by chance, so this says nothing about the two
competing. It does show how little budget is left once either fires.

## Sticking points, ranked by expected impact

### 1. The pre-badge world is exhausted, and the only way on is behind Falkner

**Evidence.**
- Walking south on Route 32 without the Zephyr Badge is blocked. Verified
  live: from a pre-Falkner Violet state (Falkner replay step 15,216), walking
  south stops at Route 32 (18,8) with "Wait up! What's the hurry? … Have you
  gone to the POKéMON GYM?", and the player is moved back to (18,6). This
  happens every time.
- Archive cells on Route 32 (`visit_archive.cells`, cell rows) show the
  same wall. Rows 0–3 have 24–49 honest visits each; every row from 4 to 42
  has exactly 1.
- That one visit is episode 234, which beat Falkner at step 18,264. It then
  reached ROUTE_32_POKECENTER_1F at 26,115 and UNION_CAVE_1F at 26,480
  (`discovery_log`). No seeded episode has entered either map
  (`seeded_map_counts`).
- Everything else reachable from Violet is a dead end before later story
  items. That covers the Ruins of Alph chambers, Earl's Academy, Dark Cave's
  entrance chamber, Route 36 up to Sudowoodo, and Sprout Tower 3F. All are in
  the archive.
- The Go-Explore pool (110 snapshots) is mostly these dead ends: Academy 15,
  Violet Gym cells 12, Violet PC 10, Research Centre 9, Dark Cave entrance 7.
  The post-Falkner state is one `checkpoint` entry among about 30 progress
  entries. `_pick_frontier_snapshot` (`PoliwhiRL/agents/PPO/vec_ppo_agent.py:1352`)
  weights each entry by 1/(1+map visit count), and Violet Gym has about 93
  visits.

**Diagnosis.** The archive plateau is not an exploration failure inside
reachable ground. The run has covered everything it can reach without the
badge. New ground needs, in order: (a) Falkner, at 3% honest; (b) budget left
after Falkner, a median 5.9k steps; (c) going back through fully-walked
Violet to Route 32. In (c) the coverage reward is silent, which is the Route 30
gap problem again. Go-Explore cannot cover for this, for two reasons. It
almost never seeds from the post-badge state. And a seeded episode starts
with an empty visit history (see point 5), so its frontier reward pulls it
around Violet instead of south.

**Fix (general).** Seed from the deepest progress captures, not from rare
locations. In `_pick_frontier_snapshot`, weight progress-pool entries by how
rarely the honest policy reaches that checkpoint
(`visit_archive.checkpoint_competence`, already tracked). A checkpoint the
honest policy rarely reaches, such as 1213, then gets most of the progress
seeds. This uses the same competence signal as checkpoint pricing and names
no map. Do it together with the visit-history fix (point 5), otherwise the
seeds re-farm Violet.

**Verify.** Seeded episodes from the 1213 capture entering Route 32 rows ≥ 4
and UNION_CAVE_1F (`seeded_map_counts`). Then honest entries to those maps. Add
"Reached Union Cave" (map 3:37) to `probe_goals` so the probe ladder shows it.

**Risk / farm check.** Seeded reward inflates. Judge only on honest
`unique_maps`/`unique_cells` and the probes, per AGENTS.md. Watch for
checkpoint-seeded reward rising while honest flag 31/66 rates fall: that
means seeds are crowding out the early legs. The `goexplore_progress_seed_fraction`
split already bounds how many seeds this can take.

### 2. A whiteout is free, so it replaces healing

**Evidence.**
- The three replays contain 16 whiteout warps (all party HP 0, then a map
  change with a full heal). The penalty paid on 1 (Flash replay, step 12,073).
  - Falkner: warps at 3,494, 5,180 and 15,215, none paid.
  - Flash: 10 warps, 1 paid.
  - Togepi: 3 warps, none paid.
- On every unpaid one, the step where HP reaches 0 has `script_active` set.
  Example: Falkner replay step 15,212, Sprout Tower 2F, "KRIS is out of
  useable POKéMON!". `_check_whiteout` only moves its baseline under a script
  (`PoliwhiRL/environment/rewards.py:1341`), as logged in
  `tools/llm_agent/bugs_found.md`.
- The warp then pays `resource_potential` +14 to +17 (Falkner replay steps
  3,494, 5,180, 15,215). That is the same refund a nurse heal earns.
- **The Falkner win follows a free whiteout.**
  - The lead (Quilava, Lv 15, the only party member) used up both damaging
    moves in Sprout Tower 2F: Tackle and Ember at 0 PP from step ~14,963.
  - It then fought 249 steps on Leer/Smokescreen.
  - It whited out at 15,212. Entering VIOLET_POKECENTER_1F at step 10,363 had
    set Violet as the respawn point, so the warp put it outside the Violet
    Pokémon Centre, healed, at 15,215.
  - It walked to the gym and won at 16,254.
- Nurse heals in the three replays: 1. That is Togepi replay step 10,561,
  Cherrygrove PC, at 86% HP / 42% PP, paying +7.25.
- The Flash episode walked 9,775 of its 25,743 free-walking steps (38%) with
  the lead fainted. It visited Cherrygrove PC 1F many times and never healed
  there.
- Paid whiteouts are rising. The share of honest episodes with any paid
  whiteout was 8–20% per bucket and is 52% in the last bucket. Paid ones are
  roughly 1 in 16, so the true rate is far higher. More trainer fights are
  causing more faints.

**Diagnosis.** Healing is visible to the policy:
- `party_hp_frac`, `party_pp_frac`, `lead_hp_frac`, `lead_attack_pp_frac` and
  `able_party_frac` are in the RAM vector (`PoliwhiRL/environment/gym_env.py`,
  `_PARTY_RESOURCE_KEYS`; `RAM.get_party_resources`), so the "PP blind" note
  is out of date;
- the replays show `lead_attack_pp_frac` reading 0.0 correctly once Tackle
  and Ember were empty.

Healing is also credited. `_resource_potential_shaping`
(`rewards.py:1314`) is PBRS on 0.5·(HP frac + PP frac), and a heal pays +Δφ.
But a whiteout pays the same +Δφ, costs nothing, and moves the player to the
last Pokémon Centre. Fainting is therefore a free heal plus fast travel, and
it beats walking to the nurse. Money is halved by the game, but money has no
reward coupling by design.

**Fix (general).**
1. Detect the whiteout under a script. Track the all-fainted transition every
   valid step, resetting the baseline only on a party-size change. Confirm it
   with the warp: party HP goes 0 → full on a map change. That guards against
   a garbage read at battle start.
2. On the confirmed whiteout warp, reset the resource-potential baseline to
   the post-warp value, so a heal earned by fainting refunds nothing. The
   potential lost on the way down then stands as the cost of fainting. That
   cost scales with how depleted the party was, and needs no new constant
   beyond the existing `whiteout_penalty`. A nurse heal keeps its +Δφ, so it
   becomes strictly better than fainting. No map, building or NPC is named.

**Verify.**
- Log `rc.whiteouts`, which is already counted, plus a "non-whiteout heal"
  counter in the terminal info. Report both in the health report.
- Scenario tests in `tools/llm_agent/scenarios.py`:
  - from a state one turn before a faint-out (cut from the Falkner replay,
    step ~15,205), the penalty pays once and `resource_potential` on arrival
    is 0;
  - from Cherrygrove PC (drive skill, "Pokémon Centers"), a nurse heal pays
    +Δφ.
- Regression guard: the honest flag 31 rate and probe "Gave Mystery Egg To
  Elm". `bugs_found.md` notes the free warp may be helping the walk back to
  Elm.

**Risk / farm check.**
- Making faints cost can teach battle avoidance, the old flee-reward
  failure. Watch honest trainer wins per episode, `battle` reward and
  `level`.
- A heal cannot be farmed: PBRS telescopes over any damage/heal cycle.
- Resetting the baseline only removes reward, so it cannot open a farm.

### 3. The navigation senses point into walls

**Evidence.**
- Wall bumps are common. A bump here means a free-walking direction press,
  already facing that way, into a tile whose ROM collision byte is a wall. In
  the three replays bumps are 24.5% (Falkner), 20% (Flash) and 16.5% (Togepi)
  of free-walking direction presses.
- The `trajectory.csv` proxy (same direction repeated with no move) gives
  10–24% across all 9 recorded episodes.
- The bumps earned nothing. Over 2,267 bumps in the Falkner replay the only
  source was `frontier_potential` −2.45 in total. Over 480 A presses facing a
  wall, the only source was `frontier_potential` −1.46, with no `talk` and
  no `menu`.
- The stuck temperature is not involved either. The stagnation clock never
  passed 0.136 in the Falkner replay, and was ≤ 0.045 in the Violet bump
  windows. That gives a sampling temperature of 1.2 or less
  (`ppo_model_implementation.py:55`).
- **Bumps follow `frontier_dir`.** The frontier vector points ≥ 0.5 into the
  pressed direction in:

  | Replay | Bumps | Successful moves |
  |---|---|---|
  | Falkner | 76% | 35% |
  | Flash | 66% | 31% |
  | Togepi | 60% | 33% |

  The exit sense explains fewer bumps (Falkner: 23% of bumps vs 33% of
  moves).
- `Rewards.frontier_direction` counts wall cells as unvisited
  (`rewards.py:2388`). A wall pocket never gets visited, so it pulls forever.
  The path-based direction (`gym_env.py:1385`) is only a fallback for when the
  window vector is exactly zero.
- **The case you saw before the gym (Falkner replay, Violet City, steps
  15,215–15,755, 540 steps).** These steps contain 157 bumps, 84 A, 59 no-op
  and 48 B presses, in three places:

  | Steps | Position | What the policy did | Sense reading |
  |---|---|---|---|
  | 15,288–15,351 and 15,450–15,483 | (24,27)–(24,29) | Faced the west wall | `frontier_dir` (−0.96, −0.28): into the wall |
  | 15,362–15,508 | (26,24), (28–29,24) | Faced the north wall | `frontier_dir` (−0.42, −0.90): into the wall |
  | 15,576–15,654 and 15,741–15,751 | (19–21,18) | Pressed up and A into the gym's front wall, one tile east of the door at (18,17) | Exit sense (−0.71, −0.71), 2 steps away |

  The gym door at (18,17) reads 0x71 in the collision field. On the gym wall
  the path is left then up, and `descent_direction`
  (`PoliwhiRL/environment/rom_collision.py:272`) sums 8 path moves into one
  unit vector, so an L-shaped path reads as a diagonal. Nothing tells the
  policy which axis comes first.
- The 59 steps at the gym's (5,10) (15,762–15,820) are a trainer's approach
  (Bird Keeper Abe), not walls. The 223 steps at (6,1) (16,032–16,254) are
  the Falkner fight and its dialogue.
- The biggest single case: Sprout Tower 1F, 1,531 steps, 878 bumps. Of those,
  960 steps were spent at (14,11)–(14,12) pressing right into the east wall,
  with `frontier_dir` around (0.5–0.9, 0.4–0.9). The unvisited cell (13,12)
  was one step left (steps 10,728–11,503).

**Diagnosis.** The two direction senses hand the policy a "go this way" that
is often a wall. The frontier sense does it because it ignores walkability.
The exit sense does it at L-shaped corners because it sums the path into a
diagonal. The policy follows them, and nothing it reads or earns penalises a
bump, so it pays in steps. Steps are the binding cost here: Falkner fires at
a median ~27k of the episode budget, and the deepest fire ever is at 90% of
it.

**Fix (general, no new dims).**
1. In `frontier_direction`, skip cells whose ROM collision is not walkable.
   `RomCollisionField` already decodes them for the 5x5 field.
2. In `descent_direction`, weight the path moves so the first one dominates,
   e.g. geometric decay. The vector then points along the next move. Both
   senses keep their meaning and size.

**Verify.**
- Add a per-episode wall-bump counter (the definition above) to the terminal
  info and the health report.
- Replay-based check: run the Falkner and Flash prefixes through the changed
  env and confirm `frontier_dir` no longer points into walls at Sprout Tower
  1F (14,12) and Violet (24,27). Confirm the exit sense at Violet (19,18)
  reads mostly left.
- In training: the honest bump share falls, steps from flag 66 to Violet /
  1213 fall, and flags 20 and 1213 rise.

**Risk.** A warm start sees the same features with new values, which is a
distribution shift. Run it as its own change and expect a short dip. There is
no reward change, so no farm risk.

### 4. The party is one Pokémon, and levelling a second happens only by accident

**Evidence.**
- Party composition over time (RAM, party struct):
  - **Falkner replay:** Cyndaquil/Quilava alone for all 16,254 steps,
    Lv 5 → 16. All 89 battles started with it.
  - **Togepi replay:** Quilava alone until Togepi at the last step.
  - **Flash replay:** six members by step 33,697. Chikorita→Bayleef Lv 16,
    Bellsprout Lv 10, Pidgey→Pidgeotto Lv 19, two Ledyba Lv 3–4, Rattata
    Lv 6.
- In the Flash replay, 141 of 261 battles started with a non-lead mon because
  the lead had fainted. There were 5 voluntary mid-battle switches. Pidgey
  levelled because the policy kept walking with a fainted lead (point 2), not
  because it chose to.
- `level` reward is paid per threshold of the party's **total** level,
  deduplicated per episode and depleted run-wide
  (`_global_level_bonus`, `rewards.py:1618`). It is suppressed across a
  party-size change.
- The honest `level` source fell from 24.9 to 6.1 per episode between the
  first and last quarter. Early totals are re-crossed every episode, so they
  have depleted.
- The policy sees `party_size` and the total level / 100 (`gym_env.py:472`),
  plus lead and aggregate HP/PP. It has no per-slot level, HP or fainted
  state.

**Diagnosis.** Nothing values the second or third mon:
- the total-level reward is nearly spent and does not care which mon levels;
- the observation cannot tell which slot is weak or fainted.

So a strong party comes only from fainting. That compounds point 2: a
one-mon party whites out the moment the lead faints.

**Fix (general).** Replace the total-level threshold reward with a potential
on the party's core: φ = c · Σ √level over the top 3 slots.
- Being concave, it pays more for a level on a weak mon than on the lead.
- Being limited to the top 3, it does not reward boxing junk catches.
- Being PBRS, catch→release or deposit→withdraw cycles telescope to ≤ 0, so
  the threshold re-cross farm in AGENTS.md cannot occur.

Keep the archive's `level_max` discovery logging. Add per-slot features
(level / 100, HP frac, fainted, damaging-move PP frac; 6 × 4) at the next
from-scratch run, because a dimension change will not warm-start
(`info.pth` `ram_dim` guard).

**Verify.**
- Per-episode top-3 levels and the share of battles started with a non-lead
  mon while the lead is healthy (voluntary use). Both should rise.
- `level` reward should stay small next to coverage.
- Farm check: per-episode maxima of `level`/`catch`/`pokedex`, and
  `duplicate_catches`, should stay flat.

**Risk.** PBRS gives a one-off gain for any catch that enters the top 3. That
is bounded at 3 slots and paid once per mon kept. Watch for duplicate catches
rising, and for honest coverage falling while catch counts rise (the
reward-hacking collapse signature).

### 5. Seeded episodes forget where they have been

**Evidence.** Open in `bugs_found.md`. `seed_explored_maps`
(`rewards.py:910`) pre-fills explored maps only for action-replay starts
(`gym_env.py:1235`), not for Go-Explore snapshot seeds. In the last quarter,
seeded episodes earn `new_map` 22.95 and `frontier` 236 per episode, against
honest 20.5 and 283. Most of that comes from re-walking Violet and Cherrygrove
from a mid-game seed.

**Diagnosis.** On its own this mostly inflates seeded reward. Combined with
point 1 it is decisive: a post-Falkner seed is paid to re-explore Violet and
not to go south.

**Fix.** Store the episode's explored maps and visited cells with each
capture. Pre-fill them on seed, as `replay_actions` already does for maps.
That covers coverage, the visited mask, `frontier_dir` and the exit sense.

**Verify.** `new_map`/`frontier` per seeded episode drops to about what the
seed point should earn. Seeded episodes from the 1213 capture reach Route 32
rows ≥ 4.

**Risk.** Seeded coverage reward falls, so seeded episodes look worse on the
headline. Judge on honest episodes only.

### 6. Smaller items

- **Dead-end attractors.** POKECENTER_2F, the Ruins chambers and Earl's
  Academy hold episodes until the stagnation watchdog ends them:
  - `Runs/N_goals_0/ep_240` spent its last 3,577 steps in POKECENTER_2F;
  - 3 of the 9 recorded episodes end in RUINS_OF_ALPH_INNER_CHAMBER.

  The stagnation cut-off already handles this; it costs the rest of the
  episode. Lower priority than points 1–3. Worth reading again once Route 32
  is open.
- **`menu` pays on scripted text.** The Route 32 push-back scene paid `menu`
  +0.25 (drive log). It is small and bounded by novelty, but it is a reward
  for meeting a blocker.
- **Stale open item.** The `bugs_found.md` entry "no menu, cursor or
  battle-identity features" is out of date. `menu_state` and
  `battle_identity` have been in the RAM vector since v3 (`gym_env.py`,
  `_MENU_KEYS`, `_BATTLE_ID_KEYS`; in `789cdba`). Tick it, or narrow it to
  what is still missing (per-slot party, item counts).

## Recommended order of work

Change one thing per step. Resume from the latest checkpoint unless the step
says otherwise. Let each change run long enough for at least one regular
probe, and compare honest buckets of 25 before and after. Steps 1–4 change
reward or the sense values only, not dimensions, so they can each be
warm-started and attributed.

| Step | Change | Warm start? | Judge by |
|---|---|---|---|
| 0 | **Done 2026-10-29.** Instrumentation only: per-episode whiteouts, non-whiteout heals, wall bumps and walk presses, lead / non-lead level gains, top-3 levels at episode end (`Rewards._track_episode_diagnostics`, `info.pth` `episode_*`); health-report section "Party health & navigation"; "Reached Union Cave" (3:37) probe goal | yes (no behaviour change) | baseline numbers for 1–4 |
| 1 | **Done 2026-10-29.** Whiteout paid on the warp that ends it (armed by party HP 0 under any script, confirmed by HP back on another map) + no resource refund on that warp (point 2) | yes, from update 3,050 | whiteouts/episode ↓, heals ↑, flag 31 and trainer wins hold |
| 1b | If heals stay flat after step 1: Go-Explore captures on low party HP/PP, in their own pool (curation ranks by location rarity, so they would be dropped from the spatial one), so the policy practises recovery from many places | yes | heals/episode ↑ |
| 2 | Wall-aware `frontier_dir` (walkable cells only; reachable within the window if BFS is cheap enough) + first-move-weighted `descent_direction` (point 3) | yes, expect a short dip | bump share ↓, 66→1213 steps ↓, flags 20/1213 ↑ |
| 3 | Seeded visit history + competence-weighted progress seeding (points 5 and 1) | yes | seeded then honest Route 32 rows ≥ 4 / Union Cave |
| 4 | Top-3 √level party potential replacing the total-level threshold reward (point 4); LLM-harness demos of switch-training (send a weak mon out first, then switch) into SIL, capped and retired on competence like the existing demos | yes | top-3 levels ↑, non-lead level gains ↑, farm maxima flat |
| 5 | Next from-scratch run: per-slot party features (level, HP frac, in-battle); anything else that changes `ram_dim` | no | probe ladder against v3 at equal updates |

Step 1 comes first, even though point 1 ranks higher. It is the smallest
change, it is already diagnosed in `bugs_found.md`, and it changes the
battle economy that every later step is measured against. Step 0 shipped
with it, so the new counters have no pre-fix baseline in this run. Compare
step 1 with the replayed recordings (16 whiteout warps across three
episodes) and with the `whiteout` reward source before update 3,050,
which paid about 1 in 16.

Voluntary switches are not counted yet. That needs a verified
current-battle-mon address. Non-lead level gains are the proxy for now.

## Open questions

- Is the free whiteout warp helping flag 31 (the walk back to Elm), as
  `bugs_found.md` suspects? Step 1's flag 31 rate answers it. If flag 31
  falls, look at the 39→31 leg (median 7,000 steps) before adding anything
  back.
- Should Falkner get reverse-curriculum practice starts? RC targets are
  capped (`reverse_curriculum_max_targets`), and the slots are held by flags
  39/31/65/66, so 1213 is not chosen until those pass the exit rate. Decide
  after step 3: competence-weighted Go-Explore seeding may cover it.
- How much of the remaining bumping is plain policy entropy (1.43 nats,
  above the band)? Re-measure after step 2 before touching the entropy coef
  again.
- Is the resource potential's weight enough once whiteouts cost? Honest
  `resource_potential` is about +15 per episode, against about 280 for
  `frontier`. Read nurse-heal counts from step 1 before changing
  `resource_potential_coef`.

## What was not verified

- Getting through Route 32 after the badge was not driven live. The harness
  found no walking route out of Violet Gym from the post-Falkner state. It is
  inferred from episode 234's `discovery_log` (Route 32 PC and Union Cave
  after flag 1213) and from the Route 32 archive rows.
- Nurse-heal frequency across all honest episodes is unknown, because no
  per-episode signal exists. The figure here is 1 heal in 3 replayed
  episodes. Step 0 adds the counter.
- The link between `frontier_dir` and bumps is a correlation over three
  replays, not an intervention. Step 2's replay check and bump counter test
  it.
- Whether the `Runs/N_goals_0/ep_*` episodes were honest or seeded: their
  folder numbers do not map cleanly onto `episode_data` indices.
- The run was live while this was written. `info.pth` moved from episode 330
  to ~343 during the analysis, so re-derive any number before acting on it.
