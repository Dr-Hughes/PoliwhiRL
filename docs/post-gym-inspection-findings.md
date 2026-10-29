# Post-gym inspection findings (open_world_v3)

Companion to [`post-gym-inspection-plan.md`](post-gym-inspection-plan.md), run
2026-10-29 against the live `open_world_v3` run (ep ~473, rollout ~3700).
Evidence files live under `Training Outputs/drive/post_gym/` (drive walks,
probes, PNGs) and `Training Outputs/drive/post_gym/step1/` (trajectory
forensics tables). No training code, config or the live run was changed during
this pass; framework bugs found are logged in
[`tools/llm_agent/bugs_found.md`](../tools/llm_agent/bugs_found.md).

## The stuck point

**Route 32 (10:1), cells (18,6)-(18,8): the Togepi-egg messenger gate.**
A Cooltrainer stands at (19,8) on the only path tile row where the road
narrows to one column (trees wall x=15-17 at rows 6-8; verified tile-by-tile
from the harness local map, §173, and by failed `left` presses at (18,6)/(18,7)
in `detour1.png`/`trap2.png`). Stepping onto (18,8) fires a scripted scene:
*"Wait up! What's the hurry?"* → he follows you → the script **pushes you back
two tiles to (18,6)** → five text boxes → free walk, re-armed. It repeats on
every entry, forever, until the party flag
`EVENT_GOT_TOGEPI_EGG_FROM_ELMS_AIDE` (45) is set. This is the game's designed
story gate, not a bug — see `maps/Route32.asm` (`Route32CooltrainerMStopsYouScene`)
in pret/pokecrystal.

Episode evidence (step-1 pass, `step1/episode_table.md`):

- `ep_394` (metrics idx 405): on Route 32 from step 12,183, tail cells
  (18,6)-(19,7), **ended at (14,2) @19,992 with stagnation clock 1.0 and ~29k
  budget unused**. It never fired 1213 — the plan's premise that this is a
  post-Falkner case is wrong; it is the same pre-egg chokepoint loop.
- `ep_304` (idx 321): the only recorded honest 1213 fire (step 27,370); then
  sat on the **gym floor for its last 2,000 steps** until stagnation cut it.
- Deepest honest south ever recorded on Route 32: **y=8** — the gate row.
  Zero episodes reached 10:13 (Route 32 PC) or 3:37 (Union Cave); zero
  Go-Explore snapshots exist past b10_m1, and the only 1213-tagged snapshots
  are 3 × `b10_m7_pre_f1213` (pre-fight, inside the gym).

The continuation probe starts from `emu_files/states/post_falkner.state` =
VIOLET_GYM (6,1), **one hurt Quilava (18/46), no egg** — representative of the
win moment (matches ep_304), and structurally on the wrong side of the gate.
Every probe goal south of the gate (Route 32 PC, Union Cave, Route 33, Azalea)
is unreachable until the agent first walks back to Violet City's PC and
completes the aide scene; the probe's 0% on the egg goal itself says the
policy never finds that key.

## Candidate verdicts

1. **World/gating — CONFIRMED, primary cause.** The messenger push-back
   scene hard-blocks the one-tile chokepoint until flag 45. Honest flag-45
   rate is **1.5% (3/328 episodes)** vs 7.5% for 1213 (health report, rollout
   3700; story-checkpoint table: aide 3 honest / Falkner 15 honest). The
   LLM-agent run passed (18,8) untouched because its history *had* completed
   the aide scene (`01_togepi_egg_done`); our hand-walk from the probe state
   re-triggered the scene on every entry (§148, §167). The plan's guess that
   the aide scene is at the *Route 32* PC is wrong — the aide is in the
   **Violet City** PC; the Route 32 PC is where Eusine waits.
2. **Reward drought south of Violet — RULED OUT.** One-call audit walk from
   the egg state (`walk_egg_log.jsonl`, 283 presses, Violet PC → Union Cave
   door, all milestones true): **delivered 397.9 ≈ 1.4/press** — new_map +40,
   event +40, first_trainer_battle +40, first_trainer_battle_win +50,
   checkpoint_honest +250 (clipped to ~150 by the balancer), pokedex +24,
   battle +31, frontier +44. Compare the gate loop: **47.9 over 856 presses
   ≈ 0.06/press** (`walk_south_log.jsonl`). The south pays; the policy just
   can't get there.
3. **Survival — CONFIRMED as compounding, not primary.** The egg-state walk
   won every Route 32 trainer fight with one full-HP Lv16 Quilava (win at
   (15,22) included). Losses come from arriving hurt: the probe state itself
   starts at 39% HP, ep_394 shows the Route 32↔Violet-PC whiteout signature
   (3 PC arrivals after battles), and honest whiteout penalty now averages
   −50/episode last-quarter (the 2026-10-29 whiteout fix is paying as
   designed). The other survival sink is **Sprout Tower 3F (3:3)**: ep_449
   spent 78% of its post-gym steps in battle there and ended via
   `battle_stagnation`; ep_340 ended the same map.
4. **Stagnation watchdog — CONFIRMED as proximate terminator, exonerated as
   cause.** On the productive southbound walk the clock never exceeded
   **0.004**; inside the push-back loop it climbed 0.19→0.26 over 200 presses
   (≈1.0 in ~2.5k). The clock correctly truncates the loop; the loop is the
   problem. Whole-run truncation: stagnation 408/475.
5. **Observation/harness — mostly RULED OUT, two real bugs logged.**
   Nothing in the observation lies about the gate — flag 45 is in the RAM
   vector (derived table `got_togepi_egg`), the collision field correctly
   calls (18,8) walkable (it *is* walkable; the scene pushes you back), and
   the push-back is visible on-screen. But: (a) the **egg-counts-as-fainted**
   bug made the LLM harness order agents into an unhealable heal-nag (fixed
   this session — see `bugs_found.md` Fixed 2026-10-29); (b) the recorder's
   `playerstate` column reads `walk` during battles (logged Open);
   (c) `ep_369`/`ep_381` CSVs match no metrics row (logged Open).
6. **Distraction — CONFIRMED as compounding.** 3 of 5 gym-reachers sank
   their post-gym tail into maps that still paid per-episode novelty:
   ep_139 49% of post-gym remainder in Ruins of Alph 3:22-28; ep_340/449 in
   Sprout Tower; ep_394 banked 855 Ruins + 938 gate steps. Rational for a
   novelty-maximiser — the gate is the only road out and it pays nothing
   while closed.

## Primary cause and compounding chain

**Primary:** the post-Falkner world is gated behind a story scene (Violet PC
aide → egg, flag 45) whose honest completion rate is 1.5%. The gate's
push-back loop is reward-blind and observation-quiet, so episodes burn
thousands of presses at (18,6)-(18,8) until the stagnation clock truncates
them. **Compounding:** (a) episodes leave the Falkner win hurt and alone
(39% HP in the probe state), so even the return trip to the PC is survival-
risky; (b) Ruins/Sprout Tower still pay novelty, pulling time off the gate;
(c) with zero snapshots south of the gate row, seeding can never bootstrap
past it — the plateau is self-locking.

## Recommended fixes (ranked, general mechanisms)

1. **Bridge the Falkner→aide rung in the reverse-curriculum / seeding pool.**
   The egg goal (gym exit → Violet crossing → PC → aide scene) is an
   80-step, 23-turn walk for the LLM agent
   (`post_gym_01_resumed_20261029_162703`, goal `togepi_egg`): record that
   demo into training (`tools/llm_agent/demos.py`) and/or add flag 45 as an
   explicit RC rung seeded from `1213`-snapshots. Expected next-run effect:
   the continuation probe scores >0 on the egg goal itself; Route 32 PC /
   Union Cave goals stop being structurally zero.
2. **Make scripted push-back cells visible to the run-frontier field.**
   Area-invariant mechanism: when a map scene re-triggers on the same cell
   without setting any recordable flag, mark that cell gated (blocked for
   route sense) until its unlock predicate changes. The frontier gradient
   then stops pointing the policy into the loop and the stagnation clock
   stops spinning there. Changes in next run: fewer 1.0-clock endings at
   gate rows; ep-394-shaped tails disappear.
3. **Post-checkpoint heal pressure.** The win moment is a hurt solo party in
   a dead-end map (ep_304's 2,000-step gym-floor sit; probe state 18/46).
   `resource_potential` already prices depletion; after the whiteout fix
   landed, watch the honest whiteout trend before adding anything — the
   −50/ep signal suggests the penalty is now doing this work.
4. **Watch Sprout Tower 3F.** It is the run's battle grinder (78% battle
   steps, `battle_stagnation` endings). The balancer's clip is absorbing the
   farm (−130 clip in the audit walk); if `battle_stagnation` episodes rise,
   it needs the same gate treatment as (2).
5. **Probe design note:** the continuation probe should include the aide
   scene as its first scored goal (it does) *and* a pre-egg vs post-egg arm
   — 0% on all goals is uninformative when one flag gates the whole ladder.

## Framework bugs found this pass

Logged in [`tools/llm_agent/bugs_found.md`](../tools/llm_agent/bugs_found.md):

- **Fixed (commit 12ff0ee):** held egg counted as fainted —
  harness heal-nag + `party_hp_frac` capped at 0.71 for every egg-holding
  episode (honest *and* seeded). Root cause, fix and probe in the entry.
- **Open:** recorder `playerstate` mis-decodes during battle; `ep_369` /
  `ep_381` have no metrics row.

## Materials corrections to the plan

- `Runs/checkpoints/1213-Beat-Falkner/` contains only `.mp4` +
  `actions.steps` — the `start.state`/`post_falkner.state` rows in the plan's
  materials table are stale (the probe state is `emu_files/states/`'s copy).
- `ep_394` never fired 1213 (verified from `flag_fire_steps`, no replay
  needed); it is a pre-gym-win gate loop, not a post-win stuck point.
- The Togepi aide is in the **Violet City** PC (10:10), not the Route 32 PC
  (10:13).
