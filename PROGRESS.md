# Progress

> **Historical experiment log.** This file preserves the commands, output
> directories, defaults and test counts used when each experiment ran; many are
> no longer current and the historical `freeform_gamearea*` configs have been
> removed. Use `configs/stages/open_world.json` and the README for the canonical
> current run:
>
> ```bash
> python -u main.py --use_config configs/stages/open_world.json
> ```

## 2026-07-16 — Exploration rebuild (branch `redo-configs`)

Diagnosis: the freeform agent plateaued at ~6–9 maps, confined to bank 24, because
the reward geometry paid more to re-sweep the known region than to discover a new
one, the value horizon couldn't reach milestones, and a stack of anti-stuck
machinery pumped undirected entropy instead of fixing either. Full rebuild toward a
simpler, per-episode curiosity signal.

### Changes

- **Reward → per-episode NGU coverage** (`rewards.py`): flat `frontier_novelty_bonus`
  per first-visit-this-episode cell (no floor, no cross-episode decay by default);
  flat `new_map_reward` + larger `new_bank_reward` so new regions beat re-coverage.
  Deleted the novelty floor, cross-episode decay farming, directional-frontier
  forecast, global-visit-count feature, and the battle-flee reward. Optional NGU
  lifelong decay behind `frontier_lifelong_decay` (default **off**).
- **Egocentric visited mask** (`gym_env.py`): 5×5 per-episode "visited near me" grid
  in the observation (resets each episode, fully egocentric). `RAM_OBS_DIM` = 141.
- **Real BPTT** (`PPOTransformer`, `vec_ppo_storage`, `vec_ppo_agent`, PPO loss):
  was `sequence_length=1` (transformer memory never trained). Now rollout = 1 frame +
  carried memory; training batches non-overlapping length-`sequence_length` (16)
  segments with causal masking + per-position outputs, so gradient flows across the
  segment.
- **Discount horizon** (`PPO.py`): `ppo_gamma:"auto"` cap 0.995 → **0.999**.
- **Deleted anti-stuck machinery** (`vec_ppo_agent.py`): entropy servo, entropy
  plateau reset/boost, stuck-action temperature, periodic archive decay. Fixed
  entropy 0.01. Kept: battle watchdog, stagnation truncation, action mask,
  milestone refire-depletion, visit archive (refire + discovery + optional lifelong).
- **Config loading** (`main.py`, `PPO.py`): **auto-resume** — a run with no explicit
  `load_checkpoint` continues from the latest checkpoint in its own output folder, so
  training can be stopped/restarted and extended. New CLI args `--add_rollouts N` /
  `--add_episodes N` set how much to train **this session** (additive), instead of an
  absolute `num_rollouts` stop.

### Run it

```bash
python main.py --use_config configs/stages/freeform_gamearea_rebuild.json
# resume + train 2000 more rollouts:
python main.py --use_config configs/stages/freeform_gamearea_rebuild.json --add_rollouts 2000
```
Trains from scratch (new architecture is checkpoint-incompatible). Success = breaking
the 9-map / bank-24 ceiling on the probe ladder. If it still plateaus, try
`frontier_lifelong_decay: true`.

### Status

Tests green (238 passed). Validated by synthetic tests (causal mask, BPTT gradient,
segment storage) + end-to-end smoke run. Not yet trained at scale — that's next.

## 2026-07-16 (later) — post-first-run fixes

First scaled run of the rebuild **collapsed**: reward exploded to 500–1290/ep then
crashed to a stuck 4-cell pocket. Two follow-ups:

- **Reward-scale schism fixed** (`default_configs/reward_settings.json`): the rebuild
  had rescaled exploration bonuses to 1/5/20 but left milestone/event rewards at the
  old 150–500 scale, so a single story flag paid **500** and blew up the shared-trunk
  critic. Rescaled onto the NGU scale — `flag 500→10`, `map_goal 250→20`,
  `pokedex_owned 150→15`, `first_sight 10→3`, `level_up 10→2`, `whiteout −100→−20`,
  `reward_clip 1000→50`. Everything now lives in ~[−20, +20].
- **Episodic frontier-direction sense** (`rewards.py` `frontier_direction`, wired
  through `gym_env.py`): the rescaled run trained stably but plateaued — full bank-24
  coverage, then no new run-wide discovery for ~1500 episodes (no gradient toward the
  sparse unfound exit). New observation feature: over a bounded window
  (`frontier_sense_radius`, default 6) it collapses "which way is
  unexplored-**this-episode**" into a unit direction vector + a local-saturation
  scalar. Reads only the per-episode visited set and emits relative direction only —
  fully egocentric + per-episode, so it teaches "head toward the unexplored", not a
  route. `RAM_OBS_DIM` 141 → 144 (checkpoint-incompatible; retrain from scratch).

Tests green (243 passed; +5 `test_frontier_direction.py`).

## 2026-07-17 — map/bank novelty decay (break the New Bark tour-optimum)

The frontier-direction run stayed stable for 16k episodes but never left bank 24
(no run-wide discovery after episode 141). Frame analysis of recorded episodes
showed the agent cycling New Bark Town's building-door cluster (town map 4 ↔ lab/
houses 6/7/8/9) — because `new_map_reward` was flat and re-paid every episode, the
optimal policy was "tour the 7 known doors" (~35 reward/ep, zero risk), and the
single doorless exit to Route 29 had no gradient pulling the agent there.

- **Map/bank novelty decay** (`rewards.py` `_new_map_bonus`, new `map_lifelong_decay`
  flag; `visit_archive.bank_count`): the NGU novelty principle already used for
  cells, applied at map/bank granularity. `new_map`/`new_bank` payouts are
  multiplied by `1/sqrt(1 + run-wide entry count)`, so re-touring a map entered
  thousands of times pays ~0 while a genuinely-new map/bank pays full. Dissolves
  the tour-optimum; reward-shaping only, so the policy still sees purely
  egocentric/per-episode state (no route memorised). Enabled in the rebuild stage.
  Reward-only change — `RAM_OBS_DIM` unchanged (144), but retrain from scratch for a
  clean read (the existing checkpoint is overfit to touring).

Tests green (246 passed; +3 map-decay cases in `test_phase3_goals.py`).

## 2026-07-17 (run 2) — undo frontier lifelong decay (double-decay starvation)

First scaled run with **both** `frontier_lifelong_decay` and `map_lifelong_decay` ON
failed: over ep 369→2993, exploration *decreased* monotonically (unique_cells
last100 42→22, unique_maps 3.56→2.39, archive growth 7→0), policy entropy *rose*
(1.44→1.70 — gradient dead, policy drifting random), net reward stuck at −10.

Diagnosis: the agent never found the Route 29 exit, so lifelong decay zeroed out
*every reward it could reach* — familiar cells paid ~0.03 each, touring paid 0.43
total — leaving the step penalty (−5.8) as the dominant term. Optimal policy became
"stop moving, get truncated fast." The design error: decaying the **per-episode**
frontier signal. That signal is the reliable always-on exploration teacher ("cover
new ground *this episode*" is always valid because it resets); decaying it lifelong
defeats its purpose.

Fix: `frontier_lifelong_decay` → **OFF**, `map_lifelong_decay` stays **ON**. The
principled split — per-episode cell novelty is flat/undecayed (reliable +1.0/new-cell,
linear+uncapped so covering more always pays more → outward gradient); run-wide
map/bank discovery decays (re-touring a 3000×-seen map pays ~0, new region pays full
→ tour-breaker). Restarted fresh; episode 2 avg_r +28 (was −10). Config-only change.

**Run-2 result @ ~1h (ep 2777): reward geometry FIXED, but hard-exploration plateau
remains.** Healthy now — last100 reward +71 (was −10), unique_cells/ep 74 (was 22),
policy entropy converging 1.9→1.33, coverage rising. BUT the objective metric is flat:
`max_unique_maps` stuck at 7 since ep 104, archive plateaued at 261 since ~ep 1400.
Archive dump shows the agent DID escape bank 24 — reached **(5,9) once** (8 cells in
bank 5) and (24,13) once — so the exit is reachable, not blocked. The bottleneck is
*reinforcement of a rare long-horizon sequence*: 1 escape in 2777 episodes can't be
credit-assigned. Undecayed frontier makes re-sweeping New Bark pay +74/ep, so there's
no pull to detour to the one-time exit. **Next fix (needs user sign-off — heavier,
injects global knowledge): frontier/save-state seeding (Go-Explore) — restart some
episodes from reached-but-underexplored frontier states so the outward region gets
training time.** Left run-2 training (it's healthy, not failing); hourly regression
watch continues.

**Run-2 @ 85% (ep 18700) — late-training degradation confirmed.** Reward peaked
~102 at ~60-65% of training (ep ~7-8k) then declined monotonically to ~43; ep_len
1365→657, unique_cells/ep 99→57, policy entropy drifted back UP 0.91→1.3. KL (0.002)
& clip (0.017) low → not a PPO blow-up; it's slow drift/forgetting from training a
SATURATED signal (archive frozen at 288 since ~ep 7000) with nothing new to learn.
The run's `best/` checkpoint (selected by probe_sr) is frozen at ep 117 — neither
latest nor best is a keeper, but this run's value was the diagnosis. **Actionable for
next run: num_rollouts 8192 over-trains this config — useful signal exhausted by ~5k
rollouts; either cap there or (better) read the degradation as the signal that the
agent needs FRESH signal → frontier/Go-Explore seeding. Let run-2 finish for a clean
baseline.**

## 2026-07-17 — Go-Explore frontier seeding (break the choke point)

User signed off on the frontier route. Built Go-Explore-style save-state seeding
that reuses the existing state-transport primitives (no new bytes-over-pipe channel).

**Mechanism (`goexplore_enabled`):**
- **Capture** (`gym_env._maybe_capture_frontier`, called in `step`): the first time
  an episode enters a map the run has RARELY seen (`visit_archive.map_count <=
  goexplore_capture_map_count_max`, default 100), on a valid non-scripted frame, the
  worker writes a PyBoy save-state file (one per (bank,num) per episode) and reports
  it in `terminal_info["frontier_captures"]`.
- **Curate** (`VecPPOAgent._ingest_frontier_captures`): agent keeps one snapshot per
  map region, ranks the pool by **LIVE** archive count (not stale capture-time count,
  so start-region snapshots sink and are evicted as the run tours them), caps at
  `goexplore_pool_size` (48).
- **Seed** (`_goexplore_cycle`, via existing `set_env_state` path transport): the first
  `goexplore_probe_fraction` (0.25) of workers incl. env 0 are NEVER seeded — they
  train from-scratch AND back the honest `probe_sr`. The rest restart from a frontier
  snapshot with prob `goexplore_seed_fraction` (0.6), sampled ∝ 1/(1+live_count) toward
  the true frontier; else return to true start.
- **Re-pay suppression is FREE**: reset()'s no-op startup step establishes all reward
  baselines (`_prev_*`, `explored_maps`, `_novel_cells_this_episode`) as a discarded
  reward, so a seeded start pays only for territory BEYOND the seed. No fact-bundling
  (the old `export_seed_state`/`apply_seed_facts` machinery is unnecessary).
- Probes use a dedicated `_probe_env` from the true start → seeding can't pollute them.

General/no-golden-path: frontier states are SELF-generated by the agent's own
exploration (never external saves); "rarely-seen map" is an area-invariant, self-
advancing rarity signal (as a frontier fills in, the next-rarest map takes over).
Config: `configs/stages/freeform_gamearea_goexplore.json`. Tests: +11
(`tests/test_goexplore.py`); full suite green. Smoke run: clean startup, 6/24 probe
split, capture firing, no crashes.

**Run @ ~2h (ep 8148): choke-1 BROKEN, choke-2 found.** Map-mode Go-Explore shattered
the bank-24 ceiling — `(5,9)` entered 916× (baseline: 2× in 18.7k eps), `probe_sr`
33%→100%, max_unique_maps 7→9. But it then stalled: bank-5 cells frozen at 24,
archive growth 0. Cause is a real limit of MAP-granularity capture — every seed lands
at the `(5,9)` *entrance*, so the agent re-covers the same 24-cell pocket 916× but
never launches from its far edge to cross the pocket's exit (choke-2).

**Added cell-granularity capture** (`goexplore_capture_granularity: "cell"`): snapshot
newly-discovered rare CELLS (`visit_archive.count <= goexplore_capture_cell_count_max`,
default 3), deduped per cell per episode, capped at `goexplore_max_captures_per_episode`
(3) for disk. Pool keyed by cell, ranked by live CELL count, so seeding returns the
agent to the perimeter of explored territory (the frontier edge) — from which a few
steps can cross the next choke point. Map-mode kept intact (additive switch). Stage:
`configs/stages/freeform_gamearea_goexplore_cell.json` (pool 128). Tests +4 (15 total
in test_goexplore.py); full suite 261 green. Launch after the map-mode baseline
finishes: `python -u main.py --use_config configs/stages/freeform_gamearea_goexplore_cell.json`.

**Cell-mode result @ ~1h (ep 2735) — BREAKTHROUGH, ceiling shattered.** Cell-granularity
Go-Explore didn't just cross choke-2, it cascaded through multiple regions. Banks
reached: 24, 5, 20, **26** (baseline ever reached only 24; map-mode reached 24+5).
Distinct maps in archive: **19** (baseline 7, map-mode 9); archive size **591** (nearly
2× the full baseline, in 1/7 the episodes). Bank 26 is a whole new region — 9 maps
(1,3–10), 251 cells, `(26,3)` entered 390×. Bank 20 map 1 also opened (97 entries).
probe_sr 100%, avg_r ~77. The frontier-edge seeding let the ratchet push past each
choke point in turn. Only 21% through — left running.

**Cell-mode @ ~2h (ep 5710): new plateau at 4 banks / 19 maps / 591 cells — likely an
EVENT GATE.** After the region cascade, archive went flat at 591 (~ep 2300) and stayed
there ~3400 eps; still 4 banks (24,5,20,26), 19 maps, no new territory. Banks 20 & 5
are tiny 24-cell interior pockets (fully explored); bank 26 is a complete 9-map
overworld region. Read: the agent has explored ~everything SPATIALLY reachable, and
further Crystal progress needs a story EVENT (NPC errand / forced rival battle / HM) —
a specific A-press on a specific NPC that spatial-novelty reward doesn't incentivise.
That's a different wall than the spatial choke points Go-Explore solves. Left running
(gate-crossing is stochastic; if it fires, Go-Explore cascades again). **Next lever
for real story progress: event/interaction-triggering (reward or curiosity over game
flags/NPC state), NOT more spatial Go-Explore.**

**Cell-mode COMPLETE (8192 rollouts, ep 16799, ~4h52m) — decisive success.** Final:
19 maps / 4 banks (5,20,24,26) / 591 cells; story flags fired 26,27,28,29,**30,39**,1735
(baseline only 26–29,1735 → +2 new flags via stochastic event triggering); pokédex
seen 10 (was 7), level max 10 (was 9). probe_sr held 100%, no crashes, plateau at the
event gate held to the end (no further cascade). vs the 18.7k-episode baseline that
never left bank 24 (7 maps). Checkpoint: `Training Outputs/00_freeform_gamearea_goexplore_cell/`.

### Session outcome & next lever
Arc: reward-collapse fix → confirmed pure curiosity can't break the choke point →
Go-Explore map-mode (broke bank 24) → cell-mode (cascaded into the open world).
The remaining wall is an EVENT gate (needs NPC/flag interaction, not spatial reach),
which is a design pivot for the user, NOT a spatial-Go-Explore tweak. Candidate next
mechanisms (user to direct): (a) intrinsic curiosity over game-flag/NPC-state changes
so the agent is drawn to interactions, (b) a flag/event-count novelty term parallel to
cell novelty, (c) seed the Go-Explore pool by flag-state as well as cell so it returns
to states on the verge of an event. Autonomous loop stopped here — experiment
succeeded; next step needs a direction call.

## 2026-07-18 — event-flag novelty reward (attack the event gate) + PPO/model bug fixes

Full-codebase review (subagents + direct read). Implemented the highest-value lever for
the event-gate wall — candidate (b), generalised — plus several verified correctness fixes.

### Event-flag novelty (the interaction drive) — `event_novelty_*`
The only story signal was a fresh 0->1 flip of one of the ~48 curated flags in
`_DERIVED_FLAG_TABLE`; the dense INTERMEDIATE event flags Crystal sets for small
interactions (talked-to-NPC, received-item, script-step) paid nothing, so nothing pulled
the policy toward the interaction that crosses a gate. New `_event_novelty_bonus`
(`rewards.py`) pays `event_novelty_bonus` for the first 0->1 flip THIS episode of ANY bit
in the whole wEventFlags region (0xDA72-0xDB71), depleted by `1/sqrt(1 + run-wide fire
count)` (new `VisitArchive._event_flag_fire_counts`, merged via `get_milestone_state` ->
`merge_milestones` like cells/milestones). Same NGU rule as cells: a genuinely-new
interaction pays full; a bit that flips every episode (clock/sprite-visibility churn — the
noisy-TV failure) depletes to ~0 in a few episodes. Area-invariant, no golden path.
**OFF by default**; excludes documented non-monotonic bits (26 transient, 1726
sprite-visibility) via `event_novelty_exclude_flags`. Reported in the reward breakdown as
`event`. Tests: +7 (`test_event_novelty.py`).

### Bug fixes
- **Dropout was active during rollout** (`ppo_dropout`, default **0.0**): the model was
  never switched to eval() during collection, so rollout (stores old_log_probs) and update
  (recomputes new_log_probs) sampled different dropout masks — PPO's importance ratio was
  corrupted by dropout noise before any policy change. Now 0 by default.
- **Trunk geometry now config-driven** (`num_layers`/`n_heads`/`d_model` in
  `core_settings.json`): previously fell through to constructor defaults (2 layers), silently
  contradicting the "4-layer" docs. Defaults preserve the 2-layer behaviour; set the keys to
  change depth/width.
- `.squeeze()` -> `.squeeze(-1)` in the legacy single-env value/tail paths (mis-broadcast
  when B==1 or L==1).
- `inference.json` repointed to the current run + `vision:false` (was a stale pre-rebuild
  checkpoint with `vision:true` — guaranteed shape/OOD mismatch).

### Deferred (documented, higher risk — need a run to verify)
- Relative-position attention bias to replace the absolute age embedding (makes rollout vs
  BPTT forwards numerically consistent).
- Per-`done` masking inside a 16-step BPTT segment (stops attention leaking across an
  in-segment episode terminal).
- True terminal-obs bootstrap for truncated GAE (currently uses the post-reset state's value).

### Run it (FROM SCRATCH — no warm-start; user chose not to carry checkpoint compat)
```bash
python -u main.py --use_config configs/stages/freeform_gamearea_event.json
```
Trains from scratch (8192 rollouts) with Go-Explore cell-mode + event novelty on from step 0,
into `Training Outputs/01_freeform_event_novelty/`. Success read: reward breakdown `event`
term non-zero and NEW story flags firing beyond the cell-mode set (26-30,39,1735) — e.g. 31
(gave_mystery_egg), catch-tutorial 65/66, or a gym-leader flag — plus `max_unique_maps`
climbing past 19. Full suite green (268).

**Historical note, since resolved:** the run exposed a stale warm-start
`episode_data` skeleton. Checkpoint loading now builds a complete fresh skeleton
from the current tracking schema before carrying compatible history forward.

## 2026-07-19 — event reward alone is inert; add flag-state Go-Explore seeding

**Run 1 result (4-layer, event_novelty_bonus 2.0, ep 2819 / ~23%):** healthy but did NOT cross the
gate. Re-cascaded to the exact baseline extent (banks 5/20/24/26, 19 maps, 568 cells) and — the
key finding — fired the SAME flags as baseline (26-30,39,1735), zero new. Cause: `event` reward
averaged **0.38/ep vs frontier 79/ep** (~0.5% of signal) — the event bonus was drowned by
undecayed frontier coverage, and the ~6 old flags it re-fired every episode are depleted to ~0.04
each. Confirms (again, cf. `reward-mechanics-clip-and-scaler-2026-07-17`) that a small
milestone-type reward is inert when frontier dominates: **fix the ratio, and — more importantly —
give repeated attempts from the verge of an event, because reward magnitude can't manufacture the
first trigger.**

**Two changes for run 2:**
- **event_novelty_bonus 2 -> 10** (on par with `new_bank`), so a genuinely-new flag is a real
  advantage spike, not a rounding error. Old-flag re-fires still deplete to ~0 (no farming).
- **Flag-state Go-Explore seeding** (`goexplore_flag_capture`, the deferred candidate (c)): when a
  RARE event flag fires (`event_flag_fire_count <= goexplore_flag_count_max`, default 3), latch it
  and snapshot at the next clean in-control frame — a walkable "verge of the next event" state.
  Reward calc exposes per-step fires (`_detect_event_fires` / `rare_event_fires`, decoupled from
  the reward so capture works even at bonus 0); the capture feeds the SAME frontier pool keyed by
  `("flag", bit)`, ranked/seeded by live fire count (`_frontier_key`/`_live_count` dispatch on
  `cap["kind"]`). Flag and cell captures coexist in one pool — as spatial saturates (high cell
  counts), rare-flag snapshots rise to the top and get seeded most, so the agent gets repeated
  shots at the next event from just past the last one. The interaction analogue of the cell-mode
  ratchet that cracked the spatial choke points. Back to **2 layers** for faster iteration (the
  bottleneck is reward/seeding, not model capacity). Tests +9 (277 green).

Run it (from scratch, into `02_freeform_event_seed/`):
```bash
python -u main.py --use_config configs/stages/freeform_gamearea_event.json
```
Success read: NEW flags beyond 26-30/39/1735 in the discovery log (e.g. 31 gave_mystery_egg, 65/66
catch tutorial, a gym flag), `max_unique_maps` past 19, and the `[GoExplore] frontier locations`
log showing `("flag", ...)` snapshots entering the pool. Watch the `event` breakdown for a
noisy-TV blow-up (junk bits firing at bonus 10) — if it balloons with no real discoveries, add
those bits to `event_novelty_exclude_flags` or lower the bonus.

## 2026-07-24 — checkpoint observability and canonical open-world config

- Durable story flags now have human-readable checkpoint metadata and a
  separately reported `checkpoint` reward source.
- The first honest reach of each checkpoint stores its action prefix, replays it
  from the canonical state, verifies the expected flag and writes labelled PNGs
  under `Runs/checkpoints/`.
- Short and long true-start probes report every ladder checkpoint separately,
  including Mr. Pokémon's house, receiving the Mystery Egg and returning it to
  Elm.
- Go-Explore frontier manifests now survive auto-resume and can rebuild missing
  checkpoint snapshots.
- `tools/training_health_report.py` reports honest/seeded checkpoint reach,
  replay status and frontier-pool health.
- Historical freeform experiment configs were consolidated into
  `configs/stages/open_world.json`, the canonical from-scratch run.
- `.opencode/skills/training-run-analysis/` captures the repeatable evidence
  order for model health, reward and RAM-correlated checkpoint analysis.
