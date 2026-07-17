# Progress

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
