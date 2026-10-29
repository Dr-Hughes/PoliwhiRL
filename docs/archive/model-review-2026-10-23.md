# Critical evaluation of the PoliwhiRL modelling setup

> **Archived.** A one-off review, written 2026-10-23 against commit `9268212`
> on branch `flag31-breakthrough-and-tuning`, snapshotting a run at 279
> episodes. It predates the settle-stepping environment rewrite, the
> `open_world_v3` run, and the LLM-harness audit — so its run numbers and
> code citations are historical, and its open work list is superseded by
> [`../roadmap.md`](../roadmap.md).
>
> Still worth reading for: the findings table (§3), what must not be broken
> (§3.1), the literature review (§5), and the "observable that proves the fix
> landed" framing of each work-list item (§6).

**Scope.** Read-only review of the modelling stack behind the currently running
`configs/stages/open_world.json` run (started 2026-10-23 12:21, `--erase yes`,
16 worker envs, writing `Training Outputs/open_world/`), plus a literature pass
over comparable systems. No code was executed, no tests run, nothing edited
except this file. All run numbers below are read from
`Training Outputs/open_world/Results/metrics/training_metrics.json` (snapshot at
2026-10-23 16:12: 279 episodes / 1,920 rollouts) and the PNGs beside it; all
code claims cite `path:line` in the working tree on
`flag31-breakthrough-and-tuning` (HEAD `9268212`).

**Method.** (1) config stack resolution; (2) full read of the environment
(observation, rewards, archive, action mask, vec wrapper), the network
(CNN / RAM encoder / Transformer-XL trunk), and the PPO loop (collection,
segment-BPTT update, minibatch/KL logic); (3) forensic pass over the live run's
metrics with the honest-vs-seeded split first, per AGENTS.md; (4) a cited
literature review of Pokémon/Minecraft/Crafter-scale sparse-reward RL, on- vs
off-policy sample economics, memory architectures, exploration methods and
reward-hacking theory.

**If you are the developer picking this up:** §3 is the findings table (every
row carries code and run evidence), §3.1 is what must not be broken, §4 is the
run forensics, and **§6 is the ordered work list — each item names the file and
lines to change, why, and the observable that proves the fix landed.** Do the
three unblock items in §6 together before judging anything else.

---

## 1. The system in one paragraph

A PPO actor-critic over a dual observation — an 80×72×3 screen frame through a
2-block CNN (`PoliwhiRL/models/PPO/PPOTransformer.py:22-42`) and a 171-dim
engineered RAM vector through a 3-layer MLP
(`PPOTransformer.py:128-150`) — fused into `d_model=128` and fed to a 2-layer,
4-head, `mem_len=128` Transformer-XL-style trunk
(`PPOTransformer.py:45-125`, geometry from
`configs/default_configs/core_settings.json:16-21`). Each agent action holds a
button for 15 of 90 emulated frames (`gym_env.py:532-533`), so one "step" ≈ 1.5 s
of game time. Reward is a ~15-channel hand-authored table (story flags, durable
checkpoints, pokedex, key items, levels, wild/trainer battle, catching,
per-episode cell/map/menu novelty, whole-region event-flag novelty) passed
through a cross-episode **domain reward balancer** that normalises four skill
domains to unit per-fire magnitude and applies priority weights
(`domain_reward_balancer.py:116-168`). Exploration is a Go-Explore-style
persistent visit archive: workers snapshot emulator savestates at rare
cells/flags/party milestones and 40 % of episodes restart from them
(`vec_ppo_agent.py:1164-1197`); 4/16 workers stay honest from the true start.
Episodes run to 24,576 actions or a stagnation watchdog (1,536 novelty-free
free-walk steps, zero-bootstrap terminal, `rewards.py:965-983`).

## 2. Headline verdict

The design is thoughtful and unusually well-instrumented. The run is **stalled at
the capability level, not dead**: it has learned real low-level skill and zero
game-advancing skill. Both halves are true and conflating them is how this run
would be mis-triaged.

**What it has learned.** Honest per-episode coverage climbed 90 → 221 novel
cells and 4.96 → 8.88 maps over 200 honest episodes; mean episode length grew
10.3k → 14.4k actions; the first story checkpoints (Talked To Mom, then the
starter variants) were reached *honestly* at episodes 1–30 with recorded video
replays under `Runs/checkpoints/`. Sustaining novel-cell discovery for ~14k
actions per episode is not random-walk behaviour — a uniform walker exhausts New
Bark Town and hits the 1,536-step stagnation watchdog far sooner — so there is
genuine state-conditional structure in the policy (leave the house, walk
coherently, clear dialogs).

**What it has not learned.** Anything past probe-ladder rung 4. The fixed probe
ladder is bit-identical at rollouts 400/800/1200/1600 and at the long probe
(rungs 1–4 = 100 %, rungs 5–13 = 0.0); honest `flag_fires` plateaued at ≈3.0
(max 4) by episode ~50; and the *marginal* action distribution is near-uniform —
policy entropy 1.84–1.92 nats against a masked maximum of ln 7 ≈ 1.946, and the
last-1000-action histogram is 107–169 per button against a uniform expectation of
143 ± 11. A policy that has decided which door to take does not look like this.
97.8 % of episodes end in the stagnation watchdog with the clock pegged at ~1.0.
The honest coverage creep above is real but slow, and part of it is the
reward-depletion machinery re-pricing territory rather than a sharpening policy.

**Hard problem, or configuration failure?** Both, and they are separable — the
discrimination decides whether the next move is "run longer" or "fix the
optimiser":

| Signal | Intrinsic-hardness signature | Config-failure signature | This run |
|---|---|---|---|
| Capability gain per 100 rollouts | slow but non-zero | zero | zero — ladder frozen across four probes |
| Policy sharpness on solved sub-skills | sharpens | pinned near uniform | pinned; entropy rose toward the masked max |
| Optimiser movement | KL tracks the LR schedule | KL ≪ schedule | 0.0047/rollout at lr 2.5e-5 |
| Same stack, earlier configuration | — | reached further | 2026-07 runs reached ~16 honest unique_maps (AGENTS.md); this run is at 8.9 |

Pokémon-from-pixels is genuinely frontier-hard — no published system gets past
~20 % of Gen 1 and PokéRL (arXiv:2502.19920) died at the Cut gate — so the
ceiling is real and some of the flatness is the problem. But a hard problem
still shows non-zero capability gain; a frozen ladder plus a near-uniform
marginal policy with a mechanistic explanation (§3 F1, F2) is a configuration
regression, and the last table row is the clincher: this codebase has reached
~2× the current coverage before.

Two optimiser-level defects (§3 F1, F2) are sufficient to explain the flatness
on their own; the reward architecture (§3 F3–F5) then determines *what* the
policy farms once it can move; the horizon and exploration design (§3 F6–F9)
determine whether it could ever cross the game's multi-thousand-action gates.

The literature is unambiguous that this configuration sits in a
documented-failure region: the nearest published analogue — Pleines et al.,
*Pokémon Red via Reinforcement Learning* (arXiv:2502.19920), PPO + PyBoy +
pixels + engineered state vector + hand-authored milestone/heal/nav rewards —
reports the same symptoms (reward-channel farming, action repetition,
discount-induced credit-assignment failure, plateau at a gate) at 32 workers ×
36 h for ~20 % of one game. Go-Explore's Nature paper (Ecoffet et al. 2021)
states that savestate-seeded exploration is useless without a robustification
phase this project does not have. Skalse et al. (NeurIPS 2022) prove that no
reweighting of a non-constant proxy reward makes it unhackable, which is exactly
what the domain balancer attempts.

---

## 3. Table of results — findings

Severity: **C** = blocks learning now; **H** = will cap or misdirect learning
once the optimiser moves; **M** = real defect, bounded impact; **L** = hygiene.

| ID | Sev | Finding | Evidence (code) | Evidence (live run) | Recommendation |
|---|---|---|---|---|---|
| F1 | C | Entropy bonus receives an **independent gradient-norm budget equal to the whole policy+value loss**. `_update_networks` backprops `policy_value_loss`, clips to `ppo_max_grad_norm` (0.2), stashes; then backprops `entropy_loss` and clips it to its *own* 0.2; then adds. Whenever the raw entropy-gradient norm exceeds 0.2 — the normal case for a ~2 M-param net with a near-uniform 7-way categorical — the clip rescales it to exactly 0.2, so the configured `ppo_entropy_coef=0.02` no longer sets its weight and the regulariser becomes a co-equal objective pushing toward uniform. | `ppo_model_implementation.py:442-490` | `policy_entropies` 1.92 → 1.84 (masked max ln 7 ≈ 1.946): entropy *rose toward uniform* over 1,920 rollouts while reward climbed; button histogram uniform | Collapse to one backward pass with the coefficient inside the loss, or scale the entropy gradient by its coefficient *before* its separate clip, or clip only the summed gradient |
| F2 | C | **The optimiser cannot move.** `ppo_learning_rate=2.5e-5` (12× below PPO standard) with cosine `T_max = num_rollouts = 163840` — an unreachable horizon, so the schedule never decays (logged `lrs` 2.5000e-5 → 2.4995e-5). Two compounding restart bugs make it worse: `vec_ppo_agent.py:134` overwrites any configured `ppo_scheduler_t_max` with `num_rollouts`, and `reset_lr_scheduler_on_load` defaults **True** (`ppo_model_implementation.py:601-612`), so every auto-resume restart rebuilds the cosine from peak LR over the full unreachable horizon — at this project's restart cadence the LR is effectively a constant 2.5e-5 forever. Rollouts hold only 128 steps × 16 envs = 2,048 transitions; 4 epochs × 4 minibatches = 16 steps/rollout. Observed `approx_kl` ≈ 0.005/rollout. | `configs/stages/open_world.json:26-38`; `ppo_model_implementation.py:227-245,601-612`; `vec_ppo_agent.py:82,134` | `lrs` flat; `policy_entropies` flat; `approx_kls` mean 0.0047; button histogram near-uniform at 279 episodes | Raise LR to 1e-4–3e-4; stop `vec_ppo_agent.py:134` clobbering a configured `ppo_scheduler_t_max` (or key the schedule to a reachable budget); set `reset_lr_scheduler_on_load: false` or persist schedule progress across restarts; raise `ppo_update_frequency` to ≥512 so each update sees ≥8k transitions |
| F3 | H | **The domain balancer structurally favours wild-battle grinding exactly when the run plateaus.** `progressed_domains()` marks `battle` active whenever *any* battle reward fired this episode, but `frontier`/`milestone`/`catching` only on run-wide *first-ever* discoveries. After 50 discovery-free episodes the activity multiplier floors those three at 0.2 while battle stays at 1.0 — and battle carries the highest priority weight (1.6). On a plateau the balancer therefore transfers reward weight onto the one channel that can never go stale: grinding. | `rewards.py:728-743`; `domain_reward_balancer.py:133-168`; `open_world.json:58` | honest `battle` raw mean 41.1 pinned at its 50.0 cap from honest ep ≈100 onward (50.00, 50.00, 48.81, 49.75, 48.51 by 25-ep bucket) while `event` depletes 35.9 → 5.4 | Give `battle` a discovery-grade progress proxy (first KO of a new species, first battle on a new map) or delete the activity multiplier; keep priority weights as the only durable lever |
| F4 | H | **In-domain coupling erases the milestone hierarchy.** The balancer divides each domain by the running mean of its *summed per-step* value. `event` (bonus 10.0) sits inside the `milestone` domain with `flag`/`checkpoint`/`key_item`/`level`, so a large event-novelty bonus inflates the domain mean and shrinks every story source's effective weight ~10×. The module docstring warns about exactly this coupling and then puts `r_event` in `milestone_raw`. | `domain_reward_balancer.py:150-168`; `rewards.py:863-871` | honest raw means: `event` 11.87 vs `flag` 3.25, `checkpoint_honest` 18.19 — story income is a rounding error next to exploration+battle | Move `event` into its own domain (the fix already applied to `catching`), or price event novelty ≤ the frontier bonus |
| F5 | H | **Two stacked non-stationary normalisations.** Domain balancer × `RewardScaler` (1/std of discounted returns) both rescale the stream the critic sees; the balancer's own amplification (`domain_scale_adjustment`) drifted from −116 to +150 per episode across the run — ~48 % of delivered reward in the last bucket is balancer gain, and it moves as the mix shifts. | `rewards.py:873-891`; `vec_ppo_agent.py:1848-1853`; `running_stats.py:100-108` | `domain_scale_adjustment` honest mean +62.6, last bucket +149.6 against delivered ≈310 | Keep one layer. If the balancer stays, feed the scaler the scaled stream and log the factor (currently never plotted, see F13) |
| F6 | H | **Credit-assignment horizon ≪ milestone distance.** `ppo_gamma="auto"` resolves to 0.999 (`PPO.py:47`) → 1,000-action horizon; GAE λ=0.95 → advantage horizon 1/(1−γλ) ≈ 20 actions. The first honest starter took 4,895 actions; ladder rung 5 (Mr Pokémon's house) needs thousands more and has never fired. The gradient cannot connect "walk west" to "got the egg". | `PPO.py:32-51`; `vec_ppo_agent.py:87`; `open_world.json:100-114` | probe ladder rungs 5–13 = 0.0 at rollouts 400/800/1200/1600 and long-probe 1000 — bit-identical | Temporal abstraction first (hold-direction-until-blocked macro-actions; the 90-frame/15-hold primitive at `gym_env.py:532-533` already half-exists), then λ→0.99 or a hierarchical director; longer memory is *not* the fix |
| F7 | H | **Go-Explore is used for GO/EXPLORE only; the robustification half of the Nature paper is absent.** Savestate-seeded episodes train open-loop-fragile segments; there is no backward-PPO / self-imitation phase converting archive trajectories into closed-loop skill, and no sub-goal-conditioned return. 26 % of episodes are seeded and their honest transfer is weak. | `vec_ppo_agent.py:1164-1197` (seed cycle); no imitation/self-imitation loss anywhere in `agents/` | seeded n=67: delivered 230.9 vs honest 215.6; seeded unique_maps 7.31 vs honest 6.72 — seeding buys ~9 % coverage for 26 % of compute | Add a robustification objective (behaviour-clone the recorded first-honest trajectories, which already exist as `actions.steps` under `Runs/checkpoints/`), or cut `goexplore_seed_fraction` and spend the compute honestly |
| F8 | M | **Probes are single-threaded, inline, and longer than they need to be — plausibly the dominant wall-clock cost.** Each probe = 5 episodes × 8,192 actions × 90 frames ≈ 3.7 M frames on one core; the long probe = 5 × 24,576 × 90 ≈ 11 M frames. Collection produces 184 k frames/rollout across 16 workers in ~4 s. Probe episode length (8,192) is also *shorter* than the mean training episode (13,623), so the ladder under-reports capability. | `vec_ppo_agent.py:724-731, 1210-1282`; `open_world.json:96-120` | mean rollout wall-clock ≈7.2 s vs ~4 s collection-only; `rollout_durations_s` p95 6.3 s, max 57.2 s | Run probes in a spare worker process (async), or shorten `probe_episode_length` to match the training mean, or accept and budget it explicitly |
| F9 | M | **The stuck-escape signal is attenuated exactly where it is needed.** The stuck-temperature reshapes sampling by `probs^(1/T)` with T up to 1+5·clock; the policy-gradient w.r.t. base logits is divided by T on those steps, so the steps the mechanism is meant to teach from carry ~1/6 gradient. Meanwhile 97.8 % of episodes die in the stagnation watchdog with clock ≈1.0, wasting the final 1,536 actions (≈11 % of the mean episode). | `ppo_model_implementation.py:77-118`; `rewards.py:965-983`; `open_world.json:64` | `truncation_causes`: 261 stagnation / 6 budget; `mean_final_stagnation_clock` 0.983 | Reward the escape itself (first new cell after clock > 0.5 pays a bonus) instead of flattening the sampler; tighten the watchdog (e.g. 512) to reclaim compute |
| F10 | M | **No pixel normalisation.** `get_screen_image` returns uint8 0–255 and collection does `.float()` with no scaling, so conv1 sees inputs ~100× the scale of every other layer; its gradient dominates the shared 0.2 global-norm budget, starving trunk and heads. | `gym_env.py:1217-1239`; `vec_ppo_agent.py:760`; `PPOTransformer.py:265-266` | consistent with F1/F2 flatness | Divide by 255 at the env boundary (or first line of `GameBoyCNN.forward`) |
| F11 | M | **The catching curriculum has no reachable reward path.** `duplicate_catch_reward=0` and `catch_reward_episode_cap=0` zero the catch channel; START/SELECT are permanently masked (`allow_menus_walking=false`), so the overworld bag/party menu is unreachable; in-battle BAG is D-pad reachable but unrewarded beyond `pokedex`. Result: zero catches in 208 honest episodes while `threshold_checkpoints` demand "Caught First Wild Pokemon" / "Caught 6 Pokemon" and the balancer gives catching priority 1.0. | `open_world.json:41-46,58`; `action_mask.py:128-130` | `catches`, `new_species_catches`, `duplicate_catches` = 0 in every honest episode; `catch` source 0.00 | Either price catching (small, capped, new-species-gated) and enable menus once the policy is non-uniform, or delete the catching domain, its priority and the catch checkpoints |
| F12 | M | **`best/` selector saturates.** Free-play best-checkpoint selection counts first-honest durable checkpoint recordings; the early-game set (flags 1735/27/28/29/39) was exhausted by episode ≈30, so `best/` has been frozen since and can no longer rank policies for the rest of the run. | `vec_ppo_agent.py:2214-2254` | `checkpoint_recordings` = {1735, 27, 28, 29, 39}, all recorded in the first ~30 episodes | Switch the free-play metric to run-wide first-ever discovery count (the archive already tracks it) instead of the fixed checkpoint list |
| F13 | M | **Dark / inconsistent reporting.** `is_probe`, `goals_at_start`, `scaler_ext`, `scaler_int` are logged as empty lists, so the goal-success and reward-scale panels are dark; `_plot_metrics` never passes the scaler factors. The on-disk `training_metrics.png` at inspection showed ~1,750 rollouts against the JSON's 1,920 and an LR panel with a *decaying* curve while the logged `lrs` are flat — plots and JSON disagree; the "Training Loss" panel (−2000→−1000) does not match the logged `losses` (≈0.0). | `vec_ppo_agent.py:2075-2140` (no `scaler_ext=`); metrics JSON arrays of len 0 | see left | Pass the scaler factors; populate `is_probe`/`goals_at_start`; treat the JSON as the record and regenerate plots on demand |
| F14 | M | **Trunk is the least stable published transformer variant for PPO.** Plain post-LN Transformer-XL blocks, no gating; GTrXL (Parisotto et al. 2020) and Gu et al. 2025 show gated variants are materially more stable under RL objectives. The per-position loop computes attention for all ≤`mem_len`+1 queries and discards all but the last (~128× wasted rows per position), and there is no key-padding mask, so fresh-episode all-zero memory slots absorb real softmax mass. `mem_len=128` actions ≈ 3 min of game time — right for battles/menus, irrelevant to F6's horizons. | `PPOTransformer.py:97-125` | n/a (structural) | If the trunk stays: GTrXL gating + key-padding mask + compute only the last query; but fix F1/F2 first — architecture is not the current bottleneck |
| F15 | M | **Sample economics.** 2,048 transitions/rollout on a synchronous lock-step vec env: the slowest of 16 workers sets the pace. 163,840 configured rollouts ≈ 13.6 days at the observed ~7.2 s/rollout, and the LR/entropy schedules are keyed to that unreachable number. | `vec_env.py:416-447`; `vec_ppo_agent.py:82,103,134` | 1,920 rollouts in 3.85 h | Bigger rollouts (F2), async or dynamic-batching collection (PufferLib, arXiv:2406.12905), and schedule horizons in *reachable* units |
| F16 | M | **Live re-sweep exploit path.** `flag_reopens_novelty=true` clears the paid-this-epoch cell set on any derived-flag flip, re-opening the whole visited region at `revisit_novelty_scale=0.85` *and* resetting the stagnation watchdog (a re-paid cell counts as progress). With up to 49 derived flags per episode and ~200 visited cells, one flag flip can re-pay ≈+42 while defeating the stuck detector. Not currently dominant (revisit reward 1,536 total vs new 7,611) but live. | `rewards.py:1143-1153, 1589-1593` | `frontier_revisit_reward` non-zero and growing | Cap re-opened re-pay per episode, or exclude revisit re-pay from resetting the stagnation clock |
| F17 | L | `step_penalty=0` removes all time pressure; the stagnation watchdog is the only budget rule. Defensible, but it means "wandering" is free until the watchdog fires, which is what makes F9's wasted tail large. | `open_world.json:48` | mean episode 13,623 of 24,576 | Leave, or pair a tiny step penalty with a shorter watchdog |
| F18 | L | `whiteout_penalty=-20` never fires (0 in every episode) — the agent has no downside to losing battles, so battle-grinding (F3) is costless. | `rewards.py:1021-1045` | `whiteout` source 0.00 everywhere | Either accept (grinding is capped) or make losing costly enough to matter once F3 is fixed |

### 3.1 What is done well (keep)

| Area | Why it is right |
|---|---|
| Honest/seeded provenance discipline | Split everywhere it matters: archive counts (`visit_archive.py:59-71`), checkpoint reward (`rewards.py:1084-1104`), metrics (`summary.honest_*`). This is the rule that caught the 2026-07-20 regression and it is applied consistently. |
| Anti-farm engineering depth | Per-episode threshold dedup (`rewards.py:1047-1082`), run-wide 1/√(1+n) depletion, independent wild/trainer caps, zero-bootstrap stagnation terminals with the *reason* written down (`rewards.py:965-983`), collection/replay KL tolerance guard (`_minibatch.py:81-108`), non-finite-gradient skip (`ppo_model_implementation.py:460-479`). |
| Observation design | Egocentric, per-episode, transferable features only: visited mask, frontier direction, ROM collision field, stagnation clock (`gym_env.py:187-245`). No global-knowledge leakage; append-only RAM-vector contract with a regression test on flag numbers. |
| Mask/rollout/update self-consistency | The action mask is a deterministic function of stored RAM, so PPO's importance ratio is exact across phases (`action_mask.py:48-52`, `ppo_model_implementation.py:60-75`). |
| Truncation-aware GAE | Budget cut-offs bootstrap from a *preserved* terminal observation; stuck cut-offs are zero-bootstrap terminals (`vec_env.py:167-229`, `vec_ppo_agent.py:1949-2022`). Textbook-correct handling of the asymmetry. |
| Observability | Probe ladder + long probe + first-honest checkpoint recordings with video + per-source reward breakdown + discovery log. `best/` chosen on discoveries, never on raw reward (`vec_ppo_agent.py:2214-2233`). |
| Documentation | `docs/` with `path:line` citations and the AGENTS.md triage rules; the "split honest vs seeded first" rule is exactly what this review used. |

---

## 4. The live run, in numbers

Counts below are from two reads of the same live JSON: the headline state at
16:12 (279 episodes / 1,920 rollouts) and the per-source / bucket tables from
the 267-episode read at ~15:40 (202 honest / 65 seeded). The 12-episode
difference does not move any figure quoted.

### 4.1 State at snapshot (279 episodes, 1,920 rollouts, 3.85 h)

| Quantity | Value | Read |
|---|---|---|
| Agent actions collected | 1,920 × 128 × 16 ≈ 3.93 M (≈354 M emulator frames) | budget spent |
| Wall-clock per rollout | ≈7.2 s mean (≈4 s collection; probes inline) | F8, F15 |
| Configured budget | 163,840 rollouts ≈ **13.6 days** at this rate | F2, F15 |
| Learning rate | 2.4995e-5, flat (cosine T_max unreachable) | F2 |
| Policy entropy | 1.92 → 1.84 nats; masked max ln 7 ≈ 1.946 | **near-uniform, F1** |
| Button histogram (last 1000) | 107–169 across 7 unmasked actions; start/select 0 (masked) | near-uniform *marginally*; state-conditional skill exists (§2) |
| approx KL / clip fraction | mean 0.0047 / 0.0136; max 0.0238 / 0.075 | policy barely moves, F2 |
| Episode end cause | 261 stagnation / 6 budget (97.8 %, 267-ep read) | F9 |
| Mean final stagnation clock | 0.983 | stuck at the watchdog |
| Probe ladder (rollouts 400/800/1200/1600, long 1000) | rungs 1–4 = 1.0; rungs 5–13 = 0.0, **identical every probe** | F6 |
| Honest unique maps / cells | 4.96 → 8.88 / 90 → 221 over 200 honest episodes | slow creep |
| Honest flag fires | plateaued at ≈3.0 from ep ≈50 (max 4) | story stalled |
| Run archive | 131 → 503 cells, step-like bursts then plateaus | frontier stalling |
| Catches / trainer wins / whiteouts | 0 / 0 / 0 in every honest episode | F11, F18 |

### 4.2 Honest reward mix (raw, pre-balancer; honest episodes only)

| Source | mean | max | read |
|---|---|---|---|
| `frontier` (cell novelty) | 44.8 | 109.2 | intended earner — growing 29.8 → 66.7 |
| `battle` (wild engagement+KOs) | 41.1 | **50.0 = cap** | **pinned at cap from ep ≈100 — F3** |
| `checkpoint_honest` | 18.2 | 80.0 | mostly at the 0.1 depletion floor |
| `event` (region flag novelty) | 11.9 | 70.0 | depleting correctly 35.9 → 5.4 |
| `pokedex` | 10.9 | 70.0 | seen-count + starter only; never a catch |
| `new_map` | 9.6 | 59.7 | working as designed |
| `menu` | 5.8 | 9.8 | bounded, fine |
| `level` | 4.3 | 33.3 | grinding income, depleted |
| `flag` | 3.3 | 20.0 | story signal is a rounding error — F4 |
| `catch`, `key_item`, `map_goal`, `maps_visited`, `whiteout`, `first_trainer_*` | 0.0 | 0.0 | six dark channels — F11, F18 |
| `domain_scale_adjustment` | +62.6 | +277.9 | balancer amplification, drifting — F5 |

### 4.3 Why it "gets stuck and retries tried and old moves"

Four mechanisms compound, in order of weight:

1. **The marginal policy is near-uniform** (F1+F2). At ~1/7 probability per
   button the policy has no directional intent, so "retrying an old move" is not
   a failure mode, it is the current behaviour policy — even though
   state-conditional low-level skill exists (§2).
2. **No cross-episode memory of failure by design.** The visited mask and
   frontier direction are per-episode and egocentric (`rewards.py:1765-1849`) —
   a deliberate transferability trade-off. The run-wide archive only prices
   territory via reward depletion; it never enters the observation. So "I tried
   this corridor last episode and failed" is unrepresentable.
3. **Go-Explore seeding returns the agent to cells it has already failed at**,
   with no record of the failed attempts and no robustification phase to convert
   archive trajectories into skill (F7).
4. **The only de-aliasing signal for "stuck" is the stagnation clock**, and the
   mechanism that fires on it (stuck temperature) divides the learning signal by
   up to 6 on precisely those steps (F9), while the flag-reopen path can reset
   the clock entirely (F16).

---

## 5. Architecture questions, answered against the literature

### 5.1 PPO vs DQN vs something else

| Option | Evidence | Verdict for this problem |
|---|---|---|
| Keep PPO, fix the optimiser | PPO's poor showings are often implementation/tuning artefacts (Engstrom et al. 2020, arXiv:2005.12729; PufferLib, arXiv:2406.12905). The nearest published Pokémon system is PPO and reaches ~20 % of the game (arXiv:2502.19920). | **Do this first.** The current run is not a PPO failure, it is a mis-configured optimiser (F1/F2). |
| DQN / Rainbow | Off-policy and far more sample-efficient per collected step, but exploration in sparse reward is value-learning's weakest point (DQN scores ~0 on Montezuma/Pitfall without RND/NGU), partial observability needs DRQN-style recurrence, and a hand-authored milestone table is what off-policy value overfitting farms fastest. | Worse fit here. Not recommended. |
| Model-based (DreamerV3 / EfficientZero / TD-MPC2) | DreamerV3 collected Minecraft diamonds from pixels, sparse reward, no curriculum, at ~30 M steps with one fixed config (arXiv:2301.04104); EfficientZero matches 200 M-frame DQN with 100 k frames (arXiv:2111.00210); on Crafter, DreamerV2 10.0 % vs PPO 4.6 % (arXiv:2109.06780). | The right *direction* for sample economics and long horizons, but a large rebuild. Revisit after F1–F6, or prototype Dreamer-style imagination on the existing observation contract. |
| Learned priors instead of tabula rasa | Every strong open-world result (Voyager, JARVIS-1, GROOT, MineDojo/MineCLIP) came from priors — LLM planners, video data, or learned rewards — not on-policy RL from scratch. | Cheapest high-leverage move available: a learned/auxiliary novelty head (RND/NGU) or an LLM-directed curriculum would attack F6 directly. |

### 5.2 Transformer trunk vs alternatives

- A small Transformer-XL trunk is *defensible*: under PPO on long-horizon memory
  tasks, Gu et al. 2025 (arXiv:2505.15040) find only Transformer-XL, GTrXL and
  Mamba-2 solve the hard memory task at all; LSTM/GRU/frame-stacking do not.
- But plain ungated Transformer-XL is the *less stable* variant under PPO; GTrXL
  (arXiv:1910.06764) is the published upgrade (gates + gated LayerNorm).
- `mem_len=128` actions ≈ 3 minutes of game time: adequate for battle/menu arcs,
  irrelevant to the thousands-of-action milestone distances. **The horizon
  problem is a temporal-abstraction problem, not a memory-length problem** — the
  same conclusion PokéRL reaches ("time dimension in the action space").
- Decision-Transformer/Gato-style trunks do not help: they perform no online
  exploration (arXiv:2106.01345, arXiv:2205.06175), which is this project's
  actual bottleneck.

### 5.3 Exploration and reward design

- Hand-authored milestone tables are *provably* hackable: over stochastic
  policies two reward functions are mutually unhackable only if one is constant
  (Skalse et al., NeurIPS 2022, arXiv:2209.13085). The domain balancer changes
  *which* channel gets farmed, not whether. Predicted and observed: channel
  rotation from `key_item` (2026-07-20) to `battle` (this run).
- Published mitigations that fit this codebase: potential-based shaping;
  NovelD-style first-visit + relative-growth gating (arXiv: NeurIPS 2021,
  d428d070…) which names the exact "farm the same state change" failure;
  RND/NGU learned novelty replacing hand-authored discovery channels
  (arXiv:1810.12894, arXiv:2002.06038); AUP-style conservative agency
  (arXiv:2006.06547).
- Go-Explore's Nature paper is explicit that savestate-restored trajectories are
  not robust to stochasticity and require a robustification (backward PPO +
  self-imitation) phase, and that removing intermediate sub-goal cells zeroes
  reward on hard games. This project implements GO/EXPLORE and skips both
  (F7).
- Action repetition: fixed frame-skip forces redundant per-frame actions;
  learned/dynamic action repetition (AAAI 2017; FiGAR, arXiv:1702.06054) and
  PokéRL's "time in action space" are the published fixes — and they are also
  the fix for F6's horizon.

---

## 6. Developer handoff — ordered work list

Read §3 for the evidence behind each item; §3.1 lists what must not be broken.
Each item states the edit, the reason, and the observable that proves it landed.
Effort: **S** < half a day including tests, **M** 1–2 days, **L** 3+ days.

| # | Item | Fix IDs | Effort | Risk |
|---|---|---|---|---|
| 1 | Entropy-grad budget (unblock) — **implemented** | F1 | S | low — local to one function, unit-testable |
| 1b | Entropy-schedule restart continuity — **implemented** | F1, §6.2 | S | low — 1 line + unit test |
| 2 | LR, schedule horizon, restart-reset (unblock) — **implemented** | F2 | S | low — config + 1-line guard |
| 3 | Pixel normalisation (unblock) — **implemented** | F10 | S | low — 1 line + test |
| 4 | Balancer: stop promoting grinding — **implemented (delete variant)** | F3 | S (delete-multiplier variant) / M (discovery-proxy variant) | medium — changes reward mix mid-run |
| 5 | Balancer: `event` out of `milestone` — **implemented** | F4 | S | medium — same |
| 6 | One normalisation layer | F5 | S | medium |
| 7 | Frontier capture budget 3 → 6 — **implemented** | §6.2 (Change B) | S | low — pure config |
| 8 | Macro-actions, then λ | F6 | M–L | high — changes action semantics; replay/recording tooling must follow |
| 9 | Escape reward + re-sweep cap | F9, F16 | S–M | medium |
| 10 | Go-Explore robustification | F7 | L (interim config cut: S) | medium — new training path |
| 11 | Catching decision | F11 | S | low |
| 12 | `best/` metric + dark reporting | F12, F13 | S | low |
| 13 | Async probes, rollout size | F8, F15 | S (config) / M–L (async) | medium |
| 14 | GTrXL trunk | F14 | M | medium — state-dict migration |

**Totals:** unblock set (1, 1b, 2, 3) ≈ half a day including tests. Reward set
(4–6) ≈ 1 day. Capability set (7–9) ≈ 2–3 days. Hygiene (11–13, config variants)
≈ half a day. Everything except 10 and the async half of 13 ≈ **one focused
week**; item 10 adds 2–4 days, item 14 adds 1–2.

Do items 1–3 together, in one run, before judging anything else — none of the
later findings can be validated while the optimiser cannot move.

### 6.0 Implementation status (2026-10-23, this review)

Implemented and tested now: **1, 1b, 2, 3, 4 (delete-multiplier variant), 5,
7.** Deferred by design: **6** (F5 — keeping both normalisation layers is a
judgement call; the scaler factor is still not logged, see item 12), **8, 9,
10, 11, 12, 13, 14.**

- **1** `ppo_model_implementation.py` — `_update_networks` now clips the
  UNIT-coefficient entropy gradient against its own budget and multiplies by
  `_get_entropy_coef(step)` afterwards; locked by
  `tests/test_entropy_grad_weight.py` (doubling the coefficient must double the
  entropy gradient norm; under the old code the ratio was 1.0).
- **1b** `vec_ppo_agent.py` — `set_entropy_offset(-self.total_rollouts)` on
  load; locked by an assertion in `tests/test_best_and_capture.py`.
- **2** `configs/stages/open_world.json` — `ppo_learning_rate` 2.5e-5 → 1e-4,
  `ppo_update_frequency` 128 → 256 (4,096 transitions/update; 512 was my first
  choice but quadruples the rollout buffers on MPS — raise it if wall-clock and
  memory allow), `ppo_lr_schedule: "constant"`, `reset_lr_scheduler_on_load:
  false`; `vec_ppo_agent.py:134` now uses `setdefault` so a configured
  `ppo_scheduler_t_max` survives.
- **3** `PPOTransformer.py` — `GameBoyCNN.forward` divides by 255. Forward is
  scale-invariant (GroupNorm, `models/CNN/GameBoy.py:7-16`), so **the existing
  checkpoint resumes validly**; only conv1's gradient scale changes.
- **4** `domain_reward_balancer.py` — progress-activity multiplier deleted
  (module docstring records why); `domain_priority_weights` remain the only
  durable lever. Old checkpoint balancer states load cleanly (staleness key
  ignored; locked by a test).
- **5** `domain_reward_balancer.py` + `rewards.py` — `event` is its own domain;
  locked by `test_event_domain_scales_independently_of_milestone`.
- **7** `configs/stages/open_world.json` — `goexplore_max_captures_per_episode`
  3 → 6.

Verification: `uv run pytest tests -q` → **430 passed, 0 failed**; merged config
confirmed through `main.load_default_config` / `load_user_config` /
`merge_configs` (lr 1e-4, update_frequency 256, constant schedule, servo off,
anneal off, captures 6, no stale balancer keys). Restart guidance: plain
auto-resume (no `--erase`) keeps the 503-cell visit archive and the frontier
pool; bound the session with `--add_rollouts` (≈4,000 ≈ 8 h at the old ~7.2
s/rollout; expect longer per rollout at 256 steps) and judge on: entropy
falling below ~1.7, `approx_kls` in 0.005–0.03 with `clip_fraction` < 0.2,
honest `battle` no longer pinned at its 50 cap, and — the run's own success
criterion — probe-ladder rung 5 firing.

### 6.1 The unblock set

**1. F1 — entropy bonus must not get its own gradient-norm budget.**
- Edit: `PoliwhiRL/models/PPO/ppo_model_implementation.py:442-490`
  (`_update_networks`). Collapse the two backward passes into one:
  `loss = policy_value_loss + entropy_loss`, single `backward()`, single
  `clip_grad_norm_`. If the two-budget split must stay for the critic-starvation
  reason in the comment, scale the entropy gradient by `_get_entropy_coef(step)`
  *before* its separate clip so the configured coefficient is what sets its
  weight.
- Why: as written, whenever the raw entropy-gradient norm exceeds
  `ppo_max_grad_norm` (0.2) — the normal case for a near-uniform 7-way
  categorical over a ~2 M-param net — the clip rescales it to exactly 0.2, so
  `ppo_entropy_coef=0.02` is bypassed and the regulariser acts as a co-equal
  objective pinning the policy at uniform.
- Acceptance: `policy_entropies` must *fall* below ~1.7 within the first few
  hundred rollouts of a fresh run while `approx_kls` rises into the 0.01–0.03
  band; the last-1000-action histogram must stop being flat. Add a unit test
  that the entropy term's effective gradient weight equals the configured
  coefficient (compare grads at coef 0.02 vs 0.04).

**1b. Entropy-schedule restart continuity** (independently found by a parallel
review; verified here against this tree).
- Edit: `PoliwhiRL/agents/PPO/vec_ppo_agent.py:2332`:
  `self.model.set_entropy_offset(0)` →
  `self.model.set_entropy_offset(-self.total_rollouts)`. `total_rollouts` is
  restored from the checkpoint at `vec_ppo_agent.py:2327`, but `_get_entropy_coef`
  is keyed to the per-process `rollout_idx` (passed at `vec_ppo_agent.py:1943`),
  which restarts at 0 on every auto-resume — so any enabled anneal rewinds to
  full strength at every restart, and this project restarts constantly.
- Why: without it, enabling the anneal (`ppo_entropy_anneal_enabled`, currently
  absent from `open_world.json`, i.e. off) is self-defeating.
- Acceptance: unit test asserting the effective anneal progress is continuous
  across a simulated save/load/restart.
- **Caution — this does not substitute for item 1.** The anneal changes the
  *coefficient*; F1 is that the coefficient is bypassed by the separate entropy
  grad clip. Annealing toward a small `ppo_entropy_coef_min` only escapes F1 at
  the *tail* of the schedule, once the raw entropy-grad norm drops below 0.2;
  through the early and mid anneal the clip still pins the effective weight (and
  pins it *relatively higher* than configured). Fix 1 first, then the anneal
  behaves as designed.

**2. F2 — learning rate and schedule horizon.**
- Edit: `configs/stages/open_world.json:26-38`: `ppo_learning_rate` → 1e-4 (try
  3e-4 in a short smoke run); `ppo_update_frequency` 128 → 512; and either set
  `ppo_lr_schedule: "constant"` or give the schedule a reachable horizon. Two
  code traps: `vec_ppo_agent.py:134` overwrites any configured
  `ppo_scheduler_t_max` with `num_rollouts` (163,840), and
  `reset_lr_scheduler_on_load` defaults True
  (`ppo_model_implementation.py:601-612`) so every restart rebuilds the cosine
  from peak LR — fix or configure around both.
- Why: 2.5e-5 is ~12× below PPO standard and each rollout holds only 2,048
  transitions, so 16 optimiser steps/rollout at that LR moved the policy 0.0047
  KL in 1,920 rollouts.
- Acceptance: `approx_kls` per rollout in 0.005–0.03 with `clip_fraction` < 0.2;
  probe-ladder rung 5 ("Entered Mr. Pokémon's House") fires at least once within
  a few thousand rollouts of a fresh run — that rung is the run's own success
  criterion, not a guarantee.

**3. F10 — normalise pixels.**
- Edit: first line of `GameBoyCNN.forward`
  (`PoliwhiRL/models/PPO/PPOTransformer.py:38-42`): `x = x / 255.0` (or scale at
  the env boundary in `gym_env.py:1217-1239`; pick one place and test it).
- Why: uint8 0–255 inputs make conv1's gradient dominate the shared 0.2
  global-norm budget, starving trunk and heads.
- Acceptance: no NaN/skip events; conv1 grad norm no longer dwarfs the trunk's
  (log once during a smoke run).

### 6.2 A parallel plan, reconciled

A separate review proposed a two-restart plan: (B)
`goexplore_max_captures_per_episode` 3 → 6, then (A) the entropy-offset fix plus
enabling the anneal (`ppo_entropy_anneal_steps: 32768`,
`ppo_entropy_coef_min: 0.005`). Reconciliation against this tree:

- **Change B is accepted and becomes item 7.** New mechanism this review had not
  identified: the per-episode capture budget is exhausted on Route 29 before an
  episode banks anything in Cherrygrove/Route 30, so the frontier pool is
  geographically concentrated and seeding restarts episodes where they already
  fail — a concrete explanation for this review's F7 observation that seeding
  buys ~9 % coverage for 26 % of episodes. Pure config, low risk, consistent
  with the project's "general mechanisms, not per-area tuning" norm.
- **Change A's code fix is accepted as item 1b** (verified: `vec_ppo_agent.py:2327`
  restores `total_rollouts`, `:2332` zeroes the offset, the schedule is keyed to
  per-process `rollout_idx`). Its *config* half (enabling the anneal) is
  downstream of item 1 for the reason in 1b's caution.
- **Their `entropy_servo_enabled` warning is real but not currently biting:**
  the key defaults to True (`vec_ppo_agent.py:2335`) and would freeze the
  coefficient via `set_entropy_coef`, but `curriculum_base.json:11` sets it false
  and `open_world.json:2` inherits that. Keep the warning: any stage config that
  does *not* inherit `curriculum_base` gets the servo silently on.
- **Their plan leaves F2, F3, F4, F6 and F10 untouched.** After both restarts the
  run would still be optimiser-starved (LR 2.5e-5, cosine rebuilt per restart)
  and still reward-misdirected (balancer promoting grinding). Their sequencing
  discipline — one change per restart, cheapest signal first — is sound; keep it,
  but make the *first* restart carry the unblock set (items 1, 1b, 2, 3), because
  no later signal is readable until the optimiser can move.
- **Status-line discrepancy to resolve before acting on their numbers:** their
  update cites 224 episodes / 1,550 rollouts and a "ppo_epochs/grad-clip fix"
  already shifting the run. This working tree is clean (`git diff` empty) with
  `ppo_epochs: 4`, `ppo_max_grad_norm: 0.2`, `ppo_learning_rate: 2.5e-5` on disk,
  and the live metrics JSON read 267 episodes at ~15:40 and 279 at 16:12 on
  2026-10-23 — monotonically increasing, no reset. Their episode counts predate
  those reads and no such fix exists in this checkout, so treat that status line
  as stale or from a different checkout/run, and re-verify against
  `Training Outputs/open_world/Results/metrics/training_metrics.json` before
  sequencing restarts around it.
- Their two watch-items agree with this review exactly:
  `domain_scale_adjustment` drift is F5, and the entropy term being computed on
  the base rather than the behaviour policy is the F9 caveat.

### 6.3 The reward-architecture set (once the optimiser moves)

**4. F3 — stop the balancer promoting grinding on a plateau.**
- Edit, cheap variant: delete `activity_multiplier`
  (`domain_reward_balancer.py:143-148`) and keep only `domain_priority_weights`.
  Edit, thorough variant: give `battle` a discovery-grade progress proxy in
  `rewards.py:728-743` (`progressed_domains`) — first KO of a species not yet
  KO'd this run, or first battle on a map not yet battled on — which needs a
  run-wide ledger in `visit_archive.py` (merge/broadcast/persist like the
  milestone ledger).
- Why: `battle` can never go stale under the current proxy, so on any plateau
  the balancer transfers weight onto grinding — exactly the farm rotation this
  project has been fighting since 2026-07.
- Acceptance: honest `battle` per-episode mean no longer pinned at its 50.0 cap
  for >50 consecutive episodes while `frontier`/`event` are dark.

**5. F4 — move `event` out of the `milestone` domain.**
- Edit: `domain_reward_balancer.py:74` (`DOMAINS`) plus the grouping in
  `rewards.py:863-871`: add an `event` domain (the same split already done for
  `catching`), or reduce `event_novelty_bonus` to ≤ `frontier_novelty_bonus`.
- Why: one shared running mean per domain means a 10.0 event bonus shrinks every
  story source ~10×; story income is currently a rounding error
  (`flag` 3.3 vs `event` 11.9 honest means).
- Acceptance: honest `flag` + `checkpoint_honest` combined mean ≥ 0.2 × honest
  `frontier` mean in the reward-source plot.

**6. F5 — one normalisation layer, not two.**
- Edit: keep the domain balancer *or* the `RewardScaler`
  (`vec_ppo_agent.py:1848-1853`); if both stay, feed the scaler the
  already-scaled stream and log `reward_scaler.scale_factor()` per rollout.
- Acceptance: `domain_scale_adjustment` no longer drifts by more than ±50 across
  a run; the scale factor is visible in the metrics JSON/plots (item 12).

### 6.4 The capability set (the actual game-playing problem)

**7. Change B — frontier capture budget.**
- Edit: `configs/stages/open_world.json:82`:
  `goexplore_max_captures_per_episode` 3 → 6 (§6.2).
- Acceptance: frontier pool live-counts stop being Route-29-dominated; seeded
  episodes start appearing in Cherrygrove/Route 30 cells.

**8. F6 — temporal abstraction before anything else horizon-related.**
- Edit: replace the fixed 90-frame/15-hold primitive
  (`gym_env.py:532-533`) with hold-until-blocked-or-changed macro-actions (a
  direction persists until a collision, a state-byte change, or a cap of ~N
  holds); then raise `ppo_gae_lambda` toward 0.99. Budget for the knock-on work:
  `actions.steps` replays, the PNG recorder and the checkpoint-recording tooling
  all encode the old action semantics and must follow.
- Why: γ=0.999 + λ=0.95 gives a ~20-action advantage horizon against milestones
  thousands of actions apart; longer `mem_len` does not fix this.
- Acceptance: mean actions-to-first-starter (honest) falls below ~2,000; ladder
  rung 5 fires in probes.

**9. F9 + F16 — make being stuck teachable, and close the re-sweep exploit.**
- Edit: pay a bonus for the first novel cell after
  `stagnation_fraction() > 0.5` instead of (or in addition to) the sampler
  temperature (`ppo_model_implementation.py:77-118`); cap per-episode re-opened
  revisit re-pay and stop it resetting the stagnation clock
  (`rewards.py:1143-1153, 1589-1593`).
- Acceptance: mean final stagnation clock falls below ~0.8;
  `frontier_revisit_reward` per episode stays bounded.

**10. F7 — add the Go-Explore robustification half.**
- Edit: new offline/auxiliary objective behaviour-cloning the recorded
  first-honest trajectories already on disk
  (`Training Outputs/<run>/Runs/checkpoints/*/actions.steps`) — a dataset builder
  can reuse the existing replay recorder to pair actions with observations — or
  backward-PPO from archive cells. Interim, config-only: cut
  `goexplore_seed_fraction` (0.4) and spend the compute honestly.
- Why: Ecoffet et al. 2021 is explicit that savestate-restored trajectories are
  not robust without robustification; seeding currently buys ~9 % coverage for
  26 % of episodes.
- Acceptance: honest (unseeded) unique_maps grows faster than seeded, i.e. the
  honest/seeded gap widens in the honest direction.

### 6.5 Hygiene set

**11. F11** decide catching: price it (small, capped, new-species-gated) and set
`allow_menus_walking=true` once the policy is non-uniform, or delete the
catching domain, its priority weight and the two catch
`threshold_checkpoints`. **12. F12 + F13** switch the free-play `best/` metric
from the fixed checkpoint list to run-wide first-ever discovery count
(`vec_ppo_agent.py:2214-2254`); pass `scaler_ext`/`scaler_int` into
`plot_metrics` (`vec_ppo_agent.py:2075-2140`), populate `is_probe` /
`goals_at_start`, and treat the metrics JSON — not the PNGs — as the record.
**13. F8 + F15** run probes in a spare worker process (M–L) or shorten
`probe_episode_length` to the training mean (S); raise rollout size with item 2;
express schedule horizons in reachable units. **14. F14** GTrXL gating +
key-padding mask + last-query-only attention in
`PPOTransformer.py:97-125`, after items 1–3 have landed and can be measured;
budget a state-dict migration for existing checkpoints.

### 6.6 Explicitly not recommended

- Switching to DQN/Rainbow (§5.1): sparse-reward exploration is value learning's
  weakest point and the milestone table is what off-policy overfitting farms
  fastest.
- Lengthening `mem_len` as a horizon fix (§5.2): the horizon problem is temporal
  abstraction, not memory length.
- Model-based (Dreamer-class) *now*: right direction, wrong moment — it cannot be
  evaluated while items 1–3 are unfixed. Revisit after the unblock set.

---

## 7. Caveats

- Run numbers are a single snapshot of a live run; buckets of 25–50 honest
  episodes are small (n≈25) and the creep in §4.1 could stall or reverse.
- F1's mechanism (entropy gradient exceeding its 0.2 clip) is structural
  reasoning from `ppo_model_implementation.py:442-490`; I did not instrument
  gradient norms. The corroborating observation is entropy pinned/rising toward
  the masked maximum while reward climbs.
- F8's wall-clock attribution to probes is arithmetic (frames per probe vs
  frames per rollout), not a measured profile; the current process's stdout is
  not captured in any log on disk, so per-probe durations are not directly
  observable. `training_run.log` and `training_status_checkins.log` on disk
  belong to the *previous* process and must not be read as this run's state.
- The literature brief's Pokémon coverage is Red/FireRed; no published system
  plays Crystal end-to-end, and none is published past a Cut-tree/HM gate from
  pixels (PokéRL explicitly failed there).
