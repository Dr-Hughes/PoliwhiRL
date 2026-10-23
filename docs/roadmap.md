# Roadmap

> Consolidates the forward-looking content of `PLANNING.md`, `next_steps.md` and
> the still-valid forward items of `PROGRESS.md`. Those files remain in place as
> written; this is the single place to read what is *next*.

This page is the project's forward plan: where training stands on this branch,
what is still open, what has already been settled or removed, and which file each
open item came from. It is prose about a moving target — where it disagrees with
the code or the configs, the code is right (see [`../AGENTS.md`](../AGENTS.md)).
Anything that cannot be dated or confirmed against the current tree is marked
**needs confirmation**.

## Where the project is now

Training runs the milestone-driven reward rebuild — per-episode NGU cell coverage
plus run-wide map/bank novelty decay, checkpoint-labelled story milestones, and an
event-flag novelty channel that pays for first-seen interactions — on top of
Go-Explore frontier seeding at cell, progress and rare-flag granularity
(`configs/stages/open_world.json:56-75`). The agent is trained by vectorised PPO
with real BPTT through a Transformer-XL trunk over multiprocess rollouts, with
auto-resume so a run can be stopped and extended without losing progress. The
canonical open-world stage now runs `episode_length: 24576`
(`configs/stages/open_world.json:14`), a deliberate increase over the 16384-step
budget that the flag-31 critical path outgrew (`PLANNING.md:50-53`).

## Immediate next step (session ending 2026-10-23)

**The live `open_world` training process has been deliberately stopped** at
the end of this session (user requested a session restart). State at
shutdown: episode 5692, `total_rollouts` 67600, archive 1212 cells,
frontier pool 110 snapshots, branch `flag31-breakthrough-and-tuning` at
commit `eb6e2ba`, pushed. Nothing is running — no training process, no
background monitoring loop, no inference process.

**To resume:** `uv run python main.py --use_config configs/stages/open_world.json
> training_run.log 2>&1 &` — it auto-resumes from `Training
Outputs/open_world/Checkpoints`, carrying forward the visit archive,
Go-Explore frontier pool, and `DomainRewardBalancer` state. Confirm a clean
resume by checking the log for `"Visit archive carried over"` and `"Domain
reward balancer state carried over from checkpoint"` with no traceback,
same as every restart this session.

**What to do first after resuming:** the most recent reward change
(`domain_priority_weights.frontier: 1.3`, commit `f48b914`) was deployed at
episode 5656 and had accumulated only ~40 honest episodes before this
session ended — far too few to judge. The single highest-priority thing
next session is **evaluating this change** once a few hundred honest
episodes have accumulated: via `uv run python tools/training_health_report.py
"Training Outputs/open_world/Checkpoints/info.pth"`, check (a) whether
honest `archive_size` growth / `unique_maps` / `unique_cells` finally move
past their ep-474 plateau (the change's whole purpose), and (b) that honest
`trainer-battle-win rate` (was ~76-77%, plateaued) and catch-count haven't
regressed (the risk a too-aggressive frontier weight could overcorrect and
starve the other two boosted domains). If the win-rate drops meaningfully
with no frontier payoff, revert to `domain_priority_weights: {"catching":
2.0, "battle": 1.6}` (drop the `"frontier"` key) — a one-line,
one-restart rollback.

The user's stated near-term goal (set 2026-10-23, supersedes the
battle/catching framing below): **maximise reaching Violet City, clearing
Sprout Tower, and beating Falkner (Gym 1)**. `probe_goals` in
`configs/stages/open_world.json` now has a `"Cleared Sprout Tower (Got
Flash)"` rung (flag 20) between "Reached Violet City" and "Beat Falkner
(Gym 1)" so this is directly visible in every probe read. As of shutdown,
all three remain at 0% on every probe — this is the thing to watch for
first movement on.

### Reward design

- **Honest second catch** — `honest_last100_mean_pokedex_owned` has been pinned at
  ~1.0; episodes reach "Learned to Catch" (flag 66) but never complete a real
  second Pokémon. Run a dedicated trace (budget exhaustion vs. failure to execute
  the catch action) *before* any reward change. `PLANNING.md:78-85`
- **Early-game leg inconsistency** — flag 30 (Mystery Egg) fires anywhere between
  roughly step 250 and step 13,500+ of an episode, so the first legs of the episode
  are their own bottleneck, separate from the egg→Elm backtrack. Needs its own
  trace-based diagnosis, not another length bump. `PLANNING.md:70-77`
- **Event-novelty signal ratio** — keep watching the `event` reward breakdown for a
  noisy-TV blow-up at `event_novelty_bonus: 10.0`; if junk bits balloon with no real
  discoveries, extend `event_novelty_exclude_flags` or lower the bonus
  (`PROGRESS.md:332-333`; current values at `configs/stages/open_world.json:70-72`).
  Whether this is still live in the current run — **needs confirmation**.
- **Non-monotonic-counter farms** — `key_item` / `pokedex` / `level` are deduped
  per episode; a resurgence in any of them means a new code path bypassed the guard
  (`AGENTS.md:105-111`). Standing vigilance, not a one-off task.

### Battle/catching push toward Gym 1 (2026-10-22)

**Goal, in priority order:** beat Falkner / clear the Sprout Tower rival fight
(primary — the first content past Violet City), winning trainer battles and
catching Pokémon (secondary, supporting skills on the way there), without
disrupting the egg-delivery arc (flags 30/31) that is currently healthy. This
was scoped deliberately as config-only, using mechanisms the codebase already
has, rather than new reward code — see `AGENTS.md`'s "prefer general
mechanisms" principle.

Diagnosis (as of episode ~4300, `archive_size` 891, honest `h_r` plateaued
~220-250 — see `tools/training_health_report.py` for a live read):

- **The "beat Falkner" reward already exists and needs no new code.** Flag
  1213 (`beat_falkner`) is in `PoliwhiRL/checkpoints.py:42`'s
  `DERIVED_FLAG_TABLE` and is `is_recordable_checkpoint` — exactly the same
  durable-checkpoint machinery `EVENT_31` ("Gave Mystery Egg To Elm") already
  uses (`checkpoint_progress_reward: 40` default, honest/seeded-aware decay,
  `PoliwhiRL/environment/rewards.py:985-1030`). Reaching it is a *reachability*
  gap, not a missing reward signal.
- **`battle_reward_episode_cap: 3.0` was an undocumented regression.** It was
  cut from 15.0 in commit `6728715` ("bump", no rationale) on 2026-10-15,
  before the trainer-battle-win premium (`dfd5925`, 2026-10-17) existed. At
  3.0, a single wild-battle engagement (`battle_engagement_reward: 3.0`)
  already exhausts the whole episode's battle-reward budget, so a trainer win
  worth up to 20.0 raw (`battle_win_reward: 8.0 * trainer_battle_premium_multiplier: 2.5`)
  gets clipped to whatever headroom is left — frequently near zero. Verified
  directly: under the old cap, a full trainer engage+win sequence paid only
  3.0/27.5 of its intended value; under the new cap (below) it pays the full
  27.5 unclipped (checked against the real merged `open_world.json` config via
  `PoliwhiRL/environment/rewards.py`'s `Rewards` class directly, not just
  config arithmetic).
- **Winning was already structurally favoured over losing** (engagement 3.0 →
  win 8.0, ×2.5 trainer premium, vs. a flat `whiteout_penalty: -20` on a full
  loss) — the cap bug just prevented that favourable asymmetry from actually
  reaching the policy. Deliberately **not** touching `whiteout_penalty` or the
  engagement/win/premium constants: project history already shows that
  rewarding the *downside* path (fleeing) teaches permanent avoidance
  (`PoliwhiRL/environment/rewards.py:364-367`, "Battle-flee reward REMOVED");
  the fix here only restores the existing upside's ability to register.
- **Go-Explore already splits "spatial" vs "progress" frontier pools** — a
  "progress" snapshot is one captured right after a durable checkpoint (or
  party-size growth from catching), i.e. already past the furthest story beat
  reached (`PoliwhiRL/agents/PPO/vec_ppo_agent.py:986-993`). Seeded episodes
  draw from it with probability `goexplore_progress_seed_fraction`. This is
  the existing, already-general lever for "run more seeded starts from later
  checkpoints" — no new capture logic needed.

Changes made (`configs/stages/open_world.json`, 2026-10-22, all config-only):

| Key | Old | New | Why |
|---|---|---|---|
| `battle_reward_episode_cap` | 3.0 | 40.0 | Undoes the accidental 2026-10-15 cut; now comfortably covers one full trainer engage+win cycle (27.5) with headroom, matching the class default's original intent (30.0) plus a margin for consecutive trainers. |
| `domain_priority_weights.battle` | unset (1.0) | 1.6 | The durable "matters more, doesn't get normalised away" lever (`domain_reward_balancer.py:48-61`) — makes winning trainer battles the more strongly reinforced secondary skill. |
| `domain_priority_weights.catching` | 1.75 | 2.0 | Modest further bump, as requested — catching was already elevated. |
| `level_up_reward` | 2 (default) | 8 | Raises leveling's *relative* share within the `milestone` domain (alongside flag/event/checkpoint/key_item) without inflating the domain's overall magnitude — mirrors the "widen relative spread, not raw magnitude" principle `dfd5925` already established for the trainer premium. |
| `goexplore_progress_seed_fraction` | 0.25 | 0.4 | More seeded episodes start from the furthest-reached checkpoint snapshot (currently around flag 65/66, "Learned To Catch") rather than re-covering earlier spatial frontier — shifts training budget toward pushing past Violet City. |
| `goexplore_progress_pool_size` | 32 | 48 | More room to hold progress-type snapshots as new, deeper ones appear (post-catch, post-Violet-City, eventually post-gym). |

Deliberately **unchanged**: `flag_progress_reward`, `checkpoint_progress_reward`
(+ seeded/floor), `pokedex_owned_reward`/`pokedex_first_sight_reward` raw
constants, `whiteout_penalty`, `battle_engagement_reward`/`battle_win_reward`/
`trainer_battle_premium_multiplier`, and `goexplore_seed_fraction` (the overall
seeded-vs-honest-start ratio) — all already well-tuned and load-bearing for the
currently-healthy egg-delivery arc; touching them was out of scope for "without
disrupting the already progressing things."

Tested before deploy: two new regression tests in `tests/test_extends_config.py`
assert (a) the fully-merged `open_world.json` config's `battle_reward_episode_cap`
covers one full trainer engage+win cycle (guards against a repeat of the
2026-10-15 regression), and (b) `battle`/`catching` domain priority weights sit
above the 1.0 baseline. Full suite (399 tests) passes. A short local smoke run
(40 rollouts, 2 envs, fresh network, throwaway output dir) completed without
errors or reward blow-up; a direct `Rewards`-class check against the real merged
config confirmed the trainer engage+win sequence now pays its full 27.5 raw
value unclipped. Deployed via a restart of the live `open_world` process (same
restart path as prior sessions — auto-resumes from the existing checkpoint,
visit archive, Go-Explore frontier pool and `DomainRewardBalancer` state, so no
progress is lost; only the reward weighting/cap and seed-distribution config
change going forward).

**How to judge this**, once a few hundred post-restart honest episodes have
accumulated (same honest-vs-seeded discipline as `AGENTS.md`): watch the
`battle` and `catching` entries in `reward_sources` for a healthy rise without
one source becoming an order-of-magnitude outlier (a farm signature); watch for
the first-ever trainer-battle-win discovery and, eventually, any honest
movement past "Learned To Catch" toward "Reached Violet City" in the probe
ladder (`configs/stages/open_world.json:83-96`); and confirm honest flag-31
recurrence does **not** regress — if it does, this is a one-config-revert,
one-restart rollback.

**Result (deployed episode 4390, read at episode ~4700):** honest flag-31
recurrence jumped from 7.63% pre-deployment (930 honest episodes, Wilson CI
6.10-9.52%) to 37.63% post-deployment (93 honest episodes, CI 28.46-47.79%) —
non-overlapping CIs, a genuine ~4-5x improvement, and an unexpected bonus on
top of the intended battle/catching push. Reward mix stayed healthy throughout
(`battle` reliably hits its new cap, `frontier` still dominates it ~5x — no
farm signature); `unique_maps`/`unique_cells` stayed flat/stable — no
exploration collapse.

### Win/catch counters and a duplicate-catch reward (2026-10-22)

Follow-on to the push above, closing two real gaps it surfaced: "has a
trainer battle ever been won" and "how many Pokémon have been caught" were
both unanswerable from existing metrics — only a combined, capped, decayed
dollar figure existed for battle (engagement + win could not be told apart
after the fact), and catching only tracked a run-wide *high-water mark*
(`pokedex_owned_max`, stuck at 6 since episode 3192), not a count of
individual catches. Also: duplicate-species catches (a Rattata when one is
already owned) paid nothing at all, with no non-zero signal to reinforce
"catching is good" once per-species novelty runs out.

All additions, no existing reward path touched:

- **`first_trainer_battle_win_reward`** (50.0, seeded 15.0, honest floor
  0.15): an uncapped, honest/seeded-decayed, checkpoint-style bonus on the
  first-ever TRAINER battle WIN this run — mirrors `first_trainer_battle`
  exactly but keyed to the `enemy_hp>0->0` KO rather than the engagement
  transition, and deliberately larger (wins are rarer and more valuable than
  merely entering a fight). Classified into the `battle` domain (not
  `milestone`, unlike the engagement bonus) — see
  `domain_reward_balancer.py`'s `discovery_domain`; this also gives the
  `battle` domain a genuine run-wide "first-ever" progress signal it
  previously lacked entirely (it fell back to a weaker per-episode "did any
  battle reward fire" proxy, see `Rewards.progressed_domains`).
- **`duplicate_catch_reward`** (20.0, capped per-episode at
  `catch_reward_episode_cap` 60.0, decaying per-map like the existing battle
  rewards): pays only when `party_size` increases while flagged as a wild
  encounter (`battle_type == 1` on this step or the one before — excludes
  starter/gift/egg pickups, which never have an adjacent wild battle_type)
  **and** `pokedex_owned` did *not* also increase that step. A genuinely new
  species still pays only through the existing, untouched
  `pokedex_owned_reward` path (already large, already correct) — this adds
  the missing "smaller reward for an already-owned species" case, it does
  not touch the new-species case at all.
- **Discrete per-episode counters** (`Rewards.get_episode_event_counts`):
  `trainer_wins`, `catches`, `new_species_catches`, `duplicate_catches` —
  plumbed through `vec_env.py`'s terminal info, `_commit_episode`, and
  `episode_data` into `training_metrics.json` as new parallel arrays
  (`episode_trainer_wins` etc.), exactly like the Phase-4 `episode_flag_fires`
  addition before them — shorter than `episode_rewards` for any run that
  predates this change, since `episode_data` persists across restarts and old
  episodes were never passed through the new code path. Also exported (and
  previously computed but silently dropped): raw per-episode `pokedex_seen`
  / `pokedex_owned` arrays, closing a gap where only the run-wide max was
  ever visible.
- **New chart** `Results/event_counts.png` (honest episodes only, rendered
  automatically every checkpoint): trainer-battle wins per episode, and
  catches split new-species vs duplicate-species, both as 100-episode
  rolling means. Renders nothing (not a misleading all-zero chart) for any
  run segment that predates this instrumentation.
- **Fixed** `Results/reward_sources.png`'s illegible overlapping bar-chart
  labels (now rotated), and it now computes its two main panels on
  **honest-only** episodes (per `AGENTS.md`'s central discipline) instead of
  mixing seeded in — with a third panel, when seeded data also exists,
  comparing honest-vs-seeded summed reward trend as a cheap divergence check.

Tested: 21 new unit tests in `tests/test_battle_and_milestone_rewards.py`
(duplicate-catch detection/gating/decay/cap, first-trainer-win
reward/decay/discovery/domain, episode counters), one updated regression
guard (`test_step_penalty.py`'s breakdown-keys set), full suite 419/419. A
60-rollout/2-env/3000-step smoke run completed with no errors; confirmed the
new JSON fields and `event_counts.png`/`reward_sources.png` render correctly
against real emulator output.

**How to judge this**: these are small, capped, decaying, additive reward
channels (not a redesign of anything existing) — the main thing to watch is
simply that the new counters finally show non-zero values as the run
continues (first trainer win, first few duplicate catches), and that
`event_counts.png`'s rolling means trend upward over time the way
`reward_sources.png`'s established panels already do for frontier/milestone.

### Parallelism experiment: num_envs 8 -> 16 (2026-10-22)

The run is CPU-bound by PyBoy emulation, not GPU/gradient-bound, and the host
has 28 cores (20P + 8E) sitting at ~91% idle with `num_envs=8` — meaningful
headroom before any distributed-training rewrite (multiple processes
synchronizing one model) would be justified. `num_envs` is a pure
infrastructure knob (already-tested `vec_env.py` multiprocessing path, no
reward-semantics change), so this is a time-boxed, independently-revertible
A/B test: `configs/stages/open_world.json`'s `num_envs: 8 -> 16`, in its own
commit so it can be reverted without touching the reward/counter work above.
Snapshot before the change: episode/archive/frontier-pool state at the
restart that also deployed the reward changes above (same restart, two
separate commits). **Judge after a few hours** of wall-clock: did rollout
throughput (`it/s` in the tqdm bar) meaningfully improve, and did episode
quality (`h_r`, `h_maps`, archive growth) hold steady rather than degrade
(e.g. from CPU contention with the main process's PPO update step)? If not
clearly beneficial, revert `num_envs` to 8 and restart — a one-line,
one-restart rollback, independent of everything else in this file.

**Result (judged at the 4-hour mark, 2026-10-22): kept at 16, not
reverted.** Throughput held a consistent ~35-48% per-env-normalized
improvement over the full window (it/s ranged 3.46-4.01s/it vs the
`num_envs=8` baseline's 2.5-2.7s/it, which normalizes to roughly
4.0-4.6 env-steps/sec vs baseline's ~3.1) with zero quality degradation —
`h_r`/`h_maps`/`h_cells`/archive growth all held or improved throughout.
Closed; do not re-litigate without new evidence of a regression.

### Antagonistic audit, infra bug fixes, and a frontier-weight push (2026-10-22/23)

A fresh, non-fork agent was asked to independently re-derive every headline
number in this file from `training_metrics.json`/`info.pth` directly
(rather than trust cached claims) and adversarially trace the reward code
for double-counting, farming loopholes, and sign/off-by-one errors. Full
findings below; the short version is that the headline numbers held up
(and were slightly understated), two real bugs were found and fixed, and
one genuine, still-open concern was surfaced (zero new-species catches).

**Confirmed, numbers updated:**
- Honest flag-31 recurrence post-restart (episode ≥4570): **66.1%**
  (154/233 honest, Wilson CI 59.8-71.9%) — higher than the ~58-63% being
  tracked live at the time, since more episodes had accumulated.
- Honest trainer-battle-win rate: **68.7%** at the time of the audit,
  continuing to climb gently afterward to ~76-77% before this session's
  frontier-weight change (see below) was deployed.
- `DomainRewardBalancer` persistence fix (`96cfae7`) confirmed genuinely
  working — real accumulated running-mean/var state across a restart, not
  a per-worker-reset artifact.
- `stuck_action_temperature: 5.0` provenance, previously flagged
  "needs confirmation" in this file — **resolved, not drift.** It is a
  deliberate, dated reintroduction from commit `43f1021`
  ("Fix stuck-exploration regression and rebalance reward toward
  fundamentals", 2026-10-15) with a documented Route-29 chokepoint
  diagnosis and empirical validation in that commit's own message. The
  *falsified-pair* entry in the "Explicitly closed" table below (9.0/0.5)
  is a separate, already-correctly-closed finding; this just closes the
  remaining "is 5.0 itself drift" ambiguity. Only remaining open question:
  whether 5.0 specifically (vs. some other magnitude) was swept — low
  priority, no evidence of a current problem.

**Sharper / more concerning than previously tracked:**
- **Honest pokedex ceiling is 4, not 6** — no honest episode has *ever*
  reached `pokedex_owned` 5 or 6 across 3397+ honest episodes (hit twice,
  both at exactly 4); all 14 episodes that ever hit 6 were Go-Explore
  seeded. Confirmed again via `training_health_report.py`'s new section
  (below) as of episode 5656: still 4, unchanged all session.
- **Zero new-species catches since the win/catch counters were added**
  (`ae23734`) — as of episode ~5656, 236 consecutive honest catches, every
  single one a duplicate species. The `catching` domain is already the
  most strongly boosted (`domain_priority_weights.catching: 2.0`); raising
  it further without first fixing reachability is very likely to just
  produce *more* duplicate-catch farming, not new species. **Open,
  unresolved — see the frontier-weight push below, which targets this
  exact bottleneck indirectly.**
- **Honest map-discovery frontier had been flat since episode ~474**
  (not ~2032 as earlier tracked) — a ~5200-episode gap by the time of this
  audit. 18 of the run's 37 total discovered maps have only ever been
  reached via Go-Explore seeding, never honestly. Per this file's own
  plateau-vs-collapse test: honest per-episode coverage
  (`unique_maps`/`unique_cells`) was still slowly *rising* throughout, so
  this reads as a plateau (out of ideas), not a collapse (unlearning) —
  but it is the single most load-bearing diagnosis behind this session's
  final change (frontier-weight boost, below).

**Two real bugs found and fixed (both committed, tested, pushed):**
1. `_catch_progress`'s "catch landed the step after battle_type already
   reverted to 0" gate (`rewards.py`, the wild-battle detection's
   previous-step half) was dead code: `_battle_progress` runs first each
   step and unconditionally overwrites the shared `_prev_battle_type` with
   the *current* step's value before `_catch_progress` could read it as
   "previous." Fixed by giving `_catch_progress` its own independent
   `_prev_catch_battle_type` tracker (commit `116702c`). Low severity —
   made duplicate-catch detection slightly less reliable in one timing
   edge case, not exploitable.
2. `training_metrics.json` appearing to be missing the new
   `trainer_wins`/`catches`/`new_species_catches`/`duplicate_catches`
   keys — turned out to be a **deploy-lag artifact**, not a code bug: the
   live process was still executing pre-fix code in memory even though
   the fix was already on disk. Resolved by the next restart; confirmed
   the JSON correctly carries all four keys afterward. No code change was
   needed, but this is a good general lesson: a "the file on disk doesn't
   match what the running process is doing" mismatch is a deploy-lag
   question first, a code-bug question second.

Also wired the new win/catch counters and the honest pokedex ceiling into
`tools/training_health_report.py` (a new `report_battle_and_catching`
section, commit `116702c`) per this file's own standing "bake findings
into the health report" item — it independently reproduces the audit's
headline numbers (trainer-win rate, honest pokedex ceiling, the
zero-new-species warning) in one command going forward.

**Inference pipeline bugs found while chasing a fresh flag-31 video**
(user wanted a video of a *current*-policy honest egg-delivery run; the
existing one predated this session's reward changes): 7 consecutive
standalone `run_inference` attempts missed flag 31 despite live training's
~60-68% honest rate. Root-caused to two stacked operator/code bugs, both
now fixed:
- Every `main.py --model inference ...` invocation was missing
  `--use_config configs/stages/open_world.json`, so it ran against bare
  defaults — no `stuck_action_temperature`, no `domain_reward_balancing`,
  none of the stage's tuned reward config. An invocation mistake, not a
  deeper model problem, but it silently invalidated every inference run
  made this way all session (including the very first demo full
  playthrough video generated earlier in the session).
- `PoliwhiRL/evaluator/model_evaluator.py`'s `_sample_action` was
  reimplementing the forward pass directly against `model.actor_critic`
  instead of calling `PPOModel.get_action`, silently bypassing
  `_behaviour_policy` (the `stuck_action_temperature` escalation training
  relies on to escape navigation stalls) and `_normalise_base_policy`.
  Fixed to delegate to `get_action` (commit `765b357`); added a
  `truncation_cause` print to the inference summary (was previously
  invisible) to make this kind of diagnosis faster next time.
- Even after both fixes, standalone single-shot inference still missed
  flag 31 on the next two attempts (consistent `checkpoint_honest=80.0`,
  `truncation_cause=stagnation` both times) — concluded this is likely
  this run's known honest-frontier plateau manifesting in a small sample,
  not a remaining infra bug. Rather than keep burning cycles on
  standalone inference, pivoted to capturing a video directly from the
  *live training process's* own honest rollouts instead (below).

**`recapture_checkpoints` mechanism (commit `2b0c657`, config removed
again in `eb6e2ba` once used):** a new config key — flag numbers to force
a fresh first-honest recording of on the run's next honest fire, even
though an older recording already exists from a weaker policy. Used once
to get a current "Gave Mystery Egg To Elm" video
(`Training Outputs/open_world/Runs/checkpoints/0031-Gave-Mystery-Egg-To-Elm/0031-Gave-Mystery-Egg-To-Elm.mp4`)
reflecting this session's reward improvements. **Caveat discovered the
hard way:** this pops the dedup entry on *every* restart it's present for,
not just once ever — left in the config, it silently re-captured (and
overwrote) the same video a second time on the very next restart. Removed
from `configs/stages/open_world.json` once the video existed; if reused
for another flag in future, remove it again promptly after the first
honest refire, or expect repeat re-captures on every subsequent restart.
Also fixed in the same commit: a latent crash in `load_model`'s
`checkpoint_recordings` key-casting (`int(k)` on every key would have
crashed on the non-numeric string keys threshold-checkpoints use, e.g.
`"pokedex_owned_6"` — only numeric keys are cast now).

**Threshold-checkpoint recordings + auto-video (commit `b9da4eb`):** a
general mechanism (`threshold_checkpoints` config) for recording the first
HONEST episode whose cumulative in-episode metric crosses a threshold
(currently `pokedex_owned >= 6`, titled "Caught 6 Pokemon") — directly
solves the seeded-vs-honest gap for metrics that only a high-water mark
previously tracked (the existing `pokedex_owned_max` ceiling of 6 was set
by a seeded episode; this will only ever fire for a genuinely honest
reach). Every checkpoint/threshold recording (not just new ones) now also
auto-converts its PNGs to an MP4 and deletes the raw frames afterward,
closing a disk-usage gap (freed ~200MB on the 9 existing milestone
recordings retroactively).

**Design review: reward-shaping proposals for pushing past the
current plateau (2026-10-23).** A dedicated review (fork, full session
context) was asked to evaluate four ideas aimed at the stalled honest
frontier / zero-new-species / no-gym-progress cluster of symptoms, and to
be genuinely adversarial about abuse vectors, not just propose raising
numbers:

1. **New-area reward should always be net-positive — shipped.** Diagnosis:
   `frontier` had no `domain_priority_weights` entry while `battle` (1.6x)
   and `catching` (2.0x) did — a plausible, durable asymmetry tipping a
   policy's EV toward exploiting known territory over pushing into new
   ground, consistent with the ep-474 stall. Fix: add
   `"frontier": 1.3` (commit `f48b914`) — a one-line, low-risk,
   zero-new-abuse-surface change, since it reweights an already-gated,
   already-decayed, already-anti-farmed signal rather than raising a raw
   constant (which the balancer's mean-based normalization would just
   shrink back down). **Not yet evaluated — too little post-deploy data
   at session end; see "Immediate next step" above.**
2. **Reward raw EXP gain, not just level-up thresholds — not shipped.**
   Would double-count with the existing battle-win reward (same KO event)
   and is farmable via trivial wild-Pokémon grinding loops. If pursued,
   belongs as a multiplier *inside* `_battle_progress`'s existing win-bonus
   computation (e.g. scaled by `log(1+exp_gained)` to differentiate a
   tough trainer win from a trivial wild KO), not a new parallel channel —
   and needs an EXP-logging pass first to size the actual farming
   exposure before building anti-farm machinery for it.
3. **Pokémon Center heal reward gated on an actual HP/PP deficit — not
   shipped.** The highest-risk of the four proposals: even with a strict
   deficit-fraction gate, a per-episode cap, and per-map decay, it opens a
   damage→heal→damage farming loop — structurally the same failure mode
   as the battle-flee reward this project already deleted for teaching
   exactly the opposite bad habit (permanent avoidance, not courting
   danger for a reward trickle). It would also be solving the wrong
   problem: the policy isn't failing gym attempts for lack of healing
   incentive, it isn't *reaching* Violet City reliably yet.
4. **Instrumentation-only follow-up, not yet done:** confirm whether the
   zero-new-species pattern is a reachability limit (duplicate catches
   confined to already-explored maps) or a policy choice, by cross-
   referencing catch events against each episode's `unique_maps` in
   `training_metrics.json` — cheap, no reward change, should settle
   whether raising `catching` priority further would ever help at all.

Also added a `"Cleared Sprout Tower (Got Flash)"` probe rung (flag 20,
already a fully-wired checkpoint — no reward code needed) between
"Reached Violet City" and "Beat Falkner (Gym 1)" in `probe_goals`, per the
user's updated near-term goal (see "Immediate next step" above).

Tested throughout: full suite (428 tests) passing after every change in
this section; several smoke/live-emulator checks (threshold-checkpoint
claiming, auto-video generation, inference-sampler delegation) confirmed
against the real checkpoint and emulator, not just unit-level.

### Exploration

- **Cross the event gate honestly and repeatedly** — the wall after spatial
  saturation is story interaction, and honest flag-31 recurrence is the live proxy
  for it. `PROGRESS.md:213-241` framed the gate; `PLANNING.md:27-28` records the
  longest recurrence gap (~230 episodes) since the breakthrough.
- **Frontier granularity follow-up** — cell-granularity capture broke the spatial
  choke points; whether further choke points need progress- or flag-keyed snapshots
  beyond what is enabled (`configs/stages/open_world.json:57-75`) is open.
  **Needs confirmation** — no dated plan for the next granularity step.

### Training stability

- **Entropy drift** — `current_policy_entropy` down to ~1.30, above the 1.0
  intervention floor but the lowest since the restart: watch, do not act
  (`PLANNING.md:29-31`).
- **Whether the longer budget is live** — `episode_length: 24576` is committed in
  the config, but `PLANNING.md:9-15` records that the running process was still on
  16384 and that restarting required explicit sign-off. Whether the live run has
  since been relaunched at 24576 — **needs confirmation**; do not restart
  unilaterally.
- **Rollout budget ceiling** — the run-2 diagnosis was that 8192 rollouts
  over-trains a saturated signal and that useful signal is exhausted by ~5k
  rollouts (`PROGRESS.md:142-152`). Whether a cap, or fresh-signal seeding, was
  adopted since — **needs confirmation**.
- **`stuck_action_temperature: 5.0`** is present at `configs/stages/open_world.json:48`,
  yet `PROGRESS.md:37-40` records the stuck-action temperature among the anti-stuck
  machinery deleted on 2026-07-16. Whether this is a deliberate reintroduction, a
  probe-only setting, or drift — **needs confirmation**. Flagged, not changed.

### Tooling

- **Bake findings into `tools/training_health_report.py`** rather than leaving
  one-off scripts, so each run's triage is one command (`AGENTS.md:129-131`).
- **ROM exporter gaps** — exported collision/walkable mask, PNG rendering, and the
  world/warp graph remain unbuilt; collision is decoded at runtime in
  `PoliwhiRL/environment/rom_collision.py` instead
  ([`reference/rom-maps.md`](reference/rom-maps.md)).

### Docs

- Keep this page current when a lever lands or is falsified; prefer a comment at the
  point of implementation over prose here (`AGENTS.md:135-144`).
- `CHANGELOG.txt` **remains the append-only history log at the repo root** — it is
  not being folded into `docs/`, and new entries keep going there.
- The documentation-consolidation pointer stubs left at root (`RAM_MAPPING.md`,
  `rom_maps.md`, `next_steps.md`) point into `docs/`; their history lives in git.
- **Code-comment cleanup pass (open, code-hygiene).** Eight in-code docstring
  pointers still name the old root paths, so they now resolve to the pointer stubs
  instead of the canonical pages: `PoliwhiRL/environment/gym_env.py:138`,
  `PoliwhiRL/environment/gym_env.py:355`, `PoliwhiRL/environment/gym_env.py:1114`;
  `PoliwhiRL/environment/RAM.py:83`, `PoliwhiRL/environment/RAM.py:88`,
  `PoliwhiRL/environment/RAM.py:124`, `PoliwhiRL/environment/RAM.py:346`;
  `tools/rom_maps/rom_map_extractor.py:7`. They need a separate pass to repoint them
  at `docs/reference/ram-mapping.md` and `docs/reference/rom-maps.md`. No `.py` file
  was touched by the documentation remediation — the stubs were reworded to be honest
  about this instead.
- `Manual Investigation States/` and `Reward Documentation/` stay exactly where they
  are at the root: the first is gitignored but consumed at runtime by
  `Scripts/verify_flags_from_states.py:29`. Neither is moved into `docs/`.

## Explicitly closed / superseded

Things the folded-in files proposed which this branch has since implemented,
removed or falsified. Do not re-open these without new evidence.

| Proposed by | Status now |
|---|---|
| `ppo_agent.py` single-environment agent | **Superseded.** Absent from the tree; the training loop is `PoliwhiRL/agents/PPO/vec_ppo_agent.py` (vectorised). |
| 4-layer transformer trunk | **Reverted.** Back to 2 layers for faster iteration — "the bottleneck is reward/seeding, not model capacity" (`PROGRESS.md:322`). Trunk geometry is now config-driven (`num_layers`/`n_heads`/`d_model`, `PROGRESS.md:268-271`). |
| `episode_length: 16384` | **Superseded** by 24576 in `configs/stages/open_world.json:14`; the rationale is the lengthened flag-31 critical path (`PLANNING.md:35-53`). |
| `PoliwhiRL/reward_evaluator/` | **Renamed** to `PoliwhiRL/reward_evaluation/` (`docs/branch-changes.md:44`). |
| `first_steps.json` / `second_steps.json` | **Absent** from the repo; superseded by the JSON config tree plus checkpoint/resume state. No root file by those names exists. |
| Historical `freeform_gamearea*` stage configs | **Removed** (`PROGRESS.md:5-6`) and consolidated into `configs/stages/open_world.json` (`PROGRESS.md:349-350`). Caveat: `configs/stages/freeform_gamearea_event.json` is **still present** on disk — **needs confirmation** whether it is current or a leftover. |
| Novelty floor, cross-episode decay farming, directional-frontier forecast, global-visit-count feature, battle-flee reward | **Deleted** in the 2026-07-16 exploration rebuild (`PROGRESS.md:23-28`). |
| Entropy servo, entropy plateau reset/boost, periodic archive decay | **Deleted** (`PROGRESS.md:37-40`); fixed entropy retained. |
| `export_seed_state` / `apply_seed_facts` fact-bundling | **Unnecessary** — reset's no-op startup step re-baselines rewards, so a seeded start pays only for territory beyond the seed (`PROGRESS.md:174-177`). |
| Stuck-temperature retune (`stuck_action_temperature: 9.0`, `onset_exponent: 0.5`) | **Falsified for `configs/stages/open_world.json`** head-to-head: 92.3% vs 67.9% honest stagnation-truncation, 1 vs 6 honest flag-31 fires (`PLANNING.md:86-93`). In `configs/stages/open_world.json:48` the knob is **not** at its default — it is set to `5.0`, while the default is `0.0` = disabled (`PoliwhiRL/models/PPO/ppo_model_implementation.py:42-44`) — and `stuck_temperature_onset_exponent` is absent from that file, so that stage runs the 5.0 peak on the linear ramp (`:45-54`). The 9.0/0.5 retune is what was falsified for this experiment; it is **not** the value the stage is shipping. **Not falsified everywhere:** the retuned pair is still shipped verbatim in `configs/stages/open_world_probe_temp.json:5-6` (`stuck_action_temperature: 9.0`, `stuck_temperature_onset_exponent: 0.5`), which `docs/configuration.md` lists as a usable stage. *Recommendation (not yet actioned):* delete `open_world_probe_temp.json` or mark it deprecated so the falsified pair is not runnable as if it were current. |
| `open_world_probe_temp` config/output path | **Retired** one-shot experiment, not a second production path (`PLANNING.md:108-110`). The file `configs/stages/open_world_probe_temp.json` is still on disk — **needs confirmation** before it is trusted or touched. |
| `next_steps.md` as a live backlog | **Cleared 2026-07-16** (`next_steps.md:3`); it was already a pointer stub before this consolidation. |

## Provenance

| Open item (above) | Came from |
|---|---|
| Honest second catch / pokédex stuck at 1.0 | `PLANNING.md:22-25`, `PLANNING.md:78-85` |
| Early-game leg variance | `PLANNING.md:70-77` |
| Event-novelty ratio and exclude-flag hygiene | `PROGRESS.md:248-261`, `PROGRESS.md:332-333` |
| Non-monotonic-counter vigilance | `AGENTS.md:105-111` (standing rule, restated) |
| Event gate / flag-31 recurrence | `PROGRESS.md:213-241`, `PLANNING.md:19-28` |
| Frontier granularity follow-up | `PROGRESS.md:154-202`, extrapolated — **needs confirmation** |
| Entropy drift watch | `PLANNING.md:29-31` |
| Live-run budget sign-off | `PLANNING.md:9-15`, `PLANNING.md:57-64` |
| Rollout budget ceiling | `PROGRESS.md:142-152` |
| `stuck_action_temperature: 5.0` discrepancy | Observed in this pass; `PROGRESS.md:37-40` vs `configs/stages/open_world.json:48` |
| Bake checks into `training_health_report.py` | `AGENTS.md:129-131` |
| ROM exporter gaps | `tools/rom_maps/README.md:100-111` (now [`reference/rom-maps.md`](reference/rom-maps.md)) |
| Evidence-first discipline / generalise over hand-authored fixes | `PLANNING.md:94-101` |
| Do-not-do list (no unilateral restart, no re-litigating stuck temperature) | `PLANNING.md:103-112` |
| `CHANGELOG.txt` as the append-only root history | `PLANNING.md:4-5`, `CHANGELOG.txt:1-6` |

## Related

- [`README.md`](README.md) — documentation index.
- [`architecture.md`](architecture.md) and [`branch-changes.md`](branch-changes.md)
  — what the current design is, and what this branch changed.
- [`configuration.md`](configuration.md) — the knobs named above.
- [`subsystems/rewards.md`](subsystems/rewards.md),
  [`subsystems/exploration.md`](subsystems/exploration.md),
  [`subsystems/training.md`](subsystems/training.md) — mechanism detail behind
  each open theme.
- [`operations/run-triage.md`](operations/run-triage.md) — the trace-based
  evidence order every item above insists on.
