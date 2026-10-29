# Roadmap

> The project's forward plan: where training stands, what is still open, what has
> already been settled or removed, and which file each open item came from.
>
> The dated design rounds that used to fill most of this page are archived in
> [`archive/roadmap-history-2026-10.md`](archive/roadmap-history-2026-10.md).
> The consolidated source notes it draws from are in
> [`archive/planning-2026-10-18.md`](archive/planning-2026-10-18.md) and
> [`archive/progress.md`](archive/progress.md).

It is prose about a moving target — where it disagrees with the code or the
configs, the code is right (see [`../AGENTS.md`](../AGENTS.md)). Anything that
cannot be dated or confirmed against the current tree is marked **needs
confirmation**.

## Where the project is now

> Current run status, bottleneck and how to check it: [`status.md`](status.md).
> The paragraph below is older context.

Training runs the milestone-driven reward rebuild — per-episode NGU cell coverage
plus run-wide map/bank novelty decay, checkpoint-labelled story milestones, and an
event-flag novelty channel that pays for first-seen interactions — on top of
Go-Explore frontier seeding at cell, progress and rare-flag granularity
(`configs/stages/open_world.json:131-143`). The agent is trained by vectorised PPO
with real BPTT through a Transformer-XL trunk over multiprocess rollouts, with
auto-resume so a run can be stopped and extended without losing progress. The
canonical open-world stage now runs `episode_length: 49152`
(`configs/stages/open_world.json:27`), doubled from 24576 once settle stepping
made each episode cover more ground.

## Open work

The sections below are the live backlog, grouped by theme. Each item says
what to do and what to check before calling it done. Items marked **needs
confirmation** have not been checked against the current tree — verify
before acting on them.

### Reward design

- **Honest second catch** — `honest_last100_mean_pokedex_owned` has been pinned at
  ~1.0; episodes reach "Learned to Catch" (flag 66) but never complete a real
  second Pokémon. Run a dedicated trace (budget exhaustion vs. failure to execute
  the catch action) *before* any reward change. `archive/planning-2026-10-18.md:84-91`
- **Early-game leg inconsistency** — flag 30 (Mystery Egg) fires anywhere between
  roughly step 250 and step 13,500+ of an episode, so the first legs of the episode
  are their own bottleneck, separate from the egg→Elm backtrack. Needs its own
  trace-based diagnosis, not another length bump. `archive/planning-2026-10-18.md:76-83`
- **Event-novelty signal ratio** — keep watching the `event` reward breakdown for a
  noisy-TV blow-up at `event_novelty_bonus: 10.0`; if junk bits balloon with no real
  discoveries, extend `event_novelty_exclude_flags` or lower the bonus
  (`archive/progress.md:337-338`; current values at `configs/stages/open_world.json:145-147`).
  Whether this is still live in the current run — **needs confirmation**.
- **Non-monotonic-counter farms** — `key_item` / `pokedex` / `level` are deduped
  per episode; a resurgence in any of them means a new code path bypassed the guard
  (`AGENTS.md:115-118`). Standing vigilance, not a one-off task.

### Exploration

- **Cross the event gate honestly and repeatedly** — the wall after spatial
  saturation is story interaction, and honest flag-31 recurrence is the live proxy
  for it. `archive/progress.md:218-246` framed the gate; `archive/planning-2026-10-18.md:33-34` records the
  longest recurrence gap (~230 episodes) since the breakthrough.
- **Frontier granularity follow-up** — cell-granularity capture broke the spatial
  choke points; whether further choke points need progress- or flag-keyed snapshots
  beyond what is enabled (`configs/stages/open_world.json:131-143`) is open.
  **Needs confirmation** — no dated plan for the next granularity step.

### Training stability

- **Entropy drift** — `current_policy_entropy` down to ~1.30, above the 1.0
  intervention floor but the lowest since the restart: watch, do not act
  (`archive/planning-2026-10-18.md:35-37`). The v3 run has since drifted the
  other way, 1.3 → ~1.65, and the coefficient was halved 0.02 → 0.01 at update
  2400 to compensate (commit `8d567af`).
- **The episode budget.** Resolved: `episode_length` is 49152
  (`configs/stages/open_world.json:27`), doubled from 24576 once settle
  stepping made each episode cover more ground (commit `4e3925d`).
- **Rollout budget ceiling** — the run-2 diagnosis was that 8192 rollouts
  over-trains a saturated signal and that useful signal is exhausted by ~5k
  rollouts (`archive/progress.md:147-157`). Whether a cap, or fresh-signal seeding, was
  adopted since — **needs confirmation**.
- **`stuck_action_temperature: 5.0`** is present at `configs/stages/open_world.json:92`,
  yet `archive/progress.md:42-45` records the stuck-action temperature among the anti-stuck
  machinery deleted on 2026-07-16. **Resolved** — this is a deliberate,
  dated reintroduction from commit `43f1021`, not drift; see
  [`archive/roadmap-history-2026-10.md`](archive/roadmap-history-2026-10.md).
  The *falsified* 9.0/0.5 pair is a separate finding, closed below.

### Tooling

- **Bake findings into `tools/training_health_report.py`** rather than leaving
  one-off scripts, so each run's triage is one command (`AGENTS.md:139-143`).
- **ROM exporter gaps** — exported collision/walkable mask, PNG rendering, and the
  world/warp graph remain unbuilt; collision is decoded at runtime in
  `PoliwhiRL/environment/rom_collision.py` instead
  ([`reference/rom-maps.md`](reference/rom-maps.md)).

### Docs

- Keep this page current when a lever lands or is falsified; prefer a comment at the
  point of implementation over prose here (`AGENTS.md:145-154`).
- `CHANGELOG.txt` **remains the append-only history log at the repo root** — it is
  not being folded into `docs/`, and new entries keep going there.
- Superseded material lives in [`archive/`](archive/), not at the repo root and
  not inline here. Each archived page carries a banner saying what replaced it;
  the roadmap's provenance table cites line ranges in them, so check the banner
  before treating an archived line as current.
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
| 4-layer transformer trunk | **Reverted.** Back to 2 layers for faster iteration — "the bottleneck is reward/seeding, not model capacity" (`archive/progress.md:322`). Trunk geometry is now config-driven (`num_layers`/`n_heads`/`d_model`, `archive/progress.md:273-276`). |
| `episode_length: 16384` | **Superseded** by 49152 in `configs/stages/open_world.json:27` (16384 → 24576 → 49152); the rationale is the lengthened flag-31 critical path (`archive/planning-2026-10-18.md:41-59`) and then settle stepping. |
| `PoliwhiRL/reward_evaluator/` | **Renamed** to `PoliwhiRL/reward_evaluation/` long before this branch (commit `6654e4b`); already the name on `main`. |
| `first_steps.json` / `second_steps.json` | **Absent** from the repo; superseded by the JSON config tree plus checkpoint/resume state. No root file by those names exists. |
| Historical `freeform_gamearea*` stage configs | **Removed** (`archive/progress.md:10-11`) and consolidated into `configs/stages/open_world.json` (`archive/progress.md:354-355`). Caveat: `configs/stages/freeform_gamearea_event.json` is **still present** on disk — **needs confirmation** whether it is current or a leftover. |
| Novelty floor, cross-episode decay farming, directional-frontier forecast, global-visit-count feature, battle-flee reward | **Deleted** in the 2026-07-16 exploration rebuild (`archive/progress.md:28-33`). |
| Entropy servo, entropy plateau reset/boost, periodic archive decay | **Deleted** (`archive/progress.md:42-45`); fixed entropy retained. |
| `export_seed_state` / `apply_seed_facts` fact-bundling | **Unnecessary** — reset's no-op startup step re-baselines rewards, so a seeded start pays only for territory beyond the seed (`archive/progress.md:179-182`). |
| Stuck-temperature retune (`stuck_action_temperature: 9.0`, `onset_exponent: 0.5`) | **Falsified for `configs/stages/open_world.json`** head-to-head: 92.3% vs 67.9% honest stagnation-truncation, 1 vs 6 honest flag-31 fires (`archive/planning-2026-10-18.md:92-99`). In `configs/stages/open_world.json:92` the knob is **not** at its default — it is set to `5.0`, while the default is `0.0` = disabled (`PoliwhiRL/models/PPO/ppo_model_implementation.py:37-39`) — and `stuck_temperature_onset_exponent` is absent from that file, so that stage runs the 5.0 peak on the linear ramp (the exponent defaults to `1.0`, `PoliwhiRL/models/PPO/ppo_model_implementation.py:47-49`). The 9.0/0.5 retune is what was falsified for this experiment; it is **not** the value the stage is shipping. **Not falsified everywhere:** the retuned pair is still shipped verbatim in `configs/stages/open_world_probe_temp.json:5-6` (`stuck_action_temperature: 9.0`, `stuck_temperature_onset_exponent: 0.5`), which `docs/configuration.md` lists as a usable stage. *Recommendation (not yet actioned):* delete `open_world_probe_temp.json` or mark it deprecated so the falsified pair is not runnable as if it were current. |
| `open_world_probe_temp` config/output path | **Retired** one-shot experiment, not a second production path (`archive/planning-2026-10-18.md:114-116`). The file `configs/stages/open_world_probe_temp.json` is still on disk — **needs confirmation** before it is trusted or touched. |
| `next_steps.md` as a live backlog | **Cleared 2026-07-16** (`next_steps.md:3`); it was already a pointer stub before this consolidation. |

## Provenance

| Open item (above) | Came from |
|---|---|
| Honest second catch / pokédex stuck at 1.0 | `archive/planning-2026-10-18.md:28-31`, `archive/planning-2026-10-18.md:84-91` |
| Early-game leg variance | `archive/planning-2026-10-18.md:76-83` |
| Event-novelty ratio and exclude-flag hygiene | `archive/progress.md:253-266`, `archive/progress.md:337-338` |
| Non-monotonic-counter vigilance | `AGENTS.md:115-118` (standing rule, restated) |
| Event gate / flag-31 recurrence | `archive/progress.md:218-246`, `archive/planning-2026-10-18.md:25-34` |
| Frontier granularity follow-up | `archive/progress.md:159-207`, extrapolated — **needs confirmation** |
| Entropy drift watch | `archive/planning-2026-10-18.md:35-37` |
| Live-run budget sign-off | `archive/planning-2026-10-18.md:15-21`, `archive/planning-2026-10-18.md:63-70` |
| Rollout budget ceiling | `archive/progress.md:147-157` |
| `stuck_action_temperature: 5.0` discrepancy | Observed in this pass; `archive/progress.md:42-45` vs `configs/stages/open_world.json:92` |
| Bake checks into `training_health_report.py` | `AGENTS.md:139-143` |
| ROM exporter gaps | `tools/rom_maps/README.md:100-111` (now [`reference/rom-maps.md`](reference/rom-maps.md)) |
| Evidence-first discipline / generalise over hand-authored fixes | `archive/planning-2026-10-18.md:100-107` |
| Do-not-do list (no unilateral restart, no re-litigating stuck temperature) | `archive/planning-2026-10-18.md:109-118` |
| `CHANGELOG.txt` as the append-only root history | `archive/planning-2026-10-18.md:10-11`, `CHANGELOG.txt:1-6` |

## Related

- [`README.md`](README.md) — documentation index.
- [`architecture.md`](architecture.md) — what the current design is.
- [`configuration.md`](configuration.md) — the knobs named above.
- [`subsystems/rewards.md`](subsystems/rewards.md),
  [`subsystems/exploration.md`](subsystems/exploration.md),
  [`subsystems/training.md`](subsystems/training.md) — mechanism detail behind
  each open theme.
- [`operations/run-triage.md`](operations/run-triage.md) — the trace-based
  evidence order every item above insists on.
