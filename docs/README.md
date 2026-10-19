# Documentation index

This is the index for `docs/`. The pages here were written against the current
tree on `flag31-breakthrough-and-tuning`, with `path:line` citations wherever they
make a claim about code, so you can check a statement instead of trusting it. Read
the page for the thing you are about to change; do not read the whole tree first.

## Start here

1. [`getting-started.md`](getting-started.md) — clean checkout to a running
   training session: prerequisites, install, ROM placement, first run, artefact
   layout, running the tests.
2. [`architecture.md`](architecture.md) — the system map: how a frame gets from
   PyBoy to a PPO weight update and back, which file owns which responsibility,
   the observation contract, where a run's artefacts land.
3. [`configuration.md`](configuration.md) — how a run's config is assembled: the
   defaults concatenation, `extends` inheritance, merge order, generated CLI flags,
   the shipped config catalogue, output layout.

Then drill into the subsystem you are touching, or the operations page for the job
at hand.

## Every page

| Path | What it is for |
|---|---|
| [`getting-started.md`](getting-started.md) | Shortest path from a clean checkout to a running training session, a recorded inference run and a passing test suite. |
| [`architecture.md`](architecture.md) | System map: emulator → observation → policy → update, file ownership, observation contract, artefact layout. |
| [`configuration.md`](configuration.md) | Config assembly: defaults, `extends`, merge order, generated CLI flags, config catalogue, output layout. |
| [`branch-changes.md`](branch-changes.md) | What actually changed on `flag31-breakthrough-and-tuning` vs `main` — work clusters, pivotal file, behaviour change, removed files and keys. |
| [`roadmap.md`](roadmap.md) | Forward plan: where the project is now, open work by theme, what is closed or superseded, provenance per item. |
| [`subsystems/environment.md`](subsystems/environment.md) | Environment/observation contract: what `step()` and `reset()` hand back and exactly what is in the observation. |
| [`subsystems/rewards.md`](subsystems/rewards.md) | The arithmetic of one reward step: stage order in `calculate_reward`, every source with its key and resolved default, anti-farming guards. |
| [`subsystems/models.md`](subsystems/models.md) | Policy/value network geometry, how it resolves from config, the two encoder branches, the Transformer-XL memory regime. |
| [`subsystems/training.md`](subsystems/training.md) | Training end to end: rollout collection, buffer tensor contract, per-env GAE, the bootstrap/truncation asymmetry, entropy control. |
| [`subsystems/exploration.md`](subsystems/exploration.md) | Go-Explore: ownership of the visit archive, cell quantisation and `cell_key`, what triggers a frontier capture and a seed. |
| [`operations/run-triage.md`](operations/run-triage.md) | Judging a run from persisted artefacts, not headline reward: evidence chain, field guide, the honest-vs-seeded rule, failure signatures. |
| [`operations/tooling.md`](operations/tooling.md) | Standalone diagnostics over run artefacts: `tools/training_health_report.py` and `tools/world_map.py` in depth, plus the supporting scripts. |
| [`operations/testing.md`](operations/testing.md) | Test suite layout, test selection without custom markers, which tests need the emulator and ROM, what CI does. |
| [`reference/ram-mapping.md`](reference/ram-mapping.md) | Canonical RAM reference: addresses, event flags, map groups, item IDs — verified against live emulator reads, with verified and unverified claims kept separate. |
| [`reference/rom-maps.md`](reference/rom-maps.md) | Canonical ROM-map guide: what the parser decodes, its inputs and CSV outputs, the 388-map claim, the live-WRAM validation, what the runtime collision field is. |
| [`reference/manual-states.md`](reference/manual-states.md) | The manual investigation save-state table. The `.state` files themselves live at the root under `Manual Investigation States/` and must stay there. |
| [`reference/LEGACY-NOTE.md`](reference/LEGACY-NOTE.md) | Warning that `Reward Documentation/rewards.md` describes a superseded reward scheme and is historical context only. |

## Pointer stubs left at the root

Several documents were relocated into `docs/` and their root paths replaced with
short pointer stubs, so any reference that still names the old path resolves to a
page that explains where the content went. The stubs carry no content of their own;
the originals and their history live in git.

| Root path | Points at |
|---|---|
| `RAM_MAPPING.md` | [`reference/ram-mapping.md`](reference/ram-mapping.md) |
| `rom_maps.md` | [`reference/rom-maps.md`](reference/rom-maps.md) |
| `next_steps.md` | [`roadmap.md`](roadmap.md) |

Two root files are deliberately **not** stubbed and stay exactly where they are:
`CHANGELOG.txt` remains the append-only history log at the repo root, and
`PROGRESS.md` / `PLANNING.md` remain the historical experiment log and the dated
plan they were — [`roadmap.md`](roadmap.md) folds their forward-looking content
forward without rewriting them. `Manual Investigation States/` and
`Reward Documentation/` likewise stay at the root: the first is gitignored and read
at runtime by `Scripts/verify_flags_from_states.py:29`.

## Implementation beats prose

Per [`../AGENTS.md`](../AGENTS.md): if a page and the code disagree, **the code is
right**. These pages describe a codebase that iterates quickly on reward design and
training mechanics, and prose describing *how* something works today is the first
thing to go stale. When you learn something true and specific about a mechanism,
prefer a comment at the point of implementation to an edit here, and cite
`path:line` when you do write it down.

## Related

- [`../AGENTS.md`](../AGENTS.md) — codebase map, run-triage rule, documentation-drift
  policy.
- [`../README.md`](../README.md) — project README.
- [`../.opencode/skills/training-run-analysis/SKILL.md`](../.opencode/skills/training-run-analysis/SKILL.md)
  — the repeatable evidence order for analysing a run.

## Known gaps

- `operations/testing.md` records a gap between the documented development environment
  and what CI actually runs; that gap is still open and unresolved here.
- Pages were written against the working tree on this branch. Anything merged to
  `main` after that point is not reflected; re-read the subsystem page's citations
  before trusting a default value.
