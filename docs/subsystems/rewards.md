# Reward system

What this page is for: the exact arithmetic of a single reward step — the order in
which `Rewards.calculate_reward` runs its eleven stages, every reward source with
its config key, resolved default and citation, the anti-farming guards each channel
carries (and the concrete failure each guard exists to prevent), how honest and
seeded income are kept in separate ledgers, the three truncation causes and their
bootstrap consequences, and the domain balancer that turns four raw domain sums into
one delivered scalar. Read this when a reward looks wrong, too large, or is buying
behaviour nobody wanted; the environment-side contract is
[environment.md](environment.md) and the seeding side is
[exploration.md](exploration.md).

**Sources** — §1–§6 of the earlier reconnaissance notes (untracked working-tree notes
under `.doc-scratch/`, never committed, therefore not citable from a fresh clone;
every claim re-checked against the files listed below), with
`PoliwhiRL/environment/rewards.py` (1627 L), `domain_reward_balancer.py` (180 L),
`visit_archive.py` (413 L), `PoliwhiRL/agents/PPO/vec_ppo_agent.py` (2287 L) and
the config JSONs as the underlying citations, on branch
`flag31-breakthrough-and-tuning`. Config keys are abbreviated `rs:` =
`configs/default_configs/reward_settings.json`, `ow:` =
`configs/stages/open_world.json`, `cb:` = `configs/curriculum_base.json`.

## Domains

Four skill domains, a fixed tuple: `("frontier", "battle", "milestone", "catching")`
(`PoliwhiRL/environment/domain_reward_balancer.py:74`). `catching` was split out of
`milestone` so pokédex rewards can be pushed hard without diluting the shared scale
factor of flag/checkpoint/event (`domain_reward_balancer.py:15-20`, `:84-94`).

Grouping happens at the **call site**, not inside the balancer
(`rewards.py:741-752`, cross-referenced from `domain_reward_balancer.py:76-94`):

| domain | sources |
|---|---|
| frontier | `r_front` + `r_map` + `r_menu` |
| battle | `r_battle` |
| milestone | flag + map_goal + maps_visited + key_item + level + event + first_trainer_battle |
| catching | `r_pokedex` |

`whiteout` and `step_penalty` are deliberately **outside** domain scaling — direct
penalties (`rewards.py:769-770`, rationale `domain_reward_balancer.py:79-82`).

## `calculate_reward` order of operations

`rewards.py:674-895`, in this order. The order is load-bearing: the balancer sees
raw domain sums, the clip sees the balanced total, and the accounting rows are
written after the clip so `delivered_total` reconciles.

| # | stage | citation |
|---|---|---|
| 1 | step counter + per-step event/checkpoint fire buffers | `rewards.py:675-678` |
| 2 | validity gate — the whole reward body is inside `if is_ram_state_valid(env_vars)`, else `total = 0.0` | `rewards.py:680`, `:830-831`; gate defined `:107-119` |
| 3 | per-source raw computation | `rewards.py:697-714` |
| 4 | accumulate into `_episode_breakdown` (checkpoints split by provenance) | `rewards.py:716-731` |
| 5 | raw per-domain sums | `rewards.py:741-752` |
| 6 | balance — one `domain_balancer.scale(domain, raw)` call per domain; the delta is recorded as accounting-only `domain_scale_adjustment` | `rewards.py:753-756`, `:757-762` |
| 7 | sum — `total = frontier + battle + milestone + catching + whiteout + step_penalty` | `rewards.py:764-771` |
| 8 | goal-rung fire-step log, optional `terminate_on_goal_complete` | `rewards.py:777-788` |
| 9 | stagnation + battle-stagnation counters | `rewards.py:796-829` |
| 10 | truncation decisions (budget / stagnation / battle_stagnation) | `rewards.py:835-881` |
| 11 | round → clip → dtype, then adjustment bookkeeping, return `(delivered, done)` | round `:885-886`, `np.clip(±self.clip)` `:887`, `np.float32` `:888`, bookkeeping `:890-893`, return `:895` |

`reward_round_dp = null` in `configs/stages/open_world.json:32` means the
canonical open-world run does **not** round, by way of the `is not None` guard at
`rewards.py:885`.

## Reward sources

Keys are the `_episode_breakdown` dict keys (`rewards.py:459-485`).

| source key | domain | config key(s) → default | meaning / mechanism |
|---|---|---|---|
| `flag` | milestone | `flag_progress_reward` rs:2 = 10 · ow:51 = 20 | durable story-flag progress; the breakdown stores `r_flag - r_checkpoint` so checkpoint income is not double-counted — `rewards.py:716`, producer `_global_flag_progress_bonus` `:986-1044` |
| `checkpoint_honest` | milestone | `checkpoint_progress_reward` rs:3 = 40 | checkpoint reached on a from-scratch (unseeded) episode; scale = `max(honest_checkpoint_reward_floor, 1/sqrt(1+n))` — `rewards.py:717-718`, `_checkpoint_progress_bonus` `:964-984` |
| `checkpoint_seeded` | milestone | `seeded_checkpoint_progress_reward` rs:4 = 10 | same checkpoint reached from a Go-Explore seed; lower budget, independent depleting counter — `rewards.py:978-979` |
| `map_goal` | milestone | `map_goal_reward` rs:6 = 20 | exact configured `(bank,num)` coordinate reached; per the docstring the only `goals` return value still paying reward — `rewards.py:691`, `:697`, `:719`, docstring `:688-689` |
| `maps_visited` | milestone | `maps_visited_reward` rs:7 = 0 (off) | count-of-maps-visited goal fire — `rewards.py:690`, `:698`, `:720` |
| `pokedex` | catching | `pokedex_owned_reward` rs:8 = 15 · `pokedex_first_sight_reward` rs:9 = 3 · ow:52 = 30, ow:53 = 8 | seen + owned pokédex thresholds — `rewards.py:701-703`, `:721`, `_global_pokedex_bonus` `:1046-1092` |
| `key_item` | milestone | `key_item_pickup_reward` rs:10 = 5 · ow:54 = 25 | `key_items_count` threshold crossings — `rewards.py:705`, `:722`, `_global_key_item_bonus` `:1132-1152` |
| `level` | milestone | `level_up_reward` rs:19 = 2 | party total-level threshold crossings — `rewards.py:704`, `:723`, `_global_level_bonus` `:1094-1130` |
| `battle` | battle | `battle_engagement_reward` rs:11 = 3.0 · `battle_win_reward` rs:12 = 8.0 · `battle_decay_coef` rs:13 = 0.2 · `trainer_battle_premium_multiplier` rs:18 = 2.5 · capped per episode | engagement on `battle_type 0 → nonzero`, win on `enemy_hp > 0 → 0` while in battle; per-`(map_key, battle_type)` decay `/(1 + coef*(n-1))`; trainer premium on both — `rewards.py:707`, `:724`, `_battle_progress` `:1154-1250` |
| `first_trainer_battle` | milestone | `first_trainer_battle_reward` rs:15 = 40.0 · `first_trainer_battle_seeded_reward` rs:16 = 10.0 · `first_trainer_battle_honest_floor` rs:17 = 0.1 · uncapped | one-time bonus on the first `battle_type == 2` entry of an episode; honest/seeded-decayed like a checkpoint; bypasses the episode cap — `rewards.py:725`, `:1208-1227`, docstring `:1174-1177` |
| `new_map` | frontier | `new_map_reward` rs:21 = 5.0 · `new_bank_reward` rs:22 = 20.0 | first entry to a `(bank,num)` this episode, plus extra for a first entry to a new BANK; decays run-wide with `map_lifelong_decay` — `rewards.py:708`, `:726`, `_new_map_bonus` `:1252-1310` |
| `frontier` | frontier | `frontier_novelty_bonus` rs:20 = 1.0 · ow:33 = 0.25 · `revisit_novelty_scale` ow:44 = 0.85 · `frontier_lifelong_decay` rs:23 / ow:34 · `frontier_lifelong_decay_floor` ow:35 | first step onto each cell THIS episode; revisits pay discounted after a flag re-open — `rewards.py:709`, `:727`, `_frontier_novelty_bonus` `:1312-1374` |
| `menu` | frontier | `menu_novelty_bonus` ow:46 = 0.25 (absent from rs) | novel menu / screen discovery — `rewards.py:710`, `:728`, `_menu_novelty_bonus` `:1376-1403` |
| `event` | milestone | `event_novelty_enabled` ow:70 · `event_novelty_bonus` ow:71 = 10.0 · `event_novelty_exclude_flags` ow:72 = [26, 1726] (absent from rs) | run-wide-first story event flag fires; gated on `_track_event_fires` — `rewards.py:711-713`, `:729`, `_detect_event_fires` `:1405-1439`, `_event_novelty_bonus` `:1441-1452` |
| `whiteout` | unscaled penalty | `whiteout_penalty` rs:27 = −20 | party wipe — `rewards.py:714`, `:730`, `_check_whiteout` `:901-925` |
| `step_penalty` | unscaled penalty | `step_penalty` rs:28 = −0.02 · ow:31 = 0.0 (off) | flat per-step cost, added directly to `total`, not domain-scaled — `rewards.py:731`, `:770` |
| `domain_scale_adjustment` | accounting | `domain_reward_balancing_enabled` ow:38 · 0 when disabled | `scaled_total − raw_total`, keeps `delivered_total` reconcilable — `rewards.py:757-762`, `:480-484` |
| `rounding_adjustment` | accounting | `reward_round_dp` rs:30 = 2 · ow:32 = null (no rounding) | bookkeeping only — `rewards.py:890` |
| `clip_adjustment` | accounting | `reward_clip` rs:29 = 50 | bookkeeping only — `rewards.py:891` |
| `delivery_cast_adjustment` | accounting | (float32 cast) · 0 | bookkeeping only — `rewards.py:892` |
| `delivered_total` | accounting | — | running sum of what was actually returned to the agent — `rewards.py:893` |

## Anti-farming guards

Every guard below exists because of something that actually happened. The dated
notes are the justification, not decoration — keep them when editing.

| guard | mechanism | failure it prevents | citation |
|---|---|---|---|
| `_milestone_refire_scale(kind, key)` | run-wide depletion `1/sqrt(1 + prior_episode_fire_count)` read from the visit archive, **plus** a per-episode dedupe set `_milestone_thresholds_paid` keyed `(str(kind), int(key))` that returns `0.0` on a re-cross. No numeric knob — it is structural. | (a) a first corridor milestone becoming a full-price annuity that outbids exploration (observed 2026-07-11); (b) a non-monotonic RAM counter re-paying the same threshold on every up-tick — `key_items_count` oscillation drove `key_item` to ~17k/episode and collapsed honest unique_maps 16 → 5.6 (observed 2026-07-20) | `rewards.py:927-962` (depletion `:960`, archive API `PoliwhiRL/environment/visit_archive.py:222`, dedupe `:956-959`); incident notes `:933-935`, `:951-954` |
| `battle_reward_episode_cap` | engagement + win clamped together per episode (headroom then `min`), decay-independent. Value rs:14 = 30.0, ow:29 = 3.0, attribute default 30.0 | grinding battles on cheap wild encounters crowding out exploration income; also bounds the trainer-premium channel so many trainers in one long episode saturate the cap instead of opening a new uncapped channel | `rewards.py:1247-1249`; `:175`; `:1162-1164`, `:1187-1191` |
| `(map_key, battle_type)` decay keying | decay counters keyed by map **and** battle type, not map alone — separate trackers for engagement and win | grinding wild battles on a map depleting what a trainer battle on that same map would pay, or vice versa ("genuinely different activities shouldn't share a decay counter") | `rewards.py:1194-1195`; `_battle_engaged_maps` `:1201-1202`, `_battle_won_maps` `:1232-1233`; rationale `:1183-1187` |
| `first_trainer_battle_bonus` one-time semantics | `_first_trainer_battle_fired_this_episode` latch (once per episode); run-wide decay via `first_trainer_battle_fire_count(seeded=...)`; honest floor rs:17 = 0.1; deliberately **uncapped**, exempt from `battle_reward_episode_cap` | farming the bonus repeatedly inside one episode; and, via the seeded discount, making a seeded revisit worth the same as consolidating the fight from the canonical start | `rewards.py:1208-1227` (latch `:1208-1210`, decay `:1211-1213`, floor `:1219-1223`); archive `visit_archive.py:239`; cap exemption `:1174-1177` |
| `map_lifelong_decay` | map term ÷ `sqrt(1 + run-wide map_count)`, bank term ÷ `sqrt(1 + run-wide bank_count)`, provenance-aware (`seeded=self.episode_seeded`); **no floor**. Attribute default False (`rewards.py:235`), `ow:36 = true` | the "re-tour the known building cluster every episode for a flat map bonus" optimum; the no-floor choice keeps it a strict novelty signal so re-sweeping can never out-pay discovery | `rewards.py:1298-1305`; `:235`; rationale `:1263-1267` |
| revisit + `flag_reopens_novelty` | each cell pays at most once per reward epoch via `_novel_cells_this_episode`; after a story-flag re-open the true-history set `_ever_visited_cells_this_episode` lets re-opened ground re-pay, discounted at `revisit_novelty_scale` (default 0.5, `ow:44 = 0.85`); before any re-open the two sets are identical, so behaviour stays flat per-episode coverage | in-episode wiggling to farm cells; un-discounted re-traversal of old ground after a flag flip | `rewards.py:1346-1367` (attributes `:252-253`, dedupe `:1346-1347`, discount `:1353`, `:1367`, identical-sets case `:1349-1352`); rationale `:1316-1317`, `:1343-1345` |

## Honest vs seeded

The rule that outranks everything else in this repo: split honest from seeded before
judging anything, because seeded episodes start deep in already-solved territory and
are where farmable channels get exercised hardest (`AGENTS.md:78-88`).

- **Provenance is set by the agent, not the env.** The `episode_seeded` config
  attribute defaults to False (`rewards.py:164`); `_goexplore_cycle` calls
  `vec_env.set_env_state(env_idx, snap["path"], episode_seeded=True)` when seeding
  and `episode_seeded=False` when handing a worker back to the true start
  (`PoliwhiRL/agents/PPO/vec_ppo_agent.py:1089-1091`, `:1098-1100`). The flag is
  queued in `env_is_seeded_pending` and promoted on auto-reset
  (`vec_ppo_agent.py:1076-1077`, `:1104`; arrays allocated `:553-554`). The env
  copies it into the fresh reward calculator at reset
  (`PoliwhiRL/environment/gym_env.py:1154-1155`).
- **Budget differences.** Honest checkpoint pays `checkpoint_progress_reward`
  (rs:3 = 40) floored at `honest_checkpoint_reward_floor` (rs:5 = 0.1) times the
  depletion factor (`rewards.py:980-983`); seeded pays the lower
  `seeded_checkpoint_progress_reward` (rs:4 = 10) divided by its own depletion count
  (`rewards.py:978-979`). Same shape for the trainer bonus: honest 40.0 with floor
  0.1 (`:1219-1223`), seeded 10.0 (`:1214-1217`).
- **Separate counters.** Provenance is carried into the archive queries —
  `checkpoint_fire_count(flag_num, seeded=...)` `visit_archive.py:232`,
  `first_trainer_battle_fire_count(seeded=...)` `:239`, `map_count(..., seeded=)`
  `:371`, `count(..., seeded=)` `:393`. Where the merge API has no metadata
  argument, provenance travels in-band on the record queues —
  `(*map_key, episode_seeded)` `rewards.py:1289`, `(*cell, episode_seeded)`
  `:1358`, rationale `:1356-1357`.
- **Metric fields.** Breakdown keys `checkpoint_honest` / `checkpoint_seeded`
  (`rewards.py:461-462`), selected by
  `checkpoint_key = "checkpoint_seeded" if self.episode_seeded else "checkpoint_honest"`
  (`rewards.py:717`). Per-episode provenance list `episode_seeded` in the agent's
  metrics (`vec_ppo_agent.py:333`, rationale `:327-332`). Honest-only aggregates are
  computed by `honest_mean(key)` over unseeded episodes (`vec_ppo_agent.py:1833`)
  and published as `h_r` / `h_maps` / `h_cells` / `h_flags` (`:1842-1851`).
  `best_discoveries` is tracked off the honest `checkpoint_recordings` count only
  (`:2014-2016`); the free-play best/ high-water mark counts first-honest durable
  checkpoints (`:207`).
- **Reading it.** Healthy = exploration sources (`frontier`, `new_map`, `menu`,
  `event`) dominate and honest `unique_maps` / `unique_cells` / `flag_fires` trend
  up. A rising headline reward over falling honest coverage is a reward-hacking
  collapse, and any single per-source maximum orders of magnitude above the others is
  a farm (`AGENTS.md:90-111`; triage procedure in
  [../operations/run-triage.md](../operations/run-triage.md)).

## Truncation causes

Decided at the tail of `calculate_reward` (`rewards.py:835-881`) and recorded on
`self.truncation_cause`, which surfaces in `terminal_info` as `truncation_cause`
alongside the bootstrap flag `truncated`
(`PoliwhiRL/environment/vec_env.py:167-177`, see
[environment.md](environment.md#worker-ipc-and-terminal_info)).

| cause | decided at | config key → resolved value | bootstrap consequence |
|---|---|---|---|
| `budget` | `rewards.py:835-843` (`steps > max_steps`) | `episode_length` ow:14 = 24576 | **BOOTSTRAPPED.** The only cause the agent bootstraps on — the episode was still progressing, so the unobserved continuation is bootstrapped via `terminal_info` / `_per_env_gae` — `rewards.py:836-843`; terminal observation preserved `vec_env.py:218-225` |
| `stagnation` | `rewards.py:845-863` | `stagnation_truncation_steps` default `"auto"` (cb:23) → `max(256, max_steps // 16)` (`rewards.py:322-324`); 0/None disables | Zero-bootstrap terminal, excluded from the GAE bootstrap mask — `rewards.py:849-851`. Rationale: bootstrapping made the critic value a stuck state at ≈V(post-reset), so a dead pocket looked like a cheap route to a high-value reset with zero advantage to leaving — `:852-857` |
| `battle_stagnation` | `rewards.py:865-881` | `battle_stagnation_truncation_steps` default `"auto"` (cb:24) → `max(512, max_steps // 16)` (`rewards.py:355-357`); 0/None disables | Zero-bootstrap terminal — `rewards.py:871-874`. Backstop for "mash A in one unwinnable battle" (observed 36,202 consecutive battle steps, 88% of an episode, ≈98% A presses — `:336-337`); the primary escape is the stuck-temperature mechanism (`stuck_action_temperature` ow:48 = 5.0) |

Counter maintenance and precedence:

- The free-walking stagnation counter **freezes** during scripted overlays and
  battles (`rewards.py:796-803`).
- Any paid cell — full novelty or discounted revisit — counts as progress, so
  productive backtracking through re-opened ground is not truncated
  (`rewards.py:1359-1363`).
- The battle counter resets on **any** change to enemy HP or party HP, and is
  cleared outside battle (`rewards.py:812-829`).
- Precedence: budget wins if set on the same step; stagnation and
  battle_stagnation only write while `truncation_cause is None`
  (`rewards.py:862-863`, `:880-881`).
- `stagnation_fraction()` is the helper behind both the `stagnation_clock`
  observation feature (`environment.md`) and `final_stagnation_clock`
  (`rewards.py:1510-1523`).

## `DomainRewardBalancer`

| aspect | behaviour | citation |
|---|---|---|
| domains | `("frontier", "battle", "milestone", "catching")`; `catching` split out of `milestone` (2026-10-16) so pokédex pressure cannot dilute flag/checkpoint/event | `domain_reward_balancer.py:74`; `:15-20`, `:84-94` |
| enable | `domain_reward_balancing_enabled` default False, `ow:38 = true`. When disabled `scale()` returns the raw value unchanged, so `total` is identical to the old flat sum | `:117`; `:157-158`, `rewards.py:737-738` |
| `scale(domain, raw_value)` | update per-domain `RunningMeanStd` on the non-zero raw value, then `raw × (1/typical_magnitude) × activity_multiplier(domain) × priority_weight[domain]`. Called once per domain per step, with sources already summed | `:152-167`; `:88-90`, `rewards.py:753-756` |
| mean-based, not std-based | a near-constant stream has ≈zero variance, and std-division would blow the factor up | `:22-36` |
| guards | `min_magnitude` floor (default 0.05, `ow:39 = 0.05`) bounds `1/magnitude` for a domain that hasn't fired, enforced in `_typical_magnitude`; `domain_progress_floor` clamped to [0,1] (default 0.2, `ow:41 = 0.2`); short-circuit on `raw_value == 0.0` | `:118-121`, `:149-150`; `:123`; `:157` |
| progress-activity multiplier | `1 − frac × (1 − floor)` with `frac = min(1, episodes_since_progress / window)`, window default 50 (`ow:40 = 50`); fed once per finished episode by `note_episode_end(progressed_domains)` | `:142-147`; `:122`; `:132-140`, producer `Rewards.progressed_domains` `rewards.py:632-647` |
| static priority weights | `domain_priority_weights`, default 1.0 per domain; `ow:42 = {"catching": 1.75}`. This is the durable lever — raising a raw source magnitude is self-defeating because the running mean absorbs it | `:126-130`; `:48-61` |
| scope | per-worker-local, **not** broadcast across workers (unlike `VisitArchive`) | `:63-70` |
| diagnostics | `diagnostics()` exposes scale_factor / activity / episodes_since_progress / priority_weight per domain | `:169-180` |
| progress signal mapping | discovery-type → domain table; battle has no run-wide discovery type, so it uses a weaker per-episode proxy in `Rewards.progressed_domains` | `:100-112`; `:96-99` |

Tuning guidance: change `domain_priority_weights`, not the raw magnitudes, and
watch `diagnostics()` alongside the honest reward mix
([../configuration.md](../configuration.md),
[../operations/run-triage.md](../operations/run-triage.md)).

## Cross-links

- [environment.md](environment.md) — where `calculate_reward` is called and how the
  breakdown reaches the agent.
- [exploration.md](exploration.md) — the archive that supplies the depletion counts.
- [training.md](training.md) — how the delivered scalar becomes an advantage.
- [../architecture.md](../architecture.md), [../branch-changes.md](../branch-changes.md)
  — this branch's reward changes at the whole-system level.
- [../configuration.md](../configuration.md) — where `rs:` / `ow:` / `cb:` keys live.
- [../operations/run-triage.md](../operations/run-triage.md) — spotting a farm in the metrics.

## Known gaps

Carried forward from the §Unverified section of those earlier reconnaissance notes,
plus this pass's limits.

- **No `rewards.py` lines were re-read in this documentation pass.** The
  `rewards.py:*` citations above are transcribed from the pre-verified scratch
  reconnaissance; treat them as second-hand for this pass and re-check any number
  you intend to act on (`rewards.py` is 1627 L — read ranges, not the whole file).
- `configs/stages/open_world.json`, `configs/default_configs/reward_settings.json`
  and `configs/curriculum_base.json` line numbers were likewise not re-checked here;
  the resolved values are quoted from scratch.
- Whether `goals.py` pays any reward beyond `map_fires` is **docstring-backed, not
  code-verified** — the `calculate_reward` docstring at `rewards.py:684-689` says
  only `map_fires` still pays; `goals.py` (568 L) was never opened.
- Defaults for `menu_novelty_bonus`, `event_novelty_bonus`, `event_novelty_enabled`
  and `revisit_novelty_scale` in `rewards.py.__init__` were not each confirmed
  beyond the attribute lines cited for revisit/re-open (`rewards.py:252-253`);
  `configs/default_configs/reward_settings.json` does not list them, so only
  open-world values are documented.
- `"auto"` for stagnation appears only in `configs/curriculum_base.json:23-24`, and
  the `max_steps` → `episode_length` binding (the attribute assignment line) was not
  read, though `rewards.py:324` / `:357` divide `self.max_steps` as if it is the
  episode budget.
- `DomainRewardBalancer` is per-worker-local, so any statement about a *global*
  domain scale is inference about an aggregate that does not exist in one place
  (`domain_reward_balancer.py:63-70`).
- No web access used; every statement here is from local source at this branch.
