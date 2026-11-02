# Post-gym inspection plan (open_world_v3)

**For:** an agent working by hand with the LLM tooling (`tools/llm_agent/`).
**Goal:** find where the policy gets stuck after Beat Falkner (flag 1213), and
explain *why* in terms of the game world, the observation and the reward. Do
not change the training code during this pass. The output is a written
diagnosis with evidence.

Read these first:
[`.claude/skills/live-env-testing/SKILL.md`](../.claude/skills/live-env-testing/SKILL.md)
(the `drive.py` loop, pitfalls, known mechanics),
[`tools/llm_agent/README.md`](../tools/llm_agent/README.md) and
[`docs/reference/ram-mapping.md`](reference/ram-mapping.md).

## Context: what the run shows (2026-10-29, ep 424, rollout 3450)

- The honest policy reaches Violet City about 70% of the time and gets Flash
  about 60% of the time (probe at rollout 3200). It beat Falkner honestly in
  12 of 299 episodes. Over the last 200 honest episodes the flag-1213 rate is 6%.
- The **continuation probe** starts from `emu_files/states/post_falkner.state`,
  runs 4 episodes of 16,384 steps, and scores **0%** on every goal: Togepi egg
  (flag 45), Route 32 PC (10:13), Union Cave (3:37), Route 33, Azalea, and so on.
- The archive has been flat at 1451 cells and 42 maps for about 160 episodes.
  Nothing has been discovered in an honest episode since episode 234.
- 93% of honest episodes end by **stagnation truncation**, not by running out
  of the 49,152-step budget.
- Whiteouts per honest episode are rising (about 3, −37 reward). The lead
  Pokémon finishes around level 14–15, and the second-best party member is
  around level 3.

The question to answer: **once Falkner is down, what stops the agent leaving
the Violet area southward (Route 32 → Union Cave)?** Do not settle on a
cause until you have evidence for it. Candidates to confirm or rule out:

1. **World/gating.** Something physical or scripted blocks Route 32 south.
   Examples: an NPC or scene blocking the path, a ledge direction, or the
   Togepi-egg aide encounter at the Route 32 PC triggering something unexpected.
2. **Reward.** Nothing on Route 32 pays enough. It may be that `frontier` /
   `new_map` reward doesn't extend past known cells, or that seeded and
   assisted episodes never start there.
3. **Survival.** Route 32 trainers and wild battles white the party out to the
   Violet PC. A one-Pokémon level-14 party loops between Route 32 and the
   PC, which shows up as rising whiteouts.
4. **Stagnation watchdog.** The truncation fires before the long Route 32
   walk pays off, because progress there looks like stagnation to the clock.
5. **Observation / harness bug.** Something the policy sees on Route 32 is
   wrong. Examples: a mis-decoded walkable field, warps missing, or the mode
   misread.
6. **Distraction.** Ruins of Alph (gates 10:12/10:16, maps 3:22–3:28) or
   Sprout Tower soak up the post-Falkner time, because those maps still pay
   per-episode novelty.

## Materials

| What | Where |
|---|---|
| Probe start state (training's own) | `emu_files/states/post_falkner.state` |
| First honest Falkner win: start state, post-win state, video, actions | `Training Outputs/open_world_v3/Runs/checkpoints/1213-Beat-Falkner/` (`start.state`, `post_falkner.state`, `.mp4`, `actions.steps`) |
| Recorded episodes (per-step trajectory: step, x, y, map, bank, battle/player state, btn, reward) | `Training Outputs/open_world_v3/Runs/N_goals_0/ep_*/trajectory.csv` |
| Go-Explore frontier snapshots (filename encodes `b<bank>_m<map>`, `f<flag>`) | `Training Outputs/open_world_v3/goexplore_snapshots/` |
| Checkpoint data | `Training Outputs/open_world_v3/Checkpoints/info.pth` (read via `tools/training_health_report.py`) |

Recorded episodes that reach the gym and beyond. These are worth tracing first:

- `ep_394` enters Violet Gym (10:7) at step ~7371. It is on **Route 32 (10:1)**
  at step 7263 and **ends on Route 32 at (14,2)** at step 19,993. The episode
  ends well short of the budget, so this is probably stagnation truncation
  just south of Violet. **This is the primary case.**
- `ep_139` reaches the gym at ~44,960 and then ends in map 3:27 (Ruins of
  Alph area) at the 49,152-step budget. Good for candidate 6.
- `ep_340` and `ep_304` reach the gym late and end in Sprout Tower or the gym itself.

Check first whether flag 1213 was actually set in each of these episodes. Do
it by replaying, or from the state at that step. Don't assume it.

## Ground rules

- **A training run is live** (`main.py --use_config configs/stages/open_world.json`).
  Do not edit `PoliwhiRL/environment/*` or the config. Run the LLM agent at
  its default `nice`, and run only one LLM agent at a time (two give HTTP 507).
  `drive.py` is local and cheap, so prefer it.
- Every `drive.py` call is a fresh episode. Per-episode rewards pay again on
  every call; that is not a bug. To judge payouts, do the sequence in **one** call.
- Keep training's episode settings. Use `training_episode_end` lines in LLM
  runs to see where training would have truncated.
- LLM endpoint: use the Tailscale URL (`http://pingu.tailed5a29.ts.net:11434/v1`).
  macOS Python can't reach 192.168.0.x.
- Put all scratch output under `Training Outputs/drive/post_gym/` and
  `Training Outputs/llm_agent/post_gym_*`.

## Steps

### 1. Map the stuck point from data (no emulator)

For every recorded episode that reaches 10:7, take the trajectory after its
first gym entry and compute the following:
- time spent per map;
- the last 2,000 steps' map/cell histogram;
- the cell range on Route 32 (10:1), meaning the furthest-south y reached;
- battle and whiteout steps (`battlestate`, or map jumps back to 10:10, the Violet PC);
- the final position.

Write the script in the scratchpad. Produce one short table per episode.
Expected question it answers: *how far down Route 32 does the policy get,
and what happens there?*

Also list which Go-Explore snapshots lie past Violet. `b10_m1` is Route 32
(about 20 snapshots); check whether any `b10_m13` or `b3_m37` snapshots
exist. With none, seeding can never start an episode past the furthest point
the honest policy reached.

### 2. Inspect the post-Falkner start state

```bash
uv run python -m tools.llm_agent.drive emu_files/states/post_falkner.state noop --out pg_start --dir "Training Outputs/drive/post_gym"
```

Record the following:
- map and position;
- party (species, levels, HP, PP);
- balls and money;
- badges;
- flags 1213 / 45 / 31 / 66;
- the harness prompt state (routes, exits, people).

Look at `pg_start.png`. Compare against
`Runs/checkpoints/1213-Beat-Falkner/post_falkner.state`. Are they the same
kind of situation, for example in the gym versus outside, or full HP versus
fainted? The probe state is what scores 0%, so it needs to be representative.

### 3. Walk the intended route by hand, one leg at a time

Each leg is a `drive.py` call chained with `--out`. Before each call, write
down what the real game should do. After it, read every line.

1. Gym → Violet City → Route 32: `--to ROUTE_32` (use the map names the tool accepts).
2. Down Route 32 in small batches. At each trainer, each ledge and each
   NPC/sign, note:
   - mode;
   - on-screen text;
   - whether the route planner still finds a path (if it says "no walking
     route", check with people ignored, per the skill's pitfalls);
   - battle outcomes with this party.
3. Enter the Route 32 Pokémon Center (10:13) and trigger the Togepi-egg aide
   (flag 45). Confirm the flag sets and see what the scene does to control.
4. Continue south to Union Cave (3:37).

For each leg, record which reward sources paid and how much. The key
question is whether a policy that walks south gets *paid* for it compared
with staying in Violet or the Ruins. Compare against a call that wanders
Violet/Ruins for the same number of steps.

If a battle with this party loses, the leg has failed. Note the trainer, the
levels and whether a heal stop makes it winnable. That separates candidate 3
from candidate 1.

### 4. Reproduce the policy's actual stuck point

For `ep_394`: at its last Route 32 position (14,2), find the nearest
snapshot or state. If none exists, drive there from `pg_start`. From there:
- Look at the frame and the harness state. Is the walkable field consistent
  with the PNG? Are exits and warps listed? Probe the bytes with a scratchpad
  script using `Game.wram(...)` and `env._collision_field`, as described in
  the skill.
- Read what the **stagnation clock** observation does over a straight walk
  south. Does it reset on new cells on Route 32, or keep climbing? This is
  candidate 4.

### 5. Let the LLM agent attempt the post-gym ladder

An LLM run gives a second, independent view of where an agent gets stuck:

```bash
uv run python -m tools.llm_agent --from togepi_egg --state emu_files/states/post_falkner.state \
    --until union_cave --model Ornith-1.5-35B-A3B-oQ8e-mtp --no-thinking \
    --url http://pingu.tailed5a29.ts.net:11434/v1 --out "Training Outputs/llm_agent/post_gym_01"
uv run python -m tools.llm_agent.report "Training Outputs/llm_agent/post_gym_01"
```

Note the following:
- where it stalls (the harness's repeated-situation warnings);
- where `training_episode_end` fires, which shows where training would have
  cut the episode;
- blackouts;
- the reward audit from `report`.

If the LLM also stalls, check whether it stalls for the same reason as the
policy. If it isn't the same reason, it's a harness-display issue, not a
finding about the world. If the LLM sails through, the world is passable,
and the cause is in reward, survival or the clock.

### 6. Check the obvious reward lever

Using `info.pth` and the metrics JSON, check what fraction of honest
episodes that set flag 1213 ever enter 10:1, and how long they stay there
before truncation.

## Deliverable

Write `docs/post-gym-inspection-findings.md` containing:
- The stuck point(s): map, cell and situation, with the episode/step evidence
  from step 1 and a PNG from step 3 or 4.
- For each of candidates 1–6: **confirmed / ruled out / unclear**, with the
  evidence (trajectory numbers, `drive.py` output lines, RAM bytes).
- The primary cause and any compounding causes.
- Recommended fixes, ranked. Prefer general, area-invariant mechanisms over
  per-map tuning, and say what each fix would change in the next run's
  continuation probe.
- Framework bugs found: log them in `tools/llm_agent/bugs_found.md`. Any
  reusable check goes in `tools/llm_agent/scenarios.py` or
  `tools/training_health_report.py`.

Do not change training code, configs or the live run as part of this pass.
