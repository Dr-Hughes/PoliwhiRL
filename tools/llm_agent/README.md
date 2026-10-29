# llm_agent — a vision LLM playing Crystal on the training env

A diagnostic and demonstration-data tool, not used by training. A vision LLM
(any OpenAI-compatible endpoint; default: omlx at the Tailscale URL in
`llm.py`) plays the game through the **real** `PyBoyEnvironment`, so every
press uses the policy's action indices, frame timing and reward function.

```
uv run python -m tools.llm_agent \
    --model Ornith-1.5-35B-A3B-oQ8e-mtp --no-thinking \
    --fallback-model Qwen3.8-Flash-Next-oQ4e-mtp --out "Training Outputs/llm_agent/<run>"
tail -f "Training Outputs/llm_agent/<run>/live.log"      # progress bar + turn log
open "Training Outputs/llm_agent/<run>/live.html"        # live captioned screen
uv run python -m tools.llm_agent --from violet --state "<run>/states/10_violet_start.state"
uv run python -m tools.llm_agent --resume "<run>"          # continue from the latest autosave
uv run python -m tools.llm_agent.report "<run>"          # text summary + reward audit
uv run python -m tools.llm_agent.probe --state ... --models A:think,B:nothink   # A/B a frame
uv run python -m tools.llm_agent.replay "<run>"            # replay a run's buttons, check determinism
uv run python -m tools.llm_agent.scenarios               # regression scenarios, no LLM
uv run python -m tools.llm_agent.drive STATE up 'a*3' --to ROUTE_31   # play by hand
uv run python -m tools.llm_agent.demos --chain "<run>:410" "<run>_resumed" --rc   # training demos
```

## Web dashboard (`web.py`)

```
uv run python -m tools.llm_agent.web          # http://<host>:8765 on the LAN
```

Starts, stops, pauses and saves one agent run from a browser, shows the live
screen, goal ladder, recent turns, console and every save, and restarts from
any save (`Resume`: state + notes + notebook) or goal start/finish state.
The server talks to the agent through `control.json` (polled between turns)
and `status.json` (rewritten every turn) in the run dir; the agent runs in its
own session so it outlives a server restart. `--resume` also takes one save's
`states/autosave_<time>.json`. Safe alongside training: the agent only reads
the training config and runs at `nice 5`. No authentication.

On macOS, Homebrew Python cannot reach LAN hosts (Local Network privacy;
curl is exempt and still works), so point `--url` at the model host's
Tailscale name, e.g. `http://pingu.tailed5a29.ts.net:11434/v1` for
192.168.0.15.

Playing by hand (`drive.py`) is documented as a skill:
[`.claude/skills/live-env-testing/SKILL.md`](../../.claude/skills/live-env-testing/SKILL.md).

## Goal ladder (`goals.py`)

stairs → mom → outside → starter → route29 → cherrygrove → mr_pokemon (flag 30)
→ back_to_elm (flag 31) → catch (optional) → violet → violet_gym → falkner
(flag 1213). Success is a RAM predicate checked by the harness; the model
never declares it. Goal text is in-game knowledge, not coordinates. A failed
attempt restarts from the goal's start save-state (a fresh *seeded* episode);
an optional goal that fails is skipped from its start state.

## Each turn

One fresh request (no growing chat): the 4x screen with a cell grid and the
player boxed (walking only); a GAME STATE block from RAM/ROM (map, position, scene/battle,
party HP/levels, balls/money/badges, the four neighbours, doors/stairs from the
ROM warp table, map-edge connections with the cells you can actually cross, whether you are
standing on an exit and which way uses it, nearby objects, and a 9x9 ASCII
walkability map from the ROM collision decode). The screen's mode (walk /
text / menu / naming / battle) decides what is shown: off the overworld the
neighbours, doors and local map are dropped, and battles show both mons'
species, level and live HP (wBattleMon / wEnemyMon); in a battle the cursor of
the 2x2 battle menu (FIGHT PKMN over PACK RUN), decoded from the tilemap, with
the presses that reach each option; memory (the model's own notes, last 8 turns
with results, lessons from failed attempts); stall warnings. The model answers
with the `act` tool (1–4 buttons); execution stops early on map change,
scene/battle start or end, party change, a blocked move (unless the next
button is a: the press turned you to face it) or goal completion. Buttons are
the policy's action set minus select (a no-op in training).

Every press is reported by what it did: "moved you to (x,y)", "BLOCKED
(person)", "the cursor is now on NO", "the battle ended" - and a press that
changed nothing says "NOTHING HAPPENED (the state did not change)". Without
that the model could not tell a dead button from a live one, and in a battle
it pressed `down down a` 126 times trying to reach RUN (in that menu up/down
change row and left/right change column, and the cursor does not wrap).

Use one model (Flash, thinking on). `--fallback-model` exists, but omlx
holds only one of these models at a time: every switch reloads (Flash ~33 s,
Ornith ~13 s), so a fallback stays for ≥6 turns once triggered. Measured
2026-10-28: Ornith (thinking off) answers in 2.6 s vs Flash 5.1 s but presses
1.2 vs 2.5 buttons per turn, so per goal it is no faster, and it misread the
nickname menu (YES = "keep default") 3/3. Mellum can't read images; Ornith
with thinking on spends its whole budget on grid arithmetic. Run one agent at
a time (two at once → HTTP 507).

When a model misbehaves on a frame, A/B it with `probe.py` before changing
the prompt.

## Harness lessons (fixed)

- The state block must not mislead: on a doormat the engine reads the
  off-map cell as a wall, so it said `down=blocked` and both models walked
  along the mat (A/B 2/12 → 12/12 after reporting "standing on the exit").
- Keep what was read: with only `a: ok` in the turn history the model
  re-talked to Elm ~12 times. The exact screen text (tilemap 0xC4A0), a log
  of recent text and a repeat-talk warning fixed it. A press that changed
  nothing now says so ("NOTHING HAPPENED"); reporting every press as "ok"
  left the model unable to tell a dead button from a live one.
- Name objects by sprite (0x54 = Poké Ball): "person/object" hid the balls.
- Stop multi-button turns when a menu appears and send menus / the naming
  keyboard to the strongest model.
- A thinking model can spend its whole budget without answering: retry the
  turn without thinking.
- `script_active`, not `ui_byte`, says whether a scene/text is running.
- Do not override training's episode settings: an early version raised
  `episode_length` and disabled the stagnation watchdog, which pinned the
  policy's `stagnation_clock` observation at 0. The harness now keeps the
  training values, ignores `done`, and logs `training_episode_end` where
  training would have ended the episode.

`report.py` also audits the observation: which RAM features never changed
over the run's demos (the exact vectors a policy sees), NaN and range.

## Reward verification (`verify.py`)

Every step is diffed for events (first map entry, story checkpoint, any event
flag, battle start/won, whiteout, catch, Pokédex seen, level up, key item,
talk). Each kind has a rule naming the reward sources that should pay within
a window: `must` (a miss is a finding) or `may` (conditional; hit rate only).
Milestone-type payouts with no matching event are listed as unexplained. The
results are in `events.jsonl`, each goal write-up and `REPORT.md`.

Training-framework issues the runs surface go in `bugs_found.md`.

## Replay and scenarios (`replay.py`, `scenarios.py`)

The env is deterministic: a run's `steps.jsonl` and its goal start states
replay it exactly (`replay.py` reports any step whose position differs from
the recording). `scenarios.py` turns that into a regression suite that needs
no LLM: a recorded button trajectory (`scenarios/*.json`) plays from the
config's start state through the reward verifier and observation invariants,
and probes branch off it to check button behaviour (down moves the YES/NO
cursor, START then A leaves the naming keyboard, START opens the menu). It
runs in a few seconds and in pytest (`tests/test_llm_scenarios.py`). After a
reward or observation change, run it; when a new bug is found, add an
invariant or probe that would have caught it. Trajectories are buttons only,
so a change to frame timing or the action set can make them drift; re-cut
them from a fresh run (`replay.py <run> --buttons --until N`).

Every 15 minutes (`--autosave-minutes`) the run writes `states/autosave_<time>.state`
and `autosave.json` (goal, attempt, notes, notebook, goals reached).
`--resume <run>` restarts that goal from the autosave with the notes and
notebook restored, into `<run>_resumed_<time>/`. The resumed episode is
seeded: the reward function's per-episode state starts fresh.

## Demonstrations for training (`demos.py`)

`demos.py` rebuilds training demonstrations by replaying runs through the
current env, so they always match the current observation and reward code.
`--chain` replays a run and its resumed continuation as one honest episode (a
resumed run restarts the reward calculator, so on its own it is seeded). It
writes one `.npz` per successful goal leg, tagged with the next story
checkpoint it leads to; with `--rc` it also writes reverse-curriculum
pre-fire rings. A run that no longer replays is cut at the first divergence.
`configs/stages/open_world.json` loads them through `sil_demo_files` and
`rc_seed_manifests`.

## Outputs (`<run>/`)

`REPORT.md` (index, regenerated after each attempt), `goals/NN_key_aK.md`
(per attempt: outcome, reward by source, verification, turn log),
`videos/NN_key_aK_{SUCCESS,FAILED}.mp4` (every step, captioned) and
`videos/event_*.mp4` (clips around catches, wins, whiteouts, checkpoints),
`turns.jsonl`, `steps.jsonl`, `events.jsonl`, `states/`, and
`demos/*.npz` — training-format `(images, ram, actions, rewards)` tagged
`source=llm_demo` and `seeded`. Keep demo data out of honest metrics.
