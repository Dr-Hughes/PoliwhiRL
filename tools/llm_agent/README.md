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
# (a bare model name follows the agent's own rule: think off the walk screen only)
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
→ leave_mr_pokemon → cherrygrove_return → heal_cherrygrove (optional) → rival
→ back_to_elm (flag 31) → catch (optional) → violet → violet_gym → falkner
(flag 1213) → togepi_egg (optional, flag 45) → route32_pc (optional) →
union_cave → route33 → azalea → slowpoke_well (flag 43) → bugsy (flag 1214)
→ farfetchd (flag 41) → cut (flag 16) → goldenrod. The steps past Falkner
match the training run's `story_milestones`; a skipped map step counts as
done once a later milestone is reached. Success is a RAM predicate checked by the harness; the model
never declares it. Goal text is in-game knowledge, not coordinates. A failed
attempt restarts from the goal's start save-state (a fresh *seeded* episode);
an optional goal that fails is skipped from its start state.

The return trip is split into steps: leave the house, reach Cherrygrove,
heal at its Pokémon Center, fight the rival (Cherrygrove's scene byte
0xD98A is 1 from the egg until that battle ends), then Elm. Each step also
counts as done once any later one is, so a model that takes them out of
order moves straight on.

Healing: whenever the party is below half its HP or half its PP, or a
Pokémon has fainted, the HARNESS block suggests a Pokémon Center as a
subgoal (outside battles), with the route to one if it is on the map
(marked `<- HEAL HERE`), what to do inside, or the centers the run has
been in. Independently of that, `PARTY_RULES` (in `agent.py`) is injected
every turn, battles included, as mandatory requirements: catch unseen wild
Pokémon when holding balls, buy balls at the next Pokémon Mart when out,
and lead with and train up a party member more than 3 levels behind.

## Each turn

One fresh request (no growing chat): the screen at its native 160x144 with a
cell grid and the player boxed (walking only) — the model gets native pixels
because its visual tokens scale with them, and only the live view and the
MP4s are upscaled; a GAME STATE block from RAM/ROM (map, position, scene/battle,
party HP/levels, balls/money/badges, the four neighbours, doors/stairs from the
ROM warp table, map-edge connections with the cells you can actually cross, whether you are
standing on an exit and which way uses it, every person and item on the map
named by its sprite (with walking routes to the cell next to each of the
nearest six), and a 9x9 ASCII walkability map from the ROM collision decode). The screen's mode (walk /
text / menu / naming / battle) decides what is shown: off the overworld the
neighbours, doors and local map are dropped, and battles show both mons'
species, level and live HP (wBattleMon / wEnemyMon); in a battle the cursor of
the 2x2 battle menu (FIGHT PKMN over PACK RUN), decoded from the tilemap, with
the presses that reach each option; memory (the model's own notes, last 8 turns
with results, lessons from failed attempts); the turns used and the elapsed time;
a stall warning that names the situation the model keeps coming back to and the
buttons it keeps pressing.
with the `act` tool (up to `MAX_BUTTONS`, currently 10, buttons); execution
stops early on map change,
scene/battle start or end, party change, a blocked move (unless the next
button is a: the press turned you to face it) or goal completion. Buttons are
the policy's action set minus select (a no-op in training).

Every press is reported by what it did: "moved you to (x,y)", "BLOCKED
(person)", "the cursor is now on NO", "the battle ended" - and a press that
changed nothing says "NOTHING HAPPENED (the state did not change)". Without
that the model could not tell a dead button from a live one, and in a battle
it pressed `down down a` 126 times trying to reach RUN (in that menu up/down
change row and left/right change column, and the cursor does not wrap).

Use one model (Flash). Thinking follows the screen: on for battles, menus,
text and the naming keyboard, off while walking, where the state block
already spells the move out and thinking was most of the wait. `--no-thinking`
turns it off everywhere; `--fallback-model` exists, but omlx
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
- Detect a repeated *situation*, not a repeated cell. The old stall counter
  only moved while the player could walk and froze while a scene or a battle
  was running, so it never advanced through a 169-turn battle; the count is
  now how many times the whole state signature has come up, in every mode,
  and the warning names the situation and the buttons that led there. The
  count does not decay when the situation changes, because a loop can
  alternate between two of them (a battle menu and its item submenu).
  Replayed over the recorded runs this fires on 51 of gemma's 856 turns, all
  of them the lab loop in the goal that was never reached, and on 348 of
  ornith's 984: 343 of them the battle loop in `cherrygrove`, and 5
  elsewhere (a shuffle around the table before a conversation, and one
  dialogue line on screen six times).
- The situation count also drives the escalation: at 12 the turn runs at the
  hot temperature (0.9) and the fallback model is engaged. Under the old
  counter that was rare; replayed over the recorded runs it would now apply
  to 38 of gemma's turns and 268 of ornith's. The fallback is sticky (it
  stays for `--fallback-min-turns` turns once engaged), so this costs about
  two model reloads per stuck stretch, not one per turn.
- Name people by sprite and list the whole map, not the 11x9 window round
  the player. The engine only loads an object while it is near the camera,
  so the objects are read from the map's object table (wMapObjects, hidden
  ones dropped by their event flag) as well as the loaded structs. From the
  door of Elm's lab gemma saw one "person", the aide, took her for Elm and
  spent 504 turns on the goal; Elm, out of view at the back, was never
  listed. Walking routes now go to people as well as exits.
- State the doormat as a fact, not an order: "press DOWN to leave" sent
  gemma back out of the lab on 38 of 49 arrivals, 18 of them with a plan to
  walk to Elm. The state now also says when you have just come in.
- Someone who says the same thing every time has nothing more to give: the
  repeat-talk warning used to say "do what it told you", which assumes the
  person told you something.
- A lost model writes down why it is lost. Ornith wrote "Verified
  CHERRYGROVE origin is (27,15)" eight times over five failed attempts, and
  each attempt read it back, plus its failed plan quoted as a "lesson". Now
  every notebook fact is settled when its attempt ends (`worked` /
  `unconfirmed`, shown as a tag; unconfirmed facts are dropped first), a
  restatement replaces the fact it restates, claims of certainty are
  stripped, and a failed attempt's plan is shown as a plan not to repeat,
  with the scratchpad cleared.
- The model's history is written by the harness, not the model
  (`history.py`). Each prompt starts with a summary of the attempt (per map:
  turns, cells stood on, exits used, who said what, battles, blackouts, and
  the situations it keeps returning to), rebuilt every 25 turns, then a
  compact log of the last 25-49 turns, which only grows at the end between
  rebuilds so a prefix-caching server need not re-read it. The model's own
  plans are shown for the last 5 turns only. The current state and the
  HARNESS messages come last.
- The model commits to a route: `act` has a required `following` field
  (a WALKING ROUTE destination or "none"). The harness checks the first
  button against that route's first move, says so next turn if it did not
  match, and reports the route's step count before and after ("71 -> 73,
  FURTHER AWAY"). Routes to places the goal names are marked. Ornith pressed
  the route's first move on 42% of its Route 29 turns and never reached
  Cherrygrove in five attempts.
- `script_active`, not `ui_byte`, says whether a scene/text is running.
- Do not override training's episode settings: an early version raised
  `episode_length` and disabled the stagnation watchdog, which pinned the
  policy's `stagnation_clock` observation at 0. The harness now keeps the
  training values, ignores `done`, and logs `training_episode_end` where
  training would have ended the episode.

`report.py` also audits the observation: which RAM features never changed
over the run's demos (the exact vectors a policy sees), NaN and range.

## What a run writes (`recorder.py`)

Everything is under the run directory; the full list is in `recorder.py`'s
docstring. For a long run the ones to know are:

- `turns.jsonl`: every model call: the exact prompt, the reply, the state, the
  route it named and whether it pressed that route's first move, latency,
  tokens, transport retries, the env steps it covered and the save-state for
  that turn (every 10th turn, in `states/turns/`).
- `steps.jsonl`: every env step: button, place, mode, reward by source, and
  the env's own view (`env_step`, `done`, `truncated`, `cause`, `frames`,
  `episode_reward`). To get back to any moment: load the nearest
  `states/turns/` state with `drive.py` and replay the buttons from here.
- `states/events/`: a save-state at every first map entry, story checkpoint,
  event flag, battle start, level-up, catch, key item, whiteout, and where
  training would have ended the episode.
- `health.jsonl` (every 25 turns): turns per minute, latency, prompt,
  completion and reasoning tokens, retries, route compliance, the worst
  stall, memory and free disk. Prompt vs completion tokens is what tells a
  slow prefill from slow decoding, so read both before tuning anything.
- `run_info.json` + `git.diff`: commit, branch, uncommitted changes (new
  Python files included), resolved env config, versions, command line.
- On Ctrl-C or a crash: a save-state, `autosave.json` (so `--resume` works),
  the attempt's demo so far as `demos/*_partial.npz`, and for a crash
  `crash.json` with the traceback.

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
