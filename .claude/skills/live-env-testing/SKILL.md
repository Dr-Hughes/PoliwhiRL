---
name: live-env-testing
description: Play the PoliwhiRL training env by hand, from a save-state, to check that buttons, screens, observation features and rewards behave as intended. Use when verifying a change to stepping, the action set, the observation, rewards or the LLM harness; when asked to "manually test" something in the game (healing, a battle, a route, a menu); or when a run or the LLM agent gets stuck somewhere and you need to see why.
compatibility: Requires the ROM at emu_files/, uv, and save-states (Training Outputs/llm_agent/*/states, Manual Investigation States/, Go-Explore snapshots).
---

# Live env testing

Drive the real `PyBoyEnvironment` (the training env, settle stepping, the
real reward function) one command at a time, read what comes back, and check
it against what the game should do. Do not reimplement the env in a script:
what you are testing is the code path training uses.

## The tool

```bash
uv run python -m tools.llm_agent.drive STATE [BUTTON ...] [--to MAP] [--out NAME]
```

- `STATE`: a `.state` path, or a `NAME` saved by an earlier call (states
  live in `--dir`, default `Training Outputs/drive/`).
- `BUTTON`: `up down left right a b start`, `noop`, and `left*7` = seven
  presses. Quote `'left*7'` in zsh.
- `--to MAP`: walk the shortest route to that map (the harness's ROM
  collision routes, re-planned every press). It stops when the screen
  leaves walk mode, for example at a battle, a trainer's challenge or a
  scene.
- `--out NAME`: saves `NAME.state` and `NAME.png`. Chain calls by passing
  `NAME` as the next `STATE`.

Each press prints one line:
- map and position;
- mode: walk / text / menu / naming / battle;
- the on-screen text, read from the tilemap rather than by OCR;
- both battle mons, once both are out;
- the reward sources it paid;
- the frames the step took.

At the end it prints the full harness prompt state (routes, neighbours, local
map, party) and saves the state.

**Every call starts a fresh episode**, like a Go-Explore restart. Per-episode
rewards (`new_map`, `first_trainer_battle`, …) pay again on each call. Do not
read that as a re-pay bug; to check per-episode payouts, do it in one call.

Find starting states in:
- `Training Outputs/llm_agent/<run>/states/NN_<goal>_start.state`: honest path milestones;
- `Training Outputs/llm_agent/<run>/states/autosave_*.state`;
- `Manual Investigation States/`;
- `Training Outputs/<run>/goexplore_snapshots/`.

To reach a state the LLM reached, replay it:
`uv run python -m tools.llm_agent.replay <run> --until N --save-state X.state`.

## The loop

1. **State the expectation before you press.** For example: "A on the nurse
   → dialogue → YES → one long step → full HP".
2. **Press in small batches**, 1 to 6 buttons, and read every line. Look at
   the PNG (Read tool) whenever the text and the RAM state disagree, or the
   screen is new to you.
3. **Check three things on every step:**
   - **Input:** did the button do what it does in the real game? One text box
     per A, one tile per direction, one cursor row per press. Is a press ever
     dropped, or applied twice?
   - **Observation:** does the mode, text, battle and party data match the
     image?
   - **Reward:** did the right source pay, at the right time, and only once?
4. **When something is off, find the engine's ground truth** before
   concluding anything:
   - RAM through `Game.wram(addr)` (WRAM bank 1 for 0xD000–0xDFFF);
   - ROM collision through `env._collision_field._value(...)`;
   - pret symbols, in `docs/reference/ram-mapping.md`.
   Write a small probe script in the scratchpad that loads the saved state
   and prints exactly the bytes in question.
5. **Record the finding and fix it at the right layer:**
   - **Env / reward / observation** (`PoliwhiRL/`): a training-framework bug.
     Log it in `tools/llm_agent/bugs_found.md`.
   - **Harness display** (`tools/llm_agent/`): what the LLM is shown.
   - **Game behaviour that traps a policy:** prefer a general fix, such as
     forcing an options bit, over an area-specific one.
6. **Lock it in:**
   - add a probe or an invariant to `tools/llm_agent/scenarios.py`, or a
     trajectory cut from a run with `replay --buttons`;
   - re-run `uv run python -m tools.llm_agent.scenarios` and `pytest tests -q`.

## Known game mechanics (verified live, 2026-10-28)

- **Text:** one A per text box. Instant text is forced by `wOptions`.
  Directions do nothing at a text prompt.
- **Battle menu:** `▶FIGHT PKMN / PACK RUN`.
  - RUN: from the menu, `down right a`, then `a` for "Got away safely".
  - In the move list, `b` returns to the menu.
- **Trainers:** a trainer who spots you walks up and talks. The script byte
  stays 0 during that speech, so the harness reports it as `text` from the
  on-screen text, not from `script_active`.
- **"Will … change POKéMON?":** with SHIFT battle style, YES then A loops on
  "is already out" for as long as A is pressed. B on the "Which PKMN?" screen
  escapes. The env forces SET style (`wOptions` bit 6), so the prompt never
  appears in training.
- **Ledges:** you walk onto a ledge cell from any side except the drop side.
  Standing on it and pressing its direction hops two cells, over the next
  cell. Route 30's gap is passed over the ledge at (4,24), beside Youngster
  Mikey.
- **Pokémon Centers:** the nurse is behind the counter. Stand below it,
  facing up, then press `a` through the dialogue (about 7 presses with YES).
  The heal is one step of about 320 frames.
- **The Dude's catching demo** (Route 29, after flag 31) plays itself on
  automatic input. It is one step under settle stepping, and `auto_frames`
  reports it.
- **Wild grass:** wild battles interrupt `--to` walks. Run, then continue.

## Pitfalls

- A route that says "no walking route" may be blocked by an NPC:
  - **Wandering NPCs** move between steps; re-plan.
  - **Stationary ones** (defeated trainers) stay. Check the route with people
    ignored before blaming the collision graph.
- After a framework change, recorded LLM runs can stop replaying
  (`replay` reports the divergence). Re-cut the trajectories rather than
  forcing old buttons.
- Never edit `PoliwhiRL/environment/*` while a training run or an LLM run is
  using it, unless you mean the next run to pick the change up. Running
  processes keep their imported code.
