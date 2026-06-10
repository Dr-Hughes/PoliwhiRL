# PoliwhiRL

<p align="left">
  <a href="https://results.pre-commit.ci/latest/github/AoifeHughes/PoliwhiRL/main">
    <img src="https://results.pre-commit.ci/badge/github/AoifeHughes/PoliwhiRL/main.svg" alt="pre-commit.ci status">
  </a>
  <a href="https://github.com/AoifeHughes/PoliwhiRL/actions/workflows/run_tests.yml">
    <img src="https://github.com/AoifeHughes/PoliwhiRL/actions/workflows/run_tests.yml/badge.svg" alt="Tests">
  </a>
</p>
<img src="./logo.png" alt="PoliwhiRL" width="50%">

## Legal Disclaimer

This project is not affiliated with, endorsed by, or sponsored by Nintendo, Game Freak, or The Pokemon Company. All Pokemon games and characters are trademarks of Nintendo, Game Freak, and The Pokemon Company. This project is intended for educational and research purposes only.

All usage of The Pokemon Company International's games is done with the understanding that software is legally obtained and that the user has the right to use it. This project does not provide any ROMs or copyrighted materials.

## Overview

PoliwhiRL trains a Proximal Policy Optimisation (PPO) agent to play Pokemon Crystal via the [PyBoy](https://github.com/Baekalfen/PyBoy) emulator. The policy observes both the screen image and a normalised RAM vector (position, party state, active goal target, exploration summary, curated story-flag bits) and selects one of nine discrete button presses per step.

Training uses a five-stage directed curriculum (leave the house → catch the starter → Route 29 → Cherrygrove City → Mr. Pokémon's house) followed by an open-ended free-play stage, chained via action replay so each stage builds on the previous one. Reward combines sparse directed milestones with a stationary per-episode exploration drive (two-stream normalised so milestones dominate churn). Both single-environment and vectorised multi-process training modes are supported.

## Key Features

- **Transformer-XL architecture** — dual-stream CNN and RAM encoders fused before a four-layer Transformer-XL trunk with per-layer cached memory
- **Progress-signal policy** — pokédex, party, exploration and curated story-flag state injected into the observation vector
- **Vectorised multi-process training** — independent subprocesses spawned via `multiprocessing.get_context("spawn")`, no shared memory or parameter averaging
- **Curriculum learning** — five directed stages (leave house → starter → Route 29 → Cherrygrove → Mr. Pokémon's) + free play, each a terminate-on-milestone `map`/`pokedex` goal
- **Action replay** — previous stage's best action sequences replayed at episode start, advancing the reward curriculum naturally without manual goal curation
- **Save-state pool** — multiple starting save-states cycled across workers for curriculum mixing
- **Per-state metrics and best-so-far checkpointing** — rolling 100-episode window tracks peak performance separately from latest weights
- **Reward normalisation** — streaming Welford statistics on discounted returns keep critic targets stable across curriculum stages

## Quick Start

```bash
# Stage 1 — leave the house (reach New Bark Town outdoors); from scratch
python main.py --use_config configs/stages/first.json

# Stages 2-5 — each loads the prior stage's best/ and replays its actions
python main.py --use_config configs/stages/second.json   # catch the starter
python main.py --use_config configs/stages/third.json    # reach Route 29
python main.py --use_config configs/stages/fourth.json   # reach Cherrygrove City
python main.py --use_config configs/stages/fifth.json    # reach Mr. Pokémon's house

# Free play — open-ended exploration (loads stage 5; no terminal goal)
python main.py --use_config configs/stages/freeplay.json

# Inference — stochastic playthrough of a trained model (reports goal-success rate)
python main.py --use_config configs/inference.json

# Override any config key via CLI
python main.py --use_config configs/stages/first.json --num_rollouts 300 --device cuda
```

## Model Architecture

```
Screen (C, H, W) -> GameBoyCNN (Conv-GroupNorm) -> (B*T, d_model=128)  \
                                                                      -> concat -> Linear -> (B*T, 128)
RAM   (ram_dim=72) -> RAMEncoder (MLP)         -> (B*T, d_ram=64)     /
                                                                    |
                                              PositionalEncoding (sinusoidal, max_len=1000)
                                                                    |
                                              4 x TransformerXLBlock (MHA, d_model=128, heads=4, FFN x4 GELU)
                                                                    |
                                              last token (B, 128)
                                                                    |
                                                    /-> fc_actor -> softmax -> (B, action_size)
                                                    \-> fc_critic -> (B, 1)
```

Memory tensors are caller-managed: `init_mems(batch_size, device)` returns per-layer `(B, mem_len=64, d_model)` zero tensors. Each forward pass receives and returns updated (detached) mems. On episode termination, mems are zeroed and sequence buffers are refilled.

## Configuration

JSON-based config system with inheritance:

- `configs/default_configs/*.json` — globally merged defaults (model, device, PPO hyperparameters, episode settings, rewards, ROM paths, outputs)
- `configs/curriculum_base.json` — shared training defaults for stage configs
- `configs/stages/*.json` — per-stage overrides, each extending `curriculum_base.json` via `"extends"`

The `extends` key supports chaining. Config values merge parent-then-child; CLI flags override everything.

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `num_rollouts` | 12 | Training budget (outer-loop iterations) |
| `num_envs` | 8 | Parallel environments (>1 uses vectorised agent) |
| `episode_length` | 50 | Step cap per episode |
| `ppo_update_frequency` | 128 | Transitions per env per PPO update |
| `ppo_learning_rate` | 3e-4 | Peak learning rate (cosine schedule) |
| `ppo_epochs` | 3 | PPO update passes per batch |
| `ppo_target_kl` | 0.01 | KL early-stop threshold |

## Curriculum

Five directed stages + free play. Each loads the previous stage's best checkpoint and replays its captured action sequences, so the agent starts each stage near where the last one finished. A stage's goal fires *during the training portion* (after the replay prefix), so a milestone the replay already walked through is not re-paid. Map IDs are verified against live RAM and `map` goals fire on entering the target map at **any** x/y.

| Stage | Goal | `terminate_on_goal_complete` | Episode length | Rollouts | Envs |
|---|---|---|---|---|---|
| 1 (`first.json`) | `map (24,4)` — leave the house (reach New Bark outdoors) | yes | 256 | 300 | 16 |
| 2 (`second.json`) | `pokedex_owned ≥ 1` — receive the starter from Elm | no | 512 | 600 | 16 |
| 3 (`third.json`) | `map (24,3)` — reach Route 29 | no | 768 | 700 | 16 |
| 4 (`fourth.json`) | `map (26,3)` — reach Cherrygrove City | no | 1024 | 900 | 16 |
| 5 (`fifth.json`) | `map (26,10)` — reach Mr. Pokémon's house | no | 2048 | 1000 | 16 |
| 5b (`freeplay.json`) | none — open-ended self-discovery | no | 3072 | 1000 | 16 |

Stages 2–5 do **not** terminate on the goal: the milestone is an *additive* per-episode bonus (`reach goal → +reward AND keep exploring beyond it`). Terminating would make the milestone a trade-off against the rest of the episode's renewable exploration reward — which led the agent to learn then abandon the goal (see `AGENTS.md` §10a). Non-termination makes grabbing the milestone strictly dominant and lets the agent explore past it (curriculum momentum). Stage 1 keeps termination (short, proven, cleaner replay demos). `best/` is selected on goal-success rate (directed) / exploration (free play), and only goal-reaching trajectories are captured into the replay pool.

> The starter is detected via `pokedex_owned ≥ 1`, **not** an event flag — `EVENT_GOT_A_POKEMON_FROM_ELM` (flag 26) is set then cleared by the script, so it can't be used as a terminal (see `gym_env.py`).

### Goal types

Configured per stage as a list under `"goals"`. Supported: `pokedex` (`kind: seen|owned`, `threshold`), `flag` (`flag_num`), `map` (reach a specific `map_bank`/`map_num`), `maps_visited` (`threshold` unique maps this episode), `level`, `xp`. See `PoliwhiRL/environment/goals.py`.

## Reward System

Per-step reward (clipped to ±1000), driven by config keys in `configs/curriculum_base.json` / `configs/default_configs/reward_settings.json`:

Reward is split into two streams, each normalised independently then recombined (`1.0·extrinsic + 0.3·intrinsic`) so sparse milestones dominate dense exploration churn:

```
extrinsic (directed milestones):
  + 500 · flag_fires                             # binary story milestone (0→1 this episode)
  + 150 · Δpokedex_owned                         # binary per species caught/received
  +  10 · Δpokedex_seen (first sighting only)    # one-shot per fresh species seen
  +   5 · Δkey_items_count                       # picking up Pokéballs etc.
  + 250 · map_goal_reached                       # reaching a configured `map` goal
  − 100 · whiteout                                # hard fail (party HP > 0 → 0)
intrinsic (stationary exploration + battle outcome):
  +   3 / (visits + 1) · new_cell · (1 − script_active)   # PER-EPISODE frontier novelty
  +  50 / (global_entries + 1) · new_map         # decaying first-discovery (not pumped by replay)
  + ( 3 · first_battle_per_map + 8 · first_win_per_map ) · decay   # win, not damage
  +   0 · Δenemy_hp                              # raw damage OFF by default (was the farm vector)
  +  10 · Δparty_total_level                     # minor levelling signal
  (battle entry+win+damage clamped to battle_reward_episode_cap = 30 / episode)
```

Frontier novelty is **per-episode** (stationary — it does not drain across a run, so it can't trigger the mid-stage collapse the old global archive caused). The `new_map` first-discovery bonus decays by how often a map has been entered run-wide (so map-bouncing stops paying) and is not written during action replay. Battle reward rewards *winning*, not raw damage, capped per episode. No step penalty by design — see `PoliwhiRL/environment/rewards.py` and `AGENTS.md` §4/§10a for the rationale and the failure modes this fixed.

A **RAM-conditional action mask** (`environment/action_mask.py`, on by default) blocks directional input during dialog and blocks `start`/`select` while walking unless a stage opts in via `allow_menus_walking`.

## File Structure

```
main.py                                    # Entry point: parse args, merge configs, dispatch by model type
PoliwhiRL/
├── PPO.py                                 # Training entry: env probe, agent dispatch
├── agents/PPO/
│   ├── ppo_agent.py                       # Single-env PPO agent
│   ├── vec_ppo_agent.py                   # Vectorised multi-process PPO agent
│   └── _minibatch.py                      # Shared minibatch iterator
├── environment/
│   ├── gym_env.py                         # PyBoy env, dict observation, RAM vector
│   ├── vec_env.py                         # Multiprocessing wrapper, replay pool
│   ├── rewards.py                         # Reward calculator
│   ├── goals.py                           # Goal types + termination predicate
│   ├── action_mask.py                     # RAM-conditional action mask
│   ├── visit_archive.py                   # Frontier-novelty visit counts
│   └── RAM.py                             # RAM address book
├── models/
│   ├── CNN/GameBoy.py                     # GameBoyBlock, CNN building blocks
│   ├── PPO/PPOTransformer.py              # Full model: CNN + RAM + Transformer-XL
│   └── PPO/ppo_model_implementation.py    # PPO losses, GAE, entropy schedule
├── replay/                                # Rollout buffers (single and vec)
├── explorer/                              # Manual/random data collection
├── reward_evaluation/                     # Predefined-action reward evaluation
├── evaluator/                             # Inference-only greedy runner
└── utils/
    ├── running_stats.py                   # RunningMeanStd, RewardScaler
    └── visuals.py                         # Plotting, step recording
configs/
├── default_configs/                       # Globally merged defaults
├── curriculum_base.json                   # Shared stage defaults
├── stages/{first,second,third}.json       # Curriculum stages
└── {inference, debug_eval*, random_walker, evaluate_reward_system}.json
tests/                                     # Unit + emulator tests
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch (with MPS or CUDA support), PyBoy, NumPy, Matplotlib, tqdm.

## Testing

```bash
pytest tests/ -v
```

Tests cover model init, losses, GAE, buffers, config inheritance, vectorised environments, action replay/masking, reward + goal calculation, event-flag numbers, and running statistics. Pure NumPy/PyTorch tests run instantly; emulator tests spin up real PyBoy subprocesses.

## Documentation

- **[AGENTS.md](./AGENTS.md)** — technical reference: step semantics, reward rules, goal types, architecture details, configuration, invariants
- **[RAM_MAPPING.md](./RAM_MAPPING.md)** — verified RAM addresses and the curated event-flag table

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Run pre-commit hooks: `pre-commit run --all-files`
5. Submit a pull request

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
