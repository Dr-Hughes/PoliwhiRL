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

PoliwhiRL trains a Proximal Policy Optimisation (PPO) agent to play Pokemon Crystal via the [PyBoy](https://github.com/Baekalfen/PyBoy) emulator. The policy observes both the screen image and a normalised RAM vector (position, party state, exploration summary, curated story-flag bits) and selects one of nine discrete button presses per step.

Training supports both a staged curriculum (leave the house → catch the starter → return the Mystery Egg → free play) and a from-scratch open-world run. Reward combines sparse directed milestones with exploration and Go-Explore frontier seeding. Training always uses the vectorised agent; `num_envs: 1` is the single-worker case.

Reward design and curriculum structure are under active iteration — see the config files under `configs/` for what's currently configured, rather than relying on this README for exact mechanics.

## Key Features

- **Transformer-XL architecture** — dual-stream CNN and RAM encoders fused before a configurable Transformer-XL trunk (two layers by default) with per-layer cached memory
- **Progress-signal policy** — pokédex, party, exploration and curated story-flag state injected into the observation vector
- **Vectorised multi-process training** — independent subprocesses spawned via `multiprocessing.get_context("spawn")`, no shared memory or parameter averaging
- **Curriculum learning** — each stage loads the previous stage's best checkpoint and continues training
- **Per-state metrics and best-so-far checkpointing** — rolling window tracks peak performance separately from latest weights
- **Reward normalisation** — running statistics keep critic targets stable across curriculum stages

## Quick Start

```bash
# Canonical open-world run — trains from scratch and auto-resumes its own output
python -u main.py --use_config configs/stages/open_world.json

# Stage 1 — leave the house (reach New Bark Town outdoors); from scratch
python main.py --use_config configs/stages/first.json

# Stages 2-6 — each loads the prior stage's best checkpoint
python main.py --use_config configs/stages/second.json   # catch the starter
python main.py --use_config configs/stages/third.json    # reach Route 29
python main.py --use_config configs/stages/fourth.json   # reach Cherrygrove City
python main.py --use_config configs/stages/fifth.json    # reach Mr. Pokémon's house
python main.py --use_config configs/stages/sixth.json    # return the Mystery Egg to Elm

# Curriculum free play — open-ended continuation from stage 6
python main.py --use_config configs/stages/freeplay.json

# Inference — stochastic playthrough of a trained model (reports goal-success rate)
python main.py --use_config configs/inference.json

# Override any config key via CLI
python main.py --use_config configs/stages/first.json --num_rollouts 300 --device cuda
```

## Model Architecture

```
Screen (C, H, W) -> GameBoyCNN (Conv-GroupNorm) -> (B*T, d_model)  \
                                                                   -> concat -> Linear -> (B*T, d_model)
RAM   (RAM_OBS_DIM) -> RAMEncoder (MLP)        -> (B*T, d_ram)     /
                                                                 |
                                        TransformerXLBlock stack (MHA + learned age embedding, GELU FFN)
                                                                 |
                                        last token (B, d_model)
                                                                 |
                                              /-> fc_actor -> softmax -> (B, action_size)
                                              \-> fc_critic -> (B, 1)
```

Memory tensors are caller-managed: `init_mems(batch_size, device)` returns per-layer `(B, mem_len, d_model)` zero tensors. Each forward pass receives and returns updated (detached) mems; the trainer feeds one frame per step, so the memory holds the last `mem_len` genuine env steps (the model's within-episode context). Recency information comes from each block's learned age embedding rather than an absolute positional encoding. On episode termination, mems are zeroed. Exact sizes (`d_model`, `d_ram`, `mem_len`, layer count) are config-driven — see `configs/default_configs/core_settings.json` and `PoliwhiRL/models/PPO/PPOTransformer.py`.

## Configuration

JSON-based config system with inheritance:

- `configs/default_configs/*.json` — globally merged defaults (model, device, PPO hyperparameters, episode settings, rewards, ROM paths, outputs)
- `configs/curriculum_base.json` — shared training defaults for stage configs
- `configs/stages/open_world.json` — canonical from-scratch open-world training
- `configs/stages/{first,second,third,fourth,fifth,sixth,freeplay}.json` — directed curriculum and its free-play continuation
- `configs/{inference,debug_eval,explore,random_walker,evaluate_reward_system}.json` — supported evaluation and data-collection modes

The `extends` key supports chaining. Config values merge parent-then-child; CLI flags override everything. Actual defaults (`num_rollouts`, `num_envs`, `episode_length`, PPO hyperparameters, etc.) live in `configs/default_configs/*.json` and change as the project iterates — check those files rather than this README for current values.

## Curriculum

A staged curriculum under `configs/stages/*.json`, each extending `curriculum_base.json`. Each stage loads the previous stage's best checkpoint and continues training from there. Map IDs are verified against live RAM; see `RAM_MAPPING.md`.

Reward and termination behavior per stage (whether a milestone ends the episode, whether it's purely open-ended, etc.) is configured per stage and evolves — read the individual stage config for a stage's current goal and behavior rather than relying on a table here.

### Goal types

Configured per stage as a list under `"goals"`. Supported: `pokedex` (`kind: seen|owned`, `threshold`), `flag` (`flag_num`), `map` (reach a specific `map_bank`/`map_num`), `maps_visited` (`threshold` unique maps this episode), `level`, `xp`. See `PoliwhiRL/environment/goals.py`. A stage can also run with an empty goal list, in which case it trains on general-purpose reward only (see below).

## Reward System

Per-step reward combines sparse directed milestones (story checkpoints, pokédex, key items and configured map goals) with exploration/discovery signals and battle outcome. Durable story flags use the separately reported `checkpoint` source and `checkpoint_progress_reward`; generic flags remain under `flag`. Exact weights and mechanics are config-driven and iterate frequently — see `configs/default_configs/reward_settings.json`, `configs/curriculum_base.json`, and `PoliwhiRL/environment/rewards.py` for the current, authoritative behaviour.

The first honest reach of each durable story checkpoint is replayed from the canonical start. Its action prefix and labelled PNG sequence are written under `Runs/checkpoints/<flag>-<title>/`, while replay status is persisted in `Checkpoints/info.pth`. Go-Explore frontier manifests are persisted in the same checkpoint and restored when their save-state files remain available.

A **RAM-conditional action mask** (`environment/action_mask.py`, on by default) blocks directional input during dialog and blocks `start`/`select` while walking unless a stage opts in via `allow_menus_walking`.

## File Structure

```
main.py                                    # Entry point: parse args, merge configs, dispatch by model type
PoliwhiRL/
├── PPO.py                                 # Training entry: env probe, agent dispatch
├── agents/PPO/
│   ├── vec_ppo_agent.py                   # Vectorised PPO agent (including one worker)
│   └── _minibatch.py                      # Shared minibatch iterator
├── environment/
│   ├── gym_env.py                         # PyBoy env, dict observation, RAM vector
│   ├── vec_env.py                         # Multiprocessing wrapper, replay pool
│   ├── rewards.py                         # Reward calculator
│   ├── goals.py                           # Goal types + termination predicate
│   ├── action_mask.py                     # RAM-conditional action mask
│   ├── visit_archive.py                   # Frontier-novelty visit counts
│   └── RAM.py                             # RAM address book
├── checkpoints.py                         # Durable checkpoint flags, titles and folder slugs
├── models/
│   ├── CNN/GameBoy.py                     # GameBoyBlock, CNN building blocks
│   ├── PPO/PPOTransformer.py              # Full model: CNN + RAM + Transformer-XL
│   └── PPO/ppo_model_implementation.py    # PPO losses, GAE, entropy schedule
├── replay/                                # Rollout storage buffers
├── explorer/                              # Manual/random data collection
├── reward_evaluation/                     # Predefined-action reward evaluation
├── evaluator/                             # Stochastic inference and debug runners
└── utils/
    ├── running_stats.py                   # RunningMeanStd, RewardScaler
    └── visuals.py                         # Plotting, step recording
configs/
├── default_configs/                       # Globally merged defaults
├── curriculum_base.json                   # Shared stage defaults
├── stages/open_world.json                 # Canonical open-world run
├── stages/{first..sixth,freeplay}.json    # Directed curriculum
└── {inference,debug_eval,explore,random_walker,evaluate_reward_system}.json
.opencode/skills/training-run-analysis/     # Repeatable checkpoint/run analysis workflow
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

Tests cover model init, losses, GAE, buffers, config inheritance, vectorised environments, action masking, reward + goal calculation, event-flag numbers, and running statistics. Pure NumPy/PyTorch tests run instantly; emulator tests spin up real PyBoy subprocesses.

## Documentation

- **[AGENTS.md](./AGENTS.md)** — project goal and a map of the codebase
- **[RAM_MAPPING.md](./RAM_MAPPING.md)** — verified RAM addresses and the curated event-flag table
- **[training-run-analysis](./.opencode/skills/training-run-analysis/SKILL.md)** — repeatable honest/seeded, checkpoint, reward and RAM-correlated run assessment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Run pre-commit hooks: `pre-commit run --all-files`
5. Submit a pull request

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
