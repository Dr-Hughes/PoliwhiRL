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

Training uses a curriculum of sequential location goals across five stages, chained via action replay so each stage builds on the previous one. Both single-environment and vectorised multi-process training modes are supported.

## Key Features

- **Transformer-XL architecture** — dual-stream CNN and RAM encoders fused before a four-layer Transformer-XL trunk with per-layer cached memory
- **Goal-conditioned policy** — active target coordinates injected into the observation vector so the model can navigate to arbitrary waypoints
- **Vectorised multi-process training** — independent subprocesses spawned via `multiprocessing.get_context("spawn")`, no shared memory or parameter averaging
- **Curriculum learning** — five-stage progression from simple in-house navigation to multi-map exploration
- **Action replay** — previous stage's best action sequences replayed at episode start, advancing the reward curriculum naturally without manual goal curation
- **Save-state pool** — multiple starting save-states cycled across workers for curriculum mixing
- **Per-state metrics and best-so-far checkpointing** — rolling 100-episode window tracks peak performance separately from latest weights
- **Reward normalisation** — streaming Welford statistics on discounted returns keep critic targets stable across curriculum stages

## Quick Start

```bash
# Stage 1 — from-scratch training (downstairs to mother)
python main.py --use_config configs/stages/first.json

# Stage 5 — vectorised mode (16 envs, 7 location goals + pokedex)
python main.py --use_config configs/stages/fifth.json

# Inference — greedy playthrough of a trained model
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
                                              4 x TransformerXLBlock (MHA, d_model=128, heads=8, FFN x4 GELU)
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

| Stage | Goals | Episode Length | Rollouts | Mode | Status |
|---|---|---|---|---|---|
| 1 | 2 location | 40 | 150 | single | Solved (97% success) |
| 2 | 4 location | 256 | 250 | single | Solved (100% success) |
| 3 | 7 location + pokedex owned:1 | 1024 | 1000 | single | Collapsed mid-training |
| 4 | 7 location + owned:1 + seen:3 | 2048 | 1000 | vec (16 envs) | Improving |
| 5 | 7 location + owned:1 + seen:4 | 4096 | 1000 | vec (16 envs) | Pending |

Each stage loads the previous stage's best checkpoint and replays its action sequences. See [`model_status.md`](./model_status.md) for detailed evaluation and recommended fixes.

## Reward System

Per-step reward formula:

```
r = goal_hit_reward (100 + sequence_bonus 50)
  + pokedex_seen_reward (50) on new species
  + pokedex_owned_reward (150) on new owned
  + all_goals_bonus + early_completion_bonus (150) on final goal
  + party_level_reward x delta (10 per level)
  + party_exp_reward x delta (0.01 per EXP point)
  + exploration_reward (1 per novel tile)
  + distance_shaping (potential-based, when approaching goal on same map)
  + step_penalty (-0.5 when enabled)
  + button_penalty (-5 for start/select)
clipped to [-1000, 1000]
```

Location goals are sequential and include map-bank disambiguation. Pokedex goals are multi-fire: a threshold of N contributes N goal slots, firing once per integer increment.

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
├── stages/{first..fifth}.json             # Curriculum stages
└── {explore, inference, evaluate_reward_system}.json
tests/                                     # 105 tests
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

105 tests covering model init, losses, GAE, buffers, config inheritance, vectorised environments, action replay, reward calculation, and running statistics. Pure NumPy/PyTorch tests run instantly; emulator tests spin up real PyBoy subprocesses.

## Documentation

- **[AGENTS.md](./AGENTS.md)** — comprehensive technical reference: step semantics, reward rules, architecture details, configuration, invariants
- **[model_status.md](./model_status.md)** — evaluation report with stage-by-stage metrics and recommended fixes
- **[issues.md](./issues.md)** — tracked issues and planned improvements

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Run pre-commit hooks: `pre-commit run --all-files`
5. Submit a pull request

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
