# PoliwhiRL - AGENTS.md

## Overview

Single-agent PPO training for Pokémon Crystal (GameBoy Color) via the PyBoy emulator. The agent navigates the game world to reach sequential location goals defined in config, receiving extrinsic rewards from the `Rewards` system.

**Architecture:** `Screen Image (C,H,W) -> CNN (ResNet-style) -> Latent Embedding -> TransformerXL -> Actor/Critic heads`

---

## File Layout

```
main.py                          # Entry point: parse args, load config, dispatch by model type
migration_plan.md                # Detailed refactoring plan (May 2026)

PoliwhiRL/
├── __init__.py                  # exports: setup_and_train_PPO
├── PPO.py                       # single-agent training entry: create env, agent, train
├── agents/
│   ├── __init__.py              # exports: PPOAgent
│   └── PPO/
│       ├── __init__.py          # exports: PPOAgent
│       └── ppo_agent.py         # training loop, episode runner, save/load, metrics
├── environment/
│   ├── __init__.py              # exports: PyBoyEnvironment
│   ├── gym_env.py               # gymnasium Env wrapping PyBoy emulator
│   ├── rewards.py               # reward calculator: location goals, pokedex, step penalties
│   └── RAM.py                   # RAM variable extraction from emulator
├── models/
│   ├── CNN/
│   │   └── GameBoy.py           # GameBoyBlock (Conv2d+BN+ReLU), GameBoyOptimizedCNN
│   ├── PPO/
│   │   ├── __init__.py          # exports: PPOModel
│   │   ├── ppo_model_implementation.py   # PPO loss computation, GAE, optimizer
│   │   └── PPOTransformer.py    # CNN + TransformerXL + actor/critic heads
│   └── transformers/
│       └── positional_encoding.py  # standard sinusoidal positional encoding
├── replay/
│   ├── __init__.py              # exports: PPOMemory
│   └── ppo_storage.py           # in-memory rollout buffer with sliding window sequences
├── explorer/
│   ├── __init__.py              # exports: memory_collector
│   └── mem_collection.py        # manual/random data collection into SQLite
├── reward_evaluator/
│   ├── __init__.py              # exports: evaluate_reward_system
│   ├── evaluator.py             # reward system evaluation tool
│   └── moves.py                 # predefined action sequences for evaluation
└── utils/
    ├── __init__.py              # exports: record_step, plot_metrics
    └── visuals.py               # matplotlib plotting, step recording

configs/
├── default_configs/
│   ├── core_settings.json       # model, device, vision, sequence_length, checkpointing
│   ├── episode_settings.json    # episode_length, num_episodes, ignored_buttons
│   ├── ppo_settings.json        # gamma, lr, epsilon, entropy, GAE, update_frequency
│   ├── reward_settings.json     # location_goals, pokedex_goals, penalties, bonuses
│   ├── outputs_settings.json    # output_base_dir, checkpoint, results_dir, record_path
│   └── rom_settings.json        # rom_path, state_path, extra_files
├── evaluate_reward_system.json  # override config for model=evaluate
├── explore.json                 # override config for model=explore
└── simple_sanity.json           # quick sanity check config

tests/
├── test_PPO.py                  # model init, forward pass, save/load, PPO losses
├── test_pyboyenv.py             # env init, reset, step, vision modes, save/load state
├── test_pyboy_ram.py            # RAM extraction tests
└── test_reward_system.py        # reward calculation, penalties, goal achievement
```

---

## Three Modes

Dispatched in `main.py` based on `config["model"]`:

| Mode | Value | Purpose |
|------|-------|---------|
| Train | `"PPO"` | Single-agent PPO training loop |
| Explore | `"explore"` | Manual or random data collection to SQLite DB |
| Evaluate | `"evaluate"` | Run predefined action sequences, evaluate reward function |

---

## Model Architecture

### `PPOTransformer` (PoliwhiRL/models/PPO/PPOTransformer.py)

```
Input:  (batch, seq_len, C, H, W)   -- sequence of screen images
  |
  v
GameBoyCNN:  GameBoyBlock(in_ch, 16) -> GameBoyBlock(16, 32) -> flatten -> Linear -> d_model
  |
  v
PositionalEncoding: sinusoidal
  |
  v
TransformerXL: N x TransformerXLBlock (MultiheadAttention + LayerNorm + FFN with GELU)
  -- maintains persistent memory buffer across forward passes
  |
  v
Last token output (batch, d_model)
  |
  +---> fc_actor -> softmax -> (batch, action_size)  -- action probabilities
  +---> fc_critic -> (batch, 1)                      -- state value
```

Default hyperparameters: `d_model=128`, `n_heads=8`, `num_layers=4`, `mem_len=16`.

### `GameBoyBlock` (PoliwhiRL/models/CNN/GameBoy.py)

ResNet-style block: `Conv2d(in, out, k=3, s=2, p=1) -> BatchNorm2d -> ReLU`

---

## Training Parameters

Key config values (from `configs/default_configs/`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `sequence_length` | 8 | Frames per PPO sequence (1 step = 1 button press ~ 60 frames) |
| `ppo_update_frequency` | 128 | Collect 128 steps before updating |
| `ppo_epochs` | 10 | Gradient descent passes per update |
| `ppo_gamma` | 0.98 | Discount factor |
| `ppo_gae_lambda` | 0.95 | GAE lambda (enables GAE when > 0) |
| `ppo_epsilon` | 0.2 | PPO clipping range |
| `ppo_entropy_coef` | 0.01 | Initial entropy coefficient |
| `ppo_entropy_coef_decay` | 0.99 | Decay per episode |
| `ppo_entropy_coef_min` | 0.001 | Floor |
| `ppo_max_grad_norm` | 0.5 | Gradient clipping |
| `episode_length` | 50 | Max steps per episode |
| `num_episodes` | 12 | Episodes per training run |

---

## Data Flow

```
1. PPOAgent.train_agent() iterates over num_episodes
2. run_episode() creates PyBoyEnvironment, runs steps:
   - model.get_action(state_sequence) -> action
   - env.step(action) -> next_state, reward, done
   - model.compute_log_prob(state_sequence, action) -> log_prob
   - PPOMemory.store_transition(...)
3. After sequence_length+ steps: update_model()
   - PPOMemory.get_all_data() -> sliding window sequences as dict of tensors
   - PPOModel.update() -> PPO losses with GAE, backprop, optimizer step
   - PPOMemory.reset()
4. Checkpoint saves every checkpoint_frequency episodes
5. Metrics plotted every 10 episodes
```

---

## Config Loading

Priority: CLI args > user config file (`--use_config`) > merged defaults.

All default configs in `configs/default_configs/*.json` are auto-loaded and merged. The `outputs_settings.json` paths are prefixed with `output_base_dir`.

---

## Reward System

Defined in `PoliwhiRL/environment/rewards.py`:

- **Goal reward:** +100 per location goal reached
- **Sequence bonus:** +50 for completing goals in order
- **Checkpoint bonus:** +200 at checkpoint_goals milestones (default: 2, 4, 6)
- **All goals bonus:** +500 for completing all N_goals_target
- **Step penalty:** -1 per step (when `punish_steps=true`)
- **Button penalty:** -5 for pressing start/select
- **Timeout penalty:** -100 per uncompleted goal when steps > max_steps
- **Pokedex:** +25 seen, +50 owned

Goals are defined in `reward_settings.json` as `location_goals` (list of [x, y, map] coordinates) and `pokedex_goals` (dict with seen/owned targets).

---

## Environment

`PyBoyEnvironment` in `gym_env.py` wraps the PyBoy emulator:

- **Actions:** 9 discrete (noop, a, b, left, right, up, down, start, select)
- **Observation:** screen image `(C, H, W)` when `vision=true`, or game area `(18, 20)` when false
- **Scaling:** `scaling_factor` applies cv2 resize (default 0.5 -> 72x80 for RGB)
- **Grayscale:** `use_grayscale` converts to 1-channel
- **Step:** each action holds for `button_hold_frames` (15), then ticks remaining frames to reach `frames_per_action` (90) total
- **State save/load:** pickle of emulator state + gym state for checkpoint continuation

---

## Running

```bash
# Basic training
python main.py

# With custom config
python main.py --use_config configs/simple_sanity.json

# Override specific settings
python main.py --vision false --num_episodes 50 --ppo_learning_rate 0.0003

# Explore mode (manual control)
python main.py --use_config configs/explore.json --manual_control true

# Reward evaluation
python main.py --use_config configs/evaluate_reward_system.json

# Multi-stage curriculum (shell script)
bash run_curriculum.sh
```

---

## Testing

```bash
pytest tests/ -v
```

Tests require the ROM and state files in `emu_files/`.

---

## Dependencies

See `requirements.txt`. Key packages: `torch`, `pyboy==2.4.1`, `gymnasium`, `opencv-python`, `numpy`, `tqdm`, `matplotlib`.

---

## What Was Removed (May 2026 Refactor)

The following subsystems were removed to simplify to barebones single-agent PPO:

- **DQN** (entire agent/model/config/test stack)
- **ICM** (intrinsic curiosity module for intrinsic rewards)
- **Exploration memory** (screen-hash-based visit tracking, enhanced variant)
- **Multiprocessing** (parallel runner, resource pool, shared model manager)
- **Macro actions** (action sequence discovery)
- **Attention modules** (spatial/feature attention, auxiliary prediction tasks)
- **N-Step returns** (unused reward computation)
- **SQLite rollout storage** (train_from_memory path)
- **Curriculum training** (progressive goal/episode_length scheduling)
