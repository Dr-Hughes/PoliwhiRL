# PoliwhiRL - AGENTS.md

## Overview

Single-agent PPO training for Pokémon Crystal (GameBoy Color) via the PyBoy emulator. The agent navigates the game world to reach sequential location goals defined in config, receiving extrinsic rewards from the `Rewards` system.

**Architecture:** `Screen Image (C,H,W) -> CNN (ResNet-style) -> Latent Embedding -> TransformerXL -> Actor/Critic heads`

---

## File Layout

```
main.py                          # Entry point: parse args, load config, dispatch by model type

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
│   │   └── PPOTransformer.py    # GameBoyCNN + TransformerXL + actor/critic heads
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
├── first_steps.json             # 2-goal sanity check: stairs + talk to mom
└── simple_sanity.json           # quick sanity check config (N_goals=0)

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
  -- memory is caller-managed: mems passed in, new_mems returned each forward pass
  -- each block concatenates cached mem onto current chunk for extended attention context
  |
  v
Last token output (batch, d_model)
  |
  +---> fc_actor -> softmax -> (batch, action_size)  -- action probabilities
  +---> fc_critic -> (batch, 1)                      -- state value
```

Default hyperparameters: `d_model=128`, `n_heads=8`, `num_layers=4`, `mem_len=16`.

**Memory lifecycle:** The agent calls `model.init_mems(batch_size=1)` at episode start, passes `mems` into `get_action()` each step, and receives `new_mems` to carry forward. Each transition stores a snapshot of its `mems` in PPOMemory so `update()` can reproduce the same attention context during replay. Memory is detached (no gradients through the cache).

### `TransformerXLBlock`

Each block receives `(x, mem)` where `mem` is `(B, mem_len, d_model)`. It concatenates `[mem, x]` for self-attention, then returns only the output for the current chunk positions. The new memory is the last `mem_len` tokens of the extended sequence, detached.

### `GameBoyBlock` (PoliwhiRL/models/CNN/GameBoy.py)

ResNet-style block: `Conv2d(in, out, k=3, s=2, p=1) -> BatchNorm2d -> ReLU`

### `GameBoyCNN` (PoliwhiRL/models/PPO/PPOTransformer.py)

Two GameBoyBlock stages (in_ch->16->32), flatten, Linear to d_model, ReLU. Used as the visual encoder within PPOTransformer.

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
| `ppo_learning_rate` | 3e-4 | Peak learning rate (CyclicLR oscillates between 1e-5 and this) |
| `episode_length` | 50 | Max steps per episode |
| `num_episodes` | 12 | Episodes per training run |

The learning rate uses a `CyclicLR` scheduler (`triangular2` mode, `step_size_up=100`), stepped once per episode.

---

## Data Flow

```
1. PPOAgent.train_agent() iterates over num_episodes
2. run_episode() creates PyBoyEnvironment, runs steps:
   - mems = model.init_mems(batch_size=1)  -- fresh each episode
   - model.get_action(state_sequence, mems) -> action, log_prob, new_mems
   - env.step(action) -> next_state, reward, done
   - PPOMemory.store_transition(state, ..., mems)  -- snapshot of mems stored
   - mems = new_mems  -- carry forward to next step
3. After sequence_length+ steps (or every update_frequency steps): update_model()
   - PPOMemory.get_all_data() -> sliding window sequences as dict of tensors
     (includes stored mems snapshots aligned to action positions)
   - PPOModel.update(data, episode) -> PPO losses with GAE, backprop, optimizer step
   - PPOMemory.reset()
4. model.step_scheduler() called each episode
5. Checkpoint saves every checkpoint_frequency episodes
6. Metrics plotted every 10 episodes
```

---

## Config Loading

Priority: CLI args > user config file (`--use_config`) > merged defaults.

All default configs in `configs/default_configs/*.json` are auto-loaded and merged. The `outputs_settings.json` paths are prefixed with `output_base_dir`.

---

## Reward System

Defined in `PoliwhiRL/environment/rewards.py`. All reward magnitudes are configurable via JSON.

| Signal | Default | Config Key | Notes |
|--------|---------|------------|-------|
| Goal reached | +100 | `goal_reward` | Per location goal |
| Sequential bonus | +50 | `sequence_bonus` | Only when `require_sequential=true` |
| Checkpoint bonus | +200 | `checkpoint_bonus` | At `checkpoint_goals` milestones |
| All goals bonus | 500 | `all_goals_bonus` | When N_goals >= N_goals_target |
| Early completion | 0 | `early_completion_bonus` | Additional bonus on final goal |
| Exploration | 0 | `exploration_reward` | Per unvisited (x,y,map) tile this episode |
| Step penalty | -1 | `step_penalty` | When `punish_steps=true`; configurable magnitude |
| Button penalty | -5 | `button_penalty` | Hardcoded for start/select |
| Pokedex seen | +25 | -- | When new Pokémon is seen |
| Pokedex owned | +50 | -- | When new Pokémon is caught |

Goals are defined in `reward_settings.json` as `location_goals` (list of `[x, y, map]` coordinate lists) and `pokedex_goals` (dict with seen/owned targets). Location goals support multiple coordinate options per goal (e.g., `[[8,4,6], [9,4,6]]` matches either position).

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

# First steps: learn to exit room and talk to mom
python main.py --use_config configs/first_steps.json

# With custom config
python main.py --use_config configs/simple_sanity.json

# Override specific settings
python main.py --vision false --num_episodes 50 --ppo_learning_rate 0.0003

# Explore mode (manual control)
python main.py --use_config configs/explore.json --manual_control true

# Reward evaluation
python main.py --use_config configs/evaluate_reward_system.json
```

---

## Testing

```bash
pytest tests/ -v
```

Tests require the ROM and state files in `emu_files/`. Two tests (`test_reward_scaling_and_clipping`, `test_reward_step_penalty_progression`) are known failures from outdated assumptions about hardcoded reward values.

---

## Dependencies

See `requirements.txt`. Key packages: `torch`, `pyboy==2.4.1`, `gymnasium`, `opencv-python`, `numpy`, `tqdm`, `matplotlib`.
