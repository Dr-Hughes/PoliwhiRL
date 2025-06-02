# PoliwhiRL Architecture Documentation

## Overview

PoliwhiRL is a sophisticated reinforcement learning library designed specifically for training AI agents to play sprite-based 2D Pokémon games. The system leverages PyBoy emulator integration and implements advanced RL techniques including Transformer-based models, intrinsic curiosity, macro action learning, and multi-agent training.

## System Architecture

```mermaid
graph TD
    A[PyBoy Emulator] --> B[Environment Interface]
    B --> C[Agent]
    C --> D[Transformer Model]
    C --> E[Exploration Memory]
    C --> F[ICM Module]
    C --> G[Macro Action Learner]

    D --> H[Action Selection]
    E --> D
    F --> I[Intrinsic Rewards]
    G --> J[Macro Actions]

    H --> K[Environment Step]
    I --> C
    J --> K
    K --> A

    C --> L[Multi-Agent Coordinator]
    L --> M[Model Averaging]
    M --> N[Shared Checkpoint]
```

## Core Components

### 1. Model Architectures

#### PPO Transformer Architecture

```mermaid
graph LR
    A[Game State] --> B[CNN Feature Extractor]
    A --> C[Exploration Memory]

    B --> D[Flexible Input Layer]
    C --> E[Exploration Encoder]

    D --> F[Positional Encoding]
    E --> G[Self-Attention]

    F --> H[Transformer Encoder]
    G --> I[Exploration Features]

    H --> J[Concatenation]
    I --> J

    J --> K[Actor Head]
    J --> L[Critic Head]

    K --> M[Action Probabilities]
    L --> N[State Value]
```

**Key Features:**
- **FlexibleInputLayer**: Handles both CNN (image) and direct state inputs
- **ExplorationEncoder**: Processes visit history using self-attention
- **Transformer Core**: Sequential state processing with positional encoding
- **Dual Outputs**: Actor (policy) and Critic (value estimation)

#### ICM (Intrinsic Curiosity Module)

```mermaid
graph TD
    A[Current State] --> B[State Encoder]
    C[Next State] --> D[State Encoder]
    E[Action] --> F[Action Encoder]

    B --> G[Forward Model]
    F --> G
    G --> H[Predicted Next State]

    B --> I[Inverse Model]
    D --> I
    I --> J[Predicted Action]

    H --> K[Forward Loss]
    D --> K

    J --> L[Inverse Loss]
    E --> L

    K --> M[Intrinsic Reward]
```

**Purpose**: Provides curiosity-driven exploration by rewarding agents for visiting novel states.

### 2. Macro Action Learning System

```mermaid
graph TD
    A[Episode Execution] --> B[Action Sequence Tracking]
    B --> C[Outcome Recording]
    C --> D[Pattern Analysis]

    D --> E{Success Rate > 70%?}
    E -->|Yes| F{Avg Reward > 0.2?}
    E -->|No| G[Discard Pattern]

    F -->|Yes| H[Create Macro Action]
    F -->|No| G

    H --> I[Add to Action Space]
    I --> J[Policy Integration]

    J --> K[Macro Action Selection]
    K --> L[Execute Primitive Sequence]
    L --> A
```

**How it Works:**
1. **Discovery Phase**: Tracks action sequences during episodes
2. **Evaluation Phase**: Assesses sequences based on success rate and reward
3. **Integration Phase**: Successful patterns become available as macro actions
4. **Execution Phase**: When selected, macro actions execute their underlying primitive sequence

**Key Benefits:**
- Automatic discovery of useful behavioral patterns
- Hierarchical action space (primitives + discovered macros)
- Temporal abstraction for complex tasks

### 3. Multi-Agent Training System

```mermaid
graph TD
    A[PPO Parallel Runner] --> B[Spawn N Agent Processes]

    B --> C[Agent 0]
    B --> D[Agent 1]
    B --> E[Agent N]

    C --> F[Individual Training]
    D --> G[Individual Training]
    E --> H[Individual Training]

    F --> I[Model Parameters]
    G --> J[Model Parameters]
    H --> K[Model Parameters]

    I --> L[Parameter Averaging]
    J --> L
    K --> L

    L --> M[Shared Checkpoint]
    M --> N[Next Iteration]
    N --> B

    C --> O[Individual Episode Data]
    D --> P[Individual Episode Data]
    E --> Q[Individual Episode Data]

    O --> R[Persistent Agent Checkpoints]
    P --> S[Persistent Agent Checkpoints]
    Q --> T[Persistent Agent Checkpoints]
```

**Key Features:**
- **Process Isolation**: Each agent runs in separate process for robustness
- **Model Averaging**: Parameters averaged across all successful agents
- **Data Persistence**: Individual agents maintain their own episode history
- **Fault Tolerance**: Training continues even if some agents fail

### 4. Curriculum Learning Pipeline

```mermaid
graph LR
    A[Stage 1: 1 Goal] --> B[Stage 2: 2 Goals]
    B --> C[Stage 3: 3 Goals]
    C --> D[...]
    D --> E[Stage 7: 7 Goals]

    A --> F[100 Steps]
    B --> G[150 Steps]
    C --> H[200 Steps]
    E --> I[700 Steps]

    F --> J[Model Transfer]
    G --> J
    H --> J
    I --> J

    J --> K[Next Stage Init]
    K --> B

    E --> L[Optimization Pass]
    L --> M[80% Step Reduction]
```

**Progressive Difficulty:**
- **Goal Complexity**: 1 → 7 location goals
- **Step Constraints**: Increasingly tight episode length limits
- **Model Transfer**: Each stage inherits previous stage's knowledge
- **Optimization**: Optional refinement pass with stricter constraints

### 5. Exploration and Memory Systems

#### Enhanced Exploration Memory

```mermaid
graph TD
    A[State Observation] --> B[Coordinate Extraction]
    B --> C[Visit Tracking]
    C --> D[Transition Mapping]

    D --> E[State Connectivity]
    E --> F[Waypoint Identification]

    C --> G[Exploration Bonus Calculation]
    G --> H[Decreasing Visit Rewards]

    F --> I[Important State Marking]
    I --> J[Navigation Assistance]

    A --> K[Action Sequence Tracking]
    K --> L[Pattern Recognition]
    L --> M[Macro Action Discovery]
```

**Capabilities:**
- **State Transition Learning**: Maps reachable states via actions
- **Waypoint Discovery**: Identifies strategically important locations
- **Exploration Bonuses**: Encourages visiting new areas
- **Sequence Learning**: Discovers recurring action patterns

#### Standard Exploration Memory

```mermaid
graph LR
    A[Screen Observation] --> B[Visit Frequency]
    B --> C[Exploration Tensor]
    C --> D[Model Input]

    A --> E[History Buffer]
    E --> F[Recent Visits]
    F --> D
```

**Simple Tracking**: Basic visit frequency and recency for exploration guidance.

## Data Flow and Training Loop

```mermaid
sequenceDiagram
    participant E as Environment
    participant A as Agent
    participant M as Model
    participant ICM as ICM Module
    participant Mem as Exploration Memory
    participant Macro as Macro Learner

    E->>A: State Observation
    A->>Mem: Update Visit History
    Mem->>M: Exploration Context
    A->>M: State + Context
    M->>A: Action Probabilities
    A->>Macro: Check for Macro Actions
    Macro->>A: Action (Primitive/Macro)
    A->>E: Execute Action
    E->>A: Next State + Reward
    A->>ICM: State Transition
    ICM->>A: Intrinsic Reward
    A->>Macro: Record Sequence + Outcome
    A->>A: Update Experience Buffer

    Note over A: Periodically
    A->>M: Train on Experiences
    M->>A: Updated Parameters
```

## Reward System Architecture

```mermaid
graph TD
    A[Environment State] --> B[Goal Tracker]
    B --> C{Goal Completed?}

    C -->|Yes| D[Goal Completion Reward]
    C -->|No| E[Distance-based Guidance]

    D --> F[Time Efficiency Check]
    F -->|Quick| G[Efficiency Bonus: 1.5x]
    F -->|Slow| H[Standard Reward]

    A --> I[Step Counter]
    I --> J[Exponential Step Penalty]

    A --> K[Pokédex Progress]
    K --> L[Species Discovery Reward]

    G --> M[Total Reward]
    H --> M
    E --> M
    J --> M
    L --> M
```

**Reward Components:**
- **Goal Completion**: 1.0-5.0 points (decreasing with time)
- **Efficiency Bonus**: 1.5x multiplier for quick completion
- **Distance Guidance**: Proportional reward for approaching goals
- **Step Penalty**: Exponentially increasing negative reward
- **Exploration**: Pokédex discovery bonuses

## Configuration System

```mermaid
graph LR
    A[Default Configs] --> D[Merged Config]
    B[User Config File] --> D
    C[Command Line Args] --> D

    D --> E[Agent Configuration]

    A --> A1[Core Settings]
    A --> A2[PPO Settings]
    A --> A3[Episode Settings]
    A --> A4[Reward Settings]
    A --> A5[Output Settings]
```

**Priority Order**: Command Line > User Config > Default Config

## File Structure Overview

```
PoliwhiRL/
├── agents/                     # RL Agent Implementations
│   ├── PPO/
│   │   ├── ppo_agent.py       # Core PPO agent
│   │   └── parallel_runner.py # Multi-agent coordinator
│   └── DQN/
│       ├── DQNPokemonAgent.py # Main DQN agent
│       └── multi_agent.py     # DQN parallel execution
├── models/                     # Neural Network Architectures
│   ├── PPO/
│   │   └── PPOTransformer.py  # Transformer-based actor-critic
│   ├── DQN/
│   │   └── DQNTransformer.py  # Transformer-based Q-network
│   └── ICM/
│       └── icm.py             # Intrinsic Curiosity Module
├── environment/                # Game Environment Interface
│   ├── gym_env.py             # PyBoy wrapper
│   └── rewards.py             # Reward calculation logic
├── utils/                      # Utility Systems
│   ├── macro_actions.py       # Macro action learning
│   └── resource_manager.py    # Multi-process coordination
└── replay/                     # Memory Systems
    ├── enhanced_exploration_memory.py
    └── exploration_memory.py
```

## Key Design Principles

### 1. Modularity
- Clear separation between environment, agent, model, and utility components
- Configurable system with JSON-based settings
- Swappable components (different models, memory systems, etc.)

### 2. Scalability
- Multi-agent training with parameter averaging
- Process-based parallelism for robustness
- Efficient memory management for long episodes

### 3. Exploration Strategy
- Multiple exploration mechanisms working in concert:
  - ICM for curiosity-driven exploration
  - Exploration memory for visit tracking
  - Macro actions for behavioral diversity
  - Curriculum learning for progressive complexity

### 4. Temporal Abstraction
- Macro action learning for hierarchical control
- Transformer models for sequential decision making
- Curriculum learning for staged skill acquisition

## Advanced Features

### Episode Data Persistence
- Each agent maintains individual episode history
- Statistics accumulate across training iterations
- Plotting and analysis preserve full training progression

### Checkpoint Management
- Shared model checkpoints for collective learning
- Individual agent checkpoints for data persistence
- Curriculum stage transfer for progressive training

### Resource Management
- Temporary file coordination across processes
- Memory-efficient exploration tracking
- Robust cleanup procedures

This architecture enables PoliwhiRL to tackle the complex, long-horizon task of playing Pokémon games efficiently through a combination of advanced RL techniques, careful engineering, and domain-specific optimizations.