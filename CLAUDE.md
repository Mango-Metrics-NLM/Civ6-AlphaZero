# CLAUDE.md - Civ6 AlphaZero Project

## Quick Start

```bash
# Setup
cd /home/user/Civ6-AlphaZero
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/ -v

# Run smoke test
python -c "
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.core.civ6_network import MicroCivNetwork
env = MicroCivEnv()
obs, _ = env.reset()
print(f'Environment OK: {env.board_size}x{env.board_size}')
net = MicroCivNetwork(board_size=env.board_size, action_space_size=env.action_space.n)
print(f'Network OK: {sum(p.numel() for p in net.parameters()):,} params')
"

# Train MicroCiv (quick test)
python -m alpha_zero.training_civ6 --dry_run

# Train MicroCiv (full)
python -m alpha_zero.training_civ6 --num_games=100 --num_iterations=100
```

## Build Commands

| Command | Purpose |
|---------|---------|
| `pytest tests/unit/ -v` | Run unit tests |
| `pytest tests/integration/ -v` | Run integration tests |
| `pytest tests/ -v --tb=short` | Run all tests |
| `python -m alpha_zero.training_civ6` | Start training |
| `python -m alpha_zero.training_civ6 --dry_run` | Quick test run |
| `python -m alpha_zero.training_civ6 --num_simulations=200` | Train with more MCTS simulations |

## Project Status

**Phase 1: MicroCiv Environment - COMPLETE**

Implemented components:
- [x] MicroCiv environment (`alpha_zero/envs/microciv.py`)
- [x] Simplified state encoder (`alpha_zero/envs/civ6_encoder.py`)
- [x] MicroCiv action space (`alpha_zero/core/civ6_actions.py`)
- [x] Full hierarchical action space for future phases
- [x] Neural network with ResNet backbone (`alpha_zero/core/civ6_network.py`)
- [x] MCTS with neural network guidance (`alpha_zero/core/civ6_mcts.py`)
- [x] SimpleMCTS for testing without trained network
- [x] Training pipeline with self-play (`alpha_zero/training_civ6.py`)
- [x] Unit and integration tests

**Phase 2: Distributed Training & Game Integration - COMPLETE**

Implemented components:
- [x] Distributed training system (`alpha_zero/distributed/`)
  - SharedStorage for process-safe weight sharing
  - DistributedReplayBuffer with compression support
  - SelfPlayActor for parallel game generation
  - Learner with gradient accumulation
  - TrainingCoordinator for orchestration
- [x] Unciv JSON interface (`alpha_zero/interfaces/unciv_interface.py`)
  - Full save file parsing and conversion
  - Bidirectional state conversion
  - Action encoding/decoding for Unciv
- [x] Enhanced neural network with attention (`alpha_zero/core/civ6_network.py`)
  - Multi-head self-attention for spatial reasoning
  - Support for 84-channel full state encoding
  - Hierarchical policy head support
- [x] Logging and debugging infrastructure (`alpha_zero/utils/logging.py`)
  - GameLogger for per-game action logging
  - TrainingLogger for training metrics
  - MetricsTracker for aggregation and analysis
- [x] Dynamic configuration system (`alpha_zero/utils/config.py`)
  - YAML configuration loading
  - Environment variable overrides
  - ConfigManager for runtime access
- [x] Full test suite for Phase 2 components

## Architecture Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-10 | 13-channel MicroCiv encoder | Simplified encoding for faster Phase 1 training |
| 2026-01-10 | Hierarchical action space | Reduces effective branching factor from ~10^6 to ~10^3 |
| 2026-01-10 | SimpleMCTS + Civ6MCTS | SimpleMCTS for testing, Civ6MCTS for training |
| 2026-01-10 | ResNet backbone | Proven architecture from AlphaZero, good spatial reasoning |
| 2026-01-11 | Multi-head attention in network | Better spatial reasoning for complex game states |
| 2026-01-11 | Process-based distributed training | Avoid GIL, maximize CPU utilization for self-play |
| 2026-01-11 | Unciv JSON interface | Open-source Civ V clone with accessible save format |
| 2026-01-11 | YAML config with env overrides | Flexible configuration for different environments |

## File Structure

```
alpha_zero/
├── __init__.py
├── envs/
│   ├── __init__.py
│   ├── microciv.py          # Simplified training environment
│   └── civ6_encoder.py      # State → Tensor encoding (13 & 84 channel)
├── core/
│   ├── __init__.py
│   ├── civ6_network.py      # Neural networks (MicroCiv & full Civ6 with attention)
│   ├── civ6_mcts.py         # MCTS implementation
│   └── civ6_actions.py      # Action space definition (flat & hierarchical)
├── distributed/             # Phase 2: Distributed training
│   ├── __init__.py
│   ├── actor.py             # Self-play actor processes
│   ├── learner.py           # Central learner with gradient accumulation
│   ├── coordinator.py       # Training orchestration
│   └── shared_storage.py    # Process-safe weight & replay buffer sharing
├── interfaces/              # Phase 2: Game interfaces
│   ├── __init__.py
│   └── unciv_interface.py   # Unciv JSON save file interface
├── utils/                   # Phase 2: Utilities
│   ├── __init__.py
│   ├── logging.py           # Game & training loggers, metrics tracker
│   └── config.py            # Configuration loading & management
└── training_civ6.py         # Training entry point

tests/
├── __init__.py
├── unit/
│   ├── test_actions.py
│   ├── test_encoder.py
│   ├── test_environment.py
│   ├── test_network.py
│   ├── test_mcts.py
│   ├── test_logging.py      # Phase 2: Logging tests
│   ├── test_config.py       # Phase 2: Config tests
│   └── test_unciv_interface.py  # Phase 2: Unciv interface tests
└── integration/
    ├── test_training.py
    └── test_phase2.py       # Phase 2: Integration tests

configs/
└── civ6_config.yaml         # Full configuration (Phase 1 & 2)

checkpoints/civ6/            # Saved model checkpoints
logs/civ6/                   # Training logs
saves/unciv/                 # Unciv save files
```

## MicroCiv Environment

The MicroCiv environment is a simplified 4X game for Phase 1 training:

- **Board**: 16x16 grid (configurable)
- **Players**: 2 players, turn-based
- **Units**: Warrior, Settler, Archer
- **Actions**: Move, Attack, Skip, Produce, End Turn
- **Victory**: Eliminate enemy or most cities at turn limit

### State Encoding (13 channels)

| Channels | Feature |
|----------|---------|
| 0-2 | Terrain (plains, hills, mountain) |
| 3-5 | Own units (warrior, settler, archer) |
| 6-8 | Enemy units |
| 9 | Own cities |
| 10 | Enemy cities |
| 11 | Visibility (all visible in MicroCiv) |
| 12 | Constant (bias term) |

### Action Space

| Actions | Count |
|---------|-------|
| END_TURN | 1 |
| MOVE (unit × 8 dirs) | 160 |
| ATTACK (unit × 8 dirs) | 160 |
| SKIP (unit) | 20 |
| PRODUCE (city × unit_type) | 15 |
| **Total** | **356** |

## Neural Network Architecture

```
Spatial Input (13, H, W)
    │
    ▼
Conv2d (13 → 64) + BN + ReLU
    │
    ▼
ResBlock × 6 (64 filters)
    │
    ├─────────────────────┐
    │                     │
    ▼                     ▼
Policy Head          Value Head
Conv2d (64→2)        Conv2d (64→1)
FC (combined)        FC (combined)
    │                     │
    ▼                     ▼
Policy Logits (A)    Value [-1, 1]
```

## Training Pipeline

1. **Self-play**: Generate games using MCTS + network
2. **Store**: Add (state, policy, value) to replay buffer
3. **Train**: Sample batch, compute loss, update network
4. **Evaluate**: Test against random/previous version
5. **Checkpoint**: Save model periodically

### Loss Function

```
L = L_policy + L_value
L_policy = -Σ π_target * log(π_pred)  # Cross-entropy
L_value = (v_pred - v_target)²         # MSE
```

## Hyperparameters

```yaml
# MicroCiv defaults (Phase 1)
board_size: 16
num_res_blocks: 6
num_filters: 64
num_simulations: 50
batch_size: 128
learning_rate: 0.001
c_puct: 1.5
dirichlet_alpha: 0.3
dirichlet_epsilon: 0.25
temperature: 1.0
temperature_threshold: 30
```

## Known Issues

| Issue | Workaround |
|-------|------------|
| No GPU available in dev environment | Training works on CPU (slower) |
| SimpleMCTS random rollout is weak | Use neural network MCTS after initial training |

## Debug Commands

```bash
# Check tensor shapes
python -c "
from alpha_zero.envs.microciv import MicroCivEnv
env = MicroCivEnv()
obs, _ = env.reset()
print(f'Spatial: {obs[\"spatial\"].shape}')
print(f'Vector: {obs[\"vector\"].shape}')
print(f'Action mask: {obs[\"action_mask\"].shape}')
print(f'Valid actions: {obs[\"action_mask\"].sum():.0f}')
"

# Run single game with rendering
python -c "
from alpha_zero.envs.microciv import MicroCivEnv
import numpy as np
env = MicroCivEnv(board_size=8, render_mode='ansi', seed=42)
env.reset()
print(env.render())
for _ in range(10):
    mask = env.get_valid_actions()
    action = np.random.choice(np.where(mask > 0)[0])
    env.step(action)
print(env.render())
"

# Test network forward pass
python -c "
import torch
from alpha_zero.core.civ6_network import MicroCivNetwork
net = MicroCivNetwork(board_size=16, action_space_size=356)
spatial = torch.randn(1, 13, 16, 16)
vector = torch.randn(1, 64)
policy, value = net(spatial, vector)
print(f'Policy shape: {policy.shape}')
print(f'Value: {value.item():.3f}')
"
```

## Phase 2: Distributed Training

### Running Distributed Training

```bash
# Start distributed training with default settings
python -m alpha_zero.distributed.coordinator

# With custom configuration
python -m alpha_zero.distributed.coordinator \
    --num_actors=8 \
    --batch_size=256 \
    --max_games=10000

# Dry run for testing
python -m alpha_zero.distributed.coordinator --dry_run
```

### Distributed Architecture

```
                    ┌─────────────────┐
                    │   Coordinator   │
                    │  (orchestrates) │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │  Actor 1 │        │  Actor 2 │   ...  │  Actor N │
   │(self-play)│        │(self-play)│        │(self-play)│
   └─────┬────┘        └─────┬────┘        └─────┬────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Shared Storage  │
                    │ (weights+buffer)│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Learner      │
                    │ (trains network)│
                    └─────────────────┘
```

### Configuration

Phase 2 settings in `configs/civ6_config.yaml`:

```yaml
distributed:
  enabled: true
  num_actors: 64
  actor_batch_size: 8
  learner_batch_size: 256
  weight_sync_interval: 100
  max_queue_size: 10000
  compression_enabled: true
```

## Phase 2: Unciv Interface

### Loading Unciv Save Files

```python
from alpha_zero.interfaces import UncivInterface, create_unciv_interface

# Create interface
interface = create_unciv_interface(board_size=80, max_players=8)

# Load save file
game_state = interface.load_game("saves/unciv/autosave.json")

# Convert to internal format for encoder
internal_state = interface.to_internal_state(game_state)

# Convert action to Unciv format
unciv_action = interface.to_unciv_action(action_dict)
```

## Phase 2: Logging & Debugging

### Using the Logging System

```python
from alpha_zero.utils import (
    setup_logging, get_logger,
    GameLogger, TrainingLogger, MetricsTracker,
    LogConfig
)

# Setup logging
config = LogConfig(level="INFO", log_dir="logs")
setup_logging(config)
logger = get_logger(__name__)

# Game logging
game_logger = GameLogger(config, game_id="game_001")
game_logger.log_action(turn=1, player=0, action_type="MOVE", action_details={...})
game_logger.end_game(winner=0, total_turns=50, reason="elimination")

# Training metrics
training_logger = TrainingLogger(config)
training_logger.log_epoch(epoch=1, total_loss=0.5, policy_loss=0.3, value_loss=0.2)

# Metrics tracking
tracker = MetricsTracker(window_size=100)
tracker.add("loss", 0.5)
stats = tracker.get_stats("loss")  # Returns mean, std, min, max
```

## Phase 2: Configuration System

### Loading Configuration

```python
from alpha_zero.utils import load_config, ConfigManager, get_default_config

# Load from file
config = load_config("configs/civ6_config.yaml")

# Or use ConfigManager with environment variable overrides
# Set CIV6AZ_TRAINING_BATCH_SIZE=512 to override
manager = ConfigManager("configs/civ6_config.yaml")
training_config = manager.get_training_config()
distributed_config = manager.get_distributed_config()
```

## Next Steps (Phase 3+)

1. **Multi-GPU Training**: Extend distributed training to multiple GPUs
2. **Population-Based Training**: Adaptive hyperparameter tuning
3. **Full Civ 6 Integration**: Direct game interface (mod or API)
4. **Tournament System**: ELO-based opponent selection
5. **Human vs AI**: Interface for human players to challenge trained models

---

*Last updated: 2026-01-11*
*Phase: 2 - Distributed Training & Game Integration*
*Status: Phase 2 Complete*
