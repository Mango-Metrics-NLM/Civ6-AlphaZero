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

## Architecture Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-10 | 13-channel MicroCiv encoder | Simplified encoding for faster Phase 1 training |
| 2026-01-10 | Hierarchical action space | Reduces effective branching factor from ~10^6 to ~10^3 |
| 2026-01-10 | SimpleMCTS + Civ6MCTS | SimpleMCTS for testing, Civ6MCTS for training |
| 2026-01-10 | ResNet backbone | Proven architecture from AlphaZero, good spatial reasoning |

## File Structure

```
alpha_zero/
├── __init__.py
├── envs/
│   ├── __init__.py
│   ├── microciv.py          # Simplified training environment
│   └── civ6_encoder.py      # State → Tensor encoding
├── core/
│   ├── __init__.py
│   ├── civ6_network.py      # Neural network architecture
│   ├── civ6_mcts.py         # MCTS implementation
│   └── civ6_actions.py      # Action space definition
└── training_civ6.py         # Training entry point

interfaces/                   # (Future: game interfaces)
├── __init__.py

tests/
├── __init__.py
├── unit/
│   ├── test_actions.py
│   ├── test_encoder.py
│   ├── test_environment.py
│   ├── test_network.py
│   └── test_mcts.py
└── integration/
    └── test_training.py

configs/
└── civ6_config.yaml         # Configuration file

checkpoints/civ6/            # Saved model checkpoints
logs/civ6/                   # Training logs
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

## Next Steps (Phase 2+)

1. **Hierarchical Actions**: Implement full hierarchical action selection
2. **Full State Encoding**: 84-channel encoder for complete Civ 6 state
3. **Game Integration**: Unciv JSON interface
4. **Distributed Training**: Multi-process self-play
5. **Scale Up**: Larger network, more simulations

---

*Last updated: 2026-01-10*
*Phase: 1 - MicroCiv Environment*
*Status: Phase 1 Complete*
