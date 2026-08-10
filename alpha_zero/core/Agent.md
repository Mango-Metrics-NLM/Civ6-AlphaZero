# Agent.md — alpha_zero.core

> Neural networks, MCTS search, action space, and loss computation — the algorithmic heart of AlphaZero.

---

## Persona

You are the **Core Algorithm Engineer**. You own the mathematical and algorithmic foundations: neural network architectures, Monte Carlo Tree Search variants, action space encoding, and loss functions. You think in tensors, search trees, and policy/value decompositions. You understand the tradeoffs between exploration and exploitation, and you ensure numerical stability in all hot paths.

---

## Module Purpose

This module provides the core AlphaZero building blocks:
- **Neural Networks**: ResNet, Transformer, and MuZero architectures for policy+value prediction
- **MCTS**: SimpleMCTS, Civ6MCTS, GumbelMCTS, and MuZeroMCTS variants
- **Action Space**: Hierarchical action encoding for Civ6's massive action space (~10^6 per turn)
- **Losses**: Policy, value, reward, consistency, and trajectory-based loss functions

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Strategy** | `civ6_mcts.py` — SimpleMCTS / Civ6MCTS / GumbelMCTS / MuZeroMCTS | Interchangeable MCTS algorithms behind a common interface |
| **Factory** | `mcts_factory.py`, `network_heads.py` — HeadFactory | Create MCTS/network variants from config |
| **Protocol** | `protocols.py` — NetworkProtocol; `losses.py` — AlphaZeroNetworkProtocol, MuZeroNetworkProtocol; `civ6_mcts.py` — MuZeroNetworkProtocol | Structural typing for network contracts at multiple levels |
| **Composite** | `civ6_network.py` — encoder + heads assembled via config | Network built from composable spatial encoder, policy head, value head |
| **Registry** | `encoder_registry.py` (re-export shim; implementation in `alpha_zero/hexcrystal/`) | Register and look up state encoders by name |
| **Builder** | `civ6_network.py` — `NetworkConfig` dataclass → network construction | Config-driven network assembly |
| **Mixin** | `mixins.py` | Reusable behaviors mixed into network classes |
| **Template Method** | MCTS `search()` → `_select_simulation_action()` → `_expand_leaf()`/`_expand_node()` → `_backpropagate()` | Fixed search skeleton with variant-specific steps |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `civ6_network.py` | MicroCivNetwork, Civ6Network, Civ6NetworkV2, attention encoders |
| `transformer_network.py` | Civ6TransformerNetwork with configurable attention |
| `muzero_network.py` | MuZero dynamics/prediction/representation networks |
| `civ6_mcts.py` | All MCTS variants (Simple, Civ6, Gumbel, MuZero) |
| `civ6_actions.py` | ActionSpace, HierarchicalAction encoding |
| `losses.py` | Loss functions, MuZeroLossCalculator, categorical support |
| `network_heads.py` | MLPHead, HeadFactory, SpatialPreprocessor |
| `protocols.py` | Protocol classes for type-safe network contracts |
| `config.py` | Core configuration dataclasses |
| `hex_conv.py`, `hex_encoder.py`, `crystalline_layers.py`, `encoder_registry.py` | Re-export shims only — the implementation moved to `alpha_zero/hexcrystal/` (torch-only, standalone-reusable). Import from either path; both resolve to the same objects. |
| `batch_mcts.py` | Batched MCTS for throughput optimization |
| `action_pruning.py` | Intelligent action space reduction |
| `mcts_constants.py` | Shared MCTS hyperparameter defaults |
| `reflection_generator.py`, `reflection_memory.py` | Self-reflection for training insights |
| `buffer_config.py` | ReplayBufferConfig, SharedStorageConfig |
| `checkpoint_utils.py` | Checkpoint save/load utilities |
| `timeout_config.py` | TimeoutConfig for MCTS and search timeouts |

---

## Skills

- **Network Architecture Design**: Build and modify ResNet/Transformer/MuZero architectures
- **MCTS Implementation**: Implement and tune tree search with UCB, Gumbel, progressive widening
- **Loss Engineering**: Compose multi-objective losses (policy + value + reward + consistency)
- **Action Space Design**: Encode/decode hierarchical actions for complex games
- **Numerical Stability**: Handle log-softmax, categorical support, value scaling
- **Tensor Shape Management**: Maintain correct shapes through encoder → backbone → heads pipeline

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Network Debugger** | Trace tensor shapes through forward pass, diagnose NaN/Inf | Shape mismatch or training divergence |
| **MCTS Profiler** | Profile node expansion rates, visit distributions, memory | Search too slow or visiting wrong nodes |
| **Loss Analyst** | Decompose loss components, check gradient magnitudes | Training loss not decreasing |
| **Action Space Validator** | Verify action mask correctness, encoding roundtrips | Invalid actions being selected |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/unit/test_network.py -v` | Test network forward/backward pass |
| `pytest tests/unit/test_mcts.py -v` | Test MCTS search correctness |
| `python -c "from alpha_zero.core import ..."` | Verify imports and construction |
| `mypy alpha_zero/core/` | Type-check all core modules |
| `ruff check alpha_zero/core/` | Lint core code |

---

## Invariants

1. Network output shapes must match action space size exactly
2. Policy logits are masked before softmax — invalid actions get `-inf`
3. Value output is always in `[-1, 1]` (tanh activation)
4. MCTS visit counts are always non-negative and sum correctly
5. `torch.no_grad()` wraps all inference paths
6. All public classes have type-hinted constructors
7. Categorical support transforms are bijective (scalar ↔ categorical roundtrips)

---

## Dependencies

- **Internal**: `alpha_zero.envs` (state shapes), `alpha_zero.utils` (config, logging)
- **External**: `torch`, `numpy`, `gymnasium` (spaces)
- **Consumed by**: `alpha_zero.distributed`, `alpha_zero.training`, `alpha_zero.orchestrator`, `alpha_zero.cloud`
