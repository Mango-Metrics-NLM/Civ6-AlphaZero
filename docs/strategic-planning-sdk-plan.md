# Strategic Planning SDK — Implementation Plan

> **Origin:** Generalizes the [Civ6-AlphaZero](https://github.com/Mango-Metrics-NLM/Civ6-AlphaZero) project into a domain-agnostic strategic planning SDK.
>
> **Date:** 2026-02-06
>
> **Status:** Planning

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Component Reuse Strategy](#3-component-reuse-strategy)
4. [Phased Implementation Plan](#4-phased-implementation-plan)
5. [Dependency Management](#5-dependency-management)
6. [Risk Analysis](#6-risk-analysis)
7. [Testing Strategy](#7-testing-strategy)
8. [Migration Path](#8-migration-path)
9. [Success Criteria Verification](#9-success-criteria-verification)

---

## 1. Executive Summary

### What We're Building

A domain-agnostic SDK that extracts AlphaZero-style hierarchical MCTS + neural network planning from the Civ6 codebase into pluggable environments for multi-agent, partial-information, multi-horizon decision-making. The SDK supports domains including logistics, energy grid management, climate policy, healthcare, trading, and beyond.

### Key Insight

The existing codebase was **designed** with generalization in mind — `protocols.py` defines 8 clean protocol interfaces that are fully domain-agnostic. However, virtually no concrete implementation actually codes against those protocols. Instead, they directly import `MicroCivEnv`, `MicroCivNetwork`, etc. **The core work is not redesigning algorithms but consistently threading dependency injection and registry patterns through existing concrete implementations so they honor the protocols that already exist.**

### Scope Summary

| Category | Count | Effort |
|----------|-------|--------|
| Reuse as-is | 13 components | Minimal (packaging only) |
| Refactor for generalization | 25 components | Medium-Large |
| Build from scratch | 12 components | Medium-Large |
| **Total estimated phases** | **7 phases** | |

---

## 2. Architecture Overview

### 2.1 SDK Package Structure

```
src/                                    # SDK root (new namespace)
├── __init__.py                         # Public API surface
├── cli.py                              # Click-based CLI entry point
├── core/                               # Domain-agnostic core engine
│   ├── __init__.py
│   ├── protocols.py                    # All Protocol definitions (extended from existing)
│   ├── types.py                        # Shared types, enums, TypeAliases
│   ├── config.py                       # Pydantic configuration models (new)
│   ├── logging.py                      # structlog setup (wraps existing utils/logging.py)
│   ├── exceptions.py                   # Custom exception hierarchy
│   ├── mcts/                           # MCTS engine (refactored from core/civ6_mcts.py)
│   │   ├── __init__.py
│   │   ├── tree.py                     # SearchTree: nodes, edges, statistics
│   │   ├── selection.py                # PUCT, Gumbel selection policies
│   │   ├── expansion.py               # Node expansion with action masking
│   │   ├── evaluation.py              # Leaf evaluation (network or rollout)
│   │   ├── backpropagation.py         # Q-value backup with virtual loss
│   │   ├── action_picker.py           # Temperature-based action selection
│   │   └── search.py                  # Orchestrates full MCTS search
│   ├── networks/                       # Neural network components
│   │   ├── __init__.py
│   │   ├── encoders.py                # StateEncoder implementations
│   │   ├── backbones.py               # ResidualTower, TransformerStack, GNN
│   │   ├── heads.py                   # PolicyHead, ValueHead, AuxHeads
│   │   ├── model.py                   # Full model assembly from config
│   │   └── utils.py                   # Weight init, gradient clipping
│   ├── training/                       # Training loop components
│   │   ├── __init__.py
│   │   ├── self_play.py               # Self-play actor (from distributed/actor.py)
│   │   ├── learner.py                 # Network training (from distributed/learner.py)
│   │   ├── evaluator.py               # Elo rating, baseline benchmarks
│   │   ├── replay_buffer.py           # Memory-mapped prioritized replay
│   │   ├── checkpoint.py              # Versioned save/load (from orchestrator/checkpointer.py)
│   │   ├── curriculum.py              # CurriculumScheduler
│   │   └── orchestrator.py            # Coordinates all training components
│   └── explain/                        # Explainability
│       ├── __init__.py
│       ├── tracer.py                   # MCTS decision trace exporter
│       ├── visualizer.py              # Tree visualization
│       └── analyzer.py                # Strategy pattern analysis
│
├── environments/                       # Domain-specific environment plugins
│   ├── __init__.py
│   ├── registry.py                    # Environment discovery & registration (NEW)
│   ├── base.py                        # Shared utilities for environments
│   ├── mini_games/                    # Curriculum mini-game environments
│   │   ├── __init__.py
│   │   ├── resource_management.py     # Resource allocation mini-game
│   │   ├── territory_control.py       # Territory expansion mini-game
│   │   └── tactical_combat.py         # Unit combat mini-game
│   ├── civ6/                          # Civ6 domain adapter (extracted from alpha_zero/)
│   │   ├── __init__.py
│   │   ├── environment.py             # MicroCivEnv wrapped as StrategicEnvironment
│   │   ├── encoder.py                 # 84-channel spatial + 512-dim vector
│   │   └── action_decoder.py          # Hierarchical action mapping
│   ├── logistics/                     # (stub for expansion)
│   ├── energy/                        # (stub for expansion)
│   └── climate_policy/               # (stub for expansion)
│
├── adapters/                          # External system adapters
│   ├── __init__.py
│   ├── mlflow_adapter.py
│   ├── tensorboard_adapter.py
│   └── distributed_adapter.py
│
configs/                               # Hydra configuration hierarchy (new)
│   ├── config.yaml                    # Root config
│   ├── environment/                   # Per-environment configs
│   ├── network/                       # Architecture configs
│   ├── mcts/                          # MCTS hyperparameter configs
│   ├── training/                      # Training loop configs
│   └── curriculum/                    # Curriculum progression configs
│
tests/                                 # Full test suite (new structure)
│   ├── conftest.py                    # Shared fixtures
│   ├── factories.py                   # Test data factories
│   ├── unit/core/mcts/               # MCTS unit tests
│   ├── unit/core/networks/           # Network unit tests
│   ├── unit/core/training/           # Training unit tests
│   ├── unit/environments/            # Environment tests
│   ├── property/                      # Hypothesis property tests
│   ├── integration/                   # Multi-component tests
│   └── smoke/                         # Quick validation tests
│
docs/                                  # Documentation
│   ├── architecture/                  # C4 Mermaid diagrams
│   ├── environments/                  # Plugin guides
│   └── training/                      # Training guides
```

### 2.2 Dependency Flow (Layered Architecture)

```
LAYER 0: FOUNDATION (no intra-SDK deps)
┌──────────────────────────────────────────────────────────┐
│  types.py   config.py   logging.py   exceptions.py       │
└──────────────────────────┬───────────────────────────────┘
                           │
LAYER 1: CORE ABSTRACTIONS (depend only on Layer 0)
┌──────────────────────────┼───────────────────────────────┐
│  protocols.py  actions/  │ environments/base  rewards/    │
│  data/         networks/ │ storage/base                   │
└──────────────────────────┼───────────────────────────────┘
                           │
LAYER 2: ALGORITHMS (depend on Layers 0-1)
┌──────────────────────────┼───────────────────────────────┐
│  mcts/   losses/   network_heads/   scheduling/           │
│  checkpointing/                                           │
└──────────────────────────┼───────────────────────────────┘
                           │
LAYER 3: TRAINING INFRASTRUCTURE (depend on Layers 0-2)
┌──────────────────────────┼───────────────────────────────┐
│  self_play/   learner/   curriculum/   replay_buffer/     │
│  distributed/actor  distributed/learner  storage/local    │
└──────────────────────────┼───────────────────────────────┘
                           │
LAYER 4: ORCHESTRATION (depend on all layers)
┌──────────────────────────┼───────────────────────────────┐
│  orchestrator/   coordinator/   rag/   advisor/   cli/    │
└──────────────────────────┴───────────────────────────────┘
```

### 2.3 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Protocol-based plugin system** (typing.Protocol) | Structural subtyping — domain implementations need zero inheritance from SDK base classes |
| **Pydantic v2 for SDK config** (coexists with existing dataclasses) | Runtime validation, JSON schema generation, CLI override support. Existing dataclass configs wrapped via adapters |
| **Hydra for config composition** | Hierarchical YAML with CLI overrides. Replaces manual `from_yaml()` + `apply_env_overrides()` pattern |
| **structlog wrapping stdlib logging** | Non-breaking: existing `logging.getLogger()` calls continue working. SDK modules get structured context binding |
| **Registry pattern for components** | String-based references in config → lazy resolution at runtime. Breaks circular deps between config and implementations |
| **Separate `src/` namespace** (not inside `alpha_zero/`) | Clean SDK boundary. Civ6-specific code becomes a domain adapter registered via the plugin system |
| **Extract data types into `core/types.py`** | `TrainingExample`, `GameTrajectory` referenced by both `storage/` and `distributed/`. Shared types module breaks implicit coupling |

---

## 3. Component Reuse Strategy

### 3.1 Reuse As-Is (13 components — minimal packaging work)

These components contain no domain-specific logic and can be lifted directly into the SDK with only namespace changes.

| Existing File | SDK Destination | Notes |
|---------------|-----------------|-------|
| `core/protocols.py` | `src/core/protocols.py` | 8 protocol interfaces — already domain-agnostic |
| `core/mcts_constants.py` | `src/core/types.py` (merge) | Numerical constants with env-var overrides |
| `core/losses.py` | `src/core/networks/losses.py` | Purely mathematical — no domain references |
| `core/network_heads.py` | `src/core/networks/heads.py` | MLPHead, SpatialPreprocessor, HeadFactory |
| `core/mixins.py` | `src/core/types.py` (merge) | SerializableMixin, ActionMaskMixin |
| `utils/logging.py` | `src/core/logging.py` (wrap) | MetricsTracker, TrainingLogger |
| `utils/llm_client.py` | `src/adapters/llm_client.py` | Provider-agnostic LLM client |
| `orchestrator/scheduler.py` | `src/core/training/scheduler.py` | HyperparameterScheduler — pure math |
| `orchestrator/checkpointer.py` | `src/core/training/checkpoint.py` | CheckpointManager — generic torch save/load |
| `storage/backend.py` | `src/core/training/storage.py` | Abstract StorageBackend — already abstract |
| `storage/factory.py` | `src/core/training/storage.py` (merge) | create_storage_backend — generic factory |
| `cloud/preemption.py` | `src/adapters/preemption.py` | Signal handling — zero domain logic |
| `core/checkpoint_utils.py` | `src/core/networks/utils.py` (merge) | Safe torch serialization |

### 3.2 Refactor (25 components — algorithm extraction + DI)

These contain valuable algorithmic logic entangled with Civ6-specific types. The refactoring pattern is consistent: **replace direct imports of concrete types with protocol-based dependency injection and factory/registry lookups.**

| Existing File | SDK Destination | Key Changes | Effort |
|---------------|-----------------|-------------|--------|
| `core/civ6_mcts.py` | `src/core/mcts/` (split into 6 files) | Rename `Civ6MCTS` → `PUCTMCTS`; extract SimpleMCTS, GumbelMCTS, MuZeroMCTS; move inline protocols to `protocols.py` | L |
| `core/civ6_network.py` | `src/core/networks/backbones.py` + `model.py` | Rename `MicroCivNetwork` → `AlphaZeroNetwork`; remove Civ6 defaults; make all dims required | M |
| `core/civ6_actions.py` | `src/core/types.py` + `environments/civ6/actions.py` | Extract generic `HierarchicalActionSpace` base; move Civ6 enums to domain adapter | M |
| `core/transformer_network.py` | `src/core/networks/backbones.py` (merge) | Remove default dims, make configurable | M |
| `core/muzero_network.py` | `src/core/networks/backbones.py` (merge) | Same treatment as transformer_network | M |
| `distributed/shared_storage.py` | `src/core/types.py` (data) + `src/core/training/replay_buffer.py` | Extract `TrainingExample`, `GameTrajectory` to shared types; generalize `SharedStorage`, `DistributedReplayBuffer` | M |
| `distributed/actor.py` | `src/core/training/self_play.py` | Replace `MicroCivEnv`/`Civ6MCTS` imports with DI (accept env_factory, mcts_factory as constructor params) | M |
| `distributed/learner.py` | `src/core/training/learner.py` | Replace `MicroCivNetwork`/`compute_loss` imports with DI | M |
| `distributed/coordinator.py` | `src/core/training/orchestrator.py` | Replace all concrete type imports with protocol/factory params; generalize `CoordinatorConfig` | L |
| `orchestrator/config.py` | `src/core/config.py` | Extract `ScheduleConfig`, `ResourceBudget` (already generic); replace `CIV6AZ_` prefix with configurable SDK prefix; make presets domain-provided | M |
| `orchestrator/orchestrator.py` | `src/core/training/orchestrator.py` | Replace 7+ direct Civ6 factory imports with generic component registry; `TrainingState`, `GameResult` are generic | L |
| `orchestrator/rag_integration.py` | `src/adapters/rag_adapter.py` | Rename "strategy" → "knowledge"; already mostly generic | S |
| `orchestrator/strategy_advisor.py` | `src/adapters/advisor.py` | Make prompt templates domain-pluggable | M |
| `utils/factories.py` | `src/core/factories.py` | Replace hardcoded type selection with registry lookups | M |
| `utils/config.py` | `src/core/config.py` (merge) | Extract `UncivConfig` to domain adapter; keep generic ConfigManager | S |
| `training_civ6.py` | `src/core/training/pipeline.py` | Refactor into generic `TrainingPipeline` class accepting domain components | M |
| `envs/microciv.py` | `src/environments/civ6/environment.py` | Wrap as `StrategicEnvironment` protocol implementation | M |
| `envs/civ6_encoder.py` | `src/environments/civ6/encoder.py` | Implement `StateEncoder` protocol | M |
| `storage/local_backend.py` | `src/core/training/storage.py` (merge) | Already clean — just relocate | S |
| `storage/replay_buffer.py` | `src/core/training/replay_buffer.py` (merge) | Rename data type references | S |
| `storage/config.py` | `src/core/config.py` (merge) | Already domain-agnostic | S |
| `cloud/config.py` | `src/adapters/cloud_config.py` | Make env var prefix configurable | S |
| `cloud/selfplay_worker.py` | `src/adapters/cloud/selfplay_worker.py` | DI for env/network/MCTS factories | M |
| `cloud/trainer.py` | `src/adapters/cloud/trainer.py` | DI for network/loss | M |
| `cloud/evaluator.py` | `src/adapters/cloud/evaluator.py` | Decouple from Civ6 env creation | M |

### 3.3 Build New (12 components)

| Component | SDK Location | Description | Effort |
|-----------|-------------|-------------|--------|
| **Domain Registry** | `src/environments/registry.py` | Plugin registration: `@sdk.register_domain("name")` with env_factory, encoder, action_space. Entry-point discovery. | L |
| **Abstract Action Space** | `src/core/types.py` | Generic `ActionSpace` protocol with `encode()`, `decode()`, `mask()`, `size` supporting flat, hierarchical, continuous | M |
| **State Encoder Registry** | `src/core/networks/encoders.py` | Framework for registering domain-specific encoders: raw state → (spatial, vector, action_mask) | M |
| **SDK Public API** | `src/__init__.py` | Clean exports, version management, namespace: `sdk.mcts`, `sdk.networks`, `sdk.training` | M |
| **Generic Training Pipeline** | `src/core/training/pipeline.py` | High-level `pipeline = TrainingPipeline(domain="chess", config=config); pipeline.run()` | M |
| **Domain Adapter Layer** | `src/environments/civ6/` | Civ6 becomes first concrete adapter — reference implementation | M |
| **Config Schema Validation** | `src/core/config.py` | Pydantic-based validation ensuring domain configs are complete before training | S |
| **SDK Test Framework** | `tests/conftest.py` + `tests/factories.py` | `assert_valid_environment(env)`, `assert_valid_network(net)`, `assert_mcts_convergence(mcts)` | M |
| **Example Mini-Game Domains** | `src/environments/mini_games/` | 3 simple games: resource management, territory control, tactical combat | M |
| **Experiment Tracking Abstraction** | `src/adapters/experiment_tracker.py` | Unified `tracker.log_metric()` for W&B, TensorBoard, MLflow | S |
| **Environment Wrapper Library** | `src/environments/base.py` | Composable wrappers: padding, normalization, reward shaping, multi-agent turn management | M |
| **CLI Entry Point** | `src/cli.py` | Click-based CLI: `train`, `evaluate`, `explain`, `health-check`, `benchmark` | M |

---

## 4. Phased Implementation Plan

### Phase 1: Foundation Layer
**Goal:** Establish SDK skeleton, type system, config, logging — the base everything else depends on.
**Dependencies:** None (leaf layer)

| Task | Files to Create/Modify | Source (Existing) | Effort |
|------|----------------------|-------------------|--------|
| 1.1 Initialize SDK package structure | `src/__init__.py`, `pyproject.toml` (new SDK section), `Makefile` | — | S |
| 1.2 Define core types and constants | `src/core/types.py` | `core/mcts_constants.py`, `core/mixins.py`, `core/civ6_actions.py` (generic parts) | M |
| 1.3 Build Pydantic config models | `src/core/config.py` | `orchestrator/config.py` (ScheduleConfig, ResourceBudget), `utils/config.py` | M |
| 1.4 Set up structlog wrapping stdlib | `src/core/logging.py` | `utils/logging.py` (MetricsTracker, TrainingLogger) | S |
| 1.5 Define exception hierarchy | `src/core/exceptions.py` | `cloud/exceptions.py` pattern | S |
| 1.6 Create Hydra config hierarchy | `configs/config.yaml`, `configs/environment/`, `configs/network/`, `configs/mcts/`, `configs/training/`, `configs/curriculum/` | `configs/civ6_config.yaml`, `configs/orchestrator*.yaml` | M |
| 1.7 Set up pyproject.toml with all deps | `pyproject.toml` (SDK section) | Existing `pyproject.toml` + new deps | S |
| 1.8 Write foundation tests | `tests/unit/core/test_config.py`, `tests/unit/core/test_logging.py`, `tests/unit/core/test_types.py` | — | M |

**Deliverable:** `src/core/types.py`, `src/core/config.py`, `src/core/logging.py`, `src/core/exceptions.py` all pass `mypy --strict` and have 100% test coverage.

---

### Phase 2: Core Abstractions
**Goal:** Define all protocol interfaces and abstract base patterns. No implementations yet — just contracts.
**Dependencies:** Phase 1

| Task | Files to Create/Modify | Source (Existing) | Effort |
|------|----------------------|-------------------|--------|
| 2.1 Extend Protocol definitions | `src/core/protocols.py` | `core/protocols.py` (8 existing protocols) + `StrategicEnvironment` + `StateEncoder` + `ActionSpace` | M |
| 2.2 Define StrategicEnvironment protocol | `src/core/protocols.py` | New — based on spec Section 6.1 | M |
| 2.3 Define HierarchicalActionSpace protocol | `src/core/protocols.py` | Generic extraction from `core/civ6_actions.py` | M |
| 2.4 Define ObservationSpec and ActionSpec | `src/core/types.py` (extend) | New Pydantic models | S |
| 2.5 Define TrainingExample, GameTrajectory | `src/core/types.py` (extend) | Extract from `distributed/shared_storage.py` | S |
| 2.6 Define RewardShaper protocol | `src/core/protocols.py` | `utils/reward_shaping.py` (extract interface) | S |
| 2.7 Define StorageBackend protocol | `src/core/protocols.py` | `storage/backend.py` (adopt) | S |
| 2.8 Write protocol compliance test helpers | `tests/conftest.py`, `tests/factories.py` | New — `assert_valid_environment()`, `assert_valid_network()` | M |

**Deliverable:** All protocols defined, documented with Google-style docstrings, and have corresponding test helpers that any domain implementor can use for validation.

---

### Phase 3: Algorithms (MCTS + Networks)
**Goal:** Generalize MCTS and neural network components. The algorithmic heart of the SDK.
**Dependencies:** Phase 2

| Task | Files to Create/Modify | Source (Existing) | Effort |
|------|----------------------|-------------------|--------|
| 3.1 Extract MCTS tree data structure | `src/core/mcts/tree.py` | `core/civ6_mcts.py` (MCTSNode dataclass) | M |
| 3.2 Implement selection policies | `src/core/mcts/selection.py` | `core/civ6_mcts.py` (PUCT formula from Civ6MCTS._select) | M |
| 3.3 Implement node expansion | `src/core/mcts/expansion.py` | `core/civ6_mcts.py` (_expand method + Dirichlet noise) | M |
| 3.4 Implement leaf evaluation | `src/core/mcts/evaluation.py` | `core/civ6_mcts.py` (_evaluate + rollout) | S |
| 3.5 Implement backpropagation | `src/core/mcts/backpropagation.py` | `core/civ6_mcts.py` (_backup + virtual loss) | S |
| 3.6 Implement action picker | `src/core/mcts/action_picker.py` | `core/civ6_mcts.py` (temperature-based selection) | S |
| 3.7 Orchestrate full MCTS search | `src/core/mcts/search.py` | `core/civ6_mcts.py` (Civ6MCTS.search → PUCTMCTS.search) | M |
| 3.8 Implement Gumbel MCTS variant | `src/core/mcts/search.py` (extend) | `core/civ6_mcts.py` (GumbelMCTS) | M |
| 3.9 Build ResNet backbone | `src/core/networks/backbones.py` | `core/civ6_network.py` (ResidualBlock, SpatialEncoder) | M |
| 3.10 Build Transformer backbone | `src/core/networks/backbones.py` (extend) | `core/transformer_network.py` | M |
| 3.11 Build network heads | `src/core/networks/heads.py` | `core/network_heads.py` (reuse) + `core/civ6_network.py` (PolicyHead, ValueHead) | M |
| 3.12 Build model assembly | `src/core/networks/model.py` | New — config-driven model construction | M |
| 3.13 Port loss functions | `src/core/networks/losses.py` | `core/losses.py` (reuse as-is, relocate) | S |
| 3.14 Port hyperparameter scheduler | `src/core/training/scheduler.py` | `orchestrator/scheduler.py` (reuse as-is) | S |
| 3.15 Port checkpoint manager | `src/core/training/checkpoint.py` | `orchestrator/checkpointer.py` (reuse as-is) | S |
| 3.16 Write MCTS unit tests | `tests/unit/core/mcts/test_*.py` (6 files) | New tests for decomposed MCTS | L |
| 3.17 Write network unit tests | `tests/unit/core/networks/test_*.py` (4 files) | Adapt from existing `tests/unit/test_network.py` | M |
| 3.18 Write MCTS property-based tests | `tests/property/test_mcts_invariants.py` | New — visit count consistency, Q-value bounds, determinism | M |

**Deliverable:** MCTS engine passes all property-based invariant tests. Networks forward pass produces correct shapes for any valid config. Coverage ≥ 85% on `src/core/mcts/` and `src/core/networks/`.

---

### Phase 4: Training Infrastructure
**Goal:** Build the training loop components: self-play, learner, evaluator, replay buffer.
**Dependencies:** Phase 3

| Task | Files to Create/Modify | Source (Existing) | Effort |
|------|----------------------|-------------------|--------|
| 4.1 Build replay buffer (memory-mapped) | `src/core/training/replay_buffer.py` | `distributed/shared_storage.py` (DistributedReplayBuffer) + `storage/replay_buffer.py` | M |
| 4.2 Build self-play actor | `src/core/training/self_play.py` | `distributed/actor.py` (DI refactor: accept env_factory, mcts_factory, network_factory) | M |
| 4.3 Build learner | `src/core/training/learner.py` | `distributed/learner.py` (DI refactor: accept network, loss_fn, optimizer) | M |
| 4.4 Build evaluator (Elo + baselines) | `src/core/training/evaluator.py` | `evaluation/` module (extract generic Elo calculation + baseline interface) | M |
| 4.5 Build curriculum scheduler | `src/core/training/curriculum.py` | `training/curriculum_pipeline.py` (generalize tier names) | M |
| 4.6 Build storage backends | `src/core/training/storage.py` | `storage/backend.py` + `storage/local_backend.py` + `storage/factory.py` (merge) | S |
| 4.7 Write training unit tests | `tests/unit/core/training/test_*.py` (6 files) | Adapt from existing `tests/unit/test_replay_buffer.py` etc. | L |
| 4.8 Write replay buffer property tests | `tests/property/test_replay_buffer.py` | New — FIFO ordering, capacity limits | M |

**Deliverable:** Self-play actor can run 1 game in any registered environment, learner can train 1 step on collected data, evaluator reports Elo. All pass tests.

---

### Phase 5: Environment System
**Goal:** Build the plugin system and first concrete environments.
**Dependencies:** Phase 2 (protocols), Phase 4 (training consumes environments)

| Task | Files to Create/Modify | Source (Existing) | Effort |
|------|----------------------|-------------------|--------|
| 5.1 Build environment registry | `src/environments/registry.py` | New — `@register_environment("name")` decorator + entry_points discovery | L |
| 5.2 Build environment base utilities | `src/environments/base.py` | New — observation padding, normalization wrappers, multi-agent turn wrapper | M |
| 5.3 Port MicroCiv as Civ6 domain adapter | `src/environments/civ6/environment.py`, `encoder.py`, `action_decoder.py` | `envs/microciv.py`, `envs/civ6_encoder.py`, `core/civ6_actions.py` | M |
| 5.4 Build mini-game: resource management | `src/environments/mini_games/resource_management.py` | New — simple resource allocation game (grid-based) | M |
| 5.5 Build mini-game: territory control | `src/environments/mini_games/territory_control.py` | New — territory capture game (simplified from MicroCiv) | M |
| 5.6 Build mini-game: tactical combat | `src/environments/mini_games/tactical_combat.py` | New — unit combat game | M |
| 5.7 Create environment config templates | `configs/environment/*.yaml`, `configs/environment/_template.yaml` | New YAML configs per environment | S |
| 5.8 Write environment tests | `tests/unit/environments/test_registry.py`, `test_mini_games.py` | New + adapt from `tests/unit/test_environment.py` | M |
| 5.9 Write protocol compliance integration test | `tests/integration/test_environment_plugins.py` | New — auto-validates all registered envs against StrategicEnvironment | M |
| 5.10 Create stub environments for expansion domains | `src/environments/logistics/`, `energy/`, `climate_policy/` | New — skeleton `__init__.py` + config YAML only | S |

**Deliverable:** Registry works. MicroCiv passes protocol compliance. 3 mini-games work independently. New environment can be added with zero core modifications.

---

### Phase 6: Orchestration and Integration
**Goal:** Wire everything together into the unified training pipeline and CLI.
**Dependencies:** Phases 3, 4, 5

| Task | Files to Create/Modify | Source (Existing) | Effort |
|------|----------------------|-------------------|--------|
| 6.1 Build training orchestrator | `src/core/training/orchestrator.py` | `orchestrator/orchestrator.py` (generalize with registry-based component creation) | L |
| 6.2 Build generic training pipeline | `src/core/training/pipeline.py` | `training_civ6.py` (refactor into class accepting domain components) | M |
| 6.3 Build Click-based CLI | `src/cli.py` | New — commands: `train`, `evaluate`, `explain`, `health-check`, `benchmark` | M |
| 6.4 Build experiment tracking abstraction | `src/adapters/experiment_tracker.py` | New — unified interface for W&B, TensorBoard, MLflow | S |
| 6.5 Port RAG integration | `src/adapters/rag_adapter.py` | `orchestrator/rag_integration.py` (rename strategy→knowledge) | S |
| 6.6 Port strategy advisor | `src/adapters/advisor.py` | `orchestrator/strategy_advisor.py` (pluggable prompt templates) | M |
| 6.7 Port LLM client | `src/adapters/llm_client.py` | `utils/llm_client.py` (reuse as-is) | S |
| 6.8 Write integration tests | `tests/integration/test_training_loop.py`, `test_curriculum.py`, `test_checkpoint_cycle.py` | New | L |
| 6.9 Write smoke tests | `tests/smoke/test_health_check.py` | New | S |

**Deliverable:** `python -m src.cli health-check` passes. `python -m src.cli train --env=mini_resource --iterations=10 --smoke` completes without error. Full training loop: self-play → train → eval → checkpoint works end-to-end.

---

### Phase 7: Documentation, Polish, and Packaging
**Goal:** Production-ready SDK with docs, CI, and clean public API.
**Dependencies:** All previous phases

| Task | Files to Create/Modify | Source (Existing) | Effort |
|------|----------------------|-------------------|--------|
| 7.1 Write C4 architecture diagrams | `docs/architecture/*.mermaid` | Spec Section 4 (adapt to actual implementation) | M |
| 7.2 Write "Creating Environments" guide | `docs/environments/creating_environments.md` | New | M |
| 7.3 Write "Quick Start" guide | `docs/training/quick_start.md` | New | S |
| 7.4 Write environment catalog | `docs/environments/environment_catalog.md` | New | S |
| 7.5 Set up pre-commit hooks | `.pre-commit-config.yaml` | New — ruff, mypy, pytest-smoke | S |
| 7.6 Finalize Makefile | `Makefile` | Spec Appendix A | S |
| 7.7 Finalize pyproject.toml | `pyproject.toml` | Spec Appendix B | S |
| 7.8 Create SDK CLAUDE.md | `CLAUDE.md` (update) | Spec Section 10.1 | S |
| 7.9 Full verification sweep | All files | Run lint, type check, full tests, coverage, smoke | M |
| 7.10 Clean public API surface | `src/__init__.py` | Explicit `__all__` exports | S |

**Deliverable:** All success criteria from spec Section 1.2 verified. Coverage ≥ 85%. Documentation renders correctly. SDK installable via `pip install -e ".[dev]"`.

---

## 5. Dependency Management

### 5.1 New Dependencies Required

| Dependency | Version | Purpose | Conflict Risk |
|------------|---------|---------|---------------|
| `pydantic>=2.0` | 2.x | Config validation, serialization | **Low** — coexists with existing dataclasses; adapters bridge the gap |
| `hydra-core>=1.3` | 1.3+ | Hierarchical config composition | **Medium** — replaces manual YAML loading; existing `from_yaml()` patterns need wrapping |
| `structlog>=23.0` | 23.x | Structured logging | **Medium** — existing `logging.getLogger()` calls continue working; structlog processes stdlib records |
| `rich>=13.0` | 13.x | Terminal UI, progress bars | **None** — purely additive |
| `click>=8.1` | 8.x | CLI framework | **Low** — replaces `argparse` in new code only; existing entry points unchanged |
| `hypothesis>=6.80` | 6.80+ | Property-based testing (dev only) | **None** — test dependency only |

### 5.2 Migration Strategy for Conflicting Dependencies

**pydantic vs dataclasses:**
- SDK config uses Pydantic models exclusively
- Existing alpha_zero dataclass configs remain unchanged
- Bridge: `SDKConfig.from_legacy(orchestrator_config)` adapter method
- Gradual migration: new code uses Pydantic, old code untouched

**hydra-core vs manual YAML:**
- SDK uses Hydra for its own config pipeline
- Existing `OrchestratorConfig.from_yaml()` still works
- Bridge: `HydraConfigAdapter` loads Hydra-composed config into legacy dataclasses
- Entry point `src/cli.py` uses Hydra; existing `__main__.py` unchanged

**structlog vs stdlib logging:**
- Configure structlog as a processor for stdlib `logging` module
- Existing `logging.getLogger(__name__)` calls produce structlog-formatted output
- SDK modules use `structlog.get_logger()` for context binding
- Zero changes to existing code required

---

## 6. Risk Analysis

### 6.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MCTS decomposition introduces bugs | Medium | High | Property-based tests (Hypothesis) verify invariants; run existing MCTS tests against refactored code |
| Hydra config conflicts with existing config system | Medium | Medium | Separate config namespaces; SDK uses `configs/` while existing code uses `configs/civ6_config.yaml` |
| Circular imports between `config` and `registry` | Medium | Medium | String-based references in config → lazy resolution at runtime |
| Performance regression from protocol indirection | Low | Medium | Benchmark critical paths (MCTS search, network forward pass); torch.compile mitigates |
| structlog migration breaks existing log parsing | Low | Low | structlog wraps stdlib logging transparently; existing formatters still work |
| Pydantic v2 serialization differs from dataclass `to_dict()` | Medium | Low | Adapter layer normalizes between the two |

### 6.2 Scope Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Mini-game environments take too long to build | Medium | Medium | Start with 1 game (resource management); other 2 can follow. MicroCiv adapter provides immediate testability |
| Domain expansion stubs create scope creep | High | Low | Stubs are just `__init__.py` + config YAML — no implementations |
| Explainability module harder than expected | Medium | Medium | Phase 7 — deprioritize if behind schedule; core training loop takes priority |
| Full test coverage target (85%) difficult to hit | Medium | Low | Focus coverage on core/mcts/ and core/training/ — adapters can have lower coverage |

### 6.3 Architectural Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Protocol-based design too flexible (hard to debug) | Low | Medium | `@runtime_checkable` enables `isinstance()` checks; test helpers validate protocol compliance |
| Registry pattern adds runtime complexity | Low | Low | Fail-fast: registry validates at registration time, not at use time |
| Two config systems (Pydantic + legacy dataclass) confuse contributors | Medium | Medium | Document clearly in CLAUDE.md; migration guide in docs/ |

---

## 7. Testing Strategy

### 7.1 Test Pyramid

```
        ╱╲
       ╱  ╲
      ╱Smoke╲       ← 5 tests: CLI health check, import validation
     ╱────────╲
    ╱Integration╲   ← 10 tests: training loop, curriculum, checkpoint cycle
   ╱──────────────╲
  ╱   Property      ╲ ← 15 tests: MCTS invariants, action space, replay buffer
 ╱────────────────────╲
╱      Unit Tests       ╲ ← 60+ tests: every public method, every protocol
╱──────────────────────────╲

Estimated total: ~90 tests
Target coverage: ≥ 85% on src/core/
```

### 7.2 Test Categories

| Category | Location | Runner | Purpose |
|----------|----------|--------|---------|
| Unit | `tests/unit/` | `pytest tests/unit -v --tb=short -x` | Isolated, fast, mocked deps |
| Property | `tests/property/` | `pytest tests/property -v` | Invariant verification via Hypothesis |
| Integration | `tests/integration/` | `pytest tests/integration -v` | Multi-component workflows |
| Smoke | `tests/smoke/` | `pytest tests/smoke -v` | Quick CI validation |

### 7.3 Key Property-Based Tests

| Test | Invariant |
|------|-----------|
| `test_visit_count_consistency` | `sum(child.visits) == root.visits` after any search |
| `test_q_values_bounded` | `-1 ≤ Q(s,a) ≤ 1` after any sequence of backpropagations |
| `test_search_deterministic` | Same seed + same state → same result |
| `test_valid_actions_nonempty` | Non-terminal states always have ≥1 valid action |
| `test_all_valid_actions_accepted` | Every action from `get_valid_actions()` is accepted by `step()` |
| `test_buffer_never_exceeds_capacity` | `len(buffer) ≤ max_size` after any insertion sequence |
| `test_serialization_round_trip` | `deserialize(serialize(state)) == state` for any state |

### 7.4 Protocol Compliance Test Framework

```python
# tests/conftest.py — auto-validates all registered environments
def assert_valid_environment(env: StrategicEnvironment) -> None:
    """Validate that env satisfies all StrategicEnvironment postconditions."""
    assert isinstance(env, StrategicEnvironment)
    obs = env.reset(seed=42)
    assert obs.shape == env.observation_spec.full_shape
    actions = env.get_valid_actions()
    assert len(actions) >= 1
    assert not env.is_terminal()
    for action in actions[:3]:  # test first 3 valid actions
        obs2, reward, done, trunc, info = env.step(action)
        assert -1.0 <= reward <= 1.0
        assert obs2.shape == env.observation_spec.full_shape
        env.reset(seed=42)  # reset for next action test
```

---

## 8. Migration Path

### 8.1 Coexistence Strategy

The SDK (`src/`) and existing code (`alpha_zero/`) coexist in the same repository:

```
Civ6-AlphaZero/
├── alpha_zero/          # Existing code — UNTOUCHED during SDK development
├── src/                 # New SDK code
├── tests/               # SDK tests (separate from alpha_zero tests)
│   └── (existing tests remain in tests/unit/, tests/integration/)
├── configs/             # SDK configs (new YAML files alongside existing)
└── pyproject.toml       # Updated with SDK entry points
```

**Rules:**
1. **No modifications to `alpha_zero/` during Phases 1-5.** The SDK imports patterns but does not modify existing code.
2. **Phase 6 (Civ6 domain adapter)** creates a thin wrapper in `src/environments/civ6/` that imports from `alpha_zero/`.
3. **Eventually** (post-SDK-v1.0), the monolithic `alpha_zero/` training pipeline can be replaced by the SDK pipeline — but this is optional and not in scope for the initial plan.

### 8.2 Incremental Verification

After each phase, run:
```bash
# SDK tests
pytest tests/unit -v --tb=short -x           # Phase-specific unit tests
mypy src/ --strict                           # Type check SDK code
ruff check src/ tests/                       # Lint SDK code

# Existing tests (verify no regression)
pytest tests/ -v --tb=line -k "not sdk"      # Existing alpha_zero tests still pass
```

### 8.3 Adapter Bridging

For components that need to bridge SDK ↔ existing code:

```python
# Example: Civ6 domain adapter bridges MicroCivEnv → StrategicEnvironment
class MicroCivAdapter:
    """Adapts alpha_zero.envs.microciv.MicroCivEnv to SDK StrategicEnvironment."""

    def __init__(self, config: MicroCivConfig):
        from alpha_zero.envs.microciv import MicroCivEnv  # lazy import
        self._env = MicroCivEnv(board_size=config.board_size)

    def reset(self, *, seed: int | None = None) -> NDArray[np.float32]:
        obs, _ = self._env.reset(seed=seed)
        return obs["spatial"]  # adapt observation format

    # ... implement full StrategicEnvironment protocol
```

---

## 9. Success Criteria Verification

### 9.1 Verification Commands (from spec Section 1.2)

| Criterion | Command | Phase |
|-----------|---------|-------|
| All unit tests pass | `pytest tests/unit -v --tb=short` | Phase 3+ |
| All integration tests pass | `pytest tests/integration -v` | Phase 6 |
| All property tests pass | `pytest tests/property -v` | Phase 3 |
| Type checking passes | `mypy src/ --strict` | Every phase |
| Lint passes | `ruff check src/ tests/` | Every phase |
| Coverage ≥ 85% on core | `pytest --cov=src/core --cov-report=term-missing` | Phase 6 |
| Health check smoke test | `python -m src.cli health-check` | Phase 6 |
| Mini-game trains 10 iterations | `python -m src.cli train --env=mini_resource --iterations=10 --smoke` | Phase 6 |
| C4 diagrams render | Manual verification in Mermaid viewer | Phase 7 |
| Zero hardcoded values | `grep -rn "hardcoded\|magic\|TODO.*hardcode" src/` returns nothing | Every phase |
| DEBUG logging actionable | Run with `--log-level=DEBUG`, verify trace | Phase 6 |
| Plugin interface validated | Register new env, train with zero core changes | Phase 5 |

### 9.2 Definition of Done Per Phase

**Phase complete when:**
1. All phase-specific tests pass
2. `mypy src/ --strict` exits 0
3. `ruff check src/ tests/` exits 0
4. Coverage ≥ 85% on newly added modules
5. No hardcoded values in any `src/` file
6. CLAUDE.md updated with new commands and decisions

---

## Appendix A: Effort Estimation Key

| Size | Description | Approximate Scope |
|------|-------------|-------------------|
| S | Small | < 100 lines, straightforward, < 1 file |
| M | Medium | 100-300 lines, moderate complexity, 1-3 files |
| L | Large | 300-600 lines, significant logic, 3-5 files |
| XL | Extra Large | 600+ lines, cross-cutting, > 5 files |

## Appendix B: File Count Summary

| Phase | New Files | Modified Files | Tests |
|-------|-----------|---------------|-------|
| Phase 1 | 7 | 1 (pyproject.toml) | 3 |
| Phase 2 | 3 | 2 | 2 |
| Phase 3 | 14 | 0 | 12 |
| Phase 4 | 7 | 0 | 8 |
| Phase 5 | 12 | 0 | 4 |
| Phase 6 | 8 | 0 | 4 |
| Phase 7 | 8 | 2 | 0 |
| **Total** | **59** | **5** | **33** |

## Appendix C: Critical Path

The minimum viable SDK (MVP) that demonstrates the core value proposition:

```
Phase 1 (Foundation) → Phase 2 (Protocols) → Phase 3 (MCTS + Networks)
    → Phase 4 (Training) → Phase 5.1-5.3 (Registry + MicroCiv adapter)
    → Phase 6.1-6.2 (Orchestrator + Pipeline)
```

This produces: a working SDK that can train MicroCiv using the generalized engine, proving the architecture works. Mini-games (Phase 5.4-5.6), CLI (Phase 6.3), and documentation (Phase 7) can follow.
