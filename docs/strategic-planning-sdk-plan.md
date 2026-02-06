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
10. [Spec Gap Analysis](#10-spec-gap-analysis)
11. [Protocol Contracts and Invariants](#11-protocol-contracts-and-invariants)
12. [Logging, Debugging, and Observability](#12-logging-debugging-and-observability)
13. [Domain Expansion Roadmap](#13-domain-expansion-roadmap)
14. [Scaffolding Templates](#14-scaffolding-templates)
15. [Execution Protocol](#15-execution-protocol)
16. [Permission Architecture](#16-permission-architecture)

**Companion Document:** [SDK Spec Reference](sdk-spec-reference.md) — Full agentic coding system prompt template with C4 diagrams, protocol definitions, config models, and appendices.

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

---

## 10. Spec Gap Analysis

Cross-reference of the full [SDK Spec Reference](sdk-spec-reference.md) against this implementation plan. Every spec section maps to one or more plan phases, with gaps identified and addressed.

### 10.1 Coverage Matrix

| Spec Section | Spec Content | Plan Coverage | Gap? | Resolution |
|---|---|---|---|---|
| **S1.1** System Intent | SDK mission statement | Section 1 (Executive Summary) | No | |
| **S1.2** Success Criteria | 12 mechanically verifiable criteria | Section 9 (Verification Commands) | **Partial** — missing `ruff format --check` | Added to Phase 1 verification (see 10.2) |
| **S1.3** Problem Description | 3-paragraph system description | Section 1 (Key Insight) | **Partial** — plan focused on reuse, not failure modes | Added to Section 11 (Invariants) |
| **S2.1** Hard Constraints | Python 3.11+, PyTorch 2.1+, deps, security | Section 5 (Dependency Management) | **Gap** — Python 3.11+ vs existing 3.10+ requirement | Resolution: SDK targets 3.11+ while `alpha_zero/` stays 3.10+ (see 10.2) |
| **S2.2** Soft Constraints | Code style, architecture, performance, testing, logging | Sections 7, 12 | **Partial** — torch.compile, vectorized MCTS, lazy imports not explicit | Added to Phase 3 tasks (see 10.2) |
| **S2.3** Anti-Constraints | Explicit freedoms | N/A (design principle) | No | Captured in plan Section 2.3 (Key Design Decisions) |
| **S3** Permission Architecture | Scope, autonomy, resource budget | Not in original plan | **Gap** | Added as Section 16 |
| **S4** C4 Architecture | 4-level Mermaid diagrams | Phase 7 task 7.1 | **Partial** — plan defers to Phase 7, but diagrams needed for Phase 2 design | Added as Phase 2 pre-req (see 10.2) |
| **S5** Project Structure | Full directory layout | Section 2.1 | No | Plan matches spec |
| **S6.1** StrategicEnvironment Protocol | 10-method protocol with postconditions | Phase 2 task 2.2 | **Partial** — plan lists protocol but not method contracts | Added to Section 11 |
| **S6.2** Config Models | 8 Pydantic models (SDKConfig tree) | Phase 1 task 1.3 | **Partial** — plan references but doesn't specify all fields | Added to Section 14 (Templates) |
| **S7** Logging/Debugging | structlog, correlation IDs, debug utilities, metrics | Phase 1 task 1.4 | **Gap** — plan only covers structlog setup, not debug utilities or metrics | Added as Section 12 |
| **S8** Testing Strategy | conftest.py, property tests, integration tests | Section 7 | **Partial** — plan has pyramid but not fixture code or test examples | Section 7 expanded with code examples in spec ref |
| **S9** Feedback Loop | Verification commands, error handling protocol | Section 9 | No | Plan matches spec |
| **S10** Context Persistence | CLAUDE.md template, info to preserve | Phase 7 task 7.8 | **Partial** — template not included inline | Added to Section 14 |
| **S11** Execution Protocol | Initial actions, implementation order, completion checklist | Phases 1-7 DoD | **Gap** — per-task workflow not specified | Added as Section 15 |
| **S12** Domain Expansion | 7-domain mapping, reusable matrix, extension points | Phase 5 task 5.10 | **Gap** — plan only has 3 stub domains, spec has 7 with full mapping | Added as Section 13 |
| **App A** Makefile | Complete Makefile template | Phase 7 task 7.6 | **Partial** — referenced but not included | Added to Section 14 |
| **App B** pyproject.toml | Complete template | Phase 7 task 7.7 | **Partial** — referenced but not included | Added to Section 14 |

### 10.2 Gap Resolutions

**Python 3.11+ Requirement:**
The spec requires Python 3.11+ for `TaskGroup` and `ExceptionGroup`. The existing `alpha_zero/` package targets 3.10+. Resolution:
- `src/` (SDK) targets Python 3.11+ and uses `TaskGroup` for concurrent self-play coordination
- `alpha_zero/` remains 3.10+ compatible
- `pyproject.toml` declares `requires-python = ">=3.11"` for the SDK package
- CI matrix tests both 3.11 and 3.12

**Missing Verification Command:**
Add `ruff format --check src/ tests/` to Phase 1+ verification. Update Phase DoD:
```bash
# Updated verification sequence (all phases)
ruff check src/ tests/                       # 1. Lint
ruff format --check src/ tests/              # 2. Format check (NEW)
mypy src/ --strict                           # 3. Type check
pytest tests/unit -v --tb=short -x           # 4. Unit tests
```

**Performance Constraints (torch.compile, vectorized MCTS):**
Add explicit tasks to Phase 3:
- Task 3.19: Add `torch.compile()` wrapper in `src/core/networks/model.py` with config toggle (`compile_model: bool`)
- Task 3.20: Implement batch MCTS evaluation in `src/core/mcts/evaluation.py` (batch multiple leaf nodes for single network forward pass)
- Task 3.21: Add lazy import guards in `src/core/networks/__init__.py` for optional backends (transformer requires `einops`, GNN requires `torch_geometric`)

**C4 Diagrams Timing:**
Phase 2 should draft component-level C4 diagrams (Level 3) before implementing protocols, not defer to Phase 7. Add task:
- Task 2.0: Draft `docs/architecture/c4_component.mermaid` based on spec Section 4.3 to guide protocol design

---

## 11. Protocol Contracts and Invariants

Derived from spec Sections 1.3 (failure modes) and 6.1 (StrategicEnvironment). These contracts are enforced by property-based tests and protocol compliance tests.

### 11.1 StrategicEnvironment Method Contracts

| Method | Preconditions | Postconditions | Enforced By |
|--------|--------------|----------------|-------------|
| `reset(seed)` | None | `get_valid_actions() >= 1`, `not is_terminal()`, obs matches `observation_spec.full_shape` | `test_environment_plugins.py` |
| `step(action)` | `action in get_valid_actions()`, `not is_terminal()` | `reward in [-1.0, 1.0]`, obs matches spec, if `terminated or truncated` then no further `step()` valid | `test_environment_plugins.py` |
| `get_valid_actions()` | `not is_terminal()` | `len(result) >= 1`, every action accepted by `step()` without `InvalidActionError` | `test_action_space.py` (property) |
| `serialize()` | Any state | `deserialize(serialize()) == state` (deterministic) | `test_mcts_invariants.py` (property) |
| `get_symmetries(obs, policy)` | Valid obs and policy | Result includes original `(obs, policy)` pair | `test_environment_plugins.py` |

### 11.2 MCTS Invariants

| Invariant | Definition | Property Test |
|-----------|-----------|---------------|
| Visit count consistency | `sum(child.visit_count for child in root.children) == root.visit_count` | `test_visit_count_consistency` |
| Q-value bounds | `-1.0 <= Q(s, a) <= 1.0` for all nodes after any backpropagation sequence | `test_q_values_bounded` |
| Deterministic search | Same seed + same state + same network → identical action selection | `test_search_deterministic_with_seed` |
| Virtual loss correctness | Virtual loss is added during selection and removed after backup | `test_virtual_loss_lifecycle` |
| Temperature convergence | As temperature → 0, action selection converges to argmax(visit_count) | `test_temperature_greedy_limit` |

### 11.3 Training Loop Invariants

| Invariant | Definition | Monitoring |
|-----------|-----------|------------|
| No hardcoded values | Every parameter sourced from config | `grep -rn` scan in CI + Pydantic validation |
| Replay buffer bounded | `len(buffer) <= config.replay_buffer_size` at all times | Property test + runtime assertion |
| Checkpoint resumability | `load(save(state)) == state` for network + optimizer + buffer | Integration test |
| Loss monotonicity (windowed) | 100-step moving average of total loss is non-increasing | WARNING alert if violated |
| Weight consistency | All actors use same network version within tolerance window | Actor heartbeat + version tracking |

### 11.4 Failure Mode Handling

| Failure Mode | Detection | Recovery | Spec Reference |
|---|---|---|---|
| Actor crash | Process monitor heartbeat timeout | Restart actor, replay last trajectory from WAL log | S1.3 P3 |
| Network divergence | Auxiliary loss spike > 3 sigma | Auto-rollback to last stable checkpoint | S1.3 P3 |
| Environment deadlock | Step timeout (configurable) | Force terminate + penalize game | S1.3 P3 |
| Replay buffer corruption | Checksum mismatch on read | Discard corrupted entry, log WARNING | S1.3 P3 |
| Curriculum regression | Elo drops below demotion threshold | Demote to previous stage (if `demotion_enabled`) | S1.3 P3 |

---

## 12. Logging, Debugging, and Observability

Derived from spec Section 7. Implementation spans Phases 1 (foundation), 3 (MCTS debug), 4 (training metrics), and 6 (CLI integration).

### 12.1 Structured Logging Architecture

**Phase 1 deliverable:** `src/core/logging.py`

```python
# Design: structlog wrapping stdlib logging
# - SDK modules: structlog.get_logger("component.subcomponent")
# - Existing alpha_zero modules: logging.getLogger(__name__) still works
# - Output: JSON (production) or pretty-print (dev) via config

# Every log line automatically includes:
bound_context = {
    "timestamp": "ISO 8601",
    "level": "INFO",
    "correlation_id": "uuid4",   # For distributed tracing
    "component": "mcts.search",  # Logger name
    "environment_id": "mini_resource",  # Bound at orchestrator level
    "actor_id": 0,               # Bound at actor level
    "iteration": 42,             # Bound at training loop level
    "step": 1000,                # Bound at training loop level
}
```

**Implementation tasks added:**

| Task | Phase | File | Description |
|------|-------|------|-------------|
| 1.4a | 1 | `src/core/logging.py` | structlog setup with `configure_logging(config: LoggingConfig)` |
| 1.4b | 1 | `src/core/logging.py` | Correlation ID middleware: `bind_correlation_id()`, `get_correlation_id()` |
| 1.4c | 1 | `configs/logging.yaml` | Component-level log config (mcts: DEBUG, training: INFO, etc.) |

### 12.2 Debug Utilities

**Phase 3 deliverable:** `src/core/mcts/` debug hooks
**Phase 4 deliverable:** `src/core/training/` debug hooks

| Utility | Module | Phase | Activation |
|---------|--------|-------|------------|
| MCTS Tree Dump | `mcts/tree.py:to_json()` | 3 | `logging.level == DEBUG` and `config.logging.log_mcts_trees` |
| Action Mask Validator | `mcts/expansion.py:validate_mask()` | 3 | `logging.level == DEBUG` |
| Network Activation Maps | `networks/utils.py:register_hooks()` | 3 | `config.logging.log_activation_maps` |
| Gradient Flow Analyzer | `training/learner.py:log_gradient_norms()` | 4 | `config.logging.log_gradient_norms` |
| Replay Buffer Inspector | `training/replay_buffer.py:get_stats()` | 4 | Every `config.logging.metrics_interval` steps |
| Checkpoint Diff | `training/checkpoint.py:diff_checkpoints()` | 4 | On demand via CLI: `spsdk checkpoint diff ckpt1 ckpt2` |
| Environment State Snapshots | `environments/base.py:SnapshotWrapper` | 5 | `logging.level == DEBUG` |

### 12.3 Metrics Collection

**Phase 4 deliverable:** `src/core/training/metrics.py`

All metric names and collection intervals are configurable via `configs/logging.yaml`:

```yaml
metrics:
  training:
    interval: 10  # Log every N steps
    fields: [policy_loss, value_loss, total_loss, learning_rate, gradient_norm_mean, gradient_norm_max, replay_buffer_size]
  self_play:
    interval: 1   # Log every game
    fields: [game_length, outcome, total_reward, mean_mcts_depth, mean_branching_factor]
  evaluation:
    interval: 1   # Log every eval round
    fields: [elo_rating, win_rate_vs_baseline, draw_rate, mean_game_length]
  curriculum:
    interval: 1   # Log every promotion/demotion event
    fields: [current_stage, games_at_stage, elo_at_stage]
  system:
    interval: 60  # Log every N seconds
    fields: [gpu_utilization, gpu_memory_mb, cpu_utilization, throughput_games_per_hour]
```

### 12.4 Experiment Tracking Adapter Interface

**Phase 6 deliverable:** `src/adapters/experiment_tracker.py`

```python
class ExperimentTracker(Protocol):
    """Unified interface for W&B, TensorBoard, MLflow."""
    def log_scalar(self, name: str, value: float, step: int) -> None: ...
    def log_histogram(self, name: str, values: NDArray, step: int) -> None: ...
    def log_artifact(self, path: str, name: str) -> None: ...
    def log_config(self, config: dict[str, Any]) -> None: ...
    def finish(self) -> None: ...
```

---

## 13. Domain Expansion Roadmap

Derived from spec Section 12. Extends the plan's 3 stub domains to the full 7-domain mapping.

### 13.1 Full Domain Mapping (7 Domains)

| # | Domain | SDK Module | Civ6 Analogy | Action Hierarchy | Key Obs Channels | Phase |
|---|--------|-----------|--------------|-----------------|-----------------|-------|
| 1 | **AI Strategy (Logistics)** | `environments/logistics/` | Territory expansion → route optimization | Mode → Route → Load → Destination | Network graph, demand forecast, capacity, in-transit | 5 (stub) |
| 2 | **Decision Models (Trading)** | `environments/trading/` | Tech tree → portfolio construction | Asset → Instrument → Order → Size | Price series, positions, risk metrics, order book | Post-v1.0 |
| 3 | **Human-AI (Education)** | `environments/education/` | Strategy advisor → tutoring | Topic → Scenario → Decision → Rationale | Student state, concept mastery, curriculum progress | Post-v1.0 |
| 4 | **Resource Mgmt (Energy)** | `environments/energy/` | City resources → grid management | Source → Distribution → Storage → Pricing | Generation capacity, demand curve, storage, weather | 5 (stub) |
| 5 | **Simulations (Military)** | `environments/wargame/` | Unit combat → tactical planning | Unit → Movement → Engagement → Support | Terrain, force disposition, fog of war, logistics | Post-v1.0 |
| 6 | **Climate Policy** | `environments/climate_policy/` | Diplomacy → negotiation | Policy → Implementation → Diplomacy → Trade | Emissions, GDP, temperature, commitments | 5 (stub) |
| 7 | **AI Research (Benchmark)** | `environments/mini_games/` | Curriculum mini-games → benchmark suite | Varies by mini-game | Varies by mini-game | 5 (built) |

### 13.2 Phase 5 Scope (What Gets Built vs Stubbed)

| Environment | Phase 5 Scope | Files Created |
|-------------|-------------|---------------|
| `mini_games/resource_management` | **Full implementation** — grid-based resource allocation | environment.py, encoder.py, action_decoder.py |
| `mini_games/territory_control` | **Full implementation** — simplified territory capture | environment.py, encoder.py, action_decoder.py |
| `mini_games/tactical_combat` | **Full implementation** — unit combat | environment.py, encoder.py, action_decoder.py |
| `civ6/` | **Adapter** — wraps existing `alpha_zero/envs/microciv.py` | environment.py (adapter), encoder.py (delegate), action_decoder.py (delegate) |
| `logistics/` | **Stub** — `__init__.py` + config YAML only | `__init__.py`, `configs/environment/logistics.yaml` |
| `energy/` | **Stub** — `__init__.py` + config YAML only | `__init__.py`, `configs/environment/energy_grid.yaml` |
| `climate_policy/` | **Stub** — `__init__.py` + config YAML only | `__init__.py`, `configs/environment/climate_policy.yaml` |
| `trading/` | **Not created** — Post-v1.0 | — |
| `education/` | **Not created** — Post-v1.0 | — |
| `wargame/` | **Not created** — Post-v1.0 | — |

### 13.3 Reusable Component Matrix

Components that transfer across ALL domains without modification:

| Component | SDK Module | Why Reusable |
|-----------|-----------|-------------|
| MCTS Search Engine | `core/mcts/` | Only needs policy/value from network and valid actions from environment |
| Hierarchical Action Picker | `core/mcts/action_picker.py` | Temperature schedule works for any action distribution |
| Replay Buffer | `core/training/replay_buffer.py` | Stores (obs, policy, value) tuples regardless of domain |
| Elo Evaluator | `core/training/evaluator.py` | Domain-agnostic strength measurement |
| Curriculum Scheduler | `core/training/curriculum.py` | Elo-gated progression works for any environment sequence |
| Checkpoint Manager | `core/training/checkpoint.py` | Serializes any model + optimizer + buffer state |
| Explainability Tracer | `core/explain/tracer.py` | MCTS trees are domain-agnostic data structures |
| Training Orchestrator | `core/training/orchestrator.py` | Coordinates actors/learner/evaluator for any environment |
| Configuration Engine | `core/config.py` | Pydantic validation for any parameter set |
| Structured Logger | `core/logging.py` | Context-bound logging works everywhere |

### 13.4 Per-Domain Implementation Checklist

When a domain expert adds a new domain (e.g., `logistics`):

```
1. [ ] Create src/environments/logistics/__init__.py
2. [ ] Implement StrategicEnvironment in environment.py
       - reset(), step(), get_valid_actions(), get_observation()
       - serialize(), deserialize()
       - observation_spec, action_spec properties
3. [ ] Implement StateEncoder in encoder.py
       - raw state dict → (spatial_tensor, vector_tensor, action_mask)
4. [ ] Implement ActionDecoder in action_decoder.py
       - HierarchicalAction → domain-specific command
5. [ ] (Optional) Implement RewardShaper for intermediate rewards
6. [ ] (Optional) Implement get_symmetries() for data augmentation
7. [ ] Create configs/environment/logistics.yaml
8. [ ] Register in src/environments/registry.py
9. [ ] Write tests/unit/environments/test_logistics.py
10. [ ] Verify: pytest tests/integration/test_environment_plugins.py -k logistics
11. [ ] ZERO modifications to any file in src/core/
```

---

## 14. Scaffolding Templates

These are Phase 1 and Phase 7 deliverables — ready-to-use templates that codify the spec's requirements.

### 14.1 pyproject.toml (Phase 1 deliverable)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "strategic-planning-sdk"
version = "0.1.0"
description = "Domain-agnostic strategic planning SDK: hierarchical MCTS + neural networks"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.1",
    "numpy>=1.24",
    "pydantic>=2.0",
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "structlog>=23.0",
    "rich>=13.0",
    "click>=8.1",
    "gymnasium>=0.29",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "pytest-xdist>=3.3",
    "hypothesis>=6.80",
    "mypy>=1.5",
    "ruff>=0.1",
    "pre-commit>=3.4",
]
adapters = ["mlflow>=2.7", "tensorboard>=2.14"]
environments = []

[project.scripts]
spsdk = "src.cli:main"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM", "TCH"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "gpu: marks tests requiring GPU",
    "integration: marks integration tests",
]
addopts = "-ra --strict-markers"

[tool.coverage.run]
source = ["src"]
omit = ["src/adapters/*", "tests/*"]

[tool.coverage.report]
fail_under = 85
show_missing = true
```

### 14.2 Makefile (Phase 1 deliverable)

```makefile
.PHONY: install test lint format clean verify health-check train-smoke

SRC_DIR := src
TEST_DIR := tests
CONFIG_DIR := configs
COVERAGE_THRESHOLD := 85

install:
	pip install -e ".[dev]"
	pre-commit install

test: test-unit test-property test-integration

test-unit:
	pytest $(TEST_DIR)/unit -v --tb=short -x

test-property:
	pytest $(TEST_DIR)/property -v --tb=short

test-integration:
	pytest $(TEST_DIR)/integration -v --tb=short

test-smoke:
	pytest $(TEST_DIR)/smoke -v

coverage:
	pytest --cov=$(SRC_DIR)/core --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD)

lint:
	ruff check $(SRC_DIR) $(TEST_DIR)
	mypy $(SRC_DIR) --strict

format:
	ruff format $(SRC_DIR) $(TEST_DIR)

format-check:
	ruff format --check $(SRC_DIR) $(TEST_DIR)

verify: lint format-check test coverage

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

health-check:
	python -m $(SRC_DIR).cli health-check

train-smoke:
	python -m $(SRC_DIR).cli train --config-name=config training=smoke
```

### 14.3 SDK CLAUDE.md Template (Phase 7 deliverable)

```markdown
# CLAUDE.md — Strategic Planning SDK

## Build Commands
| Command | Purpose |
|---------|---------|
| `pip install -e ".[dev]"` | Install with dev dependencies |
| `make install` | Full install with pre-commit hooks |

## Test Commands
| Command | Purpose |
|---------|---------|
| `make test` | Full test suite |
| `make test-unit` | Unit tests only (fast) |
| `make test-property` | Property-based tests |
| `make test-integration` | Integration tests |
| `make test-smoke` | Smoke tests (CI) |
| `make coverage` | Coverage report |

## Lint/Format Commands
| Command | Purpose |
|---------|---------|
| `make lint` | Ruff check + mypy --strict |
| `make format` | Ruff format |
| `make format-check` | Verify formatting without changes |
| `make verify` | Full lint + format + test + coverage |

## Training Commands
| Command | Purpose |
|---------|---------|
| `python -m src.cli train --config-name=config` | Default training |
| `python -m src.cli train --env=mini_resource --iterations=10` | Mini-game |
| `python -m src.cli train training=smoke` | Smoke test (1 iteration) |
| `python -m src.cli evaluate --checkpoint=path/to/ckpt` | Evaluate |
| `python -m src.cli explain --checkpoint=path/to/ckpt --env=mini_resource` | Decision trace |
| `python -m src.cli health-check` | Validate installation |

## Architecture Decisions
| Date | Decision | Rationale |
|------|----------|-----------|
| | Protocol-based environment plugins | Structural subtyping, zero inheritance |
| | Hydra + Pydantic for config | Composition + validation, zero hardcoded values |
| | structlog for logging | JSON production, pretty-print dev, context binding |
| | Memory-mapped replay buffer | Avoids OOM on large training runs |

## Environment Plugin Checklist
1. Create directory: `src/environments/{name}/`
2. Implement `StrategicEnvironment` protocol in `environment.py`
3. Implement `StateEncoder` in `encoder.py`
4. Implement action decoder in `action_decoder.py`
5. Add config: `configs/environment/{name}.yaml`
6. Register in `src/environments/registry.py`
7. Add tests: `tests/unit/environments/test_{name}.py`
8. Verify: `pytest tests/integration/test_environment_plugins.py -k {name}`
```

### 14.4 Hydra Root Config Template (Phase 1 deliverable)

```yaml
# configs/config.yaml — Root configuration
# Composes all config groups. Override via CLI:
#   python -m src.cli train network=resnet_large mcts=gumbel_muzero

defaults:
  - environment: mini_resource
  - network: resnet_small
  - mcts: alphazero
  - training: default
  - curriculum: default
  - _self_

seed: 42
project_name: "strategic-planning-sdk"
output_dir: "./outputs"
```

### 14.5 Environment Config Template (Phase 5 deliverable)

```yaml
# configs/environment/_template.yaml
# Copy this file to create a new environment config.
# All values must be explicitly set — no defaults are assumed.

name: "my_environment"
description: "Description of this environment"

observation:
  spatial_channels: 13        # Number of spatial feature planes
  spatial_height: 16          # Board/grid height
  spatial_width: 16           # Board/grid width
  vector_dim: 64              # Global feature vector dimension
  temporal_depth: 1           # Number of historical observations

action_space:
  type: "hierarchical"        # flat | hierarchical
  hierarchy_levels: 4         # Category -> Selector -> Subtype -> Target
  total_actions: 356          # Flat action space size
  categories: []              # List of action category names

game:
  max_players: 2
  max_steps: 200
  reward_range: [-1.0, 1.0]
  symmetries: "none"          # none | rotational | reflective | dihedral
```

---

## 15. Execution Protocol

Per-task implementation workflow for agents working on any phase. Derived from spec Section 11.

### 15.1 Per-Task Workflow

```
For each task in the phase table:

1. READ: Read all source files listed in "Source (Existing)" column
   - Understand patterns, data structures, function signatures
   - Note any domain-specific logic that must be abstracted

2. CONFIGURE: If the task introduces new parameters:
   - Add Pydantic fields to src/core/config.py
   - Add YAML defaults to appropriate configs/ file
   - Verify: no magic numbers in implementation code

3. PROTOCOL: If the task defines or extends an interface:
   - Add to src/core/protocols.py with @runtime_checkable
   - Write Google-style docstring with Args, Returns, Raises, Examples
   - Add postconditions as inline comments

4. IMPLEMENT: Write the code
   - Import only from protocols and config, not concrete types
   - Add structlog calls at appropriate levels
   - Guard debug-only code with log level checks

5. TEST: Write corresponding test file
   - Mirror src/ structure in tests/unit/
   - Use fixtures from conftest.py, not hardcoded values
   - Include edge cases and error conditions

6. VERIFY: Run the verification sequence
   ruff check src/ tests/
   ruff format --check src/ tests/
   mypy src/ --strict
   pytest tests/unit/path/to/test_file.py -v

7. COMMIT: If all verification passes
   - Clear commit message describing what and why
   - Reference the plan task number (e.g., "Phase 3.7: MCTS search orchestration")
```

### 15.2 Definition of Done (Per Task)

```
A task is DONE when:
[ ] Implementation file exists at the specified SDK path
[ ] All public functions have type hints and Google-style docstrings
[ ] No hardcoded values — all parameters from config
[ ] structlog logging at appropriate levels
[ ] Corresponding test file exists and all tests pass
[ ] mypy --strict passes on the new file
[ ] ruff check passes on the new file
[ ] Coverage >= 85% on the new file
```

### 15.3 Definition of Done (Per Phase)

```
A phase is DONE when:
[ ] All tasks in the phase table are individually DONE
[ ] Full verification sequence passes (all 8 commands)
[ ] No regressions in existing alpha_zero tests
[ ] CLAUDE.md updated with new commands and decisions
[ ] Phase deliverable verified (stated at end of each phase section)
```

---

## 16. Permission Architecture

Guardrails for agents implementing the SDK. Derived from spec Section 3.

### 16.1 Scope

```
IN SCOPE (agent can create/modify):
  - All files in src/
  - All files in tests/ (SDK tests)
  - Configuration files in configs/
  - Documentation in docs/
  - pyproject.toml, Makefile, .pre-commit-config.yaml
  - CLAUDE.md

OUT OF SCOPE (agent must NOT modify):
  - alpha_zero/ (existing code — read-only reference)
  - .git/
  - .env (contains secrets)
  - checkpoints/ (trained model files)
  - data/ (user data)
```

### 16.2 Autonomy Levels

```
AUTONOMOUS (proceed without asking):
  - File creation within src/, tests/, configs/, docs/
  - Running tests, linters, type checkers
  - Refactoring for consistency within SDK code
  - Adding logging, error handling, type hints
  - Creating new test files
  - Installing dev dependencies

CONFIRM FIRST (ask human before proceeding):
  - Architectural changes affecting > 5 SDK modules
  - Breaking changes to StrategicEnvironment protocol
  - Deletions > 200 lines
  - Adding new required (non-optional) dependencies
  - Changes to training loop invariants
  - Modifications to checkpoint format

PROHIBITED:
  - Modifying alpha_zero/ source code
  - Hardcoding values that should be configurable
  - Writing code without tests
  - Using print() instead of structlog
  - Committing to main branch
  - Skipping existing tests without documented reason
  - Ignoring type errors or lint warnings
```

### 16.3 Resource Budget

| Constraint | Limit |
|-----------|-------|
| Max failed attempts before asking for guidance | 5 |
| Max files modified in single pass | 25 |
| Exploration time before asking | 10 min |
| Unit test suite runtime target | < 5 min |
| Full test suite runtime target | < 15 min |
