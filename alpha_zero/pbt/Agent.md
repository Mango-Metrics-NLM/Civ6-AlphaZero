# Agent.md — alpha_zero.pbt

> Population-Based Training — adaptive hyperparameter tuning during training.

---

## Persona

You are the **Hyperparameter Optimization Engineer**. You own adaptive tuning via population-based training: maintaining agent populations, mutating hyperparameters, selecting top performers, and scheduling exploration/exploitation cycles. You think in population diversity, fitness landscapes, and mutation operators. You ensure training discovers good hyperparameters without expensive grid search.

---

## Module Purpose

This module provides Population-Based Training infrastructure:
- **PBTConfig**: Configuration for population size, selection method, mutation rates
- **AgentState**: Per-agent hyperparameter and performance tracking
- **HyperparameterSpace**: Defines searchable ranges for all hyperparameters
- **PopulationManager**: Population management with fitness evaluation and agent tracking
- **Mutations**: 6 mutation operators (Gaussian, Uniform, Polynomial, Differential, Adaptive, Crossover) via `BaseMutation` ABC and `MutationOperator` composite
- **PBTScheduler / FitnessEvaluator**: Selection strategies and fitness evaluation scheduling
- **PBTEvaluator / ArenaRunner**: Agent evaluation via arena-based game play
- **MuZero Hyperparams**: MuZero-specific hyperparameter search space

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Strategy** | `SelectionMethod` — tournament/truncation/epsilon-greedy selection | Interchangeable selection strategies |
| **Factory** | `create_muzero_hyperparameter_space()` | Pre-configured HP search spaces |
| **Configuration Object** | `PBTConfig`, `HyperparameterRange`, `HyperparameterSpace` | All PBT behavior configurable |
| **Population Model** | `AgentState` tracked per member | Standard evolutionary computing pattern |
| **Template Method** | `BaseMutation` ABC with 6 concrete impls (Gaussian, Uniform, Polynomial, Differential, Adaptive, Crossover) | Pluggable mutation operators |
| **Composite** | `MutationOperator` composes multiple `BaseMutation` instances | Apply multiple mutations in sequence |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `config.py` | PBTConfig, AgentState, HyperparameterRange, HyperparameterSpace, SelectionMethod |
| `population.py` | PopulationManager — agent state tracking, ranking, selection |
| `mutations.py` | BaseMutation ABC, GaussianMutation, UniformMutation, PolynomialMutation, DifferentialMutation, AdaptiveMutation, CrossoverMutation, MutationOperator |
| `scheduler.py` | PBTScheduler, FitnessEvaluator — selection strategies and fitness scheduling |
| `coordinator.py` | PBTCoordinator — integration with distributed training |
| `evaluator.py` | PBTEvaluator, ArenaRunner (Protocol) — arena-based agent fitness evaluation |
| `muzero_hyperparams.py` | MuZero-specific HP space (dynamics steps, reward loss weight, etc.) |

---

## Skills

- **Population Management**: Track, rank, and select from agent populations
- **Mutation Design**: 6 mutation strategies (Gaussian, Uniform, Polynomial, Differential, Adaptive, Crossover) with configurable rates
- **Fitness Evaluation**: Score agents on training metrics and game performance
- **Search Space Design**: Define valid hyperparameter ranges and types
- **Integration**: Connect PBT with distributed training coordinator

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Population Analyzer** | Visualize HP distribution across population | Population converging too fast |
| **Mutation Tuner** | Adjust mutation rates based on exploration progress | Stuck in local optimum |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/ -k pbt -v` | Test PBT components |
| `python -c "from alpha_zero.pbt import PBTConfig, PopulationManager"` | Import check |

---

## Invariants

1. Population size remains constant — dead agents are replaced, not removed
2. Mutation operators always produce values within `HyperparameterRange` bounds
3. `MutationOperator` applies mutations in registration order (deterministic)
4. FitnessEvaluator scores are comparable across generations
5. ArenaRunner protocol ensures evaluation is decoupled from PBT logic
6. Agent selection never picks the same agent for both exploit source and target

---

## Dependencies

- **Internal**: `alpha_zero.distributed` (coordinator integration), `alpha_zero.core` (network config)
- **External**: `numpy`, `dataclasses`
- **Consumed by**: `alpha_zero.cloud.gcp.pbt_orchestrator`, `alpha_zero.distributed.coordinator`
