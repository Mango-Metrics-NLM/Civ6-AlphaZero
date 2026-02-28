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
- **Population**: Population management with fitness evaluation
- **Mutations**: Hyperparameter mutation operators (perturbation, resample)
- **Scheduler**: Selection strategies and tournament-based fitness
- **MuZero Hyperparams**: MuZero-specific hyperparameter search space

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Strategy** | `SelectionMethod` — tournament/truncation/epsilon-greedy selection | Interchangeable selection strategies |
| **Factory** | `create_muzero_hyperparameter_space()` | Pre-configured HP search spaces |
| **Configuration Object** | `PBTConfig`, `HyperparameterRange`, `HyperparameterSpace` | All PBT behavior configurable |
| **Population Model** | `AgentState` tracked per member | Standard evolutionary computing pattern |
| **Mutation Operators** | Perturbation (scale factor), resample (random within range) | Standard PBT mutation strategies |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `config.py` | PBTConfig, AgentState, HyperparameterRange, HyperparameterSpace, SelectionMethod |
| `population.py` | Population management, agent state tracking |
| `mutations.py` | Hyperparameter mutation operators |
| `scheduler.py` | Selection strategies, fitness evaluation scheduling |
| `coordinator.py` | PBT integration with distributed training |
| `evaluator.py` | Agent fitness evaluation |
| `muzero_hyperparams.py` | MuZero-specific HP space (dynamics steps, reward loss weight, etc.) |

---

## Skills

- **Population Management**: Track, rank, and select from agent populations
- **Mutation Design**: Perturbation and resample strategies with configurable rates
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

## Dependencies

- **Internal**: `alpha_zero.distributed` (coordinator integration), `alpha_zero.core` (network config)
- **External**: `numpy`, `dataclasses`
- **Consumed by**: `alpha_zero.cloud.gcp.pbt_orchestrator`, `alpha_zero.distributed.coordinator`
