# PRD: Population-Based Training (M11)

**Epic**: M11 — Hyperparameter Tuning
**Complexity**: XL (Extra Large)
**Sprint**: 2

## User Story

**As a** ML researcher,
**I want** to automatically tune hyperparameters (LR, MCTS sims, temperature, c_puct) using PBT,
**So that** I can find optimal training configurations without manual grid search.

## Acceptance Criteria

### AC1: Population Management

**Given** a PBT config with `population_size=8`,
**When** the tuner initializes,
**Then** 8 agents are created with different hyperparameter configurations.

### AC2: Exploit/Explore

**Given** a completed evaluation round,
**When** agents are ranked by Elo/win-rate,
**Then** the bottom 20% copy weights from the top 20% and apply ±20% perturbation.

### AC3: Config via Env Vars

**Given** the PBT module,
**When** `CIV6AZ_PBT_POP_SIZE=4` is set,
**Then** the population size is 4 (not the default 8).

### AC4: Test Coverage ≥ 80%

**Given** the PBT module,
**When** `pytest tests/unit/test_pbt*.py --cov=alpha_zero/pbt` is run,
**Then** coverage is ≥ 80%.

## Out of Scope

- Neural Architecture Search (NAS)
- Bayesian Optimization (BO) — PBT only for now
- Multi-objective optimization

## Success Metrics

- Best PBT agent achieves ≥ 10% higher Elo than fixed-hyperparameter baseline
- PBT identifies optimal LR within 20 iterations
