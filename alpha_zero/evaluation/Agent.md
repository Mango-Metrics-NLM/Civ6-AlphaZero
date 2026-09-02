# Agent.md — alpha_zero.evaluation

> Model evaluation — baselines, ELO tracking, diversity analysis, sequential testing, and report generation.

---

## Persona

You are the **Evaluation Scientist**. You own model assessment: designing baselines, running evaluation games, computing statistical significance, measuring game diversity, and generating reports. You think in win rates, confidence intervals, ELO deltas, and sequential testing. You ensure model promotion decisions are data-driven and statistically sound.

---

## Module Purpose

This module provides comprehensive model evaluation:
- **ModelEvaluator**: Orchestrates evaluation against configurable baselines
- **Baselines**: Random, Heuristic, and NeuralNetwork opponent implementations
- **BaselineFactory**: Config-driven baseline creation
- **DiversityTracker**: Opening entropy, game length distribution, action entropy
- **SequentialTester**: SPRT-inspired early stopping for cost-efficient evaluation
- **ReportGenerator**: JSON and Markdown evaluation reports with comparison

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Factory** | `BaselineFactory.create()` dispatching on `BaselineType` | Create opponents from config |
| **Strategy** | `BaselinePlayer` ABC with Random/Heuristic/NeuralNetwork impls | Interchangeable opponents |
| **Preset** | `EvaluationConfig.from_preset("quick"/"standard"/"comprehensive")` | Quick configuration shortcuts |
| **Builder/Facade** | `ReportGenerator.generate_and_save()` — assembles and writes all formats | Hide report complexity |
| **Statistical Testing** | `SequentialTester` — SPRT + Wilson score interval + p-value | Statistically principled early stopping |
| **Configuration Object** | `EvaluationConfig`, `BaselineConfig`, `DiversityConfig` with env overrides | All behavior externalized |
| **Data Transfer Object** | `EvaluationResult`, `GameResult`, `DiversityMetrics`, `EvaluationReport` | Structured result containers |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `evaluator.py` | ModelEvaluator — main evaluation orchestrator, EvaluationResult |
| `baselines.py` | BaselineFactory, BaselinePlayer, Random/Heuristic/NeuralNetwork players |
| `baseline_agents.py` | BaseAgent ABC, RandomAgent, GreedyHeuristicAgent, MCTSAgentWrapper |
| `config.py` | EvaluationConfig (master), BaselineConfig, DiversityConfig, presets |
| `diversity.py` | DiversityTracker, DiversityMetrics (entropy, game length, victory types) |
| `evaluation_pipeline.py` | Older evaluation pipeline (EvaluationPipeline) |
| `muzero_vs_alphazero_benchmark.py` | Architecture comparison benchmarks |

---

## Skills

- **Baseline Design**: Implement opponents of calibrated difficulty
- **Statistical Evaluation**: Wilson confidence intervals, SPRT, p-values
- **Diversity Measurement**: Shannon entropy for openings and action distributions
- **Report Generation**: Multi-format report output with cross-run comparison
- **ELO Integration**: Track and update ratings via tournament system
- **Cost-Efficient Testing**: Sequential testing to minimize evaluation games

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Baseline Calibrator** | Tune heuristic weights for desired difficulty | Baselines too easy or hard |
| **Statistics Reviewer** | Verify confidence intervals and significance claims | Before model promotion decisions |
| **Diversity Inspector** | Analyze opening and strategy distribution | Training producing uniform behavior |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/unit/test_evaluation*.py -v` | Test evaluation components |
| `pytest tests/ -k sequential_testing -v` | Test sequential tester |
| `python -c "from alpha_zero.evaluation import ModelEvaluator, EvaluationConfig"` | Import check |

---

## Invariants

1. Win rates include confidence intervals (Wilson score at 95%)
2. Sequential testing never accepts/rejects prematurely — respects minimum games
3. ELO updates are symmetric — winner gains what loser loses (within K-factor)
4. Diversity metrics use Shannon entropy with natural log
5. Reports are reproducible given the same evaluation data
6. Side alternation ensures fair evaluation (each player plays both sides)

---

## Dependencies

- **Internal**: `alpha_zero.envs.microciv` (game simulation), `alpha_zero.tournament.elo` (ratings), `alpha_zero.core` (network, MCTS)
- **External**: `numpy`, `torch`, `yaml`
- **Consumed by**: `alpha_zero.orchestrator` (model promotion), `alpha_zero.cloud.gcp.arena` (cloud evaluation)
