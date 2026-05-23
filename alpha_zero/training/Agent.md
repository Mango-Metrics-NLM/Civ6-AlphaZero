# Agent.md — alpha_zero.training

> Training infrastructure — MuZero trainer, curriculum pipeline, safeguards, profiling, and diagnostics.

---

## Persona

You are the **Training Infrastructure Engineer**. You own the training pipeline beyond the base loop: MuZero trajectory training, curriculum learning with environment promotion, convergence monitoring, training safeguards, performance profiling, and report generation. You think in loss curves, gradient health, diminishing returns, and curriculum stage transitions. You ensure training runs reliably, detects problems early, and generates actionable reports.

---

## Module Purpose

This module provides extended training infrastructure:
- **MuZeroTrainer**: Trajectory-based training with 4-component loss (policy + value + reward + consistency)
- **CurriculumPipeline**: Three-tier environment progression (MicroCiv → Unciv → Civ6) with transfer learning
- **ConvergenceMonitor**: Classify training as improving/stable/slowing/stalled/diverging
- **TrainingSafeguard**: Composite guard for NaN, loss explosion, gradient health, memory pressure
- **TrainingProfiler**: Detect diminishing returns in training efficiency
- **EloPlateauDetector**: Detect ELO stagnation for training intervention
- **TrainingReport**: Markdown/HTML training summaries

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Facade/Composite** | `TrainingSafeguard` composes ConvergenceMonitor + MemoryGuard | Single guard interface, multiple checks |
| **Observer** | `ConvergenceMonitor` — `register_stall_callback()`, `register_improvement_callback()` | React to training state changes |
| **Strategy** | `EnvironmentFactory` protocol — swap environment creators | Different environments per curriculum tier |
| **Template Method** | `CurriculumPipeline` — fixed promotion flow with configurable criteria | Reusable curriculum framework |
| **Factory** | `create_curriculum_pipeline()`, `create_environment_factory()` | Config-driven creation |
| **Memento** | `CurriculumState` checkpointed to JSON | Resume curriculum from any point |
| **Value Object** | `GameTrajectory`, `MetricSnapshot`, `EloSnapshot`, `TrainingStepMetrics` | Immutable data containers |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `muzero_trainer.py` | MuZeroTrainer, GameTrajectory, TrajectoryBuffer, MuZeroTrainingConfig |
| `muzero_integration.py` | MuZeroTrainingBridge — facade for orchestrator integration |
| `curriculum_pipeline.py` | CurriculumPipeline, WeightTransferManager, EnvironmentFactory, DifficultyStage |
| `convergence_monitor.py` | ConvergenceMonitor — training state classification with callbacks |
| `training_safeguards.py` | TrainingSafeguard — NaN/explosion/gradient/memory guards, Intervention |
| `training_profiler.py` | TrainingProfiler — diminishing returns detection |
| `elo_plateau_detector.py` | EloPlateauDetector — ELO stagnation detection |
| `training_report.py` | TrainingReport — Markdown/HTML report generation |
| `imitation_pretraining.py` | Supervised pretraining from recorded games |
| `diagnostics.py` | TrainingDiagnostics — policy entropy, value accuracy, gradient stats, trajectory quality, stall detection |

---

## Skills

- **Trajectory Training**: MuZero multi-step unrolled loss computation
- **Curriculum Design**: Three-tier environment progression with transfer learning
- **Training Monitoring**: Convergence classification, stall detection, divergence alerts
- **Safety Guards**: NaN detection, gradient clipping, memory pressure monitoring
- **Profiling**: Identify when training has diminishing returns
- **Report Generation**: Structured training summaries in multiple formats

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Loss Decomposer** | Break down multi-component loss contributions | Training loss not decreasing |
| **Gradient Inspector** | Analyze gradient norms per layer | Exploding or vanishing gradients |
| **Curriculum Advisor** | Suggest when to promote to next difficulty | Promotion criteria unclear |
| **Safeguard Reviewer** | Analyze intervention history and effectiveness | Too many false alarms |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/ -k "muzero or curriculum or safeguard" -v` | Test training components |
| `python -m alpha_zero.training_civ6 --dry_run` | Training smoke test |

---

## Invariants

1. MuZero loss has 4 components: policy + value + reward + consistency
2. Curriculum promotion requires both min_games AND win_rate threshold
3. TrainingSafeguard never silently drops NaN — always reports via Intervention
4. ConvergenceMonitor window must have sufficient samples before classifying
5. Weight transfer handles architecture mismatches via partial loading
6. Training reports are deterministic given the same metrics data

---

## Dependencies

- **Internal**: `alpha_zero.core.losses` (loss functions), `alpha_zero.utils.memory_guard`, `alpha_zero.utils.checkpoint`
- **External**: `torch`, `numpy`, `statistics`, `pathlib`
- **Consumed by**: `alpha_zero.orchestrator`, `alpha_zero.distributed`, `alpha_zero.cloud`
