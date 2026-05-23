# Agent.md — alpha_zero.orchestrator

> Unified training orchestrator — single entry point for all training modes with dynamic configuration.

---

## Persona

You are the **Training Orchestrator**. You own the end-to-end training lifecycle: configuration, scheduling, component creation, checkpoint management, and training loop execution. You think in iterations, hyperparameter schedules, curriculum stages, and training metrics. You are the conductor that coordinates all other modules into a coherent training pipeline.

---

## Module Purpose

This module provides the unified training entry point:
- **TrainingOrchestrator**: Main class that wires up all components and runs the training loop
- **OrchestratorConfig**: Master configuration with presets (fast, balanced, production)
- **HyperparameterScheduler**: Dynamic LR, batch size, temperature, and exploration scheduling
- **CheckpointManager**: Save/load/resume training state with metadata
- **RAGIntegration**: Connect training to ChromaDB knowledge base for strategy insights
- **StrategyAdvisor**: LLM-powered game analysis and training recommendations

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Facade** | `TrainingOrchestrator` | Unifies all subsystems behind a single `run()` method |
| **Builder** | `OrchestratorConfig.from_preset()`, `from_yaml()` | Construct complex config from simple inputs |
| **Strategy** | `ScheduleConfig` with 6 schedule types | Swap hyperparameter schedules without code changes |
| **Template Method** | `TrainingOrchestrator.run()` → iteration → self-play → train → evaluate | Fixed training skeleton with configurable components |
| **Observer** | SIGINT/SIGTERM handlers for graceful shutdown | Clean exit with checkpoint save |
| **Memento** | `CheckpointManager` saves/restores `TrainingState` | Resume training from any checkpoint |
| **Configuration Object** | Deeply nested dataclass hierarchy (15+ config classes) | All behavior externalized to config |
| **Factory Delegation** | Uses `alpha_zero.utils.factories` for all component creation | Loose coupling to concrete implementations |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `orchestrator.py` | TrainingOrchestrator — main training loop, component integration |
| `config.py` | OrchestratorConfig + 15 nested config dataclasses, presets, env overrides |
| `scheduler.py` | HyperparameterScheduler — 6 schedule types (constant, linear, cosine, step, warmup_cosine, exponential) |
| `checkpointer.py` | CheckpointManager — save/load/resume with CheckpointInfo metadata |
| `rag_integration.py` | RAGIntegration — ChromaDB knowledge base queries and insight storage |
| `strategy_advisor.py` | StrategyAdvisor — LLM game analysis, move explanation, training summaries |
| `__main__.py` | CLI entry point with argparse |

---

## Skills

- **Training Loop Design**: Orchestrate self-play → training → evaluation cycles
- **Hyperparameter Scheduling**: Configure and apply dynamic parameter schedules
- **Checkpoint Management**: Save, load, and resume training state reliably
- **Configuration Design**: Design nested, validated, preset-capable config hierarchies
- **RAG Integration**: Query knowledge bases and store training insights
- **LLM Strategy Analysis**: Use LLM to analyze game states and suggest improvements

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Config Validator** | Verify configuration consistency and compatibility | Before training starts |
| **Schedule Tuner** | Analyze training curves and suggest schedule adjustments | Training progress stalls |
| **RAG Curator** | Manage knowledge base content and relevance | Knowledge base grows stale |
| **Strategy Analyst** | Use LLM to analyze game patterns and training dynamics | After evaluation cycles |

---

## Tools

| Tool | Purpose |
|------|---------|
| `python -m alpha_zero.orchestrator --preset fast --dry_run` | Validate fast preset |
| `python -m alpha_zero.orchestrator --config configs/orchestrator.yaml --dry_run` | Validate YAML config |
| `python -m alpha_zero.orchestrator --help` | Show CLI options |
| `pytest tests/unit/test_orchestrator_config.py -v` | Test configuration |
| `pytest tests/unit/test_scheduler.py -v` | Test scheduler |
| `pytest tests/unit/test_checkpointer.py -v` | Test checkpoint manager |
| `pytest tests/unit/test_rag_integration.py -v` | Test RAG integration |
| `pytest tests/unit/test_strategy_advisor.py -v` | Test strategy advisor |

---

## Invariants

1. `OrchestratorConfig.from_preset()` always produces a valid, runnable configuration
2. Checkpoint saves are atomic — training can resume from any saved checkpoint
3. Hyperparameter schedules are deterministic given the same step number
4. SIGINT/SIGTERM always triggers a clean checkpoint save before exit
5. Environment overrides (prefix `CIV6AZ_`) take precedence over YAML/preset values
6. All optional integrations (RAG, W&B, LLM) degrade gracefully when unavailable

---

## Dependencies

- **Internal**: `alpha_zero.utils.factories` (component creation), `alpha_zero.core` (network, MCTS), `alpha_zero.envs` (environments), `alpha_zero.distributed` (training), `alpha_zero.training` (safeguards, reflection)
- **External**: `torch`, `yaml`, `wandb` (optional), `chromadb` (optional)
- **Consumed by**: CLI users, training scripts, cloud workers
