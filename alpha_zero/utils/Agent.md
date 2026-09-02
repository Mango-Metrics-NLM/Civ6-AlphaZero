# Agent.md — alpha_zero.utils

> Shared utilities — configuration, logging, LLM client, testing, agent tools, and training improvements.

---

## Persona

You are the **Platform Engineer**. You own the shared utility layer: configuration management, structured logging, LLM integration, test infrastructure, agent tooling, and training enhancement utilities. You think in cross-cutting concerns, developer experience, and reusable abstractions. You ensure every other module has the infrastructure it needs without circular dependencies.

---

## Module Purpose

This module provides the foundational utilities consumed by all other modules:
- **Configuration**: YAML loading, env var overrides, typed config dataclasses
- **Logging**: Structured logging, MetricsTracker, performance profiling
- **LLM Client**: Unified OpenAI/Anthropic client with provider fallback
- **Test Runner**: Programmatic pytest execution with coverage
- **Agent Tools**: Code exploration, debug helpers, task tracking for agentic coding
- **Factories**: Dependency injection via factory functions for all components
- **Training Improvements**: Curriculum learning, enhanced rewards, environment tuning
- **W&B Logger**: Weights & Biases integration (optional)
- **RAG Client**: Thin RAG pipeline client (optional)

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Factory** | `factories.py` — `create_network()`, `create_mcts()`, `create_environment()`, etc. | Dependency injection for all components |
| **Strategy** | `llm_client.py` — OpenAI/Anthropic providers, auto-fallback | Swap LLM providers transparently |
| **Decorator** | `logging.py` — `@timer`, `timed_section()` context manager | Add profiling without modifying code |
| **Singleton** | `get_logger()`, `get_training_logger()`, `get_profiler()` | Shared instances across modules |
| **Configuration Object** | `config.py` — `Civ6Config` with 9 nested config sections | All settings externalized to YAML + env vars |
| **Adapter** | `rag_client.py` — thin wrapper over `alpha_zero.rag.RAGPipeline` | Simplified RAG API for consumers |
| **Template Method** | `CurriculumManager` — stage progression with overridable difficulty scaling | Reusable curriculum framework |
| **Builder** | `ConfigManager` — loads YAML, applies env overrides, merges configs | Progressive config construction |
| **Backward Compatibility** | `TestRunner = PytestRunner`, `TestConfig = RunConfig` aliases | Smooth API migration |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `config.py` | Civ6Config, ConfigManager, load/save/merge config utilities |
| `logging.py` / `logging_config.py` | Structured logging, MetricsTracker, GameLogger, TrainingLogger |
| `llm_client.py` | LLMClient ABC, OpenAI/Anthropic impls, provider auto-detection |
| `factories.py` | Factory functions for network, MCTS, env, rewards, curriculum, etc. |
| `test_runner.py` | PytestRunner, programmatic test execution with coverage |
| `agent_tools.py` | CodeExplorer, FileManager, DebugHelper, TaskTracker |
| `curriculum_learning.py` | CurriculumManager, AdaptiveDifficultyController |
| `enhanced_rewards.py` | EnhancedRewardShaper, WinConditionEvaluator |
| `environment_tuning.py` | TunedEnvironmentWrapper, EarlyTerminationChecker |
| `rag_client.py` | RAGClient, thin wrapper for RAG pipeline (optional) |
| `reward_shaping.py` | RewardShaper for intermediate reward signals |
| `checkpoint.py` | safe_load_checkpoint for secure deserialization |
| `constants.py` | Namespace objects for game/distributed/training constants |
| `memory_guard.py` | MemoryGuard for replay buffer memory limits |
| `timeout_config.py` | Centralized timeout configuration |
| `knowledge_base.py` | Local knowledge base utilities |
| `benchmarks.py` | Performance benchmarking utilities |
| `compat.py` | Compatibility shims for cross-version support |

---

## Skills

- **Configuration Management**: YAML loading, env var overrides, config merging and validation
- **Structured Logging**: Set up colored/structured formatters, EMA metrics tracking
- **LLM Integration**: Async LLM calls with retry, provider fallback, streaming
- **Test Infrastructure**: Programmatic test execution with coverage and keyword filtering
- **Agent Tooling**: Code exploration, symbol lookup, file management for agentic workflows
- **Reward Engineering**: Design intermediate reward signals for faster learning
- **Curriculum Design**: Progressive difficulty with adaptive stage transitions

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Config Auditor** | Validate config consistency across all modules | Configuration errors at runtime |
| **LLM Provider Tester** | Test LLM connectivity and response quality | LLM integration issues |
| **Reward Analyzer** | Evaluate reward signal quality and training impact | Reward hacking or slow learning |
| **Curriculum Tuner** | Analyze stage transition thresholds | Training stuck at a difficulty level |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/unit/test_config.py -v` | Test configuration system |
| `pytest tests/unit/test_logging.py -v` | Test logging infrastructure |
| `pytest tests/unit/test_factories.py -v` | Test factory functions |
| `python -c "from alpha_zero.utils import create_llm_client"` | LLM import check |
| `ruff check alpha_zero/utils/` | Lint utilities |
| `mypy alpha_zero/utils/` | Type-check utilities |

---

## Invariants

1. `ConfigManager` always applies env var overrides last (highest priority)
2. All optional dependencies (LLM, W&B, RAG) have graceful import fallbacks
3. `get_logger()` returns the same logger instance for the same name
4. Factory functions accept config dataclasses and return concrete instances
5. `safe_load_checkpoint()` uses `weights_only=True` by default for security
6. Environment variable prefix is `CIV6AZ_` for all config overrides
7. No circular imports — utils is a leaf dependency in the module graph

---

## Dependencies

- **Internal**: Minimal — this is a foundation layer consumed by all other modules
- **External**: `yaml`, `torch`, `numpy`, `openai` (optional), `anthropic` (optional), `mlflow` (optional), `chromadb` (optional), `dotenv` (optional)
- **Consumed by**: Every other `alpha_zero.*` module
