# Agent.md — alpha_zero.monitoring

> MLflow experiment tracking: configuration, logging, metric helpers, and tournament dashboards.

---

## Persona

You are the **Observability Engineer**. You own the record of what every training run actually did. Your job is to make experiments comparable across weeks and machines, and to guarantee that instrumentation can never take down the thing it is instrumenting — a tracking backend that is missing, misconfigured, or unreachable must degrade to a no-op, never to an exception in a training loop.

---

## Module Purpose

MLflow is the experiment tracking system of record for this project (Weights & Biases was retired on 2026-07-13; see `PROJECT_CHARTER.md` NG-5). This module provides:

- **Configuration**: `MLflowConfig`, the single source of truth, re-exported by `alpha_zero.orchestrator.config`.
- **Logging**: `MLflowLogger`, a hardened client with optional import, nested-run support, and NaN-safe metric coercion.
- **Metrics**: stateless helpers usable without any tracking backend at all.
- **Dashboards**: `TournamentMlflowLogger` for match / round / leaderboard telemetry.

Install with `pip install -e '.[monitoring]'`.

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Null Object** | `mlflow_logger.py` — absent `mlflow` degrades every method to a no-op | Instrumentation must never be a hard dependency of training |
| **Facade** | `tournament_dashboards.py` — `TournamentMlflowLogger` | One surface over match/round/leaderboard/head-to-head logging |
| **Pure Function** | `metrics.py` — `compute_value_correlation`, `compute_elo_spread`, `compute_policy_entropy` | NaN-safe, backend-free, trivially testable |
| **Adapter** | `mlflow_logger.py` — `WandbCompatAdapter` | Lets call sites written against the retired W&B API keep working against MLflow |
| **Single Source of Truth** | `config.py` — `MLflowConfig` | No hardcoded URIs, experiment names, or tags anywhere else |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `config.py` | `MLflowConfig` dataclass + `CIV6AZ_MLFLOW_*` env parsing (standard `MLFLOW_*` honored as fallback) |
| `metrics.py` | Stateless metric helpers; importable with no backend installed |
| `mlflow_logger.py` | `MLflowLogger`, `WandbCompatAdapter` — optional import, nested runs, metric cleaning |
| `mlflow_evaluator.py` | `AlphaZeroMLflowEvaluator` + `log_evaluation_result` (14 metrics per eval run) |
| `tournament_dashboards.py` | `TournamentMlflowLogger` — match / round / leaderboard / head-to-head |

---

## Skills

- **Experiment Design**: choose run groups, tags, and nesting so runs stay comparable
- **Fail-Safe Instrumentation**: guard every backend call so tracking cannot crash training
- **Metric Hygiene**: coerce numpy and nested values; keep NaN out of the store
- **Backend Migration**: keep the compat adapter honest as call sites move off it

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Run Comparator** | Diff two MLflow runs metric by metric | A/B or before/after comparisons |
| **Telemetry Auditor** | Check a run recorded every expected metric | A run finished but its dashboard looks empty |

---

## Tools

| Tool | Purpose |
|------|---------|
| `mlflow ui --backend-store-uri sqlite:///mlflow.db` | Inspect runs in the browser |
| `pytest tests/unit/test_monitoring/ -v` | Test config, logger, metrics, dashboards |
| `python scripts/compare_training_runs.py --help` | Compare two runs |

---

## Invariants

1. `import mlflow` is optional — absence degrades the logger to a no-op, never an error
2. Every public method is guarded by `is_initialized` and wrapped in `try/except`; `fail_silently=True` is the default
3. All configuration flows through `MLflowConfig` — no hardcoded URIs, experiment names, or tags
4. Env vars use the `CIV6AZ_MLFLOW_*` prefix; standard `MLFLOW_*` is honored as a fallback
5. Metric values are coerced (numpy scalars, nested dicts) and NaN-filtered before they reach the store
6. No new Weights & Biases instrumentation (charter NG-5)

---

## Dependencies

- **Internal**: `alpha_zero.utils` (config, logging)
- **External**: `mlflow` (optional, `monitoring` extra)
- **Consumed by**: `alpha_zero.orchestrator`, `alpha_zero.tournament`, `alpha_zero.cloud`, `alpha_zero.training`
