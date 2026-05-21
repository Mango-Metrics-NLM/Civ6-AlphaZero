# `alpha_zero.monitoring` — MLflow Tracking Subpackage

## Purpose

Optional MLflow-based experiment tracking for the orchestrator and tournament
paths. Runs in parallel with the existing W&B integration; controlled via the
`monitoring` extras (`pip install -e '.[monitoring]'`).

## Modules

| Module | Role |
|---|---|
| `config.py` | `MLflowConfig` dataclass + env-var parsing. Single source of truth (re-exported by `alpha_zero.orchestrator.config`). |
| `metrics.py` | Stateless helpers: `compute_value_correlation`, `compute_elo_spread`, `compute_policy_entropy`. |
| `mlflow_logger.py` | `MLflowLogger` — hardened client with optional import, nested-run support, NaN-safe metric cleaning. |
| `tournament_dashboards.py` | `TournamentMlflowLogger` — match/round/leaderboard helpers. |

## Hardening guarantees

- `import mlflow` is optional; absence degrades the logger to a no-op.
- Every public method is guarded by `is_initialized` and wrapped in
  `try/except`. `fail_silently=True` (default) prevents tracking errors from
  crashing training.
- All configuration values flow through `MLflowConfig`. There are no
  hardcoded URIs, experiment names, or tags.
- Environment variables use the `CIV6AZ_MLFLOW_*` prefix; standard
  `MLFLOW_*` is also honored as a fallback.

## Why not just use W&B?

W&B remains in place for `cloud/`, `pbt/`, `evaluation/`, and `optimizations/`.
MLflow was added so a single end-to-end 50-iteration production run can be
inspected via the MLflow UI with parent + nested per-iteration runs. The two
backends coexist; either can be disabled independently via config.
