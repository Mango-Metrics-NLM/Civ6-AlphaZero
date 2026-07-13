# DEV-SQE → Pre-PR Handoff

**Date:** 2026-07-13
**Spec:** user-approved in-session plan (W&B → MLflow migration completion; deletions and `--no-wandb` behavior change explicitly approved via AskUserQuestion)
**Files Changed:** 50 files, +1,070 / −2,674 (net −1,604) — highlights:
- `alpha_zero/monitoring/config.py` — `resolve_tracker_enabled()` three-tier toggle (`CIV6AZ_MLFLOW_ENABLED` > `MLFLOW_ENABLED` > legacy `WANDB_ENABLED` w/ DeprecationWarning), wired into `from_env()`
- `alpha_zero/monitoring/mlflow_logger.py` — `_coerce_metric_value` (numpy scalars/0-d/size-1 arrays, one-level dict flattening), `run_name` property, `alert()` also `logger.warning`s, `watch_model()` tags the request, fidelity-map docstring
- `alpha_zero/orchestrator/config.py` — `WandbConfig.to_mlflow_config()` bridge (correct tags/entity/mode/log_model mapping)
- `alpha_zero/orchestrator/orchestrator.py` — single-tracker init (mlflow wins; wandb section bridges + warns); dead `_wandb_logger` attr/paths removed
- `alpha_zero/orchestrator/__main__.py` — `--no-tracking` added; `--no-wandb` now disables BOTH sections [approved behavior change]
- `alpha_zero/utils/factories.py` — `create_wandb_logger` = deprecated delegating wrapper (fixes `dict(list[str])` tags mangling)
- `alpha_zero/utils/__init__.py` — legacy tracker names served via lazy PEP 562 `__getattr__` with DeprecationWarning
- `alpha_zero/cloud/gcp/vertex_orchestrator.py` — run-group propagation fixed via `run_name` (never worked before: read `.name`/`.id` off MLflow run objects)
- `alpha_zero/cloud/gcp/{tournament_worker,alert_manager,vertex_jobs}.py`, `training_report.py` + call site — toggles/env/naming unified
- `run_enhanced_training.py`, `run_muzero_training.py` — migrated off the deleted shim; **fixes `--wandb` TypeError crash** (constructed W&B-shaped config against the MLflow class)
- `configs/vertex_*.yaml`, `cloud_training_v1.yaml` — `WANDB_*` env blocks → `MLFLOW_*`
- DELETED [approved]: `utils/wandb_logger.py` shim; `vertex_jobs` W&B secret functions; 5 module-skipped suites + skipped classes in 3 more files (~2,400 lines)
- RENAMED: `test_wandb_coordination.py` → `test_vertex_coordination.py` (9 active tests kept); `test_enhanced_training_wandb.py` → `test_enhanced_training_tracking.py` (unit + integration)

**Tests Added:** 66 new/ported tests, landed BEFORE deletions: `test_mlflow_logger_adapter.py` +14 (numpy contract, `log_reflection_stats`, `run_name`, alert caplog, baseline fan-out), `test_monitoring/test_tracker_enabled_resolution.py` (13, precedence matrix), `test_pbt_mlflow_dashboards.py` (6), `test_orchestrator_mlflow.py` +7 (main() lifecycle, exit codes 130/1, run-group propagation), `test_orchestrator_wandb_bridge.py` (11), `test_experiment_tracker.py` rewritten (5).
**Test Results:** full CI-equivalent suite run in progress at handoff-write time; all targeted suites green (see PR for final numbers). Coverage gate 85% enforced.
**Lint Status:** ruff clean across `alpha_zero/ tests/ scripts/ infra/ run_*.py` (modern 0.15.8) AND CI-pinned 0.8.0 (`check` + `format --check` on `alpha_zero/`).
**Type Check Status:** no new semantic mypy errors in changed files (remaining are pre-existing `no-any-return`/`unused-ignore`/stub noise).
**Smoke Tests:** `WANDB_ENABLED=true` → `MLflowConfig.from_env().enabled` ✓; `--no-tracking`/`--no-wandb` in orchestrator help ✓; both `run_*_training.py --help` ✓; shim absence verified ✓.
**Known Issues:**
- `run_sweep.py` is the one deliberate W&B remnant (sweep primitives; self-deprecates loudly; Optuna port tracked).
- `scripts/setup_secrets.sh` still provisions the old wandb secret — harmless provisioning helper, left for a shell-scripts cleanup pass.
- `TournamentWandbLogger` and serialized `wandb_enabled` config fields keep legacy names for compat (documented in CLAUDE.md §5.3).
