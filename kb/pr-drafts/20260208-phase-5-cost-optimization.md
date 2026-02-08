# PR Draft: Phase 5 Cost Optimization & Monitoring

**Title:** feat: Phase 5 Cost Optimization & Monitoring
**Linked Spec:** `kb/specs/20260208-phase-5-cost-optimization.md`

## Changes

- **New Feature**: `CostEstimator` class (`alpha_zero/cloud/gcp/cost.py`) to estimate GCP run costs.
- **Enhancement**: Added `SignalHandler` to `training_worker.py` to gracefully handle Spot VM preemptions (`SIGTERM`) by saving checkpoints.
- **Enhancement**: Integrated cost reporting into `vertex_orchestrator.py` summary logs.
- **Config**: Added `configs/wandb_dashboard.json` for unified W&B metrics.
- **Tests**: Added `tests/cloud/unit/test_cost.py` (100% coverage for cost logic).

## Learnings & Decisions

- **Preemption**: Vertex AI sends `SIGTERM` on preemption. We catch this to save state immediately, allowing jobs to resume efficiently on restart.
- **Cost Model**: We use a simplified cost model (hourly rate * duration) based on `GCPConfig`. Spot instances are estimated at ~30% of on-demand cost where specific pricing isn't configured.
- **Dashboards**: Unified W&B dashboard ensures all metrics (Loss, ELO, System) are visible in one place.

## Verification

- **Automated**: Passed 251 tests (including new cost tests).
- **Manual**: Verified `vertex_orchestrator` dry-run logs show cost estimates.
