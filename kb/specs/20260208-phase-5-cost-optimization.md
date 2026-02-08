# Phase 5: Cost Optimization & Monitoring

**Objective:** Implement cost estimation to track GCP spending, validation of robust preemption handling for Spot VMs, and unified W&B dashboarding for system observability.

## Files to Touch

- `alpha_zero/cloud/gcp/cost.py`: [NEW] Implementation of `CostEstimator` class.
- `alpha_zero/cloud/gcp/vertex_orchestrator.py`: [MODIFY] Integation of cost logging at job completion.
- `alpha_zero/cloud/gcp/training_worker.py`: [MODIFY] Verification/Enhancement of `SIGTERM` handling for preemption.
- `configs/wandb_dashboard.json`: [NEW] Configuration for standard W&B dashboard layout.
- `tests/cloud/unit/test_cost.py`: [NEW] Unit tests for cost estimation logic.

## Detailed Requirements

### 1. Cost Estimation (`cost.py`)

- **Class `CostEstimator`**:
  - method `estimate_job_cost(machine_type, accelerator_type, accelerator_count, duration_hours, is_spot) -> float`
  - Load pricing from `GCPConfig`.
  - Support standard machine types used in project (n1-standard, a2-highgpu).

### 2. Preemption Handling (`training_worker.py`)

- **Signal Handling**:
  - Register `signal.signal(signal.SIGTERM, handler)`.
  - Handler must trigger `save_checkpoint()` immediately.
  - Handler should exit gracefully with status 0 (or appropriate retry code for Vertex).

### 3. Dashboarding (`wandb_dashboard.json`)

- **Layout**:
  - **Training Metrics**: Loss, Policy Loss, Value Loss.
  - **System Metrics**: GPU Utilization, Memory Usage.
  - **Evaluation**: ELO (if available), Win Rate.

## Verification Steps

1. **Unit Testing**:
   - `test_cost.py` must pass with 100% coverage of `CostEstimator`.
2. **Integration Verification**:
   - `vertex_orchestrator` dry-run must log an estimated cost (mocked duration).
   - `training_worker` must catch `SIGTERM` in a mock environment (using `pytest-mock` or standard `subprocess` tests).

## Definition of Done

- [ ] `CostEstimator` implemented and tested.
- [ ] Preemption logic verified in `training_worker.py`.
- [ ] W&B Dashboard config created.
- [ ] All tests pass.
- [ ] Linter/Mypy checks pass.
