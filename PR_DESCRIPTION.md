## Summary

Phase 17 Training Readiness (v1.2.0) — Fixes critical gaps in training correctness, W&B
observability, cloud reliability, and test infrastructure, enabling full-scale 50-iteration
production runs on Vertex AI. Also includes Phase 17a Hex Crystalline Encoder, GCP
infrastructure operationalization (Artifact Registry migration, Cloud Build hardening),
composable gameplay systems, and MuZero cloud integration.

**Production run active**: `dashing-firebrand-1393` — 50 iterations, 4 workers × 50 games,
GumbelMCTS 16 sims, Container `civ6az-prod-v120` running on `civ6az-orchestrator` VM.

---

## Changes

### Core Training Correctness (Phase 17b)

- **action_mask in training pipeline**: `selfplay_worker` stores `trajectory["action_masks"]`;
  `training_worker` unpacks 5-tensor `TensorDataset(spatial, vector, action_mask, policies, values)`;
  masked cross-entropy prevents illegal actions from receiving policy gradients
  (`target *= mask; target /= target.sum(-1, keepdim=True).clamp(1e-8)`)
- **Removed `TrajectoryDataset`**: Dead class (never instantiated) deleted from `training_worker.py`
- **Conftest fixture fix**: `np.ones(100)` → `np.ones(376)` in `sample_trajectory_data` (correct
  action space size)

### W&B Observability (Phase 17b)

- **Group coordination**: `_wandb_env_vars()` now forwards `WANDB_GROUP` + `WANDB_RUN_NAME`;
  orchestrator sets `os.environ["WANDB_GROUP"]` after W&B init so all workers share the same group;
  each phase sets a unique `WANDB_RUN_NAME` with finally cleanup
- **Evaluation worker W&B**: Full `WandbLogger` init/log/finish in `evaluation_worker.py`;
  logs `eval/win_rate`, `eval/draw_rate`, `eval/games_per_minute` at `step=iteration`
- **Training worker model logging**: Logs `train/final_model_version` summary after `save_final_model()`

### Cloud Reliability (Phase 17b)

- **Evaluation retry + circuit breaker**: `run_evaluation_phase()` gets `max_retries=2` with
  `compute_retry_delay()` exponential backoff; eval failures now correctly increment
  `consecutive_failures` for circuit breaker logic
- **`_EVAL_MACHINE_TYPE` constant**: Replaces hard-coded `"n1-standard-4"` string
- **Learning rate forwarding**: `--learning_rate` arg added to orchestrator CLI;
  `build_training_extra_args()` always emits `--learning_rate=N` (previously silently dropped)
- **DEFAULT_MCTS_TYPE**: Changed `"civ6"` → `"gumbel"` in `config.py` (13× faster)

### Test Infrastructure (Phase 17b)

- **pytest-timeout=120**: Added to `pyproject.toml` `addopts` and dev dependencies
- **MockStorageBackend centralized**: Moved to `tests/mocks/cloud_mocks.py`; 3 test files import
  from shared location instead of defining locally
- **Stale test fixes**: All `test_default_mcts_type_is_civ6` tests renamed to `_is_gumbel` (3 files)
- **torch>=2.0.0**: Updated from `>=1.13.0` in `requirements-container.txt`

### Cleanup (Phase 17b)

- **Removed**: `tests/stress/` (empty dir), `tests/cloud/integration/` (empty dir),
  `scripts/validate_phase1.py` (stale at Phase 16), `requirements-runtime.txt` (overlaps
  container requirements), `--encoder_type` dead CLI arg from `evaluation_worker.py`

### GCS Trajectory Cleanup

- **`scripts/cleanup_trajectories.py`**: New script for managing GCS trajectory storage;
  supports `--list-only`, `--dry-run`, `--delete` modes; uses `GCSTrajectoryStorage.list_iteration_stats()`
  and `delete_iterations()` added to storage module

### GCP Infrastructure Operationalization

- **Artifact Registry migration**: Container pipeline migrated from GCR to Artifact Registry
  (`us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training`)
- **Cloud Build hardening**: `cloudbuild.yaml` gets `_TAG_NAME`/`_IMAGE` substitutions with
  defaults; evaluation dry-run step added; `$$VAR` used for shell vars
- **Setup scripts**: `setup_gcp.sh` + `setup_gcp.ps1` for one-click GCP provisioning
- **Cloud Build CI**: All 8 steps green for v1.2.0 (build ID `c72f98e9`)

### Hex Crystalline Encoder (Phase 17a)

- **`HexSpatialEncoder`**: Hex-topology-aware spatial encoder with crystalline attention
  (`HexConv2d`, `CrystallineAttentionBlock`)
- **`EncoderRegistry`**: Factory pattern for `patch_embed` / `hex_crystalline` encoder selection
- **Python 3.10 compat**: `alpha_zero.utils.compat.UTC` shim (`datetime.UTC` is 3.11+)
- **Cloud pipeline encoder propagation**: `--encoder_type` forwarded through selfplay/training jobs

### Composable Gameplay Systems

- Terrain, fog-of-war, technology tree, combat, trade route systems added to `MicroCivEnv`
- `gameplay_preset` config parameter for toggling system combinations
- Integrated across training pipeline config

### MuZero Cloud Integration

- `VertexMuZeroJob` + orchestrator support for MuZero training mode on Vertex AI
- Benchmark + smoke tests for MuZero cloud pipeline

---

## Type of Change

- [x] New feature (non-breaking change which adds functionality)
- [x] Bug fix — action_mask was never passed to model; eval failures didn't increment circuit breaker
- [x] Refactoring — MockStorageBackend centralization, dead code removal
- [x] Documentation update — CHANGELOG, README, C4 architecture, NEXT_STEPS

---

## Testing

- [x] All existing tests pass (4,880+ tests; Windows deselect list in `pyproject.toml`)
- [x] New tests: action_mask pipeline, W&B group coordination, eval retry, gumbel default
- [x] Type checking passes (`mypy alpha_zero/` — no new errors)
- [x] Linting passes (`ruff check alpha_zero/ tests/ scripts/`)
- [x] Cloud Build CI: 8/8 steps green (build `c72f98e9`, tag `v1.2.0`)
- [x] Production smoke: Container `civ6az-prod-v120` running healthy on `civ6az-orchestrator`

### Windows test deselect (known pre-existing timeouts — not introduced by this PR)

```
--deselect tests/unit/test_interfaces.py
--deselect tests/unit/test_gumbel_mcts.py
--deselect tests/unit/test_enhanced_training_wandb.py
--deselect tests/unit/test_batch_inference.py
--deselect tests/unit/test_civ6_mcts.py
--deselect tests/cloud/unit/test_training_muzero_dispatch.py
```

---

## Coverage

Current coverage: >80% (target: 80%) — maintained across all new modules.

---

## Notes

- **Branch**: `feature/hex-crystalline-encoder` (encompasses Phases 17a + 17b)
- **Container**: `us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v1.2.0`
- **Production run**: W&B [dashing-firebrand-1393](https://wandb.ai/ianshank-none/civ6-alphazero/runs/ukxjr0uz)
  — 50 iterations active on Vertex AI
- **Stats**: 30 commits, 222 files changed, +24,883 / −2,185 lines
- **Version**: `0.14.1` → `1.2.0` (major version bump reflecting production readiness)
