# MuZero Pipeline — Post-Bug-Fix Training Sprint

> **Sprint Goal:** Execute a GCP Vertex AI training run after stabilizing the codebase with DDP, lint, and cloud worker fixes.

## Current State

| Area | Status | Detail |
|------|--------|--------|
| DDP/Distributed | ✅ Fixed | DDPLearner init stabilized, 32/32 tests pass |
| Lint (Ruff) | ✅ Clean | 0 E/F violations, format check pass |
| Cloud Workers | ✅ Validated | selfplay + training worker imports verified |
| Container Pre-Build | ✅ Ready | All 4 gates pass, context 20.5 MB |
| Test Suite | ✅ Passed | 3,795+ tests across 6 gates |
| W&B | ⚠️ Windows | External log_code timeout — Linux/Container unaffected |
| Container | ⬜ Pending | Cloud Build submission awaiting user |

## Completed Milestones

### M1: Code Quality Gate — Phase 0 ✅

- 31 E501 + 2 I001 + 2 W293 lint violations fixed
- Stale files and broken tests removed
- Terrain ratio float comparison fixed
- **Commit**: `chore: Phase 0 code quality gate`

### M2: DDP Stabilization — Phase 1 ✅

- DDPLearner `initialize()` removed, init reordered
- 32 DDP tests passing (multinode + initialization)
- **Commit**: `fix: stabilize DDPLearner initialization`

### M3: Cloud Worker Validation — Phase 2 ✅

- Imports verified, no deprecated API calls
- Timeout configs updated in vertex_training.yaml
- requirements-container.txt deps confirmed

### M4: Pre-Build Validation — Phase 3A ✅

- Gate 1: ruff E/F clean
- Gate 2: 4/4 container imports OK
- Gate 3: 509 cloud unit tests pass
- Gate 4: Docker context 20.5 MB

### M5: Full Test Suite — Phase 4 ✅

- Unit: 2,798 passed | Sanity: 14/14 | Security: 8/8
- Integration: 466 | Cloud Unit: 509
- 3 test drift bugs fixed (MCTSConfig defaults)
- **Commits**: 7 total on `feature/inference-service-v1.2.0-stabilized`

## Pending Milestones

### M6: Container Build — Phase 3B (S)

- Cloud Build: `gcloud builds submit --config=dockerfiles/cloudbuild.yaml --substitutions=_IMAGE_TAG=v15-stabilized .`
- Verify 9 steps pass + image in Artifact Registry

### M7: GCP Auth & Dry Run — Phase 5 (S)

- ADC login + env vars (CIV6AZ_GCS_BUCKET, VERTEX_PROJECT, VERTEX_REGION)
- Dry-run validation: `python -m alpha_zero.cloud.gcp.cli train --dry-run`

### M8: Training Execution — Phase 6 (L)

- Init → Self-Play (5 workers × 100 games) → Training → Monitor
- Budget tier: T4 GPU, fallback to A100 if quota exceeded

### M9: Post-Training Validation — Phase 7 (M)

- GCS checkpoint verification
- W&B dashboard review
- GCP_INFRA.md generation

### M10: Multi-Iteration Loop (XL — Sprint 2)

- 10 full AlphaZero iterations via vertex_orchestrator
- Evaluation gate: 55% win rate threshold

### M11: Hyperparameter Tuning (XL — Sprint 2)

- PBT module for population-based training
- Tune: LR, MCTS sims, temperature, c_puct
