# AQA Report — Remediation Sprint

**Date:** 2026-02-20  
**Feature:** MuZero Pipeline Remediation (Epics 5, 6, 7)

## Acceptance Criteria Test Results

| # | Criterion | Source PRD | Status | Evidence |
|---|-----------|-----------|--------|----------|
| AC1 | `test_no_command_field_set` matches production code | prd-fix-tournament-test | ✅ PASS | L996 assertion updated: `["python", "-m", "alpha_zero.cloud.gcp.entrypoint"]` matches `vertex_jobs.py:1023` |
| AC2 | No regression in tournament test suite | prd-fix-tournament-test | ⏳ BLOCKED | pytest hangs on PyTorch import; needs `--import-mode=importlib` or CI |
| AC3 | Container builds with all smoke tests | prd-container-rebuild | ⏳ IN PROGRESS | Cloud Build submitted, awaiting result |
| AC4 | `.env` updated with Artifact Registry URI | prd-container-rebuild | ✅ PASS | `CIV6AZ_CONTAINER_IMAGE_URI=us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v1.1.0` |
| AC5 | W&B API key configured | prd-container-rebuild | ✅ PASS | `.env:63` has `WANDB_API_KEY=26a08...` — already present |
| AC6 | Selfplay produces trajectories | prd-container-rebuild | ⏳ PENDING | Requires successful container build + smoke test |
| AC7 | Training consumes trajectories | prd-container-rebuild | ⏳ PENDING | Requires AC6 first |

## Tests Run

### Unit Tests

- **Modified:** `test_tournament_worker.py::test_no_command_field_set` — assertion aligned with production code
- **Execution:** pytest runner hangs due to heavy PyTorch import chains in local environment
- **Recommendation:** Run via CI/Cloud Build smoke test step (step 5 in `cloudbuild.yaml`)

### Code Inspection Verification

- Verified `vertex_jobs.py:1023` sets `command=["python", "-m", "alpha_zero.cloud.gcp.entrypoint"]`
- Verified `test_tournament_worker.py:996` now asserts the same value
- Verified `_wandb_env_vars()` at `vertex_jobs.py:31` reads `WANDB_API_KEY` from `os.environ`
- Verified `.env:63` has the key set

### Integration Tests

- Cloud Build pipeline runs lint, unit tests, build, and 4 smoke tests (selfplay dry-run, training dry-run, MuZero selfplay dry-run, orchestrator guard)
- This validates the integration end-to-end inside the container

## Coverage Gaps

| Module | Coverage | Gap |
|--------|----------|-----|
| `vertex_jobs.py` (VertexTournamentJob) | Covered | Test now matches production |
| `selfplay_worker.py` (MuZero path) | Gap | MuZero MCTS integration test needs GPU; covered by Cloud Build dry-run |
| `.env` loading | Gap | No automated test verifies `.env` is loaded before dispatch |

## Recommended Sign-Off Checklist

- [x] Tournament test assertion matches production code
- [x] `.env` container URI updated to Artifact Registry
- [x] W&B API key present in `.env`
- [ ] Cloud Build completes successfully (8/8 steps)
- [ ] Post-build smoke test: 1 iteration selfplay + training
- [ ] W&B dashboard shows new metrics
- [ ] GCS manifest for new iteration shows `total_examples > 0`
