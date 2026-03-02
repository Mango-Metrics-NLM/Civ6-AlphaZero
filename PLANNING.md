# MuZero Pipeline — Post-Bug-Fix Training Sprint

> **Sprint Goal:** Execute a GCP Vertex AI training run after stabilizing the codebase with DDP, lint, and cloud worker fixes.

## Current State

| Area | Status | Detail |
|------|--------|--------|
| DDP/Distributed | ✅ Fixed | DDPLearner init stabilized, 32/32 tests pass |
| Lint (Ruff) | ✅ Clean | 0 E501/I001/W293 violations, format check pass |
| Cloud Workers | ✅ Validated | selfplay + training worker imports verified |
| Container | ⚠️ Stale | Must rebuild with all fixes applied |
| W&B | ⚠️ Timeout | External W&B hashing timeout in cloud tests |
| Terrain Ratios | ✅ Fixed | Float comparison via `math.isclose` |

## Milestones

### M1: Code Quality Gate (Done) — Phase 0

- ✅ Fixed 31 E501, 2 I001, 2 W293 lint violations
- ✅ Removed stale files and broken test_atari_web
- ✅ Fixed terrain ratio float comparison
- **Commit**: `chore: Phase 0 code quality gate`

### M2: DDP Stabilization (Done) — Phase 1

- ✅ DDPLearner `initialize()` removed, init reordered
- ✅ 32 DDP tests passing (multinode + initialization)
- **Commit**: `fix: stabilize DDPLearner initialization`

### M3: Cloud Worker Validation (Done) — Phase 2

- ✅ Imports verified, no deprecated API calls
- ✅ Timeout configs added to vertex_training.yaml
- ✅ requirements-container.txt deps confirmed

### M4: Container Rebuild (S) — Phase 3

- [ ] Cloud Build: `gcloud builds submit --config=dockerfiles/cloudbuild.yaml --substitutions=_IMAGE_TAG=v15-stabilized`
- **Dependency**: All code fixes committed

### M5: Full Test Suite (M) — Phase 4

- [ ] Unit, integration, cloud, security, sanity tests
- [ ] Coverage ≥ 80%

### M6: GCP Auth & Training (L) — Phase 5-6

- [ ] Setup auth, env vars, dry-run
- [ ] Init → Self-play → Training → Monitor
