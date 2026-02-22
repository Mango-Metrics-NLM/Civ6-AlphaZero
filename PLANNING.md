# MuZero Pipeline — Remediation Sprint

> **Sprint Goal:** Unblock the training pipeline by fixing the root cause of 100% selfplay game failures, then rebuild the container and resume training.

## Current State

| Area | Status | Detail |
|------|--------|--------|
| Selfplay | ❌ Broken | All games fail: `MuZeroMCTS.search() missing 1 required positional argument: 'action_mask'` |
| Training | ❌ Blocked | 0 trajectories → exits with code 1 |
| Evaluation | ⚠️ Stale test | `test_no_command_field_set` asserts wrong command spec |
| Container | ❌ Stale | `v14-muzero` has old code; local fixes not deployed |
| W&B | ⚠️ Partial | Injection code exists but host key not set |

## Milestones

### M1: Fix Stale Test (S) — Epic 7

- **Goal:** Align `test_no_command_field_set` with actual `VertexTournamentJob.launch()` behavior
- **Complexity:** S (1 file, 1-line change)
- **Dependencies:** None

### M2: Container Rebuild (M) — Epic 5

- **Goal:** Build + push container with all local fixes, including MuZero MCTS fix
- **Complexity:** M (Cloud Build pipeline, smoke tests)
- **Dependencies:** All code fixes committed locally
- **Risk:** Cloud Build quotas, Docker build cache invalidation

### M3: W&B Key Configuration (S)

- **Goal:** Ensure `WANDB_API_KEY` is available in host environment for job dispatch
- **Complexity:** S (env var setup)
- **Dependencies:** None

### M4: Smoke Test & Resume Training (L)

- **Goal:** Run 1 iteration (selfplay + training + eval), verify full pipeline
- **Complexity:** L (end-to-end cloud validation)
- **Dependencies:** M1, M2, M3

### M5: Stale GCS Data Cleanup (S)

- **Goal:** Delete empty iter_028-033 shard files to avoid confusion
- **Complexity:** S (gsutil command)
- **Dependencies:** M4 (wait until pipeline is confirmed working)

## Sprint Candidates (2-week)

### Week 1: Fix & Rebuild

- [x] Root cause analysis (Epic 6) — DONE
- [ ] M1: Fix stale tournament test
- [ ] M2: Container rebuild via Cloud Build
- [ ] M3: W&B key configuration

### Week 2: Validate & Resume

- [ ] M4: End-to-end smoke test
- [ ] M5: GCS cleanup
- [ ] Resume full training run

## Blockers & Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Cloud Build failure | Medium | Cloudbuild.yaml has lint + unit test gates; can fallback to local Docker build |
| GCP quota exhaustion | Low | Already using T4 GPUs with higher default quota |
| Container image registry confusion | Low | Standardize on Artifact Registry (`us-central1-docker.pkg.dev`) |
