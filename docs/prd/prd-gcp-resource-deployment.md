# PRD: GCP Resource Deployment & Validation

> **Epic**: Deploy Isolated GCP Resources for MuZero-World-Model
> **Date**: 2026-02-20 | **Status**: Draft

---

## User Stories

### US-1: Artifact Registry Provisioning

**As a** ML engineer deploying MuZero training jobs,
**I want** a dedicated Artifact Registry repo (`civ6az-muzero-world-model`) in `us-central1`,
**So that** container images are isolated from other projects in the `hrmtrain` GCP project.

### US-2: GCS Bucket Provisioning

**As a** ML engineer running self-play and training,
**I want** a dedicated GCS bucket (`<muzero-bucket>`) in `us-central1`,
**So that** model checkpoints, trajectories, and logs don't collide with other training pipelines.

### US-3: Container Rebuild

**As a** ML engineer deploying code changes,
**I want** the container rebuilt from current source and tagged `v16-full-fix`,
**So that** all GCP resource isolation changes are reflected in the deployed container.

### US-4: Full Test Coverage

**As a** developer maintaining quality,
**I want** the full test suite to pass with 80%+ coverage,
**So that** the resource isolation changes are validated before production deployment.

---

## Acceptance Criteria

### AC-1: AR Repo Exists

- **Given** gcloud CLI is authenticated with `hrmtrain` project
- **When** `gcloud artifacts repositories describe civ6az-muzero-world-model --location=us-central1` is run
- **Then** the repo is returned with `format: DOCKER` and `location: us-central1`

### AC-2: GCS Bucket Exists

- **Given** gsutil is authenticated with `hrmtrain` project
- **When** `gsutil ls gs://<muzero-bucket>` is run
- **Then** the bucket is accessible (exit code 0)

### AC-3: Container Image Pushed

- **Given** AR repo `civ6az-muzero-world-model` exists
- **When** `gcloud artifacts docker images list us-central1-docker.pkg.dev/hrmtrain/civ6az-muzero-world-model` is run
- **Then** `civ6az-muzero:v16-full-fix` appears in the list

### AC-4: Container Smoke Test

- **Given** container is built
- **When** Cloud Build smoke test step runs `python -c "from alpha_zero.cloud.gcp.config import GCPConfig"`
- **Then** it exits successfully (exit code 0)

### AC-5: Test Suite Passes

- **Given** torch-enabled venv is activated
- **When** `pytest tests/ --cov=alpha_zero --cov-report=html` is run
- **Then** all tests pass and coverage is >= 80%

### AC-6: No Stale References

- **Given** all changes are applied
- **When** `grep -r "muzero-world-model/muzero-training" --include="*.py" --include="*.yaml"` is run
- **Then** 0 matches are returned

---

## Out of Scope

- Renaming resources in non-MuZero projects (PBT, enhanced training, orchestrator configs)
- W&B project migration (existing runs stay in `civ6-alphazero`)
- Data migration from the `<training-bucket>` bucket to `<muzero-bucket>`
- Submitting actual training jobs (separate Epic 4)

---

## Success Metrics

| Metric | Target |
|--------|--------|
| AR repo accessible | Yes |
| GCS bucket accessible | Yes |
| Container image pushed | `v16-full-fix` tag in AR |
| Test pass rate | 100% |
| Code coverage | >= 80% |
| Stale reference count | 0 |

---

## Open Questions

1. **Data Migration**: Should existing models/trajectories be copied from `gs://<training-bucket>` to `gs://<muzero-bucket>`? (Recommend: No, fresh start for MuZero.)
2. **Old Resources**: Should the old `muzero-world-model` AR repo and images be deleted? (Recommend: Keep for rollback, delete after 30 days.)
3. **Venv Location**: Where is the torch-enabled venv on this machine? (Need from user if `anaconda3` base doesn't have torch.)
