# Civ6 AlphaZero — CI/CD Pipeline

## Overview

The CI/CD pipeline builds and tests the training container image, then pushes it to
Artifact Registry for Vertex AI Custom Jobs.

**Image**: `us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:latest`

---

## Pipeline Stages

```
┌─────────┐   ┌────────────┐   ┌─────────┐   ┌────────────┐   ┌──────┐
│  Lint   │──▶│ Unit Tests │──▶│  Build  │──▶│ Smoke Test │──▶│ Push │
│ (ruff)  │   │  (pytest)  │   │ (Docker)│   │ + dry-runs │   │ (AR) │
└─────────┘   └────────────┘   └─────────┘   └────────────┘   └──────┘
```

| Stage | Description | Duration |
|-------|-------------|----------|
| **lint** | `ruff check` — E/F rules, ignores E501 | ~15s |
| **unit-tests** | `pytest tests/cloud/unit/` | ~30s |
| **build** | `docker build -f Dockerfile.vertex` (NVIDIA PyTorch 24.09-py3 base) | ~8 min |
| **smoke-test** | Import validation inside container | ~5s |
| **test-selfplay** | `selfplay_worker --dry_run` | ~10s |
| **test-training** | `training_worker --dry_run` | ~10s |
| **test-muzero** | MuZero selfplay dry-run | ~10s |
| **test-orchestrator-guard** | Verify orchestrator mode blocked in container | ~5s |
| **push** | Push to Artifact Registry | ~2 min |

**Total**: ~12 minutes | **Machine**: `E2_HIGHCPU_8` | **Timeout**: 30 min

---

## Manual Build

From Cloud Shell (or any machine with `gcloud` authenticated):

```bash
cd ~/Civ6-AlphaZero-real

# Build + push with default tag (latest)
gcloud builds submit --config=dockerfiles/cloudbuild.yaml \
  --project=hrmtrain .

# Build with custom tag
gcloud builds submit --config=dockerfiles/cloudbuild.yaml \
  --project=hrmtrain \
  --substitutions=_IMAGE_TAG=v2.0.0 .
```

## Automated Trigger

### Option A: Webhook Trigger (no GitHub App needed)

```bash
# Run the setup script on Cloud Shell
bash scripts/setup_webhook_trigger.sh

# Then configure GitHub webhook:
# Repo → Settings → Webhooks → Add webhook
# Payload URL = <webhook URL from script output>
# Content type = application/json
# Events = Just the push event
```

### Option B: GitHub App Trigger (requires GitHub App install)

1. Install the Google Cloud Build GitHub App:
   <https://github.com/apps/google-cloud-build/installations/select_target>
2. Select `Mango-Metrics-NLM` org → `Civ6-AlphaZero`
3. Run:

   ```bash
   gcloud builds triggers create github \
     --project=hrmtrain \
     --name=civ6az-container-build \
     --repo-name=Civ6-AlphaZero \
     --repo-owner=Mango-Metrics-NLM \
     --branch-pattern="^main$" \
     --build-config=dockerfiles/cloudbuild.yaml \
     --substitutions=_IMAGE_TAG=latest,_REGION=us-central1,_AR_REPO=civ6az
   ```

> **Note**: The repo name `Civ6-AlphaZero` has uppercase letters which may cause
> `INVALID_ARGUMENT` errors with the GitHub trigger API. The webhook approach avoids this.

---

## Key Files

| File | Purpose |
|------|---------|
| `dockerfiles/Dockerfile.vertex` | Container image (NVIDIA PyTorch 24.09 + GCP deps) |
| `dockerfiles/cloudbuild.yaml` | Full 9-stage build pipeline |
| `requirements-container.txt` | Slim Python deps for container |
| `scripts/setup_webhook_trigger.sh` | Automated webhook trigger setup |

## Debugging Failures

```bash
# View build logs
gcloud builds log <BUILD_ID> --project=hrmtrain

# List recent builds
gcloud builds list --project=hrmtrain --limit=5

# Describe a specific build
gcloud builds describe <BUILD_ID> --project=hrmtrain

# Local Docker build test
docker build -f dockerfiles/Dockerfile.vertex -t civ6az-test .
docker run --rm civ6az-test python -c "from alpha_zero.cloud.gcp.config import GCPConfig; print('OK')"
```

## Environment & Secrets

- **No hardcoded credentials** — all secrets via GCP Secret Manager or env vars
- `CIV6AZ_CONTAINER_IMAGE_URI` — set in orchestrator systemd service Environment=
- `WANDB_API_KEY` — set in `~/.env.civ6az` on the VM
- Cloud Build uses the default Cloud Build service account with:
  - `roles/artifactregistry.writer`
  - `roles/storage.objectAdmin`
