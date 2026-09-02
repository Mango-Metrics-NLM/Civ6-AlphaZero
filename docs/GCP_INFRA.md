# GCP Infrastructure — Civ6 AlphaZero Training

## Project Overview

| Field | Value |
|-------|-------|
| **Project ID** | `hrmtrain` |
| **Project Number** | `1019219508001` |
| **Region** | `us-central1` |
| **Account** | `ianshank@gmail.com` |

## Resources

### Artifact Registry

| Repository | Format | Location | Purpose |
|------------|--------|----------|---------|
| `civ6az` | DOCKER | us-central1 | Training container images |
| `civ6az-muzero-world-model` | DOCKER | us-central1 | MuZero world model images |
| `alphagalerkin` | DOCKER | us-central1 | AlphaGalerkin images |
| `muzero-world-model` | DOCKER | us-central1 | Legacy MuZero images |
| `vertex-ai-training` | DOCKER | us-central1 | General Vertex AI training |

**Active Image:**

```
us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v1.2.0
```

**Stabilized Image (post bug-fix):**

```
us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v15-stabilized
```

### GCS Buckets

| Bucket | Purpose |
|--------|---------|
| `civ6az-training` | Models, trajectories, checkpoints, logs |
| `civ6az-muzero-training` | MuZero world model artifacts |
| `hrmtrain_cloudbuild` | Cloud Build source tarballs |

**Bucket Structure — `gs://civ6az-training/`:**

```
models/civ6az/v000/           # Initial model (v0)
models/civ6az/v001/..v219/    # 219 trained versions
trajectories/civ6az/iter_001..iter_049/  # Self-play data
checkpoints/civ6az/           # Training checkpoints
logs/civ6az/                  # Training logs
```

### Vertex AI Custom Jobs

**Training Run Summary (49 iterations):**

| Metric | Value |
|--------|-------|
| Total iterations | 49 |
| Model versions | v000 → v219 |
| Container image | `civ6az-training:v1.2.0` |
| W&B project | `civ6-alphazero` |
| W&B group | `dashing-firebrand-1393` |

**Iteration 49 — Training Job:**

| Field | Value |
|-------|-------|
| Display Name | `civ6az-training-iter049` |
| Job ID | `4029264727690444800` |
| State | ✅ `JOB_STATE_SUCCEEDED` |
| Duration | 5m 32s (21:24:44 → 21:30:16 UTC) |
| Machine | `n1-standard-8` + `NVIDIA_TESLA_T4` × 1 |
| Disk | 100 GB `pd-ssd` |
| Epochs | 10 |
| Trajectory window | iterations 45–49 |
| LR schedule | warmup_cosine (0.001 → 0.0001) |
| LR warmup steps | 500 |
| Value LR multiplier | 0.1 |
| Architecture | 8×8 board, 376 actions, 6 layers, 64 embed |

**Iteration 49 — Evaluation Job:**

| Field | Value |
|-------|-------|
| Display Name | `civ6az-eval-iter049` |
| Job ID | `6998262782034444288` |
| State | ✅ `JOB_STATE_SUCCEEDED` |
| Duration | 21m 7s (21:34:56 → 21:56:03 UTC) |
| Machine | `n1-standard-4` (CPU-only) |
| Matchup | v219 vs v218 |
| Games | 30 |
| MCTS | Gumbel (16 sims, 16 top actions) |
| Max turns | 30 |

## Environment Variables

| Variable | Value | Source |
|----------|-------|--------|
| `CIV6AZ_VERTEX_PROJECT` | `hrmtrain` | Required |
| `CIV6AZ_VERTEX_REGION` | `us-central1` | Required |
| `CIV6AZ_GCS_BUCKET` | `civ6az-training` | Required |
| `CIV6AZ_CONTAINER_IMAGE_URI` | `us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v15-stabilized` | Required |
| `CIV6AZ_ENCODER_TYPE` | `patch_embed` | Default |
| `CIV6AZ_MLFLOW_ENABLED` | `true` | Optional |
| `CIV6AZ_MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Optional |
| `CIV6AZ_MLFLOW_EXPERIMENT_NAME` | `civ6-alphazero` | Optional |
| ~~`WANDB_ENABLED` / `WANDB_PROJECT` / `WANDB_MODE`~~ | — | **Legacy.** W&B was retired on 2026-07-13; MLflow is the sole tracker. `resolve_tracker_enabled()` still honours `WANDB_ENABLED` as a last-resort fallback and warns. The iteration-49 run above predates the switch and used them. |

## IAM & Service Accounts

- **Active Account**: `ianshank@gmail.com` (owner)
- **ADC**: Application Default Credentials configured
- **Container SA**: Uses Vertex AI default compute service account

## Cost Estimate (Iteration 49)

| Job | Machine | GPU | Duration | Est. Cost |
|-----|---------|-----|----------|-----------|
| Training | n1-standard-8 | T4 × 1 | 5.5 min | ~$0.06 |
| Evaluation | n1-standard-4 | None | 21 min | ~$0.02 |
| **Total iter 49** | | | 26.5 min | **~$0.08** |
| **Full 49 iters** | | | ~hours | **~$4-10** |

## Cloud Build Pipeline

**Config**: `dockerfiles/cloudbuild.yaml`

9-step pipeline (5m 7s):

1. Lint (ruff)
2. Unit tests
3. Docker build (NVIDIA PyTorch 24.09-py3 base)
4. Import smoke test
5. Self-play dry run
6. Training dry run
7. MuZero self-play test
8. Orchestrator guard test
9. Push to Artifact Registry

## Rollback Instructions

```bash
# Revert to previous container image
gcloud builds submit --config=dockerfiles/cloudbuild.yaml \
  --substitutions=_IMAGE_TAG=v1.2.0 .

# Roll back model version
python -m alpha_zero.cloud.gcp.cli init --upload-model --version 218
```

## Teardown

```bash
# Delete training artifacts (DESTRUCTIVE)
gcloud storage rm -r gs://civ6az-training/trajectories/
gcloud storage rm -r gs://civ6az-training/checkpoints/

# Delete container images
gcloud artifacts docker images delete \
  us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v15-stabilized

# Cancel running jobs
gcloud ai custom-jobs cancel JOB_ID --region=us-central1
```
