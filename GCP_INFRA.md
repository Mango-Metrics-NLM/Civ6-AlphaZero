# GCP Infrastructure Reference

Infrastructure documentation for the Civ6 AlphaZero MuZero training pipeline on Google Cloud Platform.

## Project Configuration

| Property | Value |
|:---|:---|
| **Project ID** | `hrmtrain` |
| **Region** | `us-central1` |
| **Service Account** | `civ6az-vertex@hrmtrain.iam.gserviceaccount.com` |
| **Provisioning Script** | `scripts/setup_gcp.sh` (bash) / `scripts/setup_gcp.ps1` (PowerShell) |

## Resources

### GCS Bucket

| Property | Value |
|:---|:---|
| **Name** | `civ6az-training` |
| **Location** | `us-central1` |
| **Storage Class** | Standard |
| **Lifecycle** | 90-day auto-delete on `tmp/` prefix |

**Path layout:**

```
gs://civ6az-training/
  models/civ6az/         # Model weights (versioned)
    v001/model.pt
    latest.json
  trajectories/civ6az/   # Self-play game data
    iter_000/
      shard_worker_00.pt.xz
      manifest.json
  checkpoints/civ6az/    # Training checkpoints
    training_step_1000.pt
    latest.pt
```

### Artifact Registry

| Property | Value |
|:---|:---|
| **Repository** | `civ6az` |
| **Format** | Docker |
| **Region** | `us-central1` |
| **Image URI** | `us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v1.0.0` |

### Secret Manager

| Secret Name | Source Env Var | Purpose |
|:---|:---|:---|
| `wandb-api-key` | `WANDB_API_KEY` | Weights & Biases experiment tracking |
| `openai-api-key` | `OPENAI_API_KEY` | LLM-augmented strategy (optional) |
| `anthropic-api-key` | `ANTHROPIC_API_KEY` | LLM-augmented strategy (optional) |

### Enabled APIs

| API | Purpose |
|:---|:---|
| `aiplatform.googleapis.com` | Vertex AI Custom Jobs |
| `storage.googleapis.com` | GCS storage |
| `cloudbuild.googleapis.com` | Container build CI |
| `artifactregistry.googleapis.com` | Docker image registry |
| `secretmanager.googleapis.com` | Secrets management |
| `monitoring.googleapis.com` | Metrics and alerting |
| `containerregistry.googleapis.com` | Legacy GCR (backwards compat) |

## IAM Bindings

Service Account: `civ6az-vertex@hrmtrain.iam.gserviceaccount.com`

| Role | Purpose |
|:---|:---|
| `roles/aiplatform.user` | Submit Vertex AI Custom Jobs |
| `roles/storage.objectAdmin` | Read/write GCS buckets |
| `roles/artifactregistry.reader` | Pull container images |
| `roles/secretmanager.secretAccessor` | Read secrets at runtime |
| `roles/logging.logWriter` | Write Cloud Logging |
| `roles/monitoring.metricWriter` | Write custom metrics |

## Cost Estimates

| Run Type | Command | Est. Cost |
|:---|:---|:---|
| **Dry Run** | `-DryRun` | Free |
| **Smoke Test** | `-Iterations 1 -Workers 1 -GamesPerWorker 2` | ~$1-2 |
| **Full PBT** | `-Iterations 10 -Workers 4 -GamesPerWorker 100` | ~$50-75 |

**Per-resource rates** (budget tier, us-central1):

- Self-play worker: ~$0.35/hr (n1-standard-8 + T4 GPU, Spot VMs for 60-91% savings)
- Training worker: ~$0.35/hr (n1-standard-8 + T4 GPU)
- GCS storage: ~$0.02/GB/month
- Secret Manager: ~$0.03/10K accesses

## Launch Commands

```powershell
# 1. Provision resources (first time only)
.\scripts\setup_gcp.ps1

# 2. Build and push container
python scripts/build_and_push.py --tag v1.0.0

# 3. Dry run (free)
.\scripts\launch_muzero_training.ps1 -DryRun

# 4. Smoke test (~$1-2)
.\scripts\launch_muzero_training.ps1 -Iterations 1 -Workers 1 -GamesPerWorker 2

# 5. Full PBT training (~$50-75)
.\scripts\launch_muzero_training.ps1 -Iterations 10 -Workers 4 -GamesPerWorker 100
```

## Monitoring

```powershell
# Check job status via CLI
python -m alpha_zero.cloud.gcp.cli status

# List running Vertex AI jobs
cmd /c "gcloud ai custom-jobs list --region=us-central1 --filter=state:JOB_STATE_RUNNING --format=table(displayName,state)"

# View logs in Cloud Console
# https://console.cloud.google.com/logs?project=hrmtrain
```

## Rollback

```powershell
# Revert to previous container image
$env:CIV6AZ_CONTAINER_IMAGE_URI = "us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v0.15.0"

# Cancel running Vertex AI job
gcloud ai custom-jobs cancel JOB_ID --region=us-central1

# List available image versions
gcloud artifacts docker images list us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training
```

## Orchestrator VM (Reboot-Resilient)

The orchestrator runs on a cheap GCE VM and dispatches Vertex AI jobs. It auto-resumes after reboots via systemd.

| Property | Value |
|:---|:---|
| **VM Name** | `civ6az-orchestrator` |
| **Machine Type** | `e2-small` (~$15/month) |
| **Zone** | `us-central1-a` |
| **Service** | `civ6az-orchestrator.service` (systemd) |
| **State** | `gs://civ6az-training/orchestrator/state.json` |
| **Heartbeat** | `gs://civ6az-training/orchestrator/heartbeat.json` |

### Service Management

```bash
sudo systemctl start civ6az-orchestrator     # Start
sudo systemctl stop civ6az-orchestrator      # Stop
sudo systemctl status civ6az-orchestrator    # Status
journalctl -u civ6az-orchestrator -f         # Live logs
```

### Resume Behavior

- State checkpointed to GCS after each completed iteration
- `--resume` flag restores iteration progress, model version, cost accumulator
- Systemd `Restart=on-failure` auto-restarts crashed processes within 30s
- VM `Automatic Restart=On` handles hardware/maintenance reboots

## Teardown

> **Warning**: This deletes ALL resources including training data and models.

```powershell
# PowerShell
.\scripts\setup_gcp.ps1 -Teardown

# Bash/WSL
bash scripts/setup_gcp.sh --teardown
```
