# Naming Conventions — MuZero / Civ6-AlphaZero Vertex AI

## Environment Variables

| Variable                       | Purpose                            | Example                                                                                  |
|--------------------------------|------------------------------------|------------------------------------------------------------------------------------------|
| `CIV6AZ_VERTEX_PROJECT`       | GCP project ID                     | `hrmtrain`                                                                               |
| `CIV6AZ_VERTEX_REGION`        | GCP region                         | `us-central1`                                                                            |
| `CIV6AZ_GCS_BUCKET`           | GCS bucket for models/trajectories | `civ6az-training`                                                                        |
| `CIV6AZ_CONTAINER_IMAGE_URI`  | Full container image URI for child jobs | `us-central1-docker.pkg.dev/hrmtrain/muzero-world-model/muzero-training:v13-compat` |
| `CIV6AZ_AGENT_ID`             | Algorithm identifier               | `muzero`                                                                                 |
| `CIV6AZ_VERTEX_SERVICE_ACCOUNT` | SA email (optional)              | —                                                                                         |

## Container Image Tags

Format: `muzero-training:<tag>`

| Tag             | Purpose                   |
|-----------------|---------------------------|
| `v11-smoke`     | ❌ Legacy, stale           |
| `v12-muzero`    | ❌ Python 3.10 UTC bug     |
| `v13-compat`    | ✅ Current — `_compat.py` shim for Py3.10 |

## Artifact Registry

| Field                | Value                                              |
|----------------------|----------------------------------------------------|
| Repository           | `muzero-world-model` (project-isolated)             |
| Full image prefix    | `us-central1-docker.pkg.dev/hrmtrain/muzero-world-model/muzero-training` |

## Vertex AI Job Names

Format: `civ6az-{agent_id}-{role}`

| Job Name Pattern                          | Role           |
|-------------------------------------------|----------------|
| `civ6az-muzero-orchestrator`              | Orchestrator   |
| `civ6az-selfplay-iter{NNN}-worker{NN}`   | Self-play      |
| `civ6az-training-iter{NNN}`              | Training       |
| `civ6az-eval-iter{NNN}`                  | Evaluation     |
| `civ6az-tuning-iter{NNN}`               | HP Tuning      |

## GCS Paths

| Path                                       | Contents              |
|--------------------------------------------|-----------------------|
| `gs://civ6az-training/models/`             | Model checkpoints     |
| `gs://civ6az-training/trajectories/`       | Self-play game data   |
| `gs://civ6az-training/logs/`               | Job logs              |

## Config Files

| File                                  | Purpose              |
|---------------------------------------|----------------------|
| `configs/vertex_orchestrator_cloud.yaml` | Production orchestrator |
| `configs/vertex_orchestrator_smoke.yaml` | Smoke tests          |
| `configs/vertex_selfplay_smoke.yaml`     | Self-play smoke test |
| `configs/vertex_training_smoke.yaml`     | Training smoke test  |
