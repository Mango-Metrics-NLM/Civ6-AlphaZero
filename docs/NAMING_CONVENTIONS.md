# Naming Conventions — MuZero / Civ6-AlphaZero Vertex AI

## Environment Variables

| Variable                       | Purpose                            | Example                                                                                  |
|--------------------------------|------------------------------------|------------------------------------------------------------------------------------------|
| `CIV6AZ_VERTEX_PROJECT`       | GCP project ID                     | `hrmtrain`                                                                               |
| `CIV6AZ_VERTEX_REGION`        | GCP region                         | `us-central1`                                                                            |
| `CIV6AZ_GCS_BUCKET`           | GCS bucket for models/trajectories | `<your-bucket>`                                                                        |
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
| `civ6az-mzwm-training-iter{NNN}`              | Training       |
| `civ6az-eval-iter{NNN}`                  | Evaluation     |
| `civ6az-tuning-iter{NNN}`               | HP Tuning      |

## GCS Paths

| Path                                       | Contents              |
|--------------------------------------------|-----------------------|
| `gs://<training-bucket>/models/`             | Model checkpoints     |
| `gs://<training-bucket>/trajectories/`       | Self-play game data   |
| `gs://<training-bucket>/logs/`               | Job logs              |

## Config Files

| File                                  | Purpose              |
|---------------------------------------|----------------------|
| `configs/vertex_orchestrator_cloud.example.yaml` | Production orchestrator |
| `configs/vertex_orchestrator_smoke.example.yaml` | Smoke tests          |
| `configs/vertex_selfplay_smoke.example.yaml`     | Self-play smoke test |
| `configs/vertex_training_smoke.example.yaml`     | Training smoke test  |

The `*.example.yaml` files carry `YOUR_*` placeholders. Copy one to the
gitignored `configs/local/` directory and fill in your project and bucket there.
