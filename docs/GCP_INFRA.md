# GCP Infrastructure — MuZero-World-Model

> **Project**: `hrmtrain` | **Region**: `us-central1`
> **Created**: 2026-02-20

---

## Resources

| Resource | Name | Purpose |
|----------|------|---------|
| **Artifact Registry** | `civ6az-muzero-world-model` | Docker container images for Vertex AI jobs |
| **GCS Bucket** | `gs://civ6az-muzero-training` | Model checkpoints, trajectories, logs |
| **Container Image** | `civ6az-muzero:v16-full-fix` | Training/Self-play/Orchestrator container |
| **Vertex AI Jobs** | `civ6az-mzwm-*` prefix | All MuZero job display names |
| **W&B Project** | `civ6az-muzero-world-model` | Experiment tracking |

---

## Full Image URI

```
us-central1-docker.pkg.dev/hrmtrain/civ6az-muzero-world-model/civ6az-muzero:v16-full-fix
```

---

## IAM Bindings

| Principal | Role | Scope |
|-----------|------|-------|
| `1019219508001-compute@developer.gserviceaccount.com` | `artifactregistry.reader` | AR repo |
| `1019219508001-compute@developer.gserviceaccount.com` | `storage.objectAdmin` | GCS bucket |
| Cloud Build SA | `artifactregistry.writer` | AR repo |
| Cloud Build SA | `storage.admin` | Cloud Build staging bucket |

---

## Secret Manager Keys

| Secret | Usage |
|--------|-------|
| `WANDB_API_KEY` | W&B experiment tracking authentication |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/gcp_provision_resources.py` | Create AR repo + GCS bucket (IPv4 bypass) |
| `scripts/cloud_build_container.py` | Build + push container via Cloud Build |
| `scripts/gcp_diagnose_and_submit.py` | Submit Vertex AI jobs |
| `scripts/build_and_push.py` | Local Docker build + push to GCR |

---

## Rollback

```bash
# Revert to previous container tag
python scripts/cloud_build_container.py --tag v15-pre-isolation

# Delete new resources (use with caution)
python -c "
import scripts.gcp_provision_resources as p
# No teardown needed — resources are empty
"
```

---

## Cost Notes

| Resource | Cost Model |
|----------|-----------|
| AR repo | Free storage up to 500MB, then $0.10/GB/month |
| GCS bucket | $0.020/GB/month (Standard), no minimum |
| Cloud Build | First 120 min/day free, then $0.003/build-min |
| Vertex AI | Per-job billing (n1-standard-4 + T4 GPU) |

> [!TIP]
> To avoid costs, delete the GCS bucket when not actively training:
> `gsutil rm -r gs://civ6az-muzero-training`
