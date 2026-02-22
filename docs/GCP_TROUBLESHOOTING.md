# GCP Troubleshooting Guide

## Quick Fix: `gcloud` Hangs on Windows

The `gcloud` CLI hangs indefinitely on Windows during component update checks.

**Fix**: Set this environment variable before using `gcloud`:

```powershell
$env:CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK = '1'
$env:CLOUDSDK_CORE_DISABLE_PROMPTS = '1'
```

Both values are included in `.env`. Source them with:

```powershell
. .\scripts\setup_env.ps1
```

---

## Common Issues

### PowerShell Execution Policy Blocks `gcloud.ps1`

**Symptom**: `SecurityError` when running `gcloud` in PowerShell.

**Fix**: Use `cmd /c "gcloud ..."` or call `gcloud.cmd` directly:

```powershell
cmd /c "gcloud ai custom-jobs list --region=us-central1"
```

### Environment Variables Not Loaded

**Symptom**: `CIV6AZ_VERTEX_PROJECT not set` in diagnostic output.

**Fix**: Source `.env` into your session:

```powershell
. .\scripts\setup_env.ps1
```

### ADC Credentials Not Found

**Symptom**: `Your default credentials were not found` during Vertex AI operations.

**Fix**:

```powershell
cmd /c "gcloud auth application-default login"
```

### GPU Quota Exceeded (L4)

**Symptom**: `429 Quota exceeded ... nvidia_l4_gpus`

**Fix**: Use T4 GPUs instead of L4 (already configured in budget tier).

---

## Diagnostic Utility

Run the built-in diagnostic tool to check all GCP connectivity:

```powershell
# Load env vars first
. .\scripts\setup_env.ps1

# Run diagnostic (text output)
python scripts/gcp_diagnose.py

# JSON output for CI/automation
python scripts/gcp_diagnose.py --json

# Auto-fix known issues
python scripts/gcp_diagnose.py --fix
```

### Checks Performed

| Check | What it validates |
|-------|-------------------|
| `network_connectivity` | HTTPS to `storage.googleapis.com` + `aiplatform.googleapis.com` |
| `gcloud_executable` | `gcloud` or `gcloud.cmd` in PATH |
| `update_check_disabled` | `CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK=1` |
| `gcloud_auth` | Active account via `gcloud auth list` |
| `python_adc` | Application Default Credentials via `google.auth.default()` |
| `env_vars` | `CIV6AZ_VERTEX_PROJECT` and `CIV6AZ_GCS_BUCKET` set |
| `gcs_connectivity` | Named bucket exists and is accessible |
| `vertex_api` | Vertex AI Custom Jobs API responds |
