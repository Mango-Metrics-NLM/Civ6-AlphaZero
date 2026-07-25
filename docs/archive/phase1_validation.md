# Phase 1: Validation Execution Guide

This guide documents how to execute and verify the Phase 1 validation suite for the Civ6 AlphaZero project.

## Overview

Phase 1 focuses on immediate validation of:

1. **GCP E2E Integration** - Validates cloud infrastructure with live credentials
2. **Local Training Smoke Test** - Validates the training pipeline locally

## Prerequisites

### Environment Setup

1. **Python 3.10+** is required:

   ```bash
   python --version  # Should show 3.10.x or higher
   ```

2. **Install dependencies**:

   ```bash
   pip install -e ".[dev,gcp]"
   ```

3. **GCP Authentication** (for E2E tests):

   ```bash
   # Login with application default credentials
   gcloud auth application-default login
   
   # Set required environment variables
   export GCP_PROJECT_ID="your-project-id"
   export GCS_BUCKET_NAME="your-bucket-name"
   ```

### Verify GCP Credentials

Run the credential validation script:

```bash
python scripts/validate_gcp_credentials.py
```

Expected output shows validation status for:

- ✅ Environment Variables
- ✅ Service Account Key / Application Default Credentials
- ✅ GCS Access
- ✅ Vertex AI Access
- ✅ Cloud Logging Access

## Validation Commands

### Full Phase 1 Validation

```bash
# Run complete Phase 1 validation
python scripts/validate_phase1.py

# Skip GCP tests (offline mode)
python scripts/validate_phase1.py --skip-gcp

# Skip training smoke test
python scripts/validate_phase1.py --skip-training

# Verbose output
python scripts/validate_phase1.py --verbose
```

### Individual Components

#### GCP E2E Tests

```bash
# Run all GCP integration tests
pytest tests/e2e/test_gcp_integration.py -v

# Run specific test class
pytest tests/e2e/test_gcp_integration.py::TestGCSIntegration -v

# Run with detailed output
pytest tests/e2e/test_gcp_integration.py -v --tb=short -s
```

#### Preemption Handler Tests

```bash
# Run preemption tests (mock mode, no GCP required)
pytest tests/e2e/test_preemption_handling.py -v
```

#### Training Smoke Tests

```bash
# Run training unit tests
pytest tests/unit/test_training_smoke.py -v

# Run full dry-run training
python -m alpha_zero.training_civ6 --dry_run
```

## Expected Outputs

### GCP E2E Tests (9 tests)

| Test | Description |
|------|-------------|
| `test_gcs_client_initialization` | Verifies GCS client setup |
| `test_gcs_upload_download_checkpoint` | Tests checkpoint upload/download |
| `test_gcs_list_checkpoints` | Tests listing checkpoints |
| `test_vertex_trainer_initialization` | Verifies Vertex AI trainer setup |
| `test_vertex_job_spec_validation` | Validates job specifications |
| `test_cloud_logger_initialization` | Verifies Cloud Logger setup |
| `test_cloud_logger_log_event` | Tests event logging |
| `test_full_workflow_initialization` | Tests CloudTrainingCoordinator |
| `test_full_workflow_config` | Validates coordinator configuration |

### Training Smoke Test Outputs

After running `--dry_run`:

- `checkpoints/civ6/` - Contains checkpoint files (`.pt`)
- `logs/civ6/` - Contains training logs

## Troubleshooting

### Common Issues

#### "GCP credentials not found"

```bash
# Verify ADC is set up
gcloud auth application-default print-access-token

# Or set explicit credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

#### "Bucket not found"

```bash
# Verify bucket exists
gsutil ls gs://${GCS_BUCKET_NAME}

# Create bucket if needed
gsutil mb -p ${GCP_PROJECT_ID} gs://${GCS_BUCKET_NAME}
```

#### "Permission denied" errors

Ensure your service account or user has these IAM roles:

- `roles/storage.objectAdmin` (for GCS)
- `roles/aiplatform.user` (for Vertex AI)
- `roles/logging.logWriter` (for Cloud Logging)

#### Windows-specific file locking

If you see "PermissionError: Access is denied":

- Close any open checkpoint files
- Use `tempfile.TemporaryDirectory()` for test directories
- Ensure file handles are properly closed

### Debug Mode

For detailed logging:

```bash
# Set log level
export CIV6AZ_LOG_LEVEL=DEBUG

# Run with verbose
pytest tests/e2e/ -v -s --log-cli-level=DEBUG
```

## Success Criteria

Phase 1 is complete when:

| Criterion | Status |
|-----------|--------|
| All 9 E2E GCP tests pass | [ ] |
| Training dry-run completes | [ ] |
| Checkpoints created | [ ] |
| Logs written | [ ] |
| No ruff/mypy errors | [ ] |
| Test coverage ≥80% | [ ] |

## Next Steps

After Phase 1 validation:

1. **Phase 2**: Extended Training Runs (GPU-accelerated)
2. **Phase 3**: Multi-node Distributed Training
3. **Phase 4**: Advanced Features (Meta-learning)
4. **Phase 5**: Production Deployment

## References

- [Cloud Infrastructure Documentation](../alpha_zero/cloud/README.md)
- [Training Pipeline Guide](../alpha_zero/training_civ6.py)
- [Test Suite Documentation](../tests/README.md)
