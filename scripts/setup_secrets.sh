#!/usr/bin/env bash
# Migrate WANDB_API_KEY to GCP Secret Manager
#
# This script:
#   1. Creates a Secret Manager secret for the W&B API key
#   2. Grants the training service account read access
#   3. Verifies the secret can be read
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - GOOGLE_CLOUD_PROJECT env var set
#   - WANDB_API_KEY env var set (the key to migrate)
#   - Service account email known
#
# Usage:
#   export GOOGLE_CLOUD_PROJECT=your-project-id
#   export WANDB_API_KEY=your-wandb-key
#   export CIV6AZ_SERVICE_ACCOUNT=sa@project.iam.gserviceaccount.com
#   bash scripts/setup_secrets.sh

set -euo pipefail

# --- Configuration (from env vars) ---
PROJECT="${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}"
WANDB_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"
SA_EMAIL="${CIV6AZ_SERVICE_ACCOUNT:?Set CIV6AZ_SERVICE_ACCOUNT}"
SECRET_NAME="${CIV6AZ_WANDB_SECRET_NAME:-wandb-api-key}"

echo "=== Secret Manager Migration ==="
echo "Project: ${PROJECT}"
echo "Secret:  ${SECRET_NAME}"
echo "SA:      ${SA_EMAIL}"
echo ""

# --- 1. Enable Secret Manager API ---
echo "[1/4] Ensuring Secret Manager API is enabled..."
gcloud services enable secretmanager.googleapis.com --project="${PROJECT}" 2>/dev/null

# --- 2. Create or update the secret ---
echo "[2/4] Creating secret '${SECRET_NAME}'..."
if gcloud secrets describe "${SECRET_NAME}" --project="${PROJECT}" &>/dev/null; then
    echo "  Secret already exists — adding new version"
    echo -n "${WANDB_KEY}" | gcloud secrets versions add "${SECRET_NAME}" \
        --project="${PROJECT}" \
        --data-file=-
else
    echo -n "${WANDB_KEY}" | gcloud secrets create "${SECRET_NAME}" \
        --project="${PROJECT}" \
        --replication-policy="automatic" \
        --data-file=-
fi

# --- 3. Grant access to training service account ---
echo "[3/4] Granting secretAccessor role to ${SA_EMAIL}..."
gcloud secrets add-iam-policy-binding "${SECRET_NAME}" \
    --project="${PROJECT}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None \
    2>/dev/null || echo "  (binding may already exist)"

# --- 4. Verify ---
echo "[4/4] Verifying secret access..."
RETRIEVED=$(gcloud secrets versions access latest \
    --secret="${SECRET_NAME}" \
    --project="${PROJECT}" 2>/dev/null)

if [ "${RETRIEVED}" = "${WANDB_KEY}" ]; then
    echo "  ✅ Secret verified successfully"
else
    echo "  ❌ Secret verification failed"
    exit 1
fi

echo ""
echo "=== Migration complete ==="
echo ""
echo "To use in Vertex AI jobs, the training code should read:"
echo "  from google.cloud import secretmanager"
echo "  client = secretmanager.SecretManagerServiceClient()"
echo "  name = f'projects/${PROJECT}/secrets/${SECRET_NAME}/versions/latest'"
echo "  response = client.access_secret_version(request={'name': name})"
echo "  wandb_key = response.payload.data.decode('UTF-8')"
