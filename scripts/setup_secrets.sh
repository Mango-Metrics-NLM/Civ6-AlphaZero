#!/usr/bin/env bash
# Provision a secret in GCP Secret Manager and grant the training service
# account read access.
#
# Generic helper — no secret name or value is hardcoded. The W&B-era default
# (wandb-api-key) is gone along with the W&B integration; a typical use today
# is provisioning an MLflow tracking token for a remote tracking server.
#
# This script:
#   1. Creates (or versions) the Secret Manager secret
#   2. Grants the service account read access
#   3. Verifies the secret can be read back
#
# Prerequisites:
#   - gcloud CLI authenticated
#
# Usage:
#   export GOOGLE_CLOUD_PROJECT=your-project-id
#   export CIV6AZ_SERVICE_ACCOUNT=sa@project.iam.gserviceaccount.com
#   export CIV6AZ_SECRET_NAME=mlflow-tracking-token
#   export CIV6AZ_SECRET_VALUE=...   # the secret material
#   bash scripts/setup_secrets.sh

set -euo pipefail

# --- Configuration (from env vars; no defaults for secret material) ---
PROJECT="${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}"
SA_EMAIL="${CIV6AZ_SERVICE_ACCOUNT:?Set CIV6AZ_SERVICE_ACCOUNT}"
SECRET_NAME="${CIV6AZ_SECRET_NAME:?Set CIV6AZ_SECRET_NAME (e.g. mlflow-tracking-token)}"
SECRET_VALUE="${CIV6AZ_SECRET_VALUE:?Set CIV6AZ_SECRET_VALUE}"

echo "=== Secret Manager Provisioning ==="
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
    echo -n "${SECRET_VALUE}" | gcloud secrets versions add "${SECRET_NAME}" \
        --project="${PROJECT}" \
        --data-file=-
else
    echo -n "${SECRET_VALUE}" | gcloud secrets create "${SECRET_NAME}" \
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

if [ "${RETRIEVED}" = "${SECRET_VALUE}" ]; then
    echo "  ✅ Secret verified successfully"
else
    echo "  ❌ Secret verification failed"
    exit 1
fi

echo ""
echo "=== Provisioning complete ==="
echo ""
echo "To read the secret at runtime:"
echo "  from google.cloud import secretmanager"
echo "  client = secretmanager.SecretManagerServiceClient()"
echo "  name = f'projects/${PROJECT}/secrets/${SECRET_NAME}/versions/latest'"
echo "  response = client.access_secret_version(request={'name': name})"
echo "  value = response.payload.data.decode('UTF-8')"
