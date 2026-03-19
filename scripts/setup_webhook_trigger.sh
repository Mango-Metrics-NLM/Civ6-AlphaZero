#!/usr/bin/env bash
# ============================================================================
# setup_webhook_trigger.sh — Create a Cloud Build webhook trigger
#
# This creates a webhook-based Cloud Build trigger that doesn't require
# the Google Cloud Build GitHub App to be installed. Instead, it uses
# a webhook URL + secret that you configure in GitHub manually.
#
# Usage (run in Cloud Shell):
#   bash scripts/setup_webhook_trigger.sh
#
# After running, configure the webhook in GitHub:
#   Repo Settings → Webhooks → Add webhook → paste the URL
# ============================================================================

set -euo pipefail

PROJECT="hrmtrain"
REGION="us-central1"
AR_REPO="civ6az"
TRIGGER_NAME="civ6az-container-build"
SECRET_NAME="civ6az-webhook-secret"

echo "=== Step 1: Enable Secret Manager API ==="
gcloud services enable secretmanager.googleapis.com --project="$PROJECT" 2>&1
echo "✓ Secret Manager API enabled"

echo ""
echo "=== Step 2: Clean up stale 2nd-gen connections ==="
gcloud builds connections delete civ6az-github-connection \
  --region="$REGION" --project="$PROJECT" --quiet 2>&1 || echo "  (no stale connection to delete)"
echo "✓ Stale connections cleaned"

echo ""
echo "=== Step 3: Delete any existing trigger with same name ==="
gcloud builds triggers delete "$TRIGGER_NAME" \
  --project="$PROJECT" --quiet 2>&1 || echo "  (no existing trigger to delete)"
echo "✓ Ready for new trigger"

echo ""
echo "=== Step 4: Create webhook trigger ==="
# Create the webhook trigger with inline build config that clones + runs cloudbuild
gcloud builds triggers create webhook \
  --project="$PROJECT" \
  --name="$TRIGGER_NAME" \
  --description="Build civ6az-training container on push to main" \
  --secret="projects/$PROJECT/secrets/$SECRET_NAME/versions/latest" \
  --substitutions='_IMAGE_TAG=latest,_REGION=us-central1,_AR_REPO=civ6az,_BRANCH=$(body.ref)' \
  --inline-config=- <<'BUILDEOF'
steps:
  # Clone the repo
  - name: 'gcr.io/cloud-builders/git'
    args: ['clone', '--depth=1', '--branch=main',
           'https://github.com/Mango-Metrics-NLM/Civ6-AlphaZero.git', '.']
    id: 'clone'

  # Lint check
  - name: 'python:3.12-slim'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        pip install --quiet ruff && \
        ruff check alpha_zero/ --select E,F --ignore E501 --exit-non-zero-on-fix
    id: 'lint'
    waitFor: ['clone']

  # Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_AR_REPO}/civ6az-training:${_IMAGE_TAG}'
      - '-f'
      - 'dockerfiles/Dockerfile.vertex'
      - '.'
    id: 'build'
    waitFor: ['lint']

  # Smoke test
  - name: '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_AR_REPO}/civ6az-training:${_IMAGE_TAG}'
    entrypoint: 'python'
    args:
      - '-c'
      - 'import sys; sys.path.insert(0, "/app"); from alpha_zero.cloud.gcp.config import GCPConfig; print("Smoke test passed")'
    dir: '/app'
    id: 'smoke-test'
    waitFor: ['build']

  # Push to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_AR_REPO}/civ6az-training:${_IMAGE_TAG}'
    id: 'push'
    waitFor: ['smoke-test']

images:
  - '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_AR_REPO}/civ6az-training:${_IMAGE_TAG}'

options:
  logging: GCS_ONLY
  machineType: 'E2_HIGHCPU_8'

timeout: '1800s'

tags:
  - 'civ6az'
  - 'training'
  - 'webhook'
BUILDEOF

TRIGGER_EXIT=$?
echo ""
if [ $TRIGGER_EXIT -eq 0 ]; then
  echo "✓ Webhook trigger created successfully!"
  echo ""
  echo "=== Next Steps ==="
  echo "1. Get the webhook URL:"
  echo "   gcloud builds triggers describe $TRIGGER_NAME --project=$PROJECT --format='value(webhookConfig.url)'"
  echo ""
  echo "2. Configure in GitHub:"
  echo "   Repo → Settings → Webhooks → Add webhook"
  echo "   - Payload URL: <webhook URL from step 1>"
  echo "   - Content type: application/json"
  echo "   - Secret: <leave empty, Cloud Build handles auth>"
  echo "   - Events: Just the push event"
  echo ""
  echo "=== Verify ===" 
  gcloud builds triggers list --project="$PROJECT" --format="table(name,webhookConfig.state)" 2>&1
else
  echo "✗ Trigger creation failed (exit=$TRIGGER_EXIT)"
  echo ""
  echo "=== Fallback: Try 1st-gen GitHub trigger with lowercase repo ==="
  echo "The repo name 'Civ6-AlphaZero' has uppercase letters which can cause INVALID_ARGUMENT."
  echo "Try using Cloud Build UI: https://console.cloud.google.com/cloud-build/triggers?project=$PROJECT"
fi
