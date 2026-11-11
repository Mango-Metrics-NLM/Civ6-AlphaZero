#!/bin/bash
# Deploy Civ6 AlphaZero Inference Service
#
# Usage:
#   ./scripts/deploy_inference.sh [cloud-run|gke] [version-tag]
#   RENDER_ONLY=1 ./scripts/deploy_inference.sh cloud-run [version-tag]
#       Render the Cloud Run manifest and print its path; build and deploy nothing.
#
# Required environment:
#   CIV6AZ_VERTEX_PROJECT   GCP project (falls back to `gcloud config get-value project`)
#   CIV6AZ_GCS_BUCKET       bucket the server loads model weights from
# Optional:
#   CIV6AZ_VERTEX_REGION              (default us-central1)
#   CIV6AZ_ARTIFACT_REPO              Artifact Registry repository (default civ6az)
#   CIV6AZ_INFERENCE_IMAGE            image name inside the repository (default alphazero-inference)
#   CIV6AZ_INFERENCE_SERVICE          Cloud Run service / k8s deployment name (default alphazero-inference)
#   CIV6AZ_INFERENCE_SERVICE_ACCOUNT  runtime service account
#                                     (default alphazero-inference@<project>.iam.gserviceaccount.com)
#
# cloud-run-inference.yaml is a template with YOUR_* placeholders. It is rendered into a
# temporary copy (the tracked file is never edited), and the deploy refuses to proceed if
# any placeholder survives rendering.

set -euo pipefail

PLATFORM="${1:-cloud-run}"
VERSION="${2:-latest}"
RENDER_ONLY="${RENDER_ONLY:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_TEMPLATE="${CIV6AZ_CLOUD_RUN_MANIFEST:-${SCRIPT_DIR}/../cloud-run-inference.yaml}"

PROJECT="${CIV6AZ_VERTEX_PROJECT:-}"
if [ -z "${PROJECT}" ] && command -v gcloud >/dev/null 2>&1; then
    PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [ -z "${PROJECT}" ]; then
    echo "ERROR: set CIV6AZ_VERTEX_PROJECT (or a gcloud default project); there is no built-in project." >&2
    exit 2
fi
BUCKET="${CIV6AZ_GCS_BUCKET:-}"
if [ -z "${BUCKET}" ]; then
    echo "ERROR: set CIV6AZ_GCS_BUCKET to the bucket the server loads weights from; there is no default." >&2
    exit 2
fi
REGION="${CIV6AZ_VERTEX_REGION:-us-central1}"
REPO="${CIV6AZ_ARTIFACT_REPO:-civ6az}"
IMAGE_REPO_NAME="${CIV6AZ_INFERENCE_IMAGE:-alphazero-inference}"
SERVICE_NAME="${CIV6AZ_INFERENCE_SERVICE:-alphazero-inference}"
SERVICE_ACCOUNT="${CIV6AZ_INFERENCE_SERVICE_ACCOUNT:-alphazero-inference@${PROJECT}.iam.gserviceaccount.com}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE_REPO_NAME}:${VERSION}"

# Render the Cloud Run manifest into a temporary file and print its path.
# Substitutes the project, bucket, service account, image and service name, then
# refuses the result if any YOUR_* placeholder is left.
render_cloud_run_manifest() {
    local rendered
    rendered="$(mktemp "${TMPDIR:-/tmp}/cloud-run-inference.XXXXXX")"
    sed \
        -e "s|^\(  name: \).*|\1${SERVICE_NAME}|" \
        -e "s|\(serviceAccountName: \)[^ ]*|\1${SERVICE_ACCOUNT}|" \
        -e "s|\(image: \).*|\1${IMAGE_URI}|" \
        -e "s|YOUR_GCP_PROJECT|${PROJECT}|g" \
        -e "s|YOUR_BUCKET|${BUCKET}|g" \
        "${MANIFEST_TEMPLATE}" > "${rendered}"
    if grep -q 'YOUR_' "${rendered}"; then
        echo "ERROR: unrendered placeholders remain in ${rendered}:" >&2
        grep -n 'YOUR_' "${rendered}" >&2
        exit 3
    fi
    echo "${rendered}"
}

if [ "${RENDER_ONLY}" = "1" ]; then
    render_cloud_run_manifest
    exit 0
fi

echo "=== Civ6 AlphaZero Inference Deployment ==="
echo "Platform: ${PLATFORM}"
echo "Version: ${VERSION}"
echo "Image: ${IMAGE_URI}"
echo ""

# Build and push Docker image
echo "[1/3] Building Docker image..."
docker build -f dockerfiles/Dockerfile.inference -t "${IMAGE_URI}" .

echo "[2/3] Pushing to Artifact Registry..."
docker push "${IMAGE_URI}"

# Deploy based on platform
if [ "${PLATFORM}" == "cloud-run" ]; then
    echo "[3/3] Deploying to Cloud Run..."

    # Render the template (never edit the tracked file) and deploy the rendered copy
    RENDERED_MANIFEST="$(render_cloud_run_manifest)"

    gcloud run services replace "${RENDERED_MANIFEST}" \
        --region="${REGION}" \
        --project="${PROJECT}"

    # Get service URL
    SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
        --region="${REGION}" \
        --project="${PROJECT}" \
        --format="value(status.url)")

    echo ""
    echo "=== Deployment Complete ==="
    echo "Service URL: ${SERVICE_URL}"
    echo "Health Check: ${SERVICE_URL}/health"
    echo "API Docs: ${SERVICE_URL}/docs"

elif [ "${PLATFORM}" == "gke" ]; then
    echo "[3/3] Deploying to GKE..."

    # Update image in K8s deployment
    kubectl set image deployment/"${SERVICE_NAME}" \
        inference="${IMAGE_URI}" \
        -n civ6az

    # Wait for rollout
    kubectl rollout status deployment/"${SERVICE_NAME}" -n civ6az

    # Get service endpoint
    SERVICE_IP=$(kubectl get svc "${SERVICE_NAME}-service" \
        -n civ6az \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

    echo ""
    echo "=== Deployment Complete ==="
    echo "Service IP: ${SERVICE_IP}"
    echo "Health Check: http://${SERVICE_IP}/health"
    echo "API Docs: http://${SERVICE_IP}/docs"

else
    echo "ERROR: Unknown platform '${PLATFORM}'. Use 'cloud-run' or 'gke'."
    exit 1
fi

echo ""
echo "Test with:"
echo "curl ${SERVICE_URL:-http://${SERVICE_IP}}/health"
