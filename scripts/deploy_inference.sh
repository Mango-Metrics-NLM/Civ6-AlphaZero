#!/bin/bash
# Deploy Civ6 AlphaZero Inference Service
#
# Usage:
#   ./scripts/deploy_inference.sh [cloud-run|gke] [version-tag]

set -euo pipefail

PLATFORM="${1:-cloud-run}"
VERSION="${2:-latest}"
PROJECT="${CIV6AZ_VERTEX_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${CIV6AZ_VERTEX_REGION:-us-central1}"
REPO="${CIV6AZ_ARTIFACT_REPO:-civ6az}"
IMAGE_NAME="${CIV6AZ_INFERENCE_SERVICE:-alphazero-inference}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE_NAME}:${VERSION}"

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

    # Update image in Cloud Run config
    sed -i.bak "s|image: .*|image: ${IMAGE_URI}|" cloud-run-inference.yaml

    # Deploy to Cloud Run
    gcloud run services replace cloud-run-inference.yaml \
        --region="${REGION}" \
        --project="${PROJECT}"

    # Get service URL
    SERVICE_URL=$(gcloud run services describe "${IMAGE_NAME}" \
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
    kubectl set image deployment/"${IMAGE_NAME}" \
        inference="${IMAGE_URI}" \
        -n civ6az

    # Wait for rollout
    kubectl rollout status deployment/"${IMAGE_NAME}" -n civ6az

    # Get service endpoint
    SERVICE_IP=$(kubectl get svc "${IMAGE_NAME}-service" \
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
