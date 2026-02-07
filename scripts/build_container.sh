#!/usr/bin/env bash
# Build and push the Civ6 AlphaZero training container to GCR.
#
# Usage:
#   ./scripts/build_container.sh          # Build and push
#   ./scripts/build_container.sh --local  # Build only (no push)

set -euo pipefail

PROJECT_ID="${CIV6AZ_VERTEX_PROJECT:-hrmtrain}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/civ6az-training"
TAG="${1:-latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Civ6 AlphaZero Container Build ==="
echo "Project: ${PROJECT_ID}"
echo "Image:   ${IMAGE_NAME}:${TAG}"
echo "Context: ${PROJECT_DIR}"
echo ""

# Build the image
echo "Building Docker image..."
docker build -t "${IMAGE_NAME}:${TAG}" "${PROJECT_DIR}"

echo ""
echo "Build complete: ${IMAGE_NAME}:${TAG}"

# Push unless --local flag
if [[ "${TAG}" == "--local" || "${2:-}" == "--local" ]]; then
    echo "Local build only - skipping push"
    exit 0
fi

echo ""
echo "Pushing to GCR..."
docker push "${IMAGE_NAME}:${TAG}"

echo ""
echo "=== Done ==="
echo "Image pushed: ${IMAGE_NAME}:${TAG}"
echo ""
echo "Verify with:"
echo "  gcloud container images list-tags ${IMAGE_NAME}"
