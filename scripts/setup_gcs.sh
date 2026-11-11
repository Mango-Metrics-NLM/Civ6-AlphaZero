#!/bin/bash
#
# Setup GCS Bucket for Civ6 AlphaZero Training
#
# This script creates the required GCS bucket structure for Vertex AI training.
# Run this once before starting training.
#
# Usage:
#   # Set environment variables first
#   export CIV6AZ_GCS_BUCKET=your-bucket-name
#   export CIV6AZ_VERTEX_REGION=us-central1
#
#   # Run the script
#   ./scripts/setup_gcs.sh
#
# Required tools:
#   - gcloud CLI (authenticated)
#   - gsutil

set -e

# Configuration from environment
BUCKET="${CIV6AZ_GCS_BUCKET:-}"
REGION="${CIV6AZ_VERTEX_REGION:-us-central1}"
PROJECT="${CIV6AZ_VERTEX_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"

# Validate configuration
if [[ -z "$BUCKET" ]]; then
    echo "ERROR: CIV6AZ_GCS_BUCKET environment variable not set"
    echo ""
    echo "Usage:"
    echo "  export CIV6AZ_GCS_BUCKET=your-bucket-name"
    echo "  ./scripts/setup_gcs.sh"
    exit 1
fi

if [[ -z "$PROJECT" ]]; then
    echo "ERROR: No GCP project configured"
    echo "  Run: gcloud config set project YOUR_PROJECT"
    echo "  Or set: export CIV6AZ_VERTEX_PROJECT=your-project-id"
    exit 1
fi

echo "=============================================="
echo "Civ6 AlphaZero GCS Bucket Setup"
echo "=============================================="
echo ""
echo "Project: $PROJECT"
echo "Region:  $REGION"
echo "Bucket:  gs://$BUCKET"
echo ""

# Create bucket if it doesn't exist
echo "[1/4] Creating bucket..."
if gsutil ls -b "gs://$BUCKET" > /dev/null 2>&1; then
    echo "  Bucket gs://$BUCKET already exists"
else
    echo "  Creating gs://$BUCKET..."
    gsutil mb -p "$PROJECT" -l "$REGION" "gs://$BUCKET"
    echo "  Created successfully"
fi

# Set bucket lifecycle rules (optional - clean up old artifacts)
echo ""
echo "[2/4] Configuring lifecycle rules..."

# Create a temporary lifecycle config file
LIFECYCLE_CONFIG=$(mktemp)
cat > "$LIFECYCLE_CONFIG" << 'EOF'
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 90,
          "matchesPrefix": ["trajectories/"]
        }
      },
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 30,
          "matchesPrefix": ["logs/"]
        }
      }
    ]
  }
}
EOF

gsutil lifecycle set "$LIFECYCLE_CONFIG" "gs://$BUCKET" || echo "  Warning: Could not set lifecycle rules"
rm -f "$LIFECYCLE_CONFIG"
echo "  Lifecycle rules configured (trajectories: 90d, logs: 30d)"

# Create directory structure using marker files
echo ""
echo "[3/4] Creating directory structure..."

# Create marker files to establish the directory structure
# GCS doesn't have real directories, but these help organize
echo '{"created": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}' | gsutil -q cp - "gs://$BUCKET/models/civ6az/.metadata.json"
echo '{"created": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}' | gsutil -q cp - "gs://$BUCKET/trajectories/civ6az/.metadata.json"
echo '{"created": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}' | gsutil -q cp - "gs://$BUCKET/checkpoints/civ6az/.metadata.json"
echo '{"created": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}' | gsutil -q cp - "gs://$BUCKET/logs/civ6az/.metadata.json"

echo "  Created:"
echo "    gs://$BUCKET/models/civ6az/"
echo "    gs://$BUCKET/trajectories/civ6az/"
echo "    gs://$BUCKET/checkpoints/civ6az/"
echo "    gs://$BUCKET/logs/civ6az/"

# Verify the setup
echo ""
echo "[4/4] Verifying setup..."
gsutil ls "gs://$BUCKET/" | head -10

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Build container: gcloud builds submit --config=dockerfiles/cloudbuild.yaml"
echo "  2. Initialize:      python -m alpha_zero.cloud.gcp init --upload-model"
echo "  3. Start training:  python -m alpha_zero.cloud.gcp run --iterations 50"
echo ""
echo "Environment variables to set:"
echo "  export CIV6AZ_VERTEX_PROJECT=$PROJECT"
echo "  export CIV6AZ_GCS_BUCKET=$BUCKET"
echo "  export CIV6AZ_VERTEX_REGION=$REGION"
echo "  export CIV6AZ_CONTAINER_IMAGE_URI=gcr.io/$PROJECT/alphazero-training:latest"
