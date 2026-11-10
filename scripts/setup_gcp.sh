#!/usr/bin/env bash
# =============================================================================
# Civ6 AlphaZero - GCP Resource Provisioning Script
#
# Creates all GCP resources required for Vertex AI training in the correct
# dependency order. Idempotent - safe to re-run.
#
# Usage:
#   bash scripts/setup_gcp.sh              # Provision all resources
#   bash scripts/setup_gcp.sh --teardown   # Delete all resources
#   bash scripts/setup_gcp.sh --dry-run    # Print commands without executing
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Billing enabled on the project
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
PROJECT_ID="${CIV6AZ_VERTEX_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${CIV6AZ_VERTEX_REGION:-us-central1}"
# No default bucket: a bucket-name default is a takeover vector.
BUCKET_NAME="${CIV6AZ_GCS_BUCKET:?CIV6AZ_GCS_BUCKET must be set (there is no default bucket)}"
AR_REPO_NAME="civ6az"
SA_NAME="civ6az-vertex"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Secret Manager keys to create (values pulled from .env or prompted)
SECRET_KEYS=("wandb-api-key" "openai-api-key" "anthropic-api-key")

# APIs to enable
REQUIRED_APIS=(
    "aiplatform.googleapis.com"
    "storage.googleapis.com"
    "cloudbuild.googleapis.com"
    "artifactregistry.googleapis.com"
    "secretmanager.googleapis.com"
    "monitoring.googleapis.com"
    "containerregistry.googleapis.com"
)

# IAM roles for the service account
SA_ROLES=(
    "roles/aiplatform.user"
    "roles/storage.objectAdmin"
    "roles/artifactregistry.reader"
    "roles/secretmanager.secretAccessor"
    "roles/logging.logWriter"
    "roles/monitoring.metricWriter"
)

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
TEARDOWN=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --teardown) TEARDOWN=true ;;
        --dry-run)  DRY_RUN=true ;;
        *)          echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
COLOR_GREEN="\033[0;32m"
COLOR_YELLOW="\033[0;33m"
COLOR_RED="\033[0;31m"
COLOR_CYAN="\033[0;36m"
COLOR_RESET="\033[0m"

info()  { echo -e "${COLOR_CYAN}[INFO]${COLOR_RESET}  $*"; }
ok()    { echo -e "${COLOR_GREEN}[OK]${COLOR_RESET}    $*"; }
warn()  { echo -e "${COLOR_YELLOW}[WARN]${COLOR_RESET}  $*"; }
err()   { echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $*"; }

run_cmd() {
    if $DRY_RUN; then
        echo "  [DRY-RUN] $*"
    else
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------
if $TEARDOWN; then
    echo ""
    echo "============================================================"
    echo " TEARDOWN - Deleting Civ6 AlphaZero GCP Resources"
    echo "============================================================"
    echo ""
    warn "This will delete ALL provisioned resources for project: $PROJECT_ID"
    read -r -p "Are you sure? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        info "Teardown cancelled."
        exit 0
    fi

    info "Deleting Secret Manager secrets..."
    for secret in "${SECRET_KEYS[@]}"; do
        run_cmd gcloud secrets delete "$secret" --project="$PROJECT_ID" --quiet 2>/dev/null || true
    done

    info "Removing IAM bindings for service account..."
    for role in "${SA_ROLES[@]}"; do
        run_cmd gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$SA_EMAIL" \
            --role="$role" --quiet 2>/dev/null || true
    done

    info "Deleting service account..."
    run_cmd gcloud iam service-accounts delete "$SA_EMAIL" \
        --project="$PROJECT_ID" --quiet 2>/dev/null || true

    info "Deleting Artifact Registry repository..."
    run_cmd gcloud artifacts repositories delete "$AR_REPO_NAME" \
        --location="$REGION" --project="$PROJECT_ID" --quiet 2>/dev/null || true

    info "Deleting GCS bucket..."
    run_cmd gcloud storage rm -r "gs://${BUCKET_NAME}" 2>/dev/null || true

    ok "Teardown complete."
    exit 0
fi

# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Civ6 AlphaZero - GCP Resource Provisioning"
echo "============================================================"
echo ""
info "Project:  $PROJECT_ID"
info "Region:   $REGION"
info "Bucket:   $BUCKET_NAME"
info "AR Repo:  $AR_REPO_NAME"
info "SA:       $SA_EMAIL"
echo ""

# Step 1: Set active project
info "Setting active project..."
run_cmd gcloud config set project "$PROJECT_ID"

# Step 2: Enable required APIs
echo ""
info "=== Step 1: Enable GCP APIs ==="
for api in "${REQUIRED_APIS[@]}"; do
    info "Enabling $api..."
    run_cmd gcloud services enable "$api" --project="$PROJECT_ID"
done
ok "All required APIs enabled."

# Step 3: Create Artifact Registry repository
echo ""
info "=== Step 2: Artifact Registry ==="
if gcloud artifacts repositories describe "$AR_REPO_NAME" \
    --location="$REGION" --project="$PROJECT_ID" &>/dev/null; then
    ok "Artifact Registry repo '$AR_REPO_NAME' already exists."
else
    info "Creating Artifact Registry Docker repo..."
    run_cmd gcloud artifacts repositories create "$AR_REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --project="$PROJECT_ID" \
        --description="Civ6 AlphaZero training container images"
    ok "Artifact Registry repo created."
fi

# Step 4: Create GCS bucket
echo ""
info "=== Step 3: GCS Bucket ==="
if gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="$PROJECT_ID" &>/dev/null; then
    ok "GCS bucket '$BUCKET_NAME' already exists."
else
    info "Creating GCS bucket..."
    run_cmd gcloud storage buckets create "gs://${BUCKET_NAME}" \
        --project="$PROJECT_ID" \
        --location="$REGION" \
        --default-storage-class=STANDARD \
        --uniform-bucket-level-access
    ok "GCS bucket created."
fi

# Set lifecycle (delete objects older than 90 days in tmp/ prefix)
info "Setting lifecycle policy (90-day cleanup for tmp/ objects)..."
LIFECYCLE_JSON=$(mktemp)
cat > "$LIFECYCLE_JSON" << 'EOF'
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 90,
          "matchesPrefix": ["tmp/"]
        }
      }
    ]
  }
}
EOF
run_cmd gcloud storage buckets update "gs://${BUCKET_NAME}" \
    --lifecycle-file="$LIFECYCLE_JSON" --project="$PROJECT_ID" 2>/dev/null || true
rm -f "$LIFECYCLE_JSON"

# Step 5: Create Service Account
echo ""
info "=== Step 4: IAM Service Account ==="
if gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
    ok "Service account '$SA_NAME' already exists."
else
    info "Creating service account..."
    run_cmd gcloud iam service-accounts create "$SA_NAME" \
        --project="$PROJECT_ID" \
        --display-name="Civ6 AlphaZero Vertex AI Service Account" \
        --description="Least-privilege SA for Vertex AI training jobs"
    ok "Service account created."
fi

# Bind IAM roles
info "Binding IAM roles..."
for role in "${SA_ROLES[@]}"; do
    run_cmd gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$role" \
        --quiet --condition=None 2>/dev/null || true
done
ok "IAM roles bound."

# Step 6: Secret Manager
echo ""
info "=== Step 5: Secret Manager ==="

# Map secret names to env var names
declare -A SECRET_ENV_MAP=(
    ["wandb-api-key"]="WANDB_API_KEY"
    ["openai-api-key"]="OPENAI_API_KEY"
    ["anthropic-api-key"]="ANTHROPIC_API_KEY"
)

for secret in "${SECRET_KEYS[@]}"; do
    env_var="${SECRET_ENV_MAP[$secret]}"
    if gcloud secrets describe "$secret" --project="$PROJECT_ID" &>/dev/null; then
        ok "Secret '$secret' already exists."
    else
        # Try to read value from environment
        value="${!env_var:-}"
        if [[ -z "$value" ]]; then
            warn "No value found for $env_var. Skipping secret '$secret'."
            warn "  Set $env_var in your environment or .env file and re-run."
            continue
        fi
        info "Creating secret '$secret'..."
        run_cmd gcloud secrets create "$secret" \
            --project="$PROJECT_ID" \
            --replication-policy="automatic"
        echo -n "$value" | run_cmd gcloud secrets versions add "$secret" \
            --project="$PROJECT_ID" --data-file=-
        ok "Secret '$secret' created with value from \$$env_var."
    fi
done

# Step 7: Configure Docker authentication for Artifact Registry
echo ""
info "=== Step 6: Docker Auth ==="
info "Configuring Docker for Artifact Registry..."
run_cmd gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet 2>/dev/null || true
ok "Docker configured for ${REGION}-docker.pkg.dev"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Provisioning Complete"
echo "============================================================"
echo ""
ok "All resources provisioned successfully!"
echo ""
info "Next steps:"
echo "  1. Build and push container:"
echo "     python scripts/build_and_push.py --tag v1.0.0"
echo ""
echo "  2. Update .env with new image URI:"
echo "     CIV6AZ_CONTAINER_IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/alphazero-training:v1.0.0"
echo ""
echo "  3. Run dry-run test:"
echo "     .\\scripts\\launch_muzero_training.ps1 -DryRun"
echo ""
echo "  4. Run smoke test (~\$1-2):"
echo "     .\\scripts\\launch_muzero_training.ps1 -Iterations 1 -Workers 1 -GamesPerWorker 2"
echo ""
