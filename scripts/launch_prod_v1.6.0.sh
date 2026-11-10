#!/usr/bin/env bash
#
# v1.6.0 production orchestrator launcher.
#
# Runs the Vertex orchestrator directly (NOT via entrypoint --mode=orchestrator,
# which is intentionally blocked inside containers). Routes experiment tracking
# to MLflow with a SQLite metadata DB + GCS-backed artifact root. No secrets
# baked in - everything from env or the VM service account.
#
# Pre-flight write probe verifies the SA has storage.objectAdmin on the bucket
# BEFORE incurring Vertex AI job spend.
#
# Usage (on civ6az-orchestrator VM):
#   bash launch_prod_v1.6.0.sh
#
# Override via env:
#   ITERATIONS=10 WORKERS=2 bash launch_prod_v1.6.0.sh
#
set -euo pipefail

# Resolve python binary — prefer 'python3', fall back to 'python'.
PYTHON="$(command -v python3 || command -v python || true)"
if [[ -z "${PYTHON}" ]]; then
    echo "FATAL: no python3 or python found in PATH" >&2
    exit 1
fi
echo "Using Python: ${PYTHON} ($(${PYTHON} --version 2>&1))"

# Debian 12+ uses externally-managed Python — all pip installs need a venv.
# Use a user-owned location to avoid permission issues from any prior sudo runs.
VENV="${HOME}/.venv/civ6az"
if [[ -d "/opt/civ6az-venv" && ! -w "/opt/civ6az-venv" ]]; then
    echo "Removing root-owned /opt/civ6az-venv (stale from prior sudo run)..."
    sudo rm -rf /opt/civ6az-venv
fi
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "Creating venv at ${VENV}..."
    mkdir -p "$(dirname "${VENV}")"
    "${PYTHON}" -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
PYTHON="${VENV}/bin/python"
echo "Venv active: ${PYTHON} ($(${PYTHON} --version 2>&1))"
# Pre-install build tools so alpha_zero's setup.py/pyproject.toml can build.
"${PYTHON}" -m pip install --quiet --upgrade pip wheel setuptools

VERSION="${VERSION:-v1.6.0}"
PROJECT="${CIV6AZ_VERTEX_PROJECT:?CIV6AZ_VERTEX_PROJECT must be set (no default project)}"
REGION="${CIV6AZ_VERTEX_REGION:-us-central1}"
BUCKET="${CIV6AZ_GCS_BUCKET:?CIV6AZ_GCS_BUCKET must be set (no default bucket)}"
IMAGE="${CIV6AZ_CONTAINER_IMAGE_URI:-us-central1-docker.pkg.dev/${PROJECT}/civ6az/alphazero-training:${VERSION}}"

ITERATIONS="${ITERATIONS:-50}"
WORKERS="${WORKERS:-4}"
GAMES_PER_WORKER="${GAMES_PER_WORKER:-50}"
EVAL_GAMES="${EVAL_GAMES:-30}"
EVAL_WIN_THRESHOLD="${EVAL_WIN_THRESHOLD:-0.55}"
MCTS_SIMS="${MCTS_SIMS:-16}"
MAX_TURNS="${MAX_TURNS:-30}"
NUM_PARALLEL_GAMES="${NUM_PARALLEL_GAMES:-8}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
TRAJECTORY_WINDOW="${TRAJECTORY_WINDOW:-5}"

RUN_TAG="${RUN_TAG:-prod-$(date +%Y%m%d-%H%M%S)}"

# MLflow: SQLite metadata DB on the VM, artifacts to GCS. The VM service account
# must have storage.objectAdmin on the bucket. No keys baked in - the SA is
# attached at VM creation time.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:////tmp/mlflow.db}"
export MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-gs://${BUCKET}/mlflow}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-${VERSION}-prod}"
export MLFLOW_RUN_NAME="${MLFLOW_RUN_NAME:-${RUN_TAG}}"
export MLFLOW_TAGS="${MLFLOW_TAGS:-version=${VERSION},iterations=${ITERATIONS},batch=${NUM_PARALLEL_GAMES}}"
export MLFLOW_RUN_GROUP="${MLFLOW_RUN_GROUP:-${VERSION}}"
export MLFLOW_ENABLED=true

# Bootstrap alpha_zero package if not already installed.
# The project is cloned at $HOME/Civ6-AlphaZero* on this VM.
# We copy source to /tmp first to avoid permission issues with egg-info.
# Always reinstall/update alpha_zero to guarantee the orchestrator is running
# the latest code that was synced from the developer's workspace.
echo "Updating alpha_zero package to the latest version..."
REPO_DIRS=(
    "${HOME}/Civ6-AlphaZero"
    "${HOME}/Civ6-AlphaZero-real"
)
INSTALLED=0
for REPO in "${REPO_DIRS[@]}"; do
    if [[ -f "${REPO}/setup.py" || -f "${REPO}/pyproject.toml" ]]; then
        echo "Copying source from ${REPO} to /tmp/alpha_zero_src ..."
        rm -rf /tmp/alpha_zero_src
        cp -r "${REPO}" /tmp/alpha_zero_src
        # Remove any stale egg-info from the copy so pip builds fresh.
        find /tmp/alpha_zero_src -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
        echo "Installing from /tmp/alpha_zero_src (with cloud extras)..."
        # Use --upgrade to force overwrite any existing site-packages install
        "${PYTHON}" -m pip install --upgrade --quiet "/tmp/alpha_zero_src[cloud]" 2>&1 \
            || "${PYTHON}" -m pip install --upgrade --quiet /tmp/alpha_zero_src 2>&1
        # Ensure cloud deps are present even if [cloud] extra isn't declared.
        "${PYTHON}" -m pip install --quiet \
            google-cloud-aiplatform google-cloud-storage mlflow 2>&1 | tail -3
        INSTALLED=1
        break
    fi
done
if [[ "${INSTALLED}" -eq 0 ]]; then
    BUNDLE="gs://${BUCKET}/src/alpha-zero-latest.tar.gz"
    TMP_BUNDLE="/tmp/alpha_zero_src.tar.gz"
    if gsutil -q stat "${BUNDLE}" 2>/dev/null; then
        echo "Fetching source bundle from ${BUNDLE}"
        gsutil cp "${BUNDLE}" "${TMP_BUNDLE}"
        "${PYTHON}" -m pip install --upgrade --quiet "${TMP_BUNDLE}"
    else
        echo "FATAL: Cannot find alpha_zero source (checked VM clones + GCS)." >&2
        exit 1
    fi
fi
"${PYTHON}" -c 'import alpha_zero; print("alpha_zero import OK:", alpha_zero.__file__)'

# Pre-flight: verify GCS write access before launching anything expensive.
PROBE_BLOB="${MLFLOW_ARTIFACT_ROOT%/}/.write-probe-$(date +%s)"
echo "Pre-flight: writing probe blob to ${PROBE_BLOB}"
if ! echo "v1.6.0-prod-probe" | gsutil cp - "${PROBE_BLOB}"; then
    echo "FATAL: VM service account cannot write to ${MLFLOW_ARTIFACT_ROOT}." >&2
    echo "       Verify storage.objectAdmin on bucket gs://${BUCKET}." >&2
    exit 1
fi
gsutil rm "${PROBE_BLOB}" >/dev/null 2>&1 || true
echo "Pre-flight OK."

echo "Launching orchestrator: ${ITERATIONS} iters, ${WORKERS}x${GAMES_PER_WORKER} games, image=${IMAGE}"

# vertex_orchestrator reads project/bucket/image via GCPConfig from CIV6AZ_* env vars.
export CIV6AZ_VERTEX_PROJECT="${PROJECT}"
export CIV6AZ_VERTEX_REGION="${REGION}"
export CIV6AZ_GCS_BUCKET="${BUCKET}"
export CIV6AZ_CONTAINER_IMAGE_URI="${IMAGE}"

exec "${PYTHON}" -u -m alpha_zero.cloud.gcp.vertex_orchestrator \
  --iterations="${ITERATIONS}" \
  --workers="${WORKERS}" \
  --games_per_worker="${GAMES_PER_WORKER}" \
  --evaluation_games="${EVAL_GAMES}" \
  --evaluation_win_threshold="${EVAL_WIN_THRESHOLD}" \
  --mcts_type=gumbel \
  --mcts_simulations="${MCTS_SIMS}" \
  --max_turns="${MAX_TURNS}" \
  --num_parallel_games="${NUM_PARALLEL_GAMES}" \
  --max_batch_size="${MAX_BATCH_SIZE}" \
  --trajectory_window="${TRAJECTORY_WINDOW}"
