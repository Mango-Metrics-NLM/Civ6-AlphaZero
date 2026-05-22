#!/usr/bin/env bash
#
# v1.5.0 production orchestrator launcher.
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
#   bash launch_prod_v1.5.0.sh
#
# Override via env:
#   ITERATIONS=10 WORKERS=2 bash launch_prod_v1.5.0.sh
#
set -euo pipefail

VERSION="${VERSION:-v1.5.0}"
PROJECT="${CIV6AZ_VERTEX_PROJECT:-hrmtrain}"
REGION="${CIV6AZ_VERTEX_REGION:-us-central1}"
BUCKET="${CIV6AZ_GCS_BUCKET:-civ6az-training}"
IMAGE="${CIV6AZ_CONTAINER_IMAGE_URI:-us-central1-docker.pkg.dev/${PROJECT}/civ6az/civ6az-training:${VERSION}}"

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

# Pre-flight: verify GCS write access before launching anything expensive.
PROBE_BLOB="${MLFLOW_ARTIFACT_ROOT%/}/.write-probe-$(date +%s)"
echo "Pre-flight: writing probe blob to ${PROBE_BLOB}"
if ! echo "v1.5.0-prod-probe" | gsutil cp - "${PROBE_BLOB}"; then
    echo "FATAL: VM service account cannot write to ${MLFLOW_ARTIFACT_ROOT}." >&2
    echo "       Verify storage.objectAdmin on bucket gs://${BUCKET}." >&2
    exit 1
fi
gsutil rm "${PROBE_BLOB}" >/dev/null 2>&1 || true
echo "Pre-flight OK."

echo "Launching orchestrator: ${ITERATIONS} iters, ${WORKERS}x${GAMES_PER_WORKER} games, image=${IMAGE}"

exec python -u -m alpha_zero.cloud.gcp.vertex_orchestrator \
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
  --trajectory_window="${TRAJECTORY_WINDOW}" \
  --container_image_uri="${IMAGE}" \
  --gcs_bucket="${BUCKET}" \
  --project="${PROJECT}" \
  --region="${REGION}"
