#!/usr/bin/env bash
# MuZero Cloud Smoke Test Suite
#
# Validates MuZero integration in GCP without affecting the running
# AlphaZero orchestrator. All GCS paths are scoped via --agent_id.
#
# Usage:
#   ./scripts/muzero_smoke_test.sh              # Dry-run only (default, no cost)
#   ./scripts/muzero_smoke_test.sh --cloud      # Live cloud run (minimal cost)
#
# Prerequisites:
#   - gcloud auth configured
#   - CIV6AZ_VERTEX_PROJECT and CIV6AZ_GCS_BUCKET set (for --cloud)
#   - Docker image built (for --cloud)
#
# Isolation:
#   All GCS artifacts scoped via --agent_id=muzero-smoke:
#     models/civ6az/muzero-smoke/
#     trajectories/civ6az/muzero-smoke/
#     checkpoints/civ6az/muzero-smoke/
#   Production AlphaZero paths (no agent_id) are untouched:
#     models/civ6az/
#     trajectories/civ6az/
#
# Exit codes:
#   0 — all tests passed
#   1 — one or more tests failed

set -euo pipefail

MODE="${1:---dry-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AGENT_ID="muzero-smoke"

echo "=== MuZero Cloud Smoke Tests (${MODE}) ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Agent ID:    ${AGENT_ID} (isolates from production)"
echo ""

cd "${PROJECT_DIR}"

# ---------------------------------------------------------------------------
# Step 1: Dry-run validation (local, no GCP cost)
# ---------------------------------------------------------------------------

echo "[1/4] MuZero self-play dry-run..."
python -m alpha_zero.cloud.gcp.selfplay_worker \
    --worker_id=0 \
    --model_version=0 \
    --iteration=1 \
    --games=2 \
    --mcts_simulations=8 \
    --board_size=6 \
    --max_turns=30 \
    --num_res_blocks=2 \
    --num_filters=32 \
    --mcts_type=muzero \
    --agent_id="${AGENT_ID}" \
    --dry_run

echo ""
echo "[2/4] MuZero orchestrator dry-run..."
python -m alpha_zero.cloud.gcp.vertex_orchestrator \
    --algorithm=muzero \
    --iterations=1 \
    --workers=1 \
    --games_per_worker=2 \
    --mcts_simulations=8 \
    --num_res_blocks=2 \
    --num_filters=32 \
    --board_size=6 \
    --dry_run

echo ""
echo "[3/4] Verify GCS path isolation + [4/4] MuZero network creation..."
python scripts/muzero_dryrun_test.py

if [[ "${MODE}" == "--dry-run" ]]; then
    echo ""
    echo "=== MuZero dry-run smoke tests PASSED ==="
    echo "Run with --cloud to submit real Vertex AI jobs."
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 2: Live cloud run (--cloud mode, minimal cost)
# ---------------------------------------------------------------------------

echo ""
echo "--- Live Cloud Run (agent_id=${AGENT_ID}) ---"
echo "WARNING: This will launch Vertex AI jobs. Estimated cost: <\$1."
echo ""

echo "[Cloud 1/2] MuZero self-play (1 worker, 5 games)..."
python -m alpha_zero.cloud.gcp.selfplay_worker \
    --worker_id=0 \
    --model_version=0 \
    --iteration=1 \
    --games=5 \
    --mcts_simulations=8 \
    --board_size=6 \
    --max_turns=50 \
    --num_res_blocks=2 \
    --num_filters=32 \
    --mcts_type=muzero \
    --agent_id="${AGENT_ID}"

echo ""
echo "[Cloud 2/2] MuZero orchestrator (1 iteration)..."
python -m alpha_zero.cloud.gcp.vertex_orchestrator \
    --algorithm=muzero \
    --iterations=1 \
    --workers=1 \
    --games_per_worker=5 \
    --mcts_simulations=8 \
    --num_res_blocks=2 \
    --num_filters=32 \
    --board_size=6 \
    --epochs=2 \
    --max_turns=50

echo ""
echo "=== MuZero cloud smoke tests PASSED ==="
echo ""
echo "Cleanup:"
echo "  gsutil -m rm -r gs://\${CIV6AZ_GCS_BUCKET}/models/civ6az/${AGENT_ID}/"
echo "  gsutil -m rm -r gs://\${CIV6AZ_GCS_BUCKET}/trajectories/civ6az/${AGENT_ID}/"
echo "  gsutil -m rm -r gs://\${CIV6AZ_GCS_BUCKET}/checkpoints/civ6az/${AGENT_ID}/"
