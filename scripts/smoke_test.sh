#!/usr/bin/env bash
# End-to-end smoke test suite for Civ6 AlphaZero.
#
# Usage:
#   ./scripts/smoke_test.sh              # Local tests only (default)
#   ./scripts/smoke_test.sh --cloud      # Local + cloud validation (no Vertex AI cost)
#   ./scripts/smoke_test.sh --all        # Local + cloud + live Vertex AI training
#
# Environment variables (for --cloud and --all):
#   CIV6AZ_VERTEX_PROJECT   GCP project ID
#   CIV6AZ_GCS_BUCKET       GCS bucket name
#   CIV6AZ_CONTAINER_IMAGE_URI  Docker image URI
#
# Exit codes:
#   0 — all tests passed
#   1 — one or more test tiers failed

set -euo pipefail

MODE="${1:---local}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONTAINER_IMAGE="${CIV6AZ_CONTAINER_IMAGE_URI:-gcr.io/hrmtrain/civ6az-training:latest}"

echo "=== Civ6 AlphaZero Smoke Tests (${MODE}) ==="
echo "Project dir: ${PROJECT_DIR}"
echo ""

cd "${PROJECT_DIR}"

# ---------------------------------------------------------------------------
# Tier 1: Local tests (always run)
# ---------------------------------------------------------------------------

echo "[1/6] Sanity & Security..."
pytest tests/sanity/ tests/security/ -v --tb=short -q

echo ""
echo "[2/6] Unit tests..."
pytest tests/unit/ --tb=short -m "not slow and not benchmark" -q

echo ""
echo "[3/6] Integration tests..."
pytest tests/integration/ --tb=short -m "not slow and not requires_env" -q

echo ""
echo "[4/6] E2E local training smoke..."
pytest tests/e2e/test_local_training_smoke.py -v --tb=short

echo ""
echo "[5/6] Cloud unit tests (mocked)..."
pytest tests/cloud/unit/ -v --tb=short

echo ""
echo "[6/6] Vertex submission validation (dry-run)..."
pytest tests/e2e/test_vertex_submission.py -v --tb=short

if [[ "${MODE}" == "--local" ]]; then
    echo ""
    echo "=== Local smoke tests PASSED ==="
    exit 0
fi

# ---------------------------------------------------------------------------
# Tier 2: Cloud validation (--cloud and --all)
# ---------------------------------------------------------------------------

echo ""
echo "--- Cloud Validation ---"
echo ""

echo "[Cloud 1/3] Live GCS tests..."
pytest tests/e2e/test_gcs_live.py -m "requires_env" -v --tb=short

echo ""
echo "[Cloud 2/3] Container health check..."
docker run --rm "${CONTAINER_IMAGE}" \
    python -m alpha_zero.cloud.gcp.container_healthcheck

echo ""
echo "[Cloud 3/3] Vertex AI dry-run..."
python -m alpha_zero.cloud.gcp.vertex_orchestrator \
    --iterations=1 --workers=1 --games_per_worker=2 \
    --mcts_type=gumbel --mcts_simulations=4 --board_size=8 --dry_run

if [[ "${MODE}" == "--cloud" ]]; then
    echo ""
    echo "=== Cloud smoke tests PASSED ==="
    exit 0
fi

# ---------------------------------------------------------------------------
# Tier 3: Live Vertex AI training (--all only)
# ---------------------------------------------------------------------------

echo ""
echo "--- Live Vertex AI Training ---"
echo ""

echo "[Live 1/2] Single iteration training..."
python -m alpha_zero.cloud.gcp.vertex_orchestrator \
    --iterations=1 --workers=1 --games_per_worker=5 \
    --mcts_type=gumbel --mcts_simulations=8 --board_size=8 \
    --epochs=2 --max_turns=15

echo ""
echo "[Live 2/2] Two iterations with evaluation..."
python -m alpha_zero.cloud.gcp.vertex_orchestrator \
    --iterations=2 --workers=1 --games_per_worker=5 \
    --mcts_type=gumbel --mcts_simulations=8 --board_size=8 \
    --epochs=2 --evaluation_games=4 --max_turns=15

echo ""
echo "=== All smoke tests PASSED ==="
