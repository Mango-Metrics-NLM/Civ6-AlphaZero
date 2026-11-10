#!/usr/bin/env bash
# =============================================================================
# Convenience wrapper for launching Civ6 AlphaZero cloud training on Vertex AI.
#
# Reads defaults from configs/cloud_training_v1.example.yaml comments, but all values
# are passed as CLI args to the orchestrator (YAML is documentation only).
#
# Usage:
#   ./scripts/run_cloud_training.sh                   # Real run
#   ./scripts/run_cloud_training.sh --dry_run          # Validate only
#   ITERATIONS=50 ./scripts/run_cloud_training.sh      # Override iterations
# =============================================================================
set -euo pipefail

# --- Configurable defaults (override via environment) ---
ITERATIONS="${ITERATIONS:-20}"
WORKERS="${WORKERS:-4}"
GAMES_PER_WORKER="${GAMES_PER_WORKER:-25}"
MAX_TURNS="${MAX_TURNS:-30}"
MCTS_TYPE="${MCTS_TYPE:-gumbel}"
MCTS_SIMULATIONS="${MCTS_SIMULATIONS:-16}"
GUMBEL_NUM_ACTIONS="${GUMBEL_NUM_ACTIONS:-8}"
BOARD_SIZE="${BOARD_SIZE:-8}"
ACTION_SPACE_SIZE="${ACTION_SPACE_SIZE:-376}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-6}"
NUM_FILTERS="${NUM_FILTERS:-64}"
EPOCHS="${EPOCHS:-10}"
EVAL_GAMES="${EVAL_GAMES:-10}"
ENABLE_WANDB="${ENABLE_WANDB:-true}"

# --- Build command ---
CMD=(
    python -m alpha_zero.cloud.gcp.vertex_orchestrator
    --iterations="${ITERATIONS}"
    --workers="${WORKERS}"
    --games_per_worker="${GAMES_PER_WORKER}"
    --max_turns="${MAX_TURNS}"
    --mcts_type="${MCTS_TYPE}"
    --mcts_simulations="${MCTS_SIMULATIONS}"
    --gumbel_num_actions="${GUMBEL_NUM_ACTIONS}"
    --board_size="${BOARD_SIZE}"
    --action_space_size="${ACTION_SPACE_SIZE}"
    --num_res_blocks="${NUM_RES_BLOCKS}"
    --num_filters="${NUM_FILTERS}"
    --epochs="${EPOCHS}"
    --evaluation_games="${EVAL_GAMES}"
)

if [[ "${ENABLE_WANDB}" == "true" ]]; then
    CMD+=(--wandb)
fi

# Pass through any extra flags (e.g. --dry_run, --curriculum)
CMD+=("$@")

echo "=== Civ6 AlphaZero Cloud Training ==="
echo "Iterations:   ${ITERATIONS}"
echo "Workers:      ${WORKERS}"
echo "Games/worker: ${GAMES_PER_WORKER}"
echo "MCTS:         ${MCTS_TYPE} (${MCTS_SIMULATIONS} sims)"
echo "Board:        ${BOARD_SIZE}x${BOARD_SIZE}"
echo "W&B:          ${ENABLE_WANDB}"
echo ""
echo "Command: ${CMD[*]}"
echo ""

exec "${CMD[@]}"
