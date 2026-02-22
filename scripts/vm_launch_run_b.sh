#!/bin/bash
# Wait for Run A to complete, then launch Run B (hex_crystalline)
LOG=/tmp/run_b_launch.log
echo "$(date) - Waiting for Run A (civ6az-ab-run-a) to complete..." > "$LOG"

# Poll every 5 minutes until Run A exits
while docker ps --format '{{.Names}}' | grep -q civ6az-ab-run-a; do
    echo "$(date) - Run A still running..." >> "$LOG"
    sleep 300
done

echo "$(date) - Run A completed. Exit code: $(docker inspect civ6az-ab-run-a --format '{{.State.ExitCode}}')" >> "$LOG"

# Remove any existing Run B container
docker rm -f civ6az-ab-run-b 2>/dev/null

echo "$(date) - Launching Run B (hex_crystalline)..." >> "$LOG"
docker run -d \
  --name civ6az-ab-run-b \
  -e CIV6AZ_VERTEX_PROJECT=hrmtrain \
  -e CIV6AZ_VERTEX_REGION=us-central1 \
  -e CIV6AZ_GCS_BUCKET=civ6az-training \
  -e CIV6AZ_CONTAINER_IMAGE_URI=gcr.io/hrmtrain/civ6az-training:v0.14.2 \
  -e WANDB_API_KEY="${WANDB_API_KEY}" \
  -e WANDB_PROJECT=civ6-alphazero \
  -e WANDB_ENABLED=true \
  gcr.io/hrmtrain/civ6az-training:v0.14.2 \
  python -u -m alpha_zero.cloud.gcp.entrypoint \
    --mode=orchestrator \
    --iterations=20 \
    --workers=2 \
    --games_per_worker=50 \
    --mcts_type=gumbel \
    --mcts_simulations=8 \
    --board_size=8 \
    --max_turns=15 \
    --epochs=10 \
    --lr_schedule_type=warmup_cosine \
    --lr_initial=0.001 \
    --lr_final=0.0001 \
    --lr_warmup_steps=100 \
    --value_lr_multiplier=0.1 \
    --evaluation_games=25 \
    --evaluation_win_threshold=0.55 \
    --trajectory_window=5 \
    --post_run_tournament \
    --tournament_games=6 \
    --tournament_mcts_sims=25 \
    --encoder_type=hex_crystalline \
    --agent_id=ab_hex_crystalline

echo "$(date) - Run B container started" >> "$LOG"
docker ps --filter name=civ6az-ab-run-b >> "$LOG"
echo "$(date) - Launch script complete" >> "$LOG"
