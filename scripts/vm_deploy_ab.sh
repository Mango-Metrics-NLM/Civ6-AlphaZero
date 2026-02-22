#!/bin/bash
set -e
LOG=/tmp/ab_deploy.log
date > "$LOG"
echo "Starting v0.14.1 A/B deploy" >> "$LOG"

echo "Pulling v0.14.1..." >> "$LOG"
docker pull gcr.io/hrmtrain/civ6az-training:v0.14.1 >> "$LOG" 2>&1
echo "Pull complete" >> "$LOG"

if docker ps -a --format '{{.Names}}' | grep -q civ6az-ab-run-a; then
    echo "Removing existing Run A container..." >> "$LOG"
    docker rm -f civ6az-ab-run-a >> "$LOG" 2>&1
fi

echo "Launching Run A (patch_embed)..." >> "$LOG"
docker run -d \
  --name civ6az-ab-run-a \
  -e CIV6AZ_VERTEX_PROJECT=hrmtrain \
  -e CIV6AZ_VERTEX_REGION=us-central1 \
  -e CIV6AZ_GCS_BUCKET=civ6az-training \
  -e CIV6AZ_CONTAINER_IMAGE_URI=gcr.io/hrmtrain/civ6az-training:v0.14.1 \
  -e WANDB_API_KEY="${WANDB_API_KEY}" \
  -e WANDB_PROJECT=civ6-alphazero \
  -e WANDB_ENABLED=true \
  gcr.io/hrmtrain/civ6az-training:v0.14.1 \
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
    --encoder_type=patch_embed \
    --agent_id=ab_patch_embed \
  >> "$LOG" 2>&1

echo "Run A container started" >> "$LOG"
docker ps --filter name=civ6az-ab-run-a >> "$LOG"
echo "Deploy complete" >> "$LOG"
cat "$LOG"
