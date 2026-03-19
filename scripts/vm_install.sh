#!/bin/bash
# Install dependencies and launch orchestrator on VM
# This script is uploaded to the VM and executed there

set -euo pipefail

echo "=== Civ6AZ VM Setup ==="
echo ""

# 1. Create venv
echo "[1/4] Creating virtual environment..."
cd ~/Civ6-AlphaZero
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo "[2/4] Installing dependencies..."
echo "  Disk space before install:"
df -h / /home /tmp | tail -n +2
python -m pip install --no-cache-dir -q --upgrade pip
echo "  Installing requirements (this may take 3-5 minutes)..."
python -m pip install --no-cache-dir -q -r requirements-container.txt || {
    echo "  ERROR: Pip install failed, checking disk space:"
    df -h
    exit 1
}

# 3. Create environment config
echo "[3/4] Creating environment configuration..."
cat > ~/.bashrc_civ6az << 'EOF'
export CIV6AZ_VERTEX_PROJECT=hrmtrain
export CIV6AZ_VERTEX_REGION=us-central1
export CIV6AZ_GCS_BUCKET=civ6az-training
export CIV6AZ_CONTAINER_IMAGE_URI=us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v1.2.0
export WANDB_API_KEY=26a08535e80a6d5d6f4a941f6a742264b25f4819
export WANDB_PROJECT=civ6-alphazero
export PYTHONPATH=~/Civ6-AlphaZero:$PYTHONPATH
EOF

# 4. Launch orchestrator in background
echo "[4/4] Launching orchestrator..."
source ~/.bashrc_civ6az

nohup python -m alpha_zero.cloud.gcp.vertex_orchestrator \
  --iterations=20 \
  --workers=4 \
  --games_per_worker=50 \
  --num_parallel_games=4 \
  --max_batch_size=32 \
  --mcts_type=gumbel \
  --mcts_simulations=16 \
  --max_turns=30 \
  --evaluation_games=30 \
  --evaluation_win_threshold=0.55 \
  --start_model_version=268 \
  --wandb_project=civ6-alphazero \
  --wandb_entity=ianshank-none \
  --wandb_tags=batch-selfplay,v1.5.0 \
  > ~/orchestrator.log 2>&1 &

echo $! > ~/orchestrator.pid

echo ""
echo "=== Setup Complete ==="
echo "Orchestrator PID: $(cat ~/orchestrator.pid)"
echo "Logs: ~/orchestrator.log"
echo ""
