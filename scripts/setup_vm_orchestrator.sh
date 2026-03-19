#!/bin/bash
# Setup orchestrator VM with codebase and dependencies
# Run this script ON the VM, not locally

set -euo pipefail

echo "========================================="
echo "Setting up Civ6AZ Orchestrator VM"
echo "========================================="
echo ""

# 1. Install system dependencies
echo "[1/6] Installing system dependencies..."
sudo yum install -y git python3-pip python3-devel gcc gcc-c++ || \
sudo apt-get update && sudo apt-get install -y git python3-pip python3-dev build-essential

# 2. Install gcloud SDK if not present (for GCS access)
echo "[2/6] Checking gcloud SDK..."
if ! command -v gcloud &> /dev/null; then
    echo "Installing gcloud SDK..."
    curl https://sdk.cloud.google.com | bash
    exec -l $SHELL
else
    echo "  gcloud SDK already installed"
fi

# 3. Clone repository from GCS (faster) or Git
echo "[3/6] Setting up codebase..."
cd ~
if [ -d "Civ6-AlphaZero" ]; then
    echo "  Repository already exists, using existing"
else
    # Try to get from GCS bucket first (faster if available)
    if gsutil -q stat gs://civ6az-training/codebase/Civ6-AlphaZero.tar.gz; then
        echo "  Downloading from GCS..."
        gsutil cp gs://civ6az-training/codebase/Civ6-AlphaZero.tar.gz ~/
        tar -xzf Civ6-AlphaZero.tar.gz
        rm Civ6-AlphaZero.tar.gz
    else
        echo "  Cloning from Git (first time may be slow)..."
        git clone https://github.com/anthropics/civ6-alphazero.git Civ6-AlphaZero 2>/dev/null || \
        echo "  Note: Using existing files or manual setup may be needed"
    fi
fi

cd ~/Civ6-AlphaZero

# 3. Create virtual environment
echo "[3/6] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
echo "[4/6] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements-container.txt

# 5. Set environment variables
echo "[5/6] Setting environment variables..."
cat > ~/.bashrc_civ6az << 'EOF'
# Civ6AZ Orchestrator Environment
export CIV6AZ_VERTEX_PROJECT=hrmtrain
export CIV6AZ_VERTEX_REGION=us-central1
export CIV6AZ_GCS_BUCKET=civ6az-training
export CIV6AZ_CONTAINER_IMAGE_URI=us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v1.2.0
export WANDB_API_KEY=26a08535e80a6d5d6f4a941f6a742264b25f4819
export WANDB_PROJECT=civ6-alphazero
export PYTHONPATH=~/Civ6-AlphaZero:$PYTHONPATH
EOF

# Add to .bashrc if not already there
grep -q "bashrc_civ6az" ~/.bashrc || echo "source ~/.bashrc_civ6az" >> ~/.bashrc

# 6. Verify installation
echo "[6/6] Verifying installation..."
source ~/.bashrc_civ6az
python3 -c "import alpha_zero; print(f'Alpha Zero version: OK')" && \
echo "✓ Installation complete!" || \
echo "✗ Installation verification failed"

# 7. Install systemd service for auto-restart on reboot
echo "[7/7] Setting up systemd service..."
SERVICE_SRC="$HOME/Civ6-AlphaZero/scripts/systemd/civ6az-orchestrator.service"
ENV_TEMPLATE="$HOME/Civ6-AlphaZero/scripts/systemd/.env.civ6az.template"

if [ -f "$SERVICE_SRC" ]; then
    # Install service file
    sudo cp "$SERVICE_SRC" /etc/systemd/system/civ6az-orchestrator.service
    sudo systemctl daemon-reload
    sudo systemctl enable civ6az-orchestrator.service

    # Install environment file if not already present
    if [ ! -f "$HOME/.env.civ6az" ]; then
        if [ -f "$ENV_TEMPLATE" ]; then
            cp "$ENV_TEMPLATE" "$HOME/.env.civ6az"
            # Update WANDB_API_KEY from existing bashrc if available
            if [ -f "$HOME/.bashrc_civ6az" ]; then
                WANDB_KEY=$(grep WANDB_API_KEY "$HOME/.bashrc_civ6az" | head -1 | cut -d= -f2)
                if [ -n "$WANDB_KEY" ]; then
                    sed -i "s/<your-wandb-api-key>/$WANDB_KEY/" "$HOME/.env.civ6az"
                fi
            fi
            echo "  Environment file created at ~/.env.civ6az"
            echo "  [!] Please verify WANDB_API_KEY in ~/.env.civ6az"
        fi
    else
        echo "  Environment file already exists at ~/.env.civ6az"
    fi

    echo "  ✓ systemd service installed and enabled"
    echo ""
    echo "  Manage the orchestrator:"
    echo "    sudo systemctl start civ6az-orchestrator    # Start"
    echo "    sudo systemctl stop civ6az-orchestrator     # Stop"
    echo "    sudo systemctl status civ6az-orchestrator   # Status"
    echo "    journalctl -u civ6az-orchestrator -f        # Live logs"
else
    echo "  [WARN] Service file not found at $SERVICE_SRC"
    echo "  Skipping systemd setup"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Option A: Run orchestrator via systemd (recommended):"
echo "  sudo systemctl start civ6az-orchestrator"
echo "  journalctl -u civ6az-orchestrator -f"
echo ""
echo "Option B: Run orchestrator manually:"
echo "  cd ~/Civ6-AlphaZero"
echo "  source venv/bin/activate"
echo "  source ~/.bashrc_civ6az"
echo "  python -m alpha_zero.cloud.gcp.vertex_orchestrator --resume --iterations=20 ..."
