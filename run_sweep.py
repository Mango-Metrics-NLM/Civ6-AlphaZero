"""Wrapper script for W&B sweeps that properly handles config parameters."""

import sys
import wandb
import subprocess

def main():
    # Initialize wandb - this gets config from sweep
    wandb.init()

    # Get sweep config
    config = wandb.config

    # Build command with actual values
    cmd = [
        sys.executable,
        "run_muzero_training.py",
        "--quick",
        "--small",
        "--wandb",
        f"--lr={config.lr}",
        f"--batch-size={config.batch_size}",
        f"--simulations={config.simulations}",
        f"--res-blocks={config.res_blocks}",
        f"--filters={config.filters}",
    ]

    print(f"Running: {' '.join(cmd)}")

    # Run the actual training
    result = subprocess.run(cmd, capture_output=False)

    wandb.finish()
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
