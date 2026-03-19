#!/usr/bin/env python3
"""
Run orchestrator locally on Windows machine.

The orchestrator just dispatches jobs to Vertex AI, so it can run anywhere
with GCP credentials. No GPU needed.
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Launch orchestrator locally."""
    project_root = Path(__file__).parent.parent

    print("="*70)
    print("Civ6 AlphaZero - Local Orchestrator Launch")
    print("="*70)
    print()
    print("Configuration:")
    print("  Iterations: 20")
    print("  Total games: 4000 (4 workers × 50 games × 20 iters)")
    print("  Batch self-play: 4 parallel games, batch size 32")
    print("  Starting from: v268")
    print()
    print("Expected:")
    print("  Runtime: 4-6 hours")
    print("  Cost: $3-5")
    print("  Your machine: just dispatching jobs (low resource usage)")
    print()
    print("W&B Dashboard:")
    print("  https://wandb.ai/ianshank-none/civ6-alphazero")
    print()
    print("="*70)
    print()

    # Set environment variables
    import os
    os.environ.update({
        "CIV6AZ_VERTEX_PROJECT": "hrmtrain",
        "CIV6AZ_VERTEX_REGION": "us-central1",
        "CIV6AZ_GCS_BUCKET": "civ6az-training",
        "CIV6AZ_CONTAINER_IMAGE_URI": "us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v1.2.0",
        "WANDB_API_KEY": "26a08535e80a6d5d6f4a941f6a742264b25f4819",
        "WANDB_PROJECT": "civ6-alphazero",
    })

    # Build command (W&B config via env vars, start_model auto-detected)
    cmd = [
        sys.executable,
        "-m", "alpha_zero.cloud.gcp.vertex_orchestrator",
        "--iterations=20",
        "--workers=4",
        "--games_per_worker=50",
        "--num_parallel_games=4",
        "--max_batch_size=32",
        "--mcts_type=gumbel",
        "--mcts_simulations=16",
        "--max_turns=30",
        "--evaluation_games=30",
        "--evaluation_win_threshold=0.55"
    ]

    print("Launching orchestrator...")
    print()

    # Run orchestrator (output to console)
    try:
        subprocess.run(cmd, cwd=str(project_root))
    except KeyboardInterrupt:
        print("\n\nOrchestrator interrupted by user")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
