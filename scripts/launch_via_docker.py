#!/usr/bin/env python3
"""Launch orchestrator via Docker on GCP VM using Python subprocess."""
import subprocess
import sys

def main():
    """Launch Docker container on VM."""

    # Docker run command for the VM (single line to avoid SSH escaping issues)
    docker_cmd = "docker run -d --name civ6az-prod-v150 --restart unless-stopped -e CIV6AZ_VERTEX_PROJECT=hrmtrain -e CIV6AZ_VERTEX_REGION=us-central1 -e CIV6AZ_GCS_BUCKET=civ6az-training -e CIV6AZ_CONTAINER_IMAGE_URI=us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v1.2.0 -e WANDB_ENABLED=true -e WANDB_API_KEY=26a08535e80a6d5d6f4a941f6a742264b25f4819 -e WANDB_PROJECT=civ6-alphazero -e WANDB_MODE=online -e PYTHONUNBUFFERED=1 us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v1.2.0 python -u -m alpha_zero.cloud.gcp.entrypoint --mode=orchestrator --iterations=20 --workers=4 --games_per_worker=50 --num_parallel_games=4 --max_batch_size=32 --mcts_type=gumbel --mcts_simulations=16 --max_turns=30 --epochs=10 --trajectory_window=5 --evaluation_games=30 --evaluation_win_threshold=0.55 --learning_rate=0.001 --lr_schedule_type=warmup_cosine --lr_initial=0.001 --lr_final=0.0001 --lr_warmup_steps=500 --weight_decay=0.01 --value_lr_multiplier=0.1 --max_consecutive_failures=3 --start_model_version=268 --post_run_tournament --tournament_games=10 --tournament_mcts_sims=25 --wandb_tags=batch-selfplay,v1.5.0"

    print("=" * 60)
    print("Launching v1.5.0 Batch Self-Play Benchmark")
    print("=" * 60)
    print()
    print("Configuration:")
    print("  Container: civ6az-prod-v150")
    print("  Iterations: 20")
    print("  Total games: 4000 (4 workers × 50 games × 20 iters)")
    print("  Batch self-play: 4 parallel games, batch size 32")
    print("  Starting from: v268")
    print()
    print("Expected:")
    print("  Runtime: 4-6 hours")
    print("  Cost: $3-5")
    print("  Speedup: 3-5x vs sequential")
    print()

    # Execute via gcloud SSH
    gcloud_cmd = r"C:\Users\iansh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"

    print("Launching Docker container on VM...")
    result = subprocess.run(
        [
            gcloud_cmd,
            "compute", "ssh", "civ6az-orchestrator",
            "--zone=us-central1-a",
            "--project=hrmtrain",
            f"--command={docker_cmd}"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print()
        print("=" * 60)
        print("[OK] Orchestrator launched!")
        print("=" * 60)
        print()
        print(f"Container ID: {result.stdout.strip()}")
        print()
        print("Monitor progress:")
        print("  W&B: https://wandb.ai/ianshank-none/civ6-alphazero")
        print()
        print("Check logs:")
        print("  gcloud compute ssh civ6az-orchestrator --zone=us-central1-a")
        print("  docker logs -f civ6az-prod-v150")
        print()
        return 0
    else:
        print()
        print("=" * 60)
        print("[FAIL] Launch failed")
        print("=" * 60)
        print(f"Error: {result.stderr}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
