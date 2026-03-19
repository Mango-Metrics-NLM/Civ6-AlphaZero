#!/usr/bin/env python3
"""Launch 20-iteration batch self-play benchmark on GCP VM.

Sprint 6 Phase 3 Task B.2: Validate batch self-play performance at scale.
"""
import platform
import subprocess
import sys
from pathlib import Path

def main():
    """Launch orchestrator on civ6az-orchestrator VM."""

    # Configuration
    iterations = 20
    workers = 4
    games_per_worker = 50
    num_parallel_games = 4
    max_batch_size = 32
    mcts_type = "gumbel"
    mcts_sims = 16
    max_turns = 30
    evaluation_games = 30
    win_threshold = 0.55
    start_model_version = 268

    # GCP settings
    project = "hrmtrain"
    zone = "us-central1-a"
    vm_name = "civ6az-orchestrator"

    # W&B settings
    wandb_project = "civ6-alphazero"
    wandb_entity = "ianshank-none"
    wandb_tags = "batch-selfplay,v1.5.0"

    print("="*60)
    print("Production Launch: v1.5.0 - Batch Self-Play Benchmark")
    print("="*60)
    print()
    print(f"Configuration:")
    print(f"  Iterations: {iterations}")
    print(f"  Workers: {workers}")
    print(f"  Games per worker: {games_per_worker}")
    print(f"  Total games/iter: {workers * games_per_worker}")
    print(f"  Total games: {iterations * workers * games_per_worker}")
    print(f"  Batch self-play: {num_parallel_games} parallel games")
    print(f"  Batch size: {max_batch_size}")
    print(f"  MCTS: {mcts_type} ({mcts_sims} sims)")
    print(f"  Starting from: v{start_model_version}")
    print()
    print(f"Expected:")
    print(f"  Runtime: 4-6 hours")
    print(f"  Cost: $3-5")
    print(f"  Speedup: 3-5x vs sequential baseline")
    print()
    print(f"W&B Dashboard:")
    print(f"  https://wandb.ai/{wandb_entity}/{wandb_project}")
    print()

    # Build orchestrator command
    orchestrator_cmd = (
        "cd /app && "
        "python -u -m alpha_zero.cloud.gcp.vertex_orchestrator "
        f"--iterations={iterations} "
        f"--workers={workers} "
        f"--games_per_worker={games_per_worker} "
        f"--num_parallel_games={num_parallel_games} "
        f"--max_batch_size={max_batch_size} "
        f"--mcts_type={mcts_type} "
        f"--mcts_simulations={mcts_sims} "
        f"--max_turns={max_turns} "
        f"--evaluation_games={evaluation_games} "
        f"--evaluation_win_threshold={win_threshold} "
        f"--start_model_version={start_model_version} "
        f"--wandb_project={wandb_project} "
        f"--wandb_entity={wandb_entity} "
        f"--wandb_tags={wandb_tags}"
    )

    # Determine gcloud path
    if platform.system() == "Windows":
        gcloud_cmd = r"C:\Users\iansh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    else:
        gcloud_cmd = "gcloud"

    # Build SSH command
    ssh_cmd = [
        gcloud_cmd,
        "compute", "ssh", vm_name,
        f"--zone={zone}",
        f"--project={project}",
        f"--command={orchestrator_cmd}"
    ]

    print("Launching orchestrator on VM...")
    print()

    # Execute
    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print()
            print("="*60)
            print("[OK] Production benchmark launched!")
            print("="*60)
            print()
            print("Monitor progress:")
            print(f"  W&B: https://wandb.ai/{wandb_entity}/{wandb_project}")
            print(f"  GCS: gsutil ls gs://civ6az-training/models/civ6az/ | tail -10")
            print()
            print("Check status:")
            print(f"  gcloud compute ssh {vm_name} --zone={zone}")
            print()
            return 0
        else:
            print()
            print("="*60)
            print("[FAIL] Launch failed")
            print("="*60)
            return 1

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
