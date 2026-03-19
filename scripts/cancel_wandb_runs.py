#!/usr/bin/env python3
"""Cancel old crashed W&B runs."""
import wandb

def main():
    """Mark old runs as failed."""
    api = wandb.Api()

    old_runs = [
        "gallant-armadillo-1860",
        "dauntless-butterfly-1859",
        "fresh-bird-1858"
    ]

    print("Canceling old crashed W&B runs...")
    for run_name in old_runs:
        try:
            # Find run by name in project
            runs = api.runs("ianshank-none/civ6-alphazero", filters={"display_name": run_name})
            for run in runs:
                print(f"  Marking {run_name} as crashed...")
                run.summary["state"] = "crashed"
                run.update()
                print(f"    [OK] {run_name} marked as crashed")
        except Exception as e:
            print(f"    [FAIL] Could not cancel {run_name}: {e}")

    print("\n[OK] Old runs cleaned up")
    print("Active run: dandy-terrain-1861")
    print("Dashboard: https://wandb.ai/ianshank-none/civ6-alphazero")

if __name__ == "__main__":
    main()
