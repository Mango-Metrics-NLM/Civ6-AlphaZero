
import argparse
import subprocess
import sys

def main():
    """
    Run the Civ6 AlphaZero AI training.
    Default: python -m alpha_zero.training_civ6 --num_games=100 --num_iterations=100
    """
    parser = argparse.ArgumentParser(description="Run Civ6 AlphaZero AI Training")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-dir", type=str, default="logs/civ6", help="Directory for logs")
    parser.add_argument("--dry-run", action="store_true", help="Quick test run")
    
    # Allow passing through other arguments if needed (for experts), but simple interface for now
    args, unknown = parser.parse_known_args()

    print("Starting Civ6 AlphaZero Training...")
    if args.debug:
        print("Debug mode enabled")

    # Base command
    cmd = [
        sys.executable,
        "-m",
        "alpha_zero.training_civ6",
        "--num_games=100",
        "--num_iterations=100"
    ]

    # Add optional flags
    if args.debug:
        cmd.append("--debug")
    
    if args.dry_run:
        cmd.append("--dry_run")
        # Override defaults for dry run to match training_civ6.py's expectation or let it handle it?
        # training_civ6.py handles --dry_run by ignoring num_games/iterations if set, 
        # so passing them is fine, but let's be clean.
    
    # Pass log dir
    cmd.append(f"--log_dir={args.log_dir}")
    
    # Pass any unknown args directly (flexibility)
    if unknown:
        print(f"Passing through unknown arguments: {unknown}")
        cmd.extend(unknown)
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        # Run using subprocess to properly handle standard I/O
        process = subprocess.run(cmd, check=True)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
