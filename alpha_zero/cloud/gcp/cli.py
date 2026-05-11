"""GCP CLI for AlphaZero.

Provides command-line utilities for managing GCP resources,
validating configuration, and launching training jobs on Vertex AI.

Usage:
    # Validate GCP configuration
    python -m alpha_zero.cloud.gcp validate --config configs/civ6_config.yaml

    # Initialize training (verify setup, upload initial model)
    python -m alpha_zero.cloud.gcp init --tier budget

    # Launch self-play workers
    python -m alpha_zero.cloud.gcp selfplay --iteration 1 --workers 10

    # Launch training job
    python -m alpha_zero.cloud.gcp train --iteration 1 --trajectory-iters 1

    # Run full training loop
    python -m alpha_zero.cloud.gcp run --iterations 100 --tier budget

    # Check job status
    python -m alpha_zero.cloud.gcp status
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from alpha_zero.cloud.gcp.config import GCPConfig, VertexTier
from alpha_zero.cloud.gcp.exceptions import ConfigurationError, VertexJobError
from alpha_zero.cloud.gcp.storage import GCSBackend, GCSModelStorage
from alpha_zero.cloud.gcp.vertex_jobs import (
    JobState,
    VertexSelfPlayJob,
    VertexTrainingJob,
)
from alpha_zero.utils.config import ConfigManager

logger = logging.getLogger(__name__)


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate GCP configuration and permissions."""
    print("Validating GCP Configuration...")

    try:
        # 1. Load Configs
        try:
            _ = ConfigManager(args.config)
            print(f"  Civ6 Config '{args.config}' loaded successfully")
        except Exception as e:
            print(f"  Warning: Failed to load '{args.config}': {e}")

        # Load GCP Config from Env
        gcp_config = GCPConfig.from_env(validate=False)
        print(f"  GCP Config: Project={gcp_config.project_id}, Region={gcp_config.region}")

        if not gcp_config.project_id or not gcp_config.bucket:
            print("  ERROR: Project ID or Bucket missing in environment")
            return 1

        # 2. Check Storage Access
        print("\nVerifying Storage Access...")
        storage = GCSBackend(gcp_config.project_id)
        try:
            storage.client.get_bucket(gcp_config.bucket)
            print(f"  Bucket gs://{gcp_config.bucket} is accessible")
        except Exception as e:
            print(f"  ERROR: Failed to access bucket '{gcp_config.bucket}': {e}")
            return 1

        # 3. Check Vertex AI
        print("\nVerifying Vertex AI initialization...")
        from google.cloud import aiplatform

        aiplatform.init(
            project=gcp_config.project_id,
            location=gcp_config.region,
            staging_bucket=f"gs://{gcp_config.bucket}",
        )
        print("  Vertex AI initialized successfully")

        print("\nValidation complete: All checks passed!")
        return 0

    except Exception as e:
        print(f"\nERROR: Validation failed - {e}")
        logger.exception("Validation failed")
        return 1


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_config(tier: str | None = None) -> GCPConfig:
    """Load GCP configuration.

    Args:
        tier: Optional tier preset (budget/ondemand).

    Returns:
        GCPConfig instance.

    Raises:
        ConfigurationError: If configuration is invalid.
    """
    if tier:
        config = GCPConfig.with_tier(VertexTier(tier))
        # Apply environment variables on top of preset
        config.project_id = os.environ.get("CIV6AZ_VERTEX_PROJECT", config.project_id)
        config.region = os.environ.get("CIV6AZ_VERTEX_REGION", config.region)
        config.bucket = os.environ.get("CIV6AZ_GCS_BUCKET", config.bucket)
        config.container_image_uri = os.environ.get(
            "CIV6AZ_CONTAINER_IMAGE_URI", config.container_image_uri
        )
        config.service_account = os.environ.get(
            "CIV6AZ_VERTEX_SERVICE_ACCOUNT", config.service_account
        )
    else:
        config = GCPConfig.from_env(validate=False)

    return config


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize training environment.

    - Validates GCP configuration
    - Verifies GCS bucket access
    - Uploads initial model if needed
    """
    print("Initializing Vertex AI training environment...")

    try:
        config = get_config(args.tier)
        config.validate()
        print(f"  Project: {config.project_id}")
        print(f"  Region: {config.region}")
        print(f"  Bucket: {config.bucket}")
        print(f"  Tier: {config.tier.value}")
    except ConfigurationError as e:
        print(f"ERROR: Configuration invalid - {e}")
        print("\nRequired environment variables:")
        print("  CIV6AZ_VERTEX_PROJECT - GCP project ID")
        print("  CIV6AZ_GCS_BUCKET - GCS bucket name")
        return 1

    # Check GCS bucket access
    print("\nVerifying GCS bucket access...")
    try:
        from google.cloud import storage as gcs

        client = gcs.Client(project=config.project_id)
        bucket = client.bucket(config.bucket)
        if not bucket.exists():
            print(f"ERROR: Bucket gs://{config.bucket} does not exist")
            print(f"  Create it with: gsutil mb -l {config.region} gs://{config.bucket}")
            return 1
        print(f"  Bucket gs://{config.bucket} is accessible")
    except ImportError:
        print("WARNING: google-cloud-storage not installed, skipping bucket check")
    except Exception as e:
        print(f"ERROR: Cannot access bucket - {e}")
        return 1

    # Check container image
    print("\nChecking container image...")
    if config.container_image_uri:
        print(f"  Container: {config.container_image_uri}")
    else:
        print("  WARNING: No container image configured")
        print("  Set CIV6AZ_CONTAINER_IMAGE_URI or build with:")
        print("    gcloud builds submit --config=dockerfiles/cloudbuild.yaml")

    # Upload initial model if requested
    if args.upload_model:
        print("\nUploading initial model...")
        try:
            # Use NetworkArchitectureConfig.from_env() as the canonical source of truth
            # for all architecture defaults — same as selfplay_worker and training_worker.
            from alpha_zero.cloud.gcp.config import NetworkArchitectureConfig
            from alpha_zero.core.transformer_network import (
                Civ6TransformerNetwork,
                TransformerConfig,
            )

            arch = NetworkArchitectureConfig.from_env()
            model_config = TransformerConfig(
                spatial_channels=arch.spatial_channels,
                spatial_size=arch.board_size,
                vector_dim=arch.vector_dim,
                action_space_size=arch.action_space_size,
                num_layers=arch.num_layers,
                embed_dim=arch.embed_dim,
                num_heads=arch.num_heads,
            )
            model = Civ6TransformerNetwork(config=model_config)

            storage = GCSModelStorage(config)
            storage.save_model(
                model,
                version=0,
                metadata={
                    "initial": True,
                    "description": "Initial random model",
                    "arch": arch.to_dict(),
                },
            )
            print(
                f"  Uploaded model v0 to GCS"
                f" (board={arch.board_size}x{arch.board_size},"
                f" actions={arch.action_space_size},"
                f" embed={arch.embed_dim})"
            )
        except Exception as e:
            print(f"ERROR: Failed to upload model - {e}")
            return 1

    print("\nInitialization complete!")
    return 0


def cmd_selfplay(args: argparse.Namespace) -> int:
    """Launch self-play workers."""
    config = get_config(args.tier)
    config.validate()

    print(f"Launching {args.workers} self-play workers for iteration {args.iteration}...")
    print(f"  Model version: {args.model_version}")
    print(f"  Games per worker: {config.self_play.games_per_worker}")
    print(f"  Spot VMs: {config.self_play.spot}")

    if args.dry_run:
        print("\n[DRY RUN] Would submit jobs to Vertex AI")
        return 0

    job_manager = VertexSelfPlayJob(config)

    try:
        job_names = job_manager.launch(
            num_workers=args.workers,
            model_version=args.model_version,
            iteration=args.iteration,
        )

        print(f"\nSubmitted {len(job_names)} jobs:")
        for name in job_names:
            print(f"  - {name}")

        if args.wait:
            print("\nWaiting for jobs to complete...")
            results = job_manager.wait_for_completion(job_names, timeout_seconds=args.timeout)

            successful = sum(1 for r in results if r.success)
            print(f"\nResults: {successful}/{len(results)} workers succeeded")

            if successful < len(results):
                for r in results:
                    if not r.success:
                        print(f"  FAILED: {r.job_name} - {r.error_message}")
                return 1 if successful == 0 else 0

        return 0

    except VertexJobError as e:
        print(f"ERROR: {e}")
        return 1


def cmd_train(args: argparse.Namespace) -> int:
    """Launch training job."""
    config = get_config(args.tier)
    config.validate()

    trajectory_iters = [int(x) for x in args.trajectory_iters.split(",")]

    print(f"Launching training job for iteration {args.iteration}...")
    print(f"  Trajectory iterations: {trajectory_iters}")
    print(f"  Epochs: {config.training.epochs_per_iteration}")
    print(f"  Machine: {config.training.machine_type} + {config.training.accelerator_type}")

    if args.dry_run:
        print("\n[DRY RUN] Would submit training job to Vertex AI")
        return 0

    job_manager = VertexTrainingJob(config)

    try:
        job_name = job_manager.launch(
            iteration=args.iteration,
            trajectory_iterations=trajectory_iters,
            resume_from_checkpoint=args.resume_from,
        )

        print(f"\nSubmitted job: {job_name}")

        if args.wait:
            print("\nWaiting for training to complete...")
            result = job_manager.wait_for_completion(timeout_seconds=args.timeout)

            if result.success:
                print("\nTraining completed successfully!")
                return 0
            else:
                print(f"\nTraining failed: {result.error_message}")
                return 1

        return 0

    except VertexJobError as e:
        print(f"ERROR: {e}")
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Run full training loop."""
    config = get_config(args.tier)
    config.validate()

    print("Starting full training loop")
    print(f"  Iterations: {args.iterations}")
    print(f"  Self-play workers: {args.selfplay_workers}")
    print(f"  Games per worker: {args.games_per_worker or config.self_play.games_per_worker}")
    print(f"  Trajectory window: {args.trajectory_window}")
    print(f"  Tier: {config.tier.value}")

    if args.dry_run:
        print("\n[DRY RUN] Would run training loop")
        return 0

    # Override games per worker if specified
    if args.games_per_worker:
        config.self_play.games_per_worker = args.games_per_worker

    selfplay_manager = VertexSelfPlayJob(config)
    training_manager = VertexTrainingJob(config)

    # Get current model version
    model_storage = GCSModelStorage(config)
    try:
        current_version = model_storage.get_latest_version()
        print(f"\nStarting from model v{current_version}")
    except Exception:
        current_version = 0
        print("\nNo existing model found, starting from v0")

    start_iteration = args.start_iteration or 1

    try:
        for iteration in range(start_iteration, start_iteration + args.iterations):
            print(f"\n{'=' * 60}")
            print(f"ITERATION {iteration}/{start_iteration + args.iterations - 1}")
            print(f"{'=' * 60}")

            # Phase 1: Self-play
            print(f"\n[Phase 1] Self-play with model v{current_version}")
            job_names = selfplay_manager.launch(
                num_workers=args.selfplay_workers,
                model_version=current_version,
                iteration=iteration,
            )

            print(f"  Waiting for {len(job_names)} workers...")
            results = selfplay_manager.wait_for_completion(job_names)
            successful = sum(1 for r in results if r.success)
            print(f"  Self-play complete: {successful}/{len(results)} workers succeeded")

            if successful == 0:
                print("ERROR: All self-play workers failed!")
                return 1

            # Phase 2: Training
            print(
                f"\n[Phase 2] Training on iterations {max(1, iteration - args.trajectory_window + 1)}-{iteration}"
            )
            trajectory_iters = list(
                range(
                    max(1, iteration - args.trajectory_window + 1),
                    iteration + 1,
                )
            )

            training_manager.launch(
                iteration=iteration,
                trajectory_iterations=trajectory_iters,
            )

            print("  Waiting for training to complete...")
            result = training_manager.wait_for_completion()

            if not result.success:
                print(f"ERROR: Training failed - {result.error_message}")
                return 1

            # Update model version
            current_version = iteration
            print(f"\n  Training complete! New model: v{current_version}")

            # Log progress
            print(f"\nIteration {iteration} complete!")

        print(f"\n{'=' * 60}")
        print("TRAINING COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Final model version: v{current_version}")
        print(f"  Total iterations: {args.iterations}")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted! Cancelling active jobs...")
        selfplay_manager.cancel_all()
        training_manager.cancel()
        return 130

    except VertexJobError as e:
        print(f"\nERROR: {e}")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Check status of recent jobs."""
    config = get_config(args.tier)
    config.validate()

    print("Checking Vertex AI job status...")
    print(f"  Project: {config.project_id}")
    print(f"  Region: {config.region}")

    try:
        from google.cloud import aiplatform

        aiplatform.init(project=config.project_id, location=config.region)

        # List recent jobs
        jobs = aiplatform.CustomJob.list(
            filter='display_name:"civ6az-mzwm-"',
            order_by="create_time desc",
        )

        jobs = list(jobs)[: args.limit]

        if not jobs:
            print("\nNo recent Civ6AZ jobs found")
            return 0

        print(f"\nRecent jobs ({len(jobs)}):")
        for job in jobs:
            state = job.state.name if job.state else "UNKNOWN"
            state_indicator = ""
            if state == "JOB_STATE_SUCCEEDED":
                state_indicator = "[OK]"
            elif state == "JOB_STATE_FAILED":
                state_indicator = "[FAIL]"
            elif JobState.is_running(state):
                state_indicator = "[RUNNING]"

            print(f"  {state_indicator:9} {job.display_name}")
            print(f"           State: {state}")
            if job.create_time:
                print(f"           Created: {job.create_time}")
            print()

        return 0

    except ImportError:
        print("ERROR: google-cloud-aiplatform not installed")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


def cmd_orchestrate(args: argparse.Namespace) -> int:
    """Run multi-iteration orchestrated training with state recovery.

    Wraps :func:`run_iterations` from the vertex orchestrator module,
    providing a CLI interface for long-running, preemption-resilient
    training loops.
    """
    from alpha_zero.cloud.gcp.vertex_jobs import VertexAIClient
    from alpha_zero.cloud.gcp.vertex_orchestrator import (
        get_initial_model_version,
        run_iterations,
    )

    config = get_config(getattr(args, "tier", None))
    dry_run = getattr(args, "dry_run", False)

    if not dry_run:
        config.validate()

    client = VertexAIClient(
        project=config.project_id,
        region=config.region,
        service_account=config.service_account,
    )

    # Resolve starting model version from GCS
    model_version = get_initial_model_version(config, dry_run)
    start_iter = getattr(args, "start_iter", 0)
    end_iter = getattr(args, "end_iter", start_iter + 10)
    eval_threshold = getattr(args, "eval_threshold", 0.55)
    max_failures = getattr(args, "max_failures", 3)
    resume = getattr(args, "resume", True)
    run_id = getattr(args, "run_id", "")

    # Build a namespace with required orchestrator fields
    orch_args = argparse.Namespace(
        workers=getattr(args, "workers", 5),
        games_per_worker=getattr(args, "games_per_worker", 50),
        max_retries=getattr(args, "max_retries", 2),
        trajectory_window=getattr(args, "trajectory_window", 5),
        evaluation_games=getattr(args, "evaluation_games", 30),
        mcts_type=getattr(args, "mcts_type", "gumbel"),
        mcts_simulations=getattr(args, "mcts_simulations", 50),
        board_size=getattr(args, "board_size", 8),
        action_space_size=getattr(args, "action_space_size", config.network.action_space_size),
        epochs=getattr(args, "epochs", 10),
        batch_size=getattr(args, "batch_size", 64),
        learning_rate=getattr(args, "learning_rate", 0.001),
        max_turns=getattr(args, "max_turns", 30),
        num_res_blocks=config.network.num_layers,
        num_filters=config.network.embed_dim,
        reward_shaping=False,
        gumbel_num_actions=getattr(args, "gumbel_num_actions", 16),
        evaluation_win_threshold=eval_threshold,
        agent_id=getattr(args, "agent_id", None),
        run_id=run_id,
    )

    print(f"[Orchestrate] Iterations {start_iter}-{end_iter - 1}")
    print(f"[Orchestrate] Model v{model_version}, resume={resume}")
    print(f"[Orchestrate] Eval threshold: {eval_threshold:.0%}")

    state = run_iterations(
        config=config,
        client=client,
        args=orch_args,
        start_iteration=start_iter,
        end_iteration=end_iter,
        initial_model_version=model_version,
        eval_win_threshold=eval_threshold,
        max_consecutive_failures=max_failures,
        dry_run=dry_run,
        resume=resume,
    )

    summary = state.to_summary()
    print(f"\n[Orchestrate] Complete: {summary}")
    return 0 if state.failed_iterations == 0 else 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vertex AI Training CLI for Civ6 AlphaZero",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate GCP configuration")
    validate_parser.add_argument(
        "--config",
        type=str,
        default="configs/civ6_config.yaml",
        help="Path to configuration file",
    )

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize training environment")
    init_parser.add_argument("--tier", choices=["budget", "ondemand"], default="budget")
    init_parser.add_argument("--upload-model", action="store_true", help="Upload initial model")

    # selfplay command
    selfplay_parser = subparsers.add_parser("selfplay", help="Launch self-play workers")
    selfplay_parser.add_argument("--iteration", type=int, required=True, help="Training iteration")
    selfplay_parser.add_argument("--model-version", type=int, default=0, help="Model version")
    selfplay_parser.add_argument("--workers", type=int, default=10, help="Number of workers")
    selfplay_parser.add_argument("--tier", choices=["budget", "ondemand"], default="budget")
    selfplay_parser.add_argument("--wait", action="store_true", help="Wait for completion")
    selfplay_parser.add_argument("--timeout", type=int, default=14400, help="Timeout in seconds")
    selfplay_parser.add_argument("--dry-run", action="store_true", help="Don't submit jobs")

    # train command
    train_parser = subparsers.add_parser("train", help="Launch training job")
    train_parser.add_argument("--iteration", type=int, required=True, help="Training iteration")
    train_parser.add_argument(
        "--trajectory-iters", required=True, help="Comma-separated trajectory iterations"
    )
    train_parser.add_argument("--tier", choices=["budget", "ondemand"], default="budget")
    train_parser.add_argument("--resume-from", help="Checkpoint path to resume from")
    train_parser.add_argument("--wait", action="store_true", help="Wait for completion")
    train_parser.add_argument("--timeout", type=int, default=86400, help="Timeout in seconds")
    train_parser.add_argument("--dry-run", action="store_true", help="Don't submit jobs")

    # run command
    run_parser = subparsers.add_parser("run", help="Run full training loop")
    run_parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    run_parser.add_argument("--start-iteration", type=int, help="Starting iteration number")
    run_parser.add_argument("--selfplay-workers", type=int, default=10, help="Self-play workers")
    run_parser.add_argument("--games-per-worker", type=int, help="Games per worker")
    run_parser.add_argument(
        "--trajectory-window", type=int, default=3, help="Trajectory window size"
    )
    run_parser.add_argument("--tier", choices=["budget", "ondemand"], default="budget")
    run_parser.add_argument("--dry-run", action="store_true", help="Don't submit jobs")

    # orchestrate command
    orch_parser = subparsers.add_parser(
        "orchestrate",
        help="Run multi-iteration training with state recovery",
    )
    orch_parser.add_argument("--start-iter", type=int, default=50, help="First iteration")
    orch_parser.add_argument("--end-iter", type=int, default=100, help="Last iteration (excl)")
    orch_parser.add_argument(
        "--eval-threshold", type=float, default=0.55, help="Min win rate (0-1)"
    )
    orch_parser.add_argument("--workers", type=int, default=5, help="Self-play workers")
    orch_parser.add_argument("--games-per-worker", type=int, default=50)
    orch_parser.add_argument("--evaluation-games", type=int, default=30)
    orch_parser.add_argument("--trajectory-window", type=int, default=5)
    orch_parser.add_argument("--epochs", type=int, default=10)
    orch_parser.add_argument("--max-failures", type=int, default=3, help="Circuit breaker")
    orch_parser.add_argument("--resume", action="store_true", default=True)
    orch_parser.add_argument("--no-resume", dest="resume", action="store_false")
    orch_parser.add_argument("--run-id", type=str, default="", help="Run ID for state scoping")
    orch_parser.add_argument("--agent-id", type=str, help="Agent ID for GCS namespace")
    orch_parser.add_argument("--tier", choices=["budget", "ondemand"], default="budget")
    orch_parser.add_argument("--dry-run", action="store_true", help="Don't submit jobs")

    # status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("--tier", choices=["budget", "ondemand"], default="budget")
    status_parser.add_argument("--limit", type=int, default=10, help="Max jobs to show")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    setup_logging(args.verbose)

    commands = {
        "validate": cmd_validate,
        "init": cmd_init,
        "selfplay": cmd_selfplay,
        "train": cmd_train,
        "run": cmd_run,
        "orchestrate": cmd_orchestrate,
        "status": cmd_status,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
