"""CLI entry point for the training orchestrator.

Usage:
    python -m alpha_zero.orchestrator --config configs/orchestrator.yaml
    python -m alpha_zero.orchestrator --preset fast --dry_run
    python -m alpha_zero.orchestrator --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="alpha_zero.orchestrator",
        description="Unified Training Orchestrator for Civ6 AlphaZero",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with config file
    python -m alpha_zero.orchestrator --config configs/orchestrator.yaml

    # Run with preset
    python -m alpha_zero.orchestrator --preset fast

    # Dry run to validate configuration
    python -m alpha_zero.orchestrator --config configs/orchestrator.yaml --dry_run

    # Override specific values
    python -m alpha_zero.orchestrator --preset balanced \\
        --iterations 10000 \\
        --games-per-iteration 50

Environment Variables:
    CIV6AZ_WANDB_ENABLED    Enable/disable W&B logging
    CIV6AZ_RANDOM_SEED      Set random seed
    CIV6AZ_DEVICE           Force specific device (cpu/cuda)
""",
    )

    # Configuration source (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    config_group.add_argument(
        "--preset",
        type=str,
        choices=["fast", "balanced", "production"],
        help="Use a preset configuration",
    )

    # Mode options
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configuration without training",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run in distributed mode (not yet implemented)",
    )

    # Override options
    parser.add_argument(
        "--iterations",
        type=int,
        help="Override number of training iterations",
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        dest="games_per_iteration",
        help="Override games per iteration",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        dest="experiment_name",
        help="Override experiment name",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        dest="checkpoint_dir",
        help="Override checkpoint directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    # Feature toggles
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: file://./mlruns)",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG integration",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable curriculum learning",
    )

    # Verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )

    return parser


def setup_logging(verbose: int, quiet: bool) -> None:
    """Configure logging based on verbosity.

    Args:
        verbose: Verbosity level (0-2).
        quiet: If True, suppress non-error output.
    """
    if quiet:
        level = logging.ERROR
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(args: argparse.Namespace) -> Any:
    """Load configuration from args.

    Args:
        args: Parsed command line arguments.

    Returns:
        Loaded OrchestratorConfig.
    """
    from alpha_zero.orchestrator.config import OrchestratorConfig

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
        config = OrchestratorConfig.from_yaml(str(config_path))
        logger.info(f"Loaded config from {config_path}")
    elif args.preset:
        config = OrchestratorConfig.from_preset(args.preset)
        logger.info(f"Using preset: {args.preset}")
    else:
        # Use default config
        config = OrchestratorConfig()
        logger.info("Using default configuration")

    # Apply command-line overrides
    if args.iterations is not None:
        config.training.num_iterations = args.iterations

    if args.games_per_iteration is not None:
        config.training.games_per_iteration = args.games_per_iteration

    if args.experiment_name:
        config.experiment_name = args.experiment_name

    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir

    if args.seed is not None:
        config.random_seed = args.seed

    # Apply feature toggles
    if args.no_wandb:
        config.wandb.enabled = False

    # MLflow flags were added by the PR #78/#79 monitoring branch; the
    # ``getattr`` defaults keep ``load_config`` callable from older test
    # fixtures (or external callers) that build a Namespace without these
    # attributes.
    if getattr(args, "no_mlflow", False):
        config.mlflow.enabled = False
    mlflow_tracking_uri = getattr(args, "mlflow_tracking_uri", None)
    if mlflow_tracking_uri:
        config.mlflow.tracking_uri = mlflow_tracking_uri
    mlflow_experiment_name = getattr(args, "mlflow_experiment_name", None)
    if mlflow_experiment_name:
        config.mlflow.experiment_name = mlflow_experiment_name

    if args.no_rag:
        config.rag.enabled = False

    if args.no_curriculum:
        config.curriculum.enabled = False

    # Apply environment variable overrides
    config = config.apply_env_overrides()

    return config


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success).
    """
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose, args.quiet)

    # Handle distributed mode
    if args.distributed:
        logger.error("Distributed mode is not yet implemented")
        return 1

    # Load configuration
    try:
        config = load_config(args)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Validate configuration
    warnings = config.validate()
    for warning in warnings:
        logger.warning(f"Config: {warning}")

    # Import orchestrator here to avoid slow imports on --help
    from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator

    # Create orchestrator
    orchestrator = TrainingOrchestrator(config)

    # Run training or dry run
    try:
        if args.dry_run:
            logger.info("Running in dry-run mode")
            success = orchestrator.dry_run()
            return 0 if success else 1
        else:
            logger.info(f"Starting training: {config.experiment_name}")
            orchestrator.run()
            return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose >= 1:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
