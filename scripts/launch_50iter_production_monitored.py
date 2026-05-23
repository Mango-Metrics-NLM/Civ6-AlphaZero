"""Launch a 50-iteration production training run with MLflow monitoring.

Loads ``configs/orchestrator_production.yaml``, applies env-var overrides so
the run is fully tracked in MLflow (win rates, value correlation, ELO
spread), and starts ``TrainingOrchestrator``.

Usage::

    python scripts/launch_50iter_production_monitored.py --dry-run
    python scripts/launch_50iter_production_monitored.py \
        --tracking-uri file://./mlruns \
        --run-name my_run \
        --iterations 50

All knobs are exposed via flags AND env vars. Flags take precedence over
env. No hardcoded values live in this script.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "orchestrator_production.yaml"
DEFAULT_TRACKING_URI = "file://./mlruns"
DEFAULT_ITERATIONS = 50
DEFAULT_FULL_EVAL_INTERVAL = 5

logger = logging.getLogger("launch_50iter_production_monitored")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Launch a 50-iteration production training run with MLflow tracking enabled.")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to orchestrator YAML config (default: configs/orchestrator_production.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without launching training.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help=f"MLflow tracking URI (default: {DEFAULT_TRACKING_URI})",
    )
    parser.add_argument(
        "--artifact-location",
        type=str,
        default=None,
        help="MLflow artifact_location (e.g. gs://bucket/path). Optional.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name (default: civ6_production_50iter_YYYYMMDD)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (default: run_YYYYMMDD)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of training iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--full-eval-interval",
        type=int,
        default=DEFAULT_FULL_EVAL_INTERVAL,
        help=(
            "Iterations between full EvaluationPipeline runs "
            f"(default: {DEFAULT_FULL_EVAL_INTERVAL})"
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("CIV6AZ_LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def apply_env_overrides(args: argparse.Namespace) -> dict[str, str]:
    """Set ``CIV6AZ_*`` env vars from CLI args. Returns the set values."""
    today = date.today().isoformat().replace("-", "")
    experiment_name = args.experiment_name or f"civ6_production_50iter_{today}"
    run_name = args.run_name or f"run_{today}"
    tracking_uri = args.tracking_uri or DEFAULT_TRACKING_URI

    overrides = {
        "CIV6AZ_MLFLOW_ENABLED": "true",
        "CIV6AZ_MLFLOW_TRACKING_URI": tracking_uri,
        "CIV6AZ_MLFLOW_EXPERIMENT_NAME": experiment_name,
        "CIV6AZ_MLFLOW_RUN_NAME": run_name,
        "CIV6AZ_TRAINING_NUM_ITERATIONS": str(args.iterations),
        "CIV6AZ_TRAINING_FULL_EVAL_INTERVAL": str(args.full_eval_interval),
    }
    if args.artifact_location:
        overrides["CIV6AZ_MLFLOW_ARTIFACT_LOCATION"] = args.artifact_location
    for key, value in overrides.items():
        os.environ[key] = value
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    overrides = apply_env_overrides(args)
    logger.info("Applied env overrides: %s", overrides)

    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        return 2

    # Import after env vars are set so apply_env_overrides() picks them up.
    from alpha_zero.orchestrator.config import OrchestratorConfig
    from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator

    config = OrchestratorConfig.from_yaml(str(args.config))
    config.apply_env_overrides()

    if args.dry_run:
        logger.info(
            "Dry run: experiment=%s run=%s iterations=%d full_eval_interval=%d "
            "tracking_uri=%s mlflow.enabled=%s",
            config.mlflow.experiment_name,
            config.mlflow.run_name,
            config.training.num_iterations,
            config.training.full_eval_interval,
            config.mlflow.tracking_uri,
            config.mlflow.enabled,
        )
        return 0

    orchestrator = TrainingOrchestrator(config)
    if not orchestrator.initialize():
        logger.error("Orchestrator initialization failed")
        return 1
    orchestrator.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
