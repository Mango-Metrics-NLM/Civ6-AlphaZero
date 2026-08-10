"""Optuna-backed hyperparameter sweep runner.

Replaces the retired W&B sweep wrapper (``wandb.sweep``/``wandb.agent``).
The search itself lives in the reusable
:mod:`alpha_zero.optimizations.hyperparameter_search` component; this script
is a thin CLI over it.

Usage::

    pip install -e '.[tuning]'                       # optuna
    python run_sweep.py                              # default config
    python run_sweep.py --config configs/sweep_muzero.yaml --n-trials 5
    python run_sweep.py --dry-run                    # validate config only

Trials are logged to MLflow when tracking is enabled
(``CIV6AZ_MLFLOW_ENABLED=true``); studies persist across restarts when an
Optuna storage URL is set (``CIV6AZ_SWEEP_STORAGE=sqlite:///sweeps.db``).
The legacy ``SWEEP_TIMEOUT_SECONDS`` per-trial timeout env var is honored.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from alpha_zero.optimizations.hyperparameter_search import (
    HyperparameterSearch,
    SweepConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/sweep_muzero.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Civ6AZ Optuna hyperparameter sweep")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Sweep YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Override trial count")
    parser.add_argument("--study-name", default=None, help="Override study name")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL for resumable studies (e.g. sqlite:///sweeps.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the resolved plan without running trials",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the sweep. Returns process exit code."""
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Sweep config not found: %s", config_path)
        return 2

    config = SweepConfig.from_yaml(config_path)
    if args.n_trials is not None:
        config.n_trials = args.n_trials
    if args.study_name is not None:
        config.study_name = args.study_name
    if args.storage is not None:
        config.storage = args.storage

    if args.dry_run:
        logger.info(
            "Dry run OK: study=%s %s %s over %d trials; params=%s; command=%s",
            config.study_name,
            config.direction,
            config.metric,
            config.n_trials,
            sorted(config.search_space),
            " ".join(config.command),
        )
        return 0

    result = HyperparameterSearch(config).run()
    logger.info(
        "Best %s=%f with params %s",
        config.metric,
        result.best_value,
        result.best_params,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
