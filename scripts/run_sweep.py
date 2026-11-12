"""Optuna-backed hyperparameter sweep runner.

Replaces the retired W&B sweep wrapper (``wandb.sweep``/``wandb.agent``).
The search itself lives in the reusable
:mod:`alpha_zero.optimizations.hyperparameter_search` component; this script
is a thin CLI over it.

Usage::

    pip install -e '.[tuning]'                       # optuna
    python scripts/run_sweep.py                              # default config
    python scripts/run_sweep.py --config configs/sweep_muzero.yaml --n-trials 5
    python scripts/run_sweep.py --dry-run                    # validate config only

Trials are logged to MLflow when tracking is enabled
(``CIV6AZ_MLFLOW_ENABLED=true``); studies persist across restarts when an
Optuna storage URL is set (``CIV6AZ_SWEEP_STORAGE=sqlite:///sweeps.db``).
The legacy ``SWEEP_TIMEOUT_SECONDS`` per-trial timeout env var is honored.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from alpha_zero.optimizations.hyperparameter_search import (
    HyperparameterSearch,
    SweepConfig,
)
from alpha_zero.utils.game_budget import add_budget_arguments, enforce_cli

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/sweep_muzero.yaml"

_MUZERO_SCRIPT = "run_muzero_training.py"


def _worst_case_value(spec: Mapping[str, Any]) -> Any:
    """The value of a search-space entry that prices highest (its upper end)."""
    if spec.get("type") == "categorical":
        choices = list(spec["choices"])
        numeric = [c for c in choices if isinstance(c, (int, float))]
        return max(numeric) if len(numeric) == len(choices) else choices[0]
    return spec["high"]


def trial_games(command: Sequence[str], search_space: Mapping[str, Any]) -> int | None:
    """Games ONE trial of *command* plays, or ``None`` if it cannot be priced.

    The command is expanded exactly as a trial would (``{python}`` and each
    ``{param}``, at the top of its range), then parsed by the MuZero runner's
    OWN parser, so an abbreviated option (``--gam``) or a preset (``--quick``)
    resolves the way the child will see it. Anything the parser rejects,
    including an ambiguous abbreviation, is unpriceable and therefore refused.
    """
    values = {"python": sys.executable}
    values.update({name: _worst_case_value(spec) for name, spec in search_space.items()})
    try:
        argv = [str(part).format(**values) for part in command]
    except (KeyError, IndexError, ValueError):
        return None
    scripts = [i for i, tok in enumerate(argv) if tok.replace("\\", "/").endswith(_MUZERO_SCRIPT)]
    if not scripts:
        return None

    scripts_dir = str(Path(__file__).resolve().parent)
    sys.path.insert(0, scripts_dir)
    try:
        runner = importlib.import_module(_MUZERO_SCRIPT.removesuffix(".py"))
    finally:
        sys.path.remove(scripts_dir)
    try:
        with contextlib.redirect_stderr(io.StringIO()):  # argparse prints its error
            args = runner.build_parser().parse_args(argv[scripts[0] + 1 :])
    except SystemExit:
        return None
    iterations, games = runner.resolve_iterations_and_games(args)
    return int(iterations) * int(games)


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
    add_budget_arguments(parser)
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

    # Total across every trial; an unrecognised trial command cannot be priced
    # and is refused unless --approval_ref is given.
    enforce_cli(
        "run_sweep",
        {
            "dry_run": args.dry_run,
            "n_trials": config.n_trials,
            "trial_games": trial_games(config.command, config.search_space),
        },
        args,
    )

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
