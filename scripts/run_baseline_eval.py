"""
Baseline evaluation of any trained AlphaZero model against RandomAgent and GreedyHeuristicAgent.

Logs win/draw/loss counts, rates, and detailed EvaluationResult statistics to an existing
MLflow run.  All 17 metrics are logged under ``baseline/random/*`` and
``baseline/heuristic/*`` namespaces.

Usage::

    # Minimal — auto-detects latest run in experiment
    python scripts/run_baseline_eval.py --model_version 91

    # Full options
    python scripts/run_baseline_eval.py \\
        --model_version 91 --n_games 20 --board_size 8 --max_turns 30 \\
        --mcts_sims 25 --run_id <mlflow_run_id> --tracking_uri mlruns

    # SQLite backend shorthand
    python scripts/run_baseline_eval.py --model_version 91 --sqlite

Environment Variables (auto-loaded from .env if present):
    CIV6AZ_GCS_BUCKET      GCS bucket name (e.g. my-bucket)
    CIV6AZ_VERTEX_PROJECT  GCP project ID (e.g. my-gcp-project)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alpha_zero.evaluation.evaluation_pipeline import EvaluationResult

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_SQLITE_URI = "sqlite:///mlflow.db"
_DEFAULT_EXPERIMENT = "civ6az-local-50iter"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list override (``None`` = ``sys.argv[1:]``).

    Returns:
        Parsed namespace.
    """
    p = argparse.ArgumentParser(
        description="Baseline evaluation of an AlphaZero model vs random/heuristic agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_baseline_eval.py --model_version 91
  python scripts/run_baseline_eval.py --model_version 91 --n_games 30 --sqlite
  python scripts/run_baseline_eval.py --model_version 91 --run_id <id> --tracking_uri mlruns
""",
    )
    p.add_argument(
        "--model_version",
        type=int,
        required=True,
        help="Model version to evaluate (e.g. 91)",
    )
    p.add_argument(
        "--run_id",
        type=str,
        default=None,
        help=(
            "MLflow run ID to attach results to. "
            "If omitted, the latest run in --experiment is used."
        ),
    )
    p.add_argument(
        "--experiment",
        type=str,
        default=_DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name used for run lookup (default: {_DEFAULT_EXPERIMENT!r})",
    )
    tracking_grp = p.add_mutually_exclusive_group()
    tracking_grp.add_argument(
        "--tracking_uri",
        type=str,
        default="mlruns",
        help="MLflow tracking URI (default: mlruns)",
    )
    tracking_grp.add_argument(
        "--sqlite",
        action="store_true",
        default=False,
        help=f"Use SQLite backend ({_DEFAULT_SQLITE_URI!r}) — shorthand for --tracking_uri",
    )
    p.add_argument(
        "--n_games",
        type=int,
        default=20,
        help="Games per baseline agent (default: 20)",
    )
    p.add_argument(
        "--board_size",
        type=int,
        default=8,
        help="Board size — must match training config (default: 8)",
    )
    p.add_argument(
        "--max_turns",
        type=int,
        default=30,
        help="Max game turns (default: 30)",
    )
    p.add_argument(
        "--mcts_sims",
        type=int,
        default=25,
        help="MCTS simulations per move (default: 25)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device (default: cpu)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base RNG seed for the evaluation. Without it a probe cannot be "
            "repeated: five 20-game probes of one checkpoint scored 0.25, 0.35, "
            "0.375, 0.375 and 0.40 (docs/evidence/validation_iteration_zero/)."
        ),
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    from alpha_zero.utils.game_budget import add_budget_arguments

    add_budget_arguments(p)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# EvaluationResult → MLflow metrics
# ---------------------------------------------------------------------------


def extract_win_rate_metrics(result: Any, prefix: str) -> dict[str, float]:
    """Extract win/draw/loss counts and rates from an ``EvaluationResult``.

    The ``EvaluationResult.win_rate`` field is a ``WinRateResult`` object with:
    - ``.wins``, ``.losses``, ``.draws``, ``.total_games``  (int counts)
    - ``.win_rate``, ``.score_rate``, ``.draw_rate``, ``.loss_rate``  (float, 0-100 %)

    This function re-scales percentages to 0-1 fractions for MLflow convention.

    Args:
        result: ``EvaluationResult`` instance.
        prefix: Metric key prefix (e.g. ``"baseline/random"``).

    Returns:
        Dict of ``{key: float}`` ready for ``mlflow.log_metrics()``.
    """
    wr = getattr(result, "win_rate", None)
    if wr is None:
        logger.warning("EvaluationResult has no 'win_rate' attribute; skipping rate metrics")
        return {}

    metrics: dict[str, float] = {}
    for count_attr in ("wins", "losses", "draws", "total_games"):
        val = getattr(wr, count_attr, None)
        if val is not None:
            try:
                metrics[f"{prefix}/{count_attr}"] = float(val)
            except (TypeError, ValueError):
                logger.debug("Could not convert %s.%s to float", count_attr, val)

    # win_rate/score_rate/draw_rate/loss_rate are 0-100 on WinRateResult — normalise to 0-1
    for rate_attr in ("win_rate", "score_rate", "draw_rate", "loss_rate"):
        val = getattr(wr, rate_attr, None)
        if val is not None:
            try:
                metrics[f"{prefix}/{rate_attr}"] = float(val) / 100.0
            except (TypeError, ValueError):
                logger.debug("Could not convert %s.%s to float", rate_attr, val)

    return metrics


def extract_game_stats_metrics(result: Any, prefix: str) -> dict[str, float]:
    """Extract game-level aggregate statistics from an ``EvaluationResult``.

    Args:
        result: ``EvaluationResult`` instance.
        prefix: Metric key prefix.

    Returns:
        Dict of ``{key: float}`` for MLflow.
    """
    metrics: dict[str, float] = {}
    stat_attrs = (
        "avg_game_length",
        "avg_cities_won",
        "avg_cities_lost",
        "avg_units_won",
        "avg_units_lost",
        "total_evaluation_time",
        "value_correlation",
    )
    for attr in stat_attrs:
        val = getattr(result, attr, None)
        if val is not None:
            try:
                metrics[f"{prefix}/{attr}"] = float(val)
            except (TypeError, ValueError):
                logger.debug("Could not convert result.%s (%r) to float", attr, val)
    return metrics


def result_to_mlflow_metrics(result: EvaluationResult, prefix: str) -> dict[str, float]:
    """Combine win-rate and game-stat metrics from an ``EvaluationResult``.

    Args:
        result: ``EvaluationResult`` instance.
        prefix: Metric key prefix (e.g. ``"baseline/random"``).

    Returns:
        Combined dict of all loggable float metrics.
    """
    metrics: dict[str, float] = {}
    metrics.update(extract_win_rate_metrics(result, prefix))
    metrics.update(extract_game_stats_metrics(result, prefix))
    return metrics


def format_win_rate(result: Any) -> str:
    """Return ASCII-safe win-rate summary string for logging.

    Args:
        result: ``EvaluationResult`` instance.

    Returns:
        Human-readable string like ``"W:4 D:2 L:14 (20.0% win)"``
    """
    wr = getattr(result, "win_rate", None)
    if wr is None:
        return "N/A"
    return (
        f"W:{wr.wins} D:{wr.draws} L:{wr.losses}"
        f" ({wr.win_rate:.1f}% win / {wr.draw_rate:.1f}% draw)"
    )


# ---------------------------------------------------------------------------
# MLflow run resolution
# ---------------------------------------------------------------------------


def resolve_run_id(
    mlflow_module: Any,
    run_id: str | None,
    experiment_name: str,
) -> str:
    """Resolve an MLflow run ID, falling back to the latest run in an experiment.

    Args:
        mlflow_module: The imported ``mlflow`` module.
        run_id: Explicit run ID (returned as-is if not ``None``).
        experiment_name: Experiment name to search for the latest run.

    Returns:
        Resolved run ID string.

    Raises:
        RuntimeError: If no run can be found.
    """
    if run_id:
        logger.debug("Using explicit run_id: %s", run_id)
        return run_id

    logger.info("No --run_id given — searching experiment %r for latest run ...", experiment_name)
    exp = mlflow_module.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(
            f"MLflow experiment {experiment_name!r} not found. "
            "Pass --run_id explicitly or check --tracking_uri."
        )

    runs = mlflow_module.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            f"No runs found in experiment {experiment_name!r}. "
            "Run training first or pass --run_id explicitly."
        )

    resolved = str(runs.iloc[0]["run_id"])
    logger.info("Resolved latest run_id: %s", resolved)
    return resolved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list override (``None`` = ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # --- Game budget: n_games against each of random and heuristic ---
    from alpha_zero.utils.game_budget import enforce_cli

    enforce_cli("run_baseline_eval", {"n_games": args.n_games}, args)

    # --- Load env vars (from .env or os.environ) ---
    from alpha_zero.utils.env_loader import load_civ6az_env

    load_civ6az_env()

    # --- Resolve tracking URI ---
    tracking_uri = _DEFAULT_SQLITE_URI if args.sqlite else args.tracking_uri
    logger.info("MLflow tracking URI: %s", tracking_uri)

    # --- Import mlflow ---
    try:
        import mlflow
    except ImportError as exc:
        logger.error("mlflow not installed: %s", exc)
        return 1

    mlflow.set_tracking_uri(tracking_uri)

    # --- Resolve run_id ---
    try:
        run_id = resolve_run_id(mlflow, args.run_id, args.experiment)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    # --- Load model from GCS ---
    logger.info("Loading model v%d from GCS ...", args.model_version)
    try:
        from alpha_zero.cloud.gcp.config import GCPConfig
        from alpha_zero.cloud.gcp.evaluation_worker import load_model

        gcp_config = GCPConfig()
        model = load_model(gcp_config, args.model_version)
        model = model.to(args.device)
        model.eval()
        logger.info("Model v%d loaded OK on device=%s", args.model_version, args.device)
    except Exception as exc:
        logger.error("Failed to load model v%d: %s", args.model_version, exc, exc_info=True)
        return 1

    # --- Wrap as Civ6MCTS agent ---
    logger.info("Creating Civ6MCTS agent (sims=%d) ...", args.mcts_sims)
    try:
        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig

        mcts_cfg = MCTSConfig(
            num_simulations=args.mcts_sims,
            # Offset past the pipeline's per-game env range (`seed + i`), so the
            # evaluated agent's search does not share a stream with the games.
            seed=None if args.seed is None else args.seed + args.n_games,
        )
        _network: Any = model  # nn.Module satisfies MCTSNetwork at runtime; cast for mypy
        mcts_agent = Civ6MCTS(network=_network, config=mcts_cfg, device=args.device)

        logger.info("Civ6MCTS agent ready")
    except Exception as exc:
        logger.error("Failed to create MCTS agent: %s", exc, exc_info=True)
        return 1

    # --- Build evaluation pipeline ---
    logger.info(
        "EvaluationPipeline(board_size=%d, max_turns=%d, seed=%s)",
        args.board_size,
        args.max_turns,
        args.seed,
    )
    try:
        from alpha_zero.evaluation.evaluation_pipeline import EvaluationPipeline

        pipeline = EvaluationPipeline(
            board_size=args.board_size,
            max_turns=args.max_turns,
            seed=args.seed,
        )
    except Exception as exc:
        logger.error("Failed to build EvaluationPipeline: %s", exc, exc_info=True)
        return 1

    # --- Evaluate vs Random ---
    logger.info("=" * 60)
    logger.info("Evaluating v%d vs RandomAgent (%d games) ...", args.model_version, args.n_games)
    t0 = time.monotonic()
    random_result = None
    try:
        random_result = pipeline.evaluate_vs_random(mcts_agent, n_games=args.n_games)
        logger.info(
            "vs Random   -> %s  (%.1fs)",
            format_win_rate(random_result),
            time.monotonic() - t0,
        )
    except Exception as exc:
        logger.error("Random eval failed: %s", exc, exc_info=True)

    # --- Evaluate vs Heuristic ---
    logger.info(
        "Evaluating v%d vs GreedyHeuristicAgent (%d games) ...",
        args.model_version,
        args.n_games,
    )
    t1 = time.monotonic()
    heuristic_result = None
    try:
        heuristic_result = pipeline.evaluate_vs_heuristic(mcts_agent, n_games=args.n_games)
        logger.info(
            "vs Heuristic -> %s  (%.1fs)",
            format_win_rate(heuristic_result),
            time.monotonic() - t1,
        )
    except Exception as exc:
        logger.error("Heuristic eval failed: %s", exc, exc_info=True)

    logger.info("=" * 60)

    # --- Aggregate metrics ---
    all_metrics: dict[str, float] = {}
    if random_result is not None:
        all_metrics.update(result_to_mlflow_metrics(random_result, "baseline/random"))
    if heuristic_result is not None:
        all_metrics.update(result_to_mlflow_metrics(heuristic_result, "baseline/heuristic"))

    if not all_metrics:
        logger.warning("No metrics to log — both evaluations failed")
        return 1

    # --- Log to MLflow ---
    logger.info("Logging %d metrics to run %s ...", len(all_metrics), run_id)
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(all_metrics)
            mlflow.set_tag("baseline_eval_model", f"v{args.model_version:03d}")
            mlflow.set_tag("baseline_eval_n_games", str(args.n_games))
            mlflow.set_tag("baseline_eval_board_size", str(args.board_size))
            mlflow.set_tag("baseline_eval_mcts_sims", str(args.mcts_sims))
            mlflow.set_tag("baseline_eval_real", "true")
        logger.info("Logged %d metrics to run %s", len(all_metrics), run_id)
    except Exception as exc:
        logger.error("MLflow logging failed: %s", exc, exc_info=True)
        return 1

    # --- ASCII-safe summary ---
    logger.info("=" * 60)
    logger.info("BASELINE EVAL SUMMARY - v%03d", args.model_version)
    logger.info("=" * 60)
    for key, val in sorted(all_metrics.items()):
        logger.info("  %-50s %.4f", key, val)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
