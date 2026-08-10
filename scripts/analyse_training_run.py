"""
Post-run analysis of AlphaZero training results from MLflow.

Analyses:
  1. Policy loss trend — slope, smoothed curve, convergence detection
  2. Value head stability — CV, monotonicity, per-epoch variance
  3. Training summary JSON — consolidated report

All results are logged back to the originating MLflow run as artifacts and
summary metrics, giving a full picture of the 50-iteration training run
without re-running training.

Usage:
    python scripts/analyse_training_run.py \\
        --run_id 06f85192a5624f8e809536d9a558c1fd \\
        --tracking_uri mlruns

    python scripts/analyse_training_run.py \\
        --experiment civ6az-local-50iter \\
        --run_name local-50iter-1779361232 \\
        --tracking_uri sqlite:///mlflow.db \\
        --output_dir results/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PolicyLossTrend:
    """Policy loss analysis over training iterations.

    Attributes:
        iterations: List of iteration indices.
        avg_policy_loss: Raw per-iteration average policy loss.
        smoothed: Rolling-mean smoothed series (same length, NaN-padded leading entries).
        slope: Linear regression slope (negative = model is learning).
        r_squared: R² of the linear fit (closer to 1 = clean trend).
        converged: True when |slope| < convergence_threshold over the last window.
        convergence_threshold: Threshold used for convergence detection.
        first_loss: Policy loss at iteration 1.
        last_loss: Policy loss at final iteration.
        total_reduction: first_loss - last_loss (positive = improvement).
    """

    iterations: list[int]
    avg_policy_loss: list[float]
    smoothed: list[float]
    slope: float
    r_squared: float
    converged: bool
    convergence_threshold: float
    first_loss: float
    last_loss: float
    total_reduction: float


@dataclass
class ValueHeadStability:
    """Value head loss stability analysis.

    Attributes:
        iterations: List of iteration indices.
        avg_value_loss: Raw per-iteration average value loss.
        mean: Mean value loss across all iterations.
        std: Standard deviation.
        cv: Coefficient of variation = std / mean (lower = more stable).
        monotone_increasing: True if value loss is non-decreasing across all iterations
            (expected during early training as the value head learns to differentiate).
        per_epoch_variance: Mean of per-epoch value loss variance within each iteration
            (uses train/value_loss step-level metrics; None if unavailable).
        first_loss: Value loss at iteration 1.
        last_loss: Value loss at final iteration.
    """

    iterations: list[int]
    avg_value_loss: list[float]
    mean: float
    std: float
    cv: float
    monotone_increasing: bool
    per_epoch_variance: float | None
    first_loss: float
    last_loss: float


@dataclass
class TrainingSummary:
    """Top-level training summary combining all analyses."""

    run_id: str
    experiment_name: str
    total_iterations: int
    total_global_steps: int
    total_elapsed_seconds: float
    policy: PolicyLossTrend
    value: ValueHeadStability
    analysis_version: str = "1.0"


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """Compute a causal rolling mean (no look-ahead).

    For positions where fewer than ``window`` values are available, the mean
    uses all available values (expanding window).

    Args:
        values: Input series.
        window: Smoothing window size.

    Returns:
        Smoothed series of the same length as ``values``.
    """
    smoothed = []
    for i, _ in enumerate(values):
        start = max(0, i - window + 1)
        smoothed.append(sum(values[start : i + 1]) / (i - start + 1))
    return smoothed


def _linear_regression(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Compute slope, intercept, and R² for a simple OLS regression.

    Args:
        x: Independent variable (e.g. iteration indices).
        y: Dependent variable (e.g. loss values).

    Returns:
        Tuple of (slope, intercept, r_squared).
    """
    n = len(x)
    if n < 2:
        return 0.0, y[0] if y else 0.0, 0.0

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y, strict=True))
    ss_xx = sum((xi - x_mean) ** 2 for xi in x)
    ss_yy = sum((yi - y_mean) ** 2 for yi in y)

    if ss_xx == 0:
        return 0.0, y_mean, 0.0

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    r_squared = (ss_xy**2 / (ss_xx * ss_yy)) if ss_yy != 0 else 1.0

    return slope, intercept, r_squared


def _fetch_metric_series(
    client: Any,
    run_id: str,
    metric_key: str,
) -> list[tuple[int, float]]:
    """Fetch full metric history from MLflow, sorted by step.

    Args:
        client: mlflow.tracking.MlflowClient.
        run_id: MLflow run ID.
        metric_key: Metric name to fetch.

    Returns:
        List of (step, value) tuples sorted ascending by step.
    """
    history = client.get_metric_history(run_id, metric_key)
    return sorted(((h.step, h.value) for h in history), key=lambda t: t[0])


def analyse_policy_loss(
    client: Any,
    run_id: str,
    *,
    smoothing_window: int = 5,
    convergence_threshold: float = 0.001,
    convergence_window: int = 10,
) -> PolicyLossTrend:
    """Analyse policy loss trend over training iterations.

    Args:
        client: MLflow tracking client.
        run_id: MLflow run ID to analyse.
        smoothing_window: Rolling-mean window size for smoothing.
        convergence_threshold: Absolute slope threshold below which training
            is considered converged.
        convergence_window: Number of tail iterations used for convergence check.

    Returns:
        PolicyLossTrend with slope, R², convergence flag, and series data.
    """
    series = _fetch_metric_series(client, run_id, "iter/avg_policy_loss")
    if not series:
        logger.warning("No iter/avg_policy_loss metrics found for run %s", run_id)
        return PolicyLossTrend(
            iterations=[],
            avg_policy_loss=[],
            smoothed=[],
            slope=0.0,
            r_squared=0.0,
            converged=False,
            convergence_threshold=convergence_threshold,
            first_loss=float("nan"),
            last_loss=float("nan"),
            total_reduction=float("nan"),
        )

    iterations = [s[0] for s in series]
    losses = [s[1] for s in series]
    smoothed = _rolling_mean(losses, smoothing_window)

    x = [float(i) for i in iterations]
    slope, _, r_squared = _linear_regression(x, losses)

    # Convergence: check slope of tail window
    tail_len = min(convergence_window, len(losses))
    if tail_len >= 2:
        tail_x = x[-tail_len:]
        tail_y = losses[-tail_len:]
        tail_slope, _, _ = _linear_regression(tail_x, tail_y)
        converged = abs(tail_slope) < convergence_threshold
    else:
        converged = False

    return PolicyLossTrend(
        iterations=iterations,
        avg_policy_loss=losses,
        smoothed=smoothed,
        slope=slope,
        r_squared=r_squared,
        converged=converged,
        convergence_threshold=convergence_threshold,
        first_loss=losses[0],
        last_loss=losses[-1],
        total_reduction=losses[0] - losses[-1],
    )


def analyse_value_stability(
    client: Any,
    run_id: str,
) -> ValueHeadStability:
    """Analyse value head stability over training iterations.

    Args:
        client: MLflow tracking client.
        run_id: MLflow run ID to analyse.

    Returns:
        ValueHeadStability with CV, monotonicity flag, and per-epoch variance.
    """
    series = _fetch_metric_series(client, run_id, "iter/avg_value_loss")
    iterations = [s[0] for s in series]
    losses = [s[1] for s in series]

    if not losses:
        return ValueHeadStability(
            iterations=[],
            avg_value_loss=[],
            mean=float("nan"),
            std=float("nan"),
            cv=float("nan"),
            monotone_increasing=False,
            per_epoch_variance=None,
            first_loss=float("nan"),
            last_loss=float("nan"),
        )

    n = len(losses)
    mean = sum(losses) / n
    variance = sum((v - mean) ** 2 for v in losses) / n
    std = math.sqrt(variance)
    cv = std / mean if mean != 0 else float("nan")

    monotone_increasing = all(losses[i] <= losses[i + 1] for i in range(n - 1))

    # Per-epoch variance: use step-level train/value_loss if available
    per_epoch_variance: float | None = None
    try:
        step_series = _fetch_metric_series(client, run_id, "train/value_loss")
        if step_series:
            step_values = [v for _, v in step_series]
            sv_mean = sum(step_values) / len(step_values)
            sv_var = sum((v - sv_mean) ** 2 for v in step_values) / len(step_values)
            per_epoch_variance = sv_var
    except Exception as exc:
        logger.debug("Could not compute per-epoch variance: %s", exc)

    return ValueHeadStability(
        iterations=iterations,
        avg_value_loss=losses,
        mean=mean,
        std=std,
        cv=cv,
        monotone_increasing=monotone_increasing,
        per_epoch_variance=per_epoch_variance,
        first_loss=losses[0],
        last_loss=losses[-1],
    )


def build_training_summary(
    policy: PolicyLossTrend,
    value: ValueHeadStability,
    run_id: str,
    experiment_name: str = "",
    total_iterations: int = 0,
    total_global_steps: int = 0,
    total_elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    """Build a consolidated training summary dictionary.

    Args:
        policy: Policy loss analysis result.
        value: Value head stability result.
        run_id: MLflow run ID.
        experiment_name: Name of the MLflow experiment.
        total_iterations: Total iterations completed.
        total_global_steps: Total gradient steps taken.
        total_elapsed_seconds: Total training time in seconds.

    Returns:
        Dictionary suitable for JSON serialisation.
    """
    return {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "total_iterations": total_iterations,
        "total_global_steps": total_global_steps,
        "total_elapsed_seconds": total_elapsed_seconds,
        "total_elapsed_minutes": round(total_elapsed_seconds / 60, 1),
        "policy_loss": {
            "first": policy.first_loss,
            "last": policy.last_loss,
            "total_reduction": policy.total_reduction,
            "slope": policy.slope,
            "r_squared": policy.r_squared,
            "converged": policy.converged,
            "convergence_threshold": policy.convergence_threshold,
        },
        "value_loss": {
            "first": value.first_loss,
            "last": value.last_loss,
            "mean": value.mean,
            "std": value.std,
            "cv": value.cv,
            "monotone_increasing": value.monotone_increasing,
            "per_epoch_variance": value.per_epoch_variance,
        },
        "analysis_version": "1.0",
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_policy_csv(trend: PolicyLossTrend, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "avg_policy_loss", "smoothed"])
        for it, raw, sm in zip(
            trend.iterations, trend.avg_policy_loss, trend.smoothed, strict=True
        ):
            writer.writerow([it, raw, sm])


def _write_value_csv(stability: ValueHeadStability, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "avg_value_loss"])
        for it, v in zip(stability.iterations, stability.avg_value_loss, strict=True):
            writer.writerow([it, v])


# ---------------------------------------------------------------------------
# MLflow interaction
# ---------------------------------------------------------------------------


def _resolve_run_id(
    client: Any,
    run_id: str | None,
    experiment: str | None,
    run_name: str | None,
) -> str:
    """Resolve a run ID from explicit ID or experiment+name lookup.

    Args:
        client: MlflowClient.
        run_id: Explicit run ID (takes precedence).
        experiment: Experiment name to search in.
        run_name: Run name to search for within the experiment.

    Returns:
        Resolved run ID string.

    Raises:
        ValueError: If the run cannot be found.
    """
    if run_id:
        return run_id

    if not experiment:
        raise ValueError("Either --run_id or --experiment must be specified")

    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise ValueError(f"Experiment not found: {experiment!r}")

    filter_str = f"tags.`mlflow.runName` = '{run_name}'" if run_name else ""
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=filter_str,
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        msg = f"No runs found in experiment {experiment!r}"
        if run_name:
            msg += f" with name {run_name!r}"
        raise ValueError(msg)

    return str(runs[0].info.run_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Analyse AlphaZero training run metrics from MLflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyse_training_run.py --run_id 06f85192a5624f8e809536d9a558c1fd
  python scripts/analyse_training_run.py --experiment civ6az-local-50iter --run_name local-50iter-1779361232
  python scripts/analyse_training_run.py --run_id <ID> --tracking_uri sqlite:///mlflow.db
""",
    )
    id_group = p.add_mutually_exclusive_group()
    id_group.add_argument("--run_id", type=str, help="MLflow run ID to analyse")
    id_group.add_argument(
        "--experiment",
        type=str,
        help="Experiment name (picks most recent run if --run_name not given)",
    )
    p.add_argument("--run_name", type=str, help="Run name within the experiment")
    p.add_argument(
        "--tracking_uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI", "mlruns"),
        help="MLflow tracking URI (default: mlruns)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="results/analysis",
        help="Directory for output CSV/JSON files (default: results/analysis)",
    )
    p.add_argument(
        "--smoothing_window",
        type=int,
        default=5,
        help="Rolling-mean window for policy loss smoothing (default: 5)",
    )
    p.add_argument(
        "--convergence_threshold",
        type=float,
        default=0.001,
        help="Tail slope threshold for convergence detection (default: 0.001)",
    )
    p.add_argument(
        "--no_upload",
        action="store_true",
        help="Skip logging analysis artifacts back to the MLflow run",
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    try:
        import mlflow
    except ImportError:
        logger.error("mlflow not installed. Run: pip install mlflow")
        return 1

    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_uri)

    try:
        run_id = _resolve_run_id(
            client,
            run_id=args.run_id,
            experiment=args.experiment,
            run_name=args.run_name,
        )
    except ValueError as exc:
        logger.error("Run resolution failed: %s", exc)
        return 1

    logger.info("Analysing run: %s", run_id)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)

    # ----- Run analysis -----
    policy = analyse_policy_loss(
        client,
        run_id,
        smoothing_window=args.smoothing_window,
        convergence_threshold=args.convergence_threshold,
    )
    value = analyse_value_stability(client, run_id)

    # Pull run-level summary metrics
    metrics = run.data.metrics
    summary = build_training_summary(
        policy=policy,
        value=value,
        run_id=run_id,
        experiment_name=experiment.name,
        total_iterations=int(metrics.get("run/iterations_completed", 0)),
        total_global_steps=int(metrics.get("run/total_global_steps", 0)),
        total_elapsed_seconds=metrics.get("run/total_elapsed_seconds", 0.0),
    )

    # ----- Print to console -----
    print("\n" + "=" * 60)
    print("TRAINING RUN ANALYSIS")
    print("=" * 60)
    print(f"Run ID    : {run_id}")
    print(f"Experiment: {experiment.name}")
    print(f"Iterations: {summary['total_iterations']}")
    print(f"Steps     : {summary['total_global_steps']:,}")
    print(f"Duration  : {summary['total_elapsed_minutes']} min")
    print()
    print("--- Policy Loss ---")
    print(f"  First -> Last : {policy.first_loss:.4f} -> {policy.last_loss:.4f}")
    print(
        f"  Reduction    : {policy.total_reduction:.4f} ({policy.total_reduction / policy.first_loss * 100:.1f}%)"
    )
    print(f"  Slope        : {policy.slope:.6f}  R²={policy.r_squared:.4f}")
    print(
        f"  Converged    : {'YES' if policy.converged else 'NO'} (threshold={policy.convergence_threshold})"
    )
    print()
    print("--- Value Head Stability ---")
    print(f"  First -> Last : {value.first_loss:.6f} -> {value.last_loss:.6f}")
    print(f"  Mean +/- Std : {value.mean:.6f} +/- {value.std:.6f}")
    print(f"  CV           : {value.cv:.3f} ({'STABLE' if value.cv < 0.5 else 'VOLATILE'})")
    print(f"  Monotone inc : {'YES' if value.monotone_increasing else 'NO'}")
    print("=" * 60)

    # ----- Write outputs -----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_csv = output_dir / "policy_loss_trend.csv"
    value_csv = output_dir / "value_loss_stability.csv"
    summary_json = output_dir / "training_summary.json"

    _write_policy_csv(policy, policy_csv)
    _write_value_csv(value, value_csv)
    summary_json.write_text(json.dumps(summary, indent=2))

    logger.info("Written: %s", policy_csv)
    logger.info("Written: %s", value_csv)
    logger.info("Written: %s", summary_json)

    # ----- Upload artifacts to MLflow -----
    if not args.no_upload:
        mlflow.set_tracking_uri(args.tracking_uri)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(policy_csv), artifact_path="analysis")
            mlflow.log_artifact(str(value_csv), artifact_path="analysis")
            mlflow.log_artifact(str(summary_json), artifact_path="analysis")
            mlflow.log_metrics(
                {
                    "analysis/policy_slope": policy.slope,
                    "analysis/policy_r_squared": policy.r_squared,
                    "analysis/policy_reduction_pct": (
                        policy.total_reduction / policy.first_loss * 100
                        if policy.first_loss
                        else 0.0
                    ),
                    "analysis/value_cv": value.cv if not math.isnan(value.cv) else 0.0,
                    "analysis/value_monotone": float(value.monotone_increasing),
                    "analysis/converged": float(policy.converged),
                }
            )
        logger.info("Analysis artifacts uploaded to MLflow run %s", run_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
