#!/usr/bin/env python3
"""Post-run analysis for the v1.6.0 production cloud run.

Reads MLflow runs from a local rsync target (typically
``./mlruns-prod-v160/``), summarises convergence + batch-speedup metrics
against the v1.2.0 baseline, and emits a markdown report suitable for
attaching to the v1.6.0 follow-up PR.

Idempotent: safe to re-run while the cloud run is still in flight; the
report just reflects whatever's landed so far.

Outcome targets (per the v1.5.0 plan / PR #81):
  - At least 1 model promotion (win_rate >= 0.55 vs incumbent on 30 games).
  - Self-play batched games/min at least 3x the v1.2.0 baseline (~4.4 g/m
    at GumbelMCTS 8 sims, 8x8, max_turns=15 per MEMORY.md).
  - Peak GPU memory per batch < 12288 MB.

Usage::

    python scripts/analyze_prod_v1.6.0.py \\
      --mlruns-dir=./mlruns-prod-v160 \\
      --baseline-games-per-min=4.4 \\
      --output=docs/runs/v1.6.0-prod-results.md
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Outcome target constants (single source of truth; do not duplicate in the
# report body — the report reads these and inlines them).
TARGET_MIN_PROMOTIONS: int = 1
TARGET_MIN_BATCH_SPEEDUP: float = 3.0
TARGET_MAX_PEAK_GPU_MB: float = 12_288.0

#: Default v1.2.0 baseline games/min (GumbelMCTS 8 sims, 8x8, max_turns=15)
#: from MEMORY.md. Override per-run with --baseline-games-per-min.
DEFAULT_BASELINE_GAMES_PER_MIN: float = 4.4


@dataclass
class IterationMetrics:
    """One row of per-iteration metrics extracted from MLflow."""

    iteration: int
    policy_loss: float | None = None
    value_loss: float | None = None
    total_loss: float | None = None
    eval_win_rate: float | None = None
    selfplay_games_per_minute: float | None = None
    peak_gpu_memory_mb: float | None = None
    promoted: bool = False
    new_version: int | None = None


@dataclass
class RunSummary:
    """Aggregated view across all iterations in the run."""

    run_name: str
    iterations: list[IterationMetrics] = field(default_factory=list)

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)

    @property
    def promotion_iterations(self) -> list[IterationMetrics]:
        return [it for it in self.iterations if it.promoted]

    @property
    def num_promotions(self) -> int:
        return len(self.promotion_iterations)

    @property
    def policy_loss_trend(self) -> tuple[float | None, float | None]:
        """Return (first_half_mean, second_half_mean) for the policy loss."""
        losses = [it.policy_loss for it in self.iterations if it.policy_loss is not None]
        if len(losses) < 2:
            return (None, None)
        half = len(losses) // 2
        first = sum(losses[:half]) / max(half, 1)
        second = sum(losses[half:]) / max(len(losses) - half, 1)
        return (first, second)

    @property
    def mean_games_per_minute(self) -> float | None:
        vals = [
            it.selfplay_games_per_minute
            for it in self.iterations
            if it.selfplay_games_per_minute is not None
        ]
        if not vals:
            return None
        return sum(vals) / len(vals)

    @property
    def peak_gpu_memory_mb(self) -> float | None:
        vals = [
            it.peak_gpu_memory_mb for it in self.iterations if it.peak_gpu_memory_mb is not None
        ]
        if not vals:
            return None
        return max(vals)


def _load_mlflow_client(mlruns_dir: Path) -> MlflowClient:  # noqa: F821
    """Construct an MlflowClient pointed at *mlruns_dir*.

    Imported lazily so the script is usable for ``--help`` without mlflow
    installed (the actual analysis obviously needs it).
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise SystemExit(
            "mlflow is required for analysis. Install with `pip install mlflow[gcs]`."
        ) from exc

    tracking_uri = mlruns_dir.resolve().as_uri()  # file:///abs/path
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def _collect_iteration_metrics(
    client: MlflowClient,  # noqa: F821
    experiment_name: str,
) -> list[IterationMetrics]:
    """Aggregate metrics across every MLflow run in *experiment_name*.

    Each iteration of the orchestrator emits multiple per-phase runs
    (selfplay-iterNNN, train-iterNNN, eval-iterNNN). We collapse them
    onto a single IterationMetrics row keyed on the iteration index.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning("MLflow experiment %r not found", experiment_name)
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=10_000,
    )
    logger.info("Loaded %d MLflow runs from experiment %r", len(runs), experiment_name)

    by_iter: dict[int, IterationMetrics] = {}

    for run in runs:
        # Iteration comes from either explicit param/metric or parsed from
        # the run name suffix (e.g. ``civ6az-train-iter003``).
        iteration = _extract_iteration(run)
        if iteration is None:
            continue
        bucket = by_iter.setdefault(iteration, IterationMetrics(iteration=iteration))

        m = run.data.metrics
        # Training metrics
        if "train/policy_loss" in m:
            bucket.policy_loss = float(m["train/policy_loss"])
        if "train/value_loss" in m:
            bucket.value_loss = float(m["train/value_loss"])
        if "train/total_loss" in m:
            bucket.total_loss = float(m["train/total_loss"])
        # Evaluation metrics
        if "eval/win_rate" in m:
            bucket.eval_win_rate = float(m["eval/win_rate"])
        # Self-play metrics
        if "selfplay/games_per_minute" in m:
            bucket.selfplay_games_per_minute = float(m["selfplay/games_per_minute"])
        if "selfplay/peak_gpu_memory_mb" in m:
            bucket.peak_gpu_memory_mb = float(m["selfplay/peak_gpu_memory_mb"])

        # Promotion tags
        tags = run.data.tags or {}
        if tags.get("promoted") == "true" or m.get("eval/promoted") == 1.0:
            bucket.promoted = True
        if "new_version" in tags:
            try:
                bucket.new_version = int(tags["new_version"])
            except ValueError:
                logger.debug("Invalid new_version tag: %r", tags["new_version"])

    return [by_iter[k] for k in sorted(by_iter)]


def _extract_iteration(run: Run) -> int | None:  # noqa: F821
    """Best-effort iteration index extraction from an MLflow Run.

    Order of preference:
    1. ``params["iteration"]``
    2. ``metrics["iteration"]``
    3. parsed from ``run_name`` suffix ``-iterNNN``
    """
    params = run.data.params or {}
    metrics = run.data.metrics or {}

    if "iteration" in params:
        try:
            return int(params["iteration"])
        except ValueError:
            pass
    if "iteration" in metrics:
        return int(metrics["iteration"])

    name = run.info.run_name or ""
    if "-iter" in name:
        suffix = name.rsplit("-iter", 1)[-1]
        digits = "".join(c for c in suffix if c.isdigit())
        if digits:
            return int(digits)
    return None


def evaluate_targets(
    summary: RunSummary, baseline_games_per_min: float
) -> dict[str, tuple[bool, str]]:
    """Compare aggregated metrics against the v1.6.0 outcome targets.

    Returns a mapping of ``{check_name: (passed, human_readable_detail)}``.
    """
    checks: dict[str, tuple[bool, str]] = {}

    # Promotions
    passed_promo = summary.num_promotions >= TARGET_MIN_PROMOTIONS
    checks["promotions"] = (
        passed_promo,
        f"{summary.num_promotions} promotion(s) (target: >= {TARGET_MIN_PROMOTIONS})",
    )

    # Batch speedup
    mean_gpm = summary.mean_games_per_minute
    if mean_gpm is None or baseline_games_per_min <= 0:
        checks["batch_speedup"] = (False, "no self-play games/min logged")
    else:
        speedup = mean_gpm / baseline_games_per_min
        passed_sp = speedup >= TARGET_MIN_BATCH_SPEEDUP
        checks["batch_speedup"] = (
            passed_sp,
            f"{speedup:.2f}x ({mean_gpm:.2f} g/m vs baseline "
            f"{baseline_games_per_min:.2f} g/m; target: >= "
            f"{TARGET_MIN_BATCH_SPEEDUP:.1f}x)",
        )

    # Peak GPU memory
    peak = summary.peak_gpu_memory_mb
    if peak is None:
        checks["peak_gpu_memory"] = (True, "not logged (CPU run or pre-D1?)")
    else:
        passed_mem = peak < TARGET_MAX_PEAK_GPU_MB
        checks["peak_gpu_memory"] = (
            passed_mem,
            f"{peak:.0f} MB peak (target: < {TARGET_MAX_PEAK_GPU_MB:.0f} MB)",
        )

    return checks


def render_report(
    summary: RunSummary,
    targets: dict[str, tuple[bool, str]],
    baseline_games_per_min: float,
) -> str:
    """Render the full markdown report body."""
    lines: list[str] = []
    lines.append(f"# v1.6.0 production run results — `{summary.run_name}`")
    lines.append("")
    lines.append(f"- Iterations completed: **{summary.num_iterations}**")
    lines.append(f"- Promotions: **{summary.num_promotions}**")
    fh, sh = summary.policy_loss_trend
    if fh is not None and sh is not None:
        ratio = sh / fh if fh > 0 else math.nan
        lines.append(
            f"- Policy loss: first-half mean **{fh:.4f}**, second-half "
            f"mean **{sh:.4f}** (ratio **{ratio:.2f}**, target <= 1.05)"
        )
    lines.append("")

    # Targets section
    lines.append("## Outcome targets")
    lines.append("")
    lines.append("| Target | Status | Detail |")
    lines.append("|---|---|---|")
    for name, (passed, detail) in targets.items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"| `{name}` | {status} | {detail} |")
    lines.append("")
    all_passed = all(p for p, _ in targets.values())
    lines.append(
        f"**Overall:** {'PASS' if all_passed else 'FAIL'} "
        f"({sum(1 for p, _ in targets.values() if p)}/"
        f"{len(targets)} checks)"
    )
    lines.append("")

    # Promotion table
    if summary.promotion_iterations:
        lines.append("## Promotion events")
        lines.append("")
        lines.append("| Iteration | Win rate vs incumbent | New version |")
        lines.append("|---|---|---|")
        for it in summary.promotion_iterations:
            wr = f"{it.eval_win_rate:.1%}" if it.eval_win_rate is not None else "—"
            ver = str(it.new_version) if it.new_version is not None else "—"
            lines.append(f"| {it.iteration} | {wr} | {ver} |")
        lines.append("")

    # Per-iteration table
    lines.append("## Per-iteration metrics")
    lines.append("")
    lines.append("| Iter | policy_loss | value_loss | eval_win_rate | games/min | peak_gpu_mb |")
    lines.append("|---|---|---|---|---|---|")
    for it in summary.iterations:
        lines.append(
            f"| {it.iteration} "
            f"| {_fmt(it.policy_loss, '.4f')} "
            f"| {_fmt(it.value_loss, '.4f')} "
            f"| {_fmt(it.eval_win_rate, '.1%')} "
            f"| {_fmt(it.selfplay_games_per_minute, '.2f')} "
            f"| {_fmt(it.peak_gpu_memory_mb, '.0f')} |"
        )
    lines.append("")
    lines.append(
        f"_Baseline v1.2.0 self-play: ~{baseline_games_per_min:.2f} games/min "
        "(GumbelMCTS 8 sims, 8x8, max_turns=15 per MEMORY.md)._"
    )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None, spec: str) -> str:
    """Format ``value`` with ``spec`` or return em-dash if None."""
    if value is None:
        return "—"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mlruns-dir",
        type=Path,
        required=True,
        help="Local MLflow tracking directory (e.g. './mlruns-prod-v160').",
    )
    parser.add_argument(
        "--experiment-name",
        default="v1.6.0-prod",
        help="MLflow experiment to read.",
    )
    parser.add_argument(
        "--baseline-games-per-min",
        type=float,
        default=DEFAULT_BASELINE_GAMES_PER_MIN,
        help="v1.2.0 baseline games/min for speedup computation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown report path. When omitted, prints to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if not args.mlruns_dir.exists():
        logger.error("MLflow directory %s does not exist", args.mlruns_dir)
        return 2

    client = _load_mlflow_client(args.mlruns_dir)
    iterations = _collect_iteration_metrics(client, args.experiment_name)
    if not iterations:
        logger.error(
            "No iteration metrics found in experiment %r under %s",
            args.experiment_name,
            args.mlruns_dir,
        )
        return 3

    summary = RunSummary(run_name=args.experiment_name, iterations=iterations)
    targets = evaluate_targets(summary, args.baseline_games_per_min)
    report = render_report(summary, targets, args.baseline_games_per_min)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        logger.info("Report written to %s", args.output)
    else:
        sys.stdout.write(report)

    all_passed = all(p for p, _ in targets.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
