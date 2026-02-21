"""Training run comparison script.

Compares two training runs side-by-side (e.g., patch_embed vs hex_crystalline encoder).
Supports both W&B-based comparison and log-file-based fallback.

Usage:
    # W&B mode
    python scripts/compare_training_runs.py --run_a indulgent-date-703 --run_b sunny-fog-469 \
        --labels "Batch,Baseline" --output_dir reports/batch_comparison

    # Log-file mode (when wandb unavailable)
    python scripts/compare_training_runs.py --log_a logs/run_a.log --log_b logs/run_b.log \
        --labels "PatchEmbed,HexCrystalline" --output_dir reports/encoder_comparison
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt

# Optional W&B support
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


@dataclasses.dataclass
class RunMetrics:
    """Metrics extracted from a training run."""

    iterations: list[int]
    policy_loss: list[float]
    value_loss: list[float]
    total_loss: list[float]
    eval_win_rate: list[float]
    cumulative_cost: list[float]
    iteration_duration_min: list[float]
    model_versions: list[int]

    # Tournament ELO data (model_id -> elo_rating)
    tournament_elos: dict[str, float] = dataclasses.field(default_factory=dict)

    # Summary stats
    total_iterations: int = 0
    successful_iterations: int = 0
    promotions: int = 0
    final_policy_loss: float = 0.0
    final_value_loss: float = 0.0
    best_eval_win_rate: float = 0.0
    total_cost: float = 0.0
    total_duration_min: float = 0.0
    final_model_version: int = 0
    encoder_type: str = "Unknown"
    param_count: int = 0


@dataclasses.dataclass
class RunSummary:
    """Summary comparison of two runs."""

    run_a_name: str
    run_b_name: str
    run_a_metrics: RunMetrics
    run_b_metrics: RunMetrics


def setup_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def fetch_wandb_metrics(run_id: str, project: str = "ianshank-none/civ6-alphazero") -> RunMetrics:
    """Fetch metrics from W&B API.

    Args:
        run_id: W&B run ID (e.g., "indulgent-date-703")
        project: W&B project path

    Returns:
        RunMetrics with extracted data
    """
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is not installed. Use log-file mode instead.")

    logger = logging.getLogger(__name__)
    logger.info(f"Fetching W&B run: {run_id}")

    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")

    # Fetch all history columns (no keys filter — avoids returning 0 rows
    # when expected column names don't exist in this run)
    history = run.history(pandas=True)

    import pandas as pd  # guaranteed available since wandb depends on it

    def _get_col(col: str, default: float = 0.0) -> "pd.Series":
        """Return history[col] as a Series, or a zero-filled Series if missing."""
        if col in history.columns:
            return history[col].fillna(default)
        return pd.Series([default] * len(history))

    # Extract metrics — orchestrator logs orchestrator/* keys per iteration.
    # train/* loss keys are not logged by the orchestrator; fill with zeros.
    iterations = list(range(len(history)))
    policy_loss = _get_col("train/policy_loss").tolist()
    value_loss = _get_col("train/value_loss").tolist()
    # Try both total-loss column names
    if "train/loss" in history.columns:
        total_loss = _get_col("train/loss").tolist()
    elif "train/total_loss" in history.columns:
        total_loss = _get_col("train/total_loss").tolist()
    else:
        total_loss = [0.0] * len(history)
    eval_win_rate = _get_col("orchestrator/eval_win_rate").tolist()
    cumulative_cost = _get_col("orchestrator/cumulative_cost").tolist()
    iteration_duration = _get_col("orchestrator/iteration_duration_min").tolist()
    model_versions = _get_col("orchestrator/model_version").tolist()

    # Fetch tournament ELO data from summary or final logged metrics
    tournament_elos: dict[str, float] = {}
    config = run.config
    summary = run.summary

    # Tournament ELO metrics are logged as tournament/elo_{player_id}
    for key, value in summary.items():
        if key.startswith("tournament/elo_") and isinstance(value, (int, float)):
            model_id = key.replace("tournament/elo_", "")
            tournament_elos[model_id] = float(value)

    metrics = RunMetrics(
        iterations=iterations,
        policy_loss=policy_loss,
        value_loss=value_loss,
        total_loss=total_loss,
        eval_win_rate=eval_win_rate,
        cumulative_cost=cumulative_cost,
        iteration_duration_min=iteration_duration,
        model_versions=model_versions,
        tournament_elos=tournament_elos,
        total_iterations=len(iterations),
        successful_iterations=len([v for v in total_loss if v > 0]),
        promotions=len(
            [i for i in range(1, len(model_versions)) if model_versions[i] > model_versions[i - 1]]
        ),
        final_policy_loss=policy_loss[-1] if policy_loss else 0.0,
        final_value_loss=value_loss[-1] if value_loss else 0.0,
        best_eval_win_rate=max(eval_win_rate) if eval_win_rate else 0.0,
        total_cost=cumulative_cost[-1] if cumulative_cost else 0.0,
        total_duration_min=sum(iteration_duration),
        final_model_version=int(model_versions[-1]) if model_versions else 0,
        encoder_type=config.get("encoder_type", "Unknown"),
        param_count=config.get("param_count", 0),
    )

    logger.info(f"Fetched {metrics.total_iterations} iterations from {run_id}")
    return metrics


def parse_log_file(log_path: Path) -> RunMetrics:
    """Parse metrics from a log file.

    Args:
        log_path: Path to log file

    Returns:
        RunMetrics with extracted data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Parsing log file: {log_path}")

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # Initialize accumulators
    iterations = []
    policy_loss = []
    value_loss = []
    total_loss = []
    eval_win_rate = []
    cumulative_cost = []
    iteration_duration = []
    model_versions = []

    # Regex patterns for common log formats
    iteration_pattern = re.compile(r"Iteration\s+(\d+)")
    policy_loss_pattern = re.compile(r"policy_loss[:\s=]+([0-9.]+)")
    value_loss_pattern = re.compile(r"value_loss[:\s=]+([0-9.]+)")
    total_loss_pattern = re.compile(r"total_loss[:\s=]+([0-9.]+)")
    win_rate_pattern = re.compile(r"win_rate[:\s=]+([0-9.]+)")
    cost_pattern = re.compile(r"cumulative_cost[:\s=]+\$?([0-9.]+)")
    duration_pattern = re.compile(r"duration[:\s=]+([0-9.]+)\s*min")
    model_version_pattern = re.compile(r"model_version[:\s=]+v?(\d+)")
    encoder_pattern = re.compile(r"encoder[_\s]?type[:\s=]+(\w+)", re.IGNORECASE)
    param_pattern = re.compile(r"param[_\s]?count[:\s=]+([0-9,]+)")

    encoder_type = "Unknown"
    param_count = 0
    current_iter = -1
    current_cost = 0.0

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            # Track iteration
            iter_match = iteration_pattern.search(line)
            if iter_match:
                current_iter = int(iter_match.group(1))
                iterations.append(current_iter)

            # Policy loss
            pl_match = policy_loss_pattern.search(line)
            if pl_match and current_iter >= 0:
                policy_loss.append(float(pl_match.group(1)))

            # Value loss
            vl_match = value_loss_pattern.search(line)
            if vl_match and current_iter >= 0:
                value_loss.append(float(vl_match.group(1)))

            # Total loss
            tl_match = total_loss_pattern.search(line)
            if tl_match and current_iter >= 0:
                total_loss.append(float(tl_match.group(1)))

            # Win rate
            wr_match = win_rate_pattern.search(line)
            if wr_match and current_iter >= 0:
                eval_win_rate.append(float(wr_match.group(1)))

            # Cost
            cost_match = cost_pattern.search(line)
            if cost_match:
                current_cost = float(cost_match.group(1))
                cumulative_cost.append(current_cost)

            # Duration
            dur_match = duration_pattern.search(line)
            if dur_match and current_iter >= 0:
                iteration_duration.append(float(dur_match.group(1)))

            # Model version
            mv_match = model_version_pattern.search(line)
            if mv_match and current_iter >= 0:
                model_versions.append(int(mv_match.group(1)))

            # Encoder type (config)
            enc_match = encoder_pattern.search(line)
            if enc_match:
                encoder_type = enc_match.group(1)

            # Param count (config)
            param_match = param_pattern.search(line)
            if param_match:
                param_count = int(param_match.group(1).replace(",", ""))

    # Align all lists to same length (use last known value for missing data)
    max_len = max(
        len(iterations),
        len(policy_loss),
        len(value_loss),
        len(total_loss),
        len(eval_win_rate),
        len(cumulative_cost),
        len(iteration_duration),
        len(model_versions),
    )

    def pad_list(lst: list, length: int, default: float = 0.0) -> list:
        """Pad list to target length using last value or default."""
        if not lst:
            return [default] * length
        padded = lst.copy()
        while len(padded) < length:
            padded.append(padded[-1] if padded else default)
        return padded[:length]

    iterations = list(range(max_len)) if not iterations else pad_list(iterations, max_len, 0)
    policy_loss = pad_list(policy_loss, max_len, 0.0)
    value_loss = pad_list(value_loss, max_len, 0.0)
    total_loss = pad_list(total_loss, max_len, 0.0)
    eval_win_rate = pad_list(eval_win_rate, max_len, 0.0)
    cumulative_cost = pad_list(cumulative_cost, max_len, 0.0)
    iteration_duration = pad_list(iteration_duration, max_len, 0.0)
    model_versions = pad_list(model_versions, max_len, 0)

    metrics = RunMetrics(
        iterations=iterations,
        policy_loss=policy_loss,
        value_loss=value_loss,
        total_loss=total_loss,
        eval_win_rate=eval_win_rate,
        cumulative_cost=cumulative_cost,
        iteration_duration_min=iteration_duration,
        model_versions=model_versions,
        total_iterations=max_len,
        successful_iterations=len([v for v in total_loss if v > 0]),
        promotions=len(
            [i for i in range(1, len(model_versions)) if model_versions[i] > model_versions[i - 1]]
        ),
        final_policy_loss=policy_loss[-1] if policy_loss else 0.0,
        final_value_loss=value_loss[-1] if value_loss else 0.0,
        best_eval_win_rate=max(eval_win_rate) if eval_win_rate else 0.0,
        total_cost=cumulative_cost[-1] if cumulative_cost else 0.0,
        total_duration_min=sum(iteration_duration),
        final_model_version=int(model_versions[-1]) if model_versions else 0,
        encoder_type=encoder_type,
        param_count=param_count,
    )

    logger.info(f"Parsed {metrics.total_iterations} iterations from {log_path}")
    return metrics


def plot_loss_curves(summary: RunSummary, output_path: Path, labels: tuple[str, str]) -> None:
    """Generate loss curves comparison plot.

    Args:
        summary: RunSummary with both runs' metrics
        output_path: Path to save plot
        labels: Tuple of (label_a, label_b) for legend
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_a = summary.run_a_metrics
    metrics_b = summary.run_b_metrics
    label_a, label_b = labels

    # Policy loss
    axes[0].plot(metrics_a.iterations, metrics_a.policy_loss, label=label_a, marker="o", alpha=0.7)
    axes[0].plot(metrics_b.iterations, metrics_b.policy_loss, label=label_b, marker="s", alpha=0.7)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Policy Loss")
    axes[0].set_title("Policy Loss Over Training")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Value loss
    axes[1].plot(metrics_a.iterations, metrics_a.value_loss, label=label_a, marker="o", alpha=0.7)
    axes[1].plot(metrics_b.iterations, metrics_b.value_loss, label=label_b, marker="s", alpha=0.7)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Value Loss")
    axes[1].set_title("Value Loss Over Training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Total loss
    axes[2].plot(metrics_a.iterations, metrics_a.total_loss, label=label_a, marker="o", alpha=0.7)
    axes[2].plot(metrics_b.iterations, metrics_b.total_loss, label=label_b, marker="s", alpha=0.7)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Total Loss")
    axes[2].set_title("Total Loss Over Training")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logging.info(f"Saved loss curves to {output_path}")


def plot_win_rates(summary: RunSummary, output_path: Path, labels: tuple[str, str]) -> None:
    """Generate win rate comparison plot.

    Args:
        summary: RunSummary with both runs' metrics
        output_path: Path to save plot
        labels: Tuple of (label_a, label_b) for legend
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_a = summary.run_a_metrics
    metrics_b = summary.run_b_metrics
    label_a, label_b = labels

    ax.plot(
        metrics_a.iterations,
        metrics_a.eval_win_rate,
        label=label_a,
        marker="o",
        alpha=0.7,
        linewidth=2,
    )
    ax.plot(
        metrics_b.iterations,
        metrics_b.eval_win_rate,
        label=label_b,
        marker="s",
        alpha=0.7,
        linewidth=2,
    )

    # Add promotion threshold line
    ax.axhline(y=0.55, color="r", linestyle="--", alpha=0.5, label="Promotion Threshold (55%)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win Rate")
    ax.set_title("Evaluation Win Rate Over Training")
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logging.info(f"Saved win rates to {output_path}")


def plot_cost_comparison(summary: RunSummary, output_path: Path, labels: tuple[str, str]) -> None:
    """Generate cost comparison plot.

    Args:
        summary: RunSummary with both runs' metrics
        output_path: Path to save plot
        labels: Tuple of (label_a, label_b) for legend
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_a = summary.run_a_metrics
    metrics_b = summary.run_b_metrics
    label_a, label_b = labels

    ax.plot(
        metrics_a.iterations,
        metrics_a.cumulative_cost,
        label=label_a,
        marker="o",
        alpha=0.7,
        linewidth=2,
    )
    ax.plot(
        metrics_b.iterations,
        metrics_b.cumulative_cost,
        label=label_b,
        marker="s",
        alpha=0.7,
        linewidth=2,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Cost ($)")
    ax.set_title("Cumulative Cost Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logging.info(f"Saved cost comparison to {output_path}")


def plot_iteration_duration(
    summary: RunSummary, output_path: Path, labels: tuple[str, str]
) -> None:
    """Generate iteration duration comparison plot.

    Shows per-iteration timing to measure computational overhead of different encoders.

    Args:
        summary: RunSummary with both runs' metrics
        output_path: Path to save plot
        labels: Tuple of (label_a, label_b) for legend
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics_a = summary.run_a_metrics
    metrics_b = summary.run_b_metrics
    label_a, label_b = labels

    # Per-iteration duration
    axes[0].plot(
        metrics_a.iterations,
        metrics_a.iteration_duration_min,
        label=label_a,
        marker="o",
        alpha=0.7,
        linewidth=2,
    )
    axes[0].plot(
        metrics_b.iterations,
        metrics_b.iteration_duration_min,
        label=label_b,
        marker="s",
        alpha=0.7,
        linewidth=2,
    )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Duration (minutes)")
    axes[0].set_title("Per-Iteration Duration")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bar chart of average duration
    avg_a = sum(metrics_a.iteration_duration_min) / max(len(metrics_a.iteration_duration_min), 1)
    avg_b = sum(metrics_b.iteration_duration_min) / max(len(metrics_b.iteration_duration_min), 1)
    bars = axes[1].bar([label_a, label_b], [avg_a, avg_b], color=["#1f77b4", "#ff7f0e"], alpha=0.8)
    axes[1].set_ylabel("Avg Duration (minutes)")
    axes[1].set_title("Average Iteration Duration")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, [avg_a, avg_b], strict=False):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.1f}m",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Overhead annotation
    if avg_b > 0:
        overhead = (avg_a - avg_b) / avg_b * 100
        if abs(overhead) > 1:
            sign = "+" if overhead > 0 else ""
            axes[1].set_xlabel(f"Overhead: {sign}{overhead:.1f}%")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logging.info(f"Saved iteration duration to {output_path}")


def plot_elo_comparison(summary: RunSummary, output_path: Path, labels: tuple[str, str]) -> bool:
    """Generate tournament ELO comparison plot.

    Shows ELO distributions from post-run tournaments for both runs.
    Returns False if no tournament data is available.

    Args:
        summary: RunSummary with both runs' metrics
        output_path: Path to save plot
        labels: Tuple of (label_a, label_b) for legend

    Returns:
        True if plot was generated, False if no tournament data
    """
    metrics_a = summary.run_a_metrics
    metrics_b = summary.run_b_metrics
    label_a, label_b = labels

    has_a = bool(metrics_a.tournament_elos)
    has_b = bool(metrics_b.tournament_elos)

    if not has_a and not has_b:
        logging.warning("No tournament ELO data available for either run — skipping ELO plot")
        return False

    ncols = 2 if (has_a and has_b) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    plot_idx = 0

    for has_data, elos, label, color in [
        (has_a, metrics_a.tournament_elos, label_a, "#1f77b4"),
        (has_b, metrics_b.tournament_elos, label_b, "#ff7f0e"),
    ]:
        if not has_data:
            continue
        ax = axes[plot_idx] if (has_a and has_b) else axes[0]
        plot_idx += 1

        # Sort models by version number for plotting
        sorted_models = sorted(elos.items(), key=lambda x: _extract_version(x[0]))
        model_ids = [m[0] for m in sorted_models]
        elo_values = [m[1] for m in sorted_models]

        bars = ax.barh(model_ids, elo_values, color=color, alpha=0.7)
        ax.set_xlabel("ELO Rating")
        ax.set_title(f"{label} Tournament ELO")
        ax.grid(True, alpha=0.3, axis="x")

        # Highlight best model
        if elo_values:
            best_idx = elo_values.index(max(elo_values))
            bars[best_idx].set_color("green")
            bars[best_idx].set_alpha(1.0)

        # Add value labels
        for bar, val in zip(bars, elo_values, strict=False):
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}",
                ha="left",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logging.info(f"Saved ELO comparison to {output_path}")
    return True


def _extract_version(model_id: str) -> int:
    """Extract numeric version from model ID string like 'v167' or 'model_167'."""
    match = re.search(r"(\d+)", model_id)
    return int(match.group(1)) if match else 0


def generate_summary_table(summary: RunSummary, output_path: Path, labels: tuple[str, str]) -> None:
    """Generate markdown summary table.

    Args:
        summary: RunSummary with both runs' metrics
        output_path: Path to save markdown table
        labels: Tuple of (label_a, label_b) for legend
    """
    metrics_a = summary.run_a_metrics
    metrics_b = summary.run_b_metrics
    label_a, label_b = labels

    def format_delta(
        a: float, b: float, is_percentage: bool = False, higher_is_better: bool = True
    ) -> str:
        """Format delta with color indicator."""
        delta = a - b
        if abs(delta) < 0.001:
            return "0"

        if is_percentage:
            delta_str = f"{delta:+.1%}"
        else:
            delta_str = f"{delta:+.2f}"

        # Determine if this is an improvement
        is_improvement = (delta > 0) == higher_is_better
        indicator = "✓" if is_improvement else "✗"

        return f"{delta_str} {indicator}"

    table_lines = [
        "# Training Run Comparison Summary",
        "",
        f"**Run A**: {label_a} ({summary.run_a_name})",
        f"**Run B**: {label_b} ({summary.run_b_name})",
        "",
        "| Metric | Run A | Run B | Delta |",
        "|--------|-------|-------|-------|",
        f"| Total iterations | {metrics_a.total_iterations} | {metrics_b.total_iterations} | {metrics_a.total_iterations - metrics_b.total_iterations:+d} |",
        f"| Successful iterations | {metrics_a.successful_iterations} | {metrics_b.successful_iterations} | {metrics_a.successful_iterations - metrics_b.successful_iterations:+d} |",
        f"| Promotions | {metrics_a.promotions} | {metrics_b.promotions} | {metrics_a.promotions - metrics_b.promotions:+d} |",
        f"| Final policy loss | {metrics_a.final_policy_loss:.4f} | {metrics_b.final_policy_loss:.4f} | {format_delta(metrics_a.final_policy_loss, metrics_b.final_policy_loss, higher_is_better=False)} |",
        f"| Final value loss | {metrics_a.final_value_loss:.4f} | {metrics_b.final_value_loss:.4f} | {format_delta(metrics_a.final_value_loss, metrics_b.final_value_loss, higher_is_better=False)} |",
        f"| Best eval win rate | {metrics_a.best_eval_win_rate:.1%} | {metrics_b.best_eval_win_rate:.1%} | {format_delta(metrics_a.best_eval_win_rate, metrics_b.best_eval_win_rate, is_percentage=True)} |",
        f"| Total cost | ${metrics_a.total_cost:.2f} | ${metrics_b.total_cost:.2f} | ${metrics_a.total_cost - metrics_b.total_cost:+.2f} |",
        f"| Total duration (min) | {metrics_a.total_duration_min:.1f} | {metrics_b.total_duration_min:.1f} | {metrics_a.total_duration_min - metrics_b.total_duration_min:+.1f} |",
        f"| Final model version | v{metrics_a.final_model_version} | v{metrics_b.final_model_version} | {metrics_a.final_model_version - metrics_b.final_model_version:+d} |",
        f"| Encoder type | {metrics_a.encoder_type} | {metrics_b.encoder_type} | - |",
        f"| Param count | {metrics_a.param_count:,} | {metrics_b.param_count:,} | {metrics_a.param_count - metrics_b.param_count:+,} |"
        if (metrics_a.param_count > 0 or metrics_b.param_count > 0)
        else "| Param count | N/A | N/A | - |",
        "",
    ]

    output_path.write_text("\n".join(table_lines), encoding="utf-8")
    logging.info(f"Saved summary table to {output_path}")


def generate_report(summary: RunSummary, output_path: Path, labels: tuple[str, str]) -> None:
    """Generate full narrative comparison report.

    Args:
        summary: RunSummary with both runs' metrics
        output_path: Path to save report
        labels: Tuple of (label_a, label_b) for legend
    """
    metrics_a = summary.run_a_metrics
    metrics_b = summary.run_b_metrics
    label_a, label_b = labels

    # Calculate key insights (with division-by-zero protection)
    if metrics_b.final_policy_loss > 0:
        policy_improvement = (
            (metrics_b.final_policy_loss - metrics_a.final_policy_loss)
            / metrics_b.final_policy_loss
            * 100
        )
    else:
        policy_improvement = 0.0

    if metrics_b.final_value_loss > 0:
        value_improvement = (
            (metrics_b.final_value_loss - metrics_a.final_value_loss)
            / metrics_b.final_value_loss
            * 100
        )
    else:
        value_improvement = 0.0

    cost_efficiency = metrics_a.total_cost / max(metrics_b.total_cost, 0.01)
    time_efficiency = metrics_a.total_duration_min / max(metrics_b.total_duration_min, 0.01)

    report_lines = [
        "# Training Run Comparison Report",
        "",
        f"**Generated**: {Path(__file__).name}",
        f"**Run A**: {label_a} ({summary.run_a_name})",
        f"**Run B**: {label_b} ({summary.run_b_name})",
        "",
        "## Executive Summary",
        "",
        "This report compares two training runs to evaluate the impact of different "
        "configurations (e.g., encoder architecture, hyperparameters, optimization strategies).",
        "",
        f"**{label_a}** completed {metrics_a.total_iterations} iterations with "
        f"{metrics_a.promotions} model promotions, achieving a final policy loss of "
        f"{metrics_a.final_policy_loss:.4f} and best win rate of {metrics_a.best_eval_win_rate:.1%}. "
        f"Total cost: ${metrics_a.total_cost:.2f}.",
        "",
        f"**{label_b}** completed {metrics_b.total_iterations} iterations with "
        f"{metrics_b.promotions} model promotions, achieving a final policy loss of "
        f"{metrics_b.final_policy_loss:.4f} and best win rate of {metrics_b.best_eval_win_rate:.1%}. "
        f"Total cost: ${metrics_b.total_cost:.2f}.",
        "",
        "## Key Findings",
        "",
        "### Training Performance",
        "",
        f"- **Policy Loss**: {label_a} achieved {policy_improvement:+.1f}% "
        f"{'improvement' if policy_improvement < 0 else 'regression'} vs {label_b} "
        f"({metrics_a.final_policy_loss:.4f} vs {metrics_b.final_policy_loss:.4f})",
        f"- **Value Loss**: {label_a} achieved {value_improvement:+.1f}% "
        f"{'improvement' if value_improvement < 0 else 'regression'} vs {label_b} "
        f"({metrics_a.final_value_loss:.4f} vs {metrics_b.final_value_loss:.4f})",
        f"- **Win Rate**: {label_a} best win rate was {metrics_a.best_eval_win_rate:.1%} vs "
        f"{label_b} {metrics_b.best_eval_win_rate:.1%} "
        f"({(metrics_a.best_eval_win_rate - metrics_b.best_eval_win_rate)*100:+.1f}pp)",
        "",
        "### Resource Efficiency",
        "",
        f"- **Cost**: {label_a} cost ${metrics_a.total_cost:.2f} vs {label_b} ${metrics_b.total_cost:.2f} "
        f"({cost_efficiency:.2f}x)",
        f"- **Time**: {label_a} took {metrics_a.total_duration_min:.1f} min vs {label_b} "
        f"{metrics_b.total_duration_min:.1f} min ({time_efficiency:.2f}x)",
        f"- **Promotions**: {metrics_a.promotions} vs {metrics_b.promotions} "
        f"({metrics_a.promotions - metrics_b.promotions:+d})",
        "",
        "### Architecture",
        "",
        f"- **{label_a}**: {metrics_a.encoder_type} encoder, {metrics_a.param_count:,} parameters",
        f"- **{label_b}**: {metrics_b.encoder_type} encoder, {metrics_b.param_count:,} parameters",
        "",
        "### Tournament Results",
        "",
    ]

    # Add tournament ELO summary if available
    if metrics_a.tournament_elos:
        elos_a = sorted(metrics_a.tournament_elos.values())
        report_lines.append(
            f"- **{label_a}**: {len(elos_a)} models, "
            f"ELO range {elos_a[0]:.0f}-{elos_a[-1]:.0f} "
            f"(spread: {elos_a[-1] - elos_a[0]:.0f})"
        )
    else:
        report_lines.append(f"- **{label_a}**: No tournament data available")

    if metrics_b.tournament_elos:
        elos_b = sorted(metrics_b.tournament_elos.values())
        report_lines.append(
            f"- **{label_b}**: {len(elos_b)} models, "
            f"ELO range {elos_b[0]:.0f}-{elos_b[-1]:.0f} "
            f"(spread: {elos_b[-1] - elos_b[0]:.0f})"
        )
    else:
        report_lines.append(f"- **{label_b}**: No tournament data available")

    report_lines.extend(
        [
            "",
            "## Visualizations",
            "",
            "- `loss_curves.png` - Policy, value, and total loss comparison",
            "- `win_rates.png` - Evaluation win rate over iterations",
            "- `cost_comparison.png` - Cumulative cost over iterations",
            "- `iteration_duration.png` - Per-iteration timing and overhead comparison",
            "- `elo_comparison.png` - Tournament ELO distributions (if available)",
            "",
            "## Recommendations",
        ]
    )

    report_lines.append("")

    # Add recommendations based on metrics
    if policy_improvement < -5:  # >5% improvement
        report_lines.append(
            f"- **Strong recommendation for {label_a}**: Policy loss improved by "
            f"{abs(policy_improvement):.1f}%, indicating better strategic learning."
        )
    elif policy_improvement > 5:
        report_lines.append(
            f"- **Caution with {label_a}**: Policy loss regressed by "
            f"{policy_improvement:.1f}%, suggesting suboptimal configuration."
        )

    if metrics_a.best_eval_win_rate > metrics_b.best_eval_win_rate + 0.1:
        report_lines.append(
            f"- **{label_a} shows stronger play**: Win rate advantage of "
            f"{(metrics_a.best_eval_win_rate - metrics_b.best_eval_win_rate)*100:.1f}pp "
            f"suggests better generalization."
        )

    if cost_efficiency < 0.8:  # 20%+ cost savings
        report_lines.append(
            f"- **{label_a} is more cost-efficient**: {(1-cost_efficiency)*100:.0f}% cost reduction "
            f"with comparable or better performance."
        )

    if metrics_a.promotions > metrics_b.promotions + 2:
        report_lines.append(
            f"- **{label_a} shows faster improvement**: {metrics_a.promotions} promotions vs "
            f"{metrics_b.promotions}, indicating more consistent progress."
        )

    report_lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Based on this comparison, the recommended configuration depends on priorities:",
            f"- For **best final performance**: {'**' + label_a + '**' if metrics_a.final_policy_loss < metrics_b.final_policy_loss else '**' + label_b + '**'}",
            f"- For **cost efficiency**: {'**' + label_a + '**' if cost_efficiency < 1.0 else '**' + label_b + '**'}",
            f"- For **fastest training**: {'**' + label_a + '**' if time_efficiency < 1.0 else '**' + label_b + '**'}",
            "",
        ]
    )

    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    logging.info(f"Saved comparison report to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare two training runs side-by-side",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # W&B mode
    parser.add_argument(
        "--run_a",
        type=str,
        help="W&B run ID for run A (e.g., 'indulgent-date-703')",
    )
    parser.add_argument(
        "--run_b",
        type=str,
        help="W&B run ID for run B (e.g., 'sunny-fog-469')",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="ianshank-none/civ6-alphazero",
        help="W&B project path (default: hrmtrain/civ6-alphazero)",
    )

    # Log-file mode
    parser.add_argument(
        "--log_a",
        type=Path,
        help="Path to log file for run A (fallback mode)",
    )
    parser.add_argument(
        "--log_b",
        type=Path,
        help="Path to log file for run B (fallback mode)",
    )

    # Output config
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/encoder_comparison"),
        help="Output directory for reports (default: reports/encoder_comparison)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="Comma-separated labels for runs (e.g., 'PatchEmbed,HexCrystalline')",
    )

    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    # Determine mode
    use_wandb = args.run_a and args.run_b
    use_logs = args.log_a and args.log_b

    if not use_wandb and not use_logs:
        parser.error(
            "Must provide either (--run_a, --run_b) for W&B mode or "
            "(--log_a, --log_b) for log-file mode"
        )

    if use_wandb and not WANDB_AVAILABLE:
        logger.error("wandb is not installed. Please install with: pip install wandb")
        logger.info("Falling back to log-file mode if --log_a and --log_b are provided")
        if not use_logs:
            sys.exit(1)
        use_wandb = False

    # Fetch metrics
    try:
        if use_wandb:
            logger.info("Using W&B mode")
            metrics_a = fetch_wandb_metrics(args.run_a, args.project)
            metrics_b = fetch_wandb_metrics(args.run_b, args.project)
            run_a_name = args.run_a
            run_b_name = args.run_b
        else:
            logger.info("Using log-file mode")
            metrics_a = parse_log_file(args.log_a)
            metrics_b = parse_log_file(args.log_b)
            run_a_name = args.log_a.name
            run_b_name = args.log_b.name
    except Exception as e:
        logger.error(f"Failed to fetch metrics: {e}", exc_info=True)
        sys.exit(1)

    # Parse labels
    if args.labels:
        labels = tuple(args.labels.split(",", 1))
        if len(labels) != 2:
            parser.error("--labels must be comma-separated pair (e.g., 'RunA,RunB')")
    else:
        labels = (run_a_name, run_b_name)

    # Create summary
    summary = RunSummary(
        run_a_name=run_a_name,
        run_b_name=run_b_name,
        run_a_metrics=metrics_a,
        run_b_metrics=metrics_b,
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Generate outputs
    try:
        plot_loss_curves(summary, args.output_dir / "loss_curves.png", labels)
        plot_win_rates(summary, args.output_dir / "win_rates.png", labels)
        plot_cost_comparison(summary, args.output_dir / "cost_comparison.png", labels)
        plot_iteration_duration(summary, args.output_dir / "iteration_duration.png", labels)
        has_elo = plot_elo_comparison(summary, args.output_dir / "elo_comparison.png", labels)
        generate_summary_table(summary, args.output_dir / "summary_table.md", labels)
        generate_report(summary, args.output_dir / "report.md", labels)
    except Exception as e:
        logger.error(f"Failed to generate outputs: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Comparison complete! Results saved to {args.output_dir.absolute()}")
    logger.info("  - loss_curves.png")
    logger.info("  - win_rates.png")
    logger.info("  - cost_comparison.png")
    logger.info("  - iteration_duration.png")
    if has_elo:
        logger.info("  - elo_comparison.png")
    logger.info("  - summary_table.md")
    logger.info("  - report.md")


if __name__ == "__main__":
    main()
