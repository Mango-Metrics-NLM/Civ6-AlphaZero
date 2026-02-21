"""Search Efficiency Comparison Report Generator.

Generates comparison plots and markdown report for AlphaZero vs MuZero
search efficiency. Works in two modes:
1. W&B mode: pulls search metrics from W&B runs
2. Local mode: reads from benchmark JSON output files

Usage:
    # From benchmark output files
    python scripts/search_efficiency_report.py \\
        --benchmark_results benchmark_results/benchmark_report.json \\
        --output_dir reports/search_efficiency

    # From W&B runs
    python scripts/search_efficiency_report.py \\
        --wandb_run <run_id> \\
        --project ianshank-none/civ6-alphazero-benchmark \\
        --output_dir reports/search_efficiency
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional matplotlib imports
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_AVAILABLE = False


@dataclass
class AlgorithmMetrics:
    """Search efficiency metrics for one algorithm."""

    name: str
    total_env_clones: int = 0
    total_dynamics_calls: int = 0
    total_network_evals: int = 0
    avg_search_time_ms: float = 0.0
    num_games: int = 0
    win_rate: float = 0.0
    games_per_minute: float = 0.0
    avg_game_length: float = 0.0

    @property
    def search_calls_per_game(self) -> float:
        """Total search operations per game."""
        total = self.total_env_clones + self.total_dynamics_calls
        return total / max(self.num_games, 1)

    @property
    def primary_search_calls(self) -> int:
        """Primary search mechanism calls (env clones for AZ, dynamics for MZ)."""
        return self.total_env_clones if self.total_env_clones > 0 else self.total_dynamics_calls


def load_from_benchmark_json(json_path: str) -> tuple[AlgorithmMetrics, AlgorithmMetrics]:
    """Load metrics from benchmark report JSON file.

    Args:
        json_path: Path to benchmark_report.json.

    Returns:
        Tuple of (alphazero_metrics, muzero_metrics).
    """
    with open(json_path) as f:
        data = json.load(f)

    az = data.get("alphazero", {})
    mz = data.get("muzero", {})
    az_search = az.get("search_metrics", {})
    mz_search = mz.get("search_metrics", {})

    az_metrics = AlgorithmMetrics(
        name="AlphaZero",
        total_env_clones=az_search.get("total_env_clones", 0),
        total_dynamics_calls=az_search.get("total_dynamics_calls", 0),
        total_network_evals=az_search.get("total_network_evals", 0),
        avg_search_time_ms=az_search.get("avg_search_time_ms", 0.0),
        num_games=az.get("num_games", 0),
        win_rate=az.get("win_rate", 0.0),
        games_per_minute=az.get("games_per_minute", 0.0),
        avg_game_length=az.get("avg_game_length", 0.0),
    )

    mz_metrics = AlgorithmMetrics(
        name="MuZero",
        total_env_clones=mz_search.get("total_env_clones", 0),
        total_dynamics_calls=mz_search.get("total_dynamics_calls", 0),
        total_network_evals=mz_search.get("total_network_evals", 0),
        avg_search_time_ms=mz_search.get("avg_search_time_ms", 0.0),
        num_games=mz.get("num_games", 0),
        win_rate=mz.get("win_rate", 0.0),
        games_per_minute=mz.get("games_per_minute", 0.0),
        avg_game_length=mz.get("avg_game_length", 0.0),
    )

    return az_metrics, mz_metrics


def load_from_wandb(run_id: str, project: str) -> tuple[AlgorithmMetrics, AlgorithmMetrics]:
    """Load metrics from a W&B benchmark run.

    Args:
        run_id: W&B run ID.
        project: W&B project path (e.g., "entity/project").

    Returns:
        Tuple of (alphazero_metrics, muzero_metrics).
    """
    import wandb

    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")

    summary = run.summary

    az_metrics = AlgorithmMetrics(
        name="AlphaZero",
        total_env_clones=summary.get("alphazero/env_clones", 0),
        total_dynamics_calls=summary.get("alphazero/dynamics_calls", 0),
        total_network_evals=summary.get("alphazero/network_evals", 0),
        win_rate=summary.get("alphazero/win_rate", 0.0),
        games_per_minute=summary.get("alphazero/games_per_minute", 0.0),
        avg_game_length=summary.get("alphazero/avg_game_length", 0.0),
    )

    mz_metrics = AlgorithmMetrics(
        name="MuZero",
        total_env_clones=summary.get("muzero/env_clones", 0),
        total_dynamics_calls=summary.get("muzero/dynamics_calls", 0),
        total_network_evals=summary.get("muzero/network_evals", 0),
        win_rate=summary.get("muzero/win_rate", 0.0),
        games_per_minute=summary.get("muzero/games_per_minute", 0.0),
        avg_game_length=summary.get("muzero/avg_game_length", 0.0),
    )

    return az_metrics, mz_metrics


def generate_bar_chart(
    az: AlgorithmMetrics,
    mz: AlgorithmMetrics,
    output_dir: Path,
) -> list[str]:
    """Generate comparison bar charts.

    Args:
        az: AlphaZero metrics.
        mz: MuZero metrics.
        output_dir: Directory to save plots.

    Returns:
        List of generated file paths.
    """
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping chart generation")
        return []

    generated: list[str] = []
    colors = ["#3498db", "#e67e22"]  # Blue for AZ, Orange for MZ

    # 1. Search calls comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Env Clones\n(AlphaZero)", "Dynamics Calls\n(MuZero)"]
    values = [az.total_env_clones, mz.total_dynamics_calls]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, values, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Total Calls", fontsize=12)
    ax.set_title("Search Mechanism: Environment Clones vs Dynamics Model Calls", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = output_dir / "search_calls.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(str(path))
    logger.info("Saved %s", path)

    # 2. Speed comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["AlphaZero", "MuZero"]
    values = [az.games_per_minute, mz.games_per_minute]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, values, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Games per Minute", fontsize=12)
    ax.set_title("Throughput: Games per Minute", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = output_dir / "throughput.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(str(path))
    logger.info("Saved %s", path)

    # 3. Win rate comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [az.win_rate * 100, mz.win_rate * 100]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, values, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title("Win Rate vs Random Baseline", fontsize=13)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = output_dir / "win_rates.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(str(path))
    logger.info("Saved %s", path)

    return generated


def generate_markdown_report(
    az: AlgorithmMetrics,
    mz: AlgorithmMetrics,
    output_dir: Path,
    chart_files: list[str],
) -> str:
    """Generate markdown comparison report.

    Args:
        az: AlphaZero metrics.
        mz: MuZero metrics.
        output_dir: Directory to write report.
        chart_files: List of chart file paths.

    Returns:
        Path to generated markdown file.
    """
    efficiency_ratio = mz.total_dynamics_calls / max(az.total_env_clones, 1)
    speed_ratio = mz.games_per_minute / max(az.games_per_minute, 0.001)

    lines = [
        "# Search Efficiency: AlphaZero vs MuZero",
        "",
        "## Summary",
        "",
        "| Metric | AlphaZero | MuZero | Ratio |",
        "|--------|----------|--------|-------|",
        f"| Search Calls | {az.total_env_clones:,} (env clones) | "
        f"{mz.total_dynamics_calls:,} (dynamics) | {efficiency_ratio:.2f}x |",
        f"| Network Evals | {az.total_network_evals:,} | "
        f"{mz.total_network_evals:,} | "
        f"{mz.total_network_evals / max(az.total_network_evals, 1):.2f}x |",
        f"| Avg Search Time | {az.avg_search_time_ms:.1f}ms | "
        f"{mz.avg_search_time_ms:.1f}ms | "
        f"{mz.avg_search_time_ms / max(az.avg_search_time_ms, 0.001):.2f}x |",
        f"| Games per Minute | {az.games_per_minute:.1f} | "
        f"{mz.games_per_minute:.1f} | {speed_ratio:.2f}x |",
        f"| Win Rate | {az.win_rate:.2%} | "
        f"{mz.win_rate:.2%} | {mz.win_rate - az.win_rate:+.2%} |",
        f"| Avg Game Length | {az.avg_game_length:.1f} | "
        f"{mz.avg_game_length:.1f} | - |",
        "",
        "## Key Insight",
        "",
        "MuZero's dynamics model eliminates the need for environment cloning during",
        "tree search. Each MCTS simulation step uses a forward pass through the",
        "learned dynamics network instead of cloning the environment state.",
        "",
        f"- **AlphaZero**: {az.total_env_clones:,} environment clones across "
        f"{az.num_games} games",
        f"- **MuZero**: {mz.total_dynamics_calls:,} dynamics model forward passes "
        f"across {mz.num_games} games",
        f"- **Efficiency ratio**: {efficiency_ratio:.2f}x",
        "",
    ]

    # Add chart references
    if chart_files:
        lines.extend(["## Charts", ""])
        for chart in chart_files:
            name = Path(chart).stem.replace("_", " ").title()
            lines.append(f"![{name}]({Path(chart).name})")
            lines.append("")

    md_path = output_dir / "search_efficiency_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Report written to %s", md_path)
    return str(md_path)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate search efficiency comparison report",
    )
    parser.add_argument(
        "--benchmark_results",
        type=str,
        help="Path to benchmark_report.json from muzero_vs_alphazero_benchmark",
    )
    parser.add_argument(
        "--wandb_run",
        type=str,
        help="W&B run ID for benchmark results",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="ianshank-none/civ6-alphazero-benchmark",
        help="W&B project path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/search_efficiency",
        help="Output directory for report and charts",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics from source
    if args.benchmark_results:
        logger.info("Loading from benchmark JSON: %s", args.benchmark_results)
        az, mz = load_from_benchmark_json(args.benchmark_results)
    elif args.wandb_run:
        logger.info("Loading from W&B run: %s", args.wandb_run)
        az, mz = load_from_wandb(args.wandb_run, args.project)
    else:
        logger.error("Must specify --benchmark_results or --wandb_run")
        sys.exit(1)

    # Generate charts
    charts = generate_bar_chart(az, mz, output_dir)

    # Generate markdown report
    report_path = generate_markdown_report(az, mz, output_dir, charts)

    logger.info("Search efficiency report complete: %s", report_path)
    logger.info(
        "Summary: AZ=%d env clones, MZ=%d dynamics calls, ratio=%.2fx",
        az.total_env_clones,
        mz.total_dynamics_calls,
        mz.total_dynamics_calls / max(az.total_env_clones, 1),
    )


if __name__ == "__main__":
    main()
