"""Evaluation report generation.

This module provides report generation in multiple formats (JSON, Markdown)
for evaluation results.

Usage:
    from alpha_zero.evaluation.reports import ReportGenerator, EvaluationReport

    generator = ReportGenerator(config)
    report = generator.generate(results, diversity_metrics, model_elo)
    report.to_json("report.json")
    print(report.to_markdown())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from alpha_zero.evaluation.config import EvaluationConfig
from alpha_zero.evaluation.diversity import DiversityMetrics
from alpha_zero.evaluation.evaluator import EvaluationResult
from alpha_zero.utils.compat import UTC


@dataclass
class EvaluationReport:
    """Complete evaluation report.

    Attributes:
        experiment_name: Name of the evaluation experiment
        model_checkpoint: Path to evaluated model
        timestamp: ISO format timestamp
        baseline_results: Results by baseline name
        diversity_metrics: Diversity metrics (if computed)
        overall_win_rate: Overall win rate across all baselines
        overall_elo: Current ELO rating (if tracked)
        total_games: Total games played
        metadata: Additional report metadata
    """

    experiment_name: str
    model_checkpoint: str
    timestamp: str
    baseline_results: dict[str, EvaluationResult]
    diversity_metrics: DiversityMetrics | None
    overall_win_rate: float
    overall_elo: float | None
    total_games: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_name": self.experiment_name,
            "model_checkpoint": self.model_checkpoint,
            "timestamp": self.timestamp,
            "baseline_results": {k: v.to_dict() for k, v in self.baseline_results.items()},
            "diversity_metrics": (
                self.diversity_metrics.to_dict() if self.diversity_metrics else None
            ),
            "overall_win_rate": self.overall_win_rate,
            "overall_elo": self.overall_elo,
            "total_games": self.total_games,
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        """Save report as JSON file.

        Args:
            path: Output path
            indent: JSON indentation level
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_json(cls, path: str | Path) -> EvaluationReport:
        """Load report from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            EvaluationReport instance
        """
        with open(path) as f:
            data = json.load(f)

        # Reconstruct EvaluationResult objects
        baseline_results = {}
        for name, result_data in data.get("baseline_results", {}).items():
            baseline_results[name] = EvaluationResult(
                baseline_name=result_data["baseline_name"],
                games_played=result_data["games_played"],
                wins=result_data["wins"],
                losses=result_data["losses"],
                draws=result_data["draws"],
                avg_game_length=result_data["avg_game_length"],
            )

        # Reconstruct DiversityMetrics if present
        diversity_data = data.get("diversity_metrics")
        diversity_metrics = DiversityMetrics.from_dict(diversity_data) if diversity_data else None

        return cls(
            experiment_name=data["experiment_name"],
            model_checkpoint=data["model_checkpoint"],
            timestamp=data["timestamp"],
            baseline_results=baseline_results,
            diversity_metrics=diversity_metrics,
            overall_win_rate=data["overall_win_rate"],
            overall_elo=data.get("overall_elo"),
            total_games=data["total_games"],
            metadata=data.get("metadata", {}),
        )

    def to_markdown(self) -> str:
        """Generate markdown report.

        Returns:
            Formatted markdown string
        """
        lines = [
            "# Model Evaluation Report",
            "",
            f"**Experiment:** {self.experiment_name}",
            f"**Model:** {self.model_checkpoint}",
            f"**Timestamp:** {self.timestamp}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Games | {self.total_games} |",
            f"| Overall Win Rate | {self.overall_win_rate:.1%} |",
        ]

        if self.overall_elo is not None:
            lines.append(f"| ELO Rating | {self.overall_elo:.0f} |")

        lines.extend(
            [
                "",
                "## Results by Baseline",
                "",
                "| Baseline | Games | Wins | Losses | Draws | Win Rate | Avg Length |",
                "|----------|-------|------|--------|-------|----------|------------|",
            ]
        )

        for name, result in self.baseline_results.items():
            lines.append(
                f"| {name} | {result.games_played} | {result.wins} | "
                f"{result.losses} | {result.draws} | {result.win_rate:.1%} | "
                f"{result.avg_game_length:.1f} |"
            )

        if self.diversity_metrics:
            lines.extend(
                [
                    "",
                    "## Diversity Metrics",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| Unique Openings | {self.diversity_metrics.unique_openings} |",
                    f"| Opening Entropy | {self.diversity_metrics.opening_entropy:.3f} |",
                    f"| Avg Game Length | {self.diversity_metrics.avg_game_length:.1f} |",
                    f"| Game Length Std | {self.diversity_metrics.game_length_std:.1f} |",
                    f"| Avg Action Entropy | {self.diversity_metrics.avg_action_entropy:.3f} |",
                ]
            )

            if self.diversity_metrics.victory_type_distribution:
                lines.extend(
                    [
                        "",
                        "### Victory Distribution",
                        "",
                        "| Type | Frequency |",
                        "|------|-----------|",
                    ]
                )
                for vtype, freq in self.diversity_metrics.victory_type_distribution.items():
                    lines.append(f"| {vtype} | {freq:.1%} |")

        lines.extend(
            [
                "",
                "---",
                "*Generated by Civ6 AlphaZero Evaluation System*",
            ]
        )

        return "\n".join(lines)

    def to_text(self) -> str:
        """Generate plain text report.

        Returns:
            Formatted text string
        """
        lines = [
            "=" * 60,
            "MODEL EVALUATION REPORT",
            "=" * 60,
            "",
            f"Experiment: {self.experiment_name}",
            f"Model: {self.model_checkpoint}",
            f"Timestamp: {self.timestamp}",
            "",
            "-" * 40,
            "SUMMARY",
            "-" * 40,
            f"Total Games: {self.total_games}",
            f"Overall Win Rate: {self.overall_win_rate:.1%}",
        ]

        if self.overall_elo is not None:
            lines.append(f"ELO Rating: {self.overall_elo:.0f}")

        lines.extend(
            [
                "",
                "-" * 40,
                "RESULTS BY BASELINE",
                "-" * 40,
            ]
        )

        for name, result in self.baseline_results.items():
            lines.extend(
                [
                    "",
                    f"  {name}:",
                    f"    Games: {result.games_played}",
                    f"    Record: {result.wins}W-{result.losses}L-{result.draws}D",
                    f"    Win Rate: {result.win_rate:.1%}",
                    f"    Avg Game Length: {result.avg_game_length:.1f}",
                ]
            )

        if self.diversity_metrics:
            lines.extend(
                [
                    "",
                    "-" * 40,
                    "DIVERSITY METRICS",
                    "-" * 40,
                    f"Unique Openings: {self.diversity_metrics.unique_openings}",
                    f"Opening Entropy: {self.diversity_metrics.opening_entropy:.3f}",
                    f"Avg Game Length: {self.diversity_metrics.avg_game_length:.1f}",
                    f"Game Length Std: {self.diversity_metrics.game_length_std:.1f}",
                    f"Avg Action Entropy: {self.diversity_metrics.avg_action_entropy:.3f}",
                ]
            )

        lines.extend(["", "=" * 60])

        return "\n".join(lines)


class ReportGenerator:
    """Generates evaluation reports in various formats.

    Usage:
        generator = ReportGenerator(config)
        report = generator.generate(results, diversity_metrics, model_elo)
        report.to_json("output/report.json")
        report.to_markdown()
    """

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize report generator.

        Args:
            config: Evaluation configuration
        """
        self.config = config

    def generate(
        self,
        results: dict[str, EvaluationResult],
        diversity_metrics: DiversityMetrics | None = None,
        model_elo: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Generate evaluation report from results.

        Args:
            results: Results by baseline name
            diversity_metrics: Optional diversity metrics
            model_elo: Current ELO rating
            metadata: Additional metadata to include

        Returns:
            EvaluationReport object
        """
        # Calculate overall win rate
        total_wins = sum(r.wins for r in results.values())
        total_games = sum(r.games_played for r in results.values())
        overall_win_rate = total_wins / total_games if total_games > 0 else 0.0

        return EvaluationReport(
            experiment_name=self.config.experiment_name,
            model_checkpoint=self.config.model_checkpoint_path,
            timestamp=datetime.now(UTC).isoformat(),
            baseline_results=results,
            diversity_metrics=diversity_metrics,
            overall_win_rate=overall_win_rate,
            overall_elo=model_elo,
            total_games=total_games,
            metadata=metadata or {},
        )

    def generate_and_save(
        self,
        results: dict[str, EvaluationResult],
        output_dir: str | Path | None = None,
        diversity_metrics: DiversityMetrics | None = None,
        model_elo: float | None = None,
        formats: list[str] | None = None,
    ) -> EvaluationReport:
        """Generate report and save to files.

        Args:
            results: Results by baseline name
            output_dir: Output directory (uses config default if None)
            diversity_metrics: Optional diversity metrics
            model_elo: Current ELO rating
            formats: List of formats to save ("json", "markdown", "text")

        Returns:
            EvaluationReport object
        """
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        formats = formats or ["json", "markdown"]

        report = self.generate(results, diversity_metrics, model_elo)

        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"evaluation_{timestamp}"

        if "json" in formats:
            report.to_json(output_dir / f"{base_name}.json")

        if "markdown" in formats:
            md_path = output_dir / f"{base_name}.md"
            md_path.write_text(report.to_markdown())

        if "text" in formats:
            txt_path = output_dir / f"{base_name}.txt"
            txt_path.write_text(report.to_text())

        return report

    def compare_reports(
        self,
        report1: EvaluationReport,
        report2: EvaluationReport,
    ) -> dict[str, Any]:
        """Compare two evaluation reports.

        Args:
            report1: First report (baseline)
            report2: Second report (comparison)

        Returns:
            Dictionary of comparison metrics
        """
        comparison = {
            "timestamp1": report1.timestamp,
            "timestamp2": report2.timestamp,
            "win_rate_diff": report2.overall_win_rate - report1.overall_win_rate,
            "games_diff": report2.total_games - report1.total_games,
        }

        if report1.overall_elo is not None and report2.overall_elo is not None:
            comparison["elo_diff"] = report2.overall_elo - report1.overall_elo

        # Compare per-baseline
        comparison["baseline_diffs"] = {}
        for name in set(report1.baseline_results.keys()) | set(report2.baseline_results.keys()):
            r1 = report1.baseline_results.get(name)
            r2 = report2.baseline_results.get(name)

            if r1 and r2:
                comparison["baseline_diffs"][name] = {
                    "win_rate_diff": r2.win_rate - r1.win_rate,
                    "games_diff": r2.games_played - r1.games_played,
                }

        # Compare diversity if both have it
        if report1.diversity_metrics and report2.diversity_metrics:
            d1 = report1.diversity_metrics
            d2 = report2.diversity_metrics
            comparison["diversity_diffs"] = {
                "opening_entropy_diff": d2.opening_entropy - d1.opening_entropy,
                "unique_openings_diff": d2.unique_openings - d1.unique_openings,
                "avg_game_length_diff": d2.avg_game_length - d1.avg_game_length,
            }

        return comparison
