"""CLI entry point for deep analysis of completed training runs.

Fetches data from W&B and optionally GCS, runs selected analysis engines,
renders results to the terminal, and optionally exports JSON reports.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from alpha_zero.analysis.data_model import RunData

logger = logging.getLogger(__name__)

ENGINE_NAMES = ("loss", "model", "cost", "diagnostics", "selfplay", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deep analysis of completed Civ6 AlphaZero training runs",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id", help="W&B run ID")
    source.add_argument("--group", help="W&B group name")

    parser.add_argument("--gcs-bucket", help="GCS bucket (default: from env)")
    parser.add_argument("--no-gcs", action="store_true", help="Skip GCS data")
    parser.add_argument(
        "--project", default="civ6-alphazero", help="W&B project"
    )
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument("--output", "-o", type=Path, help="JSON output path")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress CLI output"
    )
    parser.add_argument(
        "--engines",
        nargs="*",
        choices=ENGINE_NAMES,
        default=["all"],
        help="Which analysis engines to run (default: all)",
    )
    return parser


def _should_run(engine: str, selected: list[str]) -> bool:
    return "all" in selected or engine in selected


def run_engines(run_data: RunData, engines: list[str]) -> dict[str, Any]:
    """Run selected analysis engines and return results dict."""
    results: dict[str, Any] = {}

    if _should_run("loss", engines):
        from alpha_zero.analysis.engines.loss_analysis import LossAnalyzer

        results["loss_analysis"] = LossAnalyzer().analyze(run_data)

    if _should_run("model", engines):
        from alpha_zero.analysis.engines.model_progression import (
            ModelProgressionAnalyzer,
        )

        results["model_progression"] = ModelProgressionAnalyzer().analyze(run_data)

    if _should_run("cost", engines):
        from alpha_zero.analysis.engines.cost_analysis import CostAnalyzer

        results["cost_analysis"] = CostAnalyzer().analyze(run_data)

    if _should_run("diagnostics", engines):
        from alpha_zero.analysis.engines.diagnostics import DiagnosticsAnalyzer

        results["diagnostics"] = DiagnosticsAnalyzer().analyze(run_data)

    if _should_run("selfplay", engines):
        from alpha_zero.analysis.engines.selfplay_quality import (
            SelfplayQualityAnalyzer,
        )

        results["selfplay_quality"] = SelfplayQualityAnalyzer().analyze(run_data)

    return results


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    # 1. Fetch data from W&B
    from alpha_zero.analysis.fetchers.wandb_fetcher import WandbRunFetcher

    fetcher = WandbRunFetcher(project=args.project, entity=args.entity)

    if args.run_id:
        run_data = fetcher.fetch_run(args.run_id)
    else:
        runs = fetcher.fetch_group(args.group)
        if not runs:
            logger.error("No runs found in group: %s", args.group)
            sys.exit(1)
        # Use the first (orchestrator) run for analysis
        run_data = runs[0]

    # 2. Augment with GCS (optional)
    if not args.no_gcs:
        try:
            from alpha_zero.analysis.fetchers.gcs_fetcher import GCSRunFetcher

            gcs_fetcher = GCSRunFetcher()
            run_data = gcs_fetcher.augment_run_data(run_data)
        except (ImportError, Exception) as e:
            logger.warning("GCS augmentation skipped: %s", e)

    # 3. Run engines
    results = run_engines(run_data, args.engines)

    # 4. CLI output
    if not args.quiet:
        from alpha_zero.analysis.report.cli_renderer import CLIRenderer

        print(CLIRenderer().render(results, run_data))

    # 5. JSON export
    if args.output:
        from alpha_zero.analysis.report.json_exporter import (
            AnalysisReport,
            JSONExporter,
        )

        report = AnalysisReport(
            run_identifier=run_data.identifier.to_dict(),
            **{
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in results.items()
            },
        )
        JSONExporter().export(report, args.output)
        logger.info("Report saved to: %s", args.output)
