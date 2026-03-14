"""JSON exporter for training run analysis reports.

Exports analysis results as JSON with a versioned schema, suitable for
archival, comparison, and downstream processing.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AnalysisReport:
    """Versioned analysis report container."""

    schema_version: str = "1.0.0"
    generated_at: str = ""
    run_identifier: dict[str, Any] = field(default_factory=dict)
    loss_analysis: dict[str, Any] | None = None
    model_progression: dict[str, Any] | None = None
    cost_analysis: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    selfplay_quality: dict[str, Any] | None = None


class JSONExporter:
    """Exports analysis results as JSON with versioned schema."""

    def to_dict(self, report: AnalysisReport) -> dict[str, Any]:
        """Convert to dict, omitting None sections."""
        d = dataclasses.asdict(report)
        return {k: v for k, v in d.items() if v is not None}

    def export(self, report: AnalysisReport, output_path: Path) -> None:
        """Write report as JSON to file."""
        if not report.generated_at:
            report.generated_at = datetime.now(timezone.utc).isoformat()
        data = self.to_dict(report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
