from __future__ import annotations

import json
from pathlib import Path

from alpha_zero.analysis.report.json_exporter import AnalysisReport, JSONExporter


class TestSchemaVersion:
    def test_schema_version_present(self) -> None:
        report = AnalysisReport(
            run_identifier={"wandb_run_id": "abc"},
            loss_analysis={"policy_loss_final": 0.5},
        )
        exporter = JSONExporter()
        d = exporter.to_dict(report)
        assert d["schema_version"] == "1.0.0"


class TestNoneSectionsOmitted:
    def test_none_sections_omitted(self) -> None:
        report = AnalysisReport(
            run_identifier={"wandb_run_id": "abc"},
            loss_analysis={"policy_loss_final": 0.5},
        )
        exporter = JSONExporter()
        d = exporter.to_dict(report)
        assert "loss_analysis" in d
        assert "diagnostics" not in d
        assert "model_progression" not in d
        assert "cost_analysis" not in d
        assert "selfplay_quality" not in d


class TestExportToFile:
    def test_export_to_file(self, tmp_path: Path) -> None:
        report = AnalysisReport(
            run_identifier={"wandb_run_id": "abc"},
            loss_analysis={"policy_loss_final": 0.5},
        )
        out = tmp_path / "subdir" / "report.json"
        exporter = JSONExporter()
        exporter.export(report, out)

        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["schema_version"] == "1.0.0"
        assert loaded["loss_analysis"]["policy_loss_final"] == 0.5


class TestGeneratedAt:
    def test_generated_at_auto_populated(self, tmp_path: Path) -> None:
        report = AnalysisReport(
            run_identifier={"wandb_run_id": "abc"},
        )
        assert report.generated_at == ""

        out = tmp_path / "report.json"
        exporter = JSONExporter()
        exporter.export(report, out)

        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert "generated_at" in loaded
        assert loaded["generated_at"] != ""
        # Should be ISO format
        assert "T" in loaded["generated_at"]
