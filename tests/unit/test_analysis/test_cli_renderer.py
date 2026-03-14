from __future__ import annotations

from alpha_zero.analysis.data_model import RunData, RunIdentifier
from alpha_zero.analysis.report.cli_renderer import CLIRenderer


class TestRenderSectionTitle:
    def test_render_section_title(self) -> None:
        assert CLIRenderer._render_section_title("FOO") == "=== FOO ==="


class TestRenderTable:
    def test_render_table(self) -> None:
        renderer = CLIRenderer()
        headers = ["Name", "Value"]
        rows = [["alpha", "1"], ["beta", "20"]]
        result = renderer._render_table(headers, rows)
        lines = result.strip().split("\n")
        # Header, separator, and two data rows
        assert len(lines) == 4
        assert "Name" in lines[0]
        assert "Value" in lines[0]
        assert "-+-" in lines[1]
        assert "alpha" in lines[2]
        assert "beta" in lines[3]

    def test_render_table_empty_rows(self) -> None:
        renderer = CLIRenderer()
        result = renderer._render_table(["A"], [])
        assert "(no data)" in result


class TestFullRender:
    def test_full_render_includes_header(self, sample_run_data: RunData) -> None:
        renderer = CLIRenderer()
        output = renderer.render({}, sample_run_data)
        assert "TRAINING RUN ANALYSIS" in output

    def test_full_render_includes_sections(self, sample_run_data: RunData) -> None:
        from alpha_zero.analysis.engines.loss_analysis import LossAnalyzer
        from alpha_zero.analysis.engines.model_progression import ModelProgressionAnalyzer
        from alpha_zero.analysis.engines.cost_analysis import CostAnalyzer
        from alpha_zero.analysis.engines.diagnostics import DiagnosticsAnalyzer
        from alpha_zero.analysis.engines.selfplay_quality import SelfplayQualityAnalyzer

        results = {
            "loss_analysis": LossAnalyzer().analyze(sample_run_data),
            "model_progression": ModelProgressionAnalyzer().analyze(sample_run_data),
            "cost_analysis": CostAnalyzer().analyze(sample_run_data),
            "diagnostics": DiagnosticsAnalyzer().analyze(sample_run_data),
            "selfplay_quality": SelfplayQualityAnalyzer().analyze(sample_run_data),
        }

        renderer = CLIRenderer()
        output = renderer.render(results, sample_run_data)

        assert "LOSS ANALYSIS" in output
        assert "MODEL PROGRESSION" in output
        assert "COST ANALYSIS" in output
        assert "DIAGNOSTICS" in output
        assert "SELFPLAY QUALITY" in output

    def test_empty_results_still_renders_header(
        self, sample_run_data: RunData
    ) -> None:
        renderer = CLIRenderer()
        output = renderer.render({}, sample_run_data)
        assert "TRAINING RUN ANALYSIS" in output
        # Should not contain section titles for engines
        assert "LOSS ANALYSIS" not in output
