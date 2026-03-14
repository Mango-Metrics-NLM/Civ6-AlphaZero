from __future__ import annotations

from alpha_zero.analysis.data_model import RunData
from alpha_zero.analysis.run_analyzer import _should_run, build_parser, run_engines


class TestBuildParser:
    def test_build_parser_run_id(self) -> None:
        args = build_parser().parse_args(["--run-id", "abc123"])
        assert args.run_id == "abc123"
        assert args.group is None

    def test_build_parser_group(self) -> None:
        args = build_parser().parse_args(["--group", "grp"])
        assert args.group == "grp"
        assert args.run_id is None

    def test_engine_selection(self) -> None:
        args = build_parser().parse_args(
            ["--run-id", "x", "--engines", "loss", "cost"]
        )
        assert args.engines == ["loss", "cost"]


class TestShouldRun:
    def test_should_run_all(self) -> None:
        assert _should_run("loss", ["all"]) is True

    def test_should_run_specific(self) -> None:
        assert _should_run("loss", ["loss", "cost"]) is True

    def test_should_run_not_selected(self) -> None:
        assert _should_run("diagnostics", ["loss"]) is False


class TestRunEngines:
    def test_run_engines_with_sample_data(self, sample_run_data: RunData) -> None:
        results = run_engines(sample_run_data, ["all"])
        assert "loss_analysis" in results
        assert "model_progression" in results
        assert "cost_analysis" in results
        assert "diagnostics" in results
        assert "selfplay_quality" in results
