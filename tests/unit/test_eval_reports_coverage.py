"""Tests for alpha_zero.evaluation.reports module.

Focuses on coverage of EvaluationReport and ReportGenerator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_zero.evaluation.config import (
    BaselineConfig,
    BaselineType,
    DiversityConfig,
    EvaluationConfig,
)
from alpha_zero.evaluation.diversity import DiversityMetrics
from alpha_zero.evaluation.evaluator import EvaluationResult
from alpha_zero.evaluation.reports import EvaluationReport, ReportGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_result(name: str = "random", games: int = 10, wins: int = 6) -> EvaluationResult:
    losses = games - wins
    return EvaluationResult(
        baseline_name=name,
        games_played=games,
        wins=wins,
        losses=losses,
        draws=0,
        avg_game_length=20.0,
    )


def _make_diversity_metrics() -> DiversityMetrics:
    return DiversityMetrics(
        unique_openings=5,
        opening_entropy=1.5,
        avg_game_length=25.0,
        game_length_std=5.0,
        game_length_min=10,
        game_length_max=40,
        victory_type_distribution={"win": 0.6, "loss": 0.3, "draw": 0.1},
        action_entropy_by_turn=[1.0, 1.1, 1.2],
        avg_action_entropy=1.1,
        total_games=10,
    )


def _make_report(
    with_diversity: bool = False,
    with_elo: bool = False,
    baselines: dict[str, EvaluationResult] | None = None,
) -> EvaluationReport:
    if baselines is None:
        baselines = {"random": _make_eval_result("random")}
    return EvaluationReport(
        experiment_name="test_eval",
        model_checkpoint="checkpoints/test.pt",
        timestamp="2026-01-01T00:00:00",
        baseline_results=baselines,
        diversity_metrics=_make_diversity_metrics() if with_diversity else None,
        overall_win_rate=0.6,
        overall_elo=1600.0 if with_elo else None,
        total_games=10,
        metadata={"key": "value"},
    )


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------


class TestEvaluationReport:
    def test_to_dict(self):
        report = _make_report(with_diversity=True, with_elo=True)
        d = report.to_dict()
        assert d["experiment_name"] == "test_eval"
        assert d["model_checkpoint"] == "checkpoints/test.pt"
        assert d["overall_win_rate"] == 0.6
        assert d["overall_elo"] == 1600.0
        assert d["total_games"] == 10
        assert "random" in d["baseline_results"]
        assert d["diversity_metrics"] is not None
        assert d["metadata"]["key"] == "value"

    def test_to_dict_no_diversity(self):
        report = _make_report(with_diversity=False)
        d = report.to_dict()
        assert d["diversity_metrics"] is None

    def test_to_json_and_from_json(self, tmp_path):
        report = _make_report(with_diversity=True, with_elo=True)
        json_path = tmp_path / "report.json"
        report.to_json(json_path)
        assert json_path.exists()

        loaded = EvaluationReport.from_json(json_path)
        assert loaded.experiment_name == "test_eval"
        assert loaded.overall_win_rate == 0.6
        assert loaded.overall_elo == 1600.0
        assert "random" in loaded.baseline_results
        assert loaded.diversity_metrics is not None
        assert loaded.diversity_metrics.unique_openings == 5

    def test_to_json_creates_parent_dirs(self, tmp_path):
        report = _make_report()
        json_path = tmp_path / "sub" / "dir" / "report.json"
        report.to_json(json_path)
        assert json_path.exists()

    def test_from_json_no_diversity(self, tmp_path):
        report = _make_report(with_diversity=False)
        json_path = tmp_path / "report.json"
        report.to_json(json_path)

        loaded = EvaluationReport.from_json(json_path)
        assert loaded.diversity_metrics is None

    def test_to_markdown_basic(self):
        report = _make_report()
        md = report.to_markdown()
        assert "# Model Evaluation Report" in md
        assert "test_eval" in md
        assert "random" in md
        assert "Overall Win Rate" in md

    def test_to_markdown_with_elo(self):
        report = _make_report(with_elo=True)
        md = report.to_markdown()
        assert "ELO Rating" in md
        assert "1600" in md

    def test_to_markdown_without_elo(self):
        report = _make_report(with_elo=False)
        md = report.to_markdown()
        assert "ELO Rating" not in md

    def test_to_markdown_with_diversity(self):
        report = _make_report(with_diversity=True)
        md = report.to_markdown()
        assert "Diversity Metrics" in md
        assert "Unique Openings" in md
        assert "Opening Entropy" in md
        assert "Victory Distribution" in md
        assert "win" in md

    def test_to_markdown_diversity_no_victory_distribution(self):
        report = _make_report(with_diversity=True)
        report.diversity_metrics.victory_type_distribution = {}
        md = report.to_markdown()
        assert "Diversity Metrics" in md
        assert "Victory Distribution" not in md

    def test_to_markdown_without_diversity(self):
        report = _make_report(with_diversity=False)
        md = report.to_markdown()
        assert "Diversity Metrics" not in md

    def test_to_markdown_multiple_baselines(self):
        baselines = {
            "random": _make_eval_result("random", games=10, wins=7),
            "heuristic": _make_eval_result("heuristic", games=10, wins=4),
        }
        report = _make_report(baselines=baselines)
        md = report.to_markdown()
        assert "random" in md
        assert "heuristic" in md

    def test_to_text_basic(self):
        report = _make_report()
        txt = report.to_text()
        assert "MODEL EVALUATION REPORT" in txt
        assert "test_eval" in txt
        assert "random" in txt

    def test_to_text_with_elo(self):
        report = _make_report(with_elo=True)
        txt = report.to_text()
        assert "ELO Rating" in txt

    def test_to_text_without_elo(self):
        report = _make_report(with_elo=False)
        txt = report.to_text()
        assert "ELO Rating" not in txt

    def test_to_text_with_diversity(self):
        report = _make_report(with_diversity=True)
        txt = report.to_text()
        assert "DIVERSITY METRICS" in txt
        assert "Unique Openings" in txt

    def test_to_text_without_diversity(self):
        report = _make_report(with_diversity=False)
        txt = report.to_text()
        assert "DIVERSITY METRICS" not in txt

    def test_to_text_multiple_baselines(self):
        baselines = {
            "random": _make_eval_result("random"),
            "heuristic": _make_eval_result("heuristic"),
        }
        report = _make_report(baselines=baselines)
        txt = report.to_text()
        assert "random" in txt
        assert "heuristic" in txt


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


class TestReportGenerator:
    @pytest.fixture()
    def config(self, tmp_path):
        return EvaluationConfig(
            experiment_name="gen_test",
            model_checkpoint_path="checkpoints/gen.pt",
            output_dir=str(tmp_path / "output"),
            baselines=[BaselineConfig(type=BaselineType.RANDOM, name="random", num_games=10)],
            diversity=DiversityConfig(enabled=False),
            elo_enabled=False,
            tensorboard_enabled=False,
        )

    @pytest.fixture()
    def generator(self, config):
        return ReportGenerator(config)

    def test_generate(self, generator):
        results = {"random": _make_eval_result("random", games=10, wins=6)}
        report = generator.generate(results)
        assert report.experiment_name == "gen_test"
        assert report.overall_win_rate == 0.6
        assert report.total_games == 10
        assert report.overall_elo is None

    def test_generate_with_diversity_and_elo(self, generator):
        results = {"random": _make_eval_result("random")}
        dm = _make_diversity_metrics()
        report = generator.generate(results, diversity_metrics=dm, model_elo=1550.0)
        assert report.diversity_metrics is not None
        assert report.overall_elo == 1550.0

    def test_generate_with_metadata(self, generator):
        results = {"random": _make_eval_result("random")}
        report = generator.generate(results, metadata={"version": "1.0"})
        assert report.metadata["version"] == "1.0"

    def test_generate_empty_results(self, generator):
        report = generator.generate({})
        assert report.overall_win_rate == 0.0
        assert report.total_games == 0

    def test_generate_and_save_json(self, generator, tmp_path):
        results = {"random": _make_eval_result("random")}
        report = generator.generate_and_save(
            results,
            output_dir=tmp_path / "save_test",
            formats=["json"],
        )
        json_files = list((tmp_path / "save_test").glob("*.json"))
        assert len(json_files) == 1

    def test_generate_and_save_markdown(self, generator, tmp_path):
        results = {"random": _make_eval_result("random")}
        report = generator.generate_and_save(
            results,
            output_dir=tmp_path / "save_test",
            formats=["markdown"],
        )
        md_files = list((tmp_path / "save_test").glob("*.md"))
        assert len(md_files) == 1

    def test_generate_and_save_text(self, generator, tmp_path):
        results = {"random": _make_eval_result("random")}
        report = generator.generate_and_save(
            results,
            output_dir=tmp_path / "save_test",
            formats=["text"],
        )
        txt_files = list((tmp_path / "save_test").glob("*.txt"))
        assert len(txt_files) == 1

    def test_generate_and_save_all_formats(self, generator, tmp_path):
        results = {"random": _make_eval_result("random")}
        dm = _make_diversity_metrics()
        report = generator.generate_and_save(
            results,
            output_dir=tmp_path / "all",
            diversity_metrics=dm,
            model_elo=1500.0,
            formats=["json", "markdown", "text"],
        )
        assert len(list((tmp_path / "all").glob("*.json"))) == 1
        assert len(list((tmp_path / "all").glob("*.md"))) == 1
        assert len(list((tmp_path / "all").glob("*.txt"))) == 1

    def test_generate_and_save_default_formats(self, generator, tmp_path):
        """Default formats are json and markdown."""
        results = {"random": _make_eval_result("random")}
        report = generator.generate_and_save(results, output_dir=tmp_path / "defaults")
        assert len(list((tmp_path / "defaults").glob("*.json"))) == 1
        assert len(list((tmp_path / "defaults").glob("*.md"))) == 1
        assert len(list((tmp_path / "defaults").glob("*.txt"))) == 0

    def test_generate_and_save_default_output_dir(self, generator, config):
        """When output_dir is None, uses config.output_dir."""
        results = {"random": _make_eval_result("random")}
        report = generator.generate_and_save(results, formats=["json"])
        json_files = list(Path(config.output_dir).glob("*.json"))
        assert len(json_files) == 1

    def test_compare_reports_basic(self, generator):
        r1 = _make_report(
            baselines={"random": _make_eval_result("random", games=10, wins=5)}
        )
        r1.overall_win_rate = 0.5
        r2 = _make_report(
            baselines={"random": _make_eval_result("random", games=10, wins=7)}
        )
        r2.overall_win_rate = 0.7

        comparison = generator.compare_reports(r1, r2)
        assert comparison["win_rate_diff"] == pytest.approx(0.2)
        assert comparison["games_diff"] == 0
        assert "random" in comparison["baseline_diffs"]

    def test_compare_reports_with_elo(self, generator):
        r1 = _make_report(with_elo=True)
        r1.overall_elo = 1500.0
        r2 = _make_report(with_elo=True)
        r2.overall_elo = 1600.0
        comparison = generator.compare_reports(r1, r2)
        assert comparison["elo_diff"] == pytest.approx(100.0)

    def test_compare_reports_without_elo(self, generator):
        r1 = _make_report(with_elo=False)
        r2 = _make_report(with_elo=False)
        comparison = generator.compare_reports(r1, r2)
        assert "elo_diff" not in comparison

    def test_compare_reports_with_diversity(self, generator):
        r1 = _make_report(with_diversity=True)
        r2 = _make_report(with_diversity=True)
        r2.diversity_metrics.opening_entropy = 2.0
        r2.diversity_metrics.unique_openings = 10
        r2.diversity_metrics.avg_game_length = 30.0
        comparison = generator.compare_reports(r1, r2)
        assert "diversity_diffs" in comparison
        assert comparison["diversity_diffs"]["opening_entropy_diff"] == pytest.approx(0.5)
        assert comparison["diversity_diffs"]["unique_openings_diff"] == 5
        assert comparison["diversity_diffs"]["avg_game_length_diff"] == pytest.approx(5.0)

    def test_compare_reports_without_diversity(self, generator):
        r1 = _make_report(with_diversity=False)
        r2 = _make_report(with_diversity=False)
        comparison = generator.compare_reports(r1, r2)
        assert "diversity_diffs" not in comparison

    def test_compare_reports_mixed_baselines(self, generator):
        """When reports have different baselines, only shared ones are compared."""
        r1 = _make_report(baselines={
            "random": _make_eval_result("random", wins=5),
            "heuristic": _make_eval_result("heuristic", wins=3),
        })
        r2 = _make_report(baselines={
            "random": _make_eval_result("random", wins=7),
            "mcts": _make_eval_result("mcts", wins=8),
        })
        comparison = generator.compare_reports(r1, r2)
        diffs = comparison["baseline_diffs"]
        # "random" is shared, so it gets a diff
        assert "random" in diffs
        # "heuristic" and "mcts" are not shared, so no diff for them
        assert "heuristic" not in diffs
        assert "mcts" not in diffs
