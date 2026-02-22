"""Tests for search efficiency report generator.

Coverage: loading from JSON, metric computation, markdown generation,
and chart generation (mocked if matplotlib unavailable).
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.search_efficiency_report import (
    AlgorithmMetrics,
    generate_markdown_report,
    load_from_benchmark_json,
)

# ---------------------------------------------------------------------------
# AlgorithmMetrics unit tests
# ---------------------------------------------------------------------------


class TestAlgorithmMetrics:
    """Test AlgorithmMetrics dataclass computed properties."""

    def test_search_calls_per_game_alphazero(self) -> None:
        m = AlgorithmMetrics(name="AZ", total_env_clones=1000, num_games=10)
        assert m.search_calls_per_game == 100.0

    def test_search_calls_per_game_muzero(self) -> None:
        m = AlgorithmMetrics(name="MZ", total_dynamics_calls=500, num_games=10)
        assert m.search_calls_per_game == 50.0

    def test_primary_search_calls_alphazero(self) -> None:
        m = AlgorithmMetrics(name="AZ", total_env_clones=1000, total_dynamics_calls=0)
        assert m.primary_search_calls == 1000

    def test_primary_search_calls_muzero(self) -> None:
        m = AlgorithmMetrics(name="MZ", total_env_clones=0, total_dynamics_calls=500)
        assert m.primary_search_calls == 500

    def test_search_calls_per_game_zero_games(self) -> None:
        m = AlgorithmMetrics(name="AZ", total_env_clones=1000, num_games=0)
        assert m.search_calls_per_game == 1000.0


# ---------------------------------------------------------------------------
# JSON loading tests
# ---------------------------------------------------------------------------


class TestLoadFromBenchmarkJson:
    """Test loading metrics from benchmark JSON output."""

    def test_load_valid_json(self, tmp_path: Path) -> None:
        data = {
            "alphazero": {
                "num_games": 50,
                "win_rate": 0.6,
                "games_per_minute": 12.5,
                "avg_game_length": 25.0,
                "search_metrics": {
                    "total_env_clones": 5000,
                    "total_dynamics_calls": 0,
                    "total_network_evals": 5000,
                    "avg_search_time_ms": 45.0,
                },
            },
            "muzero": {
                "num_games": 50,
                "win_rate": 0.54,
                "games_per_minute": 15.0,
                "avg_game_length": 22.0,
                "search_metrics": {
                    "total_env_clones": 0,
                    "total_dynamics_calls": 5000,
                    "total_network_evals": 5000,
                    "avg_search_time_ms": 38.0,
                },
            },
        }
        json_path = tmp_path / "benchmark_report.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        az, mz = load_from_benchmark_json(str(json_path))

        assert az.name == "AlphaZero"
        assert az.total_env_clones == 5000
        assert az.total_dynamics_calls == 0
        assert az.num_games == 50
        assert az.win_rate == 0.6

        assert mz.name == "MuZero"
        assert mz.total_env_clones == 0
        assert mz.total_dynamics_calls == 5000
        assert mz.win_rate == 0.54

    def test_load_partial_json(self, tmp_path: Path) -> None:
        """Missing fields should default to 0."""
        data = {"alphazero": {}, "muzero": {}}
        json_path = tmp_path / "benchmark_report.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        az, mz = load_from_benchmark_json(str(json_path))
        assert az.total_env_clones == 0
        assert mz.total_dynamics_calls == 0


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


class TestGenerateMarkdownReport:
    """Test markdown report generation."""

    def test_report_contains_key_sections(self, tmp_path: Path) -> None:
        az = AlgorithmMetrics(
            name="AlphaZero",
            total_env_clones=10000,
            total_network_evals=10000,
            avg_search_time_ms=50.0,
            num_games=100,
            win_rate=0.55,
            games_per_minute=10.0,
            avg_game_length=25.0,
        )
        mz = AlgorithmMetrics(
            name="MuZero",
            total_dynamics_calls=10000,
            total_network_evals=10000,
            avg_search_time_ms=40.0,
            num_games=100,
            win_rate=0.52,
            games_per_minute=12.0,
            avg_game_length=23.0,
        )

        report_path = generate_markdown_report(az, mz, tmp_path, [])
        content = Path(report_path).read_text()

        assert "# Search Efficiency" in content
        assert "AlphaZero" in content
        assert "MuZero" in content
        assert "10,000" in content  # formatted numbers
        assert "Env Clones" in content or "env clones" in content
        assert "Dynamics" in content or "dynamics" in content

    def test_report_handles_zero_values(self, tmp_path: Path) -> None:
        az = AlgorithmMetrics(name="AlphaZero")
        mz = AlgorithmMetrics(name="MuZero")

        # Should not raise
        report_path = generate_markdown_report(az, mz, tmp_path, [])
        assert Path(report_path).exists()

    def test_report_includes_chart_references(self, tmp_path: Path) -> None:
        az = AlgorithmMetrics(name="AlphaZero", total_env_clones=100)
        mz = AlgorithmMetrics(name="MuZero", total_dynamics_calls=100)

        charts = ["search_calls.png", "throughput.png"]
        report_path = generate_markdown_report(az, mz, tmp_path, charts)
        content = Path(report_path).read_text()

        assert "search_calls.png" in content
        assert "throughput.png" in content
        assert "## Charts" in content
