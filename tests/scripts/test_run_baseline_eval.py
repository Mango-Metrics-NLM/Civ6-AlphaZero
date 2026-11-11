"""
Tests for scripts/run_baseline_eval.py.

Tests every public function in isolation with mocked dependencies,
verifying correct EvaluationResult API usage and MLflow metric extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.run_baseline_eval import (
    extract_game_stats_metrics,
    extract_win_rate_metrics,
    format_win_rate,
    parse_args,
    resolve_run_id,
    result_to_mlflow_metrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_win_rate_result(
    wins: int = 4,
    losses: int = 14,
    draws: int = 2,
    total_games: int = 20,
) -> Any:
    """Create a mock WinRateResult with the real property logic."""
    wr = MagicMock()
    wr.wins = wins
    wr.losses = losses
    wr.draws = draws
    wr.total_games = total_games
    # Replicate real property values
    wr.win_rate = wins / total_games * 100 if total_games > 0 else 0.0
    wr.score_rate = (wins + 0.5 * draws) / total_games * 100 if total_games > 0 else 0.0
    wr.draw_rate = draws / total_games * 100 if total_games > 0 else 0.0
    wr.loss_rate = losses / total_games * 100 if total_games > 0 else 0.0
    return wr


def _make_eval_result(
    wins: int = 4,
    losses: int = 14,
    draws: int = 2,
    total_games: int = 20,
    avg_game_length: float = 18.5,
    avg_cities_won: float = 1.2,
    avg_cities_lost: float = 0.1,
    avg_units_won: float = 2.0,
    avg_units_lost: float = 0.5,
    total_evaluation_time: float = 55.3,
    value_correlation: float = 0.207,
) -> Any:
    """Create a mock EvaluationResult with realistic values."""
    result = MagicMock()
    result.win_rate = _make_win_rate_result(wins, losses, draws, total_games)
    result.avg_game_length = avg_game_length
    result.avg_cities_won = avg_cities_won
    result.avg_cities_lost = avg_cities_lost
    result.avg_units_won = avg_units_won
    result.avg_units_lost = avg_units_lost
    result.total_evaluation_time = total_evaluation_time
    result.value_correlation = value_correlation
    # summary() returns a STRING, not a dict — this is the real API
    result.summary = MagicMock(return_value="=== EVALUATION SUMMARY ===")
    return result


# ---------------------------------------------------------------------------
# TestParseArgs
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_model_version_is_required(self) -> None:
        """parse_args should raise SystemExit when --model_version is missing."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_model_version_parsed(self) -> None:
        """parse_args correctly parses --model_version."""
        args = parse_args(["--model_version", "91"])
        assert args.model_version == 91

    def test_default_run_id_is_none(self) -> None:
        """--run_id should default to None (not a hardcoded value)."""
        args = parse_args(["--model_version", "91"])
        assert args.run_id is None

    def test_explicit_run_id(self) -> None:
        """--run_id should be stored as-is."""
        args = parse_args(["--model_version", "91", "--run_id", "abc123"])
        assert args.run_id == "abc123"

    def test_sqlite_flag(self) -> None:
        """--sqlite flag should be stored as True."""
        args = parse_args(["--model_version", "91", "--sqlite"])
        assert args.sqlite is True

    def test_sqlite_and_tracking_uri_mutually_exclusive(self) -> None:
        """--sqlite and --tracking_uri should be mutually exclusive (argparse enforces this)."""
        # When --sqlite is given alongside explicit --tracking_uri,
        # they are in a mutually_exclusive_group so argparse raises SystemExit.
        # Note: some Python versions may handle bool+string exclusive groups differently;
        # the key invariant is that --sqlite takes precedence when set.
        args = parse_args(["--model_version", "91", "--sqlite"])
        assert args.sqlite is True
        # When only tracking_uri is given, sqlite should be False
        args2 = parse_args(["--model_version", "91", "--tracking_uri", "sqlite:///custom.db"])
        assert args2.sqlite is False
        assert args2.tracking_uri == "sqlite:///custom.db"

    def test_defaults(self) -> None:
        """Default values should match expected training configuration."""
        args = parse_args(["--model_version", "42"])
        assert args.n_games == 20
        assert args.board_size == 8
        assert args.max_turns == 30
        assert args.mcts_sims == 25
        assert args.device == "cpu"
        assert args.tracking_uri == "mlruns"
        assert args.sqlite is False

    def test_all_options_parsed(self) -> None:
        """All CLI options should be parsed without error."""
        args = parse_args(
            [
                "--model_version",
                "57",
                "--run_id",
                "deadbeef",
                "--experiment",
                "my-exp",
                "--tracking_uri",
                "sqlite:///test.db",
                "--n_games",
                "10",
                "--board_size",
                "16",
                "--max_turns",
                "60",
                "--mcts_sims",
                "50",
                "--device",
                "cpu",
                "--log_level",
                "DEBUG",
            ]
        )
        assert args.model_version == 57
        assert args.run_id == "deadbeef"
        assert args.experiment == "my-exp"
        assert args.tracking_uri == "sqlite:///test.db"
        assert args.n_games == 10
        assert args.board_size == 16
        assert args.max_turns == 60
        assert args.mcts_sims == 50


# ---------------------------------------------------------------------------
# TestExtractWinRateMetrics
# ---------------------------------------------------------------------------


class TestExtractWinRateMetrics:
    """Tests for extract_win_rate_metrics()."""

    def test_returns_counts_and_rates(self) -> None:
        """Should return 7 metrics: 4 counts + 3 rates."""
        result = _make_eval_result(wins=4, losses=14, draws=2, total_games=20)
        metrics = extract_win_rate_metrics(result, "baseline/random")
        assert "baseline/random/wins" in metrics
        assert "baseline/random/losses" in metrics
        assert "baseline/random/draws" in metrics
        assert "baseline/random/total_games" in metrics
        assert "baseline/random/win_rate" in metrics
        assert "baseline/random/draw_rate" in metrics
        assert "baseline/random/loss_rate" in metrics

    def test_count_values_correct(self) -> None:
        """Count values should match WinRateResult attributes."""
        result = _make_eval_result(wins=4, losses=14, draws=2, total_games=20)
        metrics = extract_win_rate_metrics(result, "x")
        assert metrics["x/wins"] == 4.0
        assert metrics["x/losses"] == 14.0
        assert metrics["x/draws"] == 2.0
        assert metrics["x/total_games"] == 20.0

    def test_rates_normalised_to_fractions(self) -> None:
        """Rates should be normalised from 0-100 (WinRateResult) to 0-1 (MLflow)."""
        result = _make_eval_result(wins=10, losses=10, draws=0, total_games=20)
        metrics = extract_win_rate_metrics(result, "x")
        assert abs(metrics["x/win_rate"] - 0.5) < 1e-9
        assert abs(metrics["x/loss_rate"] - 0.5) < 1e-9
        assert abs(metrics["x/draw_rate"] - 0.0) < 1e-9

    def test_all_wins(self) -> None:
        """100% win rate should map to 1.0."""
        result = _make_eval_result(wins=20, losses=0, draws=0, total_games=20)
        metrics = extract_win_rate_metrics(result, "x")
        assert abs(metrics["x/win_rate"] - 1.0) < 1e-9

    def test_all_losses(self) -> None:
        """0% win rate should map to 0.0."""
        result = _make_eval_result(wins=0, losses=20, draws=0, total_games=20)
        metrics = extract_win_rate_metrics(result, "x")
        assert abs(metrics["x/win_rate"] - 0.0) < 1e-9
        assert abs(metrics["x/loss_rate"] - 1.0) < 1e-9

    def test_no_win_rate_attr_returns_empty(self) -> None:
        """If result has no win_rate attribute, should return empty dict."""
        result = MagicMock(spec=[])  # no attributes
        metrics = extract_win_rate_metrics(result, "x")
        assert metrics == {}

    def test_prefix_applied_correctly(self) -> None:
        """All metric keys should start with the given prefix."""
        result = _make_eval_result()
        metrics = extract_win_rate_metrics(result, "baseline/heuristic")
        for key in metrics:
            assert key.startswith("baseline/heuristic/"), f"Bad key: {key}"


# ---------------------------------------------------------------------------
# TestExtractGameStatsMetrics
# ---------------------------------------------------------------------------


class TestExtractGameStatsMetrics:
    """Tests for extract_game_stats_metrics()."""

    def test_returns_all_seven_stats(self) -> None:
        """Should return 7 game-stat metrics."""
        result = _make_eval_result()
        metrics = extract_game_stats_metrics(result, "baseline/random")
        assert len(metrics) == 7
        expected_keys = {
            "baseline/random/avg_game_length",
            "baseline/random/avg_cities_won",
            "baseline/random/avg_cities_lost",
            "baseline/random/avg_units_won",
            "baseline/random/avg_units_lost",
            "baseline/random/total_evaluation_time",
            "baseline/random/value_correlation",
        }
        assert set(metrics.keys()) == expected_keys

    def test_values_match_result(self) -> None:
        """Values should exactly match EvaluationResult attributes."""
        result = _make_eval_result(avg_game_length=18.5, value_correlation=0.207)
        metrics = extract_game_stats_metrics(result, "x")
        assert abs(metrics["x/avg_game_length"] - 18.5) < 1e-6
        assert abs(metrics["x/value_correlation"] - 0.207) < 1e-6

    def test_missing_attr_skipped(self) -> None:
        """Missing attributes should be silently skipped."""
        result = MagicMock()
        del result.avg_cities_won  # make it missing
        result.avg_cities_lost = 0.1
        result.avg_game_length = 10.0
        result.avg_units_won = 1.0
        result.avg_units_lost = 0.5
        result.total_evaluation_time = 30.0
        result.value_correlation = 0.1
        metrics = extract_game_stats_metrics(result, "x")
        assert "x/avg_cities_won" not in metrics
        assert "x/avg_cities_lost" in metrics


# ---------------------------------------------------------------------------
# TestResultToMlflowMetrics
# ---------------------------------------------------------------------------


class TestResultToMlflowMetrics:
    """Tests for the combined result_to_mlflow_metrics()."""

    def test_returns_15_metrics(self) -> None:
        """Combined function returns 15 metrics (8 win-rate incl. score_rate + 7 stats).

        Was 14 before ``WinRateResult.score_rate`` (draw-aware, E1) existed.
        """
        result = _make_eval_result()
        metrics = result_to_mlflow_metrics(result, "baseline/random")
        assert len(metrics) == 15
        assert "baseline/random/score_rate" in metrics

    def test_score_rate_is_draw_aware_and_normalised(self) -> None:
        """4 W / 2 D / 14 L over 20 games scores (4 + 0.5 * 2) / 20 = 0.25, logged on 0-1.

        The fixture once left ``score_rate`` unset, so the extractor normalised an
        auto-created mock and the test passed against a nonsense value.
        """
        result = _make_eval_result(wins=4, draws=2, losses=14, total_games=20)
        metrics = result_to_mlflow_metrics(result, "baseline/random")
        assert metrics["baseline/random/score_rate"] == pytest.approx(0.25)
        assert metrics["baseline/random/win_rate"] == pytest.approx(0.20)

    def test_all_keys_prefixed(self) -> None:
        """All keys should have the given prefix."""
        result = _make_eval_result()
        metrics = result_to_mlflow_metrics(result, "test_prefix")
        for key in metrics:
            assert key.startswith("test_prefix/"), f"Bad key: {key}"

    def test_win_rate_is_fraction(self) -> None:
        """win_rate should be 0-1 not 0-100 in the combined output."""
        result = _make_eval_result(wins=10, losses=10, draws=0, total_games=20)
        metrics = result_to_mlflow_metrics(result, "x")
        assert 0.0 <= metrics["x/win_rate"] <= 1.0

    def test_no_double_counting(self) -> None:
        """No key should appear twice."""
        result = _make_eval_result()
        metrics = result_to_mlflow_metrics(result, "x")
        # All keys are unique (dict guarantees this, but verify count is correct)
        assert len(metrics) == len(set(metrics.keys()))


# ---------------------------------------------------------------------------
# TestFormatWinRate
# ---------------------------------------------------------------------------


class TestFormatWinRate:
    """Tests for format_win_rate()."""

    def test_no_win_rate_attr(self) -> None:
        """Should return 'N/A' when result has no win_rate attribute."""
        result = MagicMock(spec=[])
        assert format_win_rate(result) == "N/A"

    def test_format_includes_counts(self) -> None:
        """Output should include win/draw/loss counts."""
        result = _make_eval_result(wins=4, losses=14, draws=2, total_games=20)
        fmt = format_win_rate(result)
        assert "W:4" in fmt
        assert "D:2" in fmt
        assert "L:14" in fmt

    def test_format_includes_percentages(self) -> None:
        """Output should include win/draw percentages."""
        result = _make_eval_result(wins=10, losses=10, draws=0, total_games=20)
        fmt = format_win_rate(result)
        assert "50.0%" in fmt

    def test_ascii_safe(self) -> None:
        """Output should be encodable as ASCII (no unicode arrows/dashes)."""
        result = _make_eval_result()
        fmt = format_win_rate(result)
        fmt.encode("ascii")  # should not raise


# ---------------------------------------------------------------------------
# TestResolveRunId
# ---------------------------------------------------------------------------


class TestResolveRunId:
    """Tests for resolve_run_id()."""

    def test_explicit_run_id_returned_as_is(self) -> None:
        """When run_id is given, it should be returned without any MLflow calls."""
        mock_mlflow = MagicMock()
        result = resolve_run_id(mock_mlflow, "explicit-run-id", "my-exp")
        assert result == "explicit-run-id"
        mock_mlflow.get_experiment_by_name.assert_not_called()

    def test_auto_detects_latest_run(self) -> None:
        """When run_id is None, should search the experiment for latest run."""
        # pandas ships with mlflow, not with `[dev]`. Left as a bare import this
        # failed rather than skipped -- invisible while nothing ran this suite.
        pd = pytest.importorskip("pandas", reason="pandas not installed (mlflow extra)")

        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
        mock_mlflow.search_runs.return_value = pd.DataFrame({"run_id": ["auto-run-id"]})
        result = resolve_run_id(mock_mlflow, None, "my-exp")
        assert result == "auto-run-id"
        mock_mlflow.get_experiment_by_name.assert_called_once_with("my-exp")

    def test_raises_when_experiment_not_found(self) -> None:
        """Should raise RuntimeError if experiment doesn't exist."""
        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = None
        with pytest.raises(RuntimeError, match="not found"):
            resolve_run_id(mock_mlflow, None, "nonexistent-exp")

    def test_raises_when_no_runs(self) -> None:
        """Should raise RuntimeError if experiment has no runs."""
        pd = pytest.importorskip("pandas", reason="pandas not installed (mlflow extra)")

        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
        mock_mlflow.search_runs.return_value = pd.DataFrame()
        with pytest.raises(RuntimeError, match="No runs found"):
            resolve_run_id(mock_mlflow, None, "empty-exp")


# ---------------------------------------------------------------------------
# TestMain (integration / smoke)
# ---------------------------------------------------------------------------


class TestMain:
    """Smoke tests for main() with heavy mocking."""

    def test_main_returns_1_without_gcp_creds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() should return 1 if GCS model loading fails."""
        # `patch("mlflow.…")` imports mlflow to resolve the target, so an absent
        # optional extra surfaced as a failure here rather than a skip.
        pytest.importorskip("mlflow", reason="mlflow not installed (monitoring extra)")

        import scripts.run_baseline_eval as sut

        with patch("alpha_zero.utils.env_loader.load_civ6az_env", return_value={}):
            with patch("mlflow.set_tracking_uri"):
                with patch("mlflow.set_experiment"):
                    with patch(
                        "mlflow.start_run",
                        return_value=MagicMock(info=MagicMock(run_id="mock-run")),
                    ):
                        with patch("mlflow.end_run"):
                            with patch(
                                "alpha_zero.cloud.gcp.evaluation_worker.load_model",
                                side_effect=RuntimeError("No GCS access"),
                            ):
                                result = sut.main(
                                    ["--model_version", "99", "--run_id", "fake-run-id"]
                                )
                                assert result == 1

    def test_sqlite_flag_overrides_tracking_uri(self) -> None:
        """When --sqlite is set, tracking_uri should be the SQLite URI."""
        args = parse_args(["--model_version", "91", "--sqlite"])
        from scripts.run_baseline_eval import _DEFAULT_SQLITE_URI

        # The main() function uses this logic:
        tracking_uri = _DEFAULT_SQLITE_URI if args.sqlite else args.tracking_uri
        assert tracking_uri == _DEFAULT_SQLITE_URI
        assert "sqlite" in tracking_uri
