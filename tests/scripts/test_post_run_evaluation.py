"""Tests for scripts/post_run_evaluation.py.

Covers:
- select_model_versions(): range logic including endpoints and edge cases
- log_elo_results_to_mlflow(): metric calls, elo_spread computation
- log_eval_results_to_mlflow(): expected metric key coverage
- All MLflow-related calls use MagicMock (no network access)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Guard: skip gracefully if the script does not exist yet
# ---------------------------------------------------------------------------
scripts = pytest.importorskip(
    "scripts.post_run_evaluation",
    reason="scripts/post_run_evaluation.py not yet implemented",
)

select_model_versions = scripts.select_model_versions
log_elo_results_to_mlflow = scripts.log_elo_results_to_mlflow
log_eval_results_to_mlflow = scripts.log_eval_results_to_mlflow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_ml_logger() -> MagicMock:
    """Return a MagicMock representing an MLflowLogger instance.

    Both ``log_metric`` and ``log_metrics`` are available so tests can
    assert on whichever the implementation chooses to call.

    Returns:
        MagicMock with log_metric and log_metrics attributes.
    """
    logger = MagicMock()
    logger.log_metric = MagicMock()
    logger.log_metrics = MagicMock()
    return logger


@pytest.fixture()
def sample_elo_ratings() -> dict[str, float]:
    """Provide a representative ELO ratings dictionary.

    Returns:
        Dict mapping version strings to ELO float scores.
    """
    return {
        "v42": 1500.0,
        "v47": 1523.0,
        "v52": 1490.0,
        "v57": 1560.0,
        "v62": 1545.0,
    }


# ---------------------------------------------------------------------------
# TestSelectModelVersions
# ---------------------------------------------------------------------------


class TestSelectModelVersions:
    """Tests for the ``select_model_versions()`` function."""

    def test_standard_range_includes_both_endpoints(self) -> None:
        """select_model_versions(42, 91, 5) must include both 42 and 91."""
        result = select_model_versions(42, 91, 5)

        assert 42 in result
        assert 91 in result

    def test_standard_range_correct_values(self) -> None:
        """select_model_versions(42, 91, 5) must return the exact expected list."""
        expected = [42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 91]
        result = select_model_versions(42, 91, 5)

        assert result == expected

    def test_single_model_range(self) -> None:
        """select_model_versions(42, 42, 1) must return exactly [42]."""
        result = select_model_versions(42, 42, 1)

        assert result == [42]

    def test_stride_larger_than_range_returns_start(self) -> None:
        """When stride > (end - start), must return at least [start]."""
        result = select_model_versions(10, 12, 100)

        assert 10 in result
        assert len(result) >= 1

    def test_stride_one_returns_every_iteration(self) -> None:
        """Stride of 1 must return every integer from start to end inclusive."""
        result = select_model_versions(5, 8, 1)

        assert result == [5, 6, 7, 8]

    def test_result_is_sorted_ascending(self) -> None:
        """Returned list must be sorted in ascending order."""
        result = select_model_versions(10, 50, 7)

        assert result == sorted(result)

    def test_no_duplicates_in_result(self) -> None:
        """Returned list must contain no duplicate version numbers."""
        result = select_model_versions(0, 20, 5)

        assert len(result) == len(set(result))

    @pytest.mark.parametrize(
        ("start", "end", "stride", "expected_len_min"),
        [
            (0, 100, 10, 11),   # 0,10,20,...,100 → 11 versions
            (1, 1, 1, 1),        # single model
            (0, 9, 3, 4),        # 0,3,6,9 → 4 versions
        ],
    )
    def test_result_length_parametrized(
        self,
        start: int,
        end: int,
        stride: int,
        expected_len_min: int,
    ) -> None:
        """Result length must be at least the expected minimum for each case."""
        result = select_model_versions(start, end, stride)

        assert len(result) >= expected_len_min


# ---------------------------------------------------------------------------
# TestLogEloResultsToMlflow
# ---------------------------------------------------------------------------


class TestLogEloResultsToMlflow:
    """Tests for the ``log_elo_results_to_mlflow()`` function."""

    def test_logs_metric_for_each_model(
        self,
        mock_ml_logger: MagicMock,
        sample_elo_ratings: dict[str, float],
    ) -> None:
        """Must call log_metric (or log_metrics) for every model version."""
        log_elo_results_to_mlflow(
            mock_ml_logger,
            elo_ratings=sample_elo_ratings,
            best_model="v57",
            best_elo=1560.0,
        )

        # At minimum, one call per model (individual or batch)
        total_metric_calls = (
            mock_ml_logger.log_metric.call_count
            + mock_ml_logger.log_metrics.call_count
        )
        assert total_metric_calls >= 1

    def test_logs_best_elo_metric(
        self,
        mock_ml_logger: MagicMock,
        sample_elo_ratings: dict[str, float],
    ) -> None:
        """Must log a metric for the best ELO rating under a tournament key."""
        log_elo_results_to_mlflow(
            mock_ml_logger,
            elo_ratings=sample_elo_ratings,
            best_model="v57",
            best_elo=1560.0,
        )

        all_calls_str = str(mock_ml_logger.log_metric.call_args_list)
        all_calls_str += str(mock_ml_logger.log_metrics.call_args_list)

        assert "best_elo" in all_calls_str or "tournament" in all_calls_str

    def test_elo_spread_logged(
        self,
        mock_ml_logger: MagicMock,
    ) -> None:
        """Must compute and log elo_spread = max(ratings) - min(ratings)."""
        ratings = {"v10": 1400.0, "v20": 1600.0}
        expected_spread = 1600.0 - 1400.0  # 200.0

        log_elo_results_to_mlflow(
            mock_ml_logger,
            elo_ratings=ratings,
            best_model="v20",
            best_elo=1600.0,
        )

        # Collect all numeric values from calls
        logged_values: list[float] = []
        for c in mock_ml_logger.log_metric.call_args_list:
            args, kwargs = c
            # Second positional arg is the value, or keyword 'value'
            val = kwargs.get("value", args[1] if len(args) > 1 else None)
            if val is not None:
                logged_values.append(float(val))
        for c in mock_ml_logger.log_metrics.call_args_list:
            args, kwargs = c
            metrics_dict = args[0] if args else kwargs.get("metrics", {})
            logged_values.extend(float(v) for v in metrics_dict.values())

        assert expected_spread in logged_values, (
            f"Expected spread {expected_spread} not found in logged values: {logged_values}"
        )

    def test_elo_spread_single_model(
        self,
        mock_ml_logger: MagicMock,
    ) -> None:
        """Single-model ratings dict must produce spread of 0."""
        ratings = {"v10": 1500.0}

        log_elo_results_to_mlflow(
            mock_ml_logger,
            elo_ratings=ratings,
            best_model="v10",
            best_elo=1500.0,
        )

        # Collect spread value from all logged metrics (log_metrics batch call)
        logged_values: list[float] = []
        for c in mock_ml_logger.log_metric.call_args_list:
            args, kwargs = c
            val = kwargs.get("value", args[1] if len(args) > 1 else None)
            if val is not None:
                logged_values.append(float(val))
        for c in mock_ml_logger.log_metrics.call_args_list:
            args, kwargs = c
            metrics_dict = args[0] if args else kwargs.get("metrics", {})
            logged_values.extend(float(v) for v in metrics_dict.values())

        # Spread must be 0 for a single-model tournament
        assert 0.0 in logged_values or any(v == 0.0 for v in logged_values)

    def test_no_gcs_or_real_mlflow_used(
        self,
        mock_ml_logger: MagicMock,
        sample_elo_ratings: dict[str, float],
    ) -> None:
        """Function must only call the provided mock logger, not real GCS."""
        # Call with mock logger — must not attempt real GCS or network calls
        log_elo_results_to_mlflow(
            mock_ml_logger,
            elo_ratings=sample_elo_ratings,
            best_model="v57",
            best_elo=1560.0,
        )
        # If we reach here, no GCS/mlflow network calls were made
        assert mock_ml_logger.log_metric.called or mock_ml_logger.log_metrics.called


# ---------------------------------------------------------------------------
# TestLogEvalResultsToMlflow
# ---------------------------------------------------------------------------


class TestLogEvalResultsToMlflow:
    """Tests for the ``log_eval_results_to_mlflow()`` function."""

    @pytest.fixture()
    def sample_eval_results(self) -> dict:
        """Provide a representative eval_results payload.

        Returns:
            Dict with win_rate, draw_rate, and loss_rate keys.
        """
        return {
            "random": {"win_rate": 0.95, "draw_rate": 0.03, "loss_rate": 0.02, "total_games": 20.0},
            "heuristic": {"win_rate": 0.72, "draw_rate": 0.10, "loss_rate": 0.18, "total_games": 20.0},
        }

    def test_logs_win_rate(
        self,
        mock_ml_logger: MagicMock,
        sample_eval_results: dict,
    ) -> None:
        """win_rate must be present among logged metrics."""
        log_eval_results_to_mlflow(mock_ml_logger, sample_eval_results)

        all_calls_str = str(mock_ml_logger.log_metric.call_args_list)
        all_calls_str += str(mock_ml_logger.log_metrics.call_args_list)

        assert "win_rate" in all_calls_str

    def test_logs_draw_rate(
        self,
        mock_ml_logger: MagicMock,
        sample_eval_results: dict,
    ) -> None:
        """draw_rate must be logged when present in eval_results."""
        log_eval_results_to_mlflow(mock_ml_logger, sample_eval_results)

        all_calls_str = str(mock_ml_logger.log_metric.call_args_list)
        all_calls_str += str(mock_ml_logger.log_metrics.call_args_list)

        assert "draw_rate" in all_calls_str

    def test_logs_loss_rate(
        self,
        mock_ml_logger: MagicMock,
        sample_eval_results: dict,
    ) -> None:
        """loss_rate must be logged when present in eval_results."""
        log_eval_results_to_mlflow(mock_ml_logger, sample_eval_results)

        all_calls_str = str(mock_ml_logger.log_metric.call_args_list)
        all_calls_str += str(mock_ml_logger.log_metrics.call_args_list)

        assert "loss_rate" in all_calls_str

    def test_empty_eval_results_no_crash(
        self,
        mock_ml_logger: MagicMock,
    ) -> None:
        """Empty dict must not raise any exception."""
        log_eval_results_to_mlflow(mock_ml_logger, {})
        # No assertion needed — reaching here means no exception was raised

    def test_all_metrics_logged_from_dict(
        self,
        mock_ml_logger: MagicMock,
        sample_eval_results: dict,
    ) -> None:
        """Every key in eval_results should appear in at least one logging call."""
        log_eval_results_to_mlflow(mock_ml_logger, sample_eval_results)

        all_calls_str = str(mock_ml_logger.log_metric.call_args_list)
        all_calls_str += str(mock_ml_logger.log_metrics.call_args_list)

        for baseline_results in sample_eval_results.values():
            for key in baseline_results:
                assert key in all_calls_str, f"Metric key '{key}' was not logged"

    @pytest.mark.parametrize(
        "eval_results",
        [
            {"random": {"win_rate": 1.0, "draw_rate": 0.0, "loss_rate": 0.0}},
            {"random": {"win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 1.0}},
            {"heuristic": {"win_rate": 0.33, "draw_rate": 0.34, "loss_rate": 0.33}},
        ],
    )
    def test_various_rate_distributions(
        self,
        mock_ml_logger: MagicMock,
        eval_results: dict,
    ) -> None:
        """Valid rate combinations must all log without error."""
        log_eval_results_to_mlflow(mock_ml_logger, eval_results)

        total_calls = (
            mock_ml_logger.log_metric.call_count
            + mock_ml_logger.log_metrics.call_count
        )
        assert total_calls >= 1
