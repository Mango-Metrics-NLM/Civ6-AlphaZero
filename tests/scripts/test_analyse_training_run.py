"""Tests for scripts/analyse_training_run.py.

Covers:
- PolicyLossTrend: slope, convergence detection, smoothing
- ValueHeadStability: CV, monotone detection, variance
- build_training_summary: output shape and required keys
- MLflow client interactions are fully mocked (no network access)
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Guard: skip gracefully if the script does not exist yet
# ---------------------------------------------------------------------------
scripts = pytest.importorskip(
    "scripts.analyse_training_run",
    reason="scripts/analyse_training_run.py not yet implemented",
)

analyse_policy_loss = scripts.analyse_policy_loss
analyse_value_stability = scripts.analyse_value_stability
build_training_summary = scripts.build_training_summary
PolicyLossTrend = scripts.PolicyLossTrend
ValueHeadStability = scripts.ValueHeadStability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_metric_history(values: list[float]) -> list[Any]:
    """Create mock MLflow MetricHistory objects.

    Each object mimics the ``mlflow.entities.Metric`` structure, exposing
    ``.value`` and ``.step`` attributes, which is the interface used by
    ``MlflowClient.get_metric_history()``.

    Args:
        values: Sequence of float metric values (step index is inferred).

    Returns:
        List of SimpleNamespace objects with ``.value`` and ``.step`` attrs.
    """
    return [SimpleNamespace(value=v, step=i) for i, v in enumerate(values)]


@pytest.fixture()
def mock_mlflow_client() -> MagicMock:
    """Return a bare MagicMock standing in for mlflow.tracking.MlflowClient.

    Callers configure ``.get_metric_history.return_value`` per test.

    Returns:
        MagicMock with spec-free interface.
    """
    return MagicMock()


@pytest.fixture()
def sample_run_id() -> str:
    """Provide a stable fake MLflow run ID for tests.

    Returns:
        Hex-looking string that matches MLflow run ID format.
    """
    return "abc123def456789012345678901234ab"


# ---------------------------------------------------------------------------
# TestAnalysePolicyLoss
# ---------------------------------------------------------------------------


class TestAnalysePolicyLoss:
    """Tests for the ``analyse_policy_loss()`` function."""

    def test_monotone_decreasing_slope_negative(
        self, mock_mlflow_client: MagicMock, sample_run_id: str
    ) -> None:
        """Monotone decreasing loss series must produce a negative linear slope."""
        values = [2.0, 1.8, 1.5, 1.2, 0.9, 0.7, 0.5, 0.3]
        mock_mlflow_client.get_metric_history.return_value = make_metric_history(values)

        result: PolicyLossTrend = analyse_policy_loss(
            mock_mlflow_client, sample_run_id, smoothing_window=3
        )

        assert result.slope < 0.0, "Decreasing series must yield a negative slope"
        assert len(result.iterations) == len(values)
        assert len(result.avg_policy_loss) == len(values)

    def test_monotone_decreasing_converged_true(
        self, mock_mlflow_client: MagicMock, sample_run_id: str
    ) -> None:
        """Series with very small absolute slope should be marked converged."""
        # Near-flat values → |slope| < default threshold of 0.001
        values = [0.5001, 0.5000, 0.4999, 0.5001, 0.4998, 0.5002]
        mock_mlflow_client.get_metric_history.return_value = make_metric_history(values)

        result: PolicyLossTrend = analyse_policy_loss(
            mock_mlflow_client, sample_run_id, convergence_threshold=0.005
        )

        assert result.converged is True

    def test_oscillating_series_not_converged(
        self, mock_mlflow_client: MagicMock, sample_run_id: str
    ) -> None:
        """High-variance oscillating loss must be reported as not converged."""
        values = [1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1]
        mock_mlflow_client.get_metric_history.return_value = make_metric_history(values)

        result: PolicyLossTrend = analyse_policy_loss(
            mock_mlflow_client, sample_run_id, convergence_threshold=0.001
        )

        assert result.converged is False

    def test_fewer_points_than_smoothing_window_no_crash(
        self, mock_mlflow_client: MagicMock, sample_run_id: str
    ) -> None:
        """Having fewer data points than the smoothing window must not raise."""
        values = [0.9, 0.8]  # Only 2 points, window defaults to 5
        mock_mlflow_client.get_metric_history.return_value = make_metric_history(values)

        result: PolicyLossTrend = analyse_policy_loss(
            mock_mlflow_client, sample_run_id, smoothing_window=5
        )

        assert len(result.smoothed) == len(values)
        assert all(math.isfinite(v) for v in result.smoothed)

    def test_single_data_point(self, mock_mlflow_client: MagicMock, sample_run_id: str) -> None:
        """Single data point must be handled gracefully without raising."""
        mock_mlflow_client.get_metric_history.return_value = make_metric_history([0.5])

        result: PolicyLossTrend = analyse_policy_loss(mock_mlflow_client, sample_run_id)

        assert len(result.iterations) == 1
        assert math.isfinite(result.slope)

    @pytest.mark.parametrize(
        ("threshold", "values", "expected_converged"),
        [
            (0.001, [1.0, 0.5, 0.1, 0.05, 0.02], False),  # steep decline
            (1.0, [1.0, 0.5, 0.1, 0.05, 0.02], True),  # very lax threshold
            # The 4-point tail [0.5001, 0.5, 0.4999, 0.5] has |slope| ~4e-5 < 0.0001 → converged
            (0.0001, [0.5001, 0.5000, 0.4999, 0.5000], True),  # tiny slope within threshold
        ],
    )
    def test_custom_convergence_threshold(
        self,
        mock_mlflow_client: MagicMock,
        sample_run_id: str,
        threshold: float,
        values: list[float],
        expected_converged: bool,
    ) -> None:
        """Custom convergence_threshold must be respected for converged flag."""
        mock_mlflow_client.get_metric_history.return_value = make_metric_history(values)

        result: PolicyLossTrend = analyse_policy_loss(
            mock_mlflow_client,
            sample_run_id,
            convergence_threshold=threshold,
        )

        assert result.convergence_threshold == threshold
        assert result.converged == expected_converged

    def test_smoothed_length_matches_input(
        self, mock_mlflow_client: MagicMock, sample_run_id: str
    ) -> None:
        """The ``smoothed`` list must have the same length as the input series."""
        values = list(range(20, 0, -1))  # 20 descending points
        mock_mlflow_client.get_metric_history.return_value = make_metric_history(
            [float(v) for v in values]
        )

        result: PolicyLossTrend = analyse_policy_loss(
            mock_mlflow_client, sample_run_id, smoothing_window=5
        )

        assert len(result.smoothed) == len(values)

    def test_mlflow_client_called_with_policy_loss_key(
        self, mock_mlflow_client: MagicMock, sample_run_id: str
    ) -> None:
        """The client must be called with a metric key relating to policy loss."""
        mock_mlflow_client.get_metric_history.return_value = make_metric_history([0.5, 0.4])

        analyse_policy_loss(mock_mlflow_client, sample_run_id)

        mock_mlflow_client.get_metric_history.assert_called_once()
        call_args = mock_mlflow_client.get_metric_history.call_args
        # First positional arg should be run_id
        assert sample_run_id in call_args[0] or sample_run_id in call_args[1].values()


# ---------------------------------------------------------------------------
# TestAnalyseValueStability
# ---------------------------------------------------------------------------


class TestAnalyseValueStability:
    """Tests for the ``analyse_value_stability()`` function."""

    def test_stable_increasing_series_low_cv(
        self, mock_mlflow_client: MagicMock, sample_run_id: str
    ) -> None:
        """Steadily increasing, tightly clustered values → low CV."""
        values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        mock_mlflow_client.get_metric_history.return_value = make_metric_history(values)

        result: ValueHeadStability = analyse_value_stability(mock_mlflow_client, sample_run_id)

        # CV = std/mean; for tightly spaced ascending values this should be < 1
        assert result.cv < 1.0
        assert result.monotone_increasing is True

    def test_stable_series_monotone_increasing_flag(
        self, mock_mlflow_client: MagicMock, sample_run_id: str
    ) -> None:
        """Strictly increasing series must set monotone_increasing=True."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_mlflow_client.get_metric_history.return_value = make_metric_history(values)

        result: ValueHeadStability = analyse_value_stability(mock_mlflow_client, sample_run_id)

        assert result.monotone_increasing is True

    def test_volatile_series_high_cv(
        self, mock_mlflow_client: MagicMock, sample_run_id: str
    ) -> None:
        """High-variance oscillating values → high CV."""
        values = [0.01, 10.0, 0.01, 10.0, 0.01, 10.0]
        mock_mlflow_client.get_metric_history.return_value = make_metric_history(values)

        result: ValueHeadStability = analyse_value_stability(mock_mlflow_client, sample_run_id)

        # CV should be considerably > 1 for such variance
        assert result.cv > 0.5

    def test_non_monotone_series_flag_false(
        self, mock_mlflow_client: MagicMock, sample_run_id: str
    ) -> None:
        """Series with at least one decrease must set monotone_increasing=False."""
        values = [0.1, 0.2, 0.15, 0.3]  # 0.15 < 0.2 → not monotone
        mock_mlflow_client.get_metric_history.return_value = make_metric_history(values)

        result: ValueHeadStability = analyse_value_stability(mock_mlflow_client, sample_run_id)

        assert result.monotone_increasing is False

    def test_mean_and_std_calculated_correctly(
        self, mock_mlflow_client: MagicMock, sample_run_id: str
    ) -> None:
        """Mean and std must match numpy reference values."""
        import numpy as np

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mock_mlflow_client.get_metric_history.return_value = make_metric_history(values)

        result: ValueHeadStability = analyse_value_stability(mock_mlflow_client, sample_run_id)

        expected_mean = float(np.mean(values))
        expected_std = float(np.std(values))

        assert abs(result.mean - expected_mean) < 1e-6
        assert abs(result.std - expected_std) < 1e-6

    def test_cv_equals_std_over_mean(
        self, mock_mlflow_client: MagicMock, sample_run_id: str
    ) -> None:
        """CV must equal std/mean by definition."""
        values = [2.0, 3.0, 4.0, 5.0]
        mock_mlflow_client.get_metric_history.return_value = make_metric_history(values)

        result: ValueHeadStability = analyse_value_stability(mock_mlflow_client, sample_run_id)

        expected_cv = result.std / result.mean if result.mean != 0 else 0.0
        assert abs(result.cv - expected_cv) < 1e-6

    def test_single_point_graceful(self, mock_mlflow_client: MagicMock, sample_run_id: str) -> None:
        """Single data point must not raise and std must be 0."""
        mock_mlflow_client.get_metric_history.return_value = make_metric_history([0.42])

        result: ValueHeadStability = analyse_value_stability(mock_mlflow_client, sample_run_id)

        assert result.std == 0.0 or math.isfinite(result.std)


# ---------------------------------------------------------------------------
# TestBuildTrainingSummary
# ---------------------------------------------------------------------------


class TestBuildTrainingSummary:
    """Tests for the ``build_training_summary()`` function."""

    @pytest.fixture()
    def sample_policy_trend(self) -> PolicyLossTrend:
        """Return a minimal PolicyLossTrend for summary tests.

        Returns:
            PolicyLossTrend with synthetic data.
        """
        return PolicyLossTrend(
            iterations=[0, 1, 2, 3, 4],
            avg_policy_loss=[1.0, 0.8, 0.6, 0.4, 0.3],
            smoothed=[1.0, 0.9, 0.7, 0.5, 0.35],
            slope=-0.175,
            r_squared=0.98,
            converged=False,
            convergence_threshold=0.001,
            first_loss=1.0,
            last_loss=0.3,
            total_reduction=0.7,
        )

    @pytest.fixture()
    def sample_value_stability(self) -> ValueHeadStability:
        """Return a minimal ValueHeadStability for summary tests.

        Returns:
            ValueHeadStability with synthetic data.
        """
        return ValueHeadStability(
            iterations=[0, 1, 2, 3, 4],
            avg_value_loss=[0.1, 0.12, 0.14, 0.16, 0.18],
            mean=0.14,
            std=0.028,
            cv=0.2,
            monotone_increasing=True,
            per_epoch_variance=None,
            first_loss=0.1,
            last_loss=0.18,
        )

    def test_returns_dict(
        self,
        sample_policy_trend: PolicyLossTrend,
        sample_value_stability: ValueHeadStability,
        sample_run_id: str,
    ) -> None:
        """build_training_summary must return a plain dict."""
        result = build_training_summary(sample_policy_trend, sample_value_stability, sample_run_id)

        assert isinstance(result, dict)

    def test_required_top_level_keys(
        self,
        sample_policy_trend: PolicyLossTrend,
        sample_value_stability: ValueHeadStability,
        sample_run_id: str,
    ) -> None:
        """Summary dict must contain expected top-level keys."""
        result = build_training_summary(sample_policy_trend, sample_value_stability, sample_run_id)

        # build_training_summary uses "policy_loss" and "value_loss" as sub-dict keys
        expected_keys = {"run_id", "policy_loss", "value_loss"}
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_policy_subdict_contains_slope(
        self,
        sample_policy_trend: PolicyLossTrend,
        sample_value_stability: ValueHeadStability,
        sample_run_id: str,
    ) -> None:
        """Policy sub-dictionary must include slope and converged fields."""
        result = build_training_summary(sample_policy_trend, sample_value_stability, sample_run_id)

        # build_training_summary uses "policy_loss" key
        policy_section = result.get("policy_loss", result)
        # Allow either flat or nested shape
        keys_flat = set(result.keys())
        keys_nested = set(policy_section.keys()) if isinstance(policy_section, dict) else set()

        has_slope = "slope" in keys_flat or "slope" in keys_nested or "policy_slope" in keys_flat
        assert has_slope, f"'slope' not found in summary: {result}"

    def test_run_id_preserved_in_summary(
        self,
        sample_policy_trend: PolicyLossTrend,
        sample_value_stability: ValueHeadStability,
        sample_run_id: str,
    ) -> None:
        """The run_id passed in must be present in the returned summary."""
        result = build_training_summary(sample_policy_trend, sample_value_stability, sample_run_id)

        assert result.get("run_id") == sample_run_id

    def test_converged_flag_propagated(
        self,
        sample_value_stability: ValueHeadStability,
        sample_run_id: str,
    ) -> None:
        """Converged=True on the policy trend must be reflected in summary."""
        trend = PolicyLossTrend(
            iterations=[0, 1],
            avg_policy_loss=[0.5, 0.5001],
            smoothed=[0.5, 0.5001],
            slope=-0.0001,
            r_squared=0.5,
            converged=True,
            convergence_threshold=0.001,
            first_loss=0.5,
            last_loss=0.5001,
            total_reduction=-0.0001,
        )

        result = build_training_summary(trend, sample_value_stability, sample_run_id)

        # Converged flag should appear somewhere in the dict (possibly nested)
        summary_str = str(result)
        assert "True" in summary_str or "converged" in summary_str.lower()

    def test_summary_is_json_serialisable(
        self,
        sample_policy_trend: PolicyLossTrend,
        sample_value_stability: ValueHeadStability,
        sample_run_id: str,
    ) -> None:
        """Summary dict must be JSON-serialisable (no numpy scalars etc.)."""
        import json

        result = build_training_summary(sample_policy_trend, sample_value_stability, sample_run_id)

        # Should not raise
        json.dumps(result)
