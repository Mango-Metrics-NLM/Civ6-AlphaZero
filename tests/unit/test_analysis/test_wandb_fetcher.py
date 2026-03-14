"""Tests for WandbRunFetcher."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alpha_zero.analysis.data_model import RunData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_run(
    history_df: pd.DataFrame,
    run_id: str = "abc123",
    group: str = "test-group",
    config: dict | None = None,
    summary: dict | None = None,
) -> MagicMock:
    """Create a mock wandb Run object."""
    run = MagicMock()
    run.id = run_id
    run.group = group
    run.config = config or {}
    run.summary = summary or {}
    run.history.return_value = history_df
    return run


def _make_history(
    n: int = 5,
    *,
    policy_loss: list[float] | None = None,
    value_loss: list[float] | None = None,
    total_loss: list[float] | None = None,
    eval_win_rate: list[float] | None = None,
    model_version: list[float] | None = None,
    extra_columns: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Build a history DataFrame with standard W&B column names."""
    data: dict[str, list] = {}
    if policy_loss is not None:
        data["train/policy_loss"] = policy_loss
    if value_loss is not None:
        data["train/value_loss"] = value_loss
    if total_loss is not None:
        data["train/loss"] = total_loss
    if eval_win_rate is not None:
        data["orchestrator/eval_win_rate"] = eval_win_rate
    if model_version is not None:
        data["orchestrator/model_version"] = model_version
    if extra_columns:
        data.update(extra_columns)
    # Ensure at least n rows even if no columns supplied
    if not data:
        data["_step"] = list(range(n))
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWandbRunFetcher:
    """Tests for WandbRunFetcher."""

    @patch("alpha_zero.analysis.fetchers.wandb_fetcher.WANDB_AVAILABLE", True)
    @patch("alpha_zero.analysis.fetchers.wandb_fetcher.wandb")
    def test_fetch_run_extracts_iteration_data(self, mock_wandb: MagicMock) -> None:
        """Mock a run with a 5-row DataFrame; verify RunData has 5 iterations."""
        from alpha_zero.analysis.fetchers.wandb_fetcher import WandbRunFetcher

        df = _make_history(
            n=5,
            policy_loss=[1.2, 1.0, 0.8, 0.6, 0.5],
            value_loss=[0.9, 0.8, 0.7, 0.65, 0.6],
            total_loss=[2.1, 1.8, 1.5, 1.25, 1.1],
            eval_win_rate=[0.4, 0.45, 0.5, 0.55, 0.6],
            model_version=[5, 5, 5, 5, 5],
        )
        mock_run = _make_mock_run(df)
        mock_wandb.Api.return_value.run.return_value = mock_run

        fetcher = WandbRunFetcher(project="test-proj")
        run_data = fetcher.fetch_run("abc123")

        assert isinstance(run_data, RunData)
        assert len(run_data.iterations) == 5
        assert run_data.iterations[0].policy_loss == pytest.approx(1.2)
        assert run_data.iterations[2].value_loss == pytest.approx(0.7)
        assert run_data.iterations[4].eval_win_rate == pytest.approx(0.6)
        assert run_data.iterations[0].total_loss == pytest.approx(2.1)
        assert run_data.identifier.wandb_run_id == "abc123"

    @patch("alpha_zero.analysis.fetchers.wandb_fetcher.WANDB_AVAILABLE", True)
    @patch("alpha_zero.analysis.fetchers.wandb_fetcher.wandb")
    def test_missing_columns_use_defaults(self, mock_wandb: MagicMock) -> None:
        """DataFrame missing train/policy_loss -> iterations have policy_loss=None."""
        from alpha_zero.analysis.fetchers.wandb_fetcher import WandbRunFetcher

        # Only include eval_win_rate, no loss columns
        df = _make_history(
            n=3,
            eval_win_rate=[0.5, 0.55, 0.6],
        )
        mock_run = _make_mock_run(df)
        mock_wandb.Api.return_value.run.return_value = mock_run

        fetcher = WandbRunFetcher(project="test-proj")
        run_data = fetcher.fetch_run("abc123")

        assert len(run_data.iterations) == 3
        for it in run_data.iterations:
            assert it.policy_loss is None
            assert it.value_loss is None
            assert it.total_loss is None
        assert run_data.iterations[1].eval_win_rate == pytest.approx(0.55)

    @patch("alpha_zero.analysis.fetchers.wandb_fetcher.WANDB_AVAILABLE", True)
    @patch("alpha_zero.analysis.fetchers.wandb_fetcher.wandb")
    def test_promotion_detection_from_model_versions(self, mock_wandb: MagicMock) -> None:
        """model_version [5, 5, 6, 6, 7] -> promoted at index 2 and 4."""
        from alpha_zero.analysis.fetchers.wandb_fetcher import WandbRunFetcher

        df = _make_history(
            n=5,
            model_version=[5, 5, 6, 6, 7],
        )
        mock_run = _make_mock_run(df)
        mock_wandb.Api.return_value.run.return_value = mock_run

        fetcher = WandbRunFetcher(project="test-proj")
        run_data = fetcher.fetch_run("abc123")

        assert run_data.iterations[0].promoted is False
        assert run_data.iterations[1].promoted is False
        assert run_data.iterations[2].promoted is True
        assert run_data.iterations[3].promoted is False
        assert run_data.iterations[4].promoted is True

        assert run_data.start_model_version == 5
        assert run_data.end_model_version == 7

    def test_wandb_unavailable_raises(self) -> None:
        """When WANDB_AVAILABLE=False, constructor raises ImportError."""
        with patch("alpha_zero.analysis.fetchers.wandb_fetcher.WANDB_AVAILABLE", False):
            from alpha_zero.analysis.fetchers.wandb_fetcher import WandbRunFetcher

            with pytest.raises(ImportError, match="wandb is not installed"):
                WandbRunFetcher(project="test-proj")

    @patch("alpha_zero.analysis.fetchers.wandb_fetcher.WANDB_AVAILABLE", True)
    @patch("alpha_zero.analysis.fetchers.wandb_fetcher.wandb")
    def test_column_mapping_configurable(self, mock_wandb: MagicMock) -> None:
        """Custom column_map={'policy_loss': 'custom/ploss'} reads from that column."""
        from alpha_zero.analysis.fetchers.wandb_fetcher import WandbRunFetcher

        df = pd.DataFrame({"custom/ploss": [0.9, 0.7, 0.5]})
        mock_run = _make_mock_run(df)
        mock_wandb.Api.return_value.run.return_value = mock_run

        fetcher = WandbRunFetcher(
            project="test-proj",
            column_map={"policy_loss": "custom/ploss"},
        )
        run_data = fetcher.fetch_run("abc123")

        assert run_data.iterations[0].policy_loss == pytest.approx(0.9)
        assert run_data.iterations[2].policy_loss == pytest.approx(0.5)

    @patch("alpha_zero.analysis.fetchers.wandb_fetcher.WANDB_AVAILABLE", True)
    @patch("alpha_zero.analysis.fetchers.wandb_fetcher.wandb")
    def test_fetch_group_returns_list(self, mock_wandb: MagicMock) -> None:
        """Mock api.runs() returning 2 runs; fetch_group() returns list of 2 RunData."""
        from alpha_zero.analysis.fetchers.wandb_fetcher import WandbRunFetcher

        df1 = _make_history(n=2, policy_loss=[1.0, 0.8])
        df2 = _make_history(n=3, policy_loss=[0.9, 0.7, 0.5])
        mock_run1 = _make_mock_run(df1, run_id="run1", group="grp")
        mock_run2 = _make_mock_run(df2, run_id="run2", group="grp")
        mock_wandb.Api.return_value.runs.return_value = [mock_run1, mock_run2]

        fetcher = WandbRunFetcher(project="test-proj")
        results = fetcher.fetch_group("grp")

        assert len(results) == 2
        assert all(isinstance(r, RunData) for r in results)
        assert len(results[0].iterations) == 2
        assert len(results[1].iterations) == 3

    @patch("alpha_zero.analysis.fetchers.wandb_fetcher.WANDB_AVAILABLE", True)
    @patch("alpha_zero.analysis.fetchers.wandb_fetcher.wandb")
    def test_nan_values_become_none(self, mock_wandb: MagicMock) -> None:
        """DataFrame with NaN values -> None in IterationData."""
        from alpha_zero.analysis.fetchers.wandb_fetcher import WandbRunFetcher

        df = pd.DataFrame({
            "train/policy_loss": [0.5, float("nan"), 0.3],
            "train/value_loss": [float("nan"), 0.4, float("nan")],
        })
        mock_run = _make_mock_run(df)
        mock_wandb.Api.return_value.run.return_value = mock_run

        fetcher = WandbRunFetcher(project="test-proj")
        run_data = fetcher.fetch_run("abc123")

        assert run_data.iterations[0].policy_loss == pytest.approx(0.5)
        assert run_data.iterations[1].policy_loss is None
        assert run_data.iterations[2].policy_loss == pytest.approx(0.3)

        assert run_data.iterations[0].value_loss is None
        assert run_data.iterations[1].value_loss == pytest.approx(0.4)
        assert run_data.iterations[2].value_loss is None

    @patch("alpha_zero.analysis.fetchers.wandb_fetcher.WANDB_AVAILABLE", True)
    @patch("alpha_zero.analysis.fetchers.wandb_fetcher.wandb")
    def test_empty_history_returns_empty(self, mock_wandb: MagicMock) -> None:
        """Empty DataFrame -> empty iterations list."""
        from alpha_zero.analysis.fetchers.wandb_fetcher import WandbRunFetcher

        df = pd.DataFrame()
        mock_run = _make_mock_run(df)
        mock_wandb.Api.return_value.run.return_value = mock_run

        fetcher = WandbRunFetcher(project="test-proj")
        run_data = fetcher.fetch_run("abc123")

        assert run_data.iterations == []
        assert run_data.start_model_version is None
        assert run_data.end_model_version is None
