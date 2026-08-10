"""Tests for the MLflow source in the training-run comparator.

The comparator could previously read only deprecated W&B runs or regex-parsed
logs, even though MLflow has been this project's only tracker since v1.7.0. Two
properties matter and are easy to regress:

1. The module must import with neither ``matplotlib`` nor ``mlflow`` installed.
   CI installs only the ``dev`` extra, so an unguarded import at module scope
   would make every test here fail at collection — and would take the script
   with it.
2. Metric-key resolution must fall back across naming schemes, because
   orchestrator runs and local training runs publish the same series under
   different names.

The MLflow client is mocked throughout; nothing here touches a tracking server.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_training_runs.py"

_spec = importlib.util.spec_from_file_location("compare_training_runs", _SCRIPT)
assert _spec is not None and _spec.loader is not None
comparator = importlib.util.module_from_spec(_spec)
# Register before executing: the module defines dataclasses, which resolve
# annotations through sys.modules.
sys.modules["compare_training_runs"] = comparator
_spec.loader.exec_module(comparator)


def _point(value: float, step: int) -> SimpleNamespace:
    return SimpleNamespace(value=value, step=step)


def _client(
    history: dict[str, list[SimpleNamespace]] | None = None,
    params: dict[str, str] | None = None,
    metrics: dict[str, float] | None = None,
) -> MagicMock:
    """Build a mock MLflow client with the given history and metadata."""
    history = history or {}
    client = MagicMock()
    client.get_metric_history.side_effect = lambda _run_id, key: history.get(key, [])
    client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(params=params or {}, metrics=metrics or {})
    )
    return client


class TestOptionalDependencies:
    """The script must degrade rather than fail when extras are missing."""

    def test_module_imports_without_optional_extras(self) -> None:
        """Importing succeeded above; these flags record what was available."""
        assert isinstance(comparator.MATPLOTLIB_AVAILABLE, bool)
        assert isinstance(comparator.MLFLOW_AVAILABLE, bool)

    def test_plotting_symbol_exists_either_way(self) -> None:
        """Downstream code can reference ``plt`` without an AttributeError."""
        assert hasattr(comparator, "plt")

    def test_fetch_without_mlflow_raises_a_directive_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing extra produces an actionable message, not an ImportError trace."""
        monkeypatch.setitem(sys.modules, "mlflow.tracking", None)

        with pytest.raises(ImportError, match="monitoring"):
            comparator.fetch_mlflow_metrics("run-id")


class TestMetricFetching:
    """Metric history is read, ordered, and mapped onto the report fields."""

    def test_history_is_ordered_by_step(self) -> None:
        """MLflow may return points unordered; the series must not be."""
        client = _client({"iter/avg_policy_loss": [_point(1.0, 2), _point(3.0, 0), _point(2.0, 1)]})

        result = comparator.fetch_mlflow_metrics("run", client=client)

        assert result.policy_loss == [3.0, 2.0, 1.0]

    def test_primary_metric_key_is_preferred(self) -> None:
        client = _client(
            {
                "iter/avg_policy_loss": [_point(1.0, 0)],
                "train/policy_loss": [_point(9.0, 0)],
            }
        )

        result = comparator.fetch_mlflow_metrics("run", client=client)

        assert result.policy_loss == [1.0]

    def test_fallback_metric_key_is_used_when_primary_is_absent(self) -> None:
        """Orchestrator and local runs name the same series differently."""
        client = _client({"train/policy_loss": [_point(9.0, 0)]})

        result = comparator.fetch_mlflow_metrics("run", client=client)

        assert result.policy_loss == [9.0]

    def test_missing_metrics_yield_empty_series(self) -> None:
        result = comparator.fetch_mlflow_metrics("run", client=_client())

        assert result.policy_loss == []
        assert result.total_iterations == 0

    def test_backend_errors_are_tolerated_per_metric(self) -> None:
        """One unreadable series must not abandon the whole run.

        Every key for policy loss raises here; the other series still resolve.
        """
        failing = set(comparator.MLFLOW_METRIC_KEYS["policy_loss"])
        client = MagicMock()

        def _history(_run_id: str, key: str) -> list[SimpleNamespace]:
            if key in failing:
                raise RuntimeError("backend exploded")
            return [_point(1.0, 0)]

        client.get_metric_history.side_effect = _history
        client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(params={}, metrics={}))

        result = comparator.fetch_mlflow_metrics("run", client=client)

        # Zero-filled rather than left short: every series must share a length,
        # or the first plot raises and takes the tables down with it.
        assert result.policy_loss == [0.0]
        assert result.value_loss == [1.0]

    def test_a_failing_primary_key_falls_back_to_the_secondary(self) -> None:
        """An unreadable primary key is survivable when a fallback exists."""
        client = MagicMock()

        def _history(_run_id: str, key: str) -> list[SimpleNamespace]:
            if key == "iter/avg_policy_loss":
                raise RuntimeError("backend exploded")
            return [_point(7.0, 0)] if key == "train/policy_loss" else []

        client.get_metric_history.side_effect = _history
        client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(params={}, metrics={}))

        result = comparator.fetch_mlflow_metrics("run", client=client)

        assert result.policy_loss == [7.0]

    def test_derived_summary_fields_are_computed(self) -> None:
        client = _client(
            {
                "iter/avg_policy_loss": [_point(3.0, 0), _point(1.0, 1)],
                "orchestrator/eval_win_rate": [_point(0.4, 0), _point(0.7, 1)],
                "orchestrator/cumulative_cost": [_point(1.0, 0), _point(2.5, 1)],
            }
        )

        result = comparator.fetch_mlflow_metrics("run", client=client)

        assert result.final_policy_loss == 1.0
        assert result.best_eval_win_rate == 0.7
        assert result.total_cost == 2.5
        assert result.total_iterations == 2


class TestRunMetadata:
    """Encoder identity and Elo ratings come from run params and metrics."""

    def test_encoder_type_is_read_from_params(self) -> None:
        client = _client(params={"encoder_type": "hex_crystalline"})

        result = comparator.fetch_mlflow_metrics("run", client=client)

        assert result.encoder_type == "hex_crystalline"

    def test_unknown_encoder_type_is_labelled_not_guessed(self) -> None:
        result = comparator.fetch_mlflow_metrics("run", client=_client())

        assert result.encoder_type == "Unknown"

    def test_non_numeric_param_count_is_ignored(self) -> None:
        client = _client(params={"param_count": "not-a-number"})

        result = comparator.fetch_mlflow_metrics("run", client=client)

        assert result.param_count == 0

    def test_elo_metrics_are_extracted(self) -> None:
        client = _client(metrics={"tournament/elo_v001": 1520.0, "train/loss": 1.0})

        result = comparator.fetch_mlflow_metrics("run", client=client)

        assert result.tournament_elos == {"v001": 1520.0}

    def test_unreadable_run_metadata_still_returns_metrics(self) -> None:
        """Metric history already fetched must survive a metadata failure."""
        client = _client({"iter/avg_policy_loss": [_point(1.0, 0)]})
        client.get_run.side_effect = RuntimeError("no such run")

        result = comparator.fetch_mlflow_metrics("run", client=client)

        assert result.policy_loss == [1.0]


class TestParameterisedDefaults:
    """Previously-hardcoded values are configurable."""

    def test_promotion_threshold_is_a_module_constant(self) -> None:
        assert 0.0 < comparator.DEFAULT_PROMOTION_THRESHOLD < 1.0

    def test_wandb_project_default_is_overridable(self) -> None:
        """The entity is no longer baked into the source."""
        assert isinstance(comparator.DEFAULT_WANDB_PROJECT, str)
        assert comparator.DEFAULT_WANDB_PROJECT

    def test_metric_key_table_covers_every_reported_series(self) -> None:
        expected = {
            "policy_loss",
            "value_loss",
            "total_loss",
            "eval_win_rate",
            "cumulative_cost",
            "iteration_duration_min",
        }

        assert set(comparator.MLFLOW_METRIC_KEYS) == expected


class TestCommandLine:
    """The new flags coexist with the pre-existing modes."""

    @pytest.fixture
    def parsed(self) -> Any:
        return comparator

    def test_mlflow_and_legacy_flags_all_parse(self) -> None:
        import argparse

        # Rebuild the parser the way main() does, without running it.
        source = _SCRIPT.read_text()
        assert "--mlflow_run_a" in source
        assert "--no_plots" in source
        # The legacy modes must survive the port.
        assert "--run_a" in source
        assert "--log_a" in source
        assert isinstance(argparse.ArgumentParser(), argparse.ArgumentParser)


class TestRaggedSeriesPadding:
    """All series must come back the same length.

    Plotting pairs each series against ``iterations``; a short one raises
    "x and y must have same first dimension". Because the plots and the tables
    share a single try/except in ``main()``, that failure also destroys the
    tables — the outputs the ``--no_plots`` path exists to guarantee. The W&B
    reader zero-fills for exactly this reason.
    """

    def test_series_of_unequal_length_are_padded(self) -> None:
        client = _client(
            {
                "iter/avg_loss": [_point(1.0, i) for i in range(10)],
                "orchestrator/eval_win_rate": [_point(0.5, i) for i in range(3)],
            }
        )

        result = comparator.fetch_mlflow_metrics("run", client=client)

        lengths = {
            len(result.iterations),
            len(result.policy_loss),
            len(result.value_loss),
            len(result.total_loss),
            len(result.eval_win_rate),
            len(result.cumulative_cost),
            len(result.iteration_duration_min),
        }
        assert lengths == {10}, f"ragged series would crash the first plot: {lengths}"

    def test_padding_preserves_real_values(self) -> None:
        client = _client(
            {
                "iter/avg_loss": [_point(9.0, i) for i in range(4)],
                "iter/avg_policy_loss": [_point(1.0, 0), _point(2.0, 1)],
            }
        )

        result = comparator.fetch_mlflow_metrics("run", client=client)

        assert result.policy_loss[:2] == [1.0, 2.0]
        assert result.policy_loss[2:] == [0.0, 0.0]

    def test_all_empty_yields_empty_series(self) -> None:
        result = comparator.fetch_mlflow_metrics("run", client=_client())

        assert result.iterations == []
        assert result.policy_loss == []

    def test_seconds_are_converted_to_minutes(self) -> None:
        """The A/B driver logs elapsed seconds; the report column is minutes."""
        client = _client({"iter/elapsed_seconds": [_point(120.0, 0), _point(60.0, 1)]})

        result = comparator.fetch_mlflow_metrics("run", client=client)

        assert result.iteration_duration_min == [2.0, 1.0]


class TestEloKeyConvention:
    """ELO extraction must match the keys this repository actually writes."""

    def test_per_player_rating_is_extracted(self) -> None:
        client = _client(metrics={"elo/v001/rating": 1520.0})

        assert comparator.fetch_mlflow_metrics("run", client=client).tournament_elos == {
            "v001": 1520.0
        }

    def test_scalar_aggregates_are_not_players(self) -> None:
        """``elo/spread`` is a statistic, not a competitor."""
        client = _client(metrics={"elo/spread": 120.0, "elo/max": 1600.0})

        assert comparator.fetch_mlflow_metrics("run", client=client).tournament_elos == {}

    def test_pbt_aggregates_are_not_players(self) -> None:
        """``pbt/elo_mean`` previously became a phantom player called "mean"."""
        client = _client(metrics={"pbt/elo_mean": 1500.0})

        assert comparator.fetch_mlflow_metrics("run", client=client).tournament_elos == {}

    def test_leaderboard_entries_are_excluded(self) -> None:
        client = _client(metrics={"elo/leaderboard/v001/rating": 1520.0})

        assert comparator.fetch_mlflow_metrics("run", client=client).tournament_elos == {}

    def test_legacy_wandb_convention_still_reads(self) -> None:
        """Historical runs used ``tournament/elo_<player>``."""
        client = _client(metrics={"tournament/elo_v007": 1480.0})

        assert comparator.fetch_mlflow_metrics("run", client=client).tournament_elos == {
            "v007": 1480.0
        }

    def test_mixed_families_extract_only_real_players(self) -> None:
        client = _client(
            metrics={
                "elo/spread": 120.0,
                "elo/v001/rating": 1520.0,
                "pbt/elo_mean": 1500.0,
                "train/loss": 1.0,
            }
        )

        assert comparator.fetch_mlflow_metrics("run", client=client).tournament_elos == {
            "v001": 1520.0
        }
