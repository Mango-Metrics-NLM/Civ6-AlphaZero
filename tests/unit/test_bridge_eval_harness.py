"""Unit tests for BridgeEvalHarness — W2.3a + hardening."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from alpha_zero.bridge.eval_harness import (
    BridgeEvalHarness,
    _DEFAULT_EXPERIMENT_NAME,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_harness(**kwargs) -> BridgeEvalHarness:
    """Return a harness with a mock MLflowLogger injected."""
    with patch("alpha_zero.bridge.eval_harness.MLflowLogger") as mock_cls:
        mock_cls.return_value = MagicMock()
        h = BridgeEvalHarness(**kwargs)
    return h


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_eval_harness_init_defaults() -> None:
    h = _make_harness()
    assert h.run_name is None
    assert h.enabled is True
    assert h.experiment_name == _DEFAULT_EXPERIMENT_NAME


def test_eval_harness_custom_run_name() -> None:
    h = _make_harness(run_name="my-run")
    assert h.run_name == "my-run"


def test_default_experiment_name_constant_is_string() -> None:
    assert isinstance(_DEFAULT_EXPERIMENT_NAME, str) and len(_DEFAULT_EXPERIMENT_NAME) > 0


def test_custom_experiment_name() -> None:
    h = _make_harness(experiment_name="custom-exp")
    assert h.experiment_name == "custom-exp"


def test_disabled_harness_is_noop() -> None:
    """When enabled=False all hooks are no-ops."""
    h = _make_harness(enabled=False)
    assert h.enabled is False

    h.on_start({"key": "value"})
    h.on_turn_end(1, {"value_estimate": 0.5})
    h.on_stop({"turns_played": 10})

    h.logger.init.assert_not_called()
    h.logger.log_metrics.assert_not_called()
    h.logger.finish.assert_not_called()


def test_on_start_calls_logger_init() -> None:
    h = _make_harness(run_name="test-run")
    h.on_start(config_dict={"lr": 0.001})
    h.logger.init.assert_called_once_with(config_dict={"lr": 0.001})


def test_on_start_no_config() -> None:
    h = _make_harness()
    h.on_start()
    h.logger.init.assert_called_once_with(config_dict=None)


def test_on_turn_end_logs_correct_metrics() -> None:
    h = _make_harness()
    h.on_turn_end(turn=5, diagnostics={"value_estimate": 0.7, "policy_kind": 1.0})
    h.logger.log_metrics.assert_called_once()
    metrics = h.logger.log_metrics.call_args[0][0]
    assert metrics["turn"] == 5.0
    assert metrics["value_estimate"] == 0.7
    assert metrics["policy_kind"] == 1.0


def test_on_turn_end_step_matches_turn() -> None:
    h = _make_harness()
    h.on_turn_end(turn=7, diagnostics={})
    _, call_kwargs = h.logger.log_metrics.call_args
    assert call_kwargs["step"] == 7


def test_on_stop_filters_non_numeric_keys() -> None:
    h = _make_harness()
    h.on_stop({"turns_played": 42, "last_error": "none", "ack_failures": 0})
    metrics = h.logger.log_metrics.call_args[0][0]
    assert "session/turns_played" in metrics
    assert "session/ack_failures" in metrics
    assert "session/last_error" not in metrics


def test_on_stop_no_numeric_stats_skips_log_metrics() -> None:
    h = _make_harness()
    h.on_stop({"label": "end"})
    h.logger.log_metrics.assert_not_called()
    h.logger.finish.assert_called_once()


def test_on_stop_always_calls_finish() -> None:
    h = _make_harness()
    h.on_stop({})
    h.logger.finish.assert_called_once()


def test_on_stop_bool_converted_to_float() -> None:
    """bool is a subclass of int — should be logged as 0.0 or 1.0."""
    h = _make_harness()
    h.on_stop({"has_winner": True})
    metrics = h.logger.log_metrics.call_args[0][0]
    assert metrics["session/has_winner"] == 1.0
