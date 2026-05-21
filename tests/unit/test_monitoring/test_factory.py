"""Tests for ``alpha_zero.utils.factories.create_mlflow_logger``."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from alpha_zero.monitoring.config import MLflowConfig
from alpha_zero.utils.factories import create_mlflow_logger


def test_disabled_returns_none():
    cfg = MLflowConfig(enabled=False)
    assert create_mlflow_logger(cfg) is None


def test_enabled_returns_logger_when_available():
    cfg = MLflowConfig(
        enabled=True,
        tracking_uri="file:///tmp/x",
        experiment_name="exp",
    )
    with patch("alpha_zero.monitoring.MLFLOW_AVAILABLE", True):
        instance = create_mlflow_logger(cfg)
    assert instance is not None
    # Should have copied configuration over.
    assert instance.config.tracking_uri == "file:///tmp/x"
    assert instance.config.experiment_name == "exp"


def test_enabled_returns_none_when_mlflow_missing():
    cfg = MLflowConfig(enabled=True)
    with patch("alpha_zero.monitoring.MLFLOW_AVAILABLE", False):
        assert create_mlflow_logger(cfg) is None


def test_factory_experiment_name_override():
    cfg = MLflowConfig(enabled=True, experiment_name="default_name")
    with patch("alpha_zero.monitoring.MLFLOW_AVAILABLE", True):
        instance = create_mlflow_logger(cfg, experiment_name="overridden")
    assert instance is not None
    assert instance.config.experiment_name == "overridden"


def test_factory_handles_duck_typed_config():
    class DuckConfig:
        enabled = True
        tracking_uri = "file:///tmp/duck"
        experiment_name = "duck"
        run_name = None
        tags = {"src": "duck"}

    with patch("alpha_zero.monitoring.MLFLOW_AVAILABLE", True):
        instance = create_mlflow_logger(DuckConfig())
    assert instance is not None
    assert instance.config.tracking_uri == "file:///tmp/duck"


def test_factory_swallows_construction_errors(monkeypatch: pytest.MonkeyPatch):
    """If ``MLflowLogger`` construction itself raises, factory returns None."""
    cfg = MLflowConfig(enabled=True)

    class _Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("init blew up")

    monkeypatch.setattr("alpha_zero.monitoring.MLflowLogger", _Broken)
    monkeypatch.setattr("alpha_zero.monitoring.MLFLOW_AVAILABLE", True)
    assert create_mlflow_logger(cfg) is None
