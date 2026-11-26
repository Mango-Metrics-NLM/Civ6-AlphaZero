"""Backwards-compatibility and size-reduction tests for ``wandb_compat`` extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from alpha_zero import monitoring
from alpha_zero.monitoring import mlflow_logger, wandb_compat
from alpha_zero.monitoring.config import MLflowConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_wandb_compat_reexport_identity() -> None:
    """Both ``mlflow_logger`` and package ``__init__`` must expose the exact class."""
    assert mlflow_logger.WandbCompatAdapter is wandb_compat.WandbCompatAdapter
    assert monitoring.WandbCompatAdapter is wandb_compat.WandbCompatAdapter


def test_mlflow_logger_line_count_under_god_file_threshold() -> None:
    """Extracting ``WandbCompatAdapter`` keeps ``mlflow_logger.py`` under 1,000 lines."""
    lines = (
        (REPO_ROOT / "alpha_zero" / "monitoring" / "mlflow_logger.py")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) < 1000


def test_wandb_compat_alert_logs_to_mlflow_logger_channel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``WandbCompatAdapter.alert`` continues logging to ``alpha_zero.monitoring.mlflow_logger``."""
    adapter = wandb_compat.WandbCompatAdapter.from_config(MLflowConfig(enabled=False))
    with caplog.at_level(logging.WARNING, logger="alpha_zero.monitoring.mlflow_logger"):
        adapter.alert("TestTitle", "body")
    assert any("TestTitle" in rec.message for rec in caplog.records)


def test_mlflow_available_patch_controls_adapter_init() -> None:
    """Patching ``alpha_zero.monitoring.mlflow_logger.MLFLOW_AVAILABLE`` governs the adapter."""
    with patch("alpha_zero.monitoring.mlflow_logger.MLFLOW_AVAILABLE", False):
        adapter = wandb_compat.WandbCompatAdapter.from_config(MLflowConfig(enabled=True))
        assert adapter.is_initialized is False
