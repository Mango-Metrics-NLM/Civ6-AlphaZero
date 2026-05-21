"""Tests for the dict-aware ``_apply_override`` in OrchestratorConfig.

The orchestrator's env-var override mechanism used to only handle scalars
(bool/int/float/str). Dict fields like ``mlflow.tags`` would be clobbered
with a raw string. This test verifies the fix correctly parses the
``"k1=v1,k2=v2"`` form.
"""

from __future__ import annotations

import pytest

from alpha_zero.orchestrator.config import OrchestratorConfig


def test_mlflow_tags_env_override_parses_dict(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CIV6AZ_MLFLOW_TAGS", "env=prod,run=test")
    config = OrchestratorConfig()
    config.apply_env_overrides()
    assert config.mlflow.tags == {"env": "prod", "run": "test"}


def test_mlflow_tags_env_override_empty_string_clears(monkeypatch: pytest.MonkeyPatch):
    config = OrchestratorConfig()
    config.mlflow.tags = {"existing": "tag"}
    monkeypatch.setenv("CIV6AZ_MLFLOW_TAGS", "")
    config.apply_env_overrides()
    # Empty string parses to {} which overwrites the existing dict.
    assert config.mlflow.tags == {}


def test_mlflow_enabled_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CIV6AZ_MLFLOW_ENABLED", "true")
    config = OrchestratorConfig()
    config.apply_env_overrides()
    assert config.mlflow.enabled is True


def test_mlflow_tracking_uri_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CIV6AZ_MLFLOW_TRACKING_URI", "file:///tmp/civ6_mlflow")
    config = OrchestratorConfig()
    config.apply_env_overrides()
    assert config.mlflow.tracking_uri == "file:///tmp/civ6_mlflow"


def test_training_full_eval_interval_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CIV6AZ_TRAINING_FULL_EVAL_INTERVAL", "10")
    config = OrchestratorConfig()
    config.apply_env_overrides()
    assert config.training.full_eval_interval == 10


def test_other_scalar_overrides_still_work(monkeypatch: pytest.MonkeyPatch):
    """Verify the dict branch doesn't break scalar handling."""
    monkeypatch.setenv("CIV6AZ_TRAINING_BATCH_SIZE", "1024")
    monkeypatch.setenv("CIV6AZ_WANDB_ENABLED", "true")
    config = OrchestratorConfig()
    config.apply_env_overrides()
    assert config.training.batch_size == 1024
    assert config.wandb.enabled is True
