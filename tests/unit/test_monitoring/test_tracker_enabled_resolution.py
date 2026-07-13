"""Tests for ``resolve_tracker_enabled`` — the single enablement toggle.

Precedence contract (highest wins):
    CIV6AZ_MLFLOW_ENABLED > MLFLOW_ENABLED > WANDB_ENABLED (legacy) > default

The legacy ``WANDB_ENABLED`` tier exists so pre-migration deployment configs
(e.g. ``configs/vertex_*_smoke.yaml``) keep working; it must emit a
``DeprecationWarning`` when — and only when — it is the deciding variable.
"""

from __future__ import annotations

import warnings

import pytest

from alpha_zero.monitoring.config import (
    LEGACY_ENABLED_ENV,
    MLflowConfig,
    resolve_tracker_enabled,
)

_ALL_VARS = ("CIV6AZ_MLFLOW_ENABLED", "MLFLOW_ENABLED", "WANDB_ENABLED")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)


def test_default_when_nothing_set():
    assert resolve_tracker_enabled() is False
    assert resolve_tracker_enabled(default=True) is True


@pytest.mark.parametrize(
    ("var", "value", "expected"),
    [
        ("CIV6AZ_MLFLOW_ENABLED", "true", True),
        ("CIV6AZ_MLFLOW_ENABLED", "0", False),
        ("MLFLOW_ENABLED", "yes", True),
        ("MLFLOW_ENABLED", "false", False),
        ("WANDB_ENABLED", "1", True),
        ("WANDB_ENABLED", "off", False),
    ],
)
def test_each_tier_decides_alone(monkeypatch, var, value, expected):
    monkeypatch.setenv(var, value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert resolve_tracker_enabled() is expected


def test_civ6az_beats_mlflow_and_legacy(monkeypatch):
    monkeypatch.setenv("CIV6AZ_MLFLOW_ENABLED", "false")
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    monkeypatch.setenv(LEGACY_ENABLED_ENV, "true")
    assert resolve_tracker_enabled() is False


def test_mlflow_beats_legacy(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    monkeypatch.setenv(LEGACY_ENABLED_ENV, "true")
    assert resolve_tracker_enabled() is False


def test_falsy_higher_tier_still_wins(monkeypatch):
    """A *set-but-false* higher tier must decide — not fall through."""
    monkeypatch.setenv("CIV6AZ_MLFLOW_ENABLED", "no")
    monkeypatch.setenv(LEGACY_ENABLED_ENV, "true")
    assert resolve_tracker_enabled(default=True) is False


def test_legacy_var_warns_only_when_deciding(monkeypatch):
    monkeypatch.setenv(LEGACY_ENABLED_ENV, "true")
    with pytest.warns(DeprecationWarning, match="WANDB_ENABLED is deprecated"):
        assert resolve_tracker_enabled() is True

    # Once a higher tier is set, the legacy var must NOT warn.
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert resolve_tracker_enabled() is True


def test_no_warning_when_legacy_unset():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert resolve_tracker_enabled() is False


def test_from_env_honors_legacy_wandb_enabled(monkeypatch):
    """MLflowConfig.from_env() routes through the resolver, so legacy
    deployment configs that only set WANDB_ENABLED still enable tracking."""
    monkeypatch.setenv(LEGACY_ENABLED_ENV, "true")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert MLflowConfig.from_env().enabled is True
