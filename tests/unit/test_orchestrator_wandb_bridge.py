"""Tests for the legacy ``WandbConfig`` → ``MLflowConfig`` bridge.

W&B is retired (PROJECT_CHARTER.md NG-5). The orchestrator's ``WandbConfig``
dataclass survives only so existing YAML ``wandb:`` sections, presets,
``--no-wandb`` and ``CIV6AZ_WANDB_*`` overrides keep parsing; at runtime it is
converted via ``to_mlflow_config()``. These tests pin that mapping and the
deprecated ``create_wandb_logger`` factory wrapper.
"""

from __future__ import annotations

import pytest

from alpha_zero.monitoring.config import MLflowConfig
from alpha_zero.orchestrator.config import OrchestratorConfig, WandbConfig
from alpha_zero.utils.factories import create_wandb_logger

# ---------------------------------------------------------------------------
# WandbConfig.to_mlflow_config mapping
# ---------------------------------------------------------------------------


def test_bridge_maps_project_to_experiment_name():
    cfg = WandbConfig(enabled=True, project="civ6-prod").to_mlflow_config()
    assert isinstance(cfg, MLflowConfig)
    assert cfg.experiment_name == "civ6-prod"
    assert cfg.enabled is True


def test_bridge_maps_list_tags_to_dict_form():
    """Regression: the old inline mapping did dict(list[str]) which mangles
    2-char tags into key/value pairs and raises on anything else."""
    cfg = WandbConfig(enabled=True, tags=["ab", "production", "x"]).to_mlflow_config()
    assert cfg.tags == {"ab": "true", "production": "true", "x": "true"}


def test_bridge_preserves_entity_as_tag():
    cfg = WandbConfig(enabled=True, entity="my-team").to_mlflow_config()
    assert cfg.tags["wandb.entity"] == "my-team"


def test_bridge_no_entity_no_tag():
    cfg = WandbConfig(enabled=True).to_mlflow_config()
    assert "wandb.entity" not in cfg.tags


def test_bridge_mode_disabled_forces_enabled_false():
    cfg = WandbConfig(enabled=True, mode="disabled").to_mlflow_config()
    assert cfg.enabled is False


def test_bridge_maps_log_model_to_log_models():
    assert WandbConfig(enabled=True, log_model=False).to_mlflow_config().log_models is False
    assert WandbConfig(enabled=True, log_model=True).to_mlflow_config().log_models is True


def test_bridge_disabled_stays_disabled():
    assert WandbConfig(enabled=False).to_mlflow_config().enabled is False


# ---------------------------------------------------------------------------
# Preset round-trip: the production preset's wandb section still resolves
# to an enabled MLflow config through the bridge.
# ---------------------------------------------------------------------------


def test_production_preset_wandb_bridges_to_enabled_mlflow():
    config = OrchestratorConfig.from_preset("production")
    bridged = config.wandb.to_mlflow_config()
    assert isinstance(bridged, MLflowConfig)
    assert bridged.enabled is config.wandb.enabled


# ---------------------------------------------------------------------------
# create_wandb_logger — deprecated delegating wrapper
# ---------------------------------------------------------------------------


def test_factory_wrapper_warns_and_returns_none_when_disabled():
    with pytest.warns(DeprecationWarning, match="create_wandb_logger is deprecated"):
        result = create_wandb_logger(WandbConfig(enabled=False))
    assert result is None


def test_factory_wrapper_accepts_legacy_config_with_list_tags():
    """The tags-mangling bug: this raised ValueError before the bridge."""
    with pytest.warns(DeprecationWarning):
        # enabled=True exercises the full conversion path; the tracker itself
        # may be None if the mlflow package is unavailable — the point is that
        # conversion no longer raises on list tags.
        create_wandb_logger(WandbConfig(enabled=True, tags=["production", "run-1"]))


def test_factory_wrapper_passes_through_mlflow_config():
    with pytest.warns(DeprecationWarning):
        result = create_wandb_logger(MLflowConfig(enabled=False))
    assert result is None
