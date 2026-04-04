"""Comprehensive tests for alpha_zero.optimizations.config.

Covers PerformanceOptimizationsConfig including presets, YAML I/O,
environment overrides, validation, and summary generation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from alpha_zero.optimizations.auxiliary_targets import AuxiliaryTargetsConfig
from alpha_zero.optimizations.config import PerformanceOptimizationsConfig
from alpha_zero.optimizations.cyclic_lr import CyclicLRConfig
from alpha_zero.optimizations.playout_cap import PlayoutCapConfig
from alpha_zero.optimizations.policy_pruning import PolicyPruningConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_config():
    return PerformanceOptimizationsConfig()


# ---------------------------------------------------------------------------
# Construction & Defaults
# ---------------------------------------------------------------------------


class TestPerformanceOptimizationsConfigDefaults:
    def test_all_enabled_by_default(self, default_config):
        assert default_config.enable_playout_cap is True
        assert default_config.enable_policy_pruning is True
        assert default_config.enable_cyclic_lr is True
        assert default_config.enable_auxiliary_targets is True

    def test_default_log_interval(self, default_config):
        assert default_config.stats_log_interval == 1000

    def test_sub_configs_are_defaults(self, default_config):
        assert isinstance(default_config.playout_cap, PlayoutCapConfig)
        assert isinstance(default_config.policy_pruning, PolicyPruningConfig)
        assert isinstance(default_config.cyclic_lr, CyclicLRConfig)
        assert isinstance(default_config.auxiliary_targets, AuxiliaryTargetsConfig)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_stats_log_interval(self):
        with pytest.raises(ValueError, match="stats_log_interval"):
            PerformanceOptimizationsConfig(stats_log_interval=0)

    def test_negative_stats_log_interval(self):
        with pytest.raises(ValueError, match="stats_log_interval"):
            PerformanceOptimizationsConfig(stats_log_interval=-5)


# ---------------------------------------------------------------------------
# Environment Overrides
# ---------------------------------------------------------------------------


class TestEnvironmentOverrides:
    def test_playout_cap_enabled_override(self):
        with patch.dict("os.environ", {"CIV6AZ_OPT_PLAYOUT_CAP_ENABLED": "false"}):
            config = PerformanceOptimizationsConfig()
            assert config.enable_playout_cap is False

    def test_policy_pruning_enabled_override(self):
        with patch.dict("os.environ", {"CIV6AZ_OPT_POLICY_PRUNING_ENABLED": "false"}):
            config = PerformanceOptimizationsConfig()
            assert config.enable_policy_pruning is False

    def test_cyclic_lr_enabled_override(self):
        with patch.dict("os.environ", {"CIV6AZ_OPT_CYCLIC_LR_ENABLED": "false"}):
            config = PerformanceOptimizationsConfig()
            assert config.enable_cyclic_lr is False

    def test_auxiliary_targets_enabled_override(self):
        with patch.dict("os.environ", {"CIV6AZ_OPT_AUXILIARY_TARGETS_ENABLED": "false"}):
            config = PerformanceOptimizationsConfig()
            assert config.enable_auxiliary_targets is False

    def test_playout_min_max_override(self):
        env = {"CIV6AZ_OPT_PLAYOUT_MIN": "30", "CIV6AZ_OPT_PLAYOUT_MAX": "300"}
        with patch.dict("os.environ", env):
            config = PerformanceOptimizationsConfig()
            assert config.playout_cap.min_playouts == 30
            assert config.playout_cap.max_playouts == 300

    def test_cyclic_lr_type_override(self):
        with patch.dict("os.environ", {"CIV6AZ_OPT_CYCLIC_LR_TYPE": "triangular"}):
            config = PerformanceOptimizationsConfig()
            assert config.cyclic_lr.schedule_type == "triangular"

    def test_cyclic_lr_base_max_override(self):
        env = {"CIV6AZ_OPT_CYCLIC_LR_BASE": "0.0001", "CIV6AZ_OPT_CYCLIC_LR_MAX": "0.01"}
        with patch.dict("os.environ", env):
            config = PerformanceOptimizationsConfig()
            assert config.cyclic_lr.base_lr == pytest.approx(0.0001)
            assert config.cyclic_lr.max_lr == pytest.approx(0.01)

    def test_enabled_true_string(self):
        with patch.dict("os.environ", {"CIV6AZ_OPT_PLAYOUT_CAP_ENABLED": "true"}):
            config = PerformanceOptimizationsConfig()
            assert config.enable_playout_cap is True


# ---------------------------------------------------------------------------
# from_dict / to_dict
# ---------------------------------------------------------------------------


class TestDictSerialization:
    def test_from_dict_simple_fields(self):
        d = {
            "enable_playout_cap": False,
            "enable_policy_pruning": False,
            "log_optimization_stats": False,
            "stats_log_interval": 500,
        }
        config = PerformanceOptimizationsConfig.from_dict(d)
        assert config.enable_playout_cap is False
        assert config.enable_policy_pruning is False
        assert config.stats_log_interval == 500

    def test_from_dict_nested_configs(self):
        d = {
            "playout_cap": {"min_playouts": 30, "max_playouts": 200},
            "cyclic_lr": {"base_lr": 0.0001},
        }
        config = PerformanceOptimizationsConfig.from_dict(d)
        assert config.playout_cap.min_playouts == 30

    def test_from_dict_all_nested(self):
        d = {
            "playout_cap": {"min_playouts": 20},
            "policy_pruning": {"min_probability": 0.005},
            "cyclic_lr": {"base_lr": 1e-5},
            "auxiliary_targets": {"enable_attention_targets": True},
        }
        config = PerformanceOptimizationsConfig.from_dict(d)
        assert config.playout_cap.min_playouts == 20
        assert config.policy_pruning.min_probability == 0.005

    def test_to_dict(self, default_config):
        d = default_config.to_dict()
        assert "enable_playout_cap" in d
        assert "playout_cap" in d
        assert isinstance(d["playout_cap"], dict)

    def test_to_dict_from_dict_roundtrip(self, default_config):
        d = default_config.to_dict()
        config2 = PerformanceOptimizationsConfig.from_dict(d)
        d2 = config2.to_dict()
        assert d == d2


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------


class TestYamlIO:
    def test_from_yaml(self):
        data = {
            "enable_playout_cap": False,
            "stats_log_interval": 250,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(data, f)
            path = Path(f.name)

        config = PerformanceOptimizationsConfig.from_yaml(path)
        assert config.enable_playout_cap is False
        assert config.stats_log_interval == 250
        path.unlink()

    def test_to_yaml(self, default_config):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = Path(f.name)

        default_config.to_yaml(path)
        loaded = yaml.safe_load(open(path))
        assert loaded["enable_playout_cap"] is True
        assert "playout_cap" in loaded
        path.unlink()

    def test_yaml_roundtrip(self, default_config):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = Path(f.name)

        default_config.to_yaml(path)
        config2 = PerformanceOptimizationsConfig.from_yaml(path)
        assert default_config.to_dict() == config2.to_dict()
        path.unlink()


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresets:
    def test_fast_preset(self):
        config = PerformanceOptimizationsConfig.fast_preset()
        assert config.enable_playout_cap is True
        assert config.enable_auxiliary_targets is False
        assert config.log_optimization_stats is False
        assert config.playout_cap.min_playouts == 25

    def test_quality_preset(self):
        config = PerformanceOptimizationsConfig.quality_preset()
        assert config.enable_auxiliary_targets is True
        assert config.playout_cap.min_playouts == 100
        assert config.auxiliary_targets.enable_future_value is True
        assert config.stats_log_interval == 500

    def test_balanced_preset(self):
        config = PerformanceOptimizationsConfig.balanced_preset()
        assert config.enable_playout_cap is True
        assert config.enable_auxiliary_targets is True
        assert config.playout_cap.min_playouts == 50
        assert config.auxiliary_targets.enable_phase_prediction is False

    def test_minimal_preset(self):
        config = PerformanceOptimizationsConfig.minimal_preset()
        assert config.enable_playout_cap is False
        assert config.enable_policy_pruning is False
        assert config.enable_cyclic_lr is False
        assert config.enable_auxiliary_targets is False
        assert config.log_optimization_stats is False

    def test_presets_are_valid(self):
        """All presets should construct without errors."""
        for preset_fn in [
            PerformanceOptimizationsConfig.fast_preset,
            PerformanceOptimizationsConfig.quality_preset,
            PerformanceOptimizationsConfig.balanced_preset,
            PerformanceOptimizationsConfig.minimal_preset,
        ]:
            config = preset_fn()
            assert isinstance(config, PerformanceOptimizationsConfig)
            # Should be serializable
            d = config.to_dict()
            assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# get_enabled_optimizations
# ---------------------------------------------------------------------------


class TestGetEnabledOptimizations:
    def test_all_enabled(self, default_config):
        enabled = default_config.get_enabled_optimizations()
        assert "playout_cap" in enabled
        assert "policy_pruning" in enabled
        assert "cyclic_lr" in enabled
        assert "auxiliary_targets" in enabled
        assert len(enabled) == 4

    def test_none_enabled(self):
        config = PerformanceOptimizationsConfig.minimal_preset()
        enabled = config.get_enabled_optimizations()
        assert enabled == []

    def test_partial_enabled(self):
        config = PerformanceOptimizationsConfig(
            enable_playout_cap=True,
            enable_policy_pruning=False,
            enable_cyclic_lr=True,
            enable_auxiliary_targets=False,
        )
        enabled = config.get_enabled_optimizations()
        assert "playout_cap" in enabled
        assert "cyclic_lr" in enabled
        assert "policy_pruning" not in enabled
        assert "auxiliary_targets" not in enabled


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_summary_all_enabled(self, default_config):
        s = default_config.get_summary()
        assert "Performance Optimizations" in s
        assert "Playout Cap:" in s
        assert "Policy Pruning:" in s
        assert "Cyclic LR:" in s
        assert "Auxiliary Targets:" in s

    def test_summary_none_enabled(self):
        config = PerformanceOptimizationsConfig.minimal_preset()
        s = config.get_summary()
        assert "None" in s
        assert "Playout Cap:" not in s

    def test_summary_partial(self):
        config = PerformanceOptimizationsConfig(
            enable_playout_cap=True,
            enable_policy_pruning=False,
            enable_cyclic_lr=False,
            enable_auxiliary_targets=False,
        )
        s = config.get_summary()
        assert "Playout Cap:" in s
        assert "Policy Pruning:" not in s

    def test_summary_auxiliary_target_details(self):
        config = PerformanceOptimizationsConfig.quality_preset()
        s = config.get_summary()
        assert "attention" in s
        assert "opponent" in s


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_all_enabled(self, default_config):
        r = repr(default_config)
        assert "PerformanceOptimizationsConfig" in r
        assert "playout_cap" in r

    def test_repr_none_enabled(self):
        config = PerformanceOptimizationsConfig.minimal_preset()
        r = repr(config)
        assert "enabled=[]" in r
