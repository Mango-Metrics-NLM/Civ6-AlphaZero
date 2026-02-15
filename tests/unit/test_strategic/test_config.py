"""Unit tests for the Strategic Planning SDK configuration system.

Tests the hierarchical configuration classes including PlanningConfig,
StrategicNetworkConfig, ObjectiveConfig, ScenarioConfig,
ExplainabilityConfig, and the master StrategicConfig. Covers defaults,
from_dict construction, preset loading, validation, serialization
(to_dict / to_yaml / from_yaml), and environment variable overrides.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import yaml

from alpha_zero.strategic.config import (
    ExplainabilityConfig,
    ObjectiveConfig,
    PlanningConfig,
    ScenarioConfig,
    StrategicConfig,
    StrategicNetworkConfig,
)


# ---------------------------------------------------------------------------
# TestPlanningConfig
# ---------------------------------------------------------------------------


class TestPlanningConfig:
    """Tests for PlanningConfig dataclass."""

    def test_defaults(self) -> None:
        """Test that default PlanningConfig values match documented expectations."""
        cfg = PlanningConfig()

        assert cfg.num_simulations == 100
        assert cfg.c_puct == 1.5
        assert cfg.dirichlet_alpha == pytest.approx(0.3)
        assert cfg.dirichlet_epsilon == pytest.approx(0.25)
        assert cfg.temperature == pytest.approx(1.0)
        assert cfg.temperature_threshold == 30
        assert cfg.num_planning_iterations == 10
        assert cfg.strategies_per_iteration == 5

    def test_from_dict(self) -> None:
        """Test creating PlanningConfig from a dictionary with partial overrides."""
        data = {
            "num_simulations": 200,
            "c_puct": 2.5,
            "temperature": 0.5,
        }
        cfg = PlanningConfig.from_dict(data)

        assert cfg.num_simulations == 200
        assert cfg.c_puct == 2.5
        assert cfg.temperature == pytest.approx(0.5)
        # Unspecified fields keep defaults
        assert cfg.dirichlet_alpha == pytest.approx(0.3)
        assert cfg.num_planning_iterations == 10

    def test_from_dict_empty(self) -> None:
        """Test creating PlanningConfig from an empty dictionary yields defaults."""
        cfg = PlanningConfig.from_dict({})

        assert cfg.num_simulations == 100
        assert cfg.c_puct == 1.5


# ---------------------------------------------------------------------------
# TestStrategicNetworkConfig
# ---------------------------------------------------------------------------


class TestStrategicNetworkConfig:
    """Tests for StrategicNetworkConfig dataclass."""

    def test_defaults(self) -> None:
        """Test that default network config is for a vector-only, single-objective setup."""
        cfg = StrategicNetworkConfig()

        assert cfg.spatial_channels == 0
        assert cfg.spatial_size == 0
        assert cfg.vector_dim == 64
        assert cfg.num_objectives == 1
        assert cfg.num_res_blocks == 4
        assert cfg.num_filters == 64
        assert cfg.value_head_hidden == 128
        assert cfg.policy_head_hidden == 128
        assert cfg.dropout == pytest.approx(0.0)
        assert cfg.use_attention is False

    def test_from_dict(self) -> None:
        """Test creating StrategicNetworkConfig from a dictionary."""
        data = {
            "spatial_channels": 13,
            "spatial_size": 16,
            "vector_dim": 128,
            "num_objectives": 3,
            "num_res_blocks": 8,
            "use_attention": True,
        }
        cfg = StrategicNetworkConfig.from_dict(data)

        assert cfg.spatial_channels == 13
        assert cfg.spatial_size == 16
        assert cfg.vector_dim == 128
        assert cfg.num_objectives == 3
        assert cfg.num_res_blocks == 8
        assert cfg.use_attention is True
        # Unspecified keep defaults
        assert cfg.num_filters == 64
        assert cfg.dropout == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestObjectiveConfig
# ---------------------------------------------------------------------------


class TestObjectiveConfig:
    """Tests for ObjectiveConfig dataclass."""

    def test_defaults(self) -> None:
        """Test that ObjectiveConfig defaults to weighted_sum with no weights."""
        cfg = ObjectiveConfig()

        assert cfg.scalarization_method == "weighted_sum"
        assert cfg.objective_weights == []
        assert cfg.constraint_penalty == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# TestScenarioConfig
# ---------------------------------------------------------------------------


class TestScenarioConfig:
    """Tests for ScenarioConfig dataclass."""

    def test_defaults(self) -> None:
        """Test that ScenarioConfig defaults to disabled with 5 scenarios."""
        cfg = ScenarioConfig()

        assert cfg.enabled is False
        assert cfg.num_scenarios == 5
        assert cfg.robustness_weight == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# TestExplainabilityConfig
# ---------------------------------------------------------------------------


class TestExplainabilityConfig:
    """Tests for ExplainabilityConfig dataclass."""

    def test_defaults(self) -> None:
        """Test that ExplainabilityConfig defaults to tracing enabled, LLM explanations off."""
        cfg = ExplainabilityConfig()

        assert cfg.enabled is True
        assert cfg.record_alternatives is True
        assert cfg.max_alternatives == 5
        assert cfg.generate_explanations is False
        assert cfg.export_format == "json"


# ---------------------------------------------------------------------------
# TestStrategicConfig
# ---------------------------------------------------------------------------


class TestStrategicConfig:
    """Tests for the master StrategicConfig class."""

    def test_defaults(self) -> None:
        """Test that master config defaults include all sub-configs with their defaults."""
        cfg = StrategicConfig()

        assert cfg.mode == "plan"
        assert cfg.experiment_name == "strategic_planning_v1"
        assert cfg.random_seed is None
        assert cfg.checkpoint_dir == "checkpoints/strategic"
        assert cfg.log_dir == "logs/strategic"
        assert cfg.config_path is None

        # Sub-configs should be default instances
        assert isinstance(cfg.planning, PlanningConfig)
        assert isinstance(cfg.objective, ObjectiveConfig)
        assert isinstance(cfg.scenario, ScenarioConfig)
        assert isinstance(cfg.network, StrategicNetworkConfig)
        assert isinstance(cfg.explainability, ExplainabilityConfig)

        assert cfg.planning.num_simulations == 100
        assert cfg.network.num_objectives == 1

    def test_from_preset_fast(self) -> None:
        """Test that the 'fast' preset has minimal simulations and small network."""
        cfg = StrategicConfig.from_preset("fast")

        assert cfg.experiment_name == "fast_test"
        assert cfg.planning.num_simulations == 10
        assert cfg.planning.num_planning_iterations == 2
        assert cfg.planning.strategies_per_iteration == 2
        assert cfg.network.num_res_blocks == 2
        assert cfg.network.num_filters == 32
        assert cfg.scenario.enabled is False
        assert cfg.explainability.generate_explanations is False
        assert cfg.explainability.record_alternatives is False

    def test_from_preset_balanced(self) -> None:
        """Test that the 'balanced' preset has moderate settings."""
        cfg = StrategicConfig.from_preset("balanced")

        assert cfg.experiment_name == "balanced_planning"
        assert cfg.planning.num_simulations == 100
        assert cfg.planning.num_planning_iterations == 10
        assert cfg.network.num_res_blocks == 4
        assert cfg.network.num_filters == 64
        assert cfg.scenario.enabled is True
        assert cfg.scenario.num_scenarios == 5
        assert cfg.explainability.enabled is True
        assert cfg.explainability.generate_explanations is False

    def test_from_preset_thorough(self) -> None:
        """Test that the 'thorough' preset has deep search and full analysis."""
        cfg = StrategicConfig.from_preset("thorough")

        assert cfg.experiment_name == "thorough_planning"
        assert cfg.planning.num_simulations == 800
        assert cfg.planning.c_puct == 2.0
        assert cfg.planning.num_planning_iterations == 50
        assert cfg.network.num_res_blocks == 8
        assert cfg.network.num_filters == 128
        assert cfg.network.use_attention is True
        assert cfg.scenario.enabled is True
        assert cfg.scenario.num_scenarios == 20
        assert cfg.scenario.robustness_weight == pytest.approx(0.5)
        assert cfg.explainability.generate_explanations is True
        assert cfg.logging.tensorboard is True
        assert cfg.logging.log_to_file is True

    def test_from_preset_invalid(self) -> None:
        """Test that an invalid preset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset.*nonexistent"):
            StrategicConfig.from_preset("nonexistent")

    def test_validate(self) -> None:
        """Test that validate returns warnings for problematic settings."""
        cfg = StrategicConfig()

        # Default config should produce no warnings
        warnings = cfg.validate()
        assert len(warnings) == 0, f"Unexpected warnings: {warnings}"

        # Set problematic values
        cfg.mode = "invalid_mode"
        cfg.planning.num_simulations = 0
        cfg.planning.c_puct = -1.0
        cfg.planning.dirichlet_epsilon = 2.0
        cfg.planning.dirichlet_alpha = -0.5
        cfg.planning.temperature = -1.0
        cfg.objective.scalarization_method = "bogus"
        cfg.objective.constraint_penalty = -5.0
        cfg.network.num_res_blocks = 0
        cfg.network.num_filters = 0
        cfg.network.dropout = 1.5
        cfg.explainability.export_format = "xml"
        cfg.explainability.max_alternatives = -1

        warnings = cfg.validate()

        # Should have multiple warnings
        assert len(warnings) >= 10, (
            f"Expected at least 10 warnings, got {len(warnings)}: {warnings}"
        )

        # Spot-check specific warnings
        warning_text = "\n".join(warnings)
        assert "Unknown mode" in warning_text
        assert "simulations" in warning_text.lower()
        assert "c_puct" in warning_text
        assert "dirichlet_epsilon" in warning_text
        assert "dirichlet_alpha" in warning_text
        assert "temperature" in warning_text
        assert "scalarization method" in warning_text or "scalarization_method" in warning_text.lower()
        assert "num_res_blocks" in warning_text
        assert "num_filters" in warning_text
        assert "dropout" in warning_text
        assert "export_format" in warning_text

    def test_validate_objective_weight_mismatch(self) -> None:
        """Test that validate warns when objective weights length != num_objectives."""
        cfg = StrategicConfig()
        cfg.network.num_objectives = 3
        cfg.objective.objective_weights = [0.5, 0.5]  # only 2, but 3 objectives

        warnings = cfg.validate()
        matching = [w for w in warnings if "objective_weights" in w]
        assert len(matching) >= 1

    def test_to_dict(self) -> None:
        """Test that to_dict produces a complete dictionary representation."""
        cfg = StrategicConfig()
        cfg.mode = "train"
        cfg.planning.num_simulations = 50

        data = cfg.to_dict()

        assert isinstance(data, dict)
        assert data["mode"] == "train"
        assert data["planning"]["num_simulations"] == 50
        assert "objective" in data
        assert "scenario" in data
        assert "network" in data
        assert "explainability" in data
        assert "logging" in data

        # Check that all expected keys are present
        assert data["network"]["vector_dim"] == 64
        assert data["explainability"]["export_format"] == "json"

    def test_apply_env_overrides(self) -> None:
        """Test that environment variables with STRATEGIC_SDK_ prefix override config."""
        env_vars = {
            "STRATEGIC_SDK_PLANNING_NUM_SIMULATIONS": "999",
            "STRATEGIC_SDK_NETWORK_NUM_RES_BLOCKS": "12",
            "STRATEGIC_SDK_SCENARIO_ENABLED": "true",
            "STRATEGIC_SDK_EXPLAINABILITY_MAX_ALTERNATIVES": "20",
            "STRATEGIC_SDK_LOGGING_LEVEL": "DEBUG",
            "UNRELATED_VAR": "should_be_ignored",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            cfg = StrategicConfig()
            result = cfg.apply_env_overrides()

        # Should return self for chaining
        assert result is cfg

        assert cfg.planning.num_simulations == 999
        assert cfg.network.num_res_blocks == 12
        assert cfg.scenario.enabled is True
        assert cfg.explainability.max_alternatives == 20
        assert cfg.logging.level == "DEBUG"

    def test_apply_env_overrides_ignores_unknown_section(self) -> None:
        """Test that env overrides with unknown sections are silently ignored."""
        env_vars = {
            "STRATEGIC_SDK_NONEXISTENT_FOO": "bar",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            cfg = StrategicConfig()
            cfg.apply_env_overrides()

        # Should not raise; config should remain at defaults
        assert cfg.planning.num_simulations == 100

    def test_apply_env_overrides_ignores_unknown_attr(self) -> None:
        """Test that env overrides with unknown attributes are silently ignored."""
        env_vars = {
            "STRATEGIC_SDK_PLANNING_NONEXISTENT_ATTR": "123",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            cfg = StrategicConfig()
            cfg.apply_env_overrides()

        assert cfg.planning.num_simulations == 100

    def test_from_yaml_and_to_yaml(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test round-trip YAML serialization and deserialization."""
        original = StrategicConfig()
        original.mode = "benchmark"
        original.experiment_name = "yaml_test"
        original.planning.num_simulations = 300
        original.network.num_objectives = 5
        original.scenario.enabled = True
        original.scenario.num_scenarios = 15
        original.explainability.generate_explanations = True

        yaml_path = str(tmp_path / "test_config.yaml")
        original.to_yaml(yaml_path)

        # Verify file was created
        assert os.path.isfile(yaml_path)

        # Load and compare
        loaded = StrategicConfig.from_yaml(yaml_path)

        assert loaded.mode == "benchmark"
        assert loaded.experiment_name == "yaml_test"
        assert loaded.planning.num_simulations == 300
        assert loaded.network.num_objectives == 5
        assert loaded.scenario.enabled is True
        assert loaded.scenario.num_scenarios == 15
        assert loaded.explainability.generate_explanations is True
        assert loaded.config_path == yaml_path

    def test_from_yaml_file_not_found(self) -> None:
        """Test that from_yaml raises FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            StrategicConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_to_yaml_creates_parent_dirs(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test that to_yaml creates parent directories if they do not exist."""
        cfg = StrategicConfig()
        nested_path = str(tmp_path / "deep" / "nested" / "config.yaml")
        cfg.to_yaml(nested_path)

        assert os.path.isfile(nested_path)

        # Verify it can be read back
        with open(nested_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["mode"] == "plan"

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """Test that to_dict followed by _from_dict preserves all values."""
        original = StrategicConfig()
        original.mode = "evaluate"
        original.planning.num_simulations = 42
        original.network.use_attention = True
        original.objective.scalarization_method = "chebyshev"

        data = original.to_dict()
        restored = StrategicConfig._from_dict(data)

        assert restored.mode == "evaluate"
        assert restored.planning.num_simulations == 42
        assert restored.network.use_attention is True
        assert restored.objective.scalarization_method == "chebyshev"

    def test_apply_env_overrides_float(self) -> None:
        """Test that float environment overrides are correctly parsed."""
        env_vars = {
            "STRATEGIC_SDK_PLANNING_C_PUCT": "3.14",
            "STRATEGIC_SDK_NETWORK_DROPOUT": "0.5",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            cfg = StrategicConfig()
            cfg.apply_env_overrides()

        assert cfg.planning.c_puct == pytest.approx(3.14)
        assert cfg.network.dropout == pytest.approx(0.5)

    def test_apply_env_overrides_bool_variants(self) -> None:
        """Test that boolean environment overrides accept 'true', '1', 'yes' variants."""
        for true_val in ["true", "1", "yes", "True", "YES"]:
            env_vars = {"STRATEGIC_SDK_SCENARIO_ENABLED": true_val}
            with patch.dict(os.environ, env_vars, clear=False):
                cfg = StrategicConfig()
                cfg.apply_env_overrides()
            assert cfg.scenario.enabled is True, (
                f"Expected True for STRATEGIC_SDK_SCENARIO_ENABLED={true_val}"
            )

        for false_val in ["false", "0", "no"]:
            env_vars = {"STRATEGIC_SDK_SCENARIO_ENABLED": false_val}
            with patch.dict(os.environ, env_vars, clear=False):
                cfg = StrategicConfig()
                cfg.apply_env_overrides()
            assert cfg.scenario.enabled is False, (
                f"Expected False for STRATEGIC_SDK_SCENARIO_ENABLED={false_val}"
            )
