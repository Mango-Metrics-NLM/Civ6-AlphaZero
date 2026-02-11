"""Comprehensive tests for src.core.config module.

Tests cover:
1.  _deep_merge: nested dicts merge recursively, scalar override replaces,
    new keys added, original not mutated, both empty, list replacement,
    scalar/dict cross-replacement, deeply nested merge
2.  _coerce_value: bool (true/false/1/0/yes/no), int, float, str enum, plain str
3.  MCTSConfig: defaults, custom values, validation (num_simulations=0, c_puct=-1)
4.  NetworkConfig: defaults, custom values, validation (dropout_rate=1.0)
5.  TrainingConfig: defaults, custom values, batch_size/learning_rate validation
6.  CurriculumStage: required name/environment, defaults, custom values
7.  CurriculumConfig: empty stages default, stages list, defaults
8.  LoggingConfig: defaults, format_ field name, format alias, model_dump both modes
9.  SDKConfig: default construction, all sub-configs accessible, sub-config defaults
10. SDKConfig.from_yaml: valid YAML, missing file, invalid content, empty, partial
11. SDKConfig.from_yaml_with_overrides: merges two files, missing files, invalid merged
12. apply_env_overrides: SPSDK_TRAINING_BATCH_SIZE, SPSDK_SEED,
    SPSDK_MCTS_NUM_SIMULATIONS, multiple simultaneous, bool, float, enum,
    unrelated prefix ignored, custom prefix, invalid coercion graceful, returns new
13. load_config: no args, with path, with overrides, path+overrides, missing path,
    env vars applied last, invalid overrides, nested overrides
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.core.config import (
    CurriculumConfig,
    CurriculumStage,
    LoggingConfig,
    MCTSConfig,
    NetworkConfig,
    SDKConfig,
    TrainingConfig,
    _coerce_value,
    _deep_merge,
    load_config,
)
from src.core.exceptions import ConfigFileNotFoundError, ConfigValidationError
from src.core.types import BackboneType, DeviceType, SelectionPolicyType

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def base_yaml(tmp_path: Path) -> Path:
    """Create a base YAML config file with top-level SDKConfig keys."""
    data = {
        "seed": 42,
        "project_name": "base-project",
        "output_dir": "./base-outputs",
        "training": {
            "batch_size": 128,
            "learning_rate": 0.01,
        },
        "mcts": {
            "num_simulations": 100,
        },
    }
    p = tmp_path / "base.yaml"
    p.write_text(yaml.dump(data))
    return p


@pytest.fixture()
def override_yaml(tmp_path: Path) -> Path:
    """Create an override YAML config that changes a subset of fields."""
    data = {
        "seed": 99,
        "training": {
            "batch_size": 512,
        },
        "mcts": {
            "num_simulations": 400,
            "c_puct": 2.0,
        },
    }
    p = tmp_path / "override.yaml"
    p.write_text(yaml.dump(data))
    return p


@pytest.fixture()
def real_config_yaml() -> Path:
    """Return the path to the real configs/config.yaml shipped with the project."""
    p = Path("configs/config.yaml")
    if not p.exists():
        pytest.skip("configs/config.yaml not found in working directory")
    return p


@pytest.fixture(autouse=True)
def _clean_spsdk_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any pre-existing SPSDK_ environment variables to isolate tests."""
    for key in list(os.environ.keys()):
        if key.startswith("SPSDK_"):
            monkeypatch.delenv(key, raising=False)


# ============================================================================
# 1. Tests: _deep_merge
# ============================================================================


class TestDeepMerge:
    """Tests for the _deep_merge(base, override) helper."""

    def test_nested_override(self) -> None:
        """Override values in nested dicts are applied recursively."""
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": 10, "c": 2}, "d": 3}

    def test_scalar_override_replaces(self) -> None:
        """A scalar override replaces the scalar in base."""
        base = {"key": "old"}
        override = {"key": "new"}
        result = _deep_merge(base, override)
        assert result["key"] == "new"

    def test_new_keys_added(self) -> None:
        """Non-overlapping keys from override are added to the result."""
        base = {"x": 1}
        override = {"y": 2}
        result = _deep_merge(base, override)
        assert result == {"x": 1, "y": 2}

    def test_does_not_mutate_base(self) -> None:
        """The original base dict is not modified."""
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1

    def test_does_not_mutate_override(self) -> None:
        """The original override dict is not modified."""
        base = {"a": 1}
        override = {"a": {"nested": [1, 2]}}
        result = _deep_merge(base, override)
        # Mutating the result should not affect the override
        result["a"]["nested"].append(3)
        assert override["a"]["nested"] == [1, 2]

    def test_empty_override(self) -> None:
        """An empty override returns a copy of the base."""
        base = {"a": {"b": 1}, "c": 3}
        result = _deep_merge(base, {})
        assert result == base
        assert result is not base

    def test_empty_base(self) -> None:
        """An empty base returns a copy of the override."""
        override = {"a": 1, "b": {"c": 2}}
        result = _deep_merge({}, override)
        assert result == override
        assert result is not override

    def test_both_empty(self) -> None:
        """Merging two empty dicts yields an empty dict."""
        result = _deep_merge({}, {})
        assert result == {}

    def test_list_replacement(self) -> None:
        """Lists in override replace lists in base (no element-level merge)."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = _deep_merge(base, override)
        assert result == {"items": [4, 5]}

    def test_scalar_replaces_dict(self) -> None:
        """A scalar override replaces an entire nested dict."""
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": "flat_value"}
        result = _deep_merge(base, override)
        assert result == {"a": "flat_value"}

    def test_dict_replaces_scalar(self) -> None:
        """A dict override replaces a scalar in base."""
        base = {"a": "flat_value"}
        override = {"a": {"b": 1}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": 1}}

    def test_deeply_nested(self) -> None:
        """Deep nesting (3+ levels) merges correctly."""
        base = {"l1": {"l2": {"l3": {"val": 1, "keep": True}}}}
        override = {"l1": {"l2": {"l3": {"val": 99}}}}
        result = _deep_merge(base, override)
        assert result["l1"]["l2"]["l3"]["val"] == 99
        assert result["l1"]["l2"]["l3"]["keep"] is True

    def test_multiple_new_keys_at_different_levels(self) -> None:
        """New keys are added at every level of nesting."""
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}, "b": {"z": 3}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 2}, "b": {"z": 3}}


# ============================================================================
# 2. Tests: _coerce_value
# ============================================================================


class TestCoerceValue:
    """Tests for the _coerce_value(current, raw) helper."""

    # -- bool ---------------------------------------------------------------

    @pytest.mark.parametrize("raw", ["true", "True", "TRUE", "1", "yes", "Yes"])
    def test_bool_truthy(self, raw: str) -> None:
        """Truthy string variants are coerced to True."""
        assert _coerce_value(True, raw) is True
        assert _coerce_value(False, raw) is True

    @pytest.mark.parametrize("raw", ["false", "False", "0", "no", "No", "random"])
    def test_bool_falsy(self, raw: str) -> None:
        """Non-truthy strings are coerced to False."""
        assert _coerce_value(True, raw) is False
        assert _coerce_value(False, raw) is False

    # -- int ----------------------------------------------------------------

    def test_int_coercion(self) -> None:
        """Numeric strings are coerced to int when current is int."""
        assert _coerce_value(0, "42") == 42
        assert isinstance(_coerce_value(0, "42"), int)

    def test_int_negative(self) -> None:
        """Negative numeric strings are coerced correctly."""
        assert _coerce_value(0, "-7") == -7

    def test_int_large(self) -> None:
        """Large integers are coerced correctly."""
        assert _coerce_value(0, "1000000") == 1_000_000

    def test_int_zero(self) -> None:
        """Zero is coerced correctly for int type."""
        assert _coerce_value(5, "0") == 0
        assert isinstance(_coerce_value(5, "0"), int)

    # -- float --------------------------------------------------------------

    def test_float_coercion(self) -> None:
        """Numeric strings are coerced to float when current is float."""
        result = _coerce_value(0.0, "3.14")
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_float_scientific(self) -> None:
        """Scientific notation strings are coerced to float."""
        result = _coerce_value(0.0, "1e-3")
        assert result == pytest.approx(0.001)

    def test_float_negative(self) -> None:
        """Negative float strings are coerced correctly."""
        result = _coerce_value(0.0, "-2.5")
        assert result == pytest.approx(-2.5)

    # -- str Enum -----------------------------------------------------------

    def test_device_type_enum(self) -> None:
        """A DeviceType str enum is coerced from its string value."""
        result = _coerce_value(DeviceType.auto, "cpu")
        assert result == DeviceType.cpu
        assert isinstance(result, DeviceType)

    def test_selection_policy_enum(self) -> None:
        """A SelectionPolicyType str enum is coerced from its string value."""
        result = _coerce_value(SelectionPolicyType.puct, "gumbel")
        assert result == SelectionPolicyType.gumbel

    def test_backbone_type_enum(self) -> None:
        """A BackboneType str enum is coerced from its string value."""
        result = _coerce_value(BackboneType.resnet, "transformer")
        assert result == BackboneType.transformer

    # -- plain str ----------------------------------------------------------

    def test_plain_str_passthrough(self) -> None:
        """When current is a plain str, raw is returned unchanged."""
        result = _coerce_value("hello", "world")
        assert result == "world"
        assert isinstance(result, str)

    def test_plain_str_empty(self) -> None:
        """An empty string is returned as-is when current is str."""
        result = _coerce_value("default", "")
        assert result == ""


# ============================================================================
# 3. Tests: MCTSConfig
# ============================================================================


class TestMCTSConfig:
    """Tests for MCTSConfig defaults and validation."""

    def test_defaults(self) -> None:
        """Verify all default values match specification."""
        cfg = MCTSConfig()
        assert cfg.num_simulations == 200
        assert cfg.c_puct == pytest.approx(1.5)
        assert cfg.dirichlet_alpha == pytest.approx(0.3)
        assert cfg.dirichlet_epsilon == pytest.approx(0.25)
        assert cfg.temperature_schedule == {0: 1.0, 30: 0.1}
        assert cfg.virtual_loss == pytest.approx(1.0)
        assert cfg.progressive_widening_alpha == pytest.approx(0.5)
        assert cfg.selection_policy == SelectionPolicyType.puct

    def test_custom_values(self) -> None:
        """Custom values are accepted and stored correctly."""
        cfg = MCTSConfig(
            num_simulations=800,
            c_puct=2.5,
            dirichlet_alpha=0.1,
            dirichlet_epsilon=0.5,
            temperature_schedule={0: 1.0, 10: 0.5, 50: 0.01},
            virtual_loss=3.0,
            progressive_widening_alpha=0.8,
            selection_policy=SelectionPolicyType.gumbel,
        )
        assert cfg.num_simulations == 800
        assert cfg.c_puct == pytest.approx(2.5)
        assert cfg.selection_policy == SelectionPolicyType.gumbel
        assert cfg.temperature_schedule == {0: 1.0, 10: 0.5, 50: 0.01}

    def test_num_simulations_zero_raises(self) -> None:
        """num_simulations=0 raises ValidationError (ge=1 constraint)."""
        with pytest.raises(ValidationError):
            MCTSConfig(num_simulations=0)

    def test_num_simulations_negative_raises(self) -> None:
        """num_simulations=-1 raises ValidationError."""
        with pytest.raises(ValidationError):
            MCTSConfig(num_simulations=-1)

    def test_num_simulations_one_is_valid(self) -> None:
        """num_simulations=1 is the minimum valid value (ge=1)."""
        cfg = MCTSConfig(num_simulations=1)
        assert cfg.num_simulations == 1

    def test_c_puct_zero_raises(self) -> None:
        """c_puct=0 raises ValidationError (gt=0 constraint)."""
        with pytest.raises(ValidationError):
            MCTSConfig(c_puct=0.0)

    def test_c_puct_negative_raises(self) -> None:
        """c_puct=-1 raises ValidationError."""
        with pytest.raises(ValidationError):
            MCTSConfig(c_puct=-1.0)

    def test_c_puct_small_positive_is_valid(self) -> None:
        """c_puct just above zero is valid (gt=0)."""
        cfg = MCTSConfig(c_puct=0.001)
        assert cfg.c_puct == pytest.approx(0.001)

    def test_dirichlet_epsilon_bounds(self) -> None:
        """dirichlet_epsilon must be in [0, 1]."""
        # Valid boundary values
        MCTSConfig(dirichlet_epsilon=0.0)
        MCTSConfig(dirichlet_epsilon=1.0)
        # Out of bounds
        with pytest.raises(ValidationError):
            MCTSConfig(dirichlet_epsilon=1.1)
        with pytest.raises(ValidationError):
            MCTSConfig(dirichlet_epsilon=-0.1)

    def test_selection_policy_alphazero(self) -> None:
        """SelectionPolicyType.alphazero is accepted."""
        cfg = MCTSConfig(selection_policy=SelectionPolicyType.alphazero)
        assert cfg.selection_policy == SelectionPolicyType.alphazero


# ============================================================================
# 4. Tests: NetworkConfig
# ============================================================================


class TestNetworkConfig:
    """Tests for NetworkConfig defaults and validation."""

    def test_defaults(self) -> None:
        """Verify backbone_type is resnet and other defaults."""
        cfg = NetworkConfig()
        assert cfg.backbone_type == BackboneType.resnet
        assert cfg.num_residual_blocks == 6
        assert cfg.num_filters == 64
        assert cfg.fc_hidden_dim == 256
        assert cfg.value_head_hidden_dim == 64
        assert cfg.policy_head_hidden_dim == 64
        assert cfg.use_attention is False
        assert cfg.attention_heads == 4
        assert cfg.dropout_rate == pytest.approx(0.0)
        assert cfg.weight_init == "kaiming"
        assert cfg.compile_model is False

    def test_custom_values(self) -> None:
        """Custom values are accepted and stored correctly."""
        cfg = NetworkConfig(
            backbone_type=BackboneType.transformer,
            num_filters=128,
            use_attention=True,
            attention_heads=8,
            weight_init="xavier",
            compile_model=True,
        )
        assert cfg.backbone_type == BackboneType.transformer
        assert cfg.num_filters == 128
        assert cfg.use_attention is True
        assert cfg.attention_heads == 8
        assert cfg.weight_init == "xavier"
        assert cfg.compile_model is True

    def test_num_residual_blocks_zero_raises(self) -> None:
        """num_residual_blocks must be >= 1."""
        with pytest.raises(ValidationError):
            NetworkConfig(num_residual_blocks=0)

    def test_dropout_rate_at_one_raises(self) -> None:
        """dropout_rate=1.0 raises ValidationError (lt=1 means strictly less than 1)."""
        with pytest.raises(ValidationError):
            NetworkConfig(dropout_rate=1.0)

    def test_dropout_rate_above_one_raises(self) -> None:
        """dropout_rate > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            NetworkConfig(dropout_rate=1.5)

    def test_dropout_rate_negative_raises(self) -> None:
        """dropout_rate < 0 raises ValidationError (ge=0)."""
        with pytest.raises(ValidationError):
            NetworkConfig(dropout_rate=-0.1)

    def test_dropout_rate_boundary_valid(self) -> None:
        """dropout_rate=0.0 and 0.99 are valid boundary values."""
        cfg0 = NetworkConfig(dropout_rate=0.0)
        assert cfg0.dropout_rate == pytest.approx(0.0)
        cfg99 = NetworkConfig(dropout_rate=0.99)
        assert cfg99.dropout_rate == pytest.approx(0.99)

    def test_gnn_backbone(self) -> None:
        """BackboneType.gnn is a valid backbone type."""
        cfg = NetworkConfig(backbone_type=BackboneType.gnn)
        assert cfg.backbone_type == BackboneType.gnn


# ============================================================================
# 5. Tests: TrainingConfig
# ============================================================================


class TestTrainingConfig:
    """Tests for TrainingConfig defaults and validation."""

    def test_defaults(self) -> None:
        """Verify batch_size=256, device=auto, and all other defaults."""
        cfg = TrainingConfig()
        assert cfg.batch_size == 256
        assert cfg.learning_rate == pytest.approx(1e-3)
        assert cfg.lr_schedule == {0: 1.0}
        assert cfg.weight_decay == pytest.approx(1e-4)
        assert cfg.max_iterations == 1000
        assert cfg.games_per_iteration == 25
        assert cfg.replay_buffer_size == 100_000
        assert cfg.checkpoint_interval == 10
        assert cfg.eval_interval == 10
        assert cfg.eval_games == 20
        assert cfg.max_steps_per_game == 200
        assert cfg.num_actors == 1
        assert cfg.device == DeviceType.auto
        assert cfg.mixed_precision is False
        assert cfg.gradient_accumulation_steps == 1

    def test_batch_size_zero_raises(self) -> None:
        """batch_size=0 fails validation (ge=1)."""
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)

    def test_batch_size_negative_raises(self) -> None:
        """batch_size=-1 fails validation."""
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=-1)

    def test_learning_rate_zero_raises(self) -> None:
        """learning_rate=0 fails validation (gt=0)."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0.0)

    def test_learning_rate_negative_raises(self) -> None:
        """learning_rate=-1 fails validation."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=-1.0)

    def test_custom_values(self) -> None:
        """Custom values are accepted and stored correctly."""
        cfg = TrainingConfig(
            batch_size=512,
            learning_rate=0.01,
            device=DeviceType.cpu,
            mixed_precision=True,
            gradient_accumulation_steps=4,
        )
        assert cfg.batch_size == 512
        assert cfg.device == DeviceType.cpu
        assert cfg.mixed_precision is True
        assert cfg.gradient_accumulation_steps == 4

    def test_weight_decay_zero_valid(self) -> None:
        """weight_decay=0 is valid (ge=0)."""
        cfg = TrainingConfig(weight_decay=0.0)
        assert cfg.weight_decay == pytest.approx(0.0)

    def test_lr_schedule_custom(self) -> None:
        """Custom lr_schedule dict is accepted."""
        schedule = {0: 1.0, 100: 0.5, 500: 0.1}
        cfg = TrainingConfig(lr_schedule=schedule)
        assert cfg.lr_schedule == schedule


# ============================================================================
# 6. Tests: CurriculumStage
# ============================================================================


class TestCurriculumStage:
    """Tests for CurriculumStage required fields and defaults."""

    def test_requires_name_and_environment(self) -> None:
        """CurriculumStage requires name and environment fields."""
        stage = CurriculumStage(name="beginner", environment="mini-env")
        assert stage.name == "beginner"
        assert stage.environment == "mini-env"

    def test_missing_name_raises(self) -> None:
        """Omitting 'name' raises a validation error."""
        with pytest.raises(ValidationError):
            CurriculumStage(environment="mini-env")  # type: ignore[call-arg]

    def test_missing_environment_raises(self) -> None:
        """Omitting 'environment' raises a validation error."""
        with pytest.raises(ValidationError):
            CurriculumStage(name="beginner")  # type: ignore[call-arg]

    def test_missing_both_raises(self) -> None:
        """Omitting both required fields raises a validation error."""
        with pytest.raises(ValidationError):
            CurriculumStage()  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        """Optional fields have correct default values."""
        stage = CurriculumStage(name="beginner", environment="mini-env")
        assert stage.min_iterations == 0
        assert stage.max_iterations == 1000
        assert stage.baseline_type == "random"

    def test_custom_values(self) -> None:
        """Custom values override defaults correctly."""
        stage = CurriculumStage(
            name="hard",
            environment="full-env",
            min_iterations=50,
            max_iterations=500,
            baseline_type="greedy",
        )
        assert stage.min_iterations == 50
        assert stage.max_iterations == 500
        assert stage.baseline_type == "greedy"


# ============================================================================
# 7. Tests: CurriculumConfig
# ============================================================================


class TestCurriculumConfig:
    """Tests for CurriculumConfig."""

    def test_empty_stages_by_default(self) -> None:
        """stages is an empty list by default."""
        cfg = CurriculumConfig()
        assert cfg.stages == []

    def test_defaults(self) -> None:
        """All default values are correct."""
        cfg = CurriculumConfig()
        assert cfg.promotion_elo_threshold == pytest.approx(100.0)
        assert cfg.promotion_window == 50
        assert cfg.demotion_enabled is False
        assert cfg.demotion_elo_threshold == pytest.approx(-100.0)

    def test_with_stages(self) -> None:
        """CurriculumConfig accepts a list of CurriculumStage objects."""
        stages = [
            CurriculumStage(name="easy", environment="env-easy"),
            CurriculumStage(name="hard", environment="env-hard", min_iterations=50),
        ]
        cfg = CurriculumConfig(stages=stages)
        assert len(cfg.stages) == 2
        assert cfg.stages[0].name == "easy"
        assert cfg.stages[1].min_iterations == 50

    def test_stages_from_dicts(self) -> None:
        """Stages can be provided as raw dicts and Pydantic coerces them."""
        cfg = CurriculumConfig.model_validate({
            "stages": [
                {"name": "stage1", "environment": "env1"},
                {"name": "stage2", "environment": "env2", "baseline_type": "greedy"},
            ]
        })
        assert len(cfg.stages) == 2
        assert isinstance(cfg.stages[0], CurriculumStage)
        assert cfg.stages[1].baseline_type == "greedy"

    def test_custom_thresholds(self) -> None:
        """Custom promotion/demotion thresholds are accepted."""
        cfg = CurriculumConfig(
            promotion_elo_threshold=200.0,
            promotion_window=100,
            demotion_enabled=True,
            demotion_elo_threshold=-50.0,
        )
        assert cfg.promotion_elo_threshold == pytest.approx(200.0)
        assert cfg.promotion_window == 100
        assert cfg.demotion_enabled is True
        assert cfg.demotion_elo_threshold == pytest.approx(-50.0)


# ============================================================================
# 8. Tests: LoggingConfig
# ============================================================================


class TestLoggingConfig:
    """Tests for LoggingConfig, especially the format_ alias."""

    def test_defaults(self) -> None:
        """All defaults are correct."""
        cfg = LoggingConfig()
        assert cfg.level == "INFO"
        assert cfg.format_ == "pretty"
        assert cfg.log_dir is None
        assert cfg.log_mcts_trees is False
        assert cfg.log_gradient_norms is False
        assert cfg.log_activation_maps is False
        assert cfg.metrics_interval == 10
        assert cfg.correlation_id_enabled is True

    def test_format_alias_from_dict(self) -> None:
        """Constructing with 'format' key (the alias) works via model_validate."""
        cfg = LoggingConfig.model_validate({"format": "json"})
        assert cfg.format_ == "json"

    def test_format_field_name(self) -> None:
        """Constructing with the Python field name 'format_' also works."""
        cfg = LoggingConfig(format_="json")
        assert cfg.format_ == "json"

    def test_format_alias_in_model_dump(self) -> None:
        """model_dump(by_alias=True) produces 'format' key."""
        cfg = LoggingConfig(format_="json")
        dumped = cfg.model_dump(by_alias=True)
        assert "format" in dumped
        assert dumped["format"] == "json"

    def test_format_field_name_in_model_dump(self) -> None:
        """model_dump(by_alias=False) produces 'format_' key."""
        cfg = LoggingConfig(format_="json")
        dumped = cfg.model_dump(by_alias=False)
        assert "format_" in dumped
        assert dumped["format_"] == "json"

    def test_custom_log_dir(self) -> None:
        """log_dir can be set to a custom path."""
        cfg = LoggingConfig(log_dir="/var/log/sdk")
        assert cfg.log_dir == "/var/log/sdk"

    def test_debug_level(self) -> None:
        """Level can be set to DEBUG."""
        cfg = LoggingConfig(level="DEBUG")
        assert cfg.level == "DEBUG"

    def test_all_bool_flags_enabled(self) -> None:
        """All boolean logging flags can be enabled."""
        cfg = LoggingConfig(
            log_mcts_trees=True,
            log_gradient_norms=True,
            log_activation_maps=True,
            correlation_id_enabled=False,
        )
        assert cfg.log_mcts_trees is True
        assert cfg.log_gradient_norms is True
        assert cfg.log_activation_maps is True
        assert cfg.correlation_id_enabled is False


# ============================================================================
# 9. Tests: SDKConfig (defaults)
# ============================================================================


class TestSDKConfigDefaults:
    """Tests for SDKConfig default construction and sub-config accessibility."""

    def test_defaults(self) -> None:
        """All sub-configs are created with defaults; seed=42."""
        cfg = SDKConfig()
        assert cfg.seed == 42
        assert cfg.project_name == "strategic-planning-sdk"
        assert cfg.output_dir == "./outputs"

    def test_all_sub_configs_accessible(self) -> None:
        """All sub-config attributes are present and correctly typed."""
        cfg = SDKConfig()
        assert isinstance(cfg.mcts, MCTSConfig)
        assert isinstance(cfg.network, NetworkConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.curriculum, CurriculumConfig)
        assert isinstance(cfg.logging, LoggingConfig)

    def test_sub_config_defaults_propagate(self) -> None:
        """Sub-configs within SDKConfig carry their own defaults."""
        cfg = SDKConfig()
        assert cfg.mcts.num_simulations == 200
        assert cfg.training.batch_size == 256
        assert cfg.network.backbone_type == BackboneType.resnet
        assert cfg.logging.level == "INFO"
        assert cfg.curriculum.stages == []

    def test_model_validate_empty_dict(self) -> None:
        """model_validate({}) produces a default SDKConfig."""
        cfg = SDKConfig.model_validate({})
        assert cfg.seed == 42
        assert cfg.training.batch_size == 256


# ============================================================================
# 10. Tests: SDKConfig.from_yaml
# ============================================================================


class TestSDKConfigFromYaml:
    """Tests for SDKConfig.from_yaml method."""

    def test_from_yaml_with_real_config(self, real_config_yaml: Path) -> None:
        """Load the real configs/config.yaml and verify top-level fields."""
        cfg = SDKConfig.from_yaml(real_config_yaml)
        assert cfg.seed == 42
        assert cfg.project_name == "strategic-planning-sdk"
        assert cfg.output_dir == "./outputs"
        # Sub-configs get defaults since file only has top-level keys
        assert cfg.mcts.num_simulations == 200
        assert cfg.training.batch_size == 256

    def test_from_yaml_missing_file_raises(self, tmp_path: Path) -> None:
        """from_yaml raises ConfigFileNotFoundError for a missing file."""
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(ConfigFileNotFoundError) as exc_info:
            SDKConfig.from_yaml(missing)
        assert exc_info.value.path == missing

    def test_from_yaml_invalid_content_raises(self, tmp_path: Path) -> None:
        """from_yaml raises ConfigValidationError for invalid field values."""
        p = tmp_path / "bad.yaml"
        # batch_size=0 violates ge=1
        p.write_text(yaml.dump({"training": {"batch_size": 0}}))
        with pytest.raises(ConfigValidationError):
            SDKConfig.from_yaml(p)

    def test_from_yaml_negative_learning_rate_raises(self, tmp_path: Path) -> None:
        """from_yaml raises ConfigValidationError for negative learning_rate."""
        p = tmp_path / "bad_lr.yaml"
        p.write_text(yaml.dump({"training": {"learning_rate": -1.0}}))
        with pytest.raises(ConfigValidationError):
            SDKConfig.from_yaml(p)

    def test_from_yaml_empty_file(self, tmp_path: Path) -> None:
        """An empty YAML file produces a default SDKConfig."""
        p = tmp_path / "empty.yaml"
        p.write_text("")
        cfg = SDKConfig.from_yaml(p)
        assert cfg.seed == 42
        assert cfg.training.batch_size == 256

    def test_from_yaml_partial_config(self, tmp_path: Path) -> None:
        """A partial YAML file fills in missing fields with defaults."""
        p = tmp_path / "partial.yaml"
        p.write_text(yaml.dump({"seed": 99, "project_name": "my-experiment"}))
        cfg = SDKConfig.from_yaml(p)
        assert cfg.seed == 99
        assert cfg.project_name == "my-experiment"
        assert cfg.output_dir == "./outputs"  # default
        assert cfg.training.batch_size == 256  # default

    def test_from_yaml_with_nested_sub_config(self, tmp_path: Path) -> None:
        """A YAML file with nested sub-config sections loads correctly."""
        p = tmp_path / "nested.yaml"
        data = {
            "seed": 7,
            "mcts": {"num_simulations": 500, "c_puct": 3.0},
            "network": {"num_filters": 128, "use_attention": True},
        }
        p.write_text(yaml.dump(data))
        cfg = SDKConfig.from_yaml(p)
        assert cfg.seed == 7
        assert cfg.mcts.num_simulations == 500
        assert cfg.mcts.c_puct == pytest.approx(3.0)
        assert cfg.network.num_filters == 128
        assert cfg.network.use_attention is True
        # Unspecified sub-configs get defaults
        assert cfg.training.batch_size == 256

    def test_from_yaml_accepts_string_path(self, tmp_path: Path) -> None:
        """from_yaml accepts a string path in addition to Path objects."""
        p = tmp_path / "str_path.yaml"
        p.write_text(yaml.dump({"seed": 55}))
        cfg = SDKConfig.from_yaml(str(p))
        assert cfg.seed == 55


# ============================================================================
# 11. Tests: SDKConfig.from_yaml_with_overrides
# ============================================================================


class TestSDKConfigFromYamlWithOverrides:
    """Tests for SDKConfig.from_yaml_with_overrides method."""

    def test_merges_two_files(
        self, base_yaml: Path, override_yaml: Path
    ) -> None:
        """Deep-merge override YAML on top of base YAML."""
        cfg = SDKConfig.from_yaml_with_overrides(base_yaml, override_yaml)
        # Overridden values
        assert cfg.seed == 99
        assert cfg.training.batch_size == 512
        assert cfg.mcts.num_simulations == 400
        assert cfg.mcts.c_puct == pytest.approx(2.0)
        # Preserved from base (not in override)
        assert cfg.project_name == "base-project"
        assert cfg.output_dir == "./base-outputs"
        assert cfg.training.learning_rate == pytest.approx(0.01)

    def test_missing_base_raises(
        self, tmp_path: Path, override_yaml: Path
    ) -> None:
        """Raises ConfigFileNotFoundError when base file is missing."""
        missing = tmp_path / "no_such_base.yaml"
        with pytest.raises(ConfigFileNotFoundError):
            SDKConfig.from_yaml_with_overrides(missing, override_yaml)

    def test_missing_override_raises(
        self, tmp_path: Path, base_yaml: Path
    ) -> None:
        """Raises ConfigFileNotFoundError when override file is missing."""
        missing = tmp_path / "no_such_override.yaml"
        with pytest.raises(ConfigFileNotFoundError):
            SDKConfig.from_yaml_with_overrides(base_yaml, missing)

    def test_invalid_merged_raises(self, tmp_path: Path) -> None:
        """Raises ConfigValidationError when merged result is invalid."""
        base = tmp_path / "base_ok.yaml"
        base.write_text(yaml.dump({"training": {"batch_size": 256}}))
        override = tmp_path / "override_bad.yaml"
        override.write_text(yaml.dump({"training": {"batch_size": 0}}))
        with pytest.raises(ConfigValidationError):
            SDKConfig.from_yaml_with_overrides(base, override)

    def test_override_adds_new_section(self, tmp_path: Path) -> None:
        """Override file can add an entirely new config section."""
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"seed": 10}))
        override = tmp_path / "override.yaml"
        override.write_text(yaml.dump({"network": {"num_filters": 256}}))
        cfg = SDKConfig.from_yaml_with_overrides(base, override)
        assert cfg.seed == 10
        assert cfg.network.num_filters == 256

    def test_empty_override_preserves_base(self, tmp_path: Path) -> None:
        """An empty override file preserves all base values."""
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"seed": 77, "project_name": "my-proj"}))
        override = tmp_path / "empty_override.yaml"
        override.write_text("")
        cfg = SDKConfig.from_yaml_with_overrides(base, override)
        assert cfg.seed == 77
        assert cfg.project_name == "my-proj"


# ============================================================================
# 12. Tests: apply_env_overrides
# ============================================================================


class TestApplyEnvOverrides:
    """Tests for SDKConfig.apply_env_overrides."""

    def test_training_batch_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SPSDK_TRAINING_BATCH_SIZE=512 overrides training.batch_size."""
        monkeypatch.setenv("SPSDK_TRAINING_BATCH_SIZE", "512")
        cfg = SDKConfig()
        new_cfg = cfg.apply_env_overrides()
        assert new_cfg.training.batch_size == 512
        # Original is not mutated
        assert cfg.training.batch_size == 256

    def test_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SPSDK_SEED=123 overrides the top-level seed field."""
        monkeypatch.setenv("SPSDK_SEED", "123")
        cfg = SDKConfig()
        new_cfg = cfg.apply_env_overrides()
        assert new_cfg.seed == 123

    def test_mcts_num_simulations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SPSDK_MCTS_NUM_SIMULATIONS=400 overrides mcts.num_simulations."""
        monkeypatch.setenv("SPSDK_MCTS_NUM_SIMULATIONS", "400")
        cfg = SDKConfig()
        new_cfg = cfg.apply_env_overrides()
        assert new_cfg.mcts.num_simulations == 400

    def test_multiple_simultaneous(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multiple env vars are applied simultaneously."""
        monkeypatch.setenv("SPSDK_TRAINING_BATCH_SIZE", "512")
        monkeypatch.setenv("SPSDK_SEED", "123")
        monkeypatch.setenv("SPSDK_MCTS_NUM_SIMULATIONS", "800")
        cfg = SDKConfig()
        new_cfg = cfg.apply_env_overrides()
        assert new_cfg.training.batch_size == 512
        assert new_cfg.seed == 123
        assert new_cfg.mcts.num_simulations == 800

    def test_bool_coercion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bool fields are coerced correctly from env vars."""
        monkeypatch.setenv("SPSDK_TRAINING_MIXED_PRECISION", "true")
        cfg = SDKConfig()
        new_cfg = cfg.apply_env_overrides()
        assert new_cfg.training.mixed_precision is True

    def test_float_coercion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Float fields are coerced correctly from env vars."""
        monkeypatch.setenv("SPSDK_MCTS_C_PUCT", "2.5")
        cfg = SDKConfig()
        new_cfg = cfg.apply_env_overrides()
        assert new_cfg.mcts.c_puct == pytest.approx(2.5)

    def test_device_enum_coercion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DeviceType enum is coerced correctly from env vars."""
        monkeypatch.setenv("SPSDK_TRAINING_DEVICE", "cpu")
        cfg = SDKConfig()
        new_cfg = cfg.apply_env_overrides()
        assert new_cfg.training.device == DeviceType.cpu

    def test_ignores_unrelated_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env vars without the SPSDK_ prefix are ignored."""
        monkeypatch.setenv("OTHER_PREFIX_SEED", "999")
        cfg = SDKConfig()
        new_cfg = cfg.apply_env_overrides()
        assert new_cfg.seed == 42

    def test_custom_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A custom prefix works for env override scanning."""
        monkeypatch.setenv("MYAPP_SEED", "777")
        cfg = SDKConfig()
        new_cfg = cfg.apply_env_overrides(prefix="MYAPP_")
        assert new_cfg.seed == 777

    def test_invalid_value_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid coercion (non-numeric for int) does not crash.

        The implementation catches ValueError/TypeError and logs a warning,
        leaving the original value in place.
        """
        monkeypatch.setenv("SPSDK_TRAINING_BATCH_SIZE", "not_a_number")
        cfg = SDKConfig()
        new_cfg = cfg.apply_env_overrides()
        # batch_size should remain at default because coercion failed
        assert new_cfg.training.batch_size == 256

    def test_returns_new_instance(self) -> None:
        """apply_env_overrides returns a new SDKConfig, not the same object."""
        cfg = SDKConfig()
        new_cfg = cfg.apply_env_overrides()
        assert new_cfg is not cfg

    def test_original_not_mutated_after_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The original SDKConfig instance is not mutated by env overrides."""
        monkeypatch.setenv("SPSDK_SEED", "999")
        monkeypatch.setenv("SPSDK_TRAINING_BATCH_SIZE", "1024")
        cfg = SDKConfig()
        original_seed = cfg.seed
        original_batch = cfg.training.batch_size
        cfg.apply_env_overrides()
        assert cfg.seed == original_seed
        assert cfg.training.batch_size == original_batch

    def test_network_num_filters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SPSDK_NETWORK_NUM_FILTERS=256 overrides network.num_filters."""
        monkeypatch.setenv("SPSDK_NETWORK_NUM_FILTERS", "256")
        cfg = SDKConfig()
        new_cfg = cfg.apply_env_overrides()
        assert new_cfg.network.num_filters == 256

    def test_validation_error_after_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env override that passes coercion but fails Pydantic validation raises ConfigValidationError."""
        # batch_size has ge=1, so 0 is coercible to int but fails validation
        monkeypatch.setenv("SPSDK_TRAINING_BATCH_SIZE", "0")
        cfg = SDKConfig()
        with pytest.raises(ConfigValidationError, match=r"Validation failed after applying env overrides"):
            cfg.apply_env_overrides()


# ============================================================================
# 13. Tests: load_config
# ============================================================================


class TestLoadConfig:
    """Tests for the load_config convenience function."""

    def test_no_args_returns_defaults(self) -> None:
        """load_config() with no arguments returns default SDKConfig."""
        cfg = load_config()
        assert cfg.seed == 42
        assert cfg.project_name == "strategic-planning-sdk"
        assert cfg.training.batch_size == 256

    def test_with_path(self, real_config_yaml: Path) -> None:
        """load_config with a YAML path loads the file."""
        cfg = load_config(path=real_config_yaml)
        assert cfg.seed == 42
        assert cfg.project_name == "strategic-planning-sdk"

    def test_with_overrides_dict(self) -> None:
        """load_config with overrides dict merges into defaults."""
        cfg = load_config(overrides={"training": {"batch_size": 512}, "seed": 99})
        assert cfg.training.batch_size == 512
        assert cfg.seed == 99
        # Other fields remain default
        assert cfg.mcts.num_simulations == 200

    def test_with_path_and_overrides(self, tmp_path: Path) -> None:
        """load_config merges overrides on top of the YAML file."""
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"seed": 10, "training": {"batch_size": 64}}))
        cfg = load_config(path=p, overrides={"training": {"batch_size": 128}})
        assert cfg.seed == 10
        assert cfg.training.batch_size == 128

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        """load_config raises ConfigFileNotFoundError for a missing file."""
        missing = tmp_path / "ghost.yaml"
        with pytest.raises(ConfigFileNotFoundError):
            load_config(path=missing)

    def test_env_overrides_applied_last(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env vars override both YAML and programmatic overrides."""
        monkeypatch.setenv("SPSDK_TRAINING_BATCH_SIZE", "1024")

        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"training": {"batch_size": 64}}))

        cfg = load_config(
            path=p,
            overrides={"training": {"batch_size": 128}},
        )
        # Env var wins over both YAML (64) and overrides dict (128)
        assert cfg.training.batch_size == 1024

    def test_invalid_overrides_raises(self) -> None:
        """Overrides that produce invalid values raise ConfigValidationError."""
        with pytest.raises(ConfigValidationError):
            load_config(overrides={"training": {"batch_size": 0}})

    def test_nested_overrides(self) -> None:
        """Nested dict overrides apply to sub-configs."""
        cfg = load_config(
            overrides={
                "mcts": {"num_simulations": 500, "c_puct": 3.0},
                "network": {"num_filters": 128},
            }
        )
        assert cfg.mcts.num_simulations == 500
        assert cfg.mcts.c_puct == pytest.approx(3.0)
        assert cfg.network.num_filters == 128
        # Unrelated defaults preserved
        assert cfg.training.batch_size == 256

    def test_overrides_only_no_path(self) -> None:
        """load_config with only overrides (no path) works from defaults."""
        cfg = load_config(overrides={"seed": 77})
        assert cfg.seed == 77
        assert cfg.training.batch_size == 256  # default

    def test_env_seed_overrides_file_seed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SPSDK_SEED env var overrides the seed from a YAML file."""
        monkeypatch.setenv("SPSDK_SEED", "999")
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"seed": 10}))
        cfg = load_config(path=p)
        assert cfg.seed == 999

    def test_load_config_returns_sdk_config_type(self) -> None:
        """load_config always returns an SDKConfig instance."""
        cfg = load_config()
        assert isinstance(cfg, SDKConfig)
