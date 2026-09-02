"""Tests for ConfigMixin base class.

This module provides comprehensive tests for the ConfigMixin class,
which is the foundation of all configuration dataclasses in the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import ClassVar

import pytest

from alpha_zero.core.config import ConfigMixin

# =============================================================================
# Test Configuration Classes
# =============================================================================


@dataclass
class BasicConfig(ConfigMixin):
    """Basic config for testing simple types."""

    ENV_PREFIX: ClassVar[str] = "TEST_BASIC_"
    value: int = 10
    name: str = "default"


@dataclass
class FloatConfig(ConfigMixin):
    """Config for testing float parsing."""

    ENV_PREFIX: ClassVar[str] = "TEST_FLOAT_"
    rate: float = 0.001
    scale: float = 1.0


@dataclass
class BoolConfig(ConfigMixin):
    """Config for testing boolean parsing."""

    ENV_PREFIX: ClassVar[str] = "TEST_BOOL_"
    enabled: bool = True
    debug: bool = False


@dataclass
class ListConfig(ConfigMixin):
    """Config for testing list parsing."""

    ENV_PREFIX: ClassVar[str] = "TEST_LIST_"
    tags: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=lambda: ["a", "b"])


@dataclass
class NestedInnerConfig(ConfigMixin):
    """Inner config for nesting tests."""

    ENV_PREFIX: ClassVar[str] = "TEST_INNER_"
    inner_value: int = 5


@dataclass
class NestedOuterConfig(ConfigMixin):
    """Outer config for nesting tests."""

    ENV_PREFIX: ClassVar[str] = "TEST_OUTER_"
    name: str = "outer"
    inner: NestedInnerConfig = field(default_factory=NestedInnerConfig)


@dataclass
class OptionalConfig(ConfigMixin):
    """Config with optional fields."""

    ENV_PREFIX: ClassVar[str] = "TEST_OPT_"
    required: int = 0
    optional: int | None = None


# =============================================================================
# TestConfigMixinFromDict
# =============================================================================


class TestConfigMixinFromDict:
    """Tests for ConfigMixin.from_dict() method."""

    @pytest.mark.parametrize(
        "input_dict,expected_value,expected_name",
        [
            ({"value": 42}, 42, "default"),
            ({"name": "custom"}, 10, "custom"),
            ({"value": 100, "name": "both"}, 100, "both"),
            ({}, 10, "default"),  # Empty dict uses defaults
        ],
    )
    def test_from_dict_creates_config(
        self, input_dict: dict, expected_value: int, expected_name: str
    ):
        """Test from_dict creates config with correct values."""
        config = BasicConfig.from_dict(input_dict)
        assert config.value == expected_value
        assert config.name == expected_name

    def test_from_dict_ignores_unknown_keys(self):
        """Test from_dict ignores keys not in the dataclass."""
        config = BasicConfig.from_dict({"value": 99, "unknown_key": "ignored", "another": 123})
        assert config.value == 99
        assert not hasattr(config, "unknown_key")

    def test_from_dict_with_nested_config(self):
        """Test from_dict handles nested ConfigMixin objects.

        Note: ConfigMixin.from_dict preserves nested dicts as-is.
        Automatic deserialization of nested dataclasses requires
        explicit handling in subclasses.
        """
        config = NestedOuterConfig.from_dict({"name": "test", "inner": {"inner_value": 42}})
        assert config.name == "test"
        # Nested dicts are preserved as dicts, not converted to dataclasses
        assert config.inner == {"inner_value": 42}

    def test_from_dict_with_partial_nested(self):
        """Test from_dict with partial nested config uses defaults."""
        config = NestedOuterConfig.from_dict({"name": "partial"})
        assert config.name == "partial"
        assert config.inner.inner_value == 5  # Default

    def test_from_dict_non_dataclass_raises(self):
        """Test from_dict raises TypeError for non-dataclass."""

        class NotADataclass(ConfigMixin):
            value: int = 10

        with pytest.raises(TypeError, match="must be a dataclass"):
            NotADataclass.from_dict({})


# =============================================================================
# TestConfigMixinToDict
# =============================================================================


class TestConfigMixinToDict:
    """Tests for ConfigMixin.to_dict() method."""

    def test_to_dict_basic(self):
        """Test to_dict returns dictionary with all fields."""
        config = BasicConfig(value=42, name="test")
        result = config.to_dict()

        assert result == {"value": 42, "name": "test"}

    def test_to_dict_with_nested(self):
        """Test to_dict serializes nested configs."""
        config = NestedOuterConfig(name="outer", inner=NestedInnerConfig(inner_value=99))
        result = config.to_dict()

        assert result == {"name": "outer", "inner": {"inner_value": 99}}

    def test_to_dict_roundtrip(self):
        """Test to_dict/from_dict roundtrip preserves values.

        Note: Nested dataclasses are serialized to dicts but not
        automatically restored to dataclass instances on from_dict.
        """
        original = NestedOuterConfig(name="roundtrip", inner=NestedInnerConfig(inner_value=123))
        data = original.to_dict()
        restored = NestedOuterConfig.from_dict(data)

        assert restored.name == original.name
        # Nested is a dict after roundtrip, not a dataclass
        assert restored.inner == {"inner_value": 123}


# =============================================================================
# TestConfigMixinFromEnv
# =============================================================================


class TestConfigMixinFromEnv:
    """Tests for ConfigMixin.from_env() method."""

    def test_from_env_reads_environment_variables(self):
        """Test from_env reads values from environment."""
        os.environ["TEST_BASIC_VALUE"] = "100"
        os.environ["TEST_BASIC_NAME"] = "from_env"

        try:
            config = BasicConfig.from_env()
            assert config.value == 100
            assert config.name == "from_env"
        finally:
            del os.environ["TEST_BASIC_VALUE"]
            del os.environ["TEST_BASIC_NAME"]

    def test_from_env_with_no_env_vars(self):
        """Test from_env returns defaults when no env vars set."""
        # Ensure no conflicting env vars
        for key in list(os.environ.keys()):
            if key.startswith("TEST_BASIC_"):
                del os.environ[key]

        config = BasicConfig.from_env()
        assert config.value == 10  # Default
        assert config.name == "default"

    def test_from_env_with_custom_prefix(self):
        """Test from_env with custom prefix override."""
        os.environ["CUSTOM_VALUE"] = "200"

        try:
            config = BasicConfig.from_env(prefix="CUSTOM_")
            assert config.value == 200
        finally:
            del os.environ["CUSTOM_VALUE"]


# =============================================================================
# TestConfigMixinApplyEnvOverrides
# =============================================================================


class TestConfigMixinApplyEnvOverrides:
    """Tests for ConfigMixin.apply_env_overrides() method."""

    def test_apply_env_overrides_modifies_config(self):
        """Test apply_env_overrides updates existing config."""
        config = BasicConfig(value=10, name="original")
        os.environ["TEST_BASIC_VALUE"] = "999"

        try:
            result = config.apply_env_overrides()
            assert result is config  # Returns self
            assert config.value == 999
            assert config.name == "original"  # Unchanged
        finally:
            del os.environ["TEST_BASIC_VALUE"]

    def test_apply_env_overrides_chain(self):
        """Test apply_env_overrides can be chained."""
        os.environ["TEST_BASIC_VALUE"] = "50"

        try:
            config = BasicConfig().apply_env_overrides()
            assert config.value == 50
        finally:
            del os.environ["TEST_BASIC_VALUE"]


# =============================================================================
# TestConfigMixinParseValue
# =============================================================================


class TestConfigMixinParseValue:
    """Tests for ConfigMixin._parse_value() static method."""

    @pytest.mark.parametrize(
        "value,target_type,expected",
        [
            # Integer parsing
            ("42", int, 42),
            ("-1", int, -1),
            ("0", int, 0),
            # Float parsing
            ("3.14", float, 3.14),
            ("1e-5", float, 1e-5),
            ("-0.5", float, -0.5),
            # String passthrough
            ("hello", str, "hello"),
            ("", str, ""),
        ],
    )
    def test_parse_value_numeric_and_string(self, value: str, target_type: type, expected):
        """Test _parse_value handles numeric and string types."""
        result = ConfigMixin._parse_value(value, target_type, None)
        assert result == expected
        assert isinstance(result, target_type)

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ],
    )
    def test_parse_value_boolean(self, value: str, expected: bool):
        """Test _parse_value handles boolean strings."""
        result = ConfigMixin._parse_value(value, bool, None)
        assert result is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("a,b,c", ["a", "b", "c"]),
            ("single", ["single"]),
            ("", []),
            ("  a  ,  b  ", ["a", "b"]),  # Whitespace stripped
        ],
    )
    def test_parse_value_list(self, value: str, expected: list):
        """Test _parse_value handles comma-separated lists."""
        result = ConfigMixin._parse_value(value, list, None)
        assert result == expected

    def test_parse_value_with_default_type_inference(self):
        """Test _parse_value infers type from default value."""
        result = ConfigMixin._parse_value("42", str, 10)  # Default is int
        assert result == 42
        assert isinstance(result, int)


# =============================================================================
# TestConfigMixinMerge
# =============================================================================


class TestConfigMixinMerge:
    """Tests for ConfigMixin.merge() method."""

    def test_merge_with_dict(self):
        """Test merge with dictionary updates values."""
        config = BasicConfig(value=10, name="original")
        config.merge({"value": 99})

        assert config.value == 99
        assert config.name == "original"

    def test_merge_with_config(self):
        """Test merge with another ConfigMixin instance."""
        config1 = BasicConfig(value=10, name="first")
        config2 = BasicConfig(value=20, name="second")

        config1.merge(config2)

        assert config1.value == 20
        assert config1.name == "second"

    def test_merge_returns_self(self):
        """Test merge returns self for chaining."""
        config = BasicConfig()
        result = config.merge({"value": 50})

        assert result is config


# =============================================================================
# TestConfigMixinRepr
# =============================================================================


class TestConfigMixinRepr:
    """Tests for ConfigMixin.__repr__() method."""

    def test_repr_format(self):
        """Test __repr__ returns readable string."""
        config = BasicConfig(value=42, name="test")
        result = repr(config)

        assert "BasicConfig" in result
        assert "value=42" in result
        assert "name='test'" in result


# =============================================================================
# TestConfigMixinEdgeCases
# =============================================================================


class TestConfigMixinEdgeCases:
    """Edge case tests for ConfigMixin."""

    def test_empty_list_env_override(self):
        """Test environment override for empty list."""
        os.environ["TEST_LIST_TAGS"] = ""

        try:
            config = ListConfig().apply_env_overrides()
            assert config.tags == []
        finally:
            del os.environ["TEST_LIST_TAGS"]

    def test_list_with_values_env_override(self):
        """Test environment override for list with values."""
        os.environ["TEST_LIST_TAGS"] = "x,y,z"

        try:
            config = ListConfig().apply_env_overrides()
            assert config.tags == ["x", "y", "z"]
        finally:
            del os.environ["TEST_LIST_TAGS"]

    def test_deeply_nested_config(self):
        """Test handling of deeply nested configs.

        Note: ConfigMixin preserves nested dicts as-is without
        automatically converting them to dataclass instances.
        """

        @dataclass
        class Level3(ConfigMixin):
            value: int = 3

        @dataclass
        class Level2(ConfigMixin):
            level3: Level3 = field(default_factory=Level3)

        @dataclass
        class Level1(ConfigMixin):
            level2: Level2 = field(default_factory=Level2)

        config = Level1.from_dict({"level2": {"level3": {"value": 42}}})

        # Nested dicts are preserved as dicts
        assert config.level2 == {"level3": {"value": 42}}

    def test_optional_field_none(self):
        """Test optional fields can be None."""
        config = OptionalConfig.from_dict({"required": 10, "optional": None})

        assert config.required == 10
        assert config.optional is None

    def test_optional_field_with_value(self):
        """Test optional fields can have values."""
        config = OptionalConfig.from_dict({"required": 10, "optional": 20})

        assert config.required == 10
        assert config.optional == 20
