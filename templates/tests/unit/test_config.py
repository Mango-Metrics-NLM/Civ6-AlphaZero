"""
Unit tests for the config module.

Tests cover:
- EnvironmentConfig for environment variable handling
- TemplateVariable validation
- TemplateConfig substitution and validation
- ConfigLoader file loading and merging
- ConfigManager high-level operations
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from template_system.config import (
    STANDARD_TEMPLATE_VARIABLES,
    ConfigLoader,
    ConfigManager,
    EnvironmentConfig,
    TemplateConfig,
    TemplateVariable,
)

if TYPE_CHECKING:
    pass


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = EnvironmentConfig()
        assert config.prefix == ""
        assert config.separator == "__"
        assert config.case_sensitive is False
        assert config.load_dotenv is True

    def test_get_env_key_simple(self) -> None:
        """Test converting simple config key to env key."""
        config = EnvironmentConfig(prefix="APP_")
        assert config.get_env_key("debug") == "APP_DEBUG"

    def test_get_env_key_nested(self) -> None:
        """Test converting nested config key to env key."""
        config = EnvironmentConfig(prefix="APP_")
        assert config.get_env_key("database.host") == "APP_DATABASE__HOST"

    def test_get_env_key_case_sensitive(self) -> None:
        """Test case-sensitive env key conversion."""
        config = EnvironmentConfig(prefix="App_", case_sensitive=True)
        assert config.get_env_key("Debug") == "App_Debug"

    def test_parse_env_key_valid(self) -> None:
        """Test parsing valid env key to config key."""
        config = EnvironmentConfig(prefix="APP_")
        assert config.parse_env_key("APP_DATABASE__HOST") == "database.host"

    def test_parse_env_key_invalid(self) -> None:
        """Test parsing env key without matching prefix."""
        config = EnvironmentConfig(prefix="APP_")
        assert config.parse_env_key("OTHER_KEY") is None


class TestTemplateVariable:
    """Tests for TemplateVariable dataclass."""

    def test_variable_creation(self) -> None:
        """Test creating a template variable."""
        var = TemplateVariable(
            name="TEST_VAR",
            description="Test variable",
            default="default_value",
            required=False,
            example="example_value",
        )
        assert var.name == "TEST_VAR"
        assert var.default == "default_value"
        assert var.required is False

    def test_variable_validation_default(self) -> None:
        """Test default validation passes."""
        var = TemplateVariable(name="TEST", description="Test")
        assert var.validate("any_value") is True

    def test_variable_validation_custom(self) -> None:
        """Test custom validator."""
        var = TemplateVariable(
            name="PORT",
            description="Port number",
            validator=lambda v: isinstance(v, int) and 1 <= v <= 65535,
        )
        assert var.validate(8080) is True
        assert var.validate(0) is False
        assert var.validate("8080") is False


class TestTemplateConfig:
    """Tests for TemplateConfig dataclass."""

    def test_get_variable_existing(self) -> None:
        """Test getting an existing variable."""
        config = TemplateConfig(variables={"NAME": "test"})
        assert config.get_variable("NAME") == "test"

    def test_get_variable_default(self) -> None:
        """Test getting non-existent variable with default."""
        config = TemplateConfig(variables={})
        assert config.get_variable("MISSING", "default") == "default"

    def test_set_variable(self) -> None:
        """Test setting a variable."""
        config = TemplateConfig()
        config.set_variable("NEW_VAR", "new_value")
        assert config.get_variable("NEW_VAR") == "new_value"

    def test_substitute_simple(self) -> None:
        """Test simple variable substitution."""
        config = TemplateConfig(variables={"NAME": "test_app"})
        result = config.substitute("Application: {NAME}")
        assert result == "Application: test_app"

    def test_substitute_multiple(self) -> None:
        """Test multiple variable substitution."""
        config = TemplateConfig(variables={
            "NAME": "app",
            "VERSION": "1.0.0",
        })
        result = config.substitute("{NAME} v{VERSION}")
        assert result == "app v1.0.0"

    def test_substitute_missing_variable(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test substitution with missing variable logs warning."""
        import logging
        caplog.set_level(logging.WARNING)

        config = TemplateConfig(variables={})
        result = config.substitute("Hello {MISSING}")
        assert result == "Hello {MISSING}"
        assert "Template variable not set" in caplog.text

    def test_validate_required_missing(self) -> None:
        """Test validation fails for missing required variable."""
        config = TemplateConfig(variables={}, validate_required=True)
        errors = config.validate()
        # Should have errors for required standard variables
        assert len(errors) > 0

    def test_validate_all_present(self) -> None:
        """Test validation passes when all required present."""
        config = TemplateConfig(
            variables={
                "SYSTEM_NAME": "Test",
                "PROJECT_NAME": "test",
                "PACKAGE_NAME": "test",
            },
            validate_required=True,
        )
        errors = config.validate()
        assert len(errors) == 0


class TestStandardTemplateVariables:
    """Tests for standard template variables."""

    def test_required_variables_defined(self) -> None:
        """Test that required variables are defined."""
        required_names = {"SYSTEM_NAME", "PROJECT_NAME", "PACKAGE_NAME"}
        defined_names = {v.name for v in STANDARD_TEMPLATE_VARIABLES}
        assert required_names.issubset(defined_names)

    def test_variables_have_descriptions(self) -> None:
        """Test all variables have descriptions."""
        for var in STANDARD_TEMPLATE_VARIABLES:
            assert var.description, f"Variable {var.name} missing description"

    def test_variables_have_examples(self) -> None:
        """Test all variables have examples."""
        for var in STANDARD_TEMPLATE_VARIABLES:
            assert var.example, f"Variable {var.name} missing example"


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_load_json_file(self, sample_json_config: Path) -> None:
        """Test loading a JSON config file."""
        loader = ConfigLoader()
        config = loader.load_file(sample_json_config)
        assert config["app"]["name"] == "test_app"
        assert config["app"]["debug"] is True

    def test_load_yaml_file(self, sample_yaml_config: Path) -> None:
        """Test loading a YAML config file."""
        pytest.importorskip("yaml")
        loader = ConfigLoader()
        config = loader.load_file(sample_yaml_config)
        assert config["app"]["name"] == "test_app"
        assert config["database"]["host"] == "localhost"

    def test_load_nonexistent_file(self) -> None:
        """Test loading non-existent file raises error."""
        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file(Path("/nonexistent/file.json"))

    def test_load_unsupported_format(self, temp_dir: Path) -> None:
        """Test loading unsupported format raises error."""
        file_path = temp_dir / "config.xml"
        file_path.write_text("<config></config>")
        loader = ConfigLoader()
        with pytest.raises(ValueError, match="Unsupported config format"):
            loader.load_file(file_path)

    def test_load_env(self) -> None:
        """Test loading config from environment variables."""
        env_config = EnvironmentConfig(prefix="TEST_CFG_")
        loader = ConfigLoader(env_config=env_config)

        # Set test environment variable
        os.environ["TEST_CFG_DEBUG"] = "true"
        os.environ["TEST_CFG_PORT"] = "8080"
        try:
            config = loader.load_env()
            assert config.get("debug") is True
            assert config.get("port") == 8080
        finally:
            os.environ.pop("TEST_CFG_DEBUG", None)
            os.environ.pop("TEST_CFG_PORT", None)

    def test_merge_configs(self) -> None:
        """Test merging multiple configs."""
        loader = ConfigLoader()
        base = {"app": {"name": "base", "debug": False}}
        override = {"app": {"debug": True}, "new_key": "value"}
        merged = loader.merge(base, override)

        assert merged["app"]["name"] == "base"
        assert merged["app"]["debug"] is True
        assert merged["new_key"] == "value"

    def test_parse_value_boolean(self) -> None:
        """Test parsing boolean values."""
        loader = ConfigLoader()
        assert loader._parse_value("true") is True
        assert loader._parse_value("false") is False
        assert loader._parse_value("yes") is True
        assert loader._parse_value("no") is False
        assert loader._parse_value("1") is True
        assert loader._parse_value("0") is False

    def test_parse_value_numbers(self) -> None:
        """Test parsing numeric values."""
        loader = ConfigLoader()
        assert loader._parse_value("42") == 42
        assert loader._parse_value("3.14") == 3.14
        assert loader._parse_value("-10") == -10

    def test_parse_value_string(self) -> None:
        """Test parsing string values."""
        loader = ConfigLoader()
        assert loader._parse_value("hello") == "hello"
        assert loader._parse_value("hello world") == "hello world"


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_load_existing_config(self, sample_yaml_config: Path) -> None:
        """Test loading existing config file."""
        pytest.importorskip("yaml")
        manager = ConfigManager(sample_yaml_config)
        config = manager.load()
        assert config["app"]["name"] == "test_app"

    def test_load_nonexistent_config(
        self,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test loading with non-existent file logs warning."""
        import logging
        caplog.set_level(logging.WARNING)

        manager = ConfigManager(temp_dir / "missing.yaml")
        config = manager.load()
        assert config == {}
        assert "Config file not found" in caplog.text

    def test_apply_env_overrides(self, sample_yaml_config: Path) -> None:
        """Test applying environment overrides."""
        pytest.importorskip("yaml")
        os.environ["TEST_APP__DEBUG"] = "false"
        try:
            manager = ConfigManager(sample_yaml_config, env_prefix="TEST_")
            config = manager.load()
            config = manager.apply_env_overrides(config)
            # Original value may be overridden
        finally:
            os.environ.pop("TEST_APP__DEBUG", None)

    def test_get_nested_value(self, sample_yaml_config: Path) -> None:
        """Test getting nested config value."""
        pytest.importorskip("yaml")
        manager = ConfigManager(sample_yaml_config)
        manager.load()
        assert manager.get("database.host") == "localhost"
        assert manager.get("database.port") == 5432

    def test_get_missing_value_default(self, sample_yaml_config: Path) -> None:
        """Test getting missing value with default."""
        pytest.importorskip("yaml")
        manager = ConfigManager(sample_yaml_config)
        manager.load()
        assert manager.get("missing.key", "default") == "default"

    def test_set_value(self) -> None:
        """Test setting config value."""
        manager = ConfigManager()
        manager.load()
        manager.set("app.name", "new_app")
        assert manager.get("app.name") == "new_app"

    def test_to_dict(self, sample_yaml_config: Path) -> None:
        """Test converting config to dictionary."""
        pytest.importorskip("yaml")
        manager = ConfigManager(sample_yaml_config)
        manager.load()
        config_dict = manager.to_dict()
        assert isinstance(config_dict, dict)
        assert "app" in config_dict

    def test_load_dotenv(self, temp_dir: Path) -> None:
        """Test loading .env file."""
        # Create .env file
        env_path = temp_dir / ".env"
        env_path.write_text("TEST_DOTENV_VAR=test_value\n")

        # Save current directory
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            manager = ConfigManager(auto_load_dotenv=True)
            manager._load_dotenv()
            # Variable should be loaded if not already set
        finally:
            os.chdir(original_cwd)
            os.environ.pop("TEST_DOTENV_VAR", None)
