"""
Configuration Management Module.

This module provides a flexible, hierarchical configuration system with:
- YAML/JSON file loading
- Environment variable overrides
- Pydantic validation
- Template variable substitution

Example:
    >>> from template_system.config import ConfigManager, TemplateConfig
    >>> manager = ConfigManager("configs/project.yaml")
    >>> config = manager.load()
    >>> config = config.apply_env_overrides()
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    TypeVar,
)

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from pydantic import BaseModel, Field, field_validator

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


logger = logging.getLogger(__name__)


# Template variable pattern: {VARIABLE_NAME}
TEMPLATE_VAR_PATTERN = re.compile(r"\{([A-Z_][A-Z0-9_]*)\}")


@dataclass
class EnvironmentConfig:
    """Configuration for environment variable handling.

    Attributes:
        prefix: Prefix for environment variables (e.g., "MYAPP_").
        separator: Separator for nested keys (e.g., "__").
        case_sensitive: Whether env var names are case-sensitive.
        load_dotenv: Whether to load from .env file.
        dotenv_path: Path to .env file.
    """

    prefix: str = ""
    separator: str = "__"
    case_sensitive: bool = False
    load_dotenv: bool = True
    dotenv_path: Path | None = None

    def get_env_key(self, config_key: str) -> str:
        """Convert config key to environment variable name.

        Args:
            config_key: Configuration key (e.g., "database.host").

        Returns:
            Environment variable name (e.g., "MYAPP_DATABASE__HOST").
        """
        key = config_key.replace(".", self.separator)
        if not self.case_sensitive:
            key = key.upper()
        return f"{self.prefix}{key}"

    def parse_env_key(self, env_key: str) -> str | None:
        """Parse environment variable name to config key.

        Args:
            env_key: Environment variable name.

        Returns:
            Config key, or None if not matching prefix.
        """
        if not env_key.startswith(self.prefix):
            return None
        key = env_key[len(self.prefix):]
        key = key.replace(self.separator, ".")
        if not self.case_sensitive:
            key = key.lower()
        return key


@dataclass
class TemplateVariable:
    """Definition of a template variable.

    Attributes:
        name: Variable name (e.g., "PACKAGE_NAME").
        description: Human-readable description.
        default: Default value if not provided.
        required: Whether the variable must be provided.
        validator: Optional validation function.
        example: Example value for documentation.
    """

    name: str
    description: str
    default: Any = None
    required: bool = False
    validator: Callable[[Any], bool] | None = None
    example: str = ""

    def validate(self, value: Any) -> bool:
        """Validate a value for this variable.

        Args:
            value: Value to validate.

        Returns:
            True if valid, False otherwise.
        """
        if self.validator:
            return self.validator(value)
        return True


# Standard template variables
STANDARD_TEMPLATE_VARIABLES: list[TemplateVariable] = [
    TemplateVariable(
        name="SYSTEM_NAME",
        description="Full system name",
        example="Distributed Task Queue",
        required=True,
    ),
    TemplateVariable(
        name="PROJECT_NAME",
        description="Project directory name",
        example="task-queue",
        required=True,
    ),
    TemplateVariable(
        name="PACKAGE_NAME",
        description="Python package name",
        example="task_queue",
        required=True,
    ),
    TemplateVariable(
        name="COVERAGE_THRESHOLD",
        description="Minimum test coverage percentage",
        default="80",
        example="80",
    ),
    TemplateVariable(
        name="FRAMEWORK",
        description="Web framework",
        default="FastAPI",
        example="FastAPI",
    ),
    TemplateVariable(
        name="PORT",
        description="Default port",
        default="8000",
        example="8000",
    ),
    TemplateVariable(
        name="DB_TECH",
        description="Database technology",
        default="PostgreSQL",
        example="PostgreSQL",
    ),
    TemplateVariable(
        name="DB_PORT",
        description="Database port",
        default="5432",
        example="5432",
    ),
    TemplateVariable(
        name="DATE",
        description="Current date",
        example="2026-01-23",
    ),
]


@dataclass
class TemplateConfig:
    """Configuration for template instantiation.

    Attributes:
        variables: Map of variable names to values.
        template_path: Path to template file.
        output_path: Path for output file.
        validate_required: Whether to validate required variables.
    """

    variables: dict[str, Any] = field(default_factory=dict)
    template_path: Path | None = None
    output_path: Path | None = None
    validate_required: bool = True

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable value with fallback to default.

        Args:
            name: Variable name.
            default: Default value if not set.

        Returns:
            Variable value or default.
        """
        return self.variables.get(name, default)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable value.

        Args:
            name: Variable name.
            value: Value to set.
        """
        self.variables[name] = value

    def substitute(self, text: str) -> str:
        """Substitute template variables in text.

        Args:
            text: Text with {VARIABLE} placeholders.

        Returns:
            Text with variables substituted.
        """
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            value = self.variables.get(var_name)
            if value is None:
                logger.warning(f"Template variable not set: {var_name}")
                return match.group(0)  # Keep original
            return str(value)

        return TEMPLATE_VAR_PATTERN.sub(replacer, text)

    def validate(
        self,
        variable_definitions: list[TemplateVariable] | None = None,
    ) -> list[str]:
        """Validate template configuration.

        Args:
            variable_definitions: Variable definitions to validate against.

        Returns:
            List of validation errors.
        """
        errors: list[str] = []
        definitions = variable_definitions or STANDARD_TEMPLATE_VARIABLES

        for var_def in definitions:
            value = self.variables.get(var_def.name)

            if var_def.required and value is None:
                if self.validate_required:
                    errors.append(f"Required variable not set: {var_def.name}")
                continue

            if value is not None and not var_def.validate(value):
                errors.append(f"Invalid value for {var_def.name}: {value}")

        return errors


T = TypeVar("T")


class ConfigLoader(Generic[T]):
    """Generic configuration loader with multiple format support.

    This class loads configuration from various sources (YAML, JSON, env)
    and validates against a schema.
    """

    _loaders: ClassVar[dict[str, Callable[[Path], dict[str, Any]]]] = {}

    def __init__(
        self,
        schema: type[T] | None = None,
        env_config: EnvironmentConfig | None = None,
    ) -> None:
        """Initialize the loader.

        Args:
            schema: Pydantic model class for validation.
            env_config: Environment variable configuration.
        """
        self.schema = schema
        self.env_config = env_config or EnvironmentConfig()
        self._register_default_loaders()

    def _register_default_loaders(self) -> None:
        """Register default file format loaders."""
        import json

        self._loaders[".json"] = lambda p: json.loads(p.read_text())

        if YAML_AVAILABLE:
            self._loaders[".yaml"] = lambda p: yaml.safe_load(p.read_text())
            self._loaders[".yml"] = self._loaders[".yaml"]

    def load_file(self, path: Path | str) -> dict[str, Any]:
        """Load configuration from a file.

        Args:
            path: Path to configuration file.

        Returns:
            Configuration dictionary.

        Raises:
            ValueError: If file format is not supported.
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        suffix = path.suffix.lower()
        loader = self._loaders.get(suffix)
        if not loader:
            raise ValueError(f"Unsupported config format: {suffix}")

        logger.debug(f"Loading config from {path}")
        return loader(path)

    def load_env(self) -> dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary from environment.
        """
        config: dict[str, Any] = {}

        for key, value in os.environ.items():
            config_key = self.env_config.parse_env_key(key)
            if config_key:
                # Handle nested keys
                self._set_nested(config, config_key, self._parse_value(value))

        return config

    def _set_nested(
        self,
        config: dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        """Set a nested configuration value.

        Args:
            config: Configuration dictionary.
            key: Dot-separated key path.
            value: Value to set.
        """
        parts = key.split(".")
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def _parse_value(self, value: str) -> Any:
        """Parse a string value to appropriate type.

        Args:
            value: String value to parse.

        Returns:
            Parsed value (int, float, bool, or str).
        """
        # Boolean
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # String
        return value

    def merge(self, *configs: dict[str, Any]) -> dict[str, Any]:
        """Merge multiple configuration dictionaries.

        Later configs override earlier ones. Nested dicts are merged recursively.

        Args:
            configs: Configuration dictionaries to merge.

        Returns:
            Merged configuration.
        """
        result: dict[str, Any] = {}
        for config in configs:
            self._deep_merge(result, config)
        return result

    def _deep_merge(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> None:
        """Deep merge override into base.

        Args:
            base: Base dictionary (modified in place).
            override: Override dictionary.
        """
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def validate(self, config: dict[str, Any]) -> T:
        """Validate configuration against schema.

        Args:
            config: Configuration dictionary.

        Returns:
            Validated configuration object.

        Raises:
            ValueError: If no schema defined.
            ValidationError: If validation fails.
        """
        if not self.schema:
            raise ValueError("No schema defined for validation")

        if PYDANTIC_AVAILABLE and issubclass(self.schema, BaseModel):
            return self.schema.model_validate(config)

        # Fallback for non-Pydantic
        return self.schema(**config)  # type: ignore[return-value]


class ConfigManager:
    """High-level configuration manager.

    This class provides a unified interface for loading, merging, and
    validating configuration from multiple sources.

    Example:
        >>> manager = ConfigManager("config.yaml")
        >>> config = manager.load()
        >>> config = manager.apply_env_overrides(config)
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        env_prefix: str = "",
        auto_load_dotenv: bool = True,
    ) -> None:
        """Initialize the manager.

        Args:
            config_path: Path to main configuration file.
            env_prefix: Prefix for environment variables.
            auto_load_dotenv: Whether to auto-load .env file.
        """
        self.config_path = Path(config_path) if config_path else None
        self.env_config = EnvironmentConfig(
            prefix=env_prefix,
            load_dotenv=auto_load_dotenv,
        )
        self.loader: ConfigLoader[dict[str, Any]] = ConfigLoader(
            env_config=self.env_config
        )
        self._config: dict[str, Any] = {}
        self._logger = logger.getChild(self.__class__.__name__)

        if auto_load_dotenv:
            self._load_dotenv()

    def _load_dotenv(self) -> None:
        """Load environment variables from .env file."""
        dotenv_path = self.env_config.dotenv_path or Path(".env")
        if dotenv_path.exists():
            try:
                # Simple .env parser (no external dependency)
                with open(dotenv_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, value = line.partition("=")
                            key = key.strip()
                            value = value.strip().strip("'\"")
                            if key not in os.environ:
                                os.environ[key] = value
                self._logger.debug(f"Loaded .env from {dotenv_path}")
            except Exception as e:
                self._logger.warning(f"Failed to load .env: {e}")

    def load(self) -> dict[str, Any]:
        """Load configuration from file.

        Returns:
            Configuration dictionary.
        """
        if self.config_path and self.config_path.exists():
            self._config = self.loader.load_file(self.config_path)
            self._logger.info(f"Loaded config from {self.config_path}")
        else:
            self._config = {}
            if self.config_path:
                self._logger.warning(f"Config file not found: {self.config_path}")
        return self._config

    def apply_env_overrides(
        self,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply environment variable overrides.

        Args:
            config: Configuration to apply overrides to.

        Returns:
            Configuration with env overrides applied.
        """
        base = config or self._config
        env_config = self.loader.load_env()
        return self.loader.merge(base, env_config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Dot-separated key path.
            default: Default value if not found.

        Returns:
            Configuration value or default.
        """
        parts = key.split(".")
        current: Any = self._config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Dot-separated key path.
            value: Value to set.
        """
        parts = key.split(".")
        current = self._config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def to_dict(self) -> dict[str, Any]:
        """Get configuration as dictionary.

        Returns:
            Configuration dictionary.
        """
        return self._config.copy()


# Pydantic base configuration model
if PYDANTIC_AVAILABLE:

    class BaseConfig(BaseModel):
        """Base configuration model with common settings."""

        debug: bool = Field(default=False, description="Enable debug mode")
        log_level: str = Field(default="INFO", description="Logging level")

        @field_validator("log_level")
        @classmethod
        def validate_log_level(cls, v: str) -> str:
            """Validate log level is valid."""
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if v.upper() not in valid_levels:
                raise ValueError(f"Invalid log level: {v}")
            return v.upper()

        model_config = {
            "extra": "allow",
            "env_prefix": "",
        }

else:
    # Fallback for non-Pydantic environments
    @dataclass
    class BaseConfig:  # type: ignore[no-redef]
        """Base configuration model with common settings."""

        debug: bool = False
        log_level: str = "INFO"
