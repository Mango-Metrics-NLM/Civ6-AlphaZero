"""Configuration base classes and utilities.

This module provides the ConfigMixin class that standardizes configuration
handling across the codebase. All config dataclasses should inherit from
ConfigMixin to gain consistent serialization and environment variable support.

Example:
    >>> from dataclasses import dataclass
    >>> from alpha_zero.core.config import ConfigMixin
    >>>
    >>> @dataclass
    ... class MyConfig(ConfigMixin):
    ...     ENV_PREFIX = "CIV6AZ_MY_"
    ...     learning_rate: float = 0.001
    ...     batch_size: int = 256
    >>>
    >>> # Create from dict
    >>> config = MyConfig.from_dict({"learning_rate": 0.01})
    >>>
    >>> # Override from environment
    >>> import os
    >>> os.environ["CIV6AZ_MY_BATCH_SIZE"] = "512"
    >>> config = config.apply_env_overrides()
"""

from __future__ import annotations

import os
from dataclasses import fields, is_dataclass
from typing import Any, ClassVar, TypeVar

T = TypeVar("T", bound="ConfigMixin")


class ConfigMixin:
    """Mixin providing standard config methods for dataclasses.

    Subclasses should:
    1. Be decorated with @dataclass
    2. Optionally define ENV_PREFIX as a class variable for environment overrides
    3. Call super().__post_init__() if overriding __post_init__

    Attributes:
        ENV_PREFIX: Class variable defining the environment variable prefix.
            Defaults to "CIV6AZ_".

    Example:
        >>> @dataclass
        ... class TrainingConfig(ConfigMixin):
        ...     ENV_PREFIX = "CIV6AZ_TRAIN_"
        ...     epochs: int = 100
        ...     lr: float = 0.001
    """

    ENV_PREFIX: ClassVar[str] = "CIV6AZ_"

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create config from dictionary, ignoring unknown keys.

        Args:
            data: Dictionary with configuration values.

        Returns:
            New config instance.

        Example:
            >>> config = MyConfig.from_dict({"lr": 0.01, "unknown": "ignored"})
        """
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")

        valid_fields = {f.name for f in fields(cls)}
        filtered = {}

        for key, value in data.items():
            if key in valid_fields:
                # Handle nested dataclass configs
                field_info = next(f for f in fields(cls) if f.name == key)
                field_type = field_info.type

                # Check if field is a ConfigMixin subclass
                if (
                    isinstance(field_type, type)
                    and issubclass(field_type, ConfigMixin)
                    and isinstance(value, dict)
                ):
                    filtered[key] = field_type.from_dict(value)
                else:
                    filtered[key] = value

        return cls(**filtered)

    @classmethod
    def from_env(cls: type[T], prefix: str | None = None) -> T:
        """Create config from environment variables.

        Environment variables are read using the pattern:
        {prefix}{FIELD_NAME} where field names are uppercased.

        Args:
            prefix: Optional override for ENV_PREFIX.

        Returns:
            New config instance with values from environment.

        Example:
            >>> # With CIV6AZ_MY_BATCH_SIZE=512 in environment
            >>> config = MyConfig.from_env()
            >>> config.batch_size
            512
        """
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")

        prefix = prefix or getattr(cls, "ENV_PREFIX", "CIV6AZ_")
        kwargs = {}

        for f in fields(cls):
            env_key = f"{prefix}{f.name.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                # Get the default value to infer type
                default_value = getattr(cls, f.name, None) if hasattr(cls, f.name) else None
                kwargs[f.name] = cls._parse_value(value, f.type, default_value)

        return cls(**kwargs) if kwargs else cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Nested ConfigMixin objects are also converted.

        Returns:
            Dictionary representation of the config.
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, ConfigMixin):
                result[f.name] = value.to_dict()
            else:
                result[f.name] = value
        return result

    def apply_env_overrides(self: T, prefix: str | None = None) -> T:
        """Apply environment variable overrides in-place.

        Modifies this config instance with values from environment variables.

        Args:
            prefix: Optional override for ENV_PREFIX.

        Returns:
            Self for method chaining.

        Example:
            >>> config = MyConfig().apply_env_overrides()
        """
        prefix = prefix or getattr(self.__class__, "ENV_PREFIX", "CIV6AZ_")

        for f in fields(self):
            env_key = f"{prefix}{f.name.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                current = getattr(self, f.name)
                parsed = self._parse_value(value, type(current), current)
                object.__setattr__(self, f.name, parsed)

        return self

    @staticmethod
    def _parse_value(value: str, target_type: type | Any, default: Any = None) -> Any:
        """Parse string value to target type.

        Args:
            value: String value from environment variable.
            target_type: Expected type (may be a type hint).
            default: Default value to infer type from if target_type is complex.

        Returns:
            Parsed value in the appropriate type.
        """
        # Handle None type hints
        if target_type is type(None):
            return None

        # Infer type from default if available
        if default is not None:
            actual_type = type(default)
        else:
            # Try to get actual type from type hint
            actual_type = target_type
            # Handle Optional, Union, etc by extracting first non-None type
            if hasattr(target_type, "__origin__"):
                args = getattr(target_type, "__args__", ())
                non_none_args = [a for a in args if a is not type(None)]
                if non_none_args:
                    actual_type = non_none_args[0]

        # Parse based on type
        if actual_type is bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif actual_type is int:
            return int(value)
        elif actual_type is float:
            return float(value)
        elif actual_type is list or (
            hasattr(actual_type, "__origin__") and actual_type.__origin__ is list
        ):
            # Simple list parsing: comma-separated values
            if not value.strip():
                return []
            return [v.strip() for v in value.split(",")]

        return value

    def merge(self: T, other: dict[str, Any] | T) -> T:
        """Merge another config or dict into this one.

        Values from other override values in self.

        Args:
            other: Config or dict to merge from.

        Returns:
            Self for method chaining.
        """
        if isinstance(other, ConfigMixin):
            other = other.to_dict()

        for key, value in other.items():
            if hasattr(self, key):
                object.__setattr__(self, key, value)

        return self

    def __repr__(self) -> str:
        """Return string representation of config."""
        if not is_dataclass(self):
            return super().__repr__()

        field_strs = []
        for f in fields(self):
            value = getattr(self, f.name)
            field_strs.append(f"{f.name}={value!r}")

        return f"{self.__class__.__name__}({', '.join(field_strs)})"
