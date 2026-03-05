"""Mixin classes for common functionality.

These mixins provide reusable implementations of common patterns
like serialization and action masking. Use them via multiple inheritance
to add functionality to your classes.

Example:
    >>> from dataclasses import dataclass
    >>> from alpha_zero.core.mixins import SerializableMixin, ActionMaskMixin
    >>>
    >>> @dataclass
    ... class MyConfig(SerializableMixin):
    ...     value: int = 10
    >>>
    >>> config = MyConfig()
    >>> data = config.to_dict()
    >>> restored = MyConfig.from_dict(data)
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor

T = TypeVar("T")


class SerializableMixin:
    """Mixin providing to_dict/from_dict serialization.

    Works with dataclasses to provide automatic serialization.
    Override _serialize_field and _deserialize_field for custom handling.

    Example:
        >>> @dataclass
        ... class Config(SerializableMixin):
        ...     learning_rate: float = 0.001
        ...     epochs: int = 100
        >>>
        >>> config = Config()
        >>> data = config.to_dict()  # {'learning_rate': 0.001, 'epochs': 100}
        >>> restored = Config.from_dict(data)
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Handles nested SerializableMixin objects, numpy arrays, and enums.

        Returns:
            Dictionary representation of the object.
        """
        if not is_dataclass(self):
            raise TypeError(f"{type(self).__name__} must be a dataclass to use SerializableMixin")

        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            result[f.name] = self._serialize_field(f.name, value)
        return result

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create instance from dictionary.

        Args:
            data: Dictionary with field values.

        Returns:
            New instance of the class.
        """
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass to use SerializableMixin")

        kwargs = {}
        for f in fields(cls):
            if f.name in data:
                value = cls._deserialize_field(f.name, data[f.name], f.type)  # type: ignore[attr-defined]
                kwargs[f.name] = value
        return cls(**kwargs)

    def _serialize_field(self, name: str, value: Any) -> Any:
        """Serialize a single field value.

        Override for custom serialization of specific fields.

        Args:
            name: Field name.
            value: Field value.

        Returns:
            Serialized value.
        """
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if hasattr(value, "value"):  # Enum
            return value.value
        if isinstance(value, (list, tuple)):
            return [self._serialize_field(name, v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_field(name, v) for k, v in value.items()}
        return value

    @classmethod
    def _deserialize_field(cls, name: str, value: Any, field_type: type | None = None) -> Any:
        """Deserialize a single field value.

        Override for custom deserialization of specific fields.

        Args:
            name: Field name.
            value: Serialized value.
            field_type: Expected field type (may be None).

        Returns:
            Deserialized value.
        """
        # Check if field_type is a SerializableMixin subclass
        if (
            field_type is not None
            and isinstance(field_type, type)
            and issubclass(field_type, SerializableMixin)
            and isinstance(value, dict)
        ):
            return field_type.from_dict(value)
        return value


class ActionMaskMixin:
    """Mixin providing action masking utilities.

    Action masking is essential for MCTS and policy networks to ensure
    only valid actions are considered during search and training.

    Example:
        >>> class MyPolicy(ActionMaskMixin):
        ...     def get_policy(self, state, mask):
        ...         logits = self.network(state)
        ...         return self.apply_action_mask(logits, mask)
    """

    MASK_VALUE: float = -1e9

    def apply_action_mask(
        self,
        logits: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Apply action mask to logits.

        Sets invalid actions to a very negative value so they have
        near-zero probability after softmax.

        Args:
            logits: Policy logits of shape (B, A).
            mask: Valid action mask of shape (B, A) where 1 = valid, 0 = invalid.

        Returns:
            Masked logits with invalid actions set to MASK_VALUE.
        """
        return logits.masked_fill(mask == 0, self.MASK_VALUE)

    def get_valid_action_indices(self, mask: np.ndarray) -> np.ndarray:
        """Get indices of valid actions from mask.

        Args:
            mask: Action mask array.

        Returns:
            Array of valid action indices.
        """
        return np.where(mask > 0)[0]

    def sample_valid_action(
        self,
        probs: np.ndarray,
        mask: np.ndarray,
    ) -> int:
        """Sample action from valid actions only.

        Masks the probability distribution and samples from valid actions.
        Falls back to uniform sampling if all probs are zero.

        Args:
            probs: Action probabilities.
            mask: Valid action mask.

        Returns:
            Sampled action index.
        """
        masked_probs = probs * mask
        sum_probs = masked_probs.sum()

        if sum_probs > 0:
            masked_probs = masked_probs / sum_probs
            return int(np.random.choice(len(probs), p=masked_probs))
        else:
            # Fallback to uniform over valid actions
            valid_indices = self.get_valid_action_indices(mask)
            if len(valid_indices) > 0:
                return int(np.random.choice(valid_indices))
            return 0

    def argmax_valid_action(
        self,
        probs: np.ndarray,
        mask: np.ndarray,
    ) -> int:
        """Select highest probability valid action.

        Args:
            probs: Action probabilities.
            mask: Valid action mask.

        Returns:
            Action index with highest probability among valid actions.
        """
        masked_probs = probs * mask
        return int(np.argmax(masked_probs))


class LoggingMixin:
    """Mixin providing structured logging utilities.

    Provides methods for debug logging with consistent formatting.
    Classes using this mixin should set self._logger in __init__.

    Example:
        >>> import logging
        >>> class MyClass(LoggingMixin):
        ...     def __init__(self):
        ...         self._logger = logging.getLogger(__name__)
        ...
        ...     def process(self):
        ...         self.log_debug("Processing", step=1, value=0.5)
    """

    _logger: Any = None  # Should be set to logging.Logger in subclass

    def log_debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with structured fields.

        Args:
            message: Log message.
            **kwargs: Additional fields to include in log.
        """
        if self._logger is None:
            return

        if kwargs:
            fields_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            self._logger.debug(f"{message}: {fields_str}")
        else:
            self._logger.debug(message)

    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message with structured fields.

        Args:
            message: Log message.
            **kwargs: Additional fields to include in log.
        """
        if self._logger is None:
            return

        if kwargs:
            fields_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            self._logger.info(f"{message}: {fields_str}")
        else:
            self._logger.info(message)

    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with structured fields.

        Args:
            message: Log message.
            **kwargs: Additional fields to include in log.
        """
        if self._logger is None:
            return

        if kwargs:
            fields_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            self._logger.warning(f"{message}: {fields_str}")
        else:
            self._logger.warning(message)


class TensorShapeMixin:
    """Mixin providing tensor shape validation utilities.

    Helps validate tensor shapes during debugging and testing.

    Example:
        >>> class MyNetwork(TensorShapeMixin, nn.Module):
        ...     def forward(self, x):
        ...         self.assert_shape(x, (None, 3, 32, 32), "input")
        ...         # ... process x ...
    """

    def assert_shape(
        self,
        tensor: Tensor | np.ndarray,
        expected: tuple[int | None, ...],
        name: str = "tensor",
    ) -> None:
        """Assert tensor has expected shape.

        None values in expected shape match any dimension.

        Args:
            tensor: Tensor to check.
            expected: Expected shape tuple (None = any value).
            name: Name for error message.

        Raises:
            AssertionError: If shape doesn't match.
        """
        actual = tensor.shape
        if len(actual) != len(expected):
            raise AssertionError(
                f"{name} has {len(actual)} dimensions, expected {len(expected)}. "
                f"Shape: {actual}, expected: {expected}"
            )

        for i, (a, e) in enumerate(zip(actual, expected, strict=False)):
            if e is not None and a != e:
                raise AssertionError(
                    f"{name} dim {i} is {a}, expected {e}. Shape: {actual}, expected: {expected}"
                )

    def get_shape_str(self, tensor: Tensor | np.ndarray) -> str:
        """Get string representation of tensor shape.

        Args:
            tensor: Tensor to describe.

        Returns:
            String like "(32, 64, 128)".
        """
        return str(tuple(tensor.shape))
