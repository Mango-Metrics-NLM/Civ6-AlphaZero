"""Component factory registry for Civ6-AlphaZero.

Provides a thread-safe registry for component factory functions, enabling
plugin-style component creation with discoverability.

Usage:
    from alpha_zero.utils.registry import default_registry, ComponentRegistry

    # Register a factory function
    @default_registry.register("network", "microciv")
    def create_microciv_network(**kwargs):
        return MicroCivNetwork(**kwargs)

    # Create a component by name
    network = default_registry.create("network", "microciv", board_size=16)

    # List available components in a category
    names = default_registry.list_components("network")

    # Check if a component is registered
    if default_registry.has_component("network", "microciv"):
        ...
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Type alias for factory callables
FactoryFunc = Callable[..., Any]


class ComponentRegistry:
    """Thread-safe registry for component factory functions.

    Organizes factories by category (e.g., "network", "mcts", "environment")
    and name (e.g., "microciv", "civ6", "simple"). Supports decorator-based
    registration and keyword-argument-based creation.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._registry: dict[str, dict[str, FactoryFunc]] = {}
        self._lock = threading.Lock()

    def register(
        self,
        category: str,
        name: str,
        *,
        allow_override: bool = False,
    ) -> Callable[[FactoryFunc], FactoryFunc]:
        """Decorator to register a factory function.

        Args:
            category: Component category (e.g., "network", "mcts").
            name: Component name within the category (e.g., "microciv").
            allow_override: If True, allow re-registration of existing names.
                Defaults to False, which raises KeyError on duplicates.

        Returns:
            Decorator that registers the wrapped function.

        Raises:
            KeyError: If the name is already registered and allow_override is False.

        Example:
            @registry.register("network", "microciv")
            def create_microciv_network(**kwargs):
                return MicroCivNetwork(**kwargs)
        """

        def decorator(func: FactoryFunc) -> FactoryFunc:
            with self._lock:
                if category not in self._registry:
                    self._registry[category] = {}

                if name in self._registry[category] and not allow_override:
                    raise KeyError(
                        f"Component '{name}' already registered in category '{category}'. "
                        f"Use allow_override=True to replace it."
                    )

                self._registry[category][name] = func
                logger.debug(f"Registered component: {category}/{name}")

            return func

        return decorator

    def register_factory(
        self,
        category: str,
        name: str,
        factory: FactoryFunc,
        *,
        allow_override: bool = False,
    ) -> None:
        """Register a factory function directly (non-decorator form).

        Args:
            category: Component category.
            name: Component name within the category.
            factory: The factory callable to register.
            allow_override: If True, allow re-registration of existing names.

        Raises:
            KeyError: If the name is already registered and allow_override is False.
        """
        with self._lock:
            if category not in self._registry:
                self._registry[category] = {}

            if name in self._registry[category] and not allow_override:
                raise KeyError(
                    f"Component '{name}' already registered in category '{category}'. "
                    f"Use allow_override=True to replace it."
                )

            self._registry[category][name] = factory
            logger.debug(f"Registered component: {category}/{name}")

    def create(self, category: str, name: str, **kwargs: Any) -> Any:
        """Create a component by looking up its factory.

        Args:
            category: Component category.
            name: Component name within the category.
            **kwargs: Arguments to pass to the factory function.

        Returns:
            The created component instance.

        Raises:
            KeyError: If the category or name is not registered.
        """
        with self._lock:
            if category not in self._registry:
                raise KeyError(
                    f"Unknown category '{category}'. "
                    f"Available: {list(self._registry.keys())}"
                )

            if name not in self._registry[category]:
                raise KeyError(
                    f"Unknown component '{name}' in category '{category}'. "
                    f"Available: {list(self._registry[category].keys())}"
                )

            factory = self._registry[category][name]

        # Call factory outside the lock to avoid holding it during construction
        logger.debug(f"Creating component: {category}/{name}")
        return factory(**kwargs)

    def list_components(self, category: str) -> list[str]:
        """List all registered component names in a category.

        Args:
            category: Component category.

        Returns:
            Sorted list of registered component names.

        Raises:
            KeyError: If the category does not exist.
        """
        with self._lock:
            if category not in self._registry:
                raise KeyError(
                    f"Unknown category '{category}'. "
                    f"Available: {list(self._registry.keys())}"
                )
            return sorted(self._registry[category].keys())

    def has_component(self, category: str, name: str) -> bool:
        """Check if a component is registered.

        Args:
            category: Component category.
            name: Component name.

        Returns:
            True if the component is registered.
        """
        with self._lock:
            return category in self._registry and name in self._registry[category]

    def list_categories(self) -> list[str]:
        """List all registered categories.

        Returns:
            Sorted list of category names.
        """
        with self._lock:
            return sorted(self._registry.keys())

    def clear(self) -> None:
        """Remove all registered components. Primarily for testing."""
        with self._lock:
            self._registry.clear()

    def __repr__(self) -> str:
        with self._lock:
            counts = {cat: len(names) for cat, names in self._registry.items()}
        return f"ComponentRegistry({counts})"


# Default global registry instance
default_registry = ComponentRegistry()
