"""
Component Factory System.

This module provides a flexible, dependency-injection ready factory system
for creating and managing components with configuration-driven instantiation.

Example:
    >>> from template_system.factories import ComponentFactory, FactoryRegistry
    >>> registry = FactoryRegistry()
    >>> registry.register("logger", LoggerFactory)
    >>> logger = registry.create("logger", name="app")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    TypeVar,
)

if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


T = TypeVar("T")
ConfigT = TypeVar("ConfigT")


class ComponentError(Exception):
    """Base exception for component errors."""

    pass


class ComponentNotFoundError(ComponentError):
    """Raised when a component is not found in the registry."""

    pass


class ComponentCreationError(ComponentError):
    """Raised when component creation fails."""

    pass


@dataclass
class ComponentMetadata:
    """Metadata for a registered component.

    Attributes:
        name: Unique component name.
        description: Human-readable description.
        version: Component version.
        dependencies: List of dependency component names.
        tags: Tags for categorization.
        singleton: Whether only one instance should exist.
    """

    name: str
    description: str = ""
    version: str = "1.0.0"
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    singleton: bool = False


class ComponentFactory(ABC, Generic[T, ConfigT]):
    """Abstract base class for component factories.

    Factories encapsulate the creation logic for components, supporting
    configuration-driven instantiation and dependency injection.
    """

    metadata: ClassVar[ComponentMetadata]

    @abstractmethod
    def create(self, config: ConfigT, **kwargs: Any) -> T:
        """Create a component instance.

        Args:
            config: Configuration for the component.
            **kwargs: Additional creation arguments.

        Returns:
            Created component instance.
        """
        ...

    def validate_config(self, config: ConfigT) -> list[str]:
        """Validate configuration for component creation.

        Args:
            config: Configuration to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        return []

    def get_dependencies(self) -> list[str]:
        """Get list of component dependencies.

        Returns:
            List of dependency component names.
        """
        if hasattr(self, "metadata"):
            return self.metadata.dependencies
        return []


class SimpleFactory(ComponentFactory[T, dict[str, Any]]):
    """Simple factory for classes with constructor injection.

    This factory creates instances by passing configuration directly
    to the class constructor.
    """

    def __init__(
        self,
        component_class: type[T],
        metadata: ComponentMetadata | None = None,
    ) -> None:
        """Initialize the factory.

        Args:
            component_class: Class to instantiate.
            metadata: Component metadata.
        """
        self.component_class = component_class
        self.metadata = metadata or ComponentMetadata(
            name=component_class.__name__,
            description=component_class.__doc__ or "",
        )

    def create(self, config: dict[str, Any], **kwargs: Any) -> T:
        """Create component by passing config to constructor.

        Args:
            config: Configuration dictionary.
            **kwargs: Additional constructor arguments.

        Returns:
            Created component instance.
        """
        merged = {**config, **kwargs}
        try:
            return self.component_class(**merged)
        except Exception as e:
            raise ComponentCreationError(
                f"Failed to create {self.component_class.__name__}: {e}"
            ) from e


class FunctionFactory(ComponentFactory[T, ConfigT]):
    """Factory that wraps a creation function.

    This factory delegates creation to a provided function, useful for
    complex creation logic or when integrating with existing code.
    """

    def __init__(
        self,
        create_fn: Callable[[ConfigT], T],
        metadata: ComponentMetadata | None = None,
        validator: Callable[[ConfigT], list[str]] | None = None,
    ) -> None:
        """Initialize the factory.

        Args:
            create_fn: Function to create component.
            metadata: Component metadata.
            validator: Optional validation function.
        """
        self.create_fn = create_fn
        self.metadata = metadata or ComponentMetadata(
            name=create_fn.__name__,
            description=create_fn.__doc__ or "",
        )
        self._validator = validator

    def create(self, config: ConfigT, **kwargs: Any) -> T:
        """Create component using the wrapped function.

        Args:
            config: Configuration for creation.
            **kwargs: Ignored (for interface compatibility).

        Returns:
            Created component instance.
        """
        try:
            return self.create_fn(config)
        except Exception as e:
            raise ComponentCreationError(
                f"Failed to create via {self.create_fn.__name__}: {e}"
            ) from e

    def validate_config(self, config: ConfigT) -> list[str]:
        """Validate configuration using optional validator.

        Args:
            config: Configuration to validate.

        Returns:
            List of validation errors.
        """
        if self._validator:
            return self._validator(config)
        return []


class FactoryRegistry:
    """Registry for component factories.

    This class maintains a registry of factories and handles component
    creation with automatic dependency resolution.

    Example:
        >>> registry = FactoryRegistry()
        >>> registry.register("database", DatabaseFactory())
        >>> registry.register("cache", CacheFactory())
        >>> db = registry.create("database", config={"host": "localhost"})
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._factories: dict[str, ComponentFactory[Any, Any]] = {}
        self._singletons: dict[str, Any] = {}
        self._logger = logger.getChild(self.__class__.__name__)

    def register(
        self,
        name: str,
        factory: ComponentFactory[T, ConfigT] | type[T],
        metadata: ComponentMetadata | None = None,
    ) -> None:
        """Register a factory or class.

        Args:
            name: Unique component name.
            factory: Factory instance or class to wrap.
            metadata: Optional metadata (for class registration).
        """
        if isinstance(factory, type):
            # Wrap class in SimpleFactory
            factory = SimpleFactory(factory, metadata)

        self._factories[name] = factory
        self._logger.debug(f"Registered factory: {name}")

    def unregister(self, name: str) -> None:
        """Unregister a factory.

        Args:
            name: Component name to unregister.
        """
        self._factories.pop(name, None)
        self._singletons.pop(name, None)
        self._logger.debug(f"Unregistered factory: {name}")

    def get_factory(self, name: str) -> ComponentFactory[Any, Any]:
        """Get a factory by name.

        Args:
            name: Component name.

        Returns:
            Factory for the component.

        Raises:
            ComponentNotFoundError: If factory not registered.
        """
        factory = self._factories.get(name)
        if not factory:
            raise ComponentNotFoundError(f"No factory registered for: {name}")
        return factory

    def create(
        self,
        name: str,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Create a component instance.

        Args:
            name: Component name.
            config: Configuration for creation.
            **kwargs: Additional creation arguments.

        Returns:
            Created component instance.

        Raises:
            ComponentNotFoundError: If factory not registered.
            ComponentCreationError: If creation fails.
        """
        factory = self.get_factory(name)

        # Check for singleton
        if hasattr(factory, "metadata") and factory.metadata.singleton:
            if name in self._singletons:
                return self._singletons[name]

        # Resolve dependencies first
        dependencies = self._resolve_dependencies(factory, config, kwargs)
        kwargs.update(dependencies)

        # Validate configuration
        errors = factory.validate_config(config)
        if errors:
            raise ComponentCreationError(f"Configuration validation failed for {name}: {errors}")

        # Create instance
        instance = factory.create(config or {}, **kwargs)

        # Store singleton
        if hasattr(factory, "metadata") and factory.metadata.singleton:
            self._singletons[name] = instance

        self._logger.debug(f"Created component: {name}")
        return instance

    def _resolve_dependencies(
        self,
        factory: ComponentFactory[Any, Any],
        config: Any,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve dependencies for a factory.

        Args:
            factory: Factory to resolve dependencies for.
            config: Configuration being used.
            kwargs: Existing kwargs.

        Returns:
            Dictionary of resolved dependencies.
        """
        dependencies: dict[str, Any] = {}

        for dep_name in factory.get_dependencies():
            if dep_name in kwargs:
                continue  # Already provided

            try:
                # Recursively create dependency
                dependencies[dep_name] = self.create(dep_name)
            except ComponentNotFoundError:
                self._logger.warning(f"Optional dependency not found: {dep_name}")

        return dependencies

    def list_components(self) -> list[str]:
        """List all registered component names.

        Returns:
            List of component names.
        """
        return list(self._factories.keys())

    def get_metadata(self, name: str) -> ComponentMetadata | None:
        """Get metadata for a component.

        Args:
            name: Component name.

        Returns:
            Component metadata, or None if not found.
        """
        factory = self._factories.get(name)
        if factory and hasattr(factory, "metadata"):
            return factory.metadata
        return None

    def clear(self) -> None:
        """Clear all registrations and singletons."""
        self._factories.clear()
        self._singletons.clear()

    def find_by_tag(self, tag: str) -> list[str]:
        """Find components by tag.

        Args:
            tag: Tag to search for.

        Returns:
            List of component names with the tag.
        """
        results = []
        for name, factory in self._factories.items():
            if hasattr(factory, "metadata") and tag in factory.metadata.tags:
                results.append(name)
        return results


class LifecycleManager:
    """Manages component lifecycle (creation, initialization, shutdown).

    This class provides hooks for managing components throughout their
    lifecycle, including ordered initialization and graceful shutdown.
    """

    def __init__(self, registry: FactoryRegistry) -> None:
        """Initialize the lifecycle manager.

        Args:
            registry: Factory registry to use.
        """
        self.registry = registry
        self._instances: dict[str, Any] = {}
        self._init_order: list[str] = []
        self._logger = logger.getChild(self.__class__.__name__)

    def create_and_initialize(
        self,
        name: str,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Create and initialize a component.

        Args:
            name: Component name.
            config: Configuration for creation.
            **kwargs: Additional creation arguments.

        Returns:
            Initialized component instance.
        """
        instance = self.registry.create(name, config, **kwargs)
        self._instances[name] = instance
        self._init_order.append(name)

        # Call initialize if available
        if hasattr(instance, "initialize"):
            self._logger.debug(f"Initializing component: {name}")
            instance.initialize()

        return instance

    def shutdown(self, name: str | None = None) -> None:
        """Shutdown components.

        Args:
            name: Specific component to shutdown, or None for all.
        """
        if name:
            self._shutdown_one(name)
        else:
            # Shutdown in reverse initialization order
            for component_name in reversed(self._init_order):
                self._shutdown_one(component_name)
            self._instances.clear()
            self._init_order.clear()

    def _shutdown_one(self, name: str) -> None:
        """Shutdown a single component.

        Args:
            name: Component name.
        """
        instance = self._instances.get(name)
        if instance and hasattr(instance, "shutdown"):
            self._logger.debug(f"Shutting down component: {name}")
            try:
                instance.shutdown()
            except Exception as e:
                self._logger.error(f"Error shutting down {name}: {e}")

    def get(self, name: str) -> Any | None:
        """Get an initialized component instance.

        Args:
            name: Component name.

        Returns:
            Component instance, or None if not created.
        """
        return self._instances.get(name)

    def get_all(self) -> dict[str, Any]:
        """Get all initialized component instances.

        Returns:
            Dictionary of component name to instance.
        """
        return self._instances.copy()


# Global registry instance
_global_registry: FactoryRegistry | None = None


def get_global_registry() -> FactoryRegistry:
    """Get the global factory registry.

    Returns:
        Global FactoryRegistry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = FactoryRegistry()
    return _global_registry


def register_component(
    name: str,
    factory_or_class: ComponentFactory[Any, Any] | type[Any],
    metadata: ComponentMetadata | None = None,
) -> None:
    """Register a component in the global registry.

    Args:
        name: Component name.
        factory_or_class: Factory or class to register.
        metadata: Optional metadata.
    """
    get_global_registry().register(name, factory_or_class, metadata)


def create_component(
    name: str,
    config: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a component from the global registry.

    Args:
        name: Component name.
        config: Configuration for creation.
        **kwargs: Additional creation arguments.

    Returns:
        Created component instance.
    """
    return get_global_registry().create(name, config, **kwargs)


# Decorator for registering classes as components
def component(
    name: str | None = None,
    singleton: bool = False,
    dependencies: list[str] | None = None,
    tags: list[str] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a class as a component.

    Args:
        name: Component name (defaults to class name).
        singleton: Whether to create only one instance.
        dependencies: List of dependency names.
        tags: Tags for categorization.

    Returns:
        Decorator function.

    Example:
        >>> @component(name="my_service", singleton=True)
        ... class MyService:
        ...     def __init__(self, config: dict):
        ...         pass
    """

    def decorator(cls: type[T]) -> type[T]:
        component_name = name or cls.__name__
        metadata = ComponentMetadata(
            name=component_name,
            description=cls.__doc__ or "",
            singleton=singleton,
            dependencies=dependencies or [],
            tags=tags or [],
        )
        register_component(component_name, cls, metadata)
        return cls

    return decorator
