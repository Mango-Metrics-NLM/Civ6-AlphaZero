"""Environment registry for the Strategic Planning SDK.

Provides a plugin-based registry for strategic environments, enabling
domain experts to register custom environments that integrate with
the planning engine.

Usage:
    from alpha_zero.strategic.registry import EnvironmentRegistry

    # Register a custom environment
    EnvironmentRegistry.register("my_domain", MyDomainEnv)

    # Create an environment instance
    env = EnvironmentRegistry.create("my_domain", config=my_config)

    # List available environments
    print(EnvironmentRegistry.list_environments())
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

# Lazy flag to track whether built-in environments have been registered
_builtins_registered = False


def _register_builtins() -> None:
    """Register built-in environments on first use.

    Uses lazy registration to avoid circular imports and unnecessary
    loading of environment modules.
    """
    global _builtins_registered
    if _builtins_registered:
        return
    _builtins_registered = True

    try:
        from alpha_zero.strategic.envs.resource_allocation import (
            ResourceAllocationEnv,
        )

        EnvironmentRegistry.register(
            "resource_allocation",
            ResourceAllocationEnv,
            description="Generic resource allocation across regions over time.",
        )
    except ImportError:
        logger.debug("ResourceAllocationEnv not available")

    try:
        from alpha_zero.strategic.envs.climate_negotiation import (
            ClimateNegotiationEnv,
        )

        EnvironmentRegistry.register(
            "climate_negotiation",
            ClimateNegotiationEnv,
            description="Multi-nation climate policy negotiation.",
        )
    except ImportError:
        logger.debug("ClimateNegotiationEnv not available")

    try:
        from alpha_zero.strategic.envs.pandemic_response import PandemicResponseEnv

        EnvironmentRegistry.register(
            "pandemic_response",
            PandemicResponseEnv,
            description="Pandemic resource allocation with SIR epidemiology.",
        )
    except ImportError:
        logger.debug("PandemicResponseEnv not available")

    try:
        from alpha_zero.strategic.envs.supply_chain import SupplyChainEnv

        EnvironmentRegistry.register(
            "supply_chain",
            SupplyChainEnv,
            description="Supply chain disruption management.",
        )
    except ImportError:
        logger.debug("SupplyChainEnv not available")


class EnvironmentRegistry:
    """Registry for strategic environment classes.

    Maintains a mapping from environment names to their implementation
    classes. Supports dynamic registration for extensibility.

    Built-in environments are registered lazily on first access to
    avoid circular imports.

    Example:
        >>> from alpha_zero.strategic.registry import EnvironmentRegistry
        >>> env = EnvironmentRegistry.create("resource_allocation")
        >>> print(env.num_objectives)
        3
    """

    _registry: ClassVar[dict[str, type]] = {}
    _descriptions: ClassVar[dict[str, str]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        env_class: type,
        description: str = "",
    ) -> None:
        """Register an environment class.

        Args:
            name: Unique name for the environment.
            env_class: Class implementing StrategicEnvironment protocol.
            description: Optional human-readable description.

        Raises:
            ValueError: If name is already registered (use overwrite=True).
        """
        if name in cls._registry:
            logger.warning(
                "Overwriting existing environment registration: %s", name
            )
        cls._registry[name] = env_class
        cls._descriptions[name] = description
        logger.debug("Registered environment: %s (%s)", name, env_class.__name__)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove an environment from the registry.

        Args:
            name: Name of the environment to remove.

        Raises:
            KeyError: If environment is not registered.
        """
        if name not in cls._registry:
            raise KeyError(f"Environment '{name}' is not registered")
        del cls._registry[name]
        cls._descriptions.pop(name, None)
        logger.debug("Unregistered environment: %s", name)

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Any:
        """Create an environment instance by name.

        Automatically registers built-in environments on first call.

        Args:
            name: Registered environment name.
            **kwargs: Arguments passed to the environment constructor.

        Returns:
            Environment instance.

        Raises:
            KeyError: If environment is not registered.
        """
        _register_builtins()
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(
                f"Environment '{name}' not registered. "
                f"Available: [{available}]"
            )
        env_class = cls._registry[name]
        logger.debug("Creating environment: %s", name)
        return env_class(**kwargs)

    @classmethod
    def list_environments(cls) -> list[str]:
        """List all registered environment names.

        Returns:
            Sorted list of environment names.
        """
        _register_builtins()
        return sorted(cls._registry.keys())

    @classmethod
    def get_description(cls, name: str) -> str:
        """Get the description for a registered environment.

        Args:
            name: Environment name.

        Returns:
            Description string, or empty string if not set.
        """
        _register_builtins()
        return cls._descriptions.get(name, "")

    @classmethod
    def get_info(cls) -> dict[str, str]:
        """Get info for all registered environments.

        Returns:
            Dictionary mapping environment names to descriptions.
        """
        _register_builtins()
        return {name: cls._descriptions.get(name, "") for name in sorted(cls._registry.keys())}

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an environment is registered.

        Args:
            name: Environment name.

        Returns:
            True if registered.
        """
        _register_builtins()
        return name in cls._registry

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Primarily for testing."""
        cls._registry.clear()
        cls._descriptions.clear()
        global _builtins_registered
        _builtins_registered = False


__all__ = ["EnvironmentRegistry"]
