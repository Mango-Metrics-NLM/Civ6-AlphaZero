"""Environment registry for discovering and instantiating environments."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.core.exceptions import (
    DuplicateRegistrationError,
    EnvironmentNotRegisteredError,
)

logger = logging.getLogger(__name__)


class EnvironmentRegistry:
    """Registry for strategic environment plugins.

    Environments are registered by name and can be instantiated by name.
    Supports both class-based and factory-function-based registration.

    Example::
        registry = EnvironmentRegistry()
        registry.register("mini_resource", MiniResourceEnv)
        env = registry.create("mini_resource", board_size=8)
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        *,
        force: bool = False,
    ) -> None:
        """Register an environment factory.

        Args:
            name: Unique environment name.
            factory: Callable that creates an environment instance.
            force: If True, overwrite existing registration.

        Raises:
            DuplicateRegistrationError: If name already registered and not force.
        """
        if name in self._factories and not force:
            raise DuplicateRegistrationError(
                f"Environment '{name}' is already registered. "
                f"Use force=True to overwrite."
            )
        self._factories[name] = factory
        logger.info("Registered environment: %s", name)

    def create(self, name: str, **kwargs: Any) -> Any:
        """Create an environment instance by name.

        Args:
            name: Registered environment name.
            **kwargs: Arguments passed to the environment factory.

        Returns:
            Environment instance.

        Raises:
            EnvironmentNotRegisteredError: If name is not registered.
        """
        if name not in self._factories:
            raise EnvironmentNotRegisteredError(
                f"Environment '{name}' is not registered. "
                f"Available: {self.list_available()}",
                name=name,
                available=self.list_available(),
            )
        return self._factories[name](**kwargs)

    def list_available(self) -> list[str]:
        """Return sorted list of registered environment names."""
        return sorted(self._factories.keys())

    def is_registered(self, name: str) -> bool:
        return name in self._factories

    def unregister(self, name: str) -> None:
        """Remove a registered environment."""
        self._factories.pop(name, None)

    def __len__(self) -> int:
        return len(self._factories)

    def __contains__(self, name: str) -> bool:
        return name in self._factories
