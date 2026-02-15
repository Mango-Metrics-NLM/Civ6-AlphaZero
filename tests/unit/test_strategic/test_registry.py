"""Unit tests for the environment registry.

Tests the plugin-based environment registry including:
- Built-in environment listing
- Environment creation by name
- Custom environment registration and unregistration
- Error handling for unknown environments
"""

from __future__ import annotations

import pytest

from alpha_zero.strategic.registry import EnvironmentRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the registry state before and after each test.

    This ensures tests are isolated from one another even though
    EnvironmentRegistry uses class-level state.
    """
    EnvironmentRegistry.clear()
    yield
    EnvironmentRegistry.clear()


class _DummyEnv:
    """Minimal dummy environment for registry tests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.num_objectives = 1


# ---------------------------------------------------------------------------
# TestEnvironmentRegistry
# ---------------------------------------------------------------------------


class TestEnvironmentRegistry:
    """Tests for the EnvironmentRegistry."""

    def test_list_environments_has_builtins(self) -> None:
        """Test that built-in environments are registered on first list call."""
        envs = EnvironmentRegistry.list_environments()

        assert isinstance(envs, list)
        assert "resource_allocation" in envs
        assert "climate_negotiation" in envs
        assert "pandemic_response" in envs
        assert "supply_chain" in envs

    def test_create_resource_allocation(self) -> None:
        """Test creating a resource allocation environment."""
        env = EnvironmentRegistry.create("resource_allocation")
        assert env is not None
        assert hasattr(env, "num_objectives")

    def test_create_climate(self) -> None:
        """Test creating a climate negotiation environment."""
        env = EnvironmentRegistry.create("climate_negotiation")
        assert env is not None
        assert hasattr(env, "num_objectives")

    def test_create_pandemic(self) -> None:
        """Test creating a pandemic response environment."""
        env = EnvironmentRegistry.create("pandemic_response")
        assert env is not None
        assert hasattr(env, "num_objectives")

    def test_create_supply_chain(self) -> None:
        """Test creating a supply chain environment."""
        env = EnvironmentRegistry.create("supply_chain")
        assert env is not None
        assert hasattr(env, "num_objectives")

    def test_create_unknown_raises(self) -> None:
        """Test that creating an unknown environment raises KeyError."""
        with pytest.raises(KeyError, match="not registered"):
            EnvironmentRegistry.create("nonexistent_env")

    def test_register_custom(self) -> None:
        """Test registering and creating a custom environment."""
        EnvironmentRegistry.register(
            "dummy",
            _DummyEnv,
            description="A dummy environment for testing.",
        )

        assert EnvironmentRegistry.is_registered("dummy")

        env = EnvironmentRegistry.create("dummy", param1="hello")
        assert isinstance(env, _DummyEnv)
        assert env.kwargs["param1"] == "hello"

    def test_unregister(self) -> None:
        """Test unregistering an environment."""
        EnvironmentRegistry.register("dummy", _DummyEnv)
        assert EnvironmentRegistry.is_registered("dummy")

        EnvironmentRegistry.unregister("dummy")
        assert not EnvironmentRegistry.is_registered("dummy")

        # Unregistering again should raise
        with pytest.raises(KeyError, match="not registered"):
            EnvironmentRegistry.unregister("dummy")

    def test_is_registered(self) -> None:
        """Test is_registered returns correct boolean."""
        # Before builtins are loaded, force load via list
        EnvironmentRegistry.list_environments()

        assert EnvironmentRegistry.is_registered("resource_allocation") is True
        assert EnvironmentRegistry.is_registered("nonexistent") is False

    def test_get_info(self) -> None:
        """Test get_info returns dict mapping names to descriptions."""
        info = EnvironmentRegistry.get_info()

        assert isinstance(info, dict)
        assert "resource_allocation" in info
        assert isinstance(info["resource_allocation"], str)
        assert len(info["resource_allocation"]) > 0
