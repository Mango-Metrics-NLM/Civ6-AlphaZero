"""Unit tests for the component factory registry.

Tests the ComponentRegistry class including:
- Registration and creation
- Duplicate registration handling
- Missing component errors
- list_components and has_component
- Thread safety
- Default registry integration with factories
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from alpha_zero.utils.registry import ComponentRegistry, default_registry


class TestComponentRegistryBasics:
    """Tests for basic registration and creation."""

    def test_register_and_create(self) -> None:
        """Test registering and creating a component."""
        registry = ComponentRegistry()

        @registry.register("network", "test")
        def create_test_network(size: int = 10) -> dict:
            return {"type": "test", "size": size}

        result = registry.create("network", "test", size=42)
        assert result == {"type": "test", "size": 42}

    def test_register_factory_non_decorator(self) -> None:
        """Test registering a factory function using the non-decorator form."""
        registry = ComponentRegistry()

        def my_factory(x: int = 0) -> int:
            return x * 2

        registry.register_factory("math", "doubler", my_factory)

        result = registry.create("math", "doubler", x=5)
        assert result == 10

    def test_create_with_no_kwargs(self) -> None:
        """Test creating a component with no keyword arguments."""
        registry = ComponentRegistry()

        @registry.register("simple", "noop")
        def create_noop() -> str:
            return "created"

        result = registry.create("simple", "noop")
        assert result == "created"

    def test_create_with_default_kwargs(self) -> None:
        """Test creating a component that uses default keyword arguments."""
        registry = ComponentRegistry()

        @registry.register("widget", "default_widget")
        def create_widget(color: str = "blue", size: int = 10) -> dict:
            return {"color": color, "size": size}

        result = registry.create("widget", "default_widget")
        assert result == {"color": "blue", "size": 10}


class TestDuplicateRegistration:
    """Tests for duplicate registration handling."""

    def test_duplicate_raises_key_error(self) -> None:
        """Test that duplicate registration raises KeyError by default."""
        registry = ComponentRegistry()

        @registry.register("network", "dup")
        def first_factory() -> str:
            return "first"

        with pytest.raises(KeyError, match="already registered"):

            @registry.register("network", "dup")
            def second_factory() -> str:
                return "second"

    def test_duplicate_with_allow_override(self) -> None:
        """Test that duplicate registration succeeds with allow_override=True."""
        registry = ComponentRegistry()

        @registry.register("network", "dup")
        def first_factory() -> str:
            return "first"

        @registry.register("network", "dup", allow_override=True)
        def second_factory() -> str:
            return "second"

        result = registry.create("network", "dup")
        assert result == "second"

    def test_duplicate_register_factory_raises(self) -> None:
        """Test that register_factory raises on duplicates."""
        registry = ComponentRegistry()
        registry.register_factory("cat", "name", lambda: "a")

        with pytest.raises(KeyError, match="already registered"):
            registry.register_factory("cat", "name", lambda: "b")

    def test_duplicate_register_factory_with_override(self) -> None:
        """Test that register_factory succeeds with allow_override."""
        registry = ComponentRegistry()
        registry.register_factory("cat", "name", lambda: "a")
        registry.register_factory("cat", "name", lambda: "b", allow_override=True)

        result = registry.create("cat", "name")
        assert result == "b"


class TestMissingComponent:
    """Tests for missing component errors."""

    def test_create_unknown_category(self) -> None:
        """Test that creating from an unknown category raises KeyError."""
        registry = ComponentRegistry()

        with pytest.raises(KeyError, match="Unknown category"):
            registry.create("nonexistent", "anything")

    def test_create_unknown_name(self) -> None:
        """Test that creating an unknown name raises KeyError."""
        registry = ComponentRegistry()

        @registry.register("network", "known")
        def factory() -> str:
            return "ok"

        with pytest.raises(KeyError, match="Unknown component"):
            registry.create("network", "unknown")

    def test_list_unknown_category(self) -> None:
        """Test that listing an unknown category raises KeyError."""
        registry = ComponentRegistry()

        with pytest.raises(KeyError, match="Unknown category"):
            registry.list_components("nonexistent")


class TestListComponents:
    """Tests for listing components."""

    def test_list_components_returns_sorted_names(self) -> None:
        """Test that list_components returns sorted component names."""
        registry = ComponentRegistry()
        registry.register_factory("env", "zebra", lambda: None)
        registry.register_factory("env", "alpha", lambda: None)
        registry.register_factory("env", "middle", lambda: None)

        names = registry.list_components("env")
        assert names == ["alpha", "middle", "zebra"]

    def test_list_categories(self) -> None:
        """Test that list_categories returns all categories sorted."""
        registry = ComponentRegistry()
        registry.register_factory("z_cat", "a", lambda: None)
        registry.register_factory("a_cat", "b", lambda: None)
        registry.register_factory("m_cat", "c", lambda: None)

        categories = registry.list_categories()
        assert categories == ["a_cat", "m_cat", "z_cat"]

    def test_list_categories_empty(self) -> None:
        """Test that list_categories returns empty list when registry is empty."""
        registry = ComponentRegistry()
        assert registry.list_categories() == []


class TestHasComponent:
    """Tests for has_component check."""

    def test_has_component_returns_true(self) -> None:
        """Test has_component returns True for registered components."""
        registry = ComponentRegistry()
        registry.register_factory("net", "test", lambda: None)

        assert registry.has_component("net", "test") is True

    def test_has_component_returns_false_wrong_name(self) -> None:
        """Test has_component returns False for unregistered name."""
        registry = ComponentRegistry()
        registry.register_factory("net", "test", lambda: None)

        assert registry.has_component("net", "other") is False

    def test_has_component_returns_false_wrong_category(self) -> None:
        """Test has_component returns False for unregistered category."""
        registry = ComponentRegistry()

        assert registry.has_component("nonexistent", "anything") is False


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_registration(self) -> None:
        """Test that concurrent registrations are safe."""
        registry = ComponentRegistry()
        errors: list[Exception] = []

        def register_component(category: str, name: str) -> None:
            try:
                registry.register_factory(category, name, lambda: name)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(50):
            t = threading.Thread(
                target=register_component,
                args=("concurrent", f"component_{i}"),
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry.list_components("concurrent")) == 50

    def test_concurrent_creation(self) -> None:
        """Test that concurrent creation calls are safe."""
        registry = ComponentRegistry()
        counter = {"calls": 0}
        lock = threading.Lock()

        def counting_factory() -> int:
            with lock:
                counter["calls"] += 1
            return counter["calls"]

        registry.register_factory("counter", "default", counting_factory)

        results: list[int] = []
        result_lock = threading.Lock()

        def create_and_collect() -> None:
            result = registry.create("counter", "default")
            with result_lock:
                results.append(result)

        threads = []
        for _ in range(20):
            t = threading.Thread(target=create_and_collect)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert counter["calls"] == 20

    def test_concurrent_read_write(self) -> None:
        """Test that reads and writes can happen concurrently."""
        registry = ComponentRegistry()
        registry.register_factory("base", "initial", lambda: "initial")

        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(50):
                    registry.has_component("base", "initial")
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(50):
                    registry.register_factory("base", f"new_{i}", lambda i=i: i)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestClear:
    """Tests for the clear method."""

    def test_clear_removes_all(self) -> None:
        """Test that clear removes all registered components."""
        registry = ComponentRegistry()
        registry.register_factory("a", "b", lambda: None)
        registry.register_factory("c", "d", lambda: None)

        registry.clear()

        assert registry.list_categories() == []

    def test_clear_allows_re_registration(self) -> None:
        """Test that cleared registry allows re-registration."""
        registry = ComponentRegistry()
        registry.register_factory("net", "test", lambda: "first")
        registry.clear()
        registry.register_factory("net", "test", lambda: "second")

        result = registry.create("net", "test")
        assert result == "second"


class TestRepr:
    """Tests for __repr__."""

    def test_repr_shows_counts(self) -> None:
        """Test that __repr__ shows category counts."""
        registry = ComponentRegistry()
        registry.register_factory("net", "a", lambda: None)
        registry.register_factory("net", "b", lambda: None)
        registry.register_factory("env", "c", lambda: None)

        result = repr(registry)
        assert "ComponentRegistry" in result
        assert "'net': 2" in result
        assert "'env': 1" in result


class TestDefaultRegistry:
    """Tests for the default global registry with factory integration."""

    def test_default_registry_is_component_registry(self) -> None:
        """Test that default_registry is a ComponentRegistry instance."""
        assert isinstance(default_registry, ComponentRegistry)

    def test_default_registry_has_factories_registered(self) -> None:
        """Test that importing factories registers components in default_registry."""
        # Importing factories triggers _register_factories()
        import alpha_zero.utils.factories  # noqa: F401

        # Check that core factory categories exist
        assert default_registry.has_component("network", "default")
        assert default_registry.has_component("mcts", "default")
        assert default_registry.has_component("environment", "default")
        assert default_registry.has_component("replay_buffer", "default")

    def test_default_registry_list_categories(self) -> None:
        """Test that default_registry has expected categories after factory import."""
        import alpha_zero.utils.factories  # noqa: F401

        categories = default_registry.list_categories()
        assert "network" in categories
        assert "mcts" in categories
        assert "environment" in categories

    def test_default_registry_create_replay_buffer(self) -> None:
        """Test creating a replay buffer through the default registry."""
        import alpha_zero.utils.factories  # noqa: F401

        buffer = default_registry.create("replay_buffer", "default", max_size=100, min_size=5)
        assert buffer is not None
        assert len(buffer) == 0
        assert hasattr(buffer, "add")
        assert hasattr(buffer, "sample")
