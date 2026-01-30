"""
Unit tests for the factories module.

Tests cover:
- ComponentMetadata dataclass
- SimpleFactory and FunctionFactory
- FactoryRegistry registration and creation
- LifecycleManager initialization and shutdown
- Global registry and component decorator
"""
from __future__ import annotations

from typing import Any

import pytest

from template_system.factories import (
    ComponentCreationError,
    ComponentError,
    ComponentMetadata,
    ComponentNotFoundError,
    FactoryRegistry,
    FunctionFactory,
    LifecycleManager,
    SimpleFactory,
    component,
    create_component,
    get_global_registry,
    register_component,
)


class TestComponentMetadata:
    """Tests for ComponentMetadata dataclass."""

    def test_metadata_creation(self) -> None:
        """Test creating component metadata."""
        metadata = ComponentMetadata(
            name="test_component",
            description="Test component",
            version="2.0.0",
            dependencies=["dep1", "dep2"],
            tags=["core", "utility"],
            singleton=True,
        )
        assert metadata.name == "test_component"
        assert metadata.version == "2.0.0"
        assert len(metadata.dependencies) == 2
        assert metadata.singleton is True

    def test_metadata_defaults(self) -> None:
        """Test default metadata values."""
        metadata = ComponentMetadata(name="minimal")
        assert metadata.description == ""
        assert metadata.version == "1.0.0"
        assert metadata.dependencies == []
        assert metadata.tags == []
        assert metadata.singleton is False


class TestComponentExceptions:
    """Tests for component exceptions."""

    def test_component_error_base(self) -> None:
        """Test base ComponentError."""
        error = ComponentError("Base error")
        assert str(error) == "Base error"

    def test_component_not_found_error(self) -> None:
        """Test ComponentNotFoundError."""
        error = ComponentNotFoundError("Component 'x' not found")
        assert "not found" in str(error)
        assert isinstance(error, ComponentError)

    def test_component_creation_error(self) -> None:
        """Test ComponentCreationError."""
        error = ComponentCreationError("Failed to create")
        assert "Failed" in str(error)
        assert isinstance(error, ComponentError)


class TestSimpleFactory:
    """Tests for SimpleFactory class."""

    def test_create_simple_component(
        self,
        sample_component_class: type,
    ) -> None:
        """Test creating a component with SimpleFactory."""
        factory = SimpleFactory(sample_component_class)
        component = factory.create({"name": "test", "value": 42})
        assert component.name == "test"
        assert component.value == 42

    def test_create_with_kwargs(
        self,
        sample_component_class: type,
    ) -> None:
        """Test creating with additional kwargs."""
        factory = SimpleFactory(sample_component_class)
        component = factory.create({"name": "from_config"}, value=100)
        assert component.name == "from_config"
        assert component.value == 100

    def test_create_with_default_metadata(
        self,
        sample_component_class: type,
    ) -> None:
        """Test factory creates default metadata."""
        factory = SimpleFactory(sample_component_class)
        assert factory.metadata.name == "SampleComponent"

    def test_create_with_custom_metadata(
        self,
        sample_component_class: type,
    ) -> None:
        """Test factory with custom metadata."""
        metadata = ComponentMetadata(
            name="custom_name",
            description="Custom description",
        )
        factory = SimpleFactory(sample_component_class, metadata)
        assert factory.metadata.name == "custom_name"

    def test_create_failure(self) -> None:
        """Test factory raises error on creation failure."""
        class BadComponent:
            def __init__(self, required: str) -> None:
                self.required = required

        factory = SimpleFactory(BadComponent)
        with pytest.raises(ComponentCreationError):
            factory.create({})  # Missing required argument


class TestFunctionFactory:
    """Tests for FunctionFactory class."""

    def test_create_with_function(self) -> None:
        """Test creating component with a function."""
        def create_dict(config: dict[str, Any]) -> dict[str, Any]:
            return {"created": True, **config}

        factory = FunctionFactory(create_dict)
        result = factory.create({"key": "value"})
        assert result["created"] is True
        assert result["key"] == "value"

    def test_create_with_metadata(self) -> None:
        """Test function factory with custom metadata."""
        def creator(config: Any) -> str:
            return "created"

        metadata = ComponentMetadata(name="string_creator")
        factory = FunctionFactory(creator, metadata)
        assert factory.metadata.name == "string_creator"

    def test_create_with_validator(self) -> None:
        """Test function factory with validator."""
        def creator(config: dict[str, Any]) -> dict[str, Any]:
            return config

        def validator(config: dict[str, Any]) -> list[str]:
            errors = []
            if "required_key" not in config:
                errors.append("Missing required_key")
            return errors

        factory = FunctionFactory(creator, validator=validator)

        # Should report validation error
        errors = factory.validate_config({})
        assert "Missing required_key" in errors

        # Should pass validation
        errors = factory.validate_config({"required_key": "value"})
        assert len(errors) == 0

    def test_create_failure(self) -> None:
        """Test function factory raises error on failure."""
        def failing_creator(config: Any) -> Any:
            raise ValueError("Creation failed")

        factory = FunctionFactory(failing_creator)
        with pytest.raises(ComponentCreationError):
            factory.create({})


class TestFactoryRegistry:
    """Tests for FactoryRegistry class."""

    def test_register_factory(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test registering a factory."""
        factory = SimpleFactory(sample_component_class)
        factory_registry.register("sample", factory)
        assert "sample" in factory_registry.list_components()

    def test_register_class_directly(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test registering a class directly."""
        factory_registry.register("direct", sample_component_class)
        assert "direct" in factory_registry.list_components()

    def test_unregister(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test unregistering a component."""
        factory_registry.register("temp", sample_component_class)
        assert "temp" in factory_registry.list_components()
        factory_registry.unregister("temp")
        assert "temp" not in factory_registry.list_components()

    def test_get_factory(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test getting a factory."""
        factory_registry.register("sample", sample_component_class)
        factory = factory_registry.get_factory("sample")
        assert factory is not None

    def test_get_factory_not_found(
        self,
        factory_registry: FactoryRegistry,
    ) -> None:
        """Test getting non-existent factory."""
        with pytest.raises(ComponentNotFoundError):
            factory_registry.get_factory("nonexistent")

    def test_create_component(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test creating a component."""
        factory_registry.register("sample", sample_component_class)
        component = factory_registry.create("sample", {"name": "test"})
        assert component.name == "test"

    def test_create_singleton(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test singleton component creation."""
        metadata = ComponentMetadata(name="singleton", singleton=True)
        factory_registry.register("singleton", sample_component_class, metadata)

        comp1 = factory_registry.create("singleton", {"name": "first"})
        comp2 = factory_registry.create("singleton", {"name": "second"})
        assert comp1 is comp2

    def test_list_components(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test listing components."""
        factory_registry.register("comp1", sample_component_class)
        factory_registry.register("comp2", sample_component_class)
        components = factory_registry.list_components()
        assert "comp1" in components
        assert "comp2" in components

    def test_get_metadata(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test getting component metadata."""
        metadata = ComponentMetadata(
            name="with_meta",
            description="Has metadata",
        )
        factory_registry.register("with_meta", sample_component_class, metadata)
        retrieved = factory_registry.get_metadata("with_meta")
        assert retrieved is not None
        assert retrieved.description == "Has metadata"

    def test_clear(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test clearing registry."""
        factory_registry.register("comp", sample_component_class)
        factory_registry.clear()
        assert len(factory_registry.list_components()) == 0

    def test_find_by_tag(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test finding components by tag."""
        meta1 = ComponentMetadata(name="tagged1", tags=["core", "utility"])
        meta2 = ComponentMetadata(name="tagged2", tags=["utility"])
        meta3 = ComponentMetadata(name="untagged", tags=[])

        factory_registry.register("tagged1", sample_component_class, meta1)
        factory_registry.register("tagged2", sample_component_class, meta2)
        factory_registry.register("untagged", sample_component_class, meta3)

        utility_components = factory_registry.find_by_tag("utility")
        assert "tagged1" in utility_components
        assert "tagged2" in utility_components
        assert "untagged" not in utility_components

    def test_validation_failure(
        self,
        factory_registry: FactoryRegistry,
    ) -> None:
        """Test creation fails on validation error."""
        def creator(config: dict[str, Any]) -> dict[str, Any]:
            return config

        def validator(config: dict[str, Any]) -> list[str]:
            return ["Validation failed"]

        factory = FunctionFactory(creator, validator=validator)
        factory_registry.register("validated", factory)

        with pytest.raises(ComponentCreationError, match="Validation failed"):
            factory_registry.create("validated", {})


class TestLifecycleManager:
    """Tests for LifecycleManager class."""

    def test_create_and_initialize(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test creating and initializing a component."""
        factory_registry.register("sample", sample_component_class)
        manager = LifecycleManager(factory_registry)

        component = manager.create_and_initialize("sample", {"name": "test"})
        assert component._initialized is True

    def test_shutdown_single(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test shutting down a single component."""
        factory_registry.register("sample", sample_component_class)
        manager = LifecycleManager(factory_registry)

        component = manager.create_and_initialize("sample", {"name": "test"})
        manager.shutdown("sample")
        assert component._initialized is False

    def test_shutdown_all(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test shutting down all components."""
        factory_registry.register("comp1", sample_component_class)
        factory_registry.register("comp2", sample_component_class)
        manager = LifecycleManager(factory_registry)

        comp1 = manager.create_and_initialize("comp1", {"name": "c1"})
        comp2 = manager.create_and_initialize("comp2", {"name": "c2"})

        manager.shutdown()
        assert comp1._initialized is False
        assert comp2._initialized is False

    def test_get_component(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test getting an initialized component."""
        factory_registry.register("sample", sample_component_class)
        manager = LifecycleManager(factory_registry)

        manager.create_and_initialize("sample", {"name": "test"})
        component = manager.get("sample")
        assert component is not None
        assert component.name == "test"

    def test_get_all(
        self,
        factory_registry: FactoryRegistry,
        sample_component_class: type,
    ) -> None:
        """Test getting all initialized components."""
        factory_registry.register("comp1", sample_component_class)
        factory_registry.register("comp2", sample_component_class)
        manager = LifecycleManager(factory_registry)

        manager.create_and_initialize("comp1", {"name": "c1"})
        manager.create_and_initialize("comp2", {"name": "c2"})

        all_components = manager.get_all()
        assert len(all_components) == 2
        assert "comp1" in all_components
        assert "comp2" in all_components


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_global_registry(self) -> None:
        """Test getting global registry."""
        registry = get_global_registry()
        assert isinstance(registry, FactoryRegistry)

    def test_global_registry_singleton(self) -> None:
        """Test global registry is a singleton."""
        reg1 = get_global_registry()
        reg2 = get_global_registry()
        assert reg1 is reg2


class TestComponentDecorator:
    """Tests for @component decorator."""

    def test_decorator_registers_class(self) -> None:
        """Test decorator registers class."""
        @component(name="decorated_component")
        class DecoratedComponent:
            """A decorated component."""

            def __init__(self, value: int = 0) -> None:
                self.value = value

        registry = get_global_registry()
        assert "decorated_component" in registry.list_components()

    def test_decorator_with_singleton(self) -> None:
        """Test decorator with singleton option."""
        @component(name="singleton_decorated", singleton=True)
        class SingletonDecorated:
            """A singleton component."""

            def __init__(self) -> None:
                pass

        registry = get_global_registry()
        metadata = registry.get_metadata("singleton_decorated")
        assert metadata is not None
        assert metadata.singleton is True

    def test_decorator_with_tags(self) -> None:
        """Test decorator with tags."""
        @component(name="tagged_decorated", tags=["tag1", "tag2"])
        class TaggedDecorated:
            """A tagged component."""

            def __init__(self) -> None:
                pass

        registry = get_global_registry()
        found = registry.find_by_tag("tag1")
        assert "tagged_decorated" in found
