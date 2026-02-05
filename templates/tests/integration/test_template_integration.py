"""
Integration tests for the template system.

Tests cover:
- End-to-end agent orchestration workflow
- Configuration loading with environment overrides
- Component lifecycle management
- Logging with structured context
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from template_system.agents import (
    AgentConfig,
    AgentOrchestrator,
    AgentRegistry,
    AgentType,
    Thoroughness,
)
from template_system.config import (
    ConfigLoader,
    ConfigManager,
    EnvironmentConfig,
    TemplateConfig,
)
from template_system.factories import (
    ComponentMetadata,
    FactoryRegistry,
    LifecycleManager,
)
from template_system.logging_config import (
    LogConfig,
    LogFormat,
    LogLevel,
    configure_logging,
    get_logger,
    log_context,
)
from template_system.testing import (
    TestConfig,
    capture_logs,
    environment_override,
    temporary_directory,
)


class TestAgentOrchestratorIntegration:
    """Integration tests for agent orchestration."""

    def test_full_workflow_suggestion(self) -> None:
        """Test suggesting and validating a complete workflow."""
        orchestrator = AgentOrchestrator()

        # Suggest workflow for implementation task
        plan = orchestrator.suggest_workflow(
            "implement a new user authentication feature"
        )

        # Validate the plan
        errors = orchestrator.validate_plan(plan)
        assert len(errors) == 0

        # Get execution order
        order = orchestrator.get_execution_order(plan)
        assert len(order) >= 1

        # Verify step types
        assert plan.steps[0].agent_type == AgentType.EXPLORE

    def test_custom_workflow_with_dependencies(self) -> None:
        """Test creating a custom workflow with dependencies."""
        orchestrator = AgentOrchestrator()
        plan = orchestrator.create_plan()

        # Add steps with dependencies
        explore_idx = plan.add_step(AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Find config files",
            prompt="Find all YAML configuration files",
            thoroughness=Thoroughness.MEDIUM,
        ))

        plan_idx = plan.add_step(
            AgentConfig(
                agent_type=AgentType.PLAN,
                description="Plan config changes",
                prompt="Plan changes to configuration system",
            ),
            depends_on=[explore_idx],
        )

        bash_idx = plan.add_step(
            AgentConfig(
                agent_type=AgentType.BASH,
                description="Run tests",
                prompt="Run pytest to validate changes",
            ),
            depends_on=[plan_idx],
        )

        # Validate
        errors = orchestrator.validate_plan(plan)
        assert len(errors) == 0

        # Get execution order
        order = orchestrator.get_execution_order(plan)
        assert len(order) == 3  # Sequential execution
        assert order[0] == [explore_idx]
        assert order[1] == [plan_idx]
        assert order[2] == [bash_idx]

    def test_parallel_workflow_execution_order(self) -> None:
        """Test parallel workflow execution order."""
        orchestrator = AgentOrchestrator(max_parallel=3)
        plan = orchestrator.create_plan()

        # Add independent parallel steps
        idx1 = plan.add_step(AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Search A",
            prompt="Find pattern A",
        ))
        idx2 = plan.add_step(AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Search B",
            prompt="Find pattern B",
        ))
        idx3 = plan.add_step(AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Search C",
            prompt="Find pattern C",
        ))

        # Mark as parallel
        plan.add_parallel_group([idx1, idx2, idx3])

        # Add dependent step
        plan.add_step(
            AgentConfig(
                agent_type=AgentType.PLAN,
                description="Combine results",
                prompt="Combine search results",
            ),
            depends_on=[idx1, idx2, idx3],
        )

        # Get execution order
        order = orchestrator.get_execution_order(plan)

        # First batch should be parallel
        assert len(order[0]) == 3
        assert set(order[0]) == {idx1, idx2, idx3}

        # Second batch is the dependent step
        assert len(order[1]) == 1


class TestConfigurationIntegration:
    """Integration tests for configuration management."""

    def test_config_loading_with_env_overrides(self) -> None:
        """Test loading config file with environment overrides."""
        with temporary_directory() as temp_dir:
            # Create config file
            config_path = temp_dir / "config.json"
            config_path.write_text(json.dumps({
                "app": {"name": "test_app", "debug": False},
                "database": {"host": "localhost", "port": 5432},
            }))

            # Set environment override
            with environment_override(APP_DEBUG="true", DATABASE_PORT="5433"):
                env_config = EnvironmentConfig(prefix="", separator="_")
                loader = ConfigLoader(env_config=env_config)

                # Load file
                file_config = loader.load_file(config_path)

                # Load env
                env_overrides = loader.load_env()

                # Merge
                merged = loader.merge(file_config, env_overrides)

                # Verify original values
                assert merged["app"]["name"] == "test_app"

    def test_template_substitution_full_workflow(self) -> None:
        """Test complete template substitution workflow."""
        config = TemplateConfig(variables={
            "SYSTEM_NAME": "My Awesome System",
            "PROJECT_NAME": "my-project",
            "PACKAGE_NAME": "my_project",
            "COVERAGE_THRESHOLD": "85",
            "PORT": "8080",
        })

        # Validate
        errors = config.validate()
        assert len(errors) == 0

        # Substitute
        template = """
# {SYSTEM_NAME}

Project: {PROJECT_NAME}
Package: {PACKAGE_NAME}

```bash
pytest --cov={PACKAGE_NAME} --cov-fail-under={COVERAGE_THRESHOLD}
python -m {PACKAGE_NAME} --port {PORT}
```
"""
        result = config.substitute(template)

        assert "My Awesome System" in result
        assert "my-project" in result
        assert "my_project" in result
        assert "--cov-fail-under=85" in result
        assert "--port 8080" in result


class TestComponentLifecycleIntegration:
    """Integration tests for component lifecycle management."""

    def test_full_lifecycle(self) -> None:
        """Test complete component lifecycle."""
        # Create a component class
        class DatabaseConnection:
            def __init__(self, host: str = "localhost", port: int = 5432) -> None:
                self.host = host
                self.port = port
                self.connected = False

            def initialize(self) -> None:
                self.connected = True

            def shutdown(self) -> None:
                self.connected = False

        class CacheService:
            def __init__(self, database: DatabaseConnection | None = None) -> None:
                self.database = database
                self.ready = False

            def initialize(self) -> None:
                self.ready = True

            def shutdown(self) -> None:
                self.ready = False

        # Setup registry
        registry = FactoryRegistry()
        registry.register(
            "database",
            DatabaseConnection,
            ComponentMetadata(name="database", singleton=True),
        )
        registry.register(
            "cache",
            CacheService,
            ComponentMetadata(name="cache", dependencies=["database"]),
        )

        # Create lifecycle manager
        manager = LifecycleManager(registry)

        # Create and initialize database
        db = manager.create_and_initialize(
            "database",
            {"host": "db.example.com", "port": 5433},
        )
        assert db.connected is True
        assert db.host == "db.example.com"

        # Create cache (should get database dependency)
        cache = manager.create_and_initialize("cache", {})
        assert cache.ready is True

        # Get all components
        all_components = manager.get_all()
        assert len(all_components) == 2

        # Shutdown all
        manager.shutdown()
        assert db.connected is False
        assert cache.ready is False


class TestLoggingIntegration:
    """Integration tests for logging infrastructure."""

    def test_structured_logging_with_context(self) -> None:
        """Test structured logging with bound context."""
        configure_logging(level="DEBUG", format="console")

        with capture_logs("test.integration.logging") as records:
            logger = get_logger("test.integration.logging")

            # Basic logging
            logger.info("Starting request")

            # Bound logging
            request_logger = logger.bind(request_id="req-123", user="test_user")
            request_logger.info("Processing request")
            request_logger.warning("Slow query detected")

            # Nested context
            with log_context(operation="database_query"):
                request_logger.debug("Executing query")

        # Verify logs captured
        assert len(records) >= 3

    def test_json_logging_format(self) -> None:
        """Test JSON logging output format."""
        configure_logging(level="INFO", format="json")

        logger = get_logger("test.integration.json")
        logger.info("Test JSON message", custom_field="custom_value")


class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    def test_complete_project_setup_workflow(self) -> None:
        """Test simulating a complete project setup workflow."""
        with temporary_directory() as project_dir:
            # 1. Create configuration
            config_dir = project_dir / "configs"
            config_dir.mkdir()

            project_config = {
                "project": {
                    "name": "test-project",
                    "version": "1.0.0",
                },
                "testing": {
                    "coverage_threshold": 80,
                    "parallel": True,
                },
                "logging": {
                    "level": "INFO",
                    "format": "json",
                },
            }
            (config_dir / "project.json").write_text(json.dumps(project_config))

            # 2. Load configuration
            manager = ConfigManager(config_dir / "project.json")
            config = manager.load()

            assert config["project"]["name"] == "test-project"

            # 3. Setup template variables
<<<<<<< HEAD
            TemplateConfig(variables={
=======
            _template_config = TemplateConfig(variables={
>>>>>>> origin/HEAD
                "SYSTEM_NAME": config["project"]["name"],
                "PROJECT_NAME": "test-project",
                "PACKAGE_NAME": "test_project",
                "COVERAGE_THRESHOLD": str(config["testing"]["coverage_threshold"]),
            })

            # 4. Create test configuration
            test_config = TestConfig(
                test_dir=project_dir / "tests",
                coverage_threshold=config["testing"]["coverage_threshold"],
                parallel=config["testing"]["parallel"],
            )

            assert test_config.coverage_threshold == 80
            assert test_config.parallel is True

            # 5. Configure logging
            log_config = LogConfig(
                level=LogLevel.from_string(config["logging"]["level"]),
                format=LogFormat(config["logging"]["format"]),
            )

            assert log_config.level == LogLevel.INFO
            assert log_config.format == LogFormat.JSON

            # 6. Setup component registry
<<<<<<< HEAD
            FactoryRegistry()
=======
            _registry = FactoryRegistry()
>>>>>>> origin/HEAD

            # 7. Plan agent workflow
            orchestrator = AgentOrchestrator()
            plan = orchestrator.suggest_workflow("setup project structure")

            errors = orchestrator.validate_plan(plan)
            assert len(errors) == 0

    def test_agent_registry_with_custom_capabilities(self) -> None:
        """Test extending agent registry with custom capabilities."""
        from template_system.agents import AgentCapability

        registry = AgentRegistry()

        # Add custom capability
        custom_cap = AgentCapability(
            name="code_review",
            description="Review code for best practices",
            required_tools=["Read", "Grep"],
            constraints=["Read-only operation"],
        )
        registry.register_capability(AgentType.EXPLORE, custom_cap)

        # Verify it's registered
        caps = registry.get_capabilities(AgentType.EXPLORE)
        cap_names = [c.name for c in caps]
        assert "code_review" in cap_names

        # Suggest agent for review task
<<<<<<< HEAD
        _suggested = registry.suggest_agent("review the authentication code")
        # Should suggest based on patterns


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    def test_config_loading_with_invalid_file(self) -> None:
        """Test handling invalid config file."""
        with temporary_directory() as temp_dir:
            # Create invalid JSON
            bad_config = temp_dir / "bad.json"
            bad_config.write_text("{ invalid json }")

            loader = ConfigLoader()
            with pytest.raises((ValueError, json.JSONDecodeError)):
                loader.load_file(bad_config)

    def test_factory_creation_with_invalid_config(self) -> None:
        """Test factory handles invalid configuration."""
        from template_system.factories import ComponentCreationError

        class StrictComponent:
            def __init__(self, required_param: str) -> None:
                if not required_param:
                    raise ValueError("required_param cannot be empty")
                self.param = required_param

        registry = FactoryRegistry()
        registry.register("strict", StrictComponent)

        # Should raise on creation with empty param
        with pytest.raises(ComponentCreationError):
            registry.create("strict", {"required_param": ""})

    def test_orchestrator_handles_invalid_plan(self) -> None:
        """Test orchestrator handles invalid plan gracefully."""
        from template_system.agents import OrchestrationPlan

        orchestrator = AgentOrchestrator()

        # Create invalid plan with circular dependency
        plan = OrchestrationPlan()
        plan.add_step(AgentConfig(
            agent_type=AgentType.EXPLORE,
            description="Step A",
            prompt="Do A",
        ))
        # Invalid: depends on itself
        plan.dependencies[0] = [0]

        errors = orchestrator.validate_plan(plan)
        # Should detect the invalid dependency
        assert any("depend" in e.lower() for e in errors)
