"""
Shared test fixtures for the template system.

This module provides pytest fixtures that are shared across all tests.
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if TYPE_CHECKING:
    from template_system.agents import AgentConfig, AgentOrchestrator
    from template_system.config import TemplateConfig
    from template_system.factories import FactoryRegistry
    from template_system.logging_config import LogConfig
    from template_system.testing import TestConfig


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests.

    Yields:
        Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Provide a temporary file for tests.

    Args:
        temp_dir: Temporary directory fixture.

    Yields:
        Path to temporary file.
    """
    file_path = temp_dir / "test_file.txt"
    file_path.touch()
    yield file_path


@pytest.fixture
def sample_yaml_config(temp_dir: Path) -> Path:
    """Create a sample YAML config file.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        Path to config file.
    """
    config_path = temp_dir / "config.yaml"
    config_path.write_text("""
app:
  name: test_app
  debug: true
  log_level: DEBUG

database:
  host: localhost
  port: 5432
  name: testdb

features:
  enabled:
    - feature1
    - feature2
""")
    return config_path


@pytest.fixture
def sample_json_config(temp_dir: Path) -> Path:
    """Create a sample JSON config file.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        Path to config file.
    """
    import json

    config_path = temp_dir / "config.json"
    config_path.write_text(json.dumps({
        "app": {
            "name": "test_app",
            "debug": True,
            "log_level": "DEBUG",
        },
        "database": {
            "host": "localhost",
            "port": 5432,
        },
    }))
    return config_path


@pytest.fixture
def env_vars() -> Generator[dict[str, str], None, None]:
    """Provide environment variable management.

    Yields:
        Dictionary for tracking set variables.
    """
    original: dict[str, str | None] = {}
    set_vars: dict[str, str] = {}

    def set_env(key: str, value: str) -> None:
        if key not in original:
            original[key] = os.environ.get(key)
        os.environ[key] = value
        set_vars[key] = value

    # Expose set function via the dict
    set_vars["_set"] = set_env  # type: ignore

    yield set_vars

    # Restore original values
    for key, orig_value in original.items():
        if orig_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = orig_value


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create a sample agent configuration.

    Returns:
        AgentConfig instance.
    """
    from template_system.agents import AgentConfig, AgentType, Thoroughness

    return AgentConfig(
        agent_type=AgentType.EXPLORE,
        description="Test exploration",
        prompt="Find all Python files",
        thoroughness=Thoroughness.MEDIUM,
    )


@pytest.fixture
def orchestrator() -> AgentOrchestrator:
    """Create an agent orchestrator.

    Returns:
        AgentOrchestrator instance.
    """
    from template_system.agents import AgentOrchestrator

    return AgentOrchestrator()


@pytest.fixture
def factory_registry() -> FactoryRegistry:
    """Create a factory registry.

    Returns:
        FactoryRegistry instance.
    """
    from template_system.factories import FactoryRegistry

    return FactoryRegistry()


@pytest.fixture
def template_config() -> TemplateConfig:
    """Create a template configuration.

    Returns:
        TemplateConfig instance.
    """
    from template_system.config import TemplateConfig

    return TemplateConfig(
        variables={
            "SYSTEM_NAME": "Test System",
            "PROJECT_NAME": "test-project",
            "PACKAGE_NAME": "test_package",
            "COVERAGE_THRESHOLD": "80",
        }
    )


@pytest.fixture
def log_config() -> LogConfig:
    """Create a log configuration.

    Returns:
        LogConfig instance.
    """
    from template_system.logging_config import LogConfig, LogFormat, LogLevel

    return LogConfig(
        level=LogLevel.DEBUG,
        format=LogFormat.CONSOLE,
    )


@pytest.fixture
def test_config(temp_dir: Path) -> TestConfig:
    """Create a test configuration.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        TestConfig instance.
    """
    from template_system.testing import TestConfig

    return TestConfig(
        test_dir=temp_dir / "tests",
        coverage_threshold=80,
        timeout_seconds=60,
    )


@pytest.fixture
def sample_component_class() -> type:
    """Create a sample component class for testing.

    Returns:
        Sample component class.
    """
    class SampleComponent:
        """A sample component for testing."""

        def __init__(
            self,
            name: str = "default",
            value: int = 0,
        ) -> None:
            self.name = name
            self.value = value
            self._initialized = False

        def initialize(self) -> None:
            self._initialized = True

        def shutdown(self) -> None:
            self._initialized = False

        def process(self, data: Any) -> Any:
            return {"name": self.name, "value": self.value, "data": data}

    return SampleComponent
