"""Shared test fixtures for agentic framework tests.

This module provides reusable fixtures for unit and integration tests,
reducing code duplication and ensuring consistent test setup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_framework.agents.base_agent import AgentContext
from agentic_framework.config import AgenticConfig
from agentic_framework.core.constraint_manager import ConstraintManager
from agentic_framework.core.permission_manager import PermissionManager
from agentic_framework.core.session_memory import SessionMemory
from agentic_framework.utils.command_runner import CommandRunner

# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def basic_config() -> AgenticConfig:
    """Create a basic configuration for testing.

    Returns:
        Configured AgenticConfig instance.
    """
    config = AgenticConfig()
    config.objective.system_intent = "Test objective"
    config.objective.success_criteria = [{"criterion": "Tests pass"}]
    config.permissions.scope.in_scope = ["**/*.py"]
    config.permissions.autonomy.autonomous = ["file_creation", "running_tests"]
    config.permissions.resource_budget.max_iterations = 5
    config.feedback.verification_commands = []
    return config


@pytest.fixture
def minimal_config() -> AgenticConfig:
    """Create a minimal configuration with defaults.

    Returns:
        AgenticConfig with minimal settings.
    """
    return AgenticConfig()


@pytest.fixture(params=["basic", "with_tools", "with_memory", "restricted"])
def agent_config_variant(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Parameterized agent configurations for testing various setups.

    Args:
        request: Pytest fixture request.

    Returns:
        Configuration dictionary for the requested variant.
    """
    configs = {
        "basic": {
            "name": "test-agent",
            "description": "Basic test agent",
        },
        "with_tools": {
            "name": "test-agent",
            "description": "Agent with tools",
            "tools": ["read", "write", "execute"],
        },
        "with_memory": {
            "name": "test-agent",
            "description": "Agent with memory",
            "use_memory": True,
            "memory_capacity": 100,
        },
        "restricted": {
            "name": "test-agent",
            "description": "Restricted agent",
            "max_iterations": 3,
            "timeout_seconds": 30,
        },
    }
    return configs[request.param]


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a reusable mock LLM client.

    Returns:
        MagicMock configured as an LLM client.
    """
    client = MagicMock()
    client.complete = AsyncMock(return_value="Test response from LLM")
    client.stream = AsyncMock()
    client.get_embedding = AsyncMock(return_value=[0.1] * 384)
    client.model_name = "test-model"
    client.is_available = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_permission_manager() -> MagicMock:
    """Create a reusable mock permission manager.

    Returns:
        MagicMock configured as a permission manager.
    """
    from agentic_framework.core.permission_manager import (
        AutonomyLevel,
        PermissionCheckResult,
    )

    manager = MagicMock(spec=PermissionManager)
    # Return a proper PermissionCheckResult object with allowed=True
    mock_result = PermissionCheckResult(
        action="test",
        allowed=True,
        level=AutonomyLevel.AUTONOMOUS,
        requires_confirmation=False,
    )
    manager.check_permission = MagicMock(return_value=mock_result)
    manager.request_permission = AsyncMock(return_value=True)
    manager.is_in_scope = MagicMock(return_value=True)
    manager.is_prohibited = MagicMock(return_value=False)
    manager.requires_confirmation = MagicMock(return_value=False)
    # Add resource_usage attribute for post_execute
    manager.resource_usage = MagicMock()
    manager.resource_usage.record_file_modification = MagicMock()
    return manager


@pytest.fixture
def mock_constraint_manager() -> MagicMock:
    """Create a reusable mock constraint manager.

    Returns:
        MagicMock configured as a constraint manager.
    """
    manager = MagicMock(spec=ConstraintManager)
    manager.check_constraints = MagicMock(return_value=[])
    manager.validate_action = MagicMock(return_value=True)
    manager.get_active_constraints = MagicMock(return_value=[])
    return manager


@pytest.fixture
def mock_command_runner() -> MagicMock:
    """Create a reusable mock command runner.

    Returns:
        MagicMock configured as a command runner.
    """
    runner = MagicMock(spec=CommandRunner)
    runner.run = MagicMock(return_value=(0, "Success", ""))
    runner.run_async = AsyncMock(return_value=(0, "Success", ""))
    return runner


# =============================================================================
# Session Memory Fixtures
# =============================================================================


@pytest.fixture
def sample_session_memory(tmp_path: Path) -> SessionMemory:
    """Create a pre-configured session memory for testing.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        SessionMemory with default sections initialized.
    """
    memory = SessionMemory(memory_file=tmp_path / "CLAUDE.md")
    memory.create_default()
    memory.add_to_section("Build Commands", {"pytest": "Run all tests"})
    memory.add_to_section("Test Commands", {"pytest -v": "Run tests verbosely"})
    return memory


@pytest.fixture
def empty_session_memory(tmp_path: Path) -> SessionMemory:
    """Create an empty session memory for testing.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Empty SessionMemory instance.
    """
    return SessionMemory(memory_file=tmp_path / "CLAUDE.md")


# =============================================================================
# Agent Context Fixtures
# =============================================================================


@pytest.fixture
def agent_context(basic_config: AgenticConfig, tmp_path: Path) -> AgentContext:
    """Create an agent context for testing.

    Args:
        basic_config: Basic configuration fixture.
        tmp_path: Pytest temporary path fixture.

    Returns:
        Configured AgentContext instance.
    """
    return AgentContext(
        config=basic_config,
        working_dir=tmp_path,
        constraint_manager=ConstraintManager(config=basic_config),
        permission_manager=PermissionManager(config=basic_config, base_path=tmp_path),
        session_memory=SessionMemory(memory_file=tmp_path / "CLAUDE.md"),
        command_runner=CommandRunner(default_cwd=tmp_path),
        request_id="test-request-001",
    )


@pytest.fixture
def agent_context_with_mocks(
    basic_config: AgenticConfig,
    tmp_path: Path,
    mock_permission_manager: MagicMock,
    mock_constraint_manager: MagicMock,
    mock_command_runner: MagicMock,
) -> AgentContext:
    """Create an agent context with mocked dependencies.

    Args:
        basic_config: Basic configuration fixture.
        tmp_path: Pytest temporary path fixture.
        mock_permission_manager: Mocked permission manager.
        mock_constraint_manager: Mocked constraint manager.
        mock_command_runner: Mocked command runner.

    Returns:
        AgentContext with mocked dependencies.
    """
    return AgentContext(
        config=basic_config,
        working_dir=tmp_path,
        constraint_manager=mock_constraint_manager,
        permission_manager=mock_permission_manager,
        session_memory=SessionMemory(memory_file=tmp_path / "CLAUDE.md"),
        command_runner=mock_command_runner,
        request_id="test-request-mocked",
    )


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_python_file(tmp_path: Path) -> Path:
    """Create a sample Python file for testing.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to the created file.
    """
    file_path = tmp_path / "sample.py"
    file_path.write_text('''"""Sample module for testing."""


def hello(name: str) -> str:
    """Return a greeting.

    Args:
        name: The name to greet.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"


class Calculator:
    """Simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b
''')
    return file_path


@pytest.fixture
def sample_test_file(tmp_path: Path) -> Path:
    """Create a sample test file for testing.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to the created test file.
    """
    (tmp_path / "tests").mkdir(exist_ok=True)
    file_path = tmp_path / "tests" / "test_sample.py"
    file_path.write_text('''"""Tests for sample module."""

import pytest


def test_hello():
    """Test the hello function."""
    from sample import hello
    assert hello("World") == "Hello, World!"


def test_calculator_add():
    """Test calculator addition."""
    from sample import Calculator
    calc = Calculator()
    assert calc.add(2, 3) == 5


def test_calculator_subtract():
    """Test calculator subtraction."""
    from sample import Calculator
    calc = Calculator()
    assert calc.subtract(5, 3) == 2
''')
    return file_path


@pytest.fixture
def sample_project_structure(tmp_path: Path) -> Path:
    """Create a sample project structure for testing.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to the project root.
    """
    # Create directories
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "unit").mkdir()
    (tmp_path / "docs").mkdir()

    # Create source files
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "main.py").write_text("def main(): pass")
    (tmp_path / "src" / "module" / "__init__.py").write_text("")
    (tmp_path / "src" / "module" / "core.py").write_text("class Core: pass")

    # Create test files
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "tests" / "conftest.py").write_text("import pytest")
    (tmp_path / "tests" / "unit" / "__init__.py").write_text("")
    (tmp_path / "tests" / "unit" / "test_main.py").write_text(
        "def test_main(): assert True"
    )

    # Create config files
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "test-project"\nversion = "0.1.0"'
    )

    return tmp_path


# =============================================================================
# Async Test Helpers
# =============================================================================


@pytest.fixture
def async_mock_response() -> AsyncMock:
    """Create an async mock for testing async operations.

    Returns:
        AsyncMock configured with a default return value.
    """
    mock = AsyncMock()
    mock.return_value = {"status": "success", "data": {}}
    return mock


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=False)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean up environment variables that might affect tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    # Remove any agentic framework environment variables
    env_prefixes = ["AGENTIC_", "MCTS_", "CIV6AZ_"]
    import os

    for key in list(os.environ.keys()):
        for prefix in env_prefixes:
            if key.startswith(prefix):
                monkeypatch.delenv(key, raising=False)
