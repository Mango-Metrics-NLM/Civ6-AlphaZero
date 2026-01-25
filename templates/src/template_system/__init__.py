"""
Template System for Agentic Coding.

This package provides reusable components for building agentic coding systems
with Claude Code, including sub-agent orchestration, logging, testing, and
configuration management.
"""
from __future__ import annotations

from template_system.agents import (
    AgentConfig,
    AgentOrchestrator,
    AgentRegistry,
    AgentResult,
    AgentType,
)
from template_system.config import (
    ConfigLoader,
    ConfigManager,
    EnvironmentConfig,
    TemplateConfig,
)
from template_system.factories import (
    ComponentFactory,
    FactoryRegistry,
)
from template_system.logging_config import (
    LogConfig,
    configure_logging,
    get_logger,
)
from template_system.testing import (
    TestConfig,
    TestRunner,
    TestSuite,
)

__version__ = "1.0.0"

__all__ = [
    # Agents
    "AgentConfig",
    "AgentOrchestrator",
    "AgentRegistry",
    "AgentResult",
    "AgentType",
    # Config
    "ConfigLoader",
    "ConfigManager",
    "EnvironmentConfig",
    "TemplateConfig",
    # Factories
    "ComponentFactory",
    "FactoryRegistry",
    # Logging
    "LogConfig",
    "configure_logging",
    "get_logger",
    # Testing
    "TestConfig",
    "TestRunner",
    "TestSuite",
]
