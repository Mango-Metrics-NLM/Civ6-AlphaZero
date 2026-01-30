"""
Sub-Agent Configuration for Agentic Coding System

Defines reusable sub-agent types for Claude Code integration.
All configurations are dynamic with no hardcoded values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..utils.logging_config import get_logger

logger = get_logger("sub_agents")


class SubAgentType(Enum):
    """Types of sub-agents available for task execution."""

    # Core agents
    EXPLORE = "Explore"
    PLAN = "Plan"
    BASH = "Bash"
    GENERAL_PURPOSE = "general-purpose"

    # Specialized agents (customizable)
    CODE_REVIEW = "code-review"
    TEST_RUNNER = "test-runner"
    DOCUMENTATION = "documentation"
    REFACTOR = "refactor"
    DEBUG = "debug"
    SECURITY_SCAN = "security-scan"

    @classmethod
    def from_string(cls, value: str) -> SubAgentType:
        """Create from string value.

        Args:
            value: String representation.

        Returns:
            SubAgentType enum value.
        """
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        raise ValueError(f"Unknown sub-agent type: {value}")


@dataclass
class SubAgentCapability:
    """Defines a capability of a sub-agent.

    Attributes:
        name: Capability name
        description: What this capability provides
        requires_tools: Tools required for this capability
        enabled: Whether capability is enabled
    """

    name: str
    description: str
    requires_tools: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class SubAgentConfig:
    """Configuration for a sub-agent.

    All values are configurable, no hardcoded defaults in the class itself.
    Defaults are provided through factory methods or configuration files.

    Attributes:
        agent_type: Type of the sub-agent
        name: Display name for the agent
        description: Description of what this agent does
        prompt_template: Template for generating the agent's prompt
        capabilities: List of agent capabilities
        timeout_seconds: Maximum execution time
        max_retries: Number of retries on failure
        model: Model to use (optional override)
        run_in_background: Whether to run asynchronously
        tools: List of tools available to this agent
        context_access: Whether agent has access to current context
        output_format: Expected output format
        metadata: Additional configuration metadata
    """

    agent_type: SubAgentType
    name: str
    description: str = ""
    prompt_template: str = ""
    capabilities: list[SubAgentCapability] = field(default_factory=list)
    timeout_seconds: float = 300.0
    max_retries: int = 3
    model: str | None = None
    run_in_background: bool = False
    tools: list[str] = field(default_factory=list)
    context_access: bool = True
    output_format: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "agent_type": self.agent_type.value,
            "name": self.name,
            "description": self.description,
            "prompt_template": self.prompt_template,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "requires_tools": cap.requires_tools,
                    "enabled": cap.enabled,
                }
                for cap in self.capabilities
            ],
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "model": self.model,
            "run_in_background": self.run_in_background,
            "tools": self.tools,
            "context_access": self.context_access,
            "output_format": self.output_format,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubAgentConfig:
        """Create from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            SubAgentConfig instance.
        """
        capabilities = [
            SubAgentCapability(
                name=cap.get("name", ""),
                description=cap.get("description", ""),
                requires_tools=cap.get("requires_tools", []),
                enabled=cap.get("enabled", True),
            )
            for cap in data.get("capabilities", [])
        ]

        return cls(
            agent_type=SubAgentType.from_string(data.get("agent_type", "general-purpose")),
            name=data.get("name", ""),
            description=data.get("description", ""),
            prompt_template=data.get("prompt_template", ""),
            capabilities=capabilities,
            timeout_seconds=data.get("timeout_seconds", 300.0),
            max_retries=data.get("max_retries", 3),
            model=data.get("model"),
            run_in_background=data.get("run_in_background", False),
            tools=data.get("tools", []),
            context_access=data.get("context_access", True),
            output_format=data.get("output_format", "text"),
            metadata=data.get("metadata", {}),
        )

    def with_overrides(self, **kwargs: Any) -> SubAgentConfig:
        """Create a new config with overridden values.

        Args:
            **kwargs: Values to override.

        Returns:
            New SubAgentConfig with overrides applied.
        """
        data = self.to_dict()
        data.update(kwargs)
        return SubAgentConfig.from_dict(data)


class SubAgentRegistry:
    """Registry for managing sub-agent configurations.

    Provides centralized management of sub-agent definitions
    with support for custom agents and overrides.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._agents: dict[str, SubAgentConfig] = {}
        self._factories: dict[str, Callable[..., SubAgentConfig]] = {}

        # Register built-in agents
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in agent configurations."""
        # Explore agent - for codebase exploration
        self.register(
            SubAgentConfig(
                agent_type=SubAgentType.EXPLORE,
                name="Explore",
                description="Fast agent specialized for exploring codebases",
                prompt_template=(
                    "Explore the codebase to answer: ${query}\n"
                    "Thoroughness level: ${thoroughness:medium}\n"
                    "Focus areas: ${focus_areas:[]}"
                ),
                capabilities=[
                    SubAgentCapability(
                        name="file_search",
                        description="Search for files by pattern",
                        requires_tools=["Glob"],
                    ),
                    SubAgentCapability(
                        name="code_search",
                        description="Search code for keywords",
                        requires_tools=["Grep"],
                    ),
                    SubAgentCapability(
                        name="read_files",
                        description="Read file contents",
                        requires_tools=["Read"],
                    ),
                ],
                tools=["Glob", "Grep", "Read", "Bash"],
                timeout_seconds=600.0,
            )
        )

        # Plan agent - for implementation planning
        self.register(
            SubAgentConfig(
                agent_type=SubAgentType.PLAN,
                name="Plan",
                description="Software architect agent for designing implementation plans",
                prompt_template=(
                    "Create an implementation plan for: ${task}\n"
                    "Constraints: ${constraints:[]}\n"
                    "Consider architectural trade-offs and identify critical files."
                ),
                capabilities=[
                    SubAgentCapability(
                        name="analyze_codebase",
                        description="Analyze codebase structure",
                        requires_tools=["Glob", "Grep", "Read"],
                    ),
                    SubAgentCapability(
                        name="identify_dependencies",
                        description="Identify module dependencies",
                        requires_tools=["Read"],
                    ),
                ],
                tools=["Glob", "Grep", "Read", "Bash", "WebSearch"],
                timeout_seconds=900.0,
            )
        )

        # Bash agent - for command execution
        self.register(
            SubAgentConfig(
                agent_type=SubAgentType.BASH,
                name="Bash",
                description="Command execution specialist for running bash commands",
                prompt_template=(
                    "Execute the following task using bash commands: ${task}\n"
                    "Working directory: ${working_dir}\n"
                    "Constraints: ${constraints:[]}"
                ),
                capabilities=[
                    SubAgentCapability(
                        name="execute_commands",
                        description="Execute shell commands",
                        requires_tools=["Bash"],
                    ),
                ],
                tools=["Bash"],
                timeout_seconds=300.0,
            )
        )

        # General purpose agent
        self.register(
            SubAgentConfig(
                agent_type=SubAgentType.GENERAL_PURPOSE,
                name="General Purpose",
                description="General-purpose agent for complex, multi-step tasks",
                prompt_template=(
                    "Complete the following task: ${task}\n"
                    "Context: ${context:}\n"
                    "This task should be completed autonomously."
                ),
                capabilities=[
                    SubAgentCapability(
                        name="full_toolset",
                        description="Access to all available tools",
                        requires_tools=["*"],
                    ),
                ],
                tools=["*"],  # All tools
                timeout_seconds=1800.0,
            )
        )

        # Test runner agent
        self.register(
            SubAgentConfig(
                agent_type=SubAgentType.TEST_RUNNER,
                name="Test Runner",
                description="Agent for running and analyzing tests",
                prompt_template=(
                    "Run tests and analyze results: ${test_command}\n"
                    "Focus: ${focus:all}\n"
                    "Report failures and suggest fixes."
                ),
                capabilities=[
                    SubAgentCapability(
                        name="run_tests",
                        description="Execute test suites",
                        requires_tools=["Bash"],
                    ),
                    SubAgentCapability(
                        name="analyze_failures",
                        description="Analyze test failures",
                        requires_tools=["Read", "Grep"],
                    ),
                ],
                tools=["Bash", "Read", "Grep"],
                timeout_seconds=600.0,
            )
        )

        # Code review agent
        self.register(
            SubAgentConfig(
                agent_type=SubAgentType.CODE_REVIEW,
                name="Code Review",
                description="Agent for reviewing code changes",
                prompt_template=(
                    "Review the following code changes: ${changes}\n"
                    "Check for: ${check_items:security,performance,style}\n"
                    "Provide actionable feedback."
                ),
                capabilities=[
                    SubAgentCapability(
                        name="static_analysis",
                        description="Perform static code analysis",
                        requires_tools=["Read", "Grep"],
                    ),
                    SubAgentCapability(
                        name="style_check",
                        description="Check code style compliance",
                        requires_tools=["Bash"],
                    ),
                ],
                tools=["Read", "Grep", "Bash", "Glob"],
                timeout_seconds=600.0,
            )
        )

        # Refactor agent
        self.register(
            SubAgentConfig(
                agent_type=SubAgentType.REFACTOR,
                name="Refactor",
                description="Agent for code refactoring tasks",
                prompt_template=(
                    "Refactor code: ${target}\n"
                    "Goal: ${goal}\n"
                    "Maintain backward compatibility: ${backward_compat:true}\n"
                    "Run tests after refactoring."
                ),
                capabilities=[
                    SubAgentCapability(
                        name="analyze_structure",
                        description="Analyze code structure",
                        requires_tools=["Read", "Grep", "Glob"],
                    ),
                    SubAgentCapability(
                        name="modify_code",
                        description="Modify source code",
                        requires_tools=["Edit", "Write"],
                    ),
                    SubAgentCapability(
                        name="validate_changes",
                        description="Validate refactoring didn't break anything",
                        requires_tools=["Bash"],
                    ),
                ],
                tools=["Read", "Write", "Edit", "Grep", "Glob", "Bash"],
                timeout_seconds=1200.0,
            )
        )

        # Debug agent
        self.register(
            SubAgentConfig(
                agent_type=SubAgentType.DEBUG,
                name="Debug",
                description="Agent for debugging issues",
                prompt_template=(
                    "Debug the following issue: ${issue}\n"
                    "Error message: ${error:}\n"
                    "Stack trace: ${stack_trace:}\n"
                    "Find root cause and suggest fix."
                ),
                capabilities=[
                    SubAgentCapability(
                        name="trace_execution",
                        description="Trace code execution paths",
                        requires_tools=["Read", "Grep"],
                    ),
                    SubAgentCapability(
                        name="analyze_logs",
                        description="Analyze log files",
                        requires_tools=["Read", "Grep", "Bash"],
                    ),
                ],
                tools=["Read", "Grep", "Bash", "Glob"],
                timeout_seconds=900.0,
            )
        )

        # Security scan agent
        self.register(
            SubAgentConfig(
                agent_type=SubAgentType.SECURITY_SCAN,
                name="Security Scan",
                description="Agent for security analysis",
                prompt_template=(
                    "Perform security analysis on: ${target}\n"
                    "Check for: ${vulnerabilities:OWASP Top 10}\n"
                    "Severity threshold: ${severity:medium}"
                ),
                capabilities=[
                    SubAgentCapability(
                        name="vulnerability_scan",
                        description="Scan for known vulnerabilities",
                        requires_tools=["Grep", "Read", "Bash"],
                    ),
                    SubAgentCapability(
                        name="dependency_audit",
                        description="Audit dependencies",
                        requires_tools=["Bash"],
                    ),
                ],
                tools=["Read", "Grep", "Bash", "Glob"],
                timeout_seconds=600.0,
            )
        )

        # Documentation agent
        self.register(
            SubAgentConfig(
                agent_type=SubAgentType.DOCUMENTATION,
                name="Documentation",
                description="Agent for documentation tasks",
                prompt_template=(
                    "Document: ${target}\n"
                    "Style: ${style:Google}\n"
                    "Include: ${include:description,parameters,returns,examples}"
                ),
                capabilities=[
                    SubAgentCapability(
                        name="analyze_code",
                        description="Analyze code for documentation",
                        requires_tools=["Read", "Grep"],
                    ),
                    SubAgentCapability(
                        name="generate_docs",
                        description="Generate documentation",
                        requires_tools=["Write", "Edit"],
                    ),
                ],
                tools=["Read", "Grep", "Write", "Edit", "Glob"],
                timeout_seconds=600.0,
            )
        )

    def register(self, config: SubAgentConfig) -> None:
        """Register a sub-agent configuration.

        Args:
            config: Sub-agent configuration to register.
        """
        key = config.agent_type.value.lower()
        self._agents[key] = config
        logger.debug(f"Registered sub-agent: {config.name} ({key})")

    def register_factory(
        self,
        name: str,
        factory: Callable[..., SubAgentConfig],
    ) -> None:
        """Register a factory function for creating agents.

        Args:
            name: Factory name.
            factory: Factory function.
        """
        self._factories[name] = factory
        logger.debug(f"Registered sub-agent factory: {name}")

    def get(self, agent_type: str | SubAgentType) -> SubAgentConfig | None:
        """Get a sub-agent configuration.

        Args:
            agent_type: Agent type to retrieve.

        Returns:
            SubAgentConfig or None if not found.
        """
        if isinstance(agent_type, SubAgentType):
            key = agent_type.value.lower()
        else:
            key = agent_type.lower()

        return self._agents.get(key)

    def get_or_create(
        self,
        agent_type: str | SubAgentType,
        **kwargs: Any,
    ) -> SubAgentConfig:
        """Get or create a sub-agent configuration.

        Args:
            agent_type: Agent type.
            **kwargs: Override values.

        Returns:
            SubAgentConfig with any overrides applied.

        Raises:
            KeyError: If agent type not found.
        """
        config = self.get(agent_type)
        if config is None:
            raise KeyError(f"Sub-agent not found: {agent_type}")

        if kwargs:
            return config.with_overrides(**kwargs)

        return config

    def create_from_factory(
        self,
        factory_name: str,
        **kwargs: Any,
    ) -> SubAgentConfig:
        """Create a sub-agent using a factory.

        Args:
            factory_name: Name of the factory.
            **kwargs: Arguments for the factory.

        Returns:
            Created SubAgentConfig.

        Raises:
            KeyError: If factory not found.
        """
        if factory_name not in self._factories:
            raise KeyError(f"Factory not found: {factory_name}")

        return self._factories[factory_name](**kwargs)

    def list_agents(self) -> list[str]:
        """List all registered agent types.

        Returns:
            List of agent type names.
        """
        return list(self._agents.keys())

    def to_dict(self) -> dict[str, Any]:
        """Convert registry to dictionary.

        Returns:
            Dictionary with all agent configurations.
        """
        return {name: config.to_dict() for name, config in self._agents.items()}


# Global registry instance
_registry: SubAgentRegistry | None = None


def get_registry() -> SubAgentRegistry:
    """Get the global sub-agent registry.

    Returns:
        SubAgentRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = SubAgentRegistry()
    return _registry


def create_sub_agent_config(
    agent_type: str | SubAgentType,
    **kwargs: Any,
) -> SubAgentConfig:
    """Factory function to create a sub-agent configuration.

    Args:
        agent_type: Type of agent to create.
        **kwargs: Override values.

    Returns:
        Configured SubAgentConfig.
    """
    registry = get_registry()
    return registry.get_or_create(agent_type, **kwargs)


def register_custom_agent(config: SubAgentConfig) -> None:
    """Register a custom sub-agent.

    Args:
        config: Configuration for the custom agent.
    """
    registry = get_registry()
    registry.register(config)
