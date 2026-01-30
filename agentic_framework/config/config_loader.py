"""
Configuration Loader for Agentic Framework

Handles loading, validation, merging, and access to configuration.
Supports environment variable substitution and runtime overrides.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Configuration validation failed: {'; '.join(errors)}")


@dataclass
class ObjectiveConfig:
    """Configuration for the objective function."""

    system_intent: str = ""
    success_criteria: list[dict[str, Any]] = field(default_factory=list)
    problem_description: dict[str, str] = field(default_factory=dict)


@dataclass
class ConstraintConfig:
    """Configuration for constraints."""

    hard: list[dict[str, Any]] = field(default_factory=list)
    soft: list[dict[str, Any]] = field(default_factory=list)
    anti: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ScopeConfig:
    """Configuration for scope boundaries."""

    in_scope: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)


@dataclass
class AutonomyConfig:
    """Configuration for autonomy levels."""

    autonomous: list[str] = field(default_factory=list)
    confirm_first: list[str] = field(default_factory=list)
    prohibited: list[str] = field(default_factory=list)


@dataclass
class ResourceBudgetConfig:
    """Configuration for resource budgets."""

    max_iterations: int = 5
    max_files_per_pass: int = 20
    exploration_timeout_seconds: int = 600


@dataclass
class PermissionConfig:
    """Configuration for permissions architecture."""

    scope: ScopeConfig = field(default_factory=ScopeConfig)
    autonomy: AutonomyConfig = field(default_factory=AutonomyConfig)
    resource_budget: ResourceBudgetConfig = field(default_factory=ResourceBudgetConfig)


@dataclass
class VerificationCommand:
    """Configuration for a verification command."""

    name: str = ""
    command: str = ""
    timeout_seconds: int = 300
    fail_fast: bool = False
    retry_count: int = 0


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling."""

    auto_fix_patterns: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FeedbackConfig:
    """Configuration for the feedback loop."""

    verification_commands: list[VerificationCommand] = field(default_factory=list)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    success_verification: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SessionSection:
    """Configuration for a session memory section."""

    name: str = ""
    persist: bool = True
    format: str = "markdown_list"


@dataclass
class SessionConfig:
    """Configuration for session memory."""

    memory_file: str = "CLAUDE.md"
    sections: list[SessionSection] = field(default_factory=list)


@dataclass
class ExecutionPhase:
    """Configuration for an execution phase."""

    phase: str = ""
    description: str = ""
    agent: str = ""


@dataclass
class ExecutionConfig:
    """Configuration for execution protocol."""

    initial_actions: list[dict[str, Any]] = field(default_factory=list)
    implementation_order: list[ExecutionPhase] = field(default_factory=list)
    completion_checklist: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LoggingHandlerConfig:
    """Configuration for a logging handler."""

    type: str = "console"
    level: str = "INFO"
    destination: str | None = None


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: list[LoggingHandlerConfig] = field(default_factory=list)


@dataclass
class AgentSpecificConfig:
    """Configuration for a specific agent."""

    timeout_seconds: int = 300
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentsConfig:
    """Configuration for all agents."""

    planning: AgentSpecificConfig = field(default_factory=AgentSpecificConfig)
    implementation: AgentSpecificConfig = field(default_factory=AgentSpecificConfig)
    testing: AgentSpecificConfig = field(default_factory=AgentSpecificConfig)
    verification: AgentSpecificConfig = field(default_factory=AgentSpecificConfig)
    documentation: AgentSpecificConfig = field(default_factory=AgentSpecificConfig)


@dataclass
class AgenticConfig:
    """Main configuration class for the Agentic Framework."""

    version: str = "1.0"
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    permissions: PermissionConfig = field(default_factory=PermissionConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgenticConfig:
        """Create configuration from dictionary."""
        config = cls()

        if "version" in data:
            config.version = data["version"]

        if "objective" in data:
            obj_data = data["objective"]
            config.objective = ObjectiveConfig(
                system_intent=obj_data.get("system_intent", ""),
                success_criteria=obj_data.get("success_criteria", []),
                problem_description=obj_data.get("problem_description", {}),
            )

        if "constraints" in data:
            const_data = data["constraints"]
            config.constraints = ConstraintConfig(
                hard=const_data.get("hard", []),
                soft=const_data.get("soft", []),
                anti=const_data.get("anti", []),
            )

        if "permissions" in data:
            perm_data = data["permissions"]
            scope_data = perm_data.get("scope", {})
            autonomy_data = perm_data.get("autonomy", {})
            budget_data = perm_data.get("resource_budget", {})

            config.permissions = PermissionConfig(
                scope=ScopeConfig(
                    in_scope=scope_data.get("in_scope", []),
                    out_of_scope=scope_data.get("out_of_scope", []),
                ),
                autonomy=AutonomyConfig(
                    autonomous=autonomy_data.get("autonomous", []),
                    confirm_first=autonomy_data.get("confirm_first", []),
                    prohibited=autonomy_data.get("prohibited", []),
                ),
                resource_budget=ResourceBudgetConfig(
                    max_iterations=budget_data.get("max_iterations", 5),
                    max_files_per_pass=budget_data.get("max_files_per_pass", 20),
                    exploration_timeout_seconds=budget_data.get(
                        "exploration_timeout_seconds", 600
                    ),
                ),
            )

        if "feedback" in data:
            fb_data = data["feedback"]
            verification_commands = []
            for vc in fb_data.get("verification_commands", []):
                verification_commands.append(
                    VerificationCommand(
                        name=vc.get("name", ""),
                        command=vc.get("command", ""),
                        timeout_seconds=vc.get("timeout_seconds", 300),
                        fail_fast=vc.get("fail_fast", False),
                        retry_count=vc.get("retry_count", 0),
                    )
                )

            error_handling_data = fb_data.get("error_handling", {})
            config.feedback = FeedbackConfig(
                verification_commands=verification_commands,
                error_handling=ErrorHandlingConfig(
                    auto_fix_patterns=error_handling_data.get("auto_fix_patterns", [])
                ),
                success_verification=fb_data.get("success_verification", []),
            )

        if "session" in data:
            sess_data = data["session"]
            sections = []
            for sec in sess_data.get("sections", []):
                sections.append(
                    SessionSection(
                        name=sec.get("name", ""),
                        persist=sec.get("persist", True),
                        format=sec.get("format", "markdown_list"),
                    )
                )

            config.session = SessionConfig(
                memory_file=sess_data.get("memory_file", "CLAUDE.md"),
                sections=sections,
            )

        if "execution" in data:
            exec_data = data["execution"]
            phases = []
            for phase in exec_data.get("implementation_order", []):
                phases.append(
                    ExecutionPhase(
                        phase=phase.get("phase", ""),
                        description=phase.get("description", ""),
                        agent=phase.get("agent", ""),
                    )
                )

            config.execution = ExecutionConfig(
                initial_actions=exec_data.get("initial_actions", []),
                implementation_order=phases,
                completion_checklist=exec_data.get("completion_checklist", []),
            )

        if "logging" in data:
            log_data = data["logging"]
            handlers = []
            for handler in log_data.get("handlers", []):
                handlers.append(
                    LoggingHandlerConfig(
                        type=handler.get("type", "console"),
                        level=handler.get("level", "INFO"),
                        destination=handler.get("destination"),
                    )
                )

            config.logging = LoggingConfig(
                level=log_data.get("level", "INFO"),
                format=log_data.get(
                    "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                ),
                handlers=handlers,
            )

        if "agents" in data:
            agents_data = data["agents"]
            config.agents = AgentsConfig()

            for agent_name in [
                "planning",
                "implementation",
                "testing",
                "verification",
                "documentation",
            ]:
                if agent_name in agents_data:
                    agent_data = agents_data[agent_name]
                    agent_config = AgentSpecificConfig(
                        timeout_seconds=agent_data.get("timeout_seconds", 300),
                        extra={
                            k: v
                            for k, v in agent_data.items()
                            if k != "timeout_seconds"
                        },
                    )
                    setattr(config.agents, agent_name, agent_config)

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "version": self.version,
            "objective": {
                "system_intent": self.objective.system_intent,
                "success_criteria": self.objective.success_criteria,
                "problem_description": self.objective.problem_description,
            },
            "constraints": {
                "hard": self.constraints.hard,
                "soft": self.constraints.soft,
                "anti": self.constraints.anti,
            },
            "permissions": {
                "scope": {
                    "in_scope": self.permissions.scope.in_scope,
                    "out_of_scope": self.permissions.scope.out_of_scope,
                },
                "autonomy": {
                    "autonomous": self.permissions.autonomy.autonomous,
                    "confirm_first": self.permissions.autonomy.confirm_first,
                    "prohibited": self.permissions.autonomy.prohibited,
                },
                "resource_budget": {
                    "max_iterations": self.permissions.resource_budget.max_iterations,
                    "max_files_per_pass": self.permissions.resource_budget.max_files_per_pass,
                    "exploration_timeout_seconds": self.permissions.resource_budget.exploration_timeout_seconds,
                },
            },
            "feedback": {
                "verification_commands": [
                    {
                        "name": vc.name,
                        "command": vc.command,
                        "timeout_seconds": vc.timeout_seconds,
                        "fail_fast": vc.fail_fast,
                        "retry_count": vc.retry_count,
                    }
                    for vc in self.feedback.verification_commands
                ],
                "error_handling": {
                    "auto_fix_patterns": self.feedback.error_handling.auto_fix_patterns
                },
                "success_verification": self.feedback.success_verification,
            },
            "session": {
                "memory_file": self.session.memory_file,
                "sections": [
                    {
                        "name": sec.name,
                        "persist": sec.persist,
                        "format": sec.format,
                    }
                    for sec in self.session.sections
                ],
            },
            "execution": {
                "initial_actions": self.execution.initial_actions,
                "implementation_order": [
                    {
                        "phase": phase.phase,
                        "description": phase.description,
                        "agent": phase.agent,
                    }
                    for phase in self.execution.implementation_order
                ],
                "completion_checklist": self.execution.completion_checklist,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "handlers": [
                    {
                        "type": h.type,
                        "level": h.level,
                        "destination": h.destination,
                    }
                    for h in self.logging.handlers
                ],
            },
        }


class ConfigLoader:
    """Loads and manages configuration from YAML files."""

    ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

    def __init__(self, config_dir: Path | None = None) -> None:
        """Initialize the config loader.

        Args:
            config_dir: Directory containing configuration files.
                       Defaults to the config directory in the package.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = Path(config_dir)
        self._config: AgenticConfig | None = None
        self._raw_config: dict[str, Any] = {}

    def _substitute_env_vars(self, value: Any) -> Any:
        """Substitute environment variables in configuration values.

        Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.
        """
        if isinstance(value, str):
            matches = self.ENV_VAR_PATTERN.findall(value)
            for match in matches:
                if ":-" in match:
                    var_name, default = match.split(":-", 1)
                else:
                    var_name = match
                    default = ""

                env_value = os.environ.get(var_name.strip(), default)
                value = value.replace(f"${{{match}}}", env_value)
            return value
        elif isinstance(value, dict):
            return {k: self._substitute_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._substitute_env_vars(item) for item in value]
        return value

    def load_yaml(self, file_path: Path) -> dict[str, Any]:
        """Load a YAML file with environment variable substitution.

        Args:
            file_path: Path to the YAML file.

        Returns:
            Parsed configuration dictionary.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        logger.debug(f"Loading configuration from {file_path}")

        with open(file_path) as f:
            data = yaml.safe_load(f) or {}

        return self._substitute_env_vars(data)

    def load(
        self,
        project_config_path: Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> AgenticConfig:
        """Load configuration with defaults, project config, and overrides.

        Configuration is merged in order:
        1. Default config (from package)
        2. Project config (if provided)
        3. Runtime overrides (if provided)

        Args:
            project_config_path: Path to project-specific configuration.
            overrides: Runtime configuration overrides.

        Returns:
            Merged configuration object.
        """
        # Load default config
        default_path = self.config_dir / "default_config.yaml"
        if default_path.exists():
            self._raw_config = self.load_yaml(default_path)
            logger.info(f"Loaded default config from {default_path}")
        else:
            logger.warning(f"Default config not found at {default_path}")
            self._raw_config = {}

        # Merge project config if provided
        if project_config_path is not None:
            project_config = self.load_yaml(Path(project_config_path))
            self._raw_config = merge_configs(self._raw_config, project_config)
            logger.info(f"Merged project config from {project_config_path}")

        # Apply runtime overrides
        if overrides:
            self._raw_config = merge_configs(self._raw_config, overrides)
            logger.info("Applied runtime configuration overrides")

        # Create typed config object
        self._config = AgenticConfig.from_dict(self._raw_config)
        return self._config

    @property
    def config(self) -> AgenticConfig:
        """Get the current configuration."""
        if self._config is None:
            self._config = self.load()
        return self._config

    @property
    def raw_config(self) -> dict[str, Any]:
        """Get the raw configuration dictionary."""
        return self._raw_config

    def validate(self, config: AgenticConfig | None = None) -> list[str]:
        """Validate the configuration.

        Args:
            config: Configuration to validate. Defaults to loaded config.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[str] = []
        config = config or self._config

        if config is None:
            errors.append("No configuration loaded")
            return errors

        # Validate objective
        if not config.objective.system_intent:
            errors.append("objective.system_intent is required")

        if not config.objective.success_criteria:
            errors.append("objective.success_criteria must have at least one criterion")

        # Validate permissions
        if not config.permissions.scope.in_scope:
            errors.append("permissions.scope.in_scope must have at least one pattern")

        # Validate feedback
        if not config.feedback.verification_commands:
            errors.append("feedback.verification_commands should have at least one command")

        for vc in config.feedback.verification_commands:
            if not vc.command:
                errors.append(f"verification command '{vc.name}' has no command specified")

        return errors


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two configuration dictionaries.

    Override values take precedence. Lists are replaced, not merged.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        Merged configuration dictionary.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


# Global config instance for convenience
_global_loader: ConfigLoader | None = None


def load_config(
    project_config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
    config_dir: Path | None = None,
) -> AgenticConfig:
    """Load configuration (convenience function).

    Args:
        project_config_path: Path to project-specific configuration.
        overrides: Runtime configuration overrides.
        config_dir: Directory containing default configuration.

    Returns:
        Loaded configuration object.
    """
    global _global_loader
    _global_loader = ConfigLoader(config_dir=config_dir)
    return _global_loader.load(project_config_path, overrides)


def get_config() -> AgenticConfig:
    """Get the current global configuration.

    Returns:
        Current configuration object.

    Raises:
        RuntimeError: If no configuration has been loaded.
    """
    global _global_loader
    if _global_loader is None:
        _global_loader = ConfigLoader()
        _global_loader.load()
    return _global_loader.config
