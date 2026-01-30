"""
Base Agent for Agentic Framework

Provides the abstract base class and common functionality for all agents.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..config import AgenticConfig
from ..core.constraint_manager import ConstraintManager
from ..core.permission_manager import PermissionManager
from ..core.session_memory import SessionMemory
from ..utils.command_runner import CommandRunner
from ..utils.logging_config import LogContext, get_logger

logger = get_logger("base_agent")


class AgentStatus(Enum):
    """Status of an agent execution."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_CONFIRMATION = "waiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentContext:
    """Context passed to agents during execution."""

    config: AgenticConfig
    working_dir: Path
    constraint_manager: ConstraintManager
    permission_manager: PermissionManager
    session_memory: SessionMemory
    command_runner: CommandRunner

    # Runtime state
    request_id: str = ""
    parent_agent: str | None = None
    iteration: int = 0
    max_iterations: int = 5

    # Shared data between agents
    shared_data: dict[str, Any] = field(default_factory=dict)

    # Callbacks
    on_status_change: Callable[[str, AgentStatus], None] | None = None
    on_confirmation_needed: Callable[[str, str], bool] | None = None
    on_progress: Callable[[str, float, str], None] | None = None

    def get_shared(self, key: str, default: Any = None) -> Any:
        """Get a value from shared data.

        Args:
            key: Key to retrieve.
            default: Default value if not found.

        Returns:
            Value or default.
        """
        return self.shared_data.get(key, default)

    def set_shared(self, key: str, value: Any) -> None:
        """Set a value in shared data.

        Args:
            key: Key to set.
            value: Value to store.
        """
        self.shared_data[key] = value

    def increment_iteration(self) -> bool:
        """Increment iteration counter.

        Returns:
            True if more iterations are allowed.
        """
        self.iteration += 1
        return self.iteration <= self.max_iterations


@dataclass
class AgentResult:
    """Result of an agent execution."""

    agent_name: str
    status: AgentStatus
    success: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    files_modified: list[Path] = field(default_factory=list)
    next_agent: str | None = None

    def __bool__(self) -> bool:
        """Return True if execution was successful."""
        return self.success

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_seconds": self.duration_seconds,
            "files_modified": [str(f) for f in self.files_modified],
            "next_agent": self.next_agent,
        }


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Features:
    - Common lifecycle management
    - Context handling
    - Permission checking
    - Progress reporting
    - Error handling
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        timeout_seconds: float = 300.0,
    ) -> None:
        """Initialize the base agent.

        Args:
            name: Unique name for this agent.
            description: Description of what this agent does.
            timeout_seconds: Default timeout for agent execution.
        """
        self.name = name
        self.description = description
        self.timeout_seconds = timeout_seconds
        self._status = AgentStatus.IDLE
        self._context: AgentContext | None = None
        self._start_time: float = 0.0
        self._logger = get_logger(f"agent.{name}")

    @property
    def status(self) -> AgentStatus:
        """Get current agent status."""
        return self._status

    @status.setter
    def status(self, value: AgentStatus) -> None:
        """Set agent status and notify."""
        old_status = self._status
        self._status = value
        self._logger.debug(f"Status changed: {old_status.value} -> {value.value}")

        if self._context and self._context.on_status_change:
            self._context.on_status_change(self.name, value)

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time since start."""
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent.

        Args:
            context: Execution context.

        Returns:
            AgentResult with execution outcome.
        """
        self._context = context
        self._start_time = time.time()
        self.status = AgentStatus.RUNNING

        with LogContext(fields={"agent": self.name, "request_id": context.request_id}):
            try:
                self._logger.info(f"Starting agent: {self.name}")

                # Pre-execution checks
                if not self._pre_execute_checks(context):
                    return AgentResult(
                        agent_name=self.name,
                        status=AgentStatus.FAILED,
                        success=False,
                        message="Pre-execution checks failed",
                        duration_seconds=self.elapsed_seconds,
                    )

                # Run the agent-specific logic
                result = self._execute(context)

                # Post-execution processing
                self._post_execute(context, result)

                self.status = AgentStatus.COMPLETED if result.success else AgentStatus.FAILED
                result.duration_seconds = self.elapsed_seconds

                self._logger.info(
                    f"Agent completed: {self.name} - "
                    f"{'success' if result.success else 'failed'} "
                    f"({result.duration_seconds:.2f}s)"
                )

                return result

            except Exception as e:
                self._logger.exception(f"Agent execution failed: {e}")
                self.status = AgentStatus.FAILED

                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.FAILED,
                    success=False,
                    message=f"Exception during execution: {str(e)}",
                    errors=[str(e)],
                    duration_seconds=self.elapsed_seconds,
                )

    @abstractmethod
    def _execute(self, context: AgentContext) -> AgentResult:
        """Agent-specific execution logic.

        Args:
            context: Execution context.

        Returns:
            AgentResult with execution outcome.
        """
        pass

    def _pre_execute_checks(self, context: AgentContext) -> bool:
        """Run pre-execution checks.

        Args:
            context: Execution context.

        Returns:
            True if checks pass.
        """
        # Check iteration limit
        if context.iteration > context.max_iterations:
            self._logger.warning(f"Max iterations ({context.max_iterations}) exceeded")
            return False

        return True

    def _post_execute(self, context: AgentContext, result: AgentResult) -> None:
        """Run post-execution processing.

        Args:
            context: Execution context.
            result: Execution result.
        """
        # Update session memory if files were modified
        if result.files_modified:
            context.permission_manager.resource_usage.record_file_modification(
                len(result.files_modified)
            )

    def report_progress(self, progress: float, message: str = "") -> None:
        """Report progress to the context.

        Args:
            progress: Progress percentage (0.0 to 1.0).
            message: Optional progress message.
        """
        if self._context and self._context.on_progress:
            self._context.on_progress(self.name, progress, message)

    def request_confirmation(self, action: str, details: str = "") -> bool:
        """Request confirmation for an action.

        Args:
            action: Description of the action.
            details: Additional details.

        Returns:
            True if confirmed.
        """
        if self._context and self._context.on_confirmation_needed:
            self.status = AgentStatus.WAITING_CONFIRMATION
            result = self._context.on_confirmation_needed(action, details)
            self.status = AgentStatus.RUNNING
            return result

        # Default to False if no callback
        self._logger.warning(f"Confirmation requested but no callback: {action}")
        return False

    def check_permission(self, action: str, path: Path | None = None) -> bool:
        """Check if an action is permitted.

        Args:
            action: Description of the action.
            path: Optional path involved.

        Returns:
            True if action is permitted.
        """
        if self._context is None:
            return False

        result = self._context.permission_manager.check_permission(action, path)

        if not result.allowed:
            self._logger.warning(f"Action not permitted: {action} - {result.reason}")
            return False

        if result.requires_confirmation:
            return self.request_confirmation(action, result.reason or "")

        return True

    def run_command(
        self,
        command: str,
        timeout: float | None = None,
        cwd: Path | None = None,
    ) -> tuple[bool, str, str]:
        """Run a shell command.

        Args:
            command: Command to run.
            timeout: Optional timeout override.
            cwd: Optional working directory.

        Returns:
            Tuple of (success, stdout, stderr).
        """
        if self._context is None:
            return False, "", "No context available"

        result = self._context.command_runner.run(
            command,
            timeout_seconds=timeout or self.timeout_seconds,
            cwd=cwd or self._context.working_dir,
        )

        return result.success, result.stdout, result.stderr

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Dot-separated key path.
            default: Default value if not found.

        Returns:
            Configuration value or default.
        """
        if self._context is None or self._context.config is None:
            return default

        # Navigate nested config
        obj = self._context.config
        for part in key.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default

        return obj


class CompositeAgent(BaseAgent):
    """Agent that composes multiple sub-agents."""

    def __init__(
        self,
        name: str,
        agents: list[BaseAgent],
        description: str = "",
        stop_on_failure: bool = True,
    ) -> None:
        """Initialize the composite agent.

        Args:
            name: Unique name for this agent.
            agents: List of sub-agents to run.
            description: Description of what this agent does.
            stop_on_failure: Stop execution on first failure.
        """
        super().__init__(name, description)
        self.agents = agents
        self.stop_on_failure = stop_on_failure
        self._results: list[AgentResult] = []

    def _execute(self, context: AgentContext) -> AgentResult:
        """Execute all sub-agents.

        Args:
            context: Execution context.

        Returns:
            Combined AgentResult.
        """
        self._results.clear()
        all_success = True
        all_files_modified: list[Path] = []
        all_errors: list[str] = []
        all_warnings: list[str] = []

        for i, agent in enumerate(self.agents):
            self.report_progress(i / len(self.agents), f"Running {agent.name}")

            result = agent.execute(context)
            self._results.append(result)

            all_files_modified.extend(result.files_modified)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

            if not result.success:
                all_success = False
                if self.stop_on_failure:
                    break

        self.report_progress(1.0, "Completed")

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.COMPLETED if all_success else AgentStatus.FAILED,
            success=all_success,
            message=f"Executed {len(self._results)} sub-agents",
            data={"sub_results": [r.to_dict() for r in self._results]},
            errors=all_errors,
            warnings=all_warnings,
            files_modified=all_files_modified,
        )

    @property
    def results(self) -> list[AgentResult]:
        """Get results from all sub-agents."""
        return self._results
