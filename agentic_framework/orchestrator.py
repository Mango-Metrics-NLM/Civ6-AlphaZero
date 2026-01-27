"""
Orchestrator for Agentic Framework

Coordinates the execution of all agents according to the execution protocol.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .config import AgenticConfig, load_config
from .core.constraint_manager import ConstraintManager
from .core.permission_manager import PermissionManager
from .core.session_memory import SessionMemory
from .core.feedback_loop import FeedbackLoop
from .core.template_parser import TemplateParser, ParsedTemplate
from .agents.base_agent import AgentContext, AgentResult, AgentStatus, BaseAgent
from .agents.planning_agent import PlanningAgent
from .agents.implementation_agent import ImplementationAgent
from .agents.testing_agent import TestingAgent
from .agents.verification_agent import VerificationAgent
from .agents.documentation_agent import DocumentationAgent
from .utils.logging_config import setup_logging, get_logger, LogContext
from .utils.command_runner import CommandRunner

logger = get_logger("orchestrator")


@dataclass
class OrchestratorResult:
    """Result of orchestrator execution."""

    success: bool
    agent_results: list[AgentResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    iterations: int = 0
    errors: list[str] = field(default_factory=list)
    change_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "agent_results": [r.to_dict() for r in self.agent_results],
            "total_duration_seconds": self.total_duration_seconds,
            "iterations": self.iterations,
            "errors": self.errors,
        }

    def summary(self) -> str:
        """Generate a summary."""
        passed = sum(1 for r in self.agent_results if r.success)
        failed = len(self.agent_results) - passed

        lines = [
            f"Orchestrator: {'SUCCESS' if self.success else 'FAILED'}",
            f"Agents run: {len(self.agent_results)} ({passed} passed, {failed} failed)",
            f"Duration: {self.total_duration_seconds:.2f}s",
            f"Iterations: {self.iterations}",
        ]

        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")

        return "\n".join(lines)


class Orchestrator:
    """Orchestrates the execution of the agentic workflow.

    Features:
    - Agent lifecycle management
    - Execution protocol following
    - Feedback loop integration
    - Progress tracking
    - Error handling and recovery
    """

    def __init__(
        self,
        config: Optional[AgenticConfig] = None,
        working_dir: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config: Configuration for the framework.
            working_dir: Working directory for operations.
            config_path: Path to project configuration file.
        """
        self.working_dir = working_dir or Path.cwd()

        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(project_config_path=config_path)
        else:
            self.config = load_config()

        # Set up logging
        setup_logging(
            level=self.config.logging.level,
            log_format=self.config.logging.format,
            handlers_config=[
                {"type": h.type, "level": h.level, "destination": h.destination}
                for h in self.config.logging.handlers
            ],
        )

        # Initialize components
        self.command_runner = CommandRunner(
            default_cwd=self.working_dir,
            default_timeout_seconds=300.0,
        )
        self.constraint_manager = ConstraintManager(
            config=self.config,
            command_runner=self.command_runner,
        )
        self.permission_manager = PermissionManager(
            config=self.config,
            base_path=self.working_dir,
        )
        self.session_memory = SessionMemory(
            memory_file=self.working_dir / self.config.session.memory_file,
            config=self.config,
        )
        self.feedback_loop = FeedbackLoop(
            config=self.config,
            command_runner=self.command_runner,
        )

        # Initialize agents
        self._agents: dict[str, BaseAgent] = {}
        self._register_default_agents()

        # Callbacks
        self._on_agent_start: Optional[Callable[[str], None]] = None
        self._on_agent_complete: Optional[Callable[[str, AgentResult], None]] = None
        self._on_progress: Optional[Callable[[str, float, str], None]] = None
        self._on_confirmation_needed: Optional[Callable[[str, str], bool]] = None

        # State
        self._request_id: str = ""
        self._agent_results: list[AgentResult] = []

    def _register_default_agents(self) -> None:
        """Register the default set of agents."""
        # Get agent configs
        agents_config = self.config.agents

        self._agents = {
            "planning": PlanningAgent(
                timeout_seconds=agents_config.planning.timeout_seconds,
            ),
            "implementation": ImplementationAgent(
                timeout_seconds=agents_config.implementation.timeout_seconds,
                max_lines_per_file=agents_config.implementation.extra.get("max_lines_per_file", 500),
            ),
            "testing": TestingAgent(
                timeout_seconds=agents_config.testing.timeout_seconds,
                coverage_threshold=agents_config.testing.extra.get("coverage_threshold", 80),
            ),
            "verification": VerificationAgent(
                timeout_seconds=agents_config.verification.timeout_seconds,
                strict_mode=agents_config.verification.extra.get("strict_mode", False),
            ),
            "documentation": DocumentationAgent(
                timeout_seconds=agents_config.documentation.timeout_seconds,
                auto_update_memory=agents_config.documentation.extra.get("auto_update_memory", True),
            ),
        }

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent.

        Args:
            agent: Agent to register.
        """
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name.

        Args:
            name: Agent name.

        Returns:
            Agent or None.
        """
        return self._agents.get(name)

    def set_callbacks(
        self,
        on_agent_start: Optional[Callable[[str], None]] = None,
        on_agent_complete: Optional[Callable[[str, AgentResult], None]] = None,
        on_progress: Optional[Callable[[str, float, str], None]] = None,
        on_confirmation_needed: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        """Set callback functions.

        Args:
            on_agent_start: Called when an agent starts.
            on_agent_complete: Called when an agent completes.
            on_progress: Called on progress updates.
            on_confirmation_needed: Called when confirmation is needed.
        """
        self._on_agent_start = on_agent_start
        self._on_agent_complete = on_agent_complete
        self._on_progress = on_progress
        self._on_confirmation_needed = on_confirmation_needed

    def _create_context(self) -> AgentContext:
        """Create an execution context.

        Returns:
            AgentContext for agent execution.
        """
        return AgentContext(
            config=self.config,
            working_dir=self.working_dir,
            constraint_manager=self.constraint_manager,
            permission_manager=self.permission_manager,
            session_memory=self.session_memory,
            command_runner=self.command_runner,
            request_id=self._request_id,
            max_iterations=self.config.permissions.resource_budget.max_iterations,
            on_status_change=self._handle_status_change,
            on_confirmation_needed=self._on_confirmation_needed,
            on_progress=self._on_progress,
        )

    def _handle_status_change(self, agent_name: str, status: AgentStatus) -> None:
        """Handle agent status changes.

        Args:
            agent_name: Name of the agent.
            status: New status.
        """
        logger.debug(f"Agent {agent_name} status: {status.value}")

    def execute(
        self,
        objective: Optional[str] = None,
        dry_run: bool = False,
    ) -> OrchestratorResult:
        """Execute the full orchestrated workflow.

        Args:
            objective: Optional objective override.
            dry_run: If True, only plan without executing.

        Returns:
            OrchestratorResult with execution outcome.
        """
        self._request_id = str(uuid.uuid4())[:8]
        self._agent_results.clear()
        start_time = time.time()

        with LogContext(fields={"request_id": self._request_id}):
            logger.info("Starting orchestrator execution")

            # Run initial actions
            self._run_initial_actions()

            # Create context
            context = self._create_context()

            # Set objective if provided
            if objective:
                context.set_shared("objective", objective)

            # Check constraints
            constraint_result = self.constraint_manager.check_all(self.working_dir)
            if not constraint_result.passed:
                logger.error("Constraint check failed")
                return OrchestratorResult(
                    success=False,
                    errors=[str(v) for v in constraint_result.hard_violations],
                    total_duration_seconds=time.time() - start_time,
                )

            # Execute agents according to implementation order
            execution_order = self.config.execution.implementation_order

            for phase in execution_order:
                agent_name = phase.agent
                if not agent_name:
                    continue

                agent = self._agents.get(agent_name)
                if agent is None:
                    logger.warning(f"Agent not found: {agent_name}")
                    continue

                # Check if dry run
                if dry_run and agent_name != "planning":
                    logger.info(f"Dry run: skipping {agent_name}")
                    continue

                # Notify start
                if self._on_agent_start:
                    self._on_agent_start(agent_name)

                # Execute agent
                logger.info(f"Executing agent: {agent_name}")
                result = agent.execute(context)
                self._agent_results.append(result)

                # Notify completion
                if self._on_agent_complete:
                    self._on_agent_complete(agent_name, result)

                # Check for failure
                if not result.success:
                    logger.warning(f"Agent {agent_name} failed")
                    # Try to recover or stop
                    if not self._attempt_recovery(context, result):
                        break

                # Increment iteration
                context.iteration += 1

            # Run completion checklist
            all_passed = self._run_completion_checklist(context)

            # Generate change summary
            change_summary = self._generate_change_summary(context)

            total_duration = time.time() - start_time
            success = all_passed and all(r.success for r in self._agent_results)

            result = OrchestratorResult(
                success=success,
                agent_results=self._agent_results,
                total_duration_seconds=total_duration,
                iterations=context.iteration,
                errors=[e for r in self._agent_results for e in r.errors],
                change_summary=change_summary,
            )

            logger.info(f"Orchestrator completed: {result.summary()}")
            return result

    def _run_initial_actions(self) -> None:
        """Run initial actions from the execution protocol."""
        for action in self.config.execution.initial_actions:
            action_name = action.get("action", "")
            command = action.get("command")
            required = action.get("required", True)

            logger.info(f"Running initial action: {action_name}")

            if action_name == "read_session_memory":
                self.session_memory.load()
            elif action_name == "scan_project_structure":
                if command:
                    self.command_runner.run(command)
            elif command:
                result = self.command_runner.run(command)
                if required and not result.success:
                    logger.warning(f"Required initial action failed: {action_name}")

    def _attempt_recovery(
        self, context: AgentContext, failed_result: AgentResult
    ) -> bool:
        """Attempt to recover from agent failure.

        Args:
            context: Execution context.
            failed_result: Result from failed agent.

        Returns:
            True if recovery was successful.
        """
        logger.info(f"Attempting recovery from {failed_result.agent_name} failure")

        # Check if we can retry
        if not context.increment_iteration():
            logger.warning("Max iterations reached, cannot recover")
            return False

        # If verification failed, try feedback loop
        if failed_result.agent_name == "verification":
            feedback_result = self.feedback_loop.run_feedback_loop(
                cwd=self.working_dir,
                fix_callback=lambda errors: self._handle_fix_request(context, errors),
            )
            return feedback_result.success

        return False

    def _handle_fix_request(
        self, context: AgentContext, errors: list[dict[str, Any]]
    ) -> bool:
        """Handle a request to fix errors.

        Args:
            context: Execution context.
            errors: List of errors to fix.

        Returns:
            True if fixes were applied.
        """
        # Store errors for implementation agent
        context.set_shared("errors_to_fix", errors)

        # Run implementation agent to fix
        impl_agent = self._agents.get("implementation")
        if impl_agent:
            result = impl_agent.execute(context)
            return result.success

        return False

    def _run_completion_checklist(self, context: AgentContext) -> bool:
        """Run the completion checklist.

        Args:
            context: Execution context.

        Returns:
            True if all checks passed.
        """
        all_passed = True

        for check_item in self.config.execution.completion_checklist:
            item = check_item.get("item", "")
            verification = check_item.get("verification")

            logger.info(f"Checking: {item}")

            if verification == "check_success_criteria":
                # Check if all success criteria are met
                for criterion in self.config.objective.success_criteria:
                    # TODO: Actually verify criteria
                    logger.info(f"  Criterion: {criterion.get('criterion', criterion)}")

            elif verification == "run_all_verifications":
                # Already run by verification agent
                verification_results = context.get_shared("verification_results", [])
                if not all(r.get("success", False) for r in verification_results):
                    all_passed = False

            elif verification == "check_memory_updated":
                if not self.session_memory.is_modified:
                    logger.info("  Session memory was not updated")

            elif verification == "generate_summary":
                # Summary already generated
                pass

        return all_passed

    def _generate_change_summary(self, context: AgentContext) -> str:
        """Generate a summary of all changes.

        Args:
            context: Execution context.

        Returns:
            Change summary string.
        """
        lines = ["# Execution Summary", ""]

        # Agent results
        lines.append("## Agent Results")
        for result in self._agent_results:
            status = "PASSED" if result.success else "FAILED"
            lines.append(f"- {result.agent_name}: {status} ({result.duration_seconds:.2f}s)")
        lines.append("")

        # Files modified
        all_files = []
        for result in self._agent_results:
            all_files.extend(result.files_modified)

        if all_files:
            lines.append("## Files Modified")
            for f in sorted(set(str(f) for f in all_files)):
                lines.append(f"- {f}")
            lines.append("")

        # Errors
        all_errors = []
        for result in self._agent_results:
            all_errors.extend(result.errors)

        if all_errors:
            lines.append("## Errors")
            for e in all_errors[:10]:
                lines.append(f"- {e}")
            lines.append("")

        return "\n".join(lines)

    def run_single_agent(
        self,
        agent_name: str,
        context: Optional[AgentContext] = None,
    ) -> AgentResult:
        """Run a single agent.

        Args:
            agent_name: Name of the agent to run.
            context: Optional context override.

        Returns:
            AgentResult from the agent.
        """
        agent = self._agents.get(agent_name)
        if agent is None:
            return AgentResult(
                agent_name=agent_name,
                status=AgentStatus.FAILED,
                success=False,
                message=f"Agent not found: {agent_name}",
            )

        if context is None:
            self._request_id = str(uuid.uuid4())[:8]
            context = self._create_context()

        return agent.execute(context)

    @classmethod
    def from_template(
        cls,
        template_path: Path,
        working_dir: Optional[Path] = None,
    ) -> Orchestrator:
        """Create an orchestrator from a template file.

        Args:
            template_path: Path to the template file.
            working_dir: Working directory.

        Returns:
            Configured Orchestrator.
        """
        parser = TemplateParser()
        template = parser.parse_file(template_path)
        config = template.to_config()

        return cls(
            config=config,
            working_dir=working_dir,
        )

    @property
    def agent_results(self) -> list[AgentResult]:
        """Get results from all agents."""
        return self._agent_results
