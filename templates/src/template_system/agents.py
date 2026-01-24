"""
Sub-Agent Configuration and Orchestration Module.

This module provides a dynamic, reusable system for defining, configuring,
and orchestrating sub-agents in Claude Code workflows.

Example:
    >>> from template_system.agents import AgentOrchestrator, AgentConfig, AgentType
    >>> config = AgentConfig(
    ...     agent_type=AgentType.EXPLORE,
    ...     thoroughness="medium",
    ...     max_turns=10,
    ... )
    >>> orchestrator = AgentOrchestrator()
    >>> result = await orchestrator.execute(config, "Find all Python files")
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    TypeVar,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable


logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Available sub-agent types in Claude Code."""

    EXPLORE = "Explore"
    PLAN = "Plan"
    BASH = "Bash"
    GENERAL_PURPOSE = "general-purpose"
    CLAUDE_CODE_GUIDE = "claude-code-guide"
    STATUSLINE_SETUP = "statusline-setup"

    @classmethod
    def from_string(cls, value: str) -> AgentType:
        """Create AgentType from string value.

        Args:
            value: String representation of agent type.

        Returns:
            Corresponding AgentType enum member.

        Raises:
            ValueError: If value doesn't match any agent type.
        """
        value_lower = value.lower().replace("-", "_").replace(" ", "_")
        for agent_type in cls:
            if agent_type.value.lower().replace("-", "_") == value_lower:
                return agent_type
            if agent_type.name.lower() == value_lower:
                return agent_type
        raise ValueError(f"Unknown agent type: {value}")


class Thoroughness(Enum):
    """Thoroughness levels for Explore agent."""

    QUICK = "quick"
    MEDIUM = "medium"
    VERY_THOROUGH = "very thorough"


class ModelType(Enum):
    """Available model types for agents."""

    SONNET = "sonnet"
    OPUS = "opus"
    HAIKU = "haiku"


@dataclass
class AgentCapability:
    """Defines a capability that an agent possesses."""

    name: str
    description: str
    required_tools: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Configuration for a sub-agent invocation.

    Attributes:
        agent_type: Type of agent to invoke.
        description: Short description (3-5 words) of the task.
        prompt: Detailed task prompt for the agent.
        thoroughness: For Explore agent, how thorough the search should be.
        model: Optional model override (sonnet, opus, haiku).
        max_turns: Maximum number of agentic turns before stopping.
        run_in_background: Whether to run the agent in the background.
        resume_id: Optional agent ID to resume from previous execution.
        timeout_seconds: Timeout for agent execution.
        retry_count: Number of retries on failure.
        metadata: Additional metadata for the agent invocation.
    """

    agent_type: AgentType
    description: str = ""
    prompt: str = ""
    thoroughness: Thoroughness | None = None
    model: ModelType | None = None
    max_turns: int | None = None
    run_in_background: bool = False
    resume_id: str | None = None
    timeout_seconds: int = 300
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.thoroughness and self.agent_type != AgentType.EXPLORE:
            logger.warning(
                "Thoroughness is only applicable to Explore agent, ignoring",
                extra={"agent_type": self.agent_type.value},
            )

    def to_task_params(self) -> dict[str, Any]:
        """Convert to parameters for Task tool invocation.

        Returns:
            Dictionary of parameters for the Task tool.
        """
        params: dict[str, Any] = {
            "subagent_type": self.agent_type.value,
            "description": self.description,
            "prompt": self._build_prompt(),
        }

        if self.model:
            params["model"] = self.model.value
        if self.max_turns:
            params["max_turns"] = self.max_turns
        if self.run_in_background:
            params["run_in_background"] = True
        if self.resume_id:
            params["resume"] = self.resume_id

        return params

    def _build_prompt(self) -> str:
        """Build the full prompt including thoroughness for Explore agent."""
        if self.agent_type == AgentType.EXPLORE and self.thoroughness:
            return f"Thoroughness: {self.thoroughness.value}\n\n{self.prompt}"
        return self.prompt


@dataclass
class AgentResult:
    """Result from a sub-agent execution.

    Attributes:
        success: Whether the agent completed successfully.
        output: Output from the agent.
        agent_id: ID of the agent for resumption.
        output_file: Path to output file (for background agents).
        error: Error message if failed.
        duration_seconds: Time taken for execution.
        metadata: Additional metadata from execution.
    """

    success: bool
    output: str = ""
    agent_id: str | None = None
    output_file: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def can_resume(self) -> bool:
        """Check if this agent can be resumed."""
        return self.agent_id is not None


T = TypeVar("T")


class AgentStrategy(ABC, Generic[T]):
    """Abstract base class for agent execution strategies."""

    @abstractmethod
    async def execute(self, config: AgentConfig, context: T) -> AgentResult:
        """Execute the agent with given configuration.

        Args:
            config: Agent configuration.
            context: Execution context.

        Returns:
            Result of agent execution.
        """
        ...

    @abstractmethod
    def validate(self, config: AgentConfig) -> list[str]:
        """Validate configuration for this strategy.

        Args:
            config: Agent configuration to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        ...


class AgentRegistry:
    """Registry for agent configurations and capabilities.

    This class maintains a registry of agent types with their capabilities,
    constraints, and recommended use cases.
    """

    _instance: ClassVar[AgentRegistry | None] = None
    _capabilities: ClassVar[dict[AgentType, list[AgentCapability]]] = {}
    _use_cases: ClassVar[dict[AgentType, list[str]]] = {}

    def __new__(cls) -> AgentRegistry:
        """Singleton pattern for registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_defaults()
        return cls._instance

    def _initialize_defaults(self) -> None:
        """Initialize default agent capabilities and use cases."""
        # Explore agent
        self._capabilities[AgentType.EXPLORE] = [
            AgentCapability(
                name="file_search",
                description="Find files by pattern matching",
                required_tools=["Glob"],
            ),
            AgentCapability(
                name="code_search",
                description="Search code for keywords and patterns",
                required_tools=["Grep"],
            ),
            AgentCapability(
                name="structure_analysis",
                description="Analyze codebase structure",
                required_tools=["Glob", "Read"],
            ),
        ]
        self._use_cases[AgentType.EXPLORE] = [
            "Finding files by pattern",
            "Searching for code keywords",
            "Understanding codebase structure",
            "Tracing imports and dependencies",
        ]

        # Plan agent
        self._capabilities[AgentType.PLAN] = [
            AgentCapability(
                name="architecture_design",
                description="Design implementation strategies",
                required_tools=["Read", "Glob", "Grep"],
            ),
            AgentCapability(
                name="dependency_analysis",
                description="Identify file dependencies",
                required_tools=["Grep", "Read"],
            ),
        ]
        self._use_cases[AgentType.PLAN] = [
            "Complex multi-file changes",
            "New feature implementation",
            "Architectural decisions",
            "Refactoring strategies",
        ]

        # Bash agent
        self._capabilities[AgentType.BASH] = [
            AgentCapability(
                name="command_execution",
                description="Execute shell commands",
                required_tools=["Bash"],
                constraints=["No interactive commands"],
            ),
            AgentCapability(
                name="git_operations",
                description="Git version control operations",
                required_tools=["Bash"],
            ),
        ]
        self._use_cases[AgentType.BASH] = [
            "Git commits and pushes",
            "Installing dependencies",
            "Running tests and linters",
            "Build and deploy operations",
        ]

        # General purpose agent
        self._capabilities[AgentType.GENERAL_PURPOSE] = [
            AgentCapability(
                name="research",
                description="Multi-step research and analysis",
                required_tools=["Read", "Glob", "Grep", "WebFetch", "WebSearch"],
            ),
        ]
        self._use_cases[AgentType.GENERAL_PURPOSE] = [
            "Multi-step investigations",
            "Complex code searches",
            "When unsure of file locations",
            "Research tasks",
        ]

    def get_capabilities(self, agent_type: AgentType) -> list[AgentCapability]:
        """Get capabilities for an agent type.

        Args:
            agent_type: Type of agent.

        Returns:
            List of capabilities for the agent.
        """
        return self._capabilities.get(agent_type, [])

    def get_use_cases(self, agent_type: AgentType) -> list[str]:
        """Get recommended use cases for an agent type.

        Args:
            agent_type: Type of agent.

        Returns:
            List of use case descriptions.
        """
        return self._use_cases.get(agent_type, [])

    def register_capability(
        self,
        agent_type: AgentType,
        capability: AgentCapability,
    ) -> None:
        """Register a new capability for an agent type.

        Args:
            agent_type: Type of agent.
            capability: Capability to register.
        """
        if agent_type not in self._capabilities:
            self._capabilities[agent_type] = []
        self._capabilities[agent_type].append(capability)
        logger.debug(
            "Registered capability",
            extra={
                "agent_type": agent_type.value,
                "capability": capability.name,
            },
        )

    def suggest_agent(self, task_description: str) -> AgentType | None:
        """Suggest an appropriate agent type for a task.

        Args:
            task_description: Description of the task.

        Returns:
            Suggested agent type, or None if no match.
        """
        task_lower = task_description.lower()

        # Pattern matching for agent selection
        patterns = {
            AgentType.EXPLORE: [
                "find", "search", "locate", "where", "structure",
                "pattern", "files", "codebase",
            ],
            AgentType.PLAN: [
                "design", "plan", "architect", "strategy", "implement",
                "refactor", "approach",
            ],
            AgentType.BASH: [
                "run", "execute", "git", "install", "test", "build",
                "deploy", "command",
            ],
            AgentType.GENERAL_PURPOSE: [
                "research", "investigate", "analyze", "complex",
                "multi-step", "understand",
            ],
        }

        scores: dict[AgentType, int] = {}
        for agent_type, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > 0:
                scores[agent_type] = score

        if scores:
            return max(scores, key=scores.get)  # type: ignore[arg-type]
        return None


@dataclass
class OrchestrationPlan:
    """Plan for orchestrating multiple agents.

    Attributes:
        steps: List of agent configurations to execute.
        parallel_groups: Groups of steps that can run in parallel.
        dependencies: Map of step index to indices it depends on.
        metadata: Additional plan metadata.
    """

    steps: list[AgentConfig] = field(default_factory=list)
    parallel_groups: list[list[int]] = field(default_factory=list)
    dependencies: dict[int, list[int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        config: AgentConfig,
        depends_on: list[int] | None = None,
    ) -> int:
        """Add a step to the plan.

        Args:
            config: Agent configuration for the step.
            depends_on: Indices of steps this step depends on.

        Returns:
            Index of the added step.
        """
        index = len(self.steps)
        self.steps.append(config)
        if depends_on:
            self.dependencies[index] = depends_on
        return index

    def add_parallel_group(self, step_indices: list[int]) -> None:
        """Mark a group of steps as parallelizable.

        Args:
            step_indices: Indices of steps that can run in parallel.
        """
        self.parallel_groups.append(step_indices)


class AgentOrchestrator:
    """Orchestrator for coordinating multiple sub-agents.

    This class provides methods for planning and executing multi-agent
    workflows with proper dependency management and parallelization.

    Example:
        >>> orchestrator = AgentOrchestrator()
        >>> plan = orchestrator.create_plan()
        >>> plan.add_step(AgentConfig(AgentType.EXPLORE, prompt="Find files"))
        >>> plan.add_step(AgentConfig(AgentType.PLAN, prompt="Plan changes"))
        >>> results = await orchestrator.execute_plan(plan)
    """

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        max_parallel: int = 3,
        default_timeout: int = 300,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            registry: Agent registry to use (default singleton).
            max_parallel: Maximum parallel agent executions.
            default_timeout: Default timeout for agent execution.
        """
        self.registry = registry or AgentRegistry()
        self.max_parallel = max_parallel
        self.default_timeout = default_timeout
        self._execution_history: list[AgentResult] = []
        self._logger = logger.getChild(self.__class__.__name__)

    def create_plan(self) -> OrchestrationPlan:
        """Create a new orchestration plan.

        Returns:
            Empty orchestration plan.
        """
        return OrchestrationPlan()

    def suggest_workflow(self, task: str) -> OrchestrationPlan:
        """Suggest a workflow for a complex task.

        Args:
            task: Description of the task.

        Returns:
            Suggested orchestration plan.
        """
        plan = self.create_plan()
        task_lower = task.lower()

        # Pattern: Explore -> Plan -> Execute
        if any(kw in task_lower for kw in ["implement", "add", "create", "build"]):
            explore_idx = plan.add_step(
                AgentConfig(
                    agent_type=AgentType.EXPLORE,
                    description="Explore codebase",
                    prompt=f"Explore the codebase to understand: {task}",
                    thoroughness=Thoroughness.MEDIUM,
                )
            )
            plan.add_step(
                AgentConfig(
                    agent_type=AgentType.PLAN,
                    description="Plan implementation",
                    prompt=f"Plan the implementation for: {task}",
                ),
                depends_on=[explore_idx],
            )
            return plan

        # Pattern: Explore only
        if any(kw in task_lower for kw in ["find", "search", "where", "locate"]):
            plan.add_step(
                AgentConfig(
                    agent_type=AgentType.EXPLORE,
                    description="Search codebase",
                    prompt=task,
                    thoroughness=Thoroughness.MEDIUM,
                )
            )
            return plan

        # Pattern: Bash only
        if any(kw in task_lower for kw in ["run", "test", "install", "git"]):
            plan.add_step(
                AgentConfig(
                    agent_type=AgentType.BASH,
                    description="Execute command",
                    prompt=task,
                )
            )
            return plan

        # Default: General purpose
        plan.add_step(
            AgentConfig(
                agent_type=AgentType.GENERAL_PURPOSE,
                description="Investigate task",
                prompt=task,
            )
        )
        return plan

    def validate_plan(self, plan: OrchestrationPlan) -> list[str]:
        """Validate an orchestration plan.

        Args:
            plan: Plan to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[str] = []

        if not plan.steps:
            errors.append("Plan has no steps")
            return errors

        # Validate dependencies
        for step_idx, deps in plan.dependencies.items():
            if step_idx >= len(plan.steps):
                errors.append(f"Invalid step index: {step_idx}")
            for dep_idx in deps:
                if dep_idx >= step_idx:
                    errors.append(
                        f"Step {step_idx} depends on later step {dep_idx}"
                    )
                if dep_idx >= len(plan.steps):
                    errors.append(f"Invalid dependency index: {dep_idx}")

        # Validate parallel groups
        for group in plan.parallel_groups:
            for step_idx in group:
                if step_idx >= len(plan.steps):
                    errors.append(f"Invalid parallel group step: {step_idx}")
                # Check if parallelized steps have dependencies on each other
                if step_idx in plan.dependencies:
                    for dep in plan.dependencies[step_idx]:
                        if dep in group:
                            errors.append(
                                f"Parallel steps {step_idx} and {dep} have dependency"
                            )

        # Validate individual configs
        for i, config in enumerate(plan.steps):
            if not config.prompt:
                errors.append(f"Step {i} has no prompt")
            if not config.description:
                errors.append(f"Step {i} has no description")

        return errors

    def get_execution_order(self, plan: OrchestrationPlan) -> list[list[int]]:
        """Determine execution order respecting dependencies.

        Args:
            plan: Orchestration plan.

        Returns:
            List of step groups, where each group can run in parallel.
        """
        if not plan.steps:
            return []

        executed: set[int] = set()
        order: list[list[int]] = []

        while len(executed) < len(plan.steps):
            # Find steps whose dependencies are all executed
            ready = []
            for i in range(len(plan.steps)):
                if i in executed:
                    continue
                deps = plan.dependencies.get(i, [])
                if all(d in executed for d in deps):
                    ready.append(i)

            if not ready:
                # Circular dependency or error
                remaining = set(range(len(plan.steps))) - executed
                self._logger.error(
                    "Circular dependency detected",
                    extra={"remaining_steps": list(remaining)},
                )
                break

            # Check if any ready steps are in a parallel group
            parallel_batch = []
            for group in plan.parallel_groups:
                group_ready = [i for i in group if i in ready]
                if len(group_ready) > 1:
                    parallel_batch.extend(group_ready)

            if parallel_batch:
                # Execute parallel group
                batch = list(set(parallel_batch))[:self.max_parallel]
            else:
                # Execute one at a time
                batch = [ready[0]]

            order.append(batch)
            executed.update(batch)

        return order

    @property
    def history(self) -> list[AgentResult]:
        """Get execution history."""
        return self._execution_history.copy()

    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()


# Selection matrix helper
AGENT_SELECTION_MATRIX: dict[str, list[AgentType]] = {
    "find_file": [AgentType.EXPLORE],
    "search_code": [AgentType.EXPLORE],
    "understand_architecture": [AgentType.EXPLORE, AgentType.PLAN],
    "design_feature": [AgentType.PLAN],
    "plan_refactoring": [AgentType.PLAN],
    "run_tests": [AgentType.BASH],
    "git_operations": [AgentType.BASH],
    "install_dependencies": [AgentType.BASH],
    "complex_investigation": [AgentType.EXPLORE, AgentType.GENERAL_PURPOSE],
    "multi_step_research": [AgentType.GENERAL_PURPOSE],
}


def select_agent(task_type: str) -> list[AgentType]:
    """Select appropriate agents for a task type.

    Args:
        task_type: Type of task from AGENT_SELECTION_MATRIX.

    Returns:
        List of recommended agent types.
    """
    return AGENT_SELECTION_MATRIX.get(task_type, [AgentType.GENERAL_PURPOSE])
