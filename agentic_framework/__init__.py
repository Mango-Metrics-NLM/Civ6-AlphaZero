"""
Agentic Framework for Claude Code Implementation

A dynamic, reusable framework for agentic coding workflows with:
- Template parsing for prompt-based configuration
- Constraint and permission management
- Specialized sub-agents for different tasks
- Feedback loop for iterative improvement
- Session memory persistence

Usage:
    from agentic_framework import Orchestrator, load_config

    # From configuration file
    orchestrator = Orchestrator(config_path=Path("config.yaml"))

    # Or from template file
    orchestrator = Orchestrator.from_template(Path("template.md"))

    # Execute the workflow
    result = orchestrator.execute(objective="Build feature X")

    # Check results
    if result.success:
        print("All tasks completed successfully")
    else:
        print(f"Failed with errors: {result.errors}")
"""

__version__ = "0.1.0"

# Configuration
# Agents
from .agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    BaseAgent,
    DocumentationAgent,
    ImplementationAgent,
    PlanningAgent,
    TestingAgent,
    VerificationAgent,
)
from .config import (
    AgenticConfig,
    ConfigLoader,
    ConfigValidationError,
    get_config,
    load_config,
)

# Core components
from .core import (
    ActionPermission,
    ArchitectureDecision,
    AutonomyLevel,
    Constraint,
    ConstraintCheckResult,
    # Constraint Manager
    ConstraintManager,
    ConstraintType,
    ConstraintViolation,
    ErrorAction,
    ErrorHandler,
    # Feedback Loop
    FeedbackLoop,
    KnownIssue,
    MemorySection,
    ParsedTemplate,
    PermissionCheckResult,
    # Permission Manager
    PermissionManager,
    ScopeChecker,
    # Session Memory
    SessionMemory,
    TemplateParseError,
    # Template Parser
    TemplateParser,
    VerificationPhase,
    VerificationResult,
)

# Orchestrator
from .orchestrator import (
    Orchestrator,
    OrchestratorResult,
)

# Utilities
from .utils import (
    CommandError,
    CommandResult,
    CommandRunner,
    LogContext,
    get_logger,
    setup_logging,
)

__all__ = [
    # Version
    "__version__",
    # Configuration
    "AgenticConfig",
    "ConfigLoader",
    "ConfigValidationError",
    "load_config",
    "get_config",
    # Core - Template
    "TemplateParser",
    "ParsedTemplate",
    "TemplateParseError",
    # Core - Constraints
    "ConstraintManager",
    "Constraint",
    "ConstraintType",
    "ConstraintViolation",
    "ConstraintCheckResult",
    # Core - Permissions
    "PermissionManager",
    "AutonomyLevel",
    "ActionPermission",
    "PermissionCheckResult",
    "ScopeChecker",
    # Core - Feedback
    "FeedbackLoop",
    "VerificationResult",
    "ErrorHandler",
    "ErrorAction",
    "VerificationPhase",
    # Core - Session
    "SessionMemory",
    "MemorySection",
    "ArchitectureDecision",
    "KnownIssue",
    # Agents
    "BaseAgent",
    "AgentContext",
    "AgentResult",
    "AgentStatus",
    "PlanningAgent",
    "ImplementationAgent",
    "TestingAgent",
    "VerificationAgent",
    "DocumentationAgent",
    # Orchestrator
    "Orchestrator",
    "OrchestratorResult",
    # Utilities
    "setup_logging",
    "get_logger",
    "LogContext",
    "CommandRunner",
    "CommandResult",
    "CommandError",
]
