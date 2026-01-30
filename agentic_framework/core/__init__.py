"""
Agentic Framework Core Module

Provides the core components for the agentic coding system:
- Template parsing
- Constraint management
- Permission and autonomy management
- Feedback loop and verification
- Session memory management
"""

from .template_parser import (
    TemplateParser,
    ParsedTemplate,
    TemplateSection,
    TemplateParseError,
)

from .constraint_manager import (
    ConstraintManager,
    Constraint,
    ConstraintType,
    ConstraintViolation,
    ConstraintCheckResult,
)

from .permission_manager import (
    PermissionManager,
    AutonomyLevel,
    ActionPermission,
    PermissionCheckResult,
    ScopeChecker,
)

from .feedback_loop import (
    FeedbackLoop,
    VerificationResult,
    ErrorHandler,
    ErrorAction,
    VerificationPhase,
)

from .session_memory import (
    SessionMemory,
    MemorySection,
    ArchitectureDecision,
    KnownIssue,
)

__all__ = [
    # Template Parser
    "TemplateParser",
    "ParsedTemplate",
    "TemplateSection",
    "TemplateParseError",
    # Constraint Manager
    "ConstraintManager",
    "Constraint",
    "ConstraintType",
    "ConstraintViolation",
    "ConstraintCheckResult",
    # Permission Manager
    "PermissionManager",
    "AutonomyLevel",
    "ActionPermission",
    "PermissionCheckResult",
    "ScopeChecker",
    # Feedback Loop
    "FeedbackLoop",
    "VerificationResult",
    "ErrorHandler",
    "ErrorAction",
    "VerificationPhase",
    # Session Memory
    "SessionMemory",
    "MemorySection",
    "ArchitectureDecision",
    "KnownIssue",
]
