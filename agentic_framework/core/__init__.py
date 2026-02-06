"""
Agentic Framework Core Module

Provides the core components for the agentic coding system:
- Template parsing
- Constraint management
- Permission and autonomy management
- Feedback loop and verification
- Session memory management
"""

from .constraint_manager import (
    Constraint,
    ConstraintCheckResult,
    ConstraintManager,
    ConstraintType,
    ConstraintViolation,
)
from .feedback_loop import (
    ErrorAction,
    ErrorHandler,
    FeedbackLoop,
    VerificationPhase,
    VerificationResult,
)
from .permission_manager import (
    ActionPermission,
    AutonomyLevel,
    PermissionCheckResult,
    PermissionManager,
    ScopeChecker,
)
from .session_memory import (
    ArchitectureDecision,
    KnownIssue,
    MemorySection,
    SessionMemory,
)
from .template_parser import (
    ParsedTemplate,
    TemplateParseError,
    TemplateParser,
    TemplateSection,
)

__all__ = [
    "ActionPermission",
    "ArchitectureDecision",
    "AutonomyLevel",
    "Constraint",
    "ConstraintCheckResult",
    # Constraint Manager
    "ConstraintManager",
    "ConstraintType",
    "ConstraintViolation",
    "ErrorAction",
    "ErrorHandler",
    # Feedback Loop
    "FeedbackLoop",
    "KnownIssue",
    "MemorySection",
    "ParsedTemplate",
    "PermissionCheckResult",
    # Permission Manager
    "PermissionManager",
    "ScopeChecker",
    # Session Memory
    "SessionMemory",
    "TemplateParseError",
    # Template Parser
    "TemplateParser",
    "TemplateSection",
    "VerificationPhase",
    "VerificationResult",
]
