"""
Agentic Coding System Prompt Template System

This module provides dynamic, reusable template generation for Claude Code
implementation using sub-agents. All values are configurable - no hardcoded values.

Features:
- Dynamic template generation with variable substitution
- Reusable sub-agent configurations
- Comprehensive verification and feedback loop components
- Full logging and debugging infrastructure
- Best testing practices enforcement
"""

from __future__ import annotations

from .template_generator import (
    TemplateGenerator,
    TemplateContext,
    TemplateVariable,
    VariableType,
)
from .sub_agents import (
    SubAgentConfig,
    SubAgentType,
    SubAgentRegistry,
    create_sub_agent_config,
)
from .verification import (
    VerificationStep,
    VerificationPipeline,
    VerificationResult,
    VerificationType,
    VerificationStatus,
    create_verification_pipeline,
    create_standard_pipeline,
)
from .feedback_handler import (
    FeedbackHandler,
    FeedbackStrategy,
    ErrorPattern,
    RecoveryAction,
)
from .template_logger import (
    TemplateLogger,
    LogLevel,
    DebugContext,
    setup_template_logging,
)
from .constraints import (
    Constraint,
    ConstraintType,
    ConstraintCategory,
    ConstraintViolation,
    ConstraintValidator,
    create_constraint,
    create_validator_from_config,
)

__all__ = [
    # Template Generator
    "TemplateGenerator",
    "TemplateContext",
    "TemplateVariable",
    "VariableType",
    # Sub-Agents
    "SubAgentConfig",
    "SubAgentType",
    "SubAgentRegistry",
    "create_sub_agent_config",
    # Verification
    "VerificationStep",
    "VerificationPipeline",
    "VerificationResult",
    "VerificationType",
    "VerificationStatus",
    "create_verification_pipeline",
    "create_standard_pipeline",
    # Feedback
    "FeedbackHandler",
    "FeedbackStrategy",
    "ErrorPattern",
    "RecoveryAction",
    # Logging
    "TemplateLogger",
    "LogLevel",
    "DebugContext",
    "setup_template_logging",
    # Constraints
    "Constraint",
    "ConstraintType",
    "ConstraintCategory",
    "ConstraintViolation",
    "ConstraintValidator",
    "create_constraint",
    "create_validator_from_config",
]

__version__ = "1.0.0"
