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

from .constraints import (
    Constraint,
    ConstraintCategory,
    ConstraintType,
    ConstraintValidator,
    ConstraintViolation,
    create_constraint,
    create_validator_from_config,
)
from .feedback_handler import (
    ErrorPattern,
    FeedbackHandler,
    FeedbackStrategy,
    RecoveryAction,
)
from .sub_agents import (
    SubAgentConfig,
    SubAgentRegistry,
    SubAgentType,
    create_sub_agent_config,
)
from .template_generator import (
    TemplateContext,
    TemplateGenerator,
    TemplateVariable,
    VariableType,
    create_template_generator,
)
from .template_logger import (
    DebugContext,
    LogLevel,
    TemplateLogger,
    setup_template_logging,
)
from .verification import (
    VerificationPipeline,
    VerificationResult,
    VerificationStatus,
    VerificationStep,
    VerificationType,
    create_standard_pipeline,
    create_verification_pipeline,
)

__all__ = [
    # Template Generator
    "TemplateGenerator",
    "TemplateContext",
    "TemplateVariable",
    "VariableType",
    "create_template_generator",
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
