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
    # Constraints
    "Constraint",
    "ConstraintCategory",
    "ConstraintType",
    "ConstraintValidator",
    "ConstraintViolation",
    "DebugContext",
    "ErrorPattern",
    # Feedback
    "FeedbackHandler",
    "FeedbackStrategy",
    "LogLevel",
    "RecoveryAction",
    # Sub-Agents
    "SubAgentConfig",
    "SubAgentRegistry",
    "SubAgentType",
    "TemplateContext",
    # Template Generator
    "TemplateGenerator",
    # Logging
    "TemplateLogger",
    "TemplateVariable",
    "VariableType",
    "VerificationPipeline",
    "VerificationResult",
    "VerificationStatus",
    # Verification
    "VerificationStep",
    "VerificationType",
    "create_constraint",
    "create_standard_pipeline",
    "create_sub_agent_config",
    "create_template_generator",
    "create_validator_from_config",
    "create_verification_pipeline",
    "setup_template_logging",
]

__version__ = "1.0.0"
