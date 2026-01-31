"""Centralized constants for the agentic framework.

All values are configurable via environment variables with AGENTIC_ prefix.
This module provides sensible defaults that can be overridden at runtime
without code changes.

Example:
    # Override max retries at runtime
    AGENTIC_MAX_RETRIES=5 python my_script.py
"""

from __future__ import annotations

import os

# =============================================================================
# Retry Configuration
# =============================================================================

DEFAULT_MAX_RETRIES: int = int(os.getenv("AGENTIC_MAX_RETRIES", "3"))
"""Maximum number of retry attempts for failed operations."""

DEFAULT_IMPORT_ERROR_MAX_RETRIES: int = int(os.getenv("AGENTIC_IMPORT_ERROR_MAX_RETRIES", "2"))
"""Maximum retries specifically for import errors (typically fewer attempts)."""

DEFAULT_RETRY_DELAY_SECONDS: float = float(os.getenv("AGENTIC_RETRY_DELAY", "1.0"))
"""Initial delay between retry attempts in seconds."""

DEFAULT_BACKOFF_MULTIPLIER: float = float(os.getenv("AGENTIC_BACKOFF_MULTIPLIER", "2.0"))
"""Multiplier for exponential backoff between retries."""

# =============================================================================
# Iteration Limits
# =============================================================================

DEFAULT_MAX_ITERATIONS: int = int(os.getenv("AGENTIC_MAX_ITERATIONS", "5"))
"""Maximum iterations for feedback loops and verification cycles."""

DEFAULT_MAX_FILES_PER_PASS: int = int(os.getenv("AGENTIC_MAX_FILES_PER_PASS", "20"))
"""Maximum files to process in a single pass during exploration."""

# =============================================================================
# Timeouts
# =============================================================================

DEFAULT_EXPLORATION_TIMEOUT_SECONDS: int = int(
    os.getenv("AGENTIC_EXPLORATION_TIMEOUT", "600")
)
"""Timeout for exploration operations (default: 10 minutes)."""

DEFAULT_OPERATION_TIMEOUT_SECONDS: int = int(
    os.getenv("AGENTIC_OPERATION_TIMEOUT", "300")
)
"""Default timeout for general operations (default: 5 minutes)."""

# =============================================================================
# Thresholds
# =============================================================================

DEFAULT_ERROR_THRESHOLD: int = int(os.getenv("AGENTIC_ERROR_THRESHOLD", "3"))
"""Number of repeated errors before escalation."""

# =============================================================================
# Session Memory Defaults
# =============================================================================

DEFAULT_CLAUDE_MD_FILENAME: str = os.getenv("AGENTIC_CLAUDE_MD_FILENAME", "CLAUDE.md")
"""Default filename for CLAUDE.md session memory file."""

DEFAULT_SECTION_ORDER: tuple[str, ...] = (
    "Build Commands",
    "Test Commands",
    "Architecture Decisions",
    "Known Issues",
    "Environment Setup",
    "Gotchas",
)
"""Default ordering for sections in session memory files."""

# =============================================================================
# Template Parser Defaults
# =============================================================================

DEFAULT_FAIL_FAST: bool = os.getenv("AGENTIC_FAIL_FAST", "false").lower() == "true"
"""Whether to fail fast on first error in template parsing."""

DEFAULT_RETRY_COUNT: int = int(os.getenv("AGENTIC_RETRY_COUNT", "0"))
"""Default retry count for template operations."""
