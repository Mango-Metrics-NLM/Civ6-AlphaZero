"""
Feedback Loop for Agentic Framework

Handles verification commands, error detection, auto-fixing,
and the iterative improvement cycle.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..config import AgenticConfig, VerificationCommand
from ..utils.command_runner import CommandRunner, RetryConfig
from ..utils.logging_config import get_logger
from .constants import (
    DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_ERROR_THRESHOLD,
    DEFAULT_IMPORT_ERROR_MAX_RETRIES,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY_SECONDS,
)

logger = get_logger("feedback_loop")


class ErrorAction(Enum):
    """Actions to take on error detection."""

    AUTO_FIX = "auto_fix"  # Automatically fix and retry
    ANALYZE_FIX = "analyze_fix"  # Analyze error, fix, retry
    REPORT = "report"  # Report and stop
    ESCALATE = "escalate"  # Escalate to human


class VerificationPhase(Enum):
    """Phases in the verification process."""

    LINT = "lint"
    TYPE_CHECK = "type_check"
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    SMOKE_TEST = "smoke_test"
    CUSTOM = "custom"


@dataclass
class VerificationResult:
    """Result of running a verification command."""

    phase: VerificationPhase
    name: str
    command: str
    success: bool
    output: str
    error_output: str
    duration_seconds: float
    retry_count: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Return True if verification passed."""
        return self.success


@dataclass
class ErrorPattern:
    """Pattern for detecting and handling errors."""

    error_type: str
    action: ErrorAction
    max_retries: int = 3
    pattern: re.Pattern | None = None
    fixer: Callable[[str, str], str | None] | None = None


@dataclass
class FeedbackLoopResult:
    """Result of running the full feedback loop."""

    success: bool
    verification_results: list[VerificationResult] = field(default_factory=list)
    iterations: int = 0
    total_duration_seconds: float = 0.0
    errors_fixed: int = 0
    errors_remaining: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failed_verifications(self) -> list[VerificationResult]:
        """Get failed verification results."""
        return [r for r in self.verification_results if not r.success]

    def summary(self) -> str:
        """Generate a summary of the feedback loop results."""
        passed = len([r for r in self.verification_results if r.success])
        failed = len(self.failed_verifications)

        lines = [
            f"Feedback Loop: {'PASSED' if self.success else 'FAILED'}",
            f"Iterations: {self.iterations}",
            f"Verifications: {passed} passed, {failed} failed",
            f"Errors fixed: {self.errors_fixed}",
            f"Duration: {self.total_duration_seconds:.2f}s",
        ]

        if self.errors_remaining:
            lines.append(f"Remaining errors: {len(self.errors_remaining)}")

        return "\n".join(lines)


class ErrorHandler:
    """Handles error detection and fixing."""

    def __init__(self) -> None:
        """Initialize the error handler."""
        self._patterns: list[ErrorPattern] = []
        self._fixers: dict[str, Callable[[str, str], str | None]] = {}

        # Register default patterns
        self._register_default_patterns()

    def _register_default_patterns(self) -> None:
        """Register default error patterns."""
        # Lint errors
        self._patterns.append(
            ErrorPattern(
                error_type="lint_error",
                action=ErrorAction.AUTO_FIX,
                max_retries=DEFAULT_MAX_RETRIES,
                pattern=re.compile(r"(?:error|warning):\s*(.+?)(?:\n|$)", re.IGNORECASE),
            )
        )

        # Type errors
        self._patterns.append(
            ErrorPattern(
                error_type="type_error",
                action=ErrorAction.ANALYZE_FIX,
                max_retries=DEFAULT_MAX_RETRIES,
                pattern=re.compile(r"error:\s*(.+?)(?:\n|$)", re.IGNORECASE),
            )
        )

        # Test failures
        self._patterns.append(
            ErrorPattern(
                error_type="test_failure",
                action=ErrorAction.ANALYZE_FIX,
                max_retries=DEFAULT_MAX_RETRIES,
                pattern=re.compile(r"FAILED\s+(.+?)(?:\s+-|$)", re.IGNORECASE),
            )
        )

        # Import errors
        self._patterns.append(
            ErrorPattern(
                error_type="import_error",
                action=ErrorAction.ANALYZE_FIX,
                max_retries=DEFAULT_IMPORT_ERROR_MAX_RETRIES,
                pattern=re.compile(
                    r"(?:ImportError|ModuleNotFoundError):\s*(.+?)(?:\n|$)", re.IGNORECASE
                ),
            )
        )

    def register_pattern(self, pattern: ErrorPattern) -> None:
        """Register an error pattern.

        Args:
            pattern: Error pattern to register.
        """
        self._patterns.append(pattern)
        logger.debug(f"Registered error pattern: {pattern.error_type}")

    def register_fixer(self, error_type: str, fixer: Callable[[str, str], str | None]) -> None:
        """Register a fixer function for an error type.

        Args:
            error_type: Type of error to fix.
            fixer: Function that takes (error_message, context) and returns fix or None.
        """
        self._fixers[error_type] = fixer
        logger.debug(f"Registered fixer for: {error_type}")

    def detect_errors(self, output: str) -> list[dict[str, Any]]:
        """Detect errors in command output.

        Args:
            output: Command output to analyze.

        Returns:
            List of detected errors with type and message.
        """
        errors: list[dict[str, Any]] = []

        for pattern in self._patterns:
            if pattern.pattern:
                matches = pattern.pattern.findall(output)
                for match in matches:
                    errors.append(
                        {
                            "type": pattern.error_type,
                            "message": match if isinstance(match, str) else match[0],
                            "action": pattern.action.value,
                            "max_retries": pattern.max_retries,
                        }
                    )

        logger.debug(f"Detected {len(errors)} errors in output")
        return errors

    def get_action(self, error_type: str) -> ErrorAction:
        """Get the action to take for an error type.

        Args:
            error_type: Type of error.

        Returns:
            ErrorAction to take.
        """
        for pattern in self._patterns:
            if pattern.error_type == error_type:
                return pattern.action
        return ErrorAction.REPORT

    def can_auto_fix(self, error_type: str) -> bool:
        """Check if an error type can be auto-fixed.

        Args:
            error_type: Type of error.

        Returns:
            True if auto-fix is available.
        """
        return error_type in self._fixers

    def attempt_fix(self, error_type: str, error_message: str, context: str) -> str | None:
        """Attempt to fix an error.

        Args:
            error_type: Type of error.
            error_message: Error message.
            context: Additional context (e.g., file content).

        Returns:
            Fix suggestion or None if no fix available.
        """
        if error_type in self._fixers:
            return self._fixers[error_type](error_message, context)
        return None


class FeedbackLoop:
    """Runs the verification and error correction loop.

    Features:
    - Sequential verification command execution
    - Error detection and categorization
    - Automatic retry with exponential backoff
    - Fix suggestion integration
    - Progress tracking
    """

    def __init__(
        self,
        config: AgenticConfig | None = None,
        command_runner: CommandRunner | None = None,
        error_handler: ErrorHandler | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        """Initialize the feedback loop.

        Args:
            config: Configuration containing verification commands.
            command_runner: Command runner for execution.
            error_handler: Error handler for detection and fixing.
            max_iterations: Maximum iterations before giving up.
        """
        self.config = config
        self.command_runner = command_runner or CommandRunner()
        self.error_handler = error_handler or ErrorHandler()
        self.max_iterations = max_iterations
        self._verification_commands: list[VerificationCommand] = []
        self._success_checks: list[dict[str, Any]] = []
        self._on_error_callback: Callable[[str, dict[str, Any]], None] | None = None
        self._on_fix_callback: Callable[[str, str], bool] | None = None

        if config:
            self.load_from_config(config)

    def load_from_config(self, config: AgenticConfig) -> None:
        """Load verification configuration from config.

        Args:
            config: Configuration containing verification settings.
        """
        self.config = config
        self._verification_commands = config.feedback.verification_commands
        self._success_checks = config.feedback.success_verification

        # Load error handling patterns
        for pattern_data in config.feedback.error_handling.auto_fix_patterns:
            action = ErrorAction(pattern_data.get("action", "analyze_fix"))
            pattern = ErrorPattern(
                error_type=pattern_data.get("error_type", "unknown"),
                action=action,
                max_retries=pattern_data.get("max_retries", DEFAULT_MAX_RETRIES),
            )
            self.error_handler.register_pattern(pattern)

        logger.info(f"Loaded {len(self._verification_commands)} verification commands")

    def set_error_callback(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Set callback for error notifications.

        Args:
            callback: Function called with (error_type, error_details).
        """
        self._on_error_callback = callback

    def set_fix_callback(self, callback: Callable[[str, str], bool]) -> None:
        """Set callback for fix confirmations.

        Args:
            callback: Function called with (error_message, fix_suggestion) returns True to apply.
        """
        self._on_fix_callback = callback

    def run_verification(
        self, command: VerificationCommand, cwd: Path | None = None
    ) -> VerificationResult:
        """Run a single verification command.

        Args:
            command: Verification command to run.
            cwd: Working directory.

        Returns:
            VerificationResult with outcome.
        """
        logger.info(f"Running verification: {command.name}")

        # Determine phase from name
        phase = self._determine_phase(command.name)

        # Run command with retry if configured
        retry_config = RetryConfig(
            max_retries=command.retry_count,
            retry_delay_seconds=DEFAULT_RETRY_DELAY_SECONDS,
            backoff_multiplier=DEFAULT_BACKOFF_MULTIPLIER,
        )

        start_time = time.time()
        result = self.command_runner.run(
            command.command,
            timeout_seconds=command.timeout_seconds,
            cwd=cwd,
            retry_config=retry_config,
        )
        duration = time.time() - start_time

        # Detect errors in output
        combined_output = f"{result.stdout}\n{result.stderr}"
        errors = self.error_handler.detect_errors(combined_output)

        # Extract warnings (non-error issues)
        warnings = [e for e in errors if "warning" in e.get("type", "").lower()]
        errors = [e for e in errors if "warning" not in e.get("type", "").lower()]

        verification_result = VerificationResult(
            phase=phase,
            name=command.name,
            command=command.command,
            success=result.success,
            output=result.stdout,
            error_output=result.stderr,
            duration_seconds=duration,
            retry_count=0,
            errors=errors if not result.success else [],
            warnings=warnings,
        )

        if verification_result.success:
            logger.info(f"Verification passed: {command.name}")
        else:
            logger.warning(f"Verification failed: {command.name}")

        return verification_result

    def _determine_phase(self, name: str) -> VerificationPhase:
        """Determine verification phase from command name.

        Args:
            name: Command name.

        Returns:
            VerificationPhase enum value.
        """
        name_lower = name.lower()

        if "lint" in name_lower or "ruff" in name_lower or "flake" in name_lower:
            return VerificationPhase.LINT
        elif "type" in name_lower or "mypy" in name_lower:
            return VerificationPhase.TYPE_CHECK
        elif "unit" in name_lower:
            return VerificationPhase.UNIT_TEST
        elif "integration" in name_lower:
            return VerificationPhase.INTEGRATION_TEST
        elif "smoke" in name_lower:
            return VerificationPhase.SMOKE_TEST
        else:
            return VerificationPhase.CUSTOM

    def run_all_verifications(
        self,
        cwd: Path | None = None,
        stop_on_failure: bool = False,
    ) -> list[VerificationResult]:
        """Run all verification commands.

        Args:
            cwd: Working directory.
            stop_on_failure: Stop after first failure.

        Returns:
            List of VerificationResult objects.
        """
        results: list[VerificationResult] = []

        for command in self._verification_commands:
            result = self.run_verification(command, cwd)
            results.append(result)

            if not result.success and (stop_on_failure or command.fail_fast):
                logger.warning(f"Stopping verifications due to failure: {command.name}")
                break

        return results

    def run_feedback_loop(
        self,
        cwd: Path | None = None,
        fix_callback: Callable[[list[dict[str, Any]]], bool] | None = None,
    ) -> FeedbackLoopResult:
        """Run the full feedback loop with error correction.

        Args:
            cwd: Working directory.
            fix_callback: Callback to apply fixes, receives errors, returns True if fixed.

        Returns:
            FeedbackLoopResult with final outcome.
        """
        logger.info("Starting feedback loop")
        start_time = time.time()

        result = FeedbackLoopResult(success=False)
        error_counts: dict[str, int] = {}

        for iteration in range(self.max_iterations):
            result.iterations = iteration + 1
            logger.info(f"Feedback loop iteration {iteration + 1}/{self.max_iterations}")

            # Run all verifications
            verification_results = self.run_all_verifications(cwd)
            result.verification_results = verification_results

            # Check if all passed
            if all(r.success for r in verification_results):
                result.success = True
                logger.info("All verifications passed!")
                break

            # Collect all errors
            all_errors: list[dict[str, Any]] = []
            for vr in verification_results:
                if not vr.success:
                    all_errors.extend(vr.errors)

            if not all_errors:
                # No errors detected but verification failed
                logger.warning("Verification failed but no errors detected")
                result.errors_remaining = [
                    {"type": "unknown", "message": "Verification failed without specific error"}
                ]
                break

            # Track error counts for repeated failure detection
            for error in all_errors:
                error_key = f"{error['type']}:{error['message'][:50]}"
                error_counts[error_key] = error_counts.get(error_key, 0) + 1

                # Check for repeated failures
                if error_counts[error_key] >= DEFAULT_ERROR_THRESHOLD:
                    logger.warning(f"Repeated failure detected: {error_key}")
                    error["action"] = ErrorAction.ESCALATE.value

            # Notify about errors
            if self._on_error_callback:
                for error in all_errors:
                    self._on_error_callback(error["type"], error)

            # Attempt to fix errors
            if fix_callback:
                if fix_callback(all_errors):
                    result.errors_fixed += len(all_errors)
                    logger.info(f"Applied fixes for {len(all_errors)} errors")
                else:
                    logger.warning("Fix callback returned False, stopping loop")
                    result.errors_remaining = all_errors
                    break
            else:
                # No fix callback, can't proceed
                logger.warning("No fix callback provided, stopping loop")
                result.errors_remaining = all_errors
                break

        result.total_duration_seconds = time.time() - start_time
        logger.info(f"Feedback loop completed: {result.summary()}")

        return result

    def check_success_criteria(
        self,
        verification_results: list[VerificationResult],
        cwd: Path | None = None,
    ) -> bool:
        """Check if all success criteria are met.

        Args:
            verification_results: Results from verification commands.
            cwd: Working directory for additional checks.

        Returns:
            True if all criteria are met.
        """
        # All verification commands must pass
        if not all(r.success for r in verification_results):
            return False

        # Run any additional success checks
        for check in self._success_checks:
            check_name = check.get("check", "")
            check_command = check.get("command")

            if check_command:
                result = self.command_runner.run(check_command, cwd=cwd)
                if not result.success:
                    logger.warning(f"Success check failed: {check_name}")
                    return False

        return True

    def generate_error_report(self, verification_results: list[VerificationResult]) -> str:
        """Generate a detailed error report.

        Args:
            verification_results: Results to report on.

        Returns:
            Formatted error report string.
        """
        lines = ["# Verification Error Report", ""]

        for result in verification_results:
            status = "PASSED" if result.success else "FAILED"
            lines.append(f"## {result.name}: {status}")
            lines.append(f"- Command: `{result.command}`")
            lines.append(f"- Duration: {result.duration_seconds:.2f}s")

            if not result.success:
                lines.append(f"- Errors: {len(result.errors)}")
                for error in result.errors[:5]:  # Limit to first 5
                    lines.append(f"  - [{error['type']}] {error['message'][:100]}")

                if result.error_output:
                    lines.append("- Error output:")
                    lines.append("```")
                    lines.append(result.error_output[:500])
                    lines.append("```")

            lines.append("")

        return "\n".join(lines)
