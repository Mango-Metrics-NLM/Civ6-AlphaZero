"""
Verification Pipeline for Agentic Coding System

Provides dynamic verification steps and pipelines with no hardcoded values.
All verification commands and thresholds are configurable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from ..utils.logging_config import get_logger
from ..utils.command_runner import CommandRunner, CommandResult

logger = get_logger("verification")


class VerificationStatus(Enum):
    """Status of a verification step."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class VerificationType(Enum):
    """Types of verification steps."""

    LINT = "lint"
    TYPE_CHECK = "type_check"
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    SECURITY_SCAN = "security_scan"
    CUSTOM = "custom"


@dataclass
class VerificationResult:
    """Result of a verification step.

    Attributes:
        step_name: Name of the verification step
        status: Verification status
        command: Command that was executed
        stdout: Standard output
        stderr: Standard error
        exit_code: Process exit code
        duration_seconds: Execution duration
        errors: List of error messages
        warnings: List of warning messages
        metrics: Additional metrics (coverage, etc.)
    """

    step_name: str
    status: VerificationStatus
    command: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if verification passed."""
        return self.status == VerificationStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "step_name": self.step_name,
            "status": self.status.value,
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


@dataclass
class VerificationStep:
    """Definition of a verification step.

    All values are configurable - no hardcoded defaults in the class.

    Attributes:
        name: Step name
        verification_type: Type of verification
        command: Command to execute
        working_dir: Working directory (optional)
        timeout_seconds: Maximum execution time
        fail_fast: Whether to stop pipeline on failure
        retry_count: Number of retries on failure
        retry_delay_seconds: Delay between retries
        success_exit_codes: Exit codes considered successful
        error_patterns: Patterns to extract errors from output
        warning_patterns: Patterns to extract warnings from output
        pre_hook: Function to run before verification
        post_hook: Function to run after verification
        enabled: Whether step is enabled
        metadata: Additional configuration
    """

    name: str
    verification_type: VerificationType
    command: str
    working_dir: Optional[Path] = None
    timeout_seconds: float = 300.0
    fail_fast: bool = False
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    success_exit_codes: list[int] = field(default_factory=lambda: [0])
    error_patterns: list[str] = field(default_factory=list)
    warning_patterns: list[str] = field(default_factory=list)
    pre_hook: Optional[Callable[[VerificationStep], bool]] = None
    post_hook: Optional[Callable[[VerificationStep, VerificationResult], None]] = None
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without callables).

        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "verification_type": self.verification_type.value,
            "command": self.command,
            "working_dir": str(self.working_dir) if self.working_dir else None,
            "timeout_seconds": self.timeout_seconds,
            "fail_fast": self.fail_fast,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "success_exit_codes": self.success_exit_codes,
            "error_patterns": self.error_patterns,
            "warning_patterns": self.warning_patterns,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationStep:
        """Create from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            VerificationStep instance.
        """
        working_dir = data.get("working_dir")
        return cls(
            name=data.get("name", ""),
            verification_type=VerificationType(data.get("verification_type", "custom")),
            command=data.get("command", ""),
            working_dir=Path(working_dir) if working_dir else None,
            timeout_seconds=data.get("timeout_seconds", 300.0),
            fail_fast=data.get("fail_fast", False),
            retry_count=data.get("retry_count", 0),
            retry_delay_seconds=data.get("retry_delay_seconds", 1.0),
            success_exit_codes=data.get("success_exit_codes", [0]),
            error_patterns=data.get("error_patterns", []),
            warning_patterns=data.get("warning_patterns", []),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {}),
        )


class VerificationPipeline:
    """Pipeline for running verification steps.

    Executes verification steps in order with configurable behavior.
    All parameters are dynamic and configurable.
    """

    def __init__(
        self,
        steps: Optional[list[VerificationStep]] = None,
        command_runner: Optional[CommandRunner] = None,
        stop_on_failure: bool = False,
        parallel_execution: bool = False,
        on_step_start: Optional[Callable[[VerificationStep], None]] = None,
        on_step_complete: Optional[Callable[[VerificationStep, VerificationResult], None]] = None,
    ) -> None:
        """Initialize the verification pipeline.

        Args:
            steps: List of verification steps.
            command_runner: Command runner for executing commands.
            stop_on_failure: Stop pipeline on first failure.
            parallel_execution: Execute steps in parallel.
            on_step_start: Callback when step starts.
            on_step_complete: Callback when step completes.
        """
        self._steps: list[VerificationStep] = steps or []
        self._command_runner = command_runner or CommandRunner()
        self._stop_on_failure = stop_on_failure
        self._parallel_execution = parallel_execution
        self._on_step_start = on_step_start
        self._on_step_complete = on_step_complete
        self._results: list[VerificationResult] = []

    def add_step(self, step: VerificationStep) -> None:
        """Add a verification step.

        Args:
            step: Step to add.
        """
        self._steps.append(step)
        logger.debug(f"Added verification step: {step.name}")

    def remove_step(self, name: str) -> bool:
        """Remove a verification step by name.

        Args:
            name: Name of step to remove.

        Returns:
            True if step was removed.
        """
        for i, step in enumerate(self._steps):
            if step.name == name:
                self._steps.pop(i)
                logger.debug(f"Removed verification step: {name}")
                return True
        return False

    def get_step(self, name: str) -> Optional[VerificationStep]:
        """Get a verification step by name.

        Args:
            name: Step name.

        Returns:
            VerificationStep or None.
        """
        for step in self._steps:
            if step.name == name:
                return step
        return None

    def _execute_step(self, step: VerificationStep) -> VerificationResult:
        """Execute a single verification step.

        Args:
            step: Step to execute.

        Returns:
            VerificationResult.
        """
        start_time = time.time()
        result = VerificationResult(
            step_name=step.name,
            status=VerificationStatus.PENDING,
            command=step.command,
        )

        # Check if enabled
        if not step.enabled:
            result.status = VerificationStatus.SKIPPED
            logger.info(f"Skipped verification step: {step.name} (disabled)")
            return result

        # Run pre-hook
        if step.pre_hook:
            try:
                if not step.pre_hook(step):
                    result.status = VerificationStatus.SKIPPED
                    logger.info(f"Skipped verification step: {step.name} (pre-hook returned False)")
                    return result
            except Exception as e:
                result.status = VerificationStatus.ERROR
                result.errors.append(f"Pre-hook error: {e}")
                return result

        # Notify step start
        if self._on_step_start:
            self._on_step_start(step)

        result.status = VerificationStatus.RUNNING
        logger.info(f"Running verification step: {step.name}")

        # Execute with retries
        attempts = 0
        max_attempts = step.retry_count + 1

        while attempts < max_attempts:
            attempts += 1

            try:
                cmd_result = self._command_runner.run(
                    step.command,
                    timeout_seconds=step.timeout_seconds,
                    cwd=step.working_dir,
                )

                result.stdout = cmd_result.stdout
                result.stderr = cmd_result.stderr
                result.exit_code = cmd_result.exit_code

                # Check success
                if cmd_result.exit_code in step.success_exit_codes:
                    result.status = VerificationStatus.PASSED
                    break
                else:
                    result.status = VerificationStatus.FAILED

                    # Extract errors
                    if step.error_patterns:
                        import re
                        for pattern in step.error_patterns:
                            for match in re.finditer(pattern, cmd_result.stdout + cmd_result.stderr):
                                result.errors.append(match.group(0))
                    else:
                        # Use stderr as errors if no patterns
                        if cmd_result.stderr:
                            result.errors.append(cmd_result.stderr[:1000])

                    # Retry if attempts remain
                    if attempts < max_attempts:
                        logger.warning(
                            f"Verification step {step.name} failed, retrying "
                            f"({attempts}/{max_attempts})"
                        )
                        time.sleep(step.retry_delay_seconds)

            except Exception as e:
                result.status = VerificationStatus.ERROR
                result.errors.append(str(e))
                logger.exception(f"Error running verification step: {step.name}")

                if attempts < max_attempts:
                    time.sleep(step.retry_delay_seconds)

        result.duration_seconds = time.time() - start_time

        # Extract warnings
        if step.warning_patterns:
            import re
            output = result.stdout + result.stderr
            for pattern in step.warning_patterns:
                for match in re.finditer(pattern, output):
                    result.warnings.append(match.group(0))

        # Run post-hook
        if step.post_hook:
            try:
                step.post_hook(step, result)
            except Exception as e:
                logger.warning(f"Post-hook error for {step.name}: {e}")

        # Notify step complete
        if self._on_step_complete:
            self._on_step_complete(step, result)

        logger.info(
            f"Verification step {step.name}: {result.status.value} "
            f"({result.duration_seconds:.2f}s)"
        )

        return result

    def run(
        self,
        working_dir: Optional[Path] = None,
    ) -> list[VerificationResult]:
        """Run all verification steps.

        Args:
            working_dir: Override working directory for all steps.

        Returns:
            List of VerificationResults.
        """
        self._results.clear()
        logger.info(f"Starting verification pipeline with {len(self._steps)} steps")

        for step in self._steps:
            # Override working directory if provided
            if working_dir and not step.working_dir:
                step.working_dir = working_dir

            result = self._execute_step(step)
            self._results.append(result)

            # Check fail fast
            if step.fail_fast and not result.passed:
                logger.warning(
                    f"Stopping pipeline: step '{step.name}' failed with fail_fast=True"
                )
                break

            # Check stop on failure
            if self._stop_on_failure and not result.passed:
                logger.warning(
                    f"Stopping pipeline: step '{step.name}' failed and stop_on_failure=True"
                )
                break

        return self._results

    @property
    def results(self) -> list[VerificationResult]:
        """Get all results."""
        return self._results

    @property
    def all_passed(self) -> bool:
        """Check if all steps passed."""
        return all(r.passed or r.status == VerificationStatus.SKIPPED for r in self._results)

    @property
    def failed_steps(self) -> list[VerificationResult]:
        """Get failed steps."""
        return [r for r in self._results if r.status == VerificationStatus.FAILED]

    def summary(self) -> dict[str, Any]:
        """Generate a summary of verification results.

        Returns:
            Summary dictionary.
        """
        passed = sum(1 for r in self._results if r.passed)
        failed = sum(1 for r in self._results if r.status == VerificationStatus.FAILED)
        skipped = sum(1 for r in self._results if r.status == VerificationStatus.SKIPPED)
        errors = sum(1 for r in self._results if r.status == VerificationStatus.ERROR)

        total_duration = sum(r.duration_seconds for r in self._results)

        return {
            "total_steps": len(self._steps),
            "executed": len(self._results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "all_passed": self.all_passed,
            "total_duration_seconds": total_duration,
            "failed_step_names": [r.step_name for r in self.failed_steps],
        }


def create_verification_pipeline(
    config: dict[str, Any],
    command_runner: Optional[CommandRunner] = None,
) -> VerificationPipeline:
    """Factory function to create a verification pipeline from configuration.

    Args:
        config: Configuration dictionary.
        command_runner: Optional command runner.

    Returns:
        Configured VerificationPipeline.
    """
    steps = []
    for step_config in config.get("verification_commands", []):
        step = VerificationStep.from_dict(step_config)
        steps.append(step)

    return VerificationPipeline(
        steps=steps,
        command_runner=command_runner,
        stop_on_failure=config.get("stop_on_failure", False),
    )


def create_standard_pipeline(
    lint_command: str = "ruff check .",
    type_check_command: str = "mypy src/",
    unit_test_command: str = "pytest tests/unit -v",
    integration_test_command: str = "pytest tests/integration -v",
    working_dir: Optional[Path] = None,
    command_runner: Optional[CommandRunner] = None,
) -> VerificationPipeline:
    """Create a standard verification pipeline.

    All parameters are configurable - no hardcoded commands.

    Args:
        lint_command: Command for linting.
        type_check_command: Command for type checking.
        unit_test_command: Command for unit tests.
        integration_test_command: Command for integration tests.
        working_dir: Working directory.
        command_runner: Command runner.

    Returns:
        Configured VerificationPipeline.
    """
    steps = [
        VerificationStep(
            name="lint",
            verification_type=VerificationType.LINT,
            command=lint_command,
            working_dir=working_dir,
            timeout_seconds=120.0,
            fail_fast=False,
            error_patterns=[r"error:.*", r"E\d+:.*"],
            warning_patterns=[r"warning:.*", r"W\d+:.*"],
        ),
        VerificationStep(
            name="type_check",
            verification_type=VerificationType.TYPE_CHECK,
            command=type_check_command,
            working_dir=working_dir,
            timeout_seconds=180.0,
            fail_fast=False,
            error_patterns=[r"error:.*", r".*: error:.*"],
        ),
        VerificationStep(
            name="unit_tests",
            verification_type=VerificationType.UNIT_TEST,
            command=unit_test_command,
            working_dir=working_dir,
            timeout_seconds=300.0,
            fail_fast=False,
            error_patterns=[r"FAILED.*", r"ERROR.*", r"AssertionError.*"],
        ),
        VerificationStep(
            name="integration_tests",
            verification_type=VerificationType.INTEGRATION_TEST,
            command=integration_test_command,
            working_dir=working_dir,
            timeout_seconds=600.0,
            fail_fast=False,
            error_patterns=[r"FAILED.*", r"ERROR.*", r"AssertionError.*"],
        ),
    ]

    return VerificationPipeline(
        steps=steps,
        command_runner=command_runner,
    )
