"""Test Runner Utilities for Automated Testing.

This module provides utilities for running tests, collecting coverage,
and generating test reports. It integrates with pytest and provides
a programmatic interface for test execution.

Usage:
    from alpha_zero.utils.test_runner import PytestRunner, RunConfig

    # Create runner from environment
    runner = PytestRunner.from_env()

    # Run all tests
    result = runner.run_all()

    # Run specific test categories
    result = runner.run_unit_tests()
    result = runner.run_integration_tests()

    # Run with coverage
    result = runner.run_with_coverage()
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RunStatus(Enum):
    """Test execution status."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    NOT_RUN = "not_run"


@dataclass
class RunResult:
    """Result of a test execution."""

    status: RunStatus
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    coverage_percent: float | None = None
    output: str = ""
    error_output: str = ""
    failed_tests: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        """Check if all tests passed."""
        return self.status == RunStatus.PASSED and self.failed == 0 and self.errors == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "duration_seconds": self.duration_seconds,
            "coverage_percent": self.coverage_percent,
            "failed_tests": self.failed_tests,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Test Results ({self.timestamp.strftime('%Y-%m-%d %H:%M:%S')})",
            "=" * 50,
            f"Status: {self.status.value.upper()}",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Total: {self.total_tests} | Passed: {self.passed} | Failed: {self.failed} | Errors: {self.errors} | Skipped: {self.skipped}",
        ]

        if self.coverage_percent is not None:
            lines.append(f"Coverage: {self.coverage_percent:.1f}%")

        if self.failed_tests:
            lines.append("\nFailed Tests:")
            for test in self.failed_tests[:10]:  # Limit to first 10
                lines.append(f"  - {test}")
            if len(self.failed_tests) > 10:
                lines.append(f"  ... and {len(self.failed_tests) - 10} more")

        return "\n".join(lines)


@dataclass
class RunConfig:
    """Configuration for test execution."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    tests_dir: str = "tests"
    unit_tests_dir: str = "tests/unit"
    integration_tests_dir: str = "tests/integration"

    # Execution settings
    timeout_seconds: int = 300
    parallel: bool = True
    num_workers: int | None = None  # None = auto
    verbose: bool = True
    fail_fast: bool = False

    # Coverage settings
    coverage_enabled: bool = True
    coverage_source: list[str] = field(default_factory=lambda: ["alpha_zero"])
    coverage_min_percent: float = 80.0
    coverage_report_formats: list[str] = field(default_factory=lambda: ["term", "html"])

    # Output settings
    output_dir: str = "test_results"
    junit_xml: bool = True
    json_report: bool = True

    # Markers and filters
    markers: list[str] = field(default_factory=list)
    exclude_markers: list[str] = field(default_factory=list)
    keywords: str | None = None

    @classmethod
    def from_env(cls) -> RunConfig:
        """Create configuration from environment variables."""
        return cls(
            project_root=Path(os.getenv("PROJECT_ROOT", Path.cwd())),
            timeout_seconds=int(os.getenv("TEST_TIMEOUT", "300")),
            parallel=os.getenv("TEST_PARALLEL", "true").lower() == "true",
            verbose=os.getenv("TEST_VERBOSE", "true").lower() == "true",
            coverage_enabled=os.getenv("TEST_COVERAGE", "true").lower() == "true",
            coverage_min_percent=float(os.getenv("TEST_COVERAGE_MIN", "80")),
            fail_fast=os.getenv("TEST_FAIL_FAST", "false").lower() == "true",
        )


class PytestRunner:
    """Test runner with comprehensive test execution capabilities."""

    def __init__(self, config: RunConfig | None = None):
        """Initialize test runner.

        Args:
            config: Test configuration. If None, uses defaults.
        """
        self.config = config or RunConfig()
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Ensure output directory exists."""
        output_path = self.config.project_root / self.config.output_dir
        output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> PytestRunner:
        """Create runner from environment variables."""
        return cls(RunConfig.from_env())

    def _build_pytest_args(
        self,
        test_path: str | None = None,
        extra_args: list[str] | None = None,
        with_coverage: bool = False,
    ) -> list[str]:
        """Build pytest command arguments.

        Args:
            test_path: Specific test path to run.
            extra_args: Additional pytest arguments.
            with_coverage: Whether to include coverage.

        Returns:
            List of command arguments.
        """
        args = [sys.executable, "-m", "pytest"]

        # Test path
        if test_path:
            args.append(str(self.config.project_root / test_path))
        else:
            args.append(str(self.config.project_root / self.config.tests_dir))

        # Verbosity
        if self.config.verbose:
            args.append("-v")

        # Parallel execution
        if self.config.parallel:
            workers = self.config.num_workers or "auto"
            args.extend(["-n", str(workers)])

        # Fail fast
        if self.config.fail_fast:
            args.append("-x")

        # Timeout
        args.extend(["--timeout", str(self.config.timeout_seconds)])

        # Coverage
        if with_coverage and self.config.coverage_enabled:
            for source in self.config.coverage_source:
                args.extend(["--cov", source])
            args.append("--cov-report=term-missing")

            if "html" in self.config.coverage_report_formats:
                output_dir = self.config.project_root / self.config.output_dir / "htmlcov"
                args.extend(["--cov-report", f"html:{output_dir}"])

        # JUnit XML
        if self.config.junit_xml:
            xml_path = self.config.project_root / self.config.output_dir / "junit.xml"
            args.extend(["--junit-xml", str(xml_path)])

        # Markers
        for marker in self.config.markers:
            args.extend(["-m", marker])

        for marker in self.config.exclude_markers:
            args.extend(["-m", f"not {marker}"])

        # Keywords
        if self.config.keywords:
            args.extend(["-k", self.config.keywords])

        # Extra args
        if extra_args:
            args.extend(extra_args)

        return args

    def _run_pytest(
        self,
        test_path: str | None = None,
        extra_args: list[str] | None = None,
        with_coverage: bool = False,
    ) -> RunResult:
        """Execute pytest and parse results.

        Args:
            test_path: Specific test path to run.
            extra_args: Additional pytest arguments.
            with_coverage: Whether to include coverage.

        Returns:
            RunResult with execution details.
        """
        args = self._build_pytest_args(test_path, extra_args, with_coverage)

        logger.info(f"Running: {' '.join(args)}")
        start_time = time.time()

        try:
            result = subprocess.run(
                args,
                cwd=str(self.config.project_root),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds + 60,  # Extra buffer
            )

            duration = time.time() - start_time
            return self._parse_pytest_output(result, duration, with_coverage)

        except subprocess.TimeoutExpired:
            return RunResult(
                status=RunStatus.ERROR,
                duration_seconds=time.time() - start_time,
                error_output="Test execution timed out",
            )
        except Exception as e:
            return RunResult(
                status=RunStatus.ERROR,
                duration_seconds=time.time() - start_time,
                error_output=str(e),
            )

    def _parse_pytest_output(
        self,
        result: subprocess.CompletedProcess,
        duration: float,
        with_coverage: bool,
    ) -> RunResult:
        """Parse pytest output to extract results.

        Args:
            result: Subprocess result.
            duration: Execution duration in seconds.
            with_coverage: Whether coverage was included.

        Returns:
            Parsed RunResult.
        """
        output = result.stdout
        error_output = result.stderr

        # Parse summary line like "5 passed, 2 failed, 1 error in 10.5s"
        total = passed = failed = errors = skipped = 0
        failed_tests: list[str] = []

        for line in output.split("\n"):
            line_lower = line.lower()

            # Parse summary
            if "passed" in line_lower or "failed" in line_lower:
                import re

                numbers = re.findall(r"(\d+)\s+(passed|failed|error|skipped)", line_lower)
                for count, status in numbers:
                    count = int(count)
                    if status == "passed":
                        passed = count
                    elif status == "failed":
                        failed = count
                    elif status == "error":
                        errors = count
                    elif status == "skipped":
                        skipped = count

            # Capture failed test names
            if line.startswith("FAILED "):
                test_name = line.replace("FAILED ", "").split(" ")[0]
                failed_tests.append(test_name)

        total = passed + failed + errors + skipped

        # Parse coverage if available
        coverage_percent = None
        if with_coverage:
            for line in output.split("\n"):
                if "TOTAL" in line and "%" in line:
                    import re

                    match = re.search(r"(\d+)%", line)
                    if match:
                        coverage_percent = float(match.group(1))

        # Determine status
        if result.returncode == 0:
            status = RunStatus.PASSED
        elif failed > 0 or errors > 0:
            status = RunStatus.FAILED
        else:
            status = RunStatus.ERROR

        return RunResult(
            status=status,
            total_tests=total,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration_seconds=duration,
            coverage_percent=coverage_percent,
            output=output,
            error_output=error_output,
            failed_tests=failed_tests,
        )

    def run_all(self, with_coverage: bool = True) -> RunResult:
        """Run all tests.

        Args:
            with_coverage: Whether to include coverage reporting.

        Returns:
            RunResult with execution details.
        """
        logger.info("Running all tests...")
        return self._run_pytest(with_coverage=with_coverage)

    def run_unit_tests(self, with_coverage: bool = False) -> RunResult:
        """Run only unit tests.

        Args:
            with_coverage: Whether to include coverage reporting.

        Returns:
            RunResult with execution details.
        """
        logger.info("Running unit tests...")
        return self._run_pytest(
            test_path=self.config.unit_tests_dir,
            with_coverage=with_coverage,
        )

    def run_integration_tests(self, with_coverage: bool = False) -> RunResult:
        """Run only integration tests.

        Args:
            with_coverage: Whether to include coverage reporting.

        Returns:
            RunResult with execution details.
        """
        logger.info("Running integration tests...")
        return self._run_pytest(
            test_path=self.config.integration_tests_dir,
            with_coverage=with_coverage,
        )

    def run_specific(
        self,
        test_path: str,
        with_coverage: bool = False,
    ) -> RunResult:
        """Run specific test file or directory.

        Args:
            test_path: Path to test file or directory.
            with_coverage: Whether to include coverage reporting.

        Returns:
            RunResult with execution details.
        """
        logger.info(f"Running tests: {test_path}")
        return self._run_pytest(
            test_path=test_path,
            with_coverage=with_coverage,
        )

    def run_by_marker(
        self,
        marker: str,
        with_coverage: bool = False,
    ) -> RunResult:
        """Run tests with specific marker.

        Args:
            marker: Pytest marker to filter by.
            with_coverage: Whether to include coverage reporting.

        Returns:
            RunResult with execution details.
        """
        logger.info(f"Running tests with marker: {marker}")
        original_markers = self.config.markers
        self.config.markers = [marker]

        try:
            return self._run_pytest(with_coverage=with_coverage)
        finally:
            self.config.markers = original_markers

    def run_by_keyword(
        self,
        keyword: str,
        with_coverage: bool = False,
    ) -> RunResult:
        """Run tests matching keyword.

        Args:
            keyword: Keyword to filter tests by.
            with_coverage: Whether to include coverage reporting.

        Returns:
            RunResult with execution details.
        """
        logger.info(f"Running tests matching: {keyword}")
        original_keyword = self.config.keywords
        self.config.keywords = keyword

        try:
            return self._run_pytest(with_coverage=with_coverage)
        finally:
            self.config.keywords = original_keyword

    def run_with_coverage(self) -> RunResult:
        """Run all tests with coverage reporting.

        Returns:
            RunResult with coverage details.
        """
        return self.run_all(with_coverage=True)

    def save_results(self, result: RunResult, filename: str | None = None) -> Path:
        """Save test results to JSON file.

        Args:
            result: Test result to save.
            filename: Optional filename. Defaults to timestamped name.

        Returns:
            Path to saved file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"

        output_path = self.config.project_root / self.config.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Results saved to: {output_path}")
        return output_path

    def verify_coverage(self, result: RunResult) -> bool:
        """Verify coverage meets minimum threshold.

        Args:
            result: Test result with coverage data.

        Returns:
            True if coverage meets threshold.
        """
        if result.coverage_percent is None:
            logger.warning("No coverage data available")
            return False

        meets_threshold = result.coverage_percent >= self.config.coverage_min_percent
        if not meets_threshold:
            logger.warning(
                f"Coverage {result.coverage_percent:.1f}% below threshold "
                f"{self.config.coverage_min_percent:.1f}%",
            )
        return meets_threshold


# Convenience functions
def run_tests(
    test_path: str | None = None,
    with_coverage: bool = True,
    verbose: bool = True,
) -> RunResult:
    """Quick function to run tests.

    Args:
        test_path: Optional specific test path.
        with_coverage: Whether to include coverage.
        verbose: Whether to show verbose output.

    Returns:
        RunResult with execution details.
    """
    config = RunConfig.from_env()
    config.verbose = verbose

    runner = PytestRunner(config)

    if test_path:
        return runner.run_specific(test_path, with_coverage=with_coverage)
    return runner.run_all(with_coverage=with_coverage)


def quick_test(keyword: str | None = None) -> bool:
    """Run a quick test and return success status.

    Args:
        keyword: Optional keyword to filter tests.

    Returns:
        True if tests passed.
    """
    config = RunConfig(
        verbose=False,
        coverage_enabled=False,
        parallel=True,
    )
    runner = PytestRunner(config)

    if keyword:
        result = runner.run_by_keyword(keyword)
    else:
        result = runner.run_unit_tests()

    return result.success
