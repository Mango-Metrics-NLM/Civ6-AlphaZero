"""
Test Framework with Fixtures and Utilities.

This module provides a comprehensive testing framework with fixtures,
utilities, and patterns for testing agentic coding systems.

Example:
    >>> from template_system.testing import TestRunner, TestConfig
    >>> config = TestConfig(coverage_threshold=80)
    >>> runner = TestRunner(config)
    >>> result = runner.run_all()
    >>> print(result.summary())
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Status of a test execution."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    RUNNING = "running"


class TestType(Enum):
    """Types of tests."""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    SMOKE = "smoke"
    PERFORMANCE = "performance"


@dataclass
class TestConfig:
    """Configuration for test execution.

    Attributes:
        test_dir: Directory containing tests.
        coverage_threshold: Minimum coverage percentage.
        timeout_seconds: Test execution timeout.
        parallel: Enable parallel test execution.
        parallel_workers: Number of parallel workers.
        verbose: Enable verbose output.
        fail_fast: Stop on first failure.
        markers: Pytest markers to include/exclude.
        coverage_omit: Patterns to omit from coverage.
    """

    test_dir: Path = field(default_factory=lambda: Path("tests"))
    coverage_threshold: int = 80
    timeout_seconds: int = 300
    parallel: bool = False
    parallel_workers: int = 4
    verbose: bool = True
    fail_fast: bool = False
    markers: list[str] = field(default_factory=list)
    coverage_omit: list[str] = field(
        default_factory=lambda: ["*/tests/*", "*/__init__.py"]
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestConfig:
        """Create TestConfig from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            TestConfig instance.
        """
        if "test_dir" in data and isinstance(data["test_dir"], str):
            data["test_dir"] = Path(data["test_dir"])
        return cls(**data)

    @classmethod
    def from_env(cls, prefix: str = "TEST_") -> TestConfig:
        """Create TestConfig from environment variables.

        Args:
            prefix: Environment variable prefix.

        Returns:
            TestConfig instance.
        """
        config = cls()

        if test_dir := os.environ.get(f"{prefix}DIR"):
            config.test_dir = Path(test_dir)

        if threshold := os.environ.get(f"{prefix}COVERAGE_THRESHOLD"):
            config.coverage_threshold = int(threshold)

        if timeout := os.environ.get(f"{prefix}TIMEOUT"):
            config.timeout_seconds = int(timeout)

        if parallel := os.environ.get(f"{prefix}PARALLEL"):
            config.parallel = parallel.lower() in ("true", "1", "yes")

        return config


@dataclass
class TestResult:
    """Result of a single test.

    Attributes:
        name: Test name.
        status: Test status.
        duration_seconds: Test duration.
        message: Status message or error.
        output: Test output.
        file: Test file path.
        line: Test line number.
    """

    name: str
    status: TestStatus
    duration_seconds: float = 0.0
    message: str = ""
    output: str = ""
    file: str = ""
    line: int = 0


@dataclass
class TestSuiteResult:
    """Result of a test suite execution.

    Attributes:
        test_type: Type of tests run.
        results: Individual test results.
        total_duration: Total execution time.
        coverage_percent: Code coverage percentage.
        start_time: When execution started.
        end_time: When execution ended.
    """

    test_type: TestType
    results: list[TestResult] = field(default_factory=list)
    total_duration: float = 0.0
    coverage_percent: float = 0.0
    start_time: str = ""
    end_time: str = ""

    @property
    def passed(self) -> int:
        """Count of passed tests."""
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        """Count of failed tests."""
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @property
    def skipped(self) -> int:
        """Count of skipped tests."""
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

    @property
    def errors(self) -> int:
        """Count of tests with errors."""
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)

    @property
    def total(self) -> int:
        """Total number of tests."""
        return len(self.results)

    @property
    def success(self) -> bool:
        """Whether all tests passed."""
        return self.failed == 0 and self.errors == 0

    def summary(self) -> str:
        """Generate a summary string.

        Returns:
            Human-readable summary.
        """
        status = "PASSED" if self.success else "FAILED"
        return (
            f"{self.test_type.value.upper()} Tests: {status}\n"
            f"  Total: {self.total}, Passed: {self.passed}, "
            f"Failed: {self.failed}, Skipped: {self.skipped}, "
            f"Errors: {self.errors}\n"
            f"  Duration: {self.total_duration:.2f}s\n"
            f"  Coverage: {self.coverage_percent:.1f}%"
        )


@dataclass
class CoverageReport:
    """Code coverage report.

    Attributes:
        total_statements: Total statements.
        covered_statements: Statements covered.
        missing_lines: Lines not covered.
        excluded_lines: Lines excluded.
        branch_coverage: Branch coverage data.
    """

    total_statements: int = 0
    covered_statements: int = 0
    missing_lines: dict[str, list[int]] = field(default_factory=dict)
    excluded_lines: dict[str, list[int]] = field(default_factory=dict)
    branch_coverage: dict[str, float] = field(default_factory=dict)

    @property
    def coverage_percent(self) -> float:
        """Calculate coverage percentage."""
        if self.total_statements == 0:
            return 0.0
        return (self.covered_statements / self.total_statements) * 100


class TestSuite:
    """Collection of tests to run together.

    This class groups related tests and provides methods for
    filtered execution.
    """

    def __init__(
        self,
        name: str,
        test_type: TestType,
        path: Path,
        markers: list[str] | None = None,
    ) -> None:
        """Initialize the test suite.

        Args:
            name: Suite name.
            test_type: Type of tests.
            path: Path to tests.
            markers: Pytest markers.
        """
        self.name = name
        self.test_type = test_type
        self.path = path
        self.markers = markers or []
        self._tests: list[str] = []

    def discover(self) -> list[str]:
        """Discover tests in the suite.

        Returns:
            List of test names.
        """
        # Use pytest --collect-only to discover tests
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.path),
            "--collect-only", "-q",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self._tests = [
                line.strip()
                for line in result.stdout.split("\n")
                if "::" in line and line.strip()
            ]
        except subprocess.TimeoutExpired:
            logger.warning(f"Test discovery timed out for {self.name}")
            self._tests = []

        return self._tests

    @property
    def test_count(self) -> int:
        """Number of tests in the suite."""
        return len(self._tests)


class TestRunner:
    """Test runner for executing test suites.

    This class provides methods for running tests with various
    configurations and collecting results.

    Example:
        >>> runner = TestRunner()
        >>> result = runner.run_unit_tests()
        >>> print(result.summary())
    """

    def __init__(
        self,
        config: TestConfig | None = None,
        package_name: str = "",
    ) -> None:
        """Initialize the test runner.

        Args:
            config: Test configuration.
            package_name: Package name for coverage.
        """
        self.config = config or TestConfig()
        self.package_name = package_name
        self._suites: dict[TestType, TestSuite] = {}
        self._logger = logger.getChild(self.__class__.__name__)

    def register_suite(self, suite: TestSuite) -> None:
        """Register a test suite.

        Args:
            suite: Test suite to register.
        """
        self._suites[suite.test_type] = suite

    def run_all(self, with_coverage: bool = True) -> TestSuiteResult:
        """Run all registered test suites.

        Args:
            with_coverage: Enable coverage collection.

        Returns:
            Combined test results.
        """
        return self._run_pytest(
            paths=[str(self.config.test_dir)],
            with_coverage=with_coverage,
            test_type=TestType.UNIT,  # Default type for combined
        )

    def run_unit_tests(self, with_coverage: bool = True) -> TestSuiteResult:
        """Run unit tests.

        Args:
            with_coverage: Enable coverage collection.

        Returns:
            Test results.
        """
        unit_path = self.config.test_dir / "unit"
        return self._run_pytest(
            paths=[str(unit_path)],
            with_coverage=with_coverage,
            test_type=TestType.UNIT,
        )

    def run_integration_tests(self, with_coverage: bool = False) -> TestSuiteResult:
        """Run integration tests.

        Args:
            with_coverage: Enable coverage collection.

        Returns:
            Test results.
        """
        int_path = self.config.test_dir / "integration"
        return self._run_pytest(
            paths=[str(int_path)],
            with_coverage=with_coverage,
            test_type=TestType.INTEGRATION,
        )

    def run_by_marker(
        self,
        marker: str,
        with_coverage: bool = False,
    ) -> TestSuiteResult:
        """Run tests with a specific marker.

        Args:
            marker: Pytest marker.
            with_coverage: Enable coverage collection.

        Returns:
            Test results.
        """
        return self._run_pytest(
            paths=[str(self.config.test_dir)],
            with_coverage=with_coverage,
            test_type=TestType.UNIT,
            extra_args=["-m", marker],
        )

    def run_by_keyword(
        self,
        keyword: str,
        with_coverage: bool = False,
    ) -> TestSuiteResult:
        """Run tests matching a keyword.

        Args:
            keyword: Keyword to match.
            with_coverage: Enable coverage collection.

        Returns:
            Test results.
        """
        return self._run_pytest(
            paths=[str(self.config.test_dir)],
            with_coverage=with_coverage,
            test_type=TestType.UNIT,
            extra_args=["-k", keyword],
        )

    def run_single_test(
        self,
        test_path: str,
        with_coverage: bool = False,
    ) -> TestSuiteResult:
        """Run a single test.

        Args:
            test_path: Path to test (file::test_name format).
            with_coverage: Enable coverage collection.

        Returns:
            Test results.
        """
        return self._run_pytest(
            paths=[test_path],
            with_coverage=with_coverage,
            test_type=TestType.UNIT,
        )

    def _run_pytest(
        self,
        paths: list[str],
        with_coverage: bool,
        test_type: TestType,
        extra_args: list[str] | None = None,
    ) -> TestSuiteResult:
        """Execute pytest with specified configuration.

        Args:
            paths: Test paths to run.
            with_coverage: Enable coverage.
            test_type: Type of tests.
            extra_args: Additional pytest arguments.

        Returns:
            Test results.
        """
        start_time = time.time()

        cmd = [sys.executable, "-m", "pytest"]
        cmd.extend(paths)

        if self.config.verbose:
            cmd.append("-v")

        if self.config.fail_fast:
            cmd.append("-x")

        if self.config.parallel and self.config.parallel_workers > 1:
            cmd.extend(["-n", str(self.config.parallel_workers)])

        if with_coverage and self.package_name:
            cmd.extend([
                f"--cov={self.package_name}",
                "--cov-report=json",
                "--cov-report=term-missing",
            ])
            for pattern in self.config.coverage_omit:
                cmd.append(f"--cov-omit={pattern}")

        # Output JSON report
        with tempfile.NamedTemporaryFile(
            suffix=".json",
            delete=False,
        ) as report_file:
            cmd.extend(["--json-report", f"--json-report-file={report_file.name}"])
            json_report_path = report_file.name

        if extra_args:
            cmd.extend(extra_args)

        self._logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            output = result.stdout + result.stderr
            _ = result.returncode  # Available for debugging if needed
        except subprocess.TimeoutExpired:
            self._logger.error("Test execution timed out")
            return TestSuiteResult(
                test_type=test_type,
                results=[
                    TestResult(
                        name="timeout",
                        status=TestStatus.ERROR,
                        message="Test execution timed out",
                    )
                ],
            )
        finally:
            # Clean up temp file
            if os.path.exists(json_report_path):
                try:
                    os.unlink(json_report_path)
                except OSError:
                    pass

        end_time = time.time()

        # Parse results
        suite_result = self._parse_output(output, test_type)
        suite_result.total_duration = end_time - start_time
        suite_result.start_time = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(start_time),
        )
        suite_result.end_time = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(end_time),
        )

        # Parse coverage
        if with_coverage:
            suite_result.coverage_percent = self._parse_coverage(output)

        return suite_result

    def _parse_output(
        self,
        output: str,
        test_type: TestType,
    ) -> TestSuiteResult:
        """Parse pytest output to extract results.

        Args:
            output: Pytest output.
            test_type: Type of tests.

        Returns:
            Parsed test results.
        """
        results: list[TestResult] = []

        for line in output.split("\n"):
            line = line.strip()

            # Parse test result lines
            if " PASSED" in line:
                name = line.split(" PASSED")[0].strip()
                results.append(TestResult(name=name, status=TestStatus.PASSED))
            elif " FAILED" in line:
                name = line.split(" FAILED")[0].strip()
                results.append(TestResult(name=name, status=TestStatus.FAILED))
            elif " SKIPPED" in line:
                name = line.split(" SKIPPED")[0].strip()
                results.append(TestResult(name=name, status=TestStatus.SKIPPED))
            elif " ERROR" in line and "::" in line:
                name = line.split(" ERROR")[0].strip()
                results.append(TestResult(name=name, status=TestStatus.ERROR))

        return TestSuiteResult(test_type=test_type, results=results)

    def _parse_coverage(self, output: str) -> float:
        """Parse coverage percentage from output.

        Args:
            output: Pytest output with coverage.

        Returns:
            Coverage percentage.
        """
        # Look for "TOTAL ... XX%"
        for line in output.split("\n"):
            if "TOTAL" in line and "%" in line:
                parts = line.split()
                for part in parts:
                    if part.endswith("%"):
                        try:
                            return float(part.rstrip("%"))
                        except ValueError:
                            pass
        return 0.0

    def check_coverage_threshold(
        self,
        result: TestSuiteResult,
    ) -> tuple[bool, str]:
        """Check if coverage meets threshold.

        Args:
            result: Test results with coverage.

        Returns:
            Tuple of (passed, message).
        """
        if result.coverage_percent >= self.config.coverage_threshold:
            return True, (
                f"Coverage {result.coverage_percent:.1f}% meets "
                f"threshold {self.config.coverage_threshold}%"
            )
        return False, (
            f"Coverage {result.coverage_percent:.1f}% below "
            f"threshold {self.config.coverage_threshold}%"
        )


T = TypeVar("T")


class MockFactory(Generic[T]):
    """Factory for creating mock objects.

    This class provides utilities for creating mock objects with
    configurable behavior for testing.
    """

    def __init__(self, base_class: type[T] | None = None) -> None:
        """Initialize the factory.

        Args:
            base_class: Base class for mocks.
        """
        self.base_class = base_class
        self._return_values: dict[str, Any] = {}
        self._side_effects: dict[str, Callable[..., Any]] = {}
        self._call_counts: dict[str, int] = {}

    def configure_return(self, method: str, value: Any) -> MockFactory[T]:
        """Configure return value for a method.

        Args:
            method: Method name.
            value: Return value.

        Returns:
            Self for chaining.
        """
        self._return_values[method] = value
        return self

    def configure_side_effect(
        self,
        method: str,
        effect: Callable[..., Any],
    ) -> MockFactory[T]:
        """Configure side effect for a method.

        Args:
            method: Method name.
            effect: Side effect function.

        Returns:
            Self for chaining.
        """
        self._side_effects[method] = effect
        return self

    def create(self) -> Any:
        """Create a mock object.

        Returns:
            Mock object with configured behavior.
        """
        factory = self

        class Mock:
            def __getattr__(self, name: str) -> Any:
                factory._call_counts[name] = factory._call_counts.get(name, 0) + 1

                if name in factory._side_effects:
                    return factory._side_effects[name]

                if name in factory._return_values:
                    return lambda *a, **kw: factory._return_values[name]

                return lambda *a, **kw: None

        return Mock()

    def get_call_count(self, method: str) -> int:
        """Get call count for a method.

        Args:
            method: Method name.

        Returns:
            Number of calls.
        """
        return self._call_counts.get(method, 0)


@contextmanager
def temporary_directory() -> Generator[Path, None, None]:
    """Context manager for a temporary directory.

    Yields:
        Path to temporary directory.
    """
    import shutil
    import tempfile

    path = Path(tempfile.mkdtemp())
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def environment_override(**env_vars: str) -> Generator[None, None, None]:
    """Context manager to temporarily override environment variables.

    Args:
        **env_vars: Environment variables to set.

    Yields:
        None.
    """
    original = {}
    for key, value in env_vars.items():
        original[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        yield
    finally:
        for key, orig_value in original.items():
            if orig_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = orig_value


@contextmanager
def capture_logs(
    logger_name: str = "",
    level: int = logging.DEBUG,
) -> Generator[list[logging.LogRecord], None, None]:
    """Context manager to capture log records.

    Args:
        logger_name: Logger name to capture (empty for root).
        level: Minimum level to capture.

    Yields:
        List of captured log records.
    """
    records: list[logging.LogRecord] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = CaptureHandler(level)
    logger_obj = logging.getLogger(logger_name)
    logger_obj.addHandler(handler)
    original_level = logger_obj.level
    logger_obj.setLevel(level)

    try:
        yield records
    finally:
        logger_obj.removeHandler(handler)
        logger_obj.setLevel(original_level)


def assert_raises_with_message(
    exception_type: type[Exception],
    message_contains: str,
) -> Callable[[Callable[..., Any]], None]:
    """Assert that a callable raises an exception with specific message.

    Args:
        exception_type: Expected exception type.
        message_contains: String that should be in message.

    Returns:
        Decorator that runs the assertion.
    """
    def decorator(func: Callable[..., Any]) -> None:
        try:
            func()
            raise AssertionError(f"Expected {exception_type.__name__} to be raised")
        except exception_type as e:
            if message_contains not in str(e):
                raise AssertionError(
                    f"Expected message to contain '{message_contains}', "
                    f"got '{str(e)}'"
                ) from None
    return decorator


# Pytest fixtures as functions (for use outside pytest)
def create_test_config(**overrides: Any) -> TestConfig:
    """Create a test configuration with overrides.

    Args:
        **overrides: Configuration overrides.

    Returns:
        TestConfig instance.
    """
    return TestConfig(**overrides)


def create_test_runner(
    config: TestConfig | None = None,
    package_name: str = "",
) -> TestRunner:
    """Create a test runner.

    Args:
        config: Test configuration.
        package_name: Package name for coverage.

    Returns:
        TestRunner instance.
    """
    return TestRunner(config, package_name)
