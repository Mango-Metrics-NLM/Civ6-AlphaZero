"""
Unit tests for the testing module.

Tests cover:
- TestConfig creation and loading
- TestResult and TestSuiteResult dataclasses
- TestSuite discovery
- TestRunner execution
- MockFactory utilities
- Test helpers and context managers
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pytest

from template_system.testing import (
    CoverageReport,
    MockFactory,
    TestConfig,
    TestResult,
    TestRunner,
    TestStatus,
    TestSuite,
    TestSuiteResult,
    TestType,
    assert_raises_with_message,
    capture_logs,
    create_test_config,
    create_test_runner,
    environment_override,
    temporary_directory,
)


class TestTestStatus:
    """Tests for TestStatus enum."""

    def test_status_values(self) -> None:
        """Test test status values."""
        assert TestStatus.PASSED.value == "passed"
        assert TestStatus.FAILED.value == "failed"
        assert TestStatus.SKIPPED.value == "skipped"
        assert TestStatus.ERROR.value == "error"
        assert TestStatus.RUNNING.value == "running"


class TestTestType:
    """Tests for TestType enum."""

    def test_type_values(self) -> None:
        """Test test type values."""
        assert TestType.UNIT.value == "unit"
        assert TestType.INTEGRATION.value == "integration"
        assert TestType.E2E.value == "e2e"
        assert TestType.SMOKE.value == "smoke"
        assert TestType.PERFORMANCE.value == "performance"


class TestTestConfig:
    """Tests for TestConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TestConfig()
        assert config.test_dir == Path("tests")
        assert config.coverage_threshold == 80
        assert config.timeout_seconds == 300
        assert config.parallel is False
        assert config.verbose is True

    def test_from_dict(self) -> None:
        """Test creating config from dictionary."""
        config = TestConfig.from_dict({
            "test_dir": "/custom/tests",
            "coverage_threshold": 90,
            "parallel": True,
        })
        assert config.test_dir == Path("/custom/tests")
        assert config.coverage_threshold == 90
        assert config.parallel is True

    def test_from_env(self) -> None:
        """Test creating config from environment."""
        os.environ["TEST_DIR"] = "/env/tests"
        os.environ["TEST_COVERAGE_THRESHOLD"] = "95"
        os.environ["TEST_PARALLEL"] = "true"
        try:
            config = TestConfig.from_env()
            assert config.test_dir == Path("/env/tests")
            assert config.coverage_threshold == 95
            assert config.parallel is True
        finally:
            os.environ.pop("TEST_DIR", None)
            os.environ.pop("TEST_COVERAGE_THRESHOLD", None)
            os.environ.pop("TEST_PARALLEL", None)


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_passed_result(self) -> None:
        """Test creating a passed result."""
        result = TestResult(
            name="test_example",
            status=TestStatus.PASSED,
            duration_seconds=0.5,
        )
        assert result.name == "test_example"
        assert result.status == TestStatus.PASSED

    def test_failed_result(self) -> None:
        """Test creating a failed result."""
        result = TestResult(
            name="test_failing",
            status=TestStatus.FAILED,
            message="Assertion failed",
            output="Expected 1, got 2",
        )
        assert result.status == TestStatus.FAILED
        assert result.message == "Assertion failed"

    def test_result_with_location(self) -> None:
        """Test result with file location."""
        result = TestResult(
            name="test_located",
            status=TestStatus.PASSED,
            file="tests/test_example.py",
            line=42,
        )
        assert result.file == "tests/test_example.py"
        assert result.line == 42


class TestTestSuiteResult:
    """Tests for TestSuiteResult dataclass."""

    def test_counts(self) -> None:
        """Test result counting properties."""
        results = [
            TestResult("t1", TestStatus.PASSED),
            TestResult("t2", TestStatus.PASSED),
            TestResult("t3", TestStatus.FAILED),
            TestResult("t4", TestStatus.SKIPPED),
            TestResult("t5", TestStatus.ERROR),
        ]
        suite_result = TestSuiteResult(
            test_type=TestType.UNIT,
            results=results,
        )
        assert suite_result.passed == 2
        assert suite_result.failed == 1
        assert suite_result.skipped == 1
        assert suite_result.errors == 1
        assert suite_result.total == 5

    def test_success_property(self) -> None:
        """Test success property."""
        passed_suite = TestSuiteResult(
            test_type=TestType.UNIT,
            results=[
                TestResult("t1", TestStatus.PASSED),
                TestResult("t2", TestStatus.SKIPPED),
            ],
        )
        assert passed_suite.success is True

        failed_suite = TestSuiteResult(
            test_type=TestType.UNIT,
            results=[
                TestResult("t1", TestStatus.PASSED),
                TestResult("t2", TestStatus.FAILED),
            ],
        )
        assert failed_suite.success is False

    def test_summary(self) -> None:
        """Test summary generation."""
        suite_result = TestSuiteResult(
            test_type=TestType.UNIT,
            results=[
                TestResult("t1", TestStatus.PASSED),
                TestResult("t2", TestStatus.FAILED),
            ],
            total_duration=1.5,
            coverage_percent=85.5,
        )
        summary = suite_result.summary()
        assert "UNIT Tests" in summary
        assert "FAILED" in summary
        assert "85.5%" in summary


class TestCoverageReport:
    """Tests for CoverageReport dataclass."""

    def test_coverage_percent(self) -> None:
        """Test coverage percentage calculation."""
        report = CoverageReport(
            total_statements=100,
            covered_statements=80,
        )
        assert report.coverage_percent == 80.0

    def test_coverage_percent_zero_statements(self) -> None:
        """Test coverage with zero statements."""
        report = CoverageReport(total_statements=0, covered_statements=0)
        assert report.coverage_percent == 0.0

    def test_missing_lines(self) -> None:
        """Test missing lines tracking."""
        report = CoverageReport(
            total_statements=100,
            covered_statements=90,
            missing_lines={"module.py": [10, 20, 30]},
        )
        assert len(report.missing_lines["module.py"]) == 3


class TestTestSuite:
    """Tests for TestSuite class."""

    def test_suite_creation(self, temp_dir: Path) -> None:
        """Test creating a test suite."""
        suite = TestSuite(
            name="unit_tests",
            test_type=TestType.UNIT,
            path=temp_dir,
            markers=["unit"],
        )
        assert suite.name == "unit_tests"
        assert suite.test_type == TestType.UNIT
        assert "unit" in suite.markers

    def test_test_count_initial(self, temp_dir: Path) -> None:
        """Test initial test count is zero."""
        suite = TestSuite(
            name="empty",
            test_type=TestType.UNIT,
            path=temp_dir,
        )
        assert suite.test_count == 0


class TestTestRunner:
    """Tests for TestRunner class."""

    def test_runner_creation(self) -> None:
        """Test creating a test runner."""
        config = TestConfig(coverage_threshold=90)
        runner = TestRunner(config, package_name="my_package")
        assert runner.config.coverage_threshold == 90
        assert runner.package_name == "my_package"

    def test_check_coverage_threshold_pass(self) -> None:
        """Test coverage threshold check passes."""
        runner = TestRunner(TestConfig(coverage_threshold=80))
        result = TestSuiteResult(
            test_type=TestType.UNIT,
            coverage_percent=85.0,
        )
        passed, message = runner.check_coverage_threshold(result)
        assert passed is True
        assert "meets" in message

    def test_check_coverage_threshold_fail(self) -> None:
        """Test coverage threshold check fails."""
        runner = TestRunner(TestConfig(coverage_threshold=80))
        result = TestSuiteResult(
            test_type=TestType.UNIT,
            coverage_percent=75.0,
        )
        passed, message = runner.check_coverage_threshold(result)
        assert passed is False
        assert "below" in message


class TestMockFactory:
    """Tests for MockFactory class."""

    def test_create_mock(self) -> None:
        """Test creating a mock object."""
        factory: MockFactory[Any] = MockFactory()
        mock = factory.create()
        assert mock is not None

    def test_configure_return_value(self) -> None:
        """Test configuring return values."""
        factory: MockFactory[Any] = MockFactory()
        factory.configure_return("get_value", 42)
        mock = factory.create()
        result = mock.get_value()
        assert result == 42

    def test_configure_side_effect(self) -> None:
        """Test configuring side effects."""
        factory: MockFactory[Any] = MockFactory()
        factory.configure_side_effect("process", lambda x: x * 2)
        mock = factory.create()
        result = mock.process(5)
        assert result == 10

    def test_call_count_tracking(self) -> None:
        """Test call count tracking."""
        factory: MockFactory[Any] = MockFactory()
        mock = factory.create()
        mock.method_a()
        mock.method_a()
        mock.method_b()
        assert factory.get_call_count("method_a") == 2
        assert factory.get_call_count("method_b") == 1
        assert factory.get_call_count("method_c") == 0

    def test_chained_configuration(self) -> None:
        """Test chained configuration calls."""
        factory: MockFactory[Any] = MockFactory()
        factory.configure_return("a", 1).configure_return("b", 2)
        mock = factory.create()
        assert mock.a() == 1
        assert mock.b() == 2


class TestTemporaryDirectory:
    """Tests for temporary_directory context manager."""

    def test_creates_directory(self) -> None:
        """Test directory is created."""
        with temporary_directory() as temp_path:
            assert temp_path.exists()
            assert temp_path.is_dir()

    def test_cleans_up_directory(self) -> None:
        """Test directory is cleaned up."""
        temp_path: Path | None = None
        with temporary_directory() as path:
            temp_path = path
            # Create a file to test cleanup
            (path / "test.txt").write_text("test")

        assert not temp_path.exists()


class TestEnvironmentOverride:
    """Tests for environment_override context manager."""

    def test_sets_environment_variable(self) -> None:
        """Test environment variable is set."""
        with environment_override(TEST_VAR="test_value"):
            assert os.environ.get("TEST_VAR") == "test_value"

    def test_restores_environment_variable(self) -> None:
        """Test environment variable is restored."""
        os.environ["EXISTING_VAR"] = "original"
        try:
            with environment_override(EXISTING_VAR="modified"):
                assert os.environ["EXISTING_VAR"] == "modified"
            assert os.environ["EXISTING_VAR"] == "original"
        finally:
            os.environ.pop("EXISTING_VAR", None)

    def test_removes_new_variable(self) -> None:
        """Test new variable is removed after context."""
        assert "NEW_TEST_VAR" not in os.environ
        with environment_override(NEW_TEST_VAR="value"):
            assert os.environ["NEW_TEST_VAR"] == "value"
        assert "NEW_TEST_VAR" not in os.environ


class TestCaptureLogs:
    """Tests for capture_logs context manager."""

    def test_captures_logs(self) -> None:
        """Test logs are captured."""
        with capture_logs("test.capture", level=logging.INFO) as records:
            logger = logging.getLogger("test.capture")
            logger.info("Test message")
            logger.warning("Warning message")

        assert len(records) == 2
        assert records[0].message == "Test message"
        assert records[1].levelname == "WARNING"

    def test_captures_at_level(self) -> None:
        """Test only logs at level or above are captured."""
        with capture_logs("test.level", level=logging.WARNING) as records:
            logger = logging.getLogger("test.level")
            logger.debug("Debug")
            logger.info("Info")
            logger.warning("Warning")
            logger.error("Error")

        # Only WARNING and ERROR should be captured
        levels = [r.levelname for r in records]
        assert "DEBUG" not in levels
        assert "INFO" not in levels
        assert "WARNING" in levels
        assert "ERROR" in levels


class TestAssertRaisesWithMessage:
    """Tests for assert_raises_with_message helper."""

    def test_passes_on_correct_exception(self) -> None:
        """Test passes when correct exception with message raised."""
        def raise_value_error() -> None:
            raise ValueError("Expected error message")

        assert_raises_with_message(ValueError, "Expected")(raise_value_error)

    def test_fails_on_no_exception(self) -> None:
        """Test fails when no exception raised."""
        def no_error() -> None:
            pass

        with pytest.raises(AssertionError, match="Expected.*to be raised"):
            assert_raises_with_message(ValueError, "any")(no_error)

    def test_fails_on_wrong_message(self) -> None:
        """Test fails when message doesn't match."""
        def raise_wrong_message() -> None:
            raise ValueError("Wrong message")

        with pytest.raises(AssertionError, match="Expected message to contain"):
            assert_raises_with_message(ValueError, "Expected")(raise_wrong_message)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_test_config(self) -> None:
        """Test create_test_config helper."""
        config = create_test_config(coverage_threshold=95, parallel=True)
        assert config.coverage_threshold == 95
        assert config.parallel is True

    def test_create_test_runner(self) -> None:
        """Test create_test_runner helper."""
        runner = create_test_runner(package_name="my_pkg")
        assert runner.package_name == "my_pkg"
