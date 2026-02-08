"""Tests for test runner module."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from alpha_zero.utils.test_runner import (
    PytestRunner,
    RunConfig,
    RunResult,
    RunStatus,
    quick_test,
    run_tests,
)

# Use new names in tests
# Use new names in tests
Runner = PytestRunner
RunnerConfig = RunConfig
RunnerResult = RunResult
RunnerStatus = RunStatus


class TestRunnerStatus:
    """Tests for RunnerStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert RunnerStatus.PASSED.value == "passed"
        assert RunnerStatus.FAILED.value == "failed"
        assert RunnerStatus.ERROR.value == "error"
        assert RunnerStatus.SKIPPED.value == "skipped"
        assert RunnerStatus.NOT_RUN.value == "not_run"


class TestRunnerResult:
    """Tests for RunnerResult dataclass."""

    def test_default_result(self):
        """Test default result values."""
        result = RunnerResult(status=RunnerStatus.NOT_RUN)
        assert result.status == RunnerStatus.NOT_RUN
        assert result.total_tests == 0
        assert result.passed == 0
        assert result.failed == 0
        assert result.coverage_percent is None

    def test_success_property_passed(self):
        """Test success property when tests pass."""
        result = RunnerResult(
            status=RunnerStatus.PASSED,
            total_tests=10,
            passed=10,
            failed=0,
            errors=0,
        )
        assert result.success is True

    def test_success_property_failed(self):
        """Test success property when tests fail."""
        result = RunnerResult(
            status=RunnerStatus.FAILED,
            total_tests=10,
            passed=8,
            failed=2,
            errors=0,
        )
        assert result.success is False

    def test_success_property_with_errors(self):
        """Test success property with errors."""
        result = RunnerResult(
            status=RunnerStatus.PASSED,
            total_tests=10,
            passed=10,
            failed=0,
            errors=1,
        )
        assert result.success is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = RunnerResult(
            status=RunnerStatus.PASSED,
            total_tests=5,
            passed=5,
            failed=0,
            errors=0,
            skipped=0,
            duration_seconds=1.5,
            coverage_percent=85.0,
        )
        data = result.to_dict()

        assert data["status"] == "passed"
        assert data["total_tests"] == 5
        assert data["passed"] == 5
        assert data["duration_seconds"] == 1.5
        assert data["coverage_percent"] == 85.0
        assert data["success"] is True

    def test_summary(self):
        """Test summary generation."""
        result = RunnerResult(
            status=RunnerStatus.PASSED,
            total_tests=10,
            passed=8,
            failed=1,
            errors=0,
            skipped=1,
            duration_seconds=5.5,
            coverage_percent=82.5,
        )
        summary = result.summary()

        assert "PASSED" in summary
        assert "5.50s" in summary
        assert "Total: 10" in summary
        assert "Passed: 8" in summary
        assert "Failed: 1" in summary
        assert "Coverage: 82.5%" in summary

    def test_summary_with_failed_tests(self):
        """Test summary with failed test names."""
        result = RunnerResult(
            status=RunnerStatus.FAILED,
            total_tests=5,
            passed=3,
            failed=2,
            failed_tests=["test_foo", "test_bar"],
        )
        summary = result.summary()

        assert "Failed Tests:" in summary
        assert "test_foo" in summary
        assert "test_bar" in summary


class TestRunnerConfig:
    """Tests for RunnerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RunnerConfig()
        assert config.tests_dir == "tests"
        assert config.unit_tests_dir == "tests/unit"
        assert config.timeout_seconds == 300
        assert config.parallel is True
        assert config.coverage_enabled is True
        assert config.coverage_min_percent == 80.0

    def test_from_env(self):
        """Test configuration from environment."""
        env_vars = {
            "TEST_TIMEOUT": "600",
            "TEST_PARALLEL": "false",
            "TEST_COVERAGE": "true",
            "TEST_COVERAGE_MIN": "90",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = RunnerConfig.from_env()
            assert config.timeout_seconds == 600
            assert config.parallel is False
            assert config.coverage_enabled is True
            assert config.coverage_min_percent == 90.0

    def test_from_env_defaults(self):
        """Test from_env uses defaults when env vars not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = RunnerConfig.from_env()
            assert config.timeout_seconds == 300
            assert config.parallel is True


class TestRunner:
    """Tests for Runner class."""

    def test_initialization(self):
        """Test runner initialization."""
        config = RunnerConfig()
        runner = Runner(config)
        assert runner.config == config

    def test_from_env(self):
        """Test runner creation from environment."""
        runner = Runner.from_env()
        assert runner.config is not None

    def test_build_pytest_args_basic(self):
        """Test building basic pytest arguments."""
        config = RunnerConfig(
            parallel=False,
            verbose=True,
            coverage_enabled=False,
            junit_xml=False,
        )
        runner = Runner(config)
        args = runner._build_pytest_args()

        assert "-v" in args
        assert "-n" not in args  # No parallel flag

    def test_build_pytest_args_with_coverage(self):
        """Test building pytest args with coverage."""
        config = RunnerConfig(
            parallel=False,
            coverage_enabled=True,
            coverage_source=["alpha_zero"],
            junit_xml=False,
        )
        runner = Runner(config)
        args = runner._build_pytest_args(with_coverage=True)

        assert "--cov" in args
        assert "alpha_zero" in args

    def test_build_pytest_args_with_path(self):
        """Test building pytest args with specific path."""
        config = RunnerConfig(parallel=False, junit_xml=False)
        runner = Runner(config)
        args = runner._build_pytest_args(test_path="tests/unit")

        # Should contain the test path
        assert any("tests" in str(a) and "unit" in str(a) for a in args)

    def test_parse_pytest_output_passed(self):
        """Test parsing pytest output for passed tests."""
        config = RunnerConfig()
        runner = Runner(config)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "5 passed in 2.5s"
        mock_result.stderr = ""

        result = runner._parse_pytest_output(mock_result, 2.5, False)

        assert result.status == RunnerStatus.PASSED
        assert result.passed == 5

    def test_parse_pytest_output_failed(self):
        """Test parsing pytest output for failed tests."""
        config = RunnerConfig()
        runner = Runner(config)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "3 passed, 2 failed in 5.0s\nFAILED test_foo.py::test_bar"
        mock_result.stderr = ""

        result = runner._parse_pytest_output(mock_result, 5.0, False)

        assert result.status == RunnerStatus.FAILED
        assert result.passed == 3
        assert result.failed == 2
        assert "test_foo.py::test_bar" in result.failed_tests

    def test_parse_pytest_output_with_coverage(self):
        """Test parsing pytest output with coverage."""
        config = RunnerConfig()
        runner = Runner(config)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "5 passed in 2.5s\nTOTAL    100    20    80%"
        mock_result.stderr = ""

        result = runner._parse_pytest_output(mock_result, 2.5, True)

        assert result.coverage_percent == 80.0

    def test_save_results(self):
        """Test saving results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunnerConfig(
                project_root=Path(tmpdir),
                output_dir="results",
            )
            runner = Runner(config)

            result = RunnerResult(
                status=RunnerStatus.PASSED,
                total_tests=5,
                passed=5,
            )

            output_path = runner.save_results(result, "test_results.json")

            assert output_path.exists()
            with open(output_path) as f:
                data = json.load(f)
            assert data["status"] == "passed"
            assert data["total_tests"] == 5

    def test_verify_coverage_meets_threshold(self):
        """Test coverage verification when threshold met."""
        config = RunnerConfig(coverage_min_percent=80.0)
        runner = Runner(config)

        result = RunnerResult(
            status=RunnerStatus.PASSED,
            coverage_percent=85.0,
        )

        assert runner.verify_coverage(result) is True

    def test_verify_coverage_below_threshold(self):
        """Test coverage verification when below threshold."""
        config = RunnerConfig(coverage_min_percent=80.0)
        runner = Runner(config)

        result = RunnerResult(
            status=RunnerStatus.PASSED,
            coverage_percent=75.0,
        )

        assert runner.verify_coverage(result) is False

    def test_verify_coverage_no_data(self):
        """Test coverage verification with no coverage data."""
        config = RunnerConfig()
        runner = Runner(config)

        result = RunnerResult(status=RunnerStatus.PASSED)

        assert runner.verify_coverage(result) is False


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @patch("alpha_zero.utils.test_runner.PytestRunner")
    def test_run_tests_basic(self, mock_runner_class):
        """Test run_tests convenience function."""
        mock_runner = MagicMock()
        mock_runner.run_all.return_value = RunnerResult(status=RunnerStatus.PASSED)
        mock_runner_class.return_value = mock_runner

        result = run_tests()

        assert result.status == RunnerStatus.PASSED
        mock_runner.run_all.assert_called_once()

    @patch("alpha_zero.utils.test_runner.PytestRunner")
    def test_run_tests_with_path(self, mock_runner_class):
        """Test run_tests with specific path."""
        mock_runner = MagicMock()
        mock_runner.run_specific.return_value = RunnerResult(status=RunnerStatus.PASSED)
        mock_runner_class.return_value = mock_runner

        _result = run_tests(test_path="tests/unit")

        mock_runner.run_specific.assert_called_once_with("tests/unit", with_coverage=True)

    @patch("alpha_zero.utils.test_runner.PytestRunner")
    def test_quick_test_passes(self, mock_runner_class):
        """Test quick_test returns True on success."""
        mock_runner = MagicMock()
        mock_runner.run_unit_tests.return_value = RunnerResult(
            status=RunnerStatus.PASSED,
            failed=0,
            errors=0,
        )
        mock_runner_class.return_value = mock_runner

        assert quick_test() is True

    @patch("alpha_zero.utils.test_runner.PytestRunner")
    def test_quick_test_fails(self, mock_runner_class):
        """Test quick_test returns False on failure."""
        mock_runner = MagicMock()
        mock_runner.run_unit_tests.return_value = RunnerResult(
            status=RunnerStatus.FAILED,
            failed=1,
            errors=0,
        )
        mock_runner_class.return_value = mock_runner

        assert quick_test() is False
