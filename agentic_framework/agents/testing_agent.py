"""
Testing Agent for Agentic Framework

Creates tests, runs test suites, and analyzes coverage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from ..utils.logging_config import get_logger

logger = get_logger("testing_agent")


@dataclass
class TestCase:
    """Represents a test case."""

    name: str
    description: str
    test_type: str = "unit"  # unit, integration, e2e
    target_function: Optional[str] = None
    target_file: Optional[str] = None
    code: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type,
            "target_function": self.target_function,
            "target_file": self.target_file,
        }


@dataclass
class TestResult:
    """Result of running a test."""

    name: str
    passed: bool
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    output: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
        }


@dataclass
class CoverageReport:
    """Test coverage report."""

    total_coverage: float = 0.0
    file_coverage: dict[str, float] = field(default_factory=dict)
    uncovered_lines: dict[str, list[int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_coverage": self.total_coverage,
            "file_coverage": self.file_coverage,
            "uncovered_lines": self.uncovered_lines,
        }


class TestingAgent(BaseAgent):
    """Agent that handles testing tasks.

    Features:
    - Test generation from implementation
    - Test suite execution
    - Coverage analysis
    - Test result reporting
    """

    def __init__(
        self,
        timeout_seconds: float = 600.0,
        coverage_threshold: float = 80.0,
        test_framework: str = "pytest",
    ) -> None:
        """Initialize the testing agent.

        Args:
            timeout_seconds: Timeout for test execution.
            coverage_threshold: Minimum coverage percentage.
            test_framework: Test framework to use.
        """
        super().__init__(
            name="testing",
            description="Creates and runs tests",
            timeout_seconds=timeout_seconds,
        )
        self.coverage_threshold = coverage_threshold
        self.test_framework = test_framework
        self._test_results: list[TestResult] = []
        self._coverage_report: Optional[CoverageReport] = None

    def _execute(self, context: AgentContext) -> AgentResult:
        """Execute testing tasks.

        Args:
            context: Execution context.

        Returns:
            AgentResult with testing outcome.
        """
        self.report_progress(0.0, "Starting tests")
        self._test_results.clear()

        # Discover existing tests
        self.report_progress(0.1, "Discovering tests")
        self._discover_tests(context.working_dir)

        # Run existing tests
        self.report_progress(0.3, "Running tests")
        run_result = self._run_tests(context)

        if not run_result["success"]:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                success=False,
                message=f"Tests failed: {run_result['failed']} failures",
                data={
                    "test_results": [r.to_dict() for r in self._test_results],
                    "passed": run_result["passed"],
                    "failed": run_result["failed"],
                },
                errors=run_result.get("errors", []),
                next_agent=None,
            )

        # Get coverage if available
        self.report_progress(0.7, "Analyzing coverage")
        self._coverage_report = self._get_coverage(context)

        # Check coverage threshold
        coverage_met = True
        if self._coverage_report:
            if self._coverage_report.total_coverage < self.coverage_threshold:
                coverage_met = False
                logger.warning(
                    f"Coverage {self._coverage_report.total_coverage:.1f}% "
                    f"below threshold {self.coverage_threshold}%"
                )

        self.report_progress(1.0, "Testing complete")

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.COMPLETED,
            success=True,
            message=f"Tests passed: {run_result['passed']} passed, {run_result['failed']} failed",
            data={
                "test_results": [r.to_dict() for r in self._test_results],
                "passed": run_result["passed"],
                "failed": run_result["failed"],
                "coverage": self._coverage_report.to_dict() if self._coverage_report else None,
                "coverage_met": coverage_met,
            },
            warnings=[] if coverage_met else [
                f"Coverage {self._coverage_report.total_coverage:.1f}% below threshold"
            ],
            next_agent="verification",
        )

    def _discover_tests(self, working_dir: Path) -> list[Path]:
        """Discover test files in the project.

        Args:
            working_dir: Directory to search.

        Returns:
            List of test file paths.
        """
        test_files: list[Path] = []

        # Look for common test patterns
        patterns = ["test_*.py", "*_test.py", "tests/**/*.py"]

        for pattern in patterns:
            test_files.extend(working_dir.glob(pattern))

        # Remove duplicates and sort
        test_files = sorted(set(test_files))
        logger.info(f"Discovered {len(test_files)} test files")

        return test_files

    def _run_tests(self, context: AgentContext) -> dict[str, Any]:
        """Run the test suite.

        Args:
            context: Execution context.

        Returns:
            Dictionary with test results.
        """
        result = {
            "success": True,
            "passed": 0,
            "failed": 0,
            "errors": [],
        }

        # Get test commands from config verification commands
        test_commands = [
            cmd for cmd in context.config.feedback.verification_commands
            if "test" in cmd.name.lower()
        ]
        
        if not test_commands:
            # Fallback to basic pytest command if no test commands in config
            logger.warning("No test commands found in config, using default pytest")
            if self.test_framework == "pytest":
                command = "python -m pytest -v --tb=short"
            else:
                command = f"python -m {self.test_framework}"
            
            success, stdout, stderr = self.run_command(command, timeout=self.timeout_seconds)
            if self.test_framework == "pytest":
                result = self._parse_pytest_output(stdout + stderr)
        else:
            # Run all test commands from config
            all_success = True
            all_output = []
            
            for test_cmd in test_commands:
                logger.info(f"Running test command: {test_cmd.name}")
                success, stdout, stderr = self.run_command(
                    test_cmd.command, 
                    timeout=test_cmd.timeout_seconds
                )
                all_output.append(stdout + stderr)
                if not success:
                    all_success = False
            
            # Parse combined output if using pytest
            if self.test_framework == "pytest":
                result = self._parse_pytest_output("\n".join(all_output))
            
            result["success"] = all_success

        return result

    def _parse_pytest_output(self, output: str) -> dict[str, Any]:
        """Parse pytest output to extract results.

        Args:
            output: Pytest output.

        Returns:
            Parsed results dictionary.
        """
        result = {
            "success": True,
            "passed": 0,
            "failed": 0,
            "errors": [],
        }

        # Count passed/failed
        passed_match = re.search(r"(\d+)\s+passed", output)
        failed_match = re.search(r"(\d+)\s+failed", output)
        error_match = re.search(r"(\d+)\s+error", output)

        if passed_match:
            result["passed"] = int(passed_match.group(1))
        if failed_match:
            result["failed"] = int(failed_match.group(1))
            result["success"] = False
        if error_match:
            result["errors"].append(f"{error_match.group(1)} errors")
            result["success"] = False

        # Extract individual test results
        test_pattern = re.compile(
            r"(test_\w+|[\w/]+::test_\w+)\s+(PASSED|FAILED|ERROR)",
            re.MULTILINE
        )
        for match in test_pattern.finditer(output):
            test_name = match.group(1)
            status = match.group(2)
            self._test_results.append(
                TestResult(
                    name=test_name,
                    passed=status == "PASSED",
                )
            )

        # Extract failure messages
        failure_pattern = re.compile(
            r"FAILED\s+(.+?)\s+-\s+(.+?)(?:\n|$)",
            re.MULTILINE
        )
        for match in failure_pattern.finditer(output):
            result["errors"].append(f"{match.group(1)}: {match.group(2)}")

        return result

    def _get_coverage(self, context: AgentContext) -> Optional[CoverageReport]:
        """Get test coverage report.

        Args:
            context: Execution context.

        Returns:
            Coverage report or None.
        """
        # Try to run coverage
        success, stdout, stderr = self.run_command(
            "python -m pytest --cov --cov-report=term-missing -q",
            timeout=self.timeout_seconds,
        )

        if not success:
            return None

        report = CoverageReport()
        output = stdout + stderr

        # Parse total coverage
        total_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
        if total_match:
            report.total_coverage = float(total_match.group(1))

        # Parse per-file coverage
        file_pattern = re.compile(
            r"^(\S+\.py)\s+\d+\s+\d+\s+(\d+)%(?:\s+(.*))?$",
            re.MULTILINE
        )
        for match in file_pattern.finditer(output):
            filename = match.group(1)
            coverage = float(match.group(2))
            missing = match.group(3)

            report.file_coverage[filename] = coverage

            if missing:
                # Parse missing lines
                lines = []
                for part in missing.split(","):
                    part = part.strip()
                    if "-" in part:
                        start, end = part.split("-")
                        lines.extend(range(int(start), int(end) + 1))
                    elif part.isdigit():
                        lines.append(int(part))
                report.uncovered_lines[filename] = lines

        return report

    def generate_test(
        self,
        function_name: str,
        file_path: Path,
        context: AgentContext,
    ) -> TestCase:
        """Generate a test case for a function.

        Args:
            function_name: Name of the function to test.
            file_path: Path to the file containing the function.
            context: Execution context.

        Returns:
            Generated test case.
        """
        # Read the file to understand the function
        content = file_path.read_text() if file_path.exists() else ""

        # Extract function signature
        func_pattern = re.compile(
            rf"def\s+{function_name}\s*\(([^)]*)\)(?:\s*->\s*(\S+))?:",
            re.MULTILINE
        )
        match = func_pattern.search(content)

        if match:

        # Generate basic test structure
        test_name = f"test_{function_name}"
        module_name = file_path.stem

        test_code = f'''def {test_name}():
    """Test {function_name}."""
    # TODO: Implement test
    from {module_name} import {function_name}

    # Arrange
    # ... setup test data

    # Act
    result = {function_name}()

    # Assert
    assert result is not None
'''

        return TestCase(
            name=test_name,
            description=f"Test for {function_name}",
            test_type="unit",
            target_function=function_name,
            target_file=str(file_path),
            code=test_code,
        )

    def get_test_summary(self) -> str:
        """Get a summary of test results.

        Returns:
            Markdown summary.
        """
        if not self._test_results:
            return "No test results available"

        passed = sum(1 for r in self._test_results if r.passed)
        failed = len(self._test_results) - passed

        lines = [
            "## Test Results",
            "",
            f"**Passed:** {passed}",
            f"**Failed:** {failed}",
            "",
        ]

        if self._coverage_report:
            lines.extend([
                "### Coverage",
                f"**Total:** {self._coverage_report.total_coverage:.1f}%",
                "",
            ])

        if failed > 0:
            lines.append("### Failed Tests")
            for result in self._test_results:
                if not result.passed:
                    lines.append(f"- `{result.name}`: {result.error_message or 'Failed'}")

        return "\n".join(lines)

    @property
    def test_results(self) -> list[TestResult]:
        """Get test results."""
        return self._test_results

    @property
    def coverage_report(self) -> Optional[CoverageReport]:
        """Get coverage report."""
        return self._coverage_report
