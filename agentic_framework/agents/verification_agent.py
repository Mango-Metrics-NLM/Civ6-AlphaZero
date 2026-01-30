"""
Verification Agent for Agentic Framework

Runs linting, type checking, and other verification commands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..core.feedback_loop import FeedbackLoop, VerificationPhase, VerificationResult
from ..utils.logging_config import get_logger
from .base_agent import AgentContext, AgentResult, AgentStatus, BaseAgent

logger = get_logger("verification_agent")


@dataclass
class VerificationIssue:
    """An issue found during verification."""

    type: str  # lint, type, security
    severity: str  # error, warning, info
    message: str
    file: str | None = None
    line: int | None = None
    column: int | None = None
    rule: str | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "rule": self.rule,
            "suggestion": self.suggestion,
        }


class VerificationAgent(BaseAgent):
    """Agent that runs verification commands.

    Features:
    - Lint checking (ruff, flake8, pylint)
    - Type checking (mypy, pyright)
    - Security scanning
    - Custom verification commands
    - Issue aggregation and reporting
    """

    def __init__(
        self,
        timeout_seconds: float = 300.0,
        strict_mode: bool = False,
        auto_fix: bool = True,
    ) -> None:
        """Initialize the verification agent.

        Args:
            timeout_seconds: Timeout for verification commands.
            strict_mode: Treat warnings as errors.
            auto_fix: Automatically fix issues when possible.
        """
        super().__init__(
            name="verification",
            description="Runs linting and type checking",
            timeout_seconds=timeout_seconds,
        )
        self.strict_mode = strict_mode
        self.auto_fix = auto_fix
        self._issues: list[VerificationIssue] = []
        self._verification_results: list[VerificationResult] = []

    def _execute(self, context: AgentContext) -> AgentResult:
        """Execute verification commands.

        Args:
            context: Execution context.

        Returns:
            AgentResult with verification outcome.
        """
        self.report_progress(0.0, "Starting verification")
        self._issues.clear()
        self._verification_results.clear()

        # Create feedback loop from config
        feedback_loop = FeedbackLoop(
            config=context.config,
            command_runner=context.command_runner,
        )

        # Run all verification commands
        self.report_progress(0.2, "Running lint checks")
        results = feedback_loop.run_all_verifications(
            cwd=context.working_dir,
            stop_on_failure=False,
        )
        self._verification_results = results

        # Parse issues from results
        self.report_progress(0.6, "Analyzing results")
        for result in results:
            issues = self._parse_verification_output(result)
            self._issues.extend(issues)

        # Auto-fix if enabled
        fixed_count = 0
        if self.auto_fix and self._issues:
            self.report_progress(0.8, "Attempting auto-fix")
            fixed_count = self._attempt_auto_fix(context)

        # Determine success
        error_count = sum(
            1
            for issue in self._issues
            if issue.severity == "error" or (self.strict_mode and issue.severity == "warning")
        )
        remaining_errors = max(error_count - fixed_count, 0)

        success = remaining_errors == 0

        self.report_progress(1.0, "Verification complete")

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.COMPLETED if success else AgentStatus.FAILED,
            success=success,
            message=f"Found {len(self._issues)} issues, {fixed_count} fixed, {remaining_errors} remaining",
            data={
                "issues": [i.to_dict() for i in self._issues],
                "verification_results": [{"name": r.name, "success": r.success} for r in results],
                "fixed_count": fixed_count,
                "error_count": error_count,
            },
            errors=[
                f"{i.file}:{i.line}: {i.message}" for i in self._issues if i.severity == "error"
            ][:10],  # Limit to first 10
            next_agent="documentation" if success else None,
        )

    def _parse_verification_output(self, result: VerificationResult) -> list[VerificationIssue]:
        """Parse verification output for issues.

        Args:
            result: Verification result.

        Returns:
            List of issues found.
        """
        issues: list[VerificationIssue] = []
        combined_output = f"{result.output}\n{result.error_output}"

        if result.phase == VerificationPhase.LINT:
            issues.extend(self._parse_lint_output(combined_output))
        elif result.phase == VerificationPhase.TYPE_CHECK:
            issues.extend(self._parse_type_output(combined_output))
        else:
            # Generic error parsing
            issues.extend(self._parse_generic_errors(combined_output, result.name))

        return issues

    def _parse_lint_output(self, output: str) -> list[VerificationIssue]:
        """Parse lint tool output.

        Args:
            output: Lint output.

        Returns:
            List of issues.
        """
        issues: list[VerificationIssue] = []

        # Ruff/flake8 format: path/to/file.py:line:col: code message
        pattern = re.compile(r"^(.+?):(\d+):(\d+):\s*(\w+)\s+(.+?)$", re.MULTILINE)

        for match in pattern.finditer(output):
            file_path = match.group(1)
            line = int(match.group(2))
            col = int(match.group(3))
            rule = match.group(4)
            message = match.group(5)

            # Determine severity from rule code
            severity = "warning"
            if rule.startswith("E") or rule.startswith("F"):
                severity = "error"

            issues.append(
                VerificationIssue(
                    type="lint",
                    severity=severity,
                    message=message,
                    file=file_path,
                    line=line,
                    column=col,
                    rule=rule,
                )
            )

        return issues

    def _parse_type_output(self, output: str) -> list[VerificationIssue]:
        """Parse type checker output.

        Args:
            output: Type checker output.

        Returns:
            List of issues.
        """
        issues: list[VerificationIssue] = []

        # Mypy format: path/to/file.py:line(:column)?: error: message
        pattern = re.compile(
            r"^(.+?):(\d+)(?::\d+)?:\s*(error|warning|note):\s*(.+?)$", re.MULTILINE
        )

        for match in pattern.finditer(output):
            file_path = match.group(1)
            line = int(match.group(2))
            severity = match.group(3)
            message = match.group(4)

            if severity == "note":
                severity = "info"

            issues.append(
                VerificationIssue(
                    type="type",
                    severity=severity,
                    message=message,
                    file=file_path,
                    line=line,
                )
            )

        return issues

    def _parse_generic_errors(self, output: str, source: str) -> list[VerificationIssue]:
        """Parse generic error patterns.

        Args:
            output: Command output.
            source: Source of the output.

        Returns:
            List of issues.
        """
        issues: list[VerificationIssue] = []

        # Generic error pattern
        error_pattern = re.compile(r"(?:error|ERROR|Error):\s*(.+?)(?:\n|$)", re.MULTILINE)

        for match in error_pattern.finditer(output):
            issues.append(
                VerificationIssue(
                    type=source,
                    severity="error",
                    message=match.group(1).strip(),
                )
            )

        return issues

    def _attempt_auto_fix(self, context: AgentContext) -> int:
        """Attempt to auto-fix issues.

        Args:
            context: Execution context.

        Returns:
            Number of issues fixed.
        """
        fixed_count = 0

        # Try running ruff with --fix
        success, stdout, stderr = self.run_command(
            "ruff check --fix .",
            cwd=context.working_dir,
        )

        if success:
            # Count fixes from output
            fix_match = re.search(r"(\d+)\s+(?:fix|Fixed)", stdout + stderr)
            if fix_match:
                fixed_count = int(fix_match.group(1))

        logger.info(f"Auto-fixed {fixed_count} issues")
        return fixed_count

    def run_lint(self, context: AgentContext, tool: str = "ruff") -> VerificationResult:
        """Run a specific lint tool.

        Args:
            context: Execution context.
            tool: Lint tool to run.

        Returns:
            Verification result.
        """
        commands = {
            "ruff": "ruff check .",
            "flake8": "flake8 .",
            "pylint": "pylint **/*.py",
        }

        command = commands.get(tool, f"{tool} .")

        success, stdout, stderr = self.run_command(command, cwd=context.working_dir)

        return VerificationResult(
            phase=VerificationPhase.LINT,
            name=tool,
            command=command,
            success=success,
            output=stdout,
            error_output=stderr,
            duration_seconds=0,
        )

    def run_type_check(self, context: AgentContext, tool: str = "mypy") -> VerificationResult:
        """Run type checking.

        Args:
            context: Execution context.
            tool: Type checker to run.

        Returns:
            Verification result.
        """
        commands = {
            "mypy": "mypy . --ignore-missing-imports",
            "pyright": "pyright .",
        }

        command = commands.get(tool, f"{tool} .")

        success, stdout, stderr = self.run_command(command, cwd=context.working_dir)

        return VerificationResult(
            phase=VerificationPhase.TYPE_CHECK,
            name=tool,
            command=command,
            success=success,
            output=stdout,
            error_output=stderr,
            duration_seconds=0,
        )

    def get_issues_summary(self) -> str:
        """Get a summary of verification issues.

        Returns:
            Markdown summary.
        """
        if not self._issues:
            return "No issues found"

        error_count = sum(1 for i in self._issues if i.severity == "error")
        warning_count = sum(1 for i in self._issues if i.severity == "warning")

        lines = [
            "## Verification Issues",
            "",
            f"**Errors:** {error_count}",
            f"**Warnings:** {warning_count}",
            "",
        ]

        # Group by file
        by_file: dict[str, list[VerificationIssue]] = {}
        for issue in self._issues:
            file_key = issue.file or "unknown"
            if file_key not in by_file:
                by_file[file_key] = []
            by_file[file_key].append(issue)

        for file_path, file_issues in sorted(by_file.items()):
            lines.append(f"### {file_path}")
            for issue in file_issues[:5]:  # Limit per file
                marker = "🔴" if issue.severity == "error" else "🟡"
                loc = f":{issue.line}" if issue.line else ""
                lines.append(f"- {marker} {loc} [{issue.rule or issue.type}] {issue.message}")
            if len(file_issues) > 5:
                lines.append(f"  ... and {len(file_issues) - 5} more")
            lines.append("")

        return "\n".join(lines)

    @property
    def issues(self) -> list[VerificationIssue]:
        """Get all verification issues."""
        return self._issues

    @property
    def verification_results(self) -> list[VerificationResult]:
        """Get all verification results."""
        return self._verification_results
