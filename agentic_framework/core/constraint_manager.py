"""
Constraint Manager for Agentic Framework

Manages hard constraints (violations = failure), soft constraints (preferences),
and anti-constraints (explicit permissions). Provides validation and checking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import AgenticConfig
from ..utils.logging_config import get_logger
from ..utils.command_runner import CommandRunner

logger = get_logger("constraint_manager")


class ConstraintType(Enum):
    """Types of constraints."""

    LANGUAGE = "language"
    DEPENDENCY = "dependency"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"
    STYLE = "style"
    ARCHITECTURE = "architecture"
    PERFORMANCE = "performance"
    TESTING = "testing"
    GENERAL = "general"


class ConstraintSeverity(Enum):
    """Severity levels for constraints."""

    HARD = "hard"  # Violations are failures
    SOFT = "soft"  # Preferences, can be relaxed
    ANTI = "anti"  # Explicit permissions


@dataclass
class Constraint:
    """Represents a single constraint."""

    name: str
    type: ConstraintType
    severity: ConstraintSeverity
    value: str
    validation_command: Optional[str] = None
    priority: int = 5  # 1-10, only for soft constraints
    scope: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any], severity: ConstraintSeverity) -> Constraint:
        """Create a Constraint from a dictionary."""
        type_str = data.get("type", "general").lower()
        try:
            constraint_type = ConstraintType(type_str)
        except ValueError:
            constraint_type = ConstraintType.GENERAL

        return cls(
            name=data.get("name", "unnamed"),
            type=constraint_type,
            severity=severity,
            value=data.get("value", ""),
            validation_command=data.get("validation_command"),
            priority=data.get("priority", 5),
            scope=data.get("scope"),
            description=data.get("description"),
        )


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""

    constraint: Constraint
    message: str
    details: Optional[str] = None
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    suggested_fix: Optional[str] = None

    def __str__(self) -> str:
        """String representation of the violation."""
        location = ""
        if self.file_path:
            location = f" at {self.file_path}"
            if self.line_number:
                location += f":{self.line_number}"

        return f"[{self.constraint.severity.value.upper()}] {self.constraint.name}{location}: {self.message}"


@dataclass
class ConstraintCheckResult:
    """Result of checking constraints."""

    passed: bool
    violations: list[ConstraintViolation] = field(default_factory=list)
    warnings: list[ConstraintViolation] = field(default_factory=list)
    permissions: list[Constraint] = field(default_factory=list)
    checked_count: int = 0
    duration_seconds: float = 0.0

    @property
    def hard_violations(self) -> list[ConstraintViolation]:
        """Get only hard constraint violations."""
        return [v for v in self.violations if v.constraint.severity == ConstraintSeverity.HARD]

    @property
    def soft_violations(self) -> list[ConstraintViolation]:
        """Get only soft constraint violations."""
        return [v for v in self.violations if v.constraint.severity == ConstraintSeverity.SOFT]

    def summary(self) -> str:
        """Generate a summary of the check results."""
        lines = [
            f"Constraint Check: {'PASSED' if self.passed else 'FAILED'}",
            f"Checked: {self.checked_count} constraints",
            f"Hard violations: {len(self.hard_violations)}",
            f"Soft violations: {len(self.soft_violations)}",
            f"Warnings: {len(self.warnings)}",
            f"Permissions: {len(self.permissions)}",
        ]
        return "\n".join(lines)


# Type alias for custom constraint validators
ConstraintValidator = Callable[[Constraint, Path], Optional[ConstraintViolation]]


class ConstraintManager:
    """Manages and validates constraints for the agentic framework.

    Features:
    - Load constraints from configuration
    - Register custom validators
    - Check constraints against files/directories
    - Generate violation reports
    """

    def __init__(
        self,
        config: Optional[AgenticConfig] = None,
        command_runner: Optional[CommandRunner] = None,
    ) -> None:
        """Initialize the constraint manager.

        Args:
            config: Configuration containing constraint definitions.
            command_runner: Command runner for validation commands.
        """
        self.config = config
        self.command_runner = command_runner or CommandRunner()
        self._constraints: list[Constraint] = []
        self._custom_validators: dict[str, ConstraintValidator] = {}
        self._builtin_validators: dict[ConstraintType, ConstraintValidator] = {}

        # Register built-in validators
        self._register_builtin_validators()

        # Load constraints from config if provided
        if config:
            self.load_from_config(config)

    def _register_builtin_validators(self) -> None:
        """Register built-in constraint validators."""
        self._builtin_validators = {
            ConstraintType.LANGUAGE: self._validate_language_constraint,
            ConstraintType.SECURITY: self._validate_security_constraint,
            ConstraintType.STYLE: self._validate_style_constraint,
            ConstraintType.TESTING: self._validate_testing_constraint,
        }

    def load_from_config(self, config: AgenticConfig) -> None:
        """Load constraints from configuration.

        Args:
            config: Configuration containing constraint definitions.
        """
        self._constraints.clear()

        # Load hard constraints
        for constraint_data in config.constraints.hard:
            constraint = Constraint.from_dict(constraint_data, ConstraintSeverity.HARD)
            self._constraints.append(constraint)
            logger.debug(f"Loaded hard constraint: {constraint.name}")

        # Load soft constraints
        for constraint_data in config.constraints.soft:
            constraint = Constraint.from_dict(constraint_data, ConstraintSeverity.SOFT)
            self._constraints.append(constraint)
            logger.debug(f"Loaded soft constraint: {constraint.name}")

        # Load anti-constraints (permissions)
        for constraint_data in config.constraints.anti:
            constraint = Constraint(
                name=constraint_data.get("permission", ""),
                type=ConstraintType.GENERAL,
                severity=ConstraintSeverity.ANTI,
                value=constraint_data.get("permission", ""),
                scope=constraint_data.get("scope"),
            )
            self._constraints.append(constraint)
            logger.debug(f"Loaded anti-constraint: {constraint.name}")

        logger.info(f"Loaded {len(self._constraints)} constraints from configuration")

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the manager.

        Args:
            constraint: Constraint to add.
        """
        self._constraints.append(constraint)
        logger.debug(f"Added constraint: {constraint.name}")

    def remove_constraint(self, name: str) -> bool:
        """Remove a constraint by name.

        Args:
            name: Name of the constraint to remove.

        Returns:
            True if constraint was found and removed.
        """
        for i, c in enumerate(self._constraints):
            if c.name == name:
                del self._constraints[i]
                logger.debug(f"Removed constraint: {name}")
                return True
        return False

    def register_validator(self, name: str, validator: ConstraintValidator) -> None:
        """Register a custom constraint validator.

        Args:
            name: Name to identify the validator.
            validator: Validation function.
        """
        self._custom_validators[name] = validator
        logger.debug(f"Registered custom validator: {name}")

    def get_constraints(
        self,
        severity: Optional[ConstraintSeverity] = None,
        constraint_type: Optional[ConstraintType] = None,
        enabled_only: bool = True,
    ) -> list[Constraint]:
        """Get constraints matching criteria.

        Args:
            severity: Filter by severity.
            constraint_type: Filter by type.
            enabled_only: Only return enabled constraints.

        Returns:
            List of matching constraints.
        """
        result = self._constraints

        if enabled_only:
            result = [c for c in result if c.enabled]

        if severity is not None:
            result = [c for c in result if c.severity == severity]

        if constraint_type is not None:
            result = [c for c in result if c.type == constraint_type]

        return result

    def get_permissions(self) -> list[Constraint]:
        """Get all anti-constraints (permissions).

        Returns:
            List of permission constraints.
        """
        return self.get_constraints(severity=ConstraintSeverity.ANTI)

    def check_all(
        self,
        target_path: Path,
        stop_on_hard_violation: bool = True,
    ) -> ConstraintCheckResult:
        """Check all constraints against a target.

        Args:
            target_path: File or directory to check.
            stop_on_hard_violation: Stop checking after first hard violation.

        Returns:
            ConstraintCheckResult with all findings.
        """
        import time

        start_time = time.time()
        result = ConstraintCheckResult(passed=True)

        hard_constraints = self.get_constraints(severity=ConstraintSeverity.HARD)
        soft_constraints = self.get_constraints(severity=ConstraintSeverity.SOFT)

        logger.info(f"Checking {len(hard_constraints)} hard and {len(soft_constraints)} soft constraints")

        # Check hard constraints first
        for constraint in hard_constraints:
            result.checked_count += 1
            violation = self._check_constraint(constraint, target_path)

            if violation:
                result.violations.append(violation)
                result.passed = False
                logger.warning(f"Hard constraint violated: {constraint.name}")

                if stop_on_hard_violation:
                    break

        # Check soft constraints
        if result.passed or not stop_on_hard_violation:
            for constraint in soft_constraints:
                result.checked_count += 1
                violation = self._check_constraint(constraint, target_path)

                if violation:
                    if constraint.priority >= 7:
                        result.violations.append(violation)
                    else:
                        result.warnings.append(violation)
                    logger.info(f"Soft constraint violated: {constraint.name}")

        # Add permissions to result
        result.permissions = self.get_permissions()

        result.duration_seconds = time.time() - start_time
        logger.info(f"Constraint check completed in {result.duration_seconds:.2f}s: {result.summary()}")

        return result

    def _check_constraint(
        self, constraint: Constraint, target_path: Path
    ) -> Optional[ConstraintViolation]:
        """Check a single constraint.

        Args:
            constraint: Constraint to check.
            target_path: Target to check against.

        Returns:
            ConstraintViolation if violated, None otherwise.
        """
        # Try custom validator first
        if constraint.name in self._custom_validators:
            return self._custom_validators[constraint.name](constraint, target_path)

        # Try built-in validator for type
        if constraint.type in self._builtin_validators:
            return self._builtin_validators[constraint.type](constraint, target_path)

        # Fall back to command-based validation
        if constraint.validation_command:
            return self._validate_with_command(constraint, target_path)

        # No validator available - assume passed
        logger.debug(f"No validator for constraint {constraint.name}, assuming passed")
        return None

    def _validate_with_command(
        self, constraint: Constraint, target_path: Path
    ) -> Optional[ConstraintViolation]:
        """Validate a constraint using its validation command.

        Args:
            constraint: Constraint with validation command.
            target_path: Target to validate.

        Returns:
            ConstraintViolation if command indicates failure.
        """
        if not constraint.validation_command:
            return None

        # Substitute {path} placeholder
        command = constraint.validation_command.replace("{path}", str(target_path))

        result = self.command_runner.run(command, cwd=target_path.parent if target_path.is_file() else target_path)

        if not result.success:
            return ConstraintViolation(
                constraint=constraint,
                message=f"Validation command failed: {constraint.validation_command}",
                details=result.stderr or result.stdout,
            )

        return None

    def _validate_language_constraint(
        self, constraint: Constraint, target_path: Path
    ) -> Optional[ConstraintViolation]:
        """Validate a language constraint."""
        # Parse version requirement from value
        value = constraint.value.lower()

        if "python" in constraint.name.lower():
            # Check Python version
            result = self.command_runner.run("python --version")
            if result.success:
                version_match = re.search(r"(\d+\.\d+)", result.stdout)
                if version_match:
                    current_version = version_match.group(1)
                    required_version = re.search(r"(\d+\.\d+)", value)

                    if required_version:
                        req = required_version.group(1)
                        if ">=" in value:
                            if float(current_version) < float(req):
                                return ConstraintViolation(
                                    constraint=constraint,
                                    message=f"Python version {current_version} does not meet requirement {value}",
                                )

        return None

    def _validate_security_constraint(
        self, constraint: Constraint, target_path: Path
    ) -> Optional[ConstraintViolation]:
        """Validate a security constraint."""
        value_lower = constraint.value.lower()

        # Check for hardcoded secrets
        if "hardcoded" in value_lower and ("secret" in value_lower or "password" in value_lower or "key" in value_lower):
            patterns = [
                r"password\s*=\s*['\"][^'\"]+['\"]",
                r"api_key\s*=\s*['\"][^'\"]+['\"]",
                r"secret\s*=\s*['\"][^'\"]+['\"]",
                r"token\s*=\s*['\"][^'\"]+['\"]",
            ]

            if target_path.is_file():
                files = [target_path]
            else:
                files = list(target_path.rglob("*.py"))

            for file_path in files:
                try:
                    content = file_path.read_text()
                    for pattern in patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            # Skip if it looks like a placeholder
                            if any(
                                placeholder in match.group()
                                for placeholder in ["${", "{{", "os.environ", "getenv"]
                            ):
                                continue

                            return ConstraintViolation(
                                constraint=constraint,
                                message="Potential hardcoded secret detected",
                                file_path=file_path,
                                details=f"Found: {match.group()[:50]}...",
                                suggested_fix="Use environment variables or a secrets manager",
                            )
                except Exception as e:
                    logger.debug(f"Error reading {file_path}: {e}")

        return None

    def _validate_style_constraint(
        self, constraint: Constraint, target_path: Path
    ) -> Optional[ConstraintViolation]:
        """Validate a style constraint."""
        value_lower = constraint.value.lower()

        # Type hints check
        if "type hint" in value_lower:
            if target_path.is_file():
                files = [target_path]
            else:
                files = list(target_path.rglob("*.py"))

            missing_hints = []
            for file_path in files:
                try:
                    content = file_path.read_text()
                    # Simple heuristic: find function definitions without return type hints
                    func_pattern = re.compile(r"def\s+(\w+)\s*\([^)]*\)\s*(?:->.*?)?:")
                    for match in func_pattern.finditer(content):
                        if "->" not in match.group():
                            missing_hints.append((file_path, match.group(1)))
                except Exception:
                    pass

            if missing_hints and len(missing_hints) > len(files) * 0.2:  # More than 20% missing
                return ConstraintViolation(
                    constraint=constraint,
                    message=f"Many functions missing type hints ({len(missing_hints)} found)",
                    details=f"Examples: {missing_hints[:3]}",
                )

        return None

    def _validate_testing_constraint(
        self, constraint: Constraint, target_path: Path
    ) -> Optional[ConstraintViolation]:
        """Validate a testing constraint."""
        value_lower = constraint.value.lower()

        # Coverage check
        if "coverage" in value_lower:
            coverage_match = re.search(r"(\d+)%", constraint.value)
            if coverage_match:
                required_coverage = int(coverage_match.group(1))

                # Try to run coverage report
                result = self.command_runner.run(
                    "python -m pytest --cov --cov-report=term-missing -q 2>/dev/null | tail -5",
                    timeout_seconds=120,
                )

                if result.success:
                    # Parse coverage percentage
                    cov_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", result.stdout)
                    if cov_match:
                        actual_coverage = int(cov_match.group(1))
                        if actual_coverage < required_coverage:
                            return ConstraintViolation(
                                constraint=constraint,
                                message=f"Test coverage {actual_coverage}% below required {required_coverage}%",
                                suggested_fix="Add more tests to increase coverage",
                            )

        return None

    def is_permitted(self, action: str) -> bool:
        """Check if an action is explicitly permitted.

        Args:
            action: Description of the action to check.

        Returns:
            True if action matches an anti-constraint (permission).
        """
        action_lower = action.lower()
        for permission in self.get_permissions():
            if permission.value.lower() in action_lower or action_lower in permission.value.lower():
                return True
        return False
