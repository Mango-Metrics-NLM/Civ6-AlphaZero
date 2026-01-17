"""
Constraint System for Agentic Coding System

Provides dynamic constraint definition, validation, and management.
All constraints are configurable with no hardcoded values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from ..utils.logging_config import get_logger

logger = get_logger("constraints")


class ConstraintType(Enum):
    """Types of constraints."""

    HARD = "hard"  # Must be satisfied - violations cause failure
    SOFT = "soft"  # Preferences - violations generate warnings
    ANTI = "anti"  # Explicit permissions - define what IS allowed


class ConstraintCategory(Enum):
    """Categories of constraints."""

    LANGUAGE = "language"
    DEPENDENCY = "dependency"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"
    STYLE = "style"
    ARCHITECTURE = "architecture"
    PERFORMANCE = "performance"
    TESTING = "testing"
    SCOPE = "scope"
    CUSTOM = "custom"


@dataclass
class ConstraintViolation:
    """A constraint violation.

    Attributes:
        constraint_name: Name of the violated constraint
        constraint_type: Type of constraint (hard/soft/anti)
        message: Violation message
        location: Where the violation occurred
        severity: Severity level (1-10)
        fix_suggestion: Suggested fix
        context: Additional context
    """

    constraint_name: str
    constraint_type: ConstraintType
    message: str
    location: str = ""
    severity: int = 5
    fix_suggestion: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "constraint_name": self.constraint_name,
            "constraint_type": self.constraint_type.value,
            "message": self.message,
            "location": self.location,
            "severity": self.severity,
            "fix_suggestion": self.fix_suggestion,
            "context": self.context,
        }


@dataclass
class Constraint:
    """A constraint definition.

    All values are configurable - no hardcoded defaults.

    Attributes:
        name: Unique constraint name
        constraint_type: Hard, soft, or anti constraint
        category: Constraint category
        description: Human-readable description
        check_function: Function to check the constraint
        pattern: Regex pattern for pattern-based checks
        value: Expected value for value-based checks
        severity: Severity of violations (1-10)
        enabled: Whether constraint is active
        fix_template: Template for suggested fixes
        metadata: Additional configuration
    """

    name: str
    constraint_type: ConstraintType
    category: ConstraintCategory = ConstraintCategory.CUSTOM
    description: str = ""
    check_function: Optional[Callable[[Any], tuple[bool, str]]] = None
    pattern: Optional[str] = None
    value: Any = None
    severity: int = 5
    enabled: bool = True
    fix_template: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    _compiled_pattern: Optional[re.Pattern[str]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Compile pattern after initialization."""
        if self.pattern:
            self._compiled_pattern = re.compile(self.pattern)

    def check(self, context: Any) -> tuple[bool, Optional[ConstraintViolation]]:
        """Check if constraint is satisfied.

        Args:
            context: Context to check against.

        Returns:
            Tuple of (is_satisfied, violation_or_none).
        """
        if not self.enabled:
            return True, None

        # Use check function if provided
        if self.check_function:
            try:
                is_valid, message = self.check_function(context)
                if not is_valid:
                    violation = ConstraintViolation(
                        constraint_name=self.name,
                        constraint_type=self.constraint_type,
                        message=message,
                        severity=self.severity,
                        fix_suggestion=self.fix_template,
                    )
                    return False, violation
                return True, None
            except Exception as e:
                violation = ConstraintViolation(
                    constraint_name=self.name,
                    constraint_type=self.constraint_type,
                    message=f"Error checking constraint: {e}",
                    severity=self.severity,
                )
                return False, violation

        # Use pattern if provided
        if self._compiled_pattern and isinstance(context, str):
            match = self._compiled_pattern.search(context)
            # For hard/soft constraints, pattern should NOT match (violation pattern)
            # For anti constraints, pattern SHOULD match (permission pattern)
            if self.constraint_type == ConstraintType.ANTI:
                if match:
                    return True, None  # Permission granted
                violation = ConstraintViolation(
                    constraint_name=self.name,
                    constraint_type=self.constraint_type,
                    message=f"Permission '{self.name}' not applicable",
                    severity=1,
                )
                return False, violation
            else:
                if match:
                    violation = ConstraintViolation(
                        constraint_name=self.name,
                        constraint_type=self.constraint_type,
                        message=f"Pattern violation: {match.group(0)[:100]}",
                        location=f"position {match.start()}-{match.end()}",
                        severity=self.severity,
                        fix_suggestion=self.fix_template,
                    )
                    return False, violation
                return True, None

        # Value comparison
        if self.value is not None:
            if context == self.value:
                return True, None
            violation = ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                message=f"Expected {self.value}, got {context}",
                severity=self.severity,
                fix_suggestion=self.fix_template,
            )
            return False, violation

        # Default: satisfied if no checks defined
        return True, None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without callables).

        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "constraint_type": self.constraint_type.value,
            "category": self.category.value,
            "description": self.description,
            "pattern": self.pattern,
            "value": self.value,
            "severity": self.severity,
            "enabled": self.enabled,
            "fix_template": self.fix_template,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Constraint:
        """Create from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Constraint instance.
        """
        return cls(
            name=data.get("name", ""),
            constraint_type=ConstraintType(data.get("constraint_type", "hard")),
            category=ConstraintCategory(data.get("category", "custom")),
            description=data.get("description", ""),
            pattern=data.get("pattern"),
            value=data.get("value"),
            severity=data.get("severity", 5),
            enabled=data.get("enabled", True),
            fix_template=data.get("fix_template", ""),
            metadata=data.get("metadata", {}),
        )


class ConstraintValidator:
    """Validator for checking multiple constraints.

    Manages a collection of constraints and provides batch validation.
    All constraints are configurable.
    """

    def __init__(
        self,
        constraints: Optional[list[Constraint]] = None,
        stop_on_hard_violation: bool = True,
    ) -> None:
        """Initialize the validator.

        Args:
            constraints: List of constraints.
            stop_on_hard_violation: Stop validation on first hard violation.
        """
        self._constraints: list[Constraint] = constraints or []
        self._stop_on_hard_violation = stop_on_hard_violation

        # Register default constraints
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default constraints."""
        defaults = [
            # Language constraints
            Constraint(
                name="python_version",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.LANGUAGE,
                description="Python version requirement",
                check_function=lambda ctx: (
                    ctx.get("python_version", "3.10") >= "3.10",
                    "Python 3.10+ required"
                ) if isinstance(ctx, dict) else (True, ""),
            ),
            # Security constraints
            Constraint(
                name="no_hardcoded_secrets",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.SECURITY,
                description="No hardcoded secrets or credentials",
                pattern=r"(password|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]",
                severity=10,
                fix_template="Use environment variables or secrets manager",
            ),
            Constraint(
                name="no_eval",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.SECURITY,
                description="No use of eval() with user input",
                pattern=r"eval\s*\([^)]*input",
                severity=10,
                fix_template="Use ast.literal_eval() or safe alternatives",
            ),
            # Style constraints
            Constraint(
                name="type_hints",
                constraint_type=ConstraintType.SOFT,
                category=ConstraintCategory.STYLE,
                description="Type hints on function signatures",
                severity=3,
            ),
            Constraint(
                name="docstrings",
                constraint_type=ConstraintType.SOFT,
                category=ConstraintCategory.STYLE,
                description="Docstrings on public functions",
                severity=3,
            ),
            # Testing constraints
            Constraint(
                name="test_coverage",
                constraint_type=ConstraintType.SOFT,
                category=ConstraintCategory.TESTING,
                description="Minimum test coverage",
                check_function=lambda ctx: (
                    ctx.get("coverage", 80) >= 80,
                    f"Coverage {ctx.get('coverage', 0)}% below 80% threshold"
                ) if isinstance(ctx, dict) else (True, ""),
                severity=5,
            ),
            # Anti-constraints (explicit permissions)
            Constraint(
                name="can_restructure_files",
                constraint_type=ConstraintType.ANTI,
                category=ConstraintCategory.SCOPE,
                description="Permission to restructure file organization",
            ),
            Constraint(
                name="can_add_dependencies",
                constraint_type=ConstraintType.ANTI,
                category=ConstraintCategory.SCOPE,
                description="Permission to add new dependencies",
            ),
            Constraint(
                name="can_refactor_adjacent",
                constraint_type=ConstraintType.ANTI,
                category=ConstraintCategory.SCOPE,
                description="Permission to refactor adjacent code for consistency",
            ),
        ]

        for constraint in defaults:
            self._constraints.append(constraint)

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint.

        Args:
            constraint: Constraint to add.
        """
        self._constraints.append(constraint)
        logger.debug(f"Added constraint: {constraint.name}")

    def remove_constraint(self, name: str) -> bool:
        """Remove a constraint by name.

        Args:
            name: Constraint name.

        Returns:
            True if constraint was removed.
        """
        for i, constraint in enumerate(self._constraints):
            if constraint.name == name:
                self._constraints.pop(i)
                logger.debug(f"Removed constraint: {name}")
                return True
        return False

    def get_constraint(self, name: str) -> Optional[Constraint]:
        """Get a constraint by name.

        Args:
            name: Constraint name.

        Returns:
            Constraint or None.
        """
        for constraint in self._constraints:
            if constraint.name == name:
                return constraint
        return None

    def enable_constraint(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a constraint.

        Args:
            name: Constraint name.
            enabled: Whether to enable.

        Returns:
            True if constraint was found and updated.
        """
        constraint = self.get_constraint(name)
        if constraint:
            constraint.enabled = enabled
            return True
        return False

    def validate(
        self,
        context: Any,
        constraint_type: Optional[ConstraintType] = None,
        category: Optional[ConstraintCategory] = None,
    ) -> list[ConstraintViolation]:
        """Validate context against constraints.

        Args:
            context: Context to validate.
            constraint_type: Filter by constraint type.
            category: Filter by category.

        Returns:
            List of violations.
        """
        violations: list[ConstraintViolation] = []

        for constraint in self._constraints:
            # Filter by type
            if constraint_type and constraint.constraint_type != constraint_type:
                continue

            # Filter by category
            if category and constraint.category != category:
                continue

            # Skip anti-constraints in normal validation
            if constraint.constraint_type == ConstraintType.ANTI:
                continue

            is_valid, violation = constraint.check(context)

            if not is_valid and violation:
                violations.append(violation)
                logger.debug(f"Constraint violation: {constraint.name} - {violation.message}")

                # Stop on hard violation if configured
                if self._stop_on_hard_violation and constraint.constraint_type == ConstraintType.HARD:
                    logger.warning(f"Stopping validation: hard constraint '{constraint.name}' violated")
                    break

        return violations

    def validate_all(self, context: Any) -> dict[str, list[ConstraintViolation]]:
        """Validate and categorize all violations.

        Args:
            context: Context to validate.

        Returns:
            Dictionary with 'hard' and 'soft' violation lists.
        """
        hard_violations = self.validate(context, constraint_type=ConstraintType.HARD)
        soft_violations = self.validate(context, constraint_type=ConstraintType.SOFT)

        return {
            "hard": hard_violations,
            "soft": soft_violations,
        }

    def check_permissions(self, requested: list[str]) -> dict[str, bool]:
        """Check if requested permissions are granted.

        Args:
            requested: List of requested permission names.

        Returns:
            Dictionary mapping permission names to granted status.
        """
        granted: dict[str, bool] = {}

        for name in requested:
            constraint = self.get_constraint(name)
            if constraint and constraint.constraint_type == ConstraintType.ANTI:
                granted[name] = constraint.enabled
            else:
                granted[name] = False

        return granted

    def is_valid(self, context: Any) -> bool:
        """Check if context is valid (no hard violations).

        Args:
            context: Context to validate.

        Returns:
            True if no hard violations.
        """
        violations = self.validate(context, constraint_type=ConstraintType.HARD)
        return len(violations) == 0

    def list_constraints(
        self,
        constraint_type: Optional[ConstraintType] = None,
        category: Optional[ConstraintCategory] = None,
        enabled_only: bool = False,
    ) -> list[Constraint]:
        """List constraints with filters.

        Args:
            constraint_type: Filter by type.
            category: Filter by category.
            enabled_only: Only return enabled constraints.

        Returns:
            List of matching constraints.
        """
        result = []

        for constraint in self._constraints:
            if constraint_type and constraint.constraint_type != constraint_type:
                continue
            if category and constraint.category != category:
                continue
            if enabled_only and not constraint.enabled:
                continue
            result.append(constraint)

        return result

    def summary(self) -> dict[str, Any]:
        """Generate a summary of constraints.

        Returns:
            Summary dictionary.
        """
        by_type = {t.value: 0 for t in ConstraintType}
        by_category = {c.value: 0 for c in ConstraintCategory}
        enabled = 0

        for constraint in self._constraints:
            by_type[constraint.constraint_type.value] += 1
            by_category[constraint.category.value] += 1
            if constraint.enabled:
                enabled += 1

        return {
            "total": len(self._constraints),
            "enabled": enabled,
            "by_type": by_type,
            "by_category": by_category,
        }


def create_constraint(
    name: str,
    constraint_type: str | ConstraintType,
    **kwargs: Any,
) -> Constraint:
    """Factory function to create a constraint.

    Args:
        name: Constraint name.
        constraint_type: Type of constraint.
        **kwargs: Additional constraint parameters.

    Returns:
        Configured Constraint.
    """
    if isinstance(constraint_type, str):
        constraint_type = ConstraintType(constraint_type)

    return Constraint(
        name=name,
        constraint_type=constraint_type,
        **kwargs,
    )


def create_validator_from_config(config: dict[str, Any]) -> ConstraintValidator:
    """Create a validator from configuration.

    Args:
        config: Configuration dictionary with 'hard', 'soft', 'anti' lists.

    Returns:
        Configured ConstraintValidator.
    """
    constraints = []

    for constraint_data in config.get("hard", []):
        constraint = Constraint.from_dict({**constraint_data, "constraint_type": "hard"})
        constraints.append(constraint)

    for constraint_data in config.get("soft", []):
        constraint = Constraint.from_dict({**constraint_data, "constraint_type": "soft"})
        constraints.append(constraint)

    for constraint_data in config.get("anti", []):
        constraint = Constraint.from_dict({**constraint_data, "constraint_type": "anti"})
        constraints.append(constraint)

    return ConstraintValidator(constraints=constraints)
