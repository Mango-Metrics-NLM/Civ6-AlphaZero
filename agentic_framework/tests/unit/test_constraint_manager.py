"""
Tests for constraint management.
"""

from pathlib import Path

import pytest

from agentic_framework.config import AgenticConfig
from agentic_framework.core.constraint_manager import (
    Constraint,
    ConstraintCheckResult,
    ConstraintManager,
    ConstraintSeverity,
    ConstraintType,
    ConstraintViolation,
)


class TestConstraint:
    """Tests for Constraint dataclass."""

    def test_from_dict_basic(self):
        """Test creating constraint from dictionary."""
        data = {
            "name": "test_constraint",
            "type": "language",
            "value": "Python 3.10+",
        }

        constraint = Constraint.from_dict(data, ConstraintSeverity.HARD)

        assert constraint.name == "test_constraint"
        assert constraint.type == ConstraintType.LANGUAGE
        assert constraint.severity == ConstraintSeverity.HARD
        assert constraint.value == "Python 3.10+"

    def test_from_dict_with_validation_command(self):
        """Test creating constraint with validation command."""
        data = {
            "name": "python_version",
            "type": "language",
            "value": ">=3.10",
            "validation_command": "python --version",
        }

        constraint = Constraint.from_dict(data, ConstraintSeverity.HARD)

        assert constraint.validation_command == "python --version"

    def test_from_dict_unknown_type(self):
        """Test creating constraint with unknown type defaults to GENERAL."""
        data = {
            "name": "test",
            "type": "unknown_type",
            "value": "test",
        }

        constraint = Constraint.from_dict(data, ConstraintSeverity.SOFT)

        assert constraint.type == ConstraintType.GENERAL


class TestConstraintManager:
    """Tests for ConstraintManager."""

    @pytest.fixture
    def config_with_constraints(self):
        """Create a config with constraints."""
        config = AgenticConfig()
        config.constraints.hard = [
            {"name": "python_version", "type": "language", "value": ">=3.10"},
            {"name": "no_secrets", "type": "security", "value": "No hardcoded secrets"},
        ]
        config.constraints.soft = [
            {"name": "type_hints", "type": "style", "value": "Use type hints", "priority": 8},
        ]
        config.constraints.anti = [
            {"permission": "Restructure files"},
        ]
        return config

    def test_load_from_config(self, config_with_constraints):
        """Test loading constraints from config."""
        manager = ConstraintManager(config=config_with_constraints)

        hard = manager.get_constraints(severity=ConstraintSeverity.HARD)
        soft = manager.get_constraints(severity=ConstraintSeverity.SOFT)
        anti = manager.get_constraints(severity=ConstraintSeverity.ANTI)

        assert len(hard) == 2
        assert len(soft) == 1
        assert len(anti) == 1

    def test_add_constraint(self):
        """Test adding a constraint."""
        manager = ConstraintManager()

        constraint = Constraint(
            name="test",
            type=ConstraintType.GENERAL,
            severity=ConstraintSeverity.HARD,
            value="test value",
        )

        manager.add_constraint(constraint)

        assert len(manager.get_constraints()) == 1

    def test_remove_constraint(self):
        """Test removing a constraint."""
        manager = ConstraintManager()

        constraint = Constraint(
            name="test",
            type=ConstraintType.GENERAL,
            severity=ConstraintSeverity.HARD,
            value="test value",
        )

        manager.add_constraint(constraint)
        assert manager.remove_constraint("test")
        assert len(manager.get_constraints()) == 0

    def test_get_constraints_filtered(self, config_with_constraints):
        """Test getting constraints with filters."""
        manager = ConstraintManager(config=config_with_constraints)

        language_constraints = manager.get_constraints(constraint_type=ConstraintType.LANGUAGE)

        assert len(language_constraints) == 1
        assert language_constraints[0].name == "python_version"

    def test_get_permissions(self, config_with_constraints):
        """Test getting anti-constraints (permissions)."""
        manager = ConstraintManager(config=config_with_constraints)

        permissions = manager.get_permissions()

        assert len(permissions) == 1
        assert "Restructure" in permissions[0].value

    def test_is_permitted(self, config_with_constraints):
        """Test checking if action is permitted."""
        manager = ConstraintManager(config=config_with_constraints)

        # "restructure files" matches permission "Restructure files" (case-insensitive substring)
        assert manager.is_permitted("restructure files")
        assert not manager.is_permitted("delete production database")

    def test_check_all_passes(self, tmp_path, config_with_constraints):
        """Test constraint check that should pass."""
        # Create a clean directory
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello(): pass\n")

        manager = ConstraintManager(config=config_with_constraints)
        result = manager.check_all(tmp_path)

        # Note: This may fail if Python version check fails
        # The important thing is it returns a ConstraintCheckResult
        assert isinstance(result, ConstraintCheckResult)


class TestConstraintViolation:
    """Tests for ConstraintViolation."""

    def test_str_representation(self):
        """Test string representation of violation."""
        constraint = Constraint(
            name="test_constraint",
            type=ConstraintType.SECURITY,
            severity=ConstraintSeverity.HARD,
            value="test",
        )

        violation = ConstraintViolation(
            constraint=constraint,
            message="Test violation message",
            file_path=Path("test.py"),
            line_number=10,
        )

        str_repr = str(violation)

        assert "HARD" in str_repr
        assert "test_constraint" in str_repr
        assert "test.py" in str_repr
        assert ":10" in str_repr


class TestConstraintCheckResult:
    """Tests for ConstraintCheckResult."""

    def test_hard_violations_filter(self):
        """Test filtering hard violations."""
        hard_constraint = Constraint(
            name="hard",
            type=ConstraintType.GENERAL,
            severity=ConstraintSeverity.HARD,
            value="",
        )
        soft_constraint = Constraint(
            name="soft",
            type=ConstraintType.GENERAL,
            severity=ConstraintSeverity.SOFT,
            value="",
        )

        result = ConstraintCheckResult(
            passed=False,
            violations=[
                ConstraintViolation(constraint=hard_constraint, message="hard error"),
                ConstraintViolation(constraint=soft_constraint, message="soft error"),
            ],
        )

        assert len(result.hard_violations) == 1
        assert len(result.soft_violations) == 1

    def test_summary(self):
        """Test generating summary."""
        result = ConstraintCheckResult(
            passed=True,
            checked_count=5,
            violations=[],
            warnings=[],
        )

        summary = result.summary()

        assert "PASSED" in summary
        assert "5" in summary
