"""
Tests for permission management.
"""

from pathlib import Path

import pytest

from agentic_framework.config import AgenticConfig
from agentic_framework.core.permission_manager import (
    PermissionManager,
    AutonomyLevel,
    PermissionCheckResult,
    ScopeChecker,
    ResourceUsage,
)


class TestScopeChecker:
    """Tests for ScopeChecker."""

    def test_simple_in_scope(self, tmp_path):
        """Test simple in-scope check."""
        checker = ScopeChecker(
            in_scope_patterns=["*.py"],
            out_of_scope_patterns=[],
            base_path=tmp_path,
        )

        test_file = tmp_path / "test.py"

        assert checker.is_in_scope(test_file)

    def test_simple_out_of_scope(self, tmp_path):
        """Test simple out-of-scope check."""
        checker = ScopeChecker(
            in_scope_patterns=["*.py"],
            out_of_scope_patterns=["vendor/**"],
            base_path=tmp_path,
        )

        vendor_file = tmp_path / "vendor" / "lib.py"

        assert not checker.is_in_scope(vendor_file)

    def test_glob_pattern(self, tmp_path):
        """Test glob pattern matching."""
        checker = ScopeChecker(
            in_scope_patterns=["src/**/*.py"],
            out_of_scope_patterns=[],
            base_path=tmp_path,
        )

        src_file = tmp_path / "src" / "module" / "test.py"
        other_file = tmp_path / "other" / "test.py"

        assert checker.is_in_scope(src_file)
        assert not checker.is_in_scope(other_file)

    def test_out_of_scope_takes_precedence(self, tmp_path):
        """Test that out-of-scope patterns take precedence."""
        checker = ScopeChecker(
            in_scope_patterns=["**/*.py"],
            out_of_scope_patterns=["__pycache__/**"],
            base_path=tmp_path,
        )

        cache_file = tmp_path / "__pycache__" / "module.cpython-310.pyc"

        assert not checker.is_in_scope(cache_file)

    def test_filter_in_scope(self, tmp_path):
        """Test filtering paths to in-scope only."""
        checker = ScopeChecker(
            in_scope_patterns=["src/**/*.py"],
            out_of_scope_patterns=[],
            base_path=tmp_path,
        )

        paths = [
            tmp_path / "src" / "main.py",
            tmp_path / "tests" / "test_main.py",
            tmp_path / "src" / "utils" / "helpers.py",
        ]

        filtered = checker.filter_in_scope(paths)

        assert len(filtered) == 2
        assert all("src" in str(p) for p in filtered)


class TestResourceUsage:
    """Tests for ResourceUsage tracking."""

    def test_initial_state(self):
        """Test initial resource usage state."""
        usage = ResourceUsage(
            max_iterations=5,
            max_files_per_pass=20,
            exploration_timeout_seconds=600,
        )

        assert usage.iterations == 0
        assert usage.can_iterate()
        assert usage.iterations_remaining == 5

    def test_record_iteration(self):
        """Test recording iterations."""
        usage = ResourceUsage(max_iterations=3)

        usage.record_iteration()
        assert usage.iterations == 1
        assert usage.can_iterate()

        usage.record_iteration()
        usage.record_iteration()
        assert usage.iterations == 3
        assert not usage.can_iterate()

    def test_record_file_modification(self):
        """Test recording file modifications."""
        usage = ResourceUsage(max_files_per_pass=5)

        usage.record_file_modification(3)
        assert usage.files_modified == 3
        assert usage.can_modify_file()

        usage.record_file_modification(3)
        assert not usage.can_modify_file()

    def test_reset(self):
        """Test resetting usage counters."""
        usage = ResourceUsage()
        usage.record_iteration()
        usage.record_file_modification()
        usage.record_exploration_time(100)

        usage.reset()

        assert usage.iterations == 0
        assert usage.files_modified == 0
        assert usage.exploration_seconds == 0.0


class TestPermissionManager:
    """Tests for PermissionManager."""

    @pytest.fixture
    def config_with_permissions(self):
        """Create a config with permissions."""
        config = AgenticConfig()
        config.permissions.scope.in_scope = ["src/**/*.py", "tests/**/*.py"]
        config.permissions.scope.out_of_scope = ["vendor/**", "__pycache__/**"]
        config.permissions.autonomy.autonomous = [
            "file_creation_in_scope",
            "running_tests",
        ]
        config.permissions.autonomy.confirm_first = [
            "architectural_changes",
            "breaking_api_changes",
        ]
        config.permissions.autonomy.prohibited = [
            "commits_to_main",
            "external_api_calls",
        ]
        config.permissions.resource_budget.max_iterations = 5
        config.permissions.resource_budget.max_files_per_pass = 10
        return config

    def test_load_from_config(self, config_with_permissions, tmp_path):
        """Test loading permissions from config."""
        manager = PermissionManager(
            config=config_with_permissions,
            base_path=tmp_path,
        )

        summary = manager.get_permission_summary()

        assert summary["autonomous_actions"] == 2
        assert summary["confirm_first_actions"] == 2
        assert summary["prohibited_actions"] == 2

    def test_check_autonomous_action(self, config_with_permissions, tmp_path):
        """Test checking autonomous action."""
        manager = PermissionManager(
            config=config_with_permissions,
            base_path=tmp_path,
        )

        result = manager.check_permission("running tests for module")

        assert result.allowed
        assert result.level == AutonomyLevel.AUTONOMOUS
        assert not result.requires_confirmation

    def test_check_confirm_first_action(self, config_with_permissions, tmp_path):
        """Test checking action requiring confirmation."""
        manager = PermissionManager(
            config=config_with_permissions,
            base_path=tmp_path,
        )

        result = manager.check_permission("making architectural changes to module")

        assert result.allowed
        assert result.level == AutonomyLevel.CONFIRM_FIRST
        assert result.requires_confirmation

    def test_check_prohibited_action(self, config_with_permissions, tmp_path):
        """Test checking prohibited action."""
        manager = PermissionManager(
            config=config_with_permissions,
            base_path=tmp_path,
        )

        result = manager.check_permission("commit to main branch")

        assert not result.allowed
        assert result.level == AutonomyLevel.PROHIBITED

    def test_check_permission_with_path(self, config_with_permissions, tmp_path):
        """Test checking permission with path scope check."""
        manager = PermissionManager(
            config=config_with_permissions,
            base_path=tmp_path,
        )

        # In scope path
        in_scope_path = tmp_path / "src" / "module.py"
        result = manager.check_permission("modify file", in_scope_path)
        assert result.scope_check is True

        # Out of scope path
        out_scope_path = tmp_path / "vendor" / "lib.py"
        result = manager.check_permission("modify file", out_scope_path)
        assert result.allowed is False

    def test_can_proceed_with_callback(self, config_with_permissions, tmp_path):
        """Test can_proceed with confirmation callback."""
        manager = PermissionManager(
            config=config_with_permissions,
            base_path=tmp_path,
        )

        # Set callback that always confirms
        manager.set_confirmation_callback(lambda msg: True)

        can_proceed = manager.can_proceed("making architectural changes")
        assert can_proceed

        # Set callback that always denies
        manager.set_confirmation_callback(lambda msg: False)

        can_proceed = manager.can_proceed("making architectural changes")
        assert not can_proceed

    def test_check_resource_budget(self, config_with_permissions, tmp_path):
        """Test checking resource budget."""
        manager = PermissionManager(
            config=config_with_permissions,
            base_path=tmp_path,
        )

        assert manager.check_resource_budget("iteration")
        assert manager.check_resource_budget("file")
        assert manager.check_resource_budget("exploration")

        # Exhaust iterations
        for _ in range(5):
            manager.resource_usage.record_iteration()

        assert not manager.check_resource_budget("iteration")
