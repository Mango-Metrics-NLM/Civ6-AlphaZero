"""
Permission Manager for Agentic Framework

Manages scope boundaries, autonomy levels, and action permissions.
Implements the permission architecture from the template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Callable

from ..config import AgenticConfig
from ..utils.logging_config import get_logger

logger = get_logger("permission_manager")


class AutonomyLevel(Enum):
    """Autonomy levels for actions."""

    AUTONOMOUS = "autonomous"  # Proceed without asking
    CONFIRM_FIRST = "confirm_first"  # Ask before proceeding
    PROHIBITED = "prohibited"  # Never attempt


class ActionCategory(Enum):
    """Categories of actions."""

    FILE_OPERATION = "file_operation"
    CODE_CHANGE = "code_change"
    DEPENDENCY = "dependency"
    TEST = "test"
    GIT = "git"
    EXTERNAL = "external"
    ARCHITECTURE = "architecture"
    API = "api"
    GENERAL = "general"


@dataclass
class ActionPermission:
    """Represents permission for an action."""

    action: str
    level: AutonomyLevel
    category: ActionCategory = ActionCategory.GENERAL
    conditions: list[str] = field(default_factory=list)
    reason: Optional[str] = None


@dataclass
class PermissionCheckResult:
    """Result of checking permission for an action."""

    action: str
    allowed: bool
    level: AutonomyLevel
    requires_confirmation: bool = False
    reason: Optional[str] = None
    matching_rule: Optional[str] = None
    scope_check: Optional[bool] = None

    def __bool__(self) -> bool:
        """Return True if action is allowed."""
        return self.allowed


@dataclass
class ResourceUsage:
    """Tracks resource usage against budget."""

    iterations: int = 0
    files_modified: int = 0
    exploration_seconds: float = 0.0

    max_iterations: int = 5
    max_files_per_pass: int = 20
    exploration_timeout_seconds: float = 600.0

    @property
    def iterations_remaining(self) -> int:
        """Get remaining iterations."""
        return max(0, self.max_iterations - self.iterations)

    @property
    def files_remaining(self) -> int:
        """Get remaining files that can be modified."""
        return max(0, self.max_files_per_pass - self.files_modified)

    @property
    def exploration_remaining(self) -> float:
        """Get remaining exploration time in seconds."""
        return max(0.0, self.exploration_timeout_seconds - self.exploration_seconds)

    def can_iterate(self) -> bool:
        """Check if more iterations are allowed."""
        return self.iterations < self.max_iterations

    def can_modify_file(self) -> bool:
        """Check if more files can be modified."""
        return self.files_modified < self.max_files_per_pass

    def can_explore(self) -> bool:
        """Check if more exploration time is available."""
        return self.exploration_seconds < self.exploration_timeout_seconds

    def record_iteration(self) -> None:
        """Record an iteration."""
        self.iterations += 1
        logger.debug(f"Iteration {self.iterations}/{self.max_iterations}")

    def record_file_modification(self, count: int = 1) -> None:
        """Record file modifications."""
        self.files_modified += count
        logger.debug(f"Files modified: {self.files_modified}/{self.max_files_per_pass}")

    def record_exploration_time(self, seconds: float) -> None:
        """Record exploration time."""
        self.exploration_seconds += seconds
        logger.debug(f"Exploration time: {self.exploration_seconds:.1f}/{self.exploration_timeout_seconds}")

    def reset(self) -> None:
        """Reset usage counters."""
        self.iterations = 0
        self.files_modified = 0
        self.exploration_seconds = 0.0


class ScopeChecker:
    """Checks if paths are within scope."""

    def __init__(
        self,
        in_scope_patterns: list[str],
        out_of_scope_patterns: list[str],
        base_path: Optional[Path] = None,
    ) -> None:
        """Initialize the scope checker.

        Args:
            in_scope_patterns: Glob patterns for files in scope.
            out_of_scope_patterns: Glob patterns for files out of scope.
            base_path: Base path for relative pattern matching.
        """
        self.in_scope_patterns = in_scope_patterns
        self.out_of_scope_patterns = out_of_scope_patterns
        self.base_path = base_path or Path.cwd()

    def is_in_scope(self, path: Path) -> bool:
        """Check if a path is within scope.

        Args:
            path: Path to check.

        Returns:
            True if path is in scope.
        """
        # Convert to relative path if absolute
        try:
            rel_path = path.relative_to(self.base_path)
        except ValueError:
            rel_path = path

        rel_path_str = str(rel_path)

        # Check if explicitly out of scope first
        for pattern in self.out_of_scope_patterns:
            if self._matches_pattern(rel_path_str, pattern):
                logger.debug(f"Path {rel_path} matches out-of-scope pattern: {pattern}")
                return False

        # Check if in scope
        for pattern in self.in_scope_patterns:
            if self._matches_pattern(rel_path_str, pattern):
                logger.debug(f"Path {rel_path} matches in-scope pattern: {pattern}")
                return True

        # Default: not in scope if no pattern matches
        logger.debug(f"Path {rel_path} does not match any in-scope pattern")
        return False

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if a path matches a glob pattern.

        Args:
            path: Path string to check.
            pattern: Glob pattern.

        Returns:
            True if path matches pattern.
            
        Note:
            Uses pathlib.Path.match() which has some limitations:
            - `src/**/*.py` won't match `src/file.py` (requires nested path)
            - Patterns without path separators match as suffixes
            For exact shell-like globbing, consider using the pathspec library.
        """
        # Use pathlib.Path.match() which handles ** patterns better than the
        # previous regex implementation
        return Path(path).match(pattern)

    def filter_in_scope(self, paths: list[Path]) -> list[Path]:
        """Filter a list of paths to only those in scope.

        Args:
            paths: List of paths to filter.

        Returns:
            List of paths that are in scope.
        """
        return [p for p in paths if self.is_in_scope(p)]


class PermissionManager:
    """Manages permissions and autonomy for agentic actions.

    Features:
    - Scope checking for file operations
    - Autonomy level determination for actions
    - Resource budget tracking
    - Confirmation workflow support
    """

    def __init__(
        self,
        config: Optional[AgenticConfig] = None,
        base_path: Optional[Path] = None,
    ) -> None:
        """Initialize the permission manager.

        Args:
            config: Configuration containing permission definitions.
            base_path: Base path for scope checking.
        """
        self.config = config
        self.base_path = base_path or Path.cwd()
        self._scope_checker: Optional[ScopeChecker] = None
        self._resource_usage = ResourceUsage()
        self._action_rules: list[ActionPermission] = []
        self._confirmation_callback: Optional[Callable[[str], bool]] = None

        if config:
            self.load_from_config(config)

    def load_from_config(self, config: AgenticConfig) -> None:
        """Load permissions from configuration.

        Args:
            config: Configuration containing permission definitions.
        """
        self.config = config

        # Set up scope checker
        self._scope_checker = ScopeChecker(
            in_scope_patterns=config.permissions.scope.in_scope,
            out_of_scope_patterns=config.permissions.scope.out_of_scope,
            base_path=self.base_path,
        )

        # Set up resource budget
        budget = config.permissions.resource_budget
        self._resource_usage = ResourceUsage(
            max_iterations=budget.max_iterations,
            max_files_per_pass=budget.max_files_per_pass,
            exploration_timeout_seconds=budget.exploration_timeout_seconds,
        )

        # Load action rules
        self._action_rules.clear()

        for action in config.permissions.autonomy.autonomous:
            self._action_rules.append(
                ActionPermission(
                    action=action,
                    level=AutonomyLevel.AUTONOMOUS,
                    category=self._categorize_action(action),
                )
            )

        for action in config.permissions.autonomy.confirm_first:
            self._action_rules.append(
                ActionPermission(
                    action=action,
                    level=AutonomyLevel.CONFIRM_FIRST,
                    category=self._categorize_action(action),
                )
            )

        for action in config.permissions.autonomy.prohibited:
            self._action_rules.append(
                ActionPermission(
                    action=action,
                    level=AutonomyLevel.PROHIBITED,
                    category=self._categorize_action(action),
                )
            )

        logger.info(f"Loaded {len(self._action_rules)} permission rules")

    def _categorize_action(self, action: str) -> ActionCategory:
        """Categorize an action based on its description.

        Args:
            action: Action description.

        Returns:
            ActionCategory for the action.
        """
        action_lower = action.lower()

        if any(kw in action_lower for kw in ["file", "create", "delete", "modify"]):
            return ActionCategory.FILE_OPERATION
        elif any(kw in action_lower for kw in ["refactor", "code", "implement"]):
            return ActionCategory.CODE_CHANGE
        elif any(kw in action_lower for kw in ["dependency", "install", "package"]):
            return ActionCategory.DEPENDENCY
        elif any(kw in action_lower for kw in ["test", "pytest", "coverage"]):
            return ActionCategory.TEST
        elif any(kw in action_lower for kw in ["git", "commit", "push", "branch"]):
            return ActionCategory.GIT
        elif any(kw in action_lower for kw in ["external", "api", "http", "request"]):
            return ActionCategory.EXTERNAL
        elif any(kw in action_lower for kw in ["architecture", "design", "structure"]):
            return ActionCategory.ARCHITECTURE
        elif any(kw in action_lower for kw in ["api", "interface", "breaking"]):
            return ActionCategory.API
        else:
            return ActionCategory.GENERAL

    @property
    def scope_checker(self) -> ScopeChecker:
        """Get the scope checker."""
        if self._scope_checker is None:
            self._scope_checker = ScopeChecker([], [], self.base_path)
        return self._scope_checker

    @property
    def resource_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        return self._resource_usage

    def set_confirmation_callback(
        self, callback: Callable[[str], bool]
    ) -> None:
        """Set a callback for confirmation requests.

        Args:
            callback: Function that takes a description and returns True to proceed.
        """
        self._confirmation_callback = callback

    def is_path_in_scope(self, path: Path) -> bool:
        """Check if a path is within scope.

        Args:
            path: Path to check.

        Returns:
            True if path is in scope.
        """
        return self.scope_checker.is_in_scope(path)

    def check_permission(self, action: str, path: Optional[Path] = None) -> PermissionCheckResult:
        """Check permission for an action.

        Args:
            action: Description of the action.
            path: Optional path involved in the action.

        Returns:
            PermissionCheckResult with decision.
        """
        action_lower = action.lower()

        # Check scope first if path is provided
        scope_check = None
        if path is not None:
            scope_check = self.is_path_in_scope(path)
            if not scope_check:
                return PermissionCheckResult(
                    action=action,
                    allowed=False,
                    level=AutonomyLevel.PROHIBITED,
                    reason=f"Path {path} is out of scope",
                    scope_check=False,
                )

        # Find matching rule
        for rule in self._action_rules:
            rule_pattern = rule.action.lower().replace("_", " ")

            # Check for keyword match
            if self._action_matches_rule(action_lower, rule_pattern):
                if rule.level == AutonomyLevel.PROHIBITED:
                    return PermissionCheckResult(
                        action=action,
                        allowed=False,
                        level=AutonomyLevel.PROHIBITED,
                        reason=f"Action is prohibited: {rule.action}",
                        matching_rule=rule.action,
                        scope_check=scope_check,
                    )
                elif rule.level == AutonomyLevel.CONFIRM_FIRST:
                    return PermissionCheckResult(
                        action=action,
                        allowed=True,
                        level=AutonomyLevel.CONFIRM_FIRST,
                        requires_confirmation=True,
                        reason=f"Action requires confirmation: {rule.action}",
                        matching_rule=rule.action,
                        scope_check=scope_check,
                    )
                else:
                    return PermissionCheckResult(
                        action=action,
                        allowed=True,
                        level=AutonomyLevel.AUTONOMOUS,
                        matching_rule=rule.action,
                        scope_check=scope_check,
                    )

        # Default: require confirmation for unrecognized actions
        return PermissionCheckResult(
            action=action,
            allowed=True,
            level=AutonomyLevel.CONFIRM_FIRST,
            requires_confirmation=True,
            reason="No matching rule found, defaulting to confirm first",
            scope_check=scope_check,
        )

    def _action_matches_rule(self, action: str, rule: str) -> bool:
        """Check if an action description matches a rule.

        Args:
            action: Action description (lowercase).
            rule: Rule pattern (lowercase).

        Returns:
            True if action matches rule.
        """
        # Split into keywords
        rule_keywords = set(rule.split())
        action_keywords = set(action.split())

        # Check for significant overlap
        common = rule_keywords & action_keywords
        if len(common) >= min(2, len(rule_keywords)):
            return True

        # Check for substring match
        if rule in action or action in rule:
            return True

        return False

    def request_confirmation(self, action: str, details: Optional[str] = None) -> bool:
        """Request confirmation for an action.

        Args:
            action: Description of the action.
            details: Additional details about the action.

        Returns:
            True if confirmed, False otherwise.
        """
        if self._confirmation_callback is not None:
            message = action
            if details:
                message += f"\n{details}"
            return self._confirmation_callback(message)

        # No callback set - log and default to False
        logger.warning(f"Confirmation requested but no callback set: {action}")
        return False

    def can_proceed(
        self,
        action: str,
        path: Optional[Path] = None,
        auto_confirm: bool = False,
    ) -> bool:
        """Check if an action can proceed.

        Args:
            action: Description of the action.
            path: Optional path involved in the action.
            auto_confirm: Automatically confirm if required.

        Returns:
            True if action can proceed.
        """
        result = self.check_permission(action, path)

        if not result.allowed:
            logger.info(f"Action not allowed: {action} - {result.reason}")
            return False

        if result.requires_confirmation:
            if auto_confirm:
                logger.info(f"Auto-confirming action: {action}")
                return True
            else:
                return self.request_confirmation(action)

        return True

    def check_resource_budget(self, resource_type: str) -> bool:
        """Check if a resource budget allows more usage.

        Args:
            resource_type: Type of resource ('iteration', 'file', 'exploration').

        Returns:
            True if more usage is allowed.
        """
        if resource_type == "iteration":
            return self._resource_usage.can_iterate()
        elif resource_type == "file":
            return self._resource_usage.can_modify_file()
        elif resource_type == "exploration":
            return self._resource_usage.can_explore()
        return True

    def get_permission_summary(self) -> dict[str, Any]:
        """Get a summary of current permissions.

        Returns:
            Dictionary with permission summary.
        """
        return {
            "autonomous_actions": len([r for r in self._action_rules if r.level == AutonomyLevel.AUTONOMOUS]),
            "confirm_first_actions": len([r for r in self._action_rules if r.level == AutonomyLevel.CONFIRM_FIRST]),
            "prohibited_actions": len([r for r in self._action_rules if r.level == AutonomyLevel.PROHIBITED]),
            "in_scope_patterns": self.scope_checker.in_scope_patterns,
            "out_of_scope_patterns": self.scope_checker.out_of_scope_patterns,
            "resource_budget": {
                "iterations": f"{self._resource_usage.iterations}/{self._resource_usage.max_iterations}",
                "files": f"{self._resource_usage.files_modified}/{self._resource_usage.max_files_per_pass}",
                "exploration": f"{self._resource_usage.exploration_seconds:.1f}/{self._resource_usage.exploration_timeout_seconds}s",
            },
        }
