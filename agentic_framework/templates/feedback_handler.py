"""
Feedback Handler for Agentic Coding System

Provides error pattern matching, recovery strategies, and feedback loop management.
All patterns and actions are configurable - no hardcoded values.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..utils.logging_config import get_logger

logger = get_logger("feedback_handler")


class FeedbackStrategy(Enum):
    """Strategies for handling feedback."""

    AUTO_FIX = "auto_fix"
    ANALYZE_FIX = "analyze_fix"
    REPORT = "report"
    ESCALATE = "escalate"
    RETRY = "retry"
    IGNORE = "ignore"


class RecoveryAction(Enum):
    """Actions to take for recovery."""

    RERUN = "rerun"
    FIX_AND_RERUN = "fix_and_rerun"
    ROLLBACK = "rollback"
    SKIP = "skip"
    STOP = "stop"
    ASK_USER = "ask_user"


@dataclass
class ErrorPattern:
    """Pattern for matching errors.

    Attributes:
        name: Pattern name for identification
        pattern: Regex pattern to match
        error_type: Type of error this pattern matches
        strategy: Strategy for handling matches
        recovery_action: Action to take for recovery
        max_occurrences: Maximum times to handle before escalating
        extract_groups: Named groups to extract from match
        fix_template: Template for auto-fix (if applicable)
        metadata: Additional configuration
    """

    name: str
    pattern: str
    error_type: str = "general"
    strategy: FeedbackStrategy = FeedbackStrategy.ANALYZE_FIX
    recovery_action: RecoveryAction = RecoveryAction.FIX_AND_RERUN
    max_occurrences: int = 3
    extract_groups: list[str] = field(default_factory=list)
    fix_template: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    _compiled: re.Pattern[str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Compile the pattern after initialization."""
        self._compiled = re.compile(self.pattern, re.MULTILINE)

    def match(self, text: str) -> list[dict[str, Any]]:
        """Find all matches in text.

        Args:
            text: Text to search.

        Returns:
            List of match dictionaries with extracted groups.
        """
        if self._compiled is None:
            self._compiled = re.compile(self.pattern, re.MULTILINE)

        matches = []
        for m in self._compiled.finditer(text):
            match_data = {
                "full_match": m.group(0),
                "start": m.start(),
                "end": m.end(),
                "groups": m.groups(),
            }

            # Extract named groups if specified
            if self.extract_groups:
                for group_name in self.extract_groups:
                    try:
                        match_data[group_name] = m.group(group_name)
                    except IndexError:
                        pass

            # Also include all named groups from the pattern
            match_data.update(m.groupdict())
            matches.append(match_data)

        return matches

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "pattern": self.pattern,
            "error_type": self.error_type,
            "strategy": self.strategy.value,
            "recovery_action": self.recovery_action.value,
            "max_occurrences": self.max_occurrences,
            "extract_groups": self.extract_groups,
            "fix_template": self.fix_template,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErrorPattern:
        """Create from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            ErrorPattern instance.
        """
        return cls(
            name=data.get("name", ""),
            pattern=data.get("pattern", ""),
            error_type=data.get("error_type", "general"),
            strategy=FeedbackStrategy(data.get("strategy", "analyze_fix")),
            recovery_action=RecoveryAction(data.get("recovery_action", "fix_and_rerun")),
            max_occurrences=data.get("max_occurrences", 3),
            extract_groups=data.get("extract_groups", []),
            fix_template=data.get("fix_template", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FeedbackEvent:
    """An event in the feedback loop.

    Attributes:
        event_type: Type of event (error, warning, info)
        source: Source of the event (step name, agent, etc.)
        message: Event message
        pattern_name: Name of matched pattern (if any)
        match_data: Data from pattern match
        timestamp: When event occurred
        handled: Whether event has been handled
        resolution: How the event was resolved
    """

    event_type: str
    source: str
    message: str
    pattern_name: str | None = None
    match_data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    handled: bool = False
    resolution: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "event_type": self.event_type,
            "source": self.source,
            "message": self.message,
            "pattern_name": self.pattern_name,
            "match_data": self.match_data,
            "timestamp": self.timestamp,
            "handled": self.handled,
            "resolution": self.resolution,
        }


class FeedbackHandler:
    """Handler for feedback loop management.

    Matches errors to patterns, determines strategies, and tracks occurrences.
    All patterns are configurable at initialization or runtime.
    """

    def __init__(
        self,
        patterns: list[ErrorPattern] | None = None,
        on_error: Callable[[FeedbackEvent], FeedbackStrategy] | None = None,
        on_fix_request: Callable[[FeedbackEvent, str], bool] | None = None,
        on_escalate: Callable[[FeedbackEvent], None] | None = None,
        max_total_errors: int = 10,
        auto_fix_enabled: bool = True,
    ) -> None:
        """Initialize the feedback handler.

        Args:
            patterns: List of error patterns.
            on_error: Callback when error is detected.
            on_fix_request: Callback to request a fix.
            on_escalate: Callback when escalation is needed.
            max_total_errors: Maximum total errors before stopping.
            auto_fix_enabled: Whether auto-fix is enabled.
        """
        self._patterns: list[ErrorPattern] = patterns or []
        self._on_error = on_error
        self._on_fix_request = on_fix_request
        self._on_escalate = on_escalate
        self._max_total_errors = max_total_errors
        self._auto_fix_enabled = auto_fix_enabled

        # Tracking
        self._events: list[FeedbackEvent] = []
        self._occurrence_counts: dict[str, int] = {}
        self._total_errors = 0

        # Register default patterns
        self._register_default_patterns()

    def _register_default_patterns(self) -> None:
        """Register default error patterns."""
        defaults = [
            ErrorPattern(
                name="lint_error",
                pattern=r"(?P<file>[^\s:]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<code>[A-Z]\d+)\s+(?P<message>.+)",
                error_type="lint",
                strategy=FeedbackStrategy.AUTO_FIX,
                recovery_action=RecoveryAction.FIX_AND_RERUN,
                extract_groups=["file", "line", "col", "code", "message"],
            ),
            ErrorPattern(
                name="type_error",
                pattern=r"(?P<file>[^\s:]+):(?P<line>\d+):\s*error:\s*(?P<message>.+)",
                error_type="type_check",
                strategy=FeedbackStrategy.ANALYZE_FIX,
                recovery_action=RecoveryAction.FIX_AND_RERUN,
                extract_groups=["file", "line", "message"],
            ),
            ErrorPattern(
                name="test_failure",
                pattern=r"FAILED\s+(?P<test>[^\s]+)\s*-\s*(?P<reason>.+)",
                error_type="test",
                strategy=FeedbackStrategy.ANALYZE_FIX,
                recovery_action=RecoveryAction.FIX_AND_RERUN,
                extract_groups=["test", "reason"],
            ),
            ErrorPattern(
                name="import_error",
                pattern=r"ImportError:\s*(?P<message>.+)",
                error_type="import",
                strategy=FeedbackStrategy.ANALYZE_FIX,
                recovery_action=RecoveryAction.FIX_AND_RERUN,
            ),
            ErrorPattern(
                name="syntax_error",
                pattern=r"SyntaxError:\s*(?P<message>.+)",
                error_type="syntax",
                strategy=FeedbackStrategy.ANALYZE_FIX,
                recovery_action=RecoveryAction.FIX_AND_RERUN,
            ),
            ErrorPattern(
                name="assertion_error",
                pattern=r"AssertionError:\s*(?P<message>.*)",
                error_type="assertion",
                strategy=FeedbackStrategy.ANALYZE_FIX,
                recovery_action=RecoveryAction.FIX_AND_RERUN,
            ),
            ErrorPattern(
                name="permission_denied",
                pattern=r"PermissionError:\s*(?P<message>.+)",
                error_type="permission",
                strategy=FeedbackStrategy.ESCALATE,
                recovery_action=RecoveryAction.ASK_USER,
            ),
            ErrorPattern(
                name="timeout_error",
                pattern=r"TimeoutError:\s*(?P<message>.+)",
                error_type="timeout",
                strategy=FeedbackStrategy.RETRY,
                recovery_action=RecoveryAction.RERUN,
                max_occurrences=2,
            ),
        ]

        for pattern in defaults:
            self._patterns.append(pattern)

    def add_pattern(self, pattern: ErrorPattern) -> None:
        """Add an error pattern.

        Args:
            pattern: Pattern to add.
        """
        self._patterns.append(pattern)
        logger.debug(f"Added error pattern: {pattern.name}")

    def remove_pattern(self, name: str) -> bool:
        """Remove a pattern by name.

        Args:
            name: Pattern name.

        Returns:
            True if pattern was removed.
        """
        for i, pattern in enumerate(self._patterns):
            if pattern.name == name:
                self._patterns.pop(i)
                logger.debug(f"Removed error pattern: {name}")
                return True
        return False

    def get_pattern(self, name: str) -> ErrorPattern | None:
        """Get a pattern by name.

        Args:
            name: Pattern name.

        Returns:
            ErrorPattern or None.
        """
        for pattern in self._patterns:
            if pattern.name == name:
                return pattern
        return None

    def process_output(
        self,
        output: str,
        source: str = "unknown",
    ) -> list[FeedbackEvent]:
        """Process output for errors.

        Args:
            output: Output text to analyze.
            source: Source identifier.

        Returns:
            List of FeedbackEvents.
        """
        events: list[FeedbackEvent] = []

        for pattern in self._patterns:
            matches = pattern.match(output)

            for match_data in matches:
                event = FeedbackEvent(
                    event_type="error",
                    source=source,
                    message=match_data.get("full_match", ""),
                    pattern_name=pattern.name,
                    match_data=match_data,
                )

                events.append(event)
                self._events.append(event)
                self._total_errors += 1

                # Track occurrences
                self._occurrence_counts[pattern.name] = (
                    self._occurrence_counts.get(pattern.name, 0) + 1
                )

                logger.debug(
                    f"Matched pattern '{pattern.name}' in {source}: "
                    f"{match_data.get('full_match', '')[:100]}"
                )

        return events

    def determine_strategy(self, event: FeedbackEvent) -> FeedbackStrategy:
        """Determine the strategy for handling an event.

        Args:
            event: Feedback event.

        Returns:
            FeedbackStrategy to use.
        """
        # Check if we've exceeded total error limit
        if self._total_errors > self._max_total_errors:
            logger.warning("Total error limit exceeded, escalating")
            return FeedbackStrategy.ESCALATE

        # Get the pattern if available
        pattern = None
        if event.pattern_name:
            pattern = self.get_pattern(event.pattern_name)

        if pattern:
            # Check occurrence limit
            occurrences = self._occurrence_counts.get(pattern.name, 0)
            if occurrences > pattern.max_occurrences:
                logger.warning(
                    f"Pattern '{pattern.name}' exceeded max occurrences "
                    f"({occurrences}/{pattern.max_occurrences}), escalating"
                )
                return FeedbackStrategy.ESCALATE

            # Check if auto-fix is disabled
            if pattern.strategy == FeedbackStrategy.AUTO_FIX and not self._auto_fix_enabled:
                return FeedbackStrategy.ANALYZE_FIX

            return pattern.strategy

        # Callback override
        if self._on_error:
            return self._on_error(event)

        # Default strategy
        return FeedbackStrategy.ANALYZE_FIX

    def determine_recovery_action(self, event: FeedbackEvent) -> RecoveryAction:
        """Determine the recovery action for an event.

        Args:
            event: Feedback event.

        Returns:
            RecoveryAction to take.
        """
        if event.pattern_name:
            pattern = self.get_pattern(event.pattern_name)
            if pattern:
                return pattern.recovery_action

        return RecoveryAction.FIX_AND_RERUN

    def handle_event(self, event: FeedbackEvent) -> bool:
        """Handle a feedback event.

        Args:
            event: Event to handle.

        Returns:
            True if handling was successful.
        """
        if event.handled:
            return True

        strategy = self.determine_strategy(event)
        action = self.determine_recovery_action(event)

        logger.info(
            f"Handling event from {event.source}: "
            f"strategy={strategy.value}, action={action.value}"
        )

        if strategy == FeedbackStrategy.ESCALATE:
            if self._on_escalate:
                self._on_escalate(event)
            event.handled = True
            event.resolution = "escalated"
            return False

        if strategy == FeedbackStrategy.IGNORE:
            event.handled = True
            event.resolution = "ignored"
            return True

        if strategy in (FeedbackStrategy.AUTO_FIX, FeedbackStrategy.ANALYZE_FIX):
            # Get fix template if available
            fix_template = ""
            if event.pattern_name:
                pattern = self.get_pattern(event.pattern_name)
                if pattern:
                    fix_template = pattern.fix_template

            # Request fix
            if self._on_fix_request:
                success = self._on_fix_request(event, fix_template)
                event.handled = success
                event.resolution = "fixed" if success else "fix_failed"
                return success

        if strategy == FeedbackStrategy.REPORT:
            event.handled = True
            event.resolution = "reported"
            return True

        if strategy == FeedbackStrategy.RETRY:
            event.handled = True
            event.resolution = "retry_requested"
            return True

        return False

    def handle_all(self) -> dict[str, int]:
        """Handle all unhandled events.

        Returns:
            Dictionary with counts of handled/failed events.
        """
        handled = 0
        failed = 0

        for event in self._events:
            if not event.handled:
                if self.handle_event(event):
                    handled += 1
                else:
                    failed += 1

        return {"handled": handled, "failed": failed}

    @property
    def events(self) -> list[FeedbackEvent]:
        """Get all events."""
        return self._events

    @property
    def unhandled_events(self) -> list[FeedbackEvent]:
        """Get unhandled events."""
        return [e for e in self._events if not e.handled]

    @property
    def has_unhandled_errors(self) -> bool:
        """Check if there are unhandled errors."""
        return len(self.unhandled_events) > 0

    def clear_events(self) -> None:
        """Clear all events."""
        self._events.clear()
        self._occurrence_counts.clear()
        self._total_errors = 0

    def summary(self) -> dict[str, Any]:
        """Generate a summary of feedback handling.

        Returns:
            Summary dictionary.
        """
        handled = sum(1 for e in self._events if e.handled)
        unhandled = len(self._events) - handled

        by_type: dict[str, int] = {}
        for event in self._events:
            key = event.pattern_name or "unknown"
            by_type[key] = by_type.get(key, 0) + 1

        return {
            "total_events": len(self._events),
            "handled": handled,
            "unhandled": unhandled,
            "by_pattern": by_type,
            "occurrence_counts": dict(self._occurrence_counts),
        }


def create_feedback_handler(
    config: dict[str, Any],
    on_fix_request: Callable[[FeedbackEvent, str], bool] | None = None,
) -> FeedbackHandler:
    """Factory function to create a feedback handler from configuration.

    Args:
        config: Configuration dictionary.
        on_fix_request: Callback for fix requests.

    Returns:
        Configured FeedbackHandler.
    """
    patterns = []
    for pattern_config in config.get("error_patterns", []):
        patterns.append(ErrorPattern.from_dict(pattern_config))

    return FeedbackHandler(
        patterns=patterns,
        on_fix_request=on_fix_request,
        max_total_errors=config.get("max_total_errors", 10),
        auto_fix_enabled=config.get("auto_fix_enabled", True),
    )
