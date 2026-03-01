"""Tests for action resolution strategies.

Tests ActionResolutionConfig, ActionResolutionStrategy, and ActionResolver
for correct behavior across all resolution strategies.
"""

from __future__ import annotations

import pytest

from alpha_zero.multiplayer.action_resolution import (
    ActionResolutionConfig,
    ActionResolutionStrategy,
    ActionResolver,
)

# =============================================================================
# ActionResolutionConfig tests
# =============================================================================


class TestActionResolutionConfig:
    """Tests for ActionResolutionConfig dataclass."""

    def test_defaults(self) -> None:
        """Default config uses independent strategy."""
        config = ActionResolutionConfig()
        assert config.strategy == ActionResolutionStrategy.INDEPENDENT.value
        assert config.allow_same_target is False

    def test_custom_strategy(self) -> None:
        """Custom strategy can be set."""
        config = ActionResolutionConfig(strategy="priority")
        assert config.strategy == "priority"

        config_random = ActionResolutionConfig(strategy="random")
        assert config_random.strategy == "random"

        config_sim = ActionResolutionConfig(strategy="simultaneous_apply")
        assert config_sim.strategy == "simultaneous_apply"

    def test_frozen(self) -> None:
        """Config is frozen and cannot be modified."""
        config = ActionResolutionConfig()
        with pytest.raises(AttributeError):
            config.strategy = "priority"  # type: ignore[misc]

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            ActionResolutionConfig(strategy="invalid_strategy")

    def test_allow_same_target_custom(self) -> None:
        """allow_same_target can be set to True."""
        config = ActionResolutionConfig(allow_same_target=True)
        assert config.allow_same_target is True


# =============================================================================
# ActionResolver tests
# =============================================================================


class TestActionResolver:
    """Tests for ActionResolver."""

    def test_independent_strategy_preserves_all(self) -> None:
        """Independent strategy preserves all actions, ordered by player ID."""
        resolver = ActionResolver(
            ActionResolutionConfig(strategy="independent")
        )
        actions = {0: 5, 1: 10, 2: 3}
        result = resolver.resolve(actions)

        assert len(result) == 3
        assert result == [(0, 5), (1, 10), (2, 3)]

    def test_priority_strategy_orders_by_player_id(self) -> None:
        """Priority strategy orders by player ID (lower = higher priority)."""
        resolver = ActionResolver(
            ActionResolutionConfig(strategy="priority")
        )
        actions = {2: 3, 0: 5, 1: 10}
        result = resolver.resolve(actions)

        assert result[0] == (0, 5)  # Player 0 first (highest priority)
        assert result[1] == (1, 10)
        assert result[2] == (2, 3)

    def test_random_strategy_all_included(self) -> None:
        """Random strategy includes all actions (order varies)."""
        resolver = ActionResolver(
            ActionResolutionConfig(strategy="random")
        )
        actions = {0: 5, 1: 10, 2: 3}
        result = resolver.resolve(actions)

        assert len(result) == 3
        # All players should be present
        player_ids = {pid for pid, _ in result}
        assert player_ids == {0, 1, 2}
        # All actions should be present
        action_set = {action for _, action in result}
        assert action_set == {5, 10, 3}

    def test_simultaneous_apply_cancels_conflicts(self) -> None:
        """Simultaneous apply cancels conflicting actions."""
        resolver = ActionResolver(
            ActionResolutionConfig(strategy="simultaneous_apply")
        )
        # Players 0 and 1 target same tile (action 5)
        actions = {0: 5, 1: 5, 2: 3}
        result = resolver.resolve(actions)

        # Only player 2's action survives (no conflict)
        assert len(result) == 1
        assert result[0] == (2, 3)

    def test_simultaneous_apply_with_target_tiles(self) -> None:
        """Simultaneous apply uses target_tiles mapping for conflict detection."""
        resolver = ActionResolver(
            ActionResolutionConfig(strategy="simultaneous_apply")
        )
        # Different action indices but same target tile
        actions = {0: 5, 1: 10, 2: 3}
        env_state = {"target_tiles": {5: 42, 10: 42, 3: 99}}
        result = resolver.resolve(actions, env_state)

        # Players 0 and 1 both target tile 42, so they cancel
        assert len(result) == 1
        assert result[0] == (2, 3)

    def test_detect_conflicts_none(self) -> None:
        """No conflicts when all actions target different tiles."""
        resolver = ActionResolver()
        actions = {0: 5, 1: 10, 2: 3}
        conflicts = resolver.detect_conflicts(actions)
        assert conflicts == []

    def test_detect_conflicts_same_target(self) -> None:
        """Detect conflicts when players target the same action."""
        resolver = ActionResolver()
        actions = {0: 5, 1: 5}
        conflicts = resolver.detect_conflicts(actions)
        assert len(conflicts) == 1
        assert conflicts[0] == (0, 1)

    def test_detect_conflicts_with_target_tiles(self) -> None:
        """Detect conflicts via target_tiles mapping."""
        resolver = ActionResolver()
        actions = {0: 1, 1: 2, 2: 3}
        env_state = {"target_tiles": {1: 42, 2: 42, 3: 99}}
        conflicts = resolver.detect_conflicts(actions, env_state)
        assert len(conflicts) == 1
        assert conflicts[0] == (0, 1)

    def test_no_conflict_different_targets(self) -> None:
        """No conflicts when all targets are different."""
        resolver = ActionResolver()
        actions = {0: 1, 1: 2, 2: 3}
        env_state = {"target_tiles": {1: 10, 2: 20, 3: 30}}
        conflicts = resolver.detect_conflicts(actions, env_state)
        assert conflicts == []

    def test_empty_actions(self) -> None:
        """Empty actions dict returns empty list."""
        resolver = ActionResolver()
        result = resolver.resolve({})
        assert result == []

        conflicts = resolver.detect_conflicts({})
        assert conflicts == []

    def test_single_player_no_conflicts(self) -> None:
        """Single player action never conflicts."""
        resolver = ActionResolver(
            ActionResolutionConfig(strategy="simultaneous_apply")
        )
        actions = {0: 5}
        result = resolver.resolve(actions)
        assert result == [(0, 5)]

    def test_default_resolver_config(self) -> None:
        """ActionResolver with no config uses defaults."""
        resolver = ActionResolver()
        assert resolver.config.strategy == "independent"
        assert resolver.config.allow_same_target is False

    def test_three_way_conflict(self) -> None:
        """Three-way conflict cancels all three players."""
        resolver = ActionResolver(
            ActionResolutionConfig(strategy="simultaneous_apply")
        )
        actions = {0: 5, 1: 5, 2: 5, 3: 10}
        result = resolver.resolve(actions)

        # Players 0, 1, 2 all conflict; only player 3 survives
        assert len(result) == 1
        assert result[0] == (3, 10)

    def test_multiple_conflict_groups(self) -> None:
        """Multiple independent conflict groups."""
        resolver = ActionResolver(
            ActionResolutionConfig(strategy="simultaneous_apply")
        )
        actions = {0: 5, 1: 5, 2: 10, 3: 10}
        result = resolver.resolve(actions)
        # Both conflict groups cancel
        assert len(result) == 0

    def test_priority_drops_lower_priority_on_conflict(self) -> None:
        """Priority strategy drops lower-priority (higher ID) player on conflict."""
        resolver = ActionResolver(
            ActionResolutionConfig(strategy="priority", allow_same_target=False)
        )
        # Players 0 and 1 target same action
        actions = {0: 5, 1: 5, 2: 3}
        result = resolver.resolve(actions)

        # Player 1 is lower priority (higher ID) so gets dropped
        assert len(result) == 2
        assert result[0] == (0, 5)
        assert result[1] == (2, 3)

    def test_priority_with_allow_same_target(self) -> None:
        """Priority with allow_same_target=True keeps all actions."""
        resolver = ActionResolver(
            ActionResolutionConfig(strategy="priority", allow_same_target=True)
        )
        actions = {0: 5, 1: 5, 2: 3}
        result = resolver.resolve(actions)

        # allow_same_target=True means no dropping
        assert len(result) == 3
        assert result == [(0, 5), (1, 5), (2, 3)]

    def test_priority_no_conflict_preserves_all(self) -> None:
        """Priority without conflicts preserves all actions."""
        resolver = ActionResolver(
            ActionResolutionConfig(strategy="priority", allow_same_target=False)
        )
        actions = {0: 1, 1: 2, 2: 3}
        result = resolver.resolve(actions)

        assert len(result) == 3
        assert result == [(0, 1), (1, 2), (2, 3)]

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy in resolve raises ValueError."""
        resolver = ActionResolver(ActionResolutionConfig(strategy="independent"))
        # Bypass normal construction to test the ValueError branch
        resolver._strategy = "not_a_strategy"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown strategy"):
            resolver.resolve({0: 5})
