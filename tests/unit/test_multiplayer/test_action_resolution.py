"""Comprehensive tests for alpha_zero.multiplayer.action_resolution.

Targets coverage of ActionResolutionConfig, ActionResolutionStrategy,
and ActionResolver from ~20% to 70%+.  Tests cover:
  - ActionResolutionStrategy enum values
  - ActionResolutionConfig construction, defaults, validation
  - ActionResolver construction with / without explicit config
  - Resolution strategies: independent, priority, random, simultaneous_apply
  - Conflict detection: no conflicts, same actions, target-tile mapping
  - Edge cases: empty actions, single player, many players, all-conflict
  - Priority: allow_same_target flag, lower-priority drops
  - Simultaneous: both parties cancelled, 3-way conflict, multiple groups
  - Random: seeded determinism check
"""

from __future__ import annotations

import random

import pytest

from alpha_zero.multiplayer.action_resolution import (
    ActionResolutionConfig,
    ActionResolutionStrategy,
    ActionResolver,
)

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture()
def independent_resolver() -> ActionResolver:
    return ActionResolver(ActionResolutionConfig(strategy="independent"))


@pytest.fixture()
def priority_resolver() -> ActionResolver:
    return ActionResolver(ActionResolutionConfig(strategy="priority"))


@pytest.fixture()
def priority_resolver_allow_same() -> ActionResolver:
    return ActionResolver(
        ActionResolutionConfig(strategy="priority", allow_same_target=True)
    )


@pytest.fixture()
def random_resolver() -> ActionResolver:
    return ActionResolver(ActionResolutionConfig(strategy="random"))


@pytest.fixture()
def simultaneous_resolver() -> ActionResolver:
    return ActionResolver(ActionResolutionConfig(strategy="simultaneous_apply"))


# =====================================================================
# ActionResolutionStrategy tests
# =====================================================================


class TestActionResolutionStrategy:
    """Tests for ActionResolutionStrategy enum."""

    def test_values(self) -> None:
        assert ActionResolutionStrategy.INDEPENDENT.value == "independent"
        assert ActionResolutionStrategy.PRIORITY.value == "priority"
        assert ActionResolutionStrategy.RANDOM.value == "random"
        assert ActionResolutionStrategy.SIMULTANEOUS_APPLY.value == "simultaneous_apply"

    def test_from_value(self) -> None:
        assert ActionResolutionStrategy("independent") == ActionResolutionStrategy.INDEPENDENT
        assert ActionResolutionStrategy("priority") == ActionResolutionStrategy.PRIORITY
        assert ActionResolutionStrategy("random") == ActionResolutionStrategy.RANDOM
        assert (
            ActionResolutionStrategy("simultaneous_apply")
            == ActionResolutionStrategy.SIMULTANEOUS_APPLY
        )

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ActionResolutionStrategy("not_a_strategy")

    def test_member_count(self) -> None:
        assert len(ActionResolutionStrategy) == 4


# =====================================================================
# ActionResolutionConfig tests
# =====================================================================


class TestActionResolutionConfig:
    """Tests for ActionResolutionConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = ActionResolutionConfig()
        assert cfg.strategy == "independent"
        assert cfg.allow_same_target is False

    def test_frozen(self) -> None:
        cfg = ActionResolutionConfig()
        with pytest.raises(AttributeError):
            cfg.strategy = "priority"  # type: ignore[misc]

    def test_all_valid_strategies(self) -> None:
        for s in ("independent", "priority", "random", "simultaneous_apply"):
            cfg = ActionResolutionConfig(strategy=s)
            assert cfg.strategy == s

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="strategy must be one of"):
            ActionResolutionConfig(strategy="bogus")

    def test_allow_same_target_true(self) -> None:
        cfg = ActionResolutionConfig(allow_same_target=True)
        assert cfg.allow_same_target is True

    def test_allow_same_target_default_false(self) -> None:
        cfg = ActionResolutionConfig()
        assert cfg.allow_same_target is False


# =====================================================================
# ActionResolver — construction
# =====================================================================


class TestActionResolverConstruction:
    """Tests for ActionResolver initialisation."""

    def test_default_config(self) -> None:
        resolver = ActionResolver()
        assert resolver.config.strategy == "independent"
        assert resolver.config.allow_same_target is False

    def test_explicit_config(self) -> None:
        cfg = ActionResolutionConfig(strategy="priority", allow_same_target=True)
        resolver = ActionResolver(cfg)
        assert resolver.config is cfg

    def test_strategy_stored(self) -> None:
        resolver = ActionResolver(ActionResolutionConfig(strategy="random"))
        assert resolver._strategy == ActionResolutionStrategy.RANDOM

    def test_none_config_uses_defaults(self) -> None:
        resolver = ActionResolver(None)
        assert resolver.config.strategy == "independent"


# =====================================================================
# ActionResolver — empty / single player
# =====================================================================


class TestEmptyAndSinglePlayer:
    """Tests for trivial action dicts."""

    def test_resolve_empty_actions(self, independent_resolver: ActionResolver) -> None:
        assert independent_resolver.resolve({}) == []

    def test_detect_conflicts_empty(self, independent_resolver: ActionResolver) -> None:
        assert independent_resolver.detect_conflicts({}) == []

    def test_single_player_independent(self, independent_resolver: ActionResolver) -> None:
        result = independent_resolver.resolve({0: 42})
        assert result == [(0, 42)]

    def test_single_player_priority(self, priority_resolver: ActionResolver) -> None:
        result = priority_resolver.resolve({0: 7})
        assert result == [(0, 7)]

    def test_single_player_simultaneous(self, simultaneous_resolver: ActionResolver) -> None:
        result = simultaneous_resolver.resolve({0: 5})
        assert result == [(0, 5)]

    def test_single_player_random(self, random_resolver: ActionResolver) -> None:
        result = random_resolver.resolve({0: 99})
        assert result == [(0, 99)]

    def test_single_player_no_conflicts(self, independent_resolver: ActionResolver) -> None:
        conflicts = independent_resolver.detect_conflicts({0: 5})
        assert conflicts == []


# =====================================================================
# ActionResolver — independent strategy
# =====================================================================


class TestIndependentStrategy:
    """Tests for the INDEPENDENT resolution strategy."""

    def test_preserves_all_actions(self, independent_resolver: ActionResolver) -> None:
        actions = {0: 10, 1: 20, 2: 30}
        result = independent_resolver.resolve(actions)
        assert result == [(0, 10), (1, 20), (2, 30)]

    def test_ordered_by_player_id(self, independent_resolver: ActionResolver) -> None:
        actions = {2: 30, 0: 10, 1: 20}
        result = independent_resolver.resolve(actions)
        assert [pid for pid, _ in result] == [0, 1, 2]

    def test_same_action_no_dropping(self, independent_resolver: ActionResolver) -> None:
        """Independent strategy doesn't drop even when actions are identical."""
        actions = {0: 5, 1: 5, 2: 5}
        result = independent_resolver.resolve(actions)
        assert len(result) == 3

    def test_many_players(self, independent_resolver: ActionResolver) -> None:
        actions = {i: i * 10 for i in range(8)}
        result = independent_resolver.resolve(actions)
        assert len(result) == 8
        assert result[0] == (0, 0)
        assert result[7] == (7, 70)


# =====================================================================
# ActionResolver — priority strategy
# =====================================================================


class TestPriorityStrategy:
    """Tests for the PRIORITY resolution strategy."""

    def test_no_conflicts_preserves_all(self, priority_resolver: ActionResolver) -> None:
        actions = {0: 1, 1: 2, 2: 3}
        result = priority_resolver.resolve(actions)
        assert len(result) == 3
        assert result == [(0, 1), (1, 2), (2, 3)]

    def test_conflict_drops_lower_priority(self, priority_resolver: ActionResolver) -> None:
        """When two players target the same action, higher-ID player is dropped."""
        actions = {0: 5, 1: 5, 2: 3}
        result = priority_resolver.resolve(actions)
        assert len(result) == 2
        assert (0, 5) in result
        assert (2, 3) in result
        assert not any(pid == 1 for pid, _ in result)

    def test_conflict_with_target_tiles(self, priority_resolver: ActionResolver) -> None:
        """Priority uses target_tiles to detect conflicts."""
        actions = {0: 1, 1: 2}
        env_state = {"target_tiles": {1: 99, 2: 99}}  # both target tile 99
        result = priority_resolver.resolve(actions, env_state)
        # Player 1 (lower priority) dropped
        assert len(result) == 1
        assert result[0] == (0, 1)

    def test_three_way_conflict(self, priority_resolver: ActionResolver) -> None:
        """Three-way conflict: all lower-priority players dropped."""
        actions = {0: 5, 1: 5, 2: 5}
        result = priority_resolver.resolve(actions)
        # Only player 0 (highest priority) survives
        assert len(result) == 1
        assert result[0] == (0, 5)

    def test_allow_same_target_keeps_all(
        self, priority_resolver_allow_same: ActionResolver
    ) -> None:
        """With allow_same_target=True, no actions are dropped."""
        actions = {0: 5, 1: 5, 2: 5}
        result = priority_resolver_allow_same.resolve(actions)
        assert len(result) == 3
        assert result == [(0, 5), (1, 5), (2, 5)]

    def test_multiple_conflict_groups(self, priority_resolver: ActionResolver) -> None:
        """Two independent conflict groups: each drops the lower-priority player."""
        actions = {0: 5, 1: 5, 2: 10, 3: 10}
        result = priority_resolver.resolve(actions)
        # Player 1 dropped (conflicts with 0), player 3 dropped (conflicts with 2)
        assert len(result) == 2
        pids = [pid for pid, _ in result]
        assert 0 in pids
        assert 2 in pids

    def test_no_conflict_with_env_state(self, priority_resolver: ActionResolver) -> None:
        """No conflict when target_tiles map to different tiles."""
        actions = {0: 1, 1: 2}
        env_state = {"target_tiles": {1: 10, 2: 20}}
        result = priority_resolver.resolve(actions, env_state)
        assert len(result) == 2


# =====================================================================
# ActionResolver — random strategy
# =====================================================================


class TestRandomStrategy:
    """Tests for the RANDOM resolution strategy."""

    def test_all_actions_present(self, random_resolver: ActionResolver) -> None:
        actions = {0: 10, 1: 20, 2: 30}
        random.seed(42)
        result = random_resolver.resolve(actions)
        pids = {pid for pid, _ in result}
        acts = {a for _, a in result}
        assert pids == {0, 1, 2}
        assert acts == {10, 20, 30}

    def test_single_element(self, random_resolver: ActionResolver) -> None:
        result = random_resolver.resolve({0: 99})
        assert result == [(0, 99)]

    def test_shuffled_order(self, random_resolver: ActionResolver) -> None:
        """Over many seeds, random should produce different orderings."""
        actions = {0: 10, 1: 20, 2: 30}
        orderings = set()
        for seed in range(100):
            random.seed(seed)
            result = random_resolver.resolve(dict(actions))
            orderings.add(tuple(pid for pid, _ in result))

        # With 100 seeds and 3! = 6 permutations, we should see > 1 ordering
        assert len(orderings) > 1

    def test_preserves_player_action_pairs(self, random_resolver: ActionResolver) -> None:
        """Random shuffles order but keeps (player, action) pairs intact."""
        actions = {0: 100, 1: 200, 2: 300}
        random.seed(0)
        result = random_resolver.resolve(actions)
        pair_set = set(result)
        assert (0, 100) in pair_set
        assert (1, 200) in pair_set
        assert (2, 300) in pair_set


# =====================================================================
# ActionResolver — simultaneous_apply strategy
# =====================================================================


class TestSimultaneousApplyStrategy:
    """Tests for the SIMULTANEOUS_APPLY resolution strategy."""

    def test_no_conflict_keeps_all(self, simultaneous_resolver: ActionResolver) -> None:
        actions = {0: 1, 1: 2, 2: 3}
        result = simultaneous_resolver.resolve(actions)
        assert len(result) == 3
        assert result == [(0, 1), (1, 2), (2, 3)]

    def test_two_player_conflict_cancels_both(
        self, simultaneous_resolver: ActionResolver
    ) -> None:
        actions = {0: 5, 1: 5}
        result = simultaneous_resolver.resolve(actions)
        assert result == []

    def test_two_player_conflict_with_survivor(
        self, simultaneous_resolver: ActionResolver
    ) -> None:
        actions = {0: 5, 1: 5, 2: 3}
        result = simultaneous_resolver.resolve(actions)
        assert len(result) == 1
        assert result[0] == (2, 3)

    def test_three_way_conflict(self, simultaneous_resolver: ActionResolver) -> None:
        actions = {0: 5, 1: 5, 2: 5, 3: 10}
        result = simultaneous_resolver.resolve(actions)
        assert len(result) == 1
        assert result[0] == (3, 10)

    def test_multiple_conflict_groups_all_cancelled(
        self, simultaneous_resolver: ActionResolver
    ) -> None:
        actions = {0: 5, 1: 5, 2: 10, 3: 10}
        result = simultaneous_resolver.resolve(actions)
        assert result == []

    def test_target_tiles_conflict_detection(
        self, simultaneous_resolver: ActionResolver
    ) -> None:
        """Different action indices but same target tile causes cancellation."""
        actions = {0: 1, 1: 2, 2: 3}
        env_state = {"target_tiles": {1: 42, 2: 42, 3: 99}}
        result = simultaneous_resolver.resolve(actions, env_state)
        assert len(result) == 1
        assert result[0] == (2, 3)

    def test_all_unique_targets(self, simultaneous_resolver: ActionResolver) -> None:
        actions = {0: 1, 1: 2, 2: 3, 3: 4}
        env_state = {"target_tiles": {1: 10, 2: 20, 3: 30, 4: 40}}
        result = simultaneous_resolver.resolve(actions, env_state)
        assert len(result) == 4

    def test_single_player_survives(self, simultaneous_resolver: ActionResolver) -> None:
        result = simultaneous_resolver.resolve({0: 42})
        assert result == [(0, 42)]

    def test_result_ordered_by_player_id(
        self, simultaneous_resolver: ActionResolver
    ) -> None:
        """Non-conflicting actions are ordered by player ID."""
        actions = {3: 30, 1: 10, 0: 0, 2: 20}
        result = simultaneous_resolver.resolve(actions)
        assert [pid for pid, _ in result] == [0, 1, 2, 3]


# =====================================================================
# ActionResolver — detect_conflicts
# =====================================================================


class TestDetectConflicts:
    """Tests for the detect_conflicts method."""

    def test_no_conflicts(self, independent_resolver: ActionResolver) -> None:
        conflicts = independent_resolver.detect_conflicts({0: 1, 1: 2, 2: 3})
        assert conflicts == []

    def test_two_player_same_action(self, independent_resolver: ActionResolver) -> None:
        conflicts = independent_resolver.detect_conflicts({0: 5, 1: 5})
        assert conflicts == [(0, 1)]

    def test_three_player_same_action(self, independent_resolver: ActionResolver) -> None:
        conflicts = independent_resolver.detect_conflicts({0: 5, 1: 5, 2: 5})
        assert len(conflicts) == 3  # (0,1), (0,2), (1,2)
        assert (0, 1) in conflicts
        assert (0, 2) in conflicts
        assert (1, 2) in conflicts

    def test_with_target_tiles_mapping(self, independent_resolver: ActionResolver) -> None:
        actions = {0: 1, 1: 2, 2: 3}
        env_state = {"target_tiles": {1: 99, 2: 99, 3: 50}}
        conflicts = independent_resolver.detect_conflicts(actions, env_state)
        assert len(conflicts) == 1
        assert conflicts[0] == (0, 1)

    def test_empty_actions(self, independent_resolver: ActionResolver) -> None:
        assert independent_resolver.detect_conflicts({}) == []

    def test_single_player_no_conflict(self, independent_resolver: ActionResolver) -> None:
        assert independent_resolver.detect_conflicts({0: 42}) == []

    def test_no_target_tiles_key(self, independent_resolver: ActionResolver) -> None:
        """Without target_tiles, actions themselves are used as targets."""
        conflicts = independent_resolver.detect_conflicts(
            {0: 5, 1: 5}, env_state={}
        )
        assert conflicts == [(0, 1)]

    def test_none_env_state(self, independent_resolver: ActionResolver) -> None:
        """None env_state is handled gracefully."""
        conflicts = independent_resolver.detect_conflicts({0: 5, 1: 5}, env_state=None)
        assert conflicts == [(0, 1)]

    def test_partial_target_tiles_fallback(
        self, independent_resolver: ActionResolver
    ) -> None:
        """Actions not in target_tiles fall back to action index as target."""
        actions = {0: 1, 1: 2}
        env_state = {"target_tiles": {1: 50}}  # Only action 1 is mapped
        conflicts = independent_resolver.detect_conflicts(actions, env_state)
        # Action 0 targets tile 50, action 1 targets tile 2 (unmapped)
        assert conflicts == []

    def test_conflicts_sorted_by_player_id(
        self, independent_resolver: ActionResolver
    ) -> None:
        """Conflict pairs are always (lower_id, higher_id)."""
        conflicts = independent_resolver.detect_conflicts({3: 5, 1: 5, 0: 5})
        for p1, p2 in conflicts:
            assert p1 < p2

    def test_multiple_independent_conflict_groups(
        self, independent_resolver: ActionResolver
    ) -> None:
        """Two separate conflict groups detected independently."""
        actions = {0: 5, 1: 5, 2: 10, 3: 10, 4: 99}
        conflicts = independent_resolver.detect_conflicts(actions)
        assert len(conflicts) == 2
        assert (0, 1) in conflicts
        assert (2, 3) in conflicts


# =====================================================================
# ActionResolver — invalid strategy branch
# =====================================================================


class TestInvalidStrategyBranch:
    """Test the unreachable ValueError branch in resolve()."""

    def test_monkey_patched_strategy_raises(self) -> None:
        """Bypassing validation to reach the else branch raises ValueError."""
        resolver = ActionResolver(ActionResolutionConfig(strategy="independent"))
        resolver._strategy = "garbage"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown strategy"):
            resolver.resolve({0: 5})


# =====================================================================
# ActionResolver — env_state handling
# =====================================================================


class TestEnvStateHandling:
    """Tests for how env_state is used across strategies."""

    def test_independent_ignores_env_state(
        self, independent_resolver: ActionResolver
    ) -> None:
        actions = {0: 5, 1: 5}
        result_no_state = independent_resolver.resolve(actions)
        result_with_state = independent_resolver.resolve(actions, {"target_tiles": {5: 99}})
        assert result_no_state == result_with_state

    def test_priority_uses_env_state_for_conflict(
        self, priority_resolver: ActionResolver
    ) -> None:
        """Priority strategy uses target_tiles for conflict detection."""
        actions = {0: 1, 1: 2}
        # Without env_state: different actions = no conflict
        result1 = priority_resolver.resolve(actions)
        assert len(result1) == 2

        # With env_state mapping both to same target tile
        result2 = priority_resolver.resolve(actions, {"target_tiles": {1: 50, 2: 50}})
        assert len(result2) == 1  # player 1 dropped

    def test_simultaneous_uses_env_state(
        self, simultaneous_resolver: ActionResolver
    ) -> None:
        actions = {0: 1, 1: 2}
        # Different actions, no conflict by default
        result1 = simultaneous_resolver.resolve(actions)
        assert len(result1) == 2

        # Map both to same target
        result2 = simultaneous_resolver.resolve(actions, {"target_tiles": {1: 7, 2: 7}})
        assert len(result2) == 0  # both cancelled


# =====================================================================
# Parametrized tests across strategies
# =====================================================================


@pytest.mark.parametrize(
    "strategy",
    ["independent", "priority", "random", "simultaneous_apply"],
)
class TestAllStrategies:
    """Tests that apply to every strategy."""

    def test_empty_actions_returns_empty(self, strategy: str) -> None:
        resolver = ActionResolver(ActionResolutionConfig(strategy=strategy))
        assert resolver.resolve({}) == []

    def test_single_player_returns_one(self, strategy: str) -> None:
        resolver = ActionResolver(ActionResolutionConfig(strategy=strategy))
        result = resolver.resolve({0: 42})
        assert len(result) == 1
        assert result[0][0] == 0
        assert result[0][1] == 42

    def test_no_exception_on_large_input(self, strategy: str) -> None:
        """Resolving many players completes without error."""
        resolver = ActionResolver(ActionResolutionConfig(strategy=strategy))
        actions = {i: i for i in range(20)}
        result = resolver.resolve(actions)
        assert len(result) > 0

    def test_result_contains_valid_pairs(self, strategy: str) -> None:
        resolver = ActionResolver(ActionResolutionConfig(strategy=strategy))
        actions = {0: 10, 1: 20, 2: 30}
        random.seed(42)
        result = resolver.resolve(actions)
        for pid, action in result:
            assert pid in actions
            assert actions[pid] == action
