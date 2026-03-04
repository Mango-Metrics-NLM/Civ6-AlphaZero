"""Action resolution strategies for simultaneous multiplayer games.

When multiple players act simultaneously, their actions may conflict
(e.g., two units targeting the same tile). This module provides
configurable strategies for resolving such conflicts.

Strategies:
- INDEPENDENT: Actions don't conflict (default for turn-based games).
- PRIORITY: Higher-priority player's action resolves first.
- RANDOM: Random order resolution.
- SIMULTANEOUS_APPLY: All actions apply simultaneously; conflicts cancel.

Example:
    >>> resolver = ActionResolver(ActionResolutionConfig(strategy="priority"))
    >>> ordered = resolver.resolve({0: 5, 1: 5}, env_state={"target_tiles": {5: 3}})
    >>> ordered[0]  # Higher-priority player goes first
    (0, 5)
"""

from __future__ import annotations

import enum
import logging
import random
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class ActionResolutionStrategy(enum.Enum):
    """Strategy for resolving simultaneous action conflicts.

    Attributes:
        INDEPENDENT: Actions don't conflict (default for turn-based).
        PRIORITY: Higher-priority player's action resolves first.
        RANDOM: Random order resolution.
        SIMULTANEOUS_APPLY: All actions apply simultaneously, conflicts cancel.
    """

    INDEPENDENT = "independent"
    PRIORITY = "priority"
    RANDOM = "random"
    SIMULTANEOUS_APPLY = "simultaneous_apply"


@dataclass(frozen=True)
class ActionResolutionConfig:
    """Configuration for action resolution.

    Attributes:
        strategy: Resolution strategy name (must match ActionResolutionStrategy value).
        allow_same_target: Whether two players can target the same tile.
    """

    strategy: str = ActionResolutionStrategy.INDEPENDENT.value
    allow_same_target: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        valid_strategies = {s.value for s in ActionResolutionStrategy}
        if self.strategy not in valid_strategies:
            msg = (
                f"strategy must be one of {valid_strategies}, got {self.strategy!r}"
            )
            raise ValueError(msg)


class ActionResolver:
    """Resolves simultaneous actions from multiple players.

    Given a dict mapping player_id -> action, produces an ordered list
    of (player_id, action) pairs according to the configured strategy.

    Args:
        config: Resolution configuration. Defaults to independent strategy.
    """

    def __init__(self, config: ActionResolutionConfig | None = None) -> None:
        self.config = config or ActionResolutionConfig()
        self._strategy = ActionResolutionStrategy(self.config.strategy)

        logger.debug(
            "ActionResolver initialized",
            extra={
                "strategy": self._strategy.value,
                "allow_same_target": self.config.allow_same_target,
            },
        )

    def resolve(
        self,
        actions: dict[int, int],
        env_state: dict[str, Any] | None = None,
    ) -> list[tuple[int, int]]:
        """Resolve simultaneous actions into an ordered list.

        Args:
            actions: Mapping of player_id -> action index.
            env_state: Optional environment state for context-aware resolution.
                May contain 'target_tiles' mapping action -> tile index.

        Returns:
            Ordered list of (player_id, action) pairs.
        """
        if not actions:
            return []

        env_state = env_state or {}

        if self._strategy == ActionResolutionStrategy.INDEPENDENT:
            result = self._resolve_independent(actions)
        elif self._strategy == ActionResolutionStrategy.PRIORITY:
            result = self._resolve_priority(actions, env_state)
        elif self._strategy == ActionResolutionStrategy.RANDOM:
            result = self._resolve_random(actions)
        elif self._strategy == ActionResolutionStrategy.SIMULTANEOUS_APPLY:
            result = self._resolve_simultaneous(actions, env_state)
        else:
            # Should be unreachable due to __post_init__ validation
            msg = f"Unknown strategy: {self._strategy}"
            raise ValueError(msg)

        logger.debug(
            "Actions resolved",
            extra={
                "strategy": self._strategy.value,
                "input_count": len(actions),
                "output_count": len(result),
            },
        )
        return result

    def detect_conflicts(
        self,
        actions: dict[int, int],
        env_state: dict[str, Any] | None = None,
    ) -> list[tuple[int, int]]:
        """Detect conflicting player pairs.

        Two players conflict if they target the same tile. Tile targets
        are determined from env_state['target_tiles'] (action -> tile),
        or by treating the action index itself as the target.

        Args:
            actions: Mapping of player_id -> action index.
            env_state: Optional state containing 'target_tiles' mapping.

        Returns:
            List of (player_i, player_j) pairs that conflict.
        """
        if not actions:
            return []

        env_state = env_state or {}
        target_tiles = env_state.get("target_tiles", {})

        # Map each player to its target tile
        player_targets: dict[int, int] = {}
        for player_id, action in actions.items():
            # Use target_tiles mapping if available, else action is the target
            target = target_tiles.get(action, action)
            player_targets[player_id] = target

        # Find players targeting the same tile
        conflicts: list[tuple[int, int]] = []
        players = sorted(player_targets.keys())
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                if player_targets[players[i]] == player_targets[players[j]]:
                    conflicts.append((players[i], players[j]))

        return conflicts

    def _resolve_independent(
        self,
        actions: dict[int, int],
    ) -> list[tuple[int, int]]:
        """Independent: preserve all actions, ordered by player ID.

        No conflict handling — all actions are applied regardless
        of whether they target the same tile.
        """
        return [(pid, action) for pid, action in sorted(actions.items())]

    def _resolve_priority(
        self,
        actions: dict[int, int],
        env_state: dict[str, Any],
    ) -> list[tuple[int, int]]:
        """Priority: lower player ID has higher priority.

        When allow_same_target is False, conflicting lower-priority
        players have their actions dropped. Actions are ordered by
        player ID (lowest = highest priority).
        """
        if self.config.allow_same_target:
            return [(pid, action) for pid, action in sorted(actions.items())]

        conflicts = self.detect_conflicts(actions, env_state)
        if not conflicts:
            return [(pid, action) for pid, action in sorted(actions.items())]

        # Identify players whose actions should be dropped (higher ID loses)
        dropped: set[int] = set()
        for _p1, p2 in conflicts:
            # _p1 < p2 (sorted by detect_conflicts), so p2 is lower priority
            dropped.add(p2)

        result = [
            (pid, action) for pid, action in sorted(actions.items())
            if pid not in dropped
        ]

        if dropped:
            logger.debug(
                "Priority resolution dropped %d actions",
                len(dropped),
                extra={"dropped_players": sorted(dropped)},
            )

        return result

    def _resolve_random(
        self,
        actions: dict[int, int],
    ) -> list[tuple[int, int]]:
        """Random: shuffle action order randomly."""
        items = list(actions.items())
        random.shuffle(items)
        return items

    def _resolve_simultaneous(
        self,
        actions: dict[int, int],
        env_state: dict[str, Any],
    ) -> list[tuple[int, int]]:
        """Simultaneous apply: conflicts cancel both actions."""
        conflicts = self.detect_conflicts(actions, env_state)
        conflicting_players: set[int] = set()
        for p1, p2 in conflicts:
            conflicting_players.add(p1)
            conflicting_players.add(p2)

        # Return non-conflicting actions, ordered by player ID
        result: list[tuple[int, int]] = []
        for pid, action in sorted(actions.items()):
            if pid not in conflicting_players:
                result.append((pid, action))

        logger.debug(
            "Simultaneous resolution",
            extra={
                "total_actions": len(actions),
                "conflicts": len(conflicts),
                "resolved_actions": len(result),
            },
        )

        return result
