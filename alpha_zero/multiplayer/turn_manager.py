"""Turn management for multiplayer games.

Handles sequential, simultaneous, and dynamic (hybrid) turn modes.
Provides ordering, timeout tracking, and war-state detection for
dynamic mode transitions.

Example:
    >>> manager = TurnManager(TurnManagerConfig(mode="dynamic"), num_players=4)
    >>> manager.begin_turn()
    >>> for player in manager.get_active_players():
    ...     # Process player's actions
    ...     pass
    >>> manager.end_turn()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from alpha_zero.multiplayer.config import TurnManagerConfig

logger = logging.getLogger(__name__)


@dataclass
class TurnState:
    """Snapshot of a single turn's state.

    Attributes:
        turn_number: Global turn counter.
        active_players: Players who can act this turn.
        actions_submitted: Actions submitted by each player.
        start_time: When this turn began (for timeout).
        mode_override: If set, overrides default turn mode for this turn.
    """

    turn_number: int = 0
    active_players: list[int] = field(default_factory=list)
    actions_submitted: dict[int, int | None] = field(default_factory=dict)
    start_time: float = 0.0
    mode_override: str | None = None


class TurnManager:
    """Manages turn ordering and transitions for N-player games.

    Supports three modes:
    - Sequential: One player acts per step, rotating through all players.
    - Simultaneous: All players submit actions concurrently.
    - Dynamic: Simultaneous during peace, switches to sequential during war.

    Attributes:
        num_players: Number of players.
        mode: Current turn mode.
        turn_number: Current turn number.
    """

    def __init__(
        self,
        config: TurnManagerConfig,
        num_players: int,
    ) -> None:
        """Initialize turn manager.

        Args:
            config: TurnManagerConfig instance.
            num_players: Number of players in the game.
        """
        self._config = config
        self._num_players = num_players
        self._turn_number = 0
        self._current_mode: str = config.mode
        self._current_player_idx = 0
        self._active_players: list[int] = list(range(num_players))
        self._eliminated: set[int] = set()
        self._war_pairs: set[frozenset[int]] = set()
        self._turn_state: TurnState | None = None
        self._turn_history: deque[TurnState] = deque(maxlen=100)

        logger.debug(
            "TurnManager initialized",
            extra={
                "num_players": num_players,
                "mode": self._current_mode,
                "timeout": config.turn_timeout_seconds,
            },
        )

    @property
    def num_players(self) -> int:
        """Total number of players."""
        return self._num_players

    @property
    def mode(self) -> str:
        """Current turn mode."""
        return self._current_mode

    @property
    def turn_number(self) -> int:
        """Current global turn number."""
        return self._turn_number

    @property
    def active_player_count(self) -> int:
        """Number of non-eliminated players."""
        return len(self._active_players)

    def begin_turn(self) -> TurnState:
        """Begin a new turn.

        Determines which players are active based on the current mode
        and war state. Returns a TurnState describing this turn.

        Returns:
            TurnState for the new turn.
        """
        self._turn_number += 1

        # Determine active players based on mode
        active = self._determine_active_players()

        state = TurnState(
            turn_number=self._turn_number,
            active_players=active,
            actions_submitted=dict.fromkeys(active),
            start_time=time.monotonic(),
        )
        self._turn_state = state

        logger.debug(
            "Turn begun",
            extra={
                "turn": self._turn_number,
                "mode": self._current_mode,
                "active_players": active,
            },
        )

        return state

    def end_turn(self) -> None:
        """End the current turn and advance state."""
        if self._turn_state is not None:
            self._turn_history.append(self._turn_state)
        self._turn_state = None

        # Advance sequential player index
        if self._current_mode == "sequential":
            self._advance_sequential_player()

    def get_active_players(self) -> list[int]:
        """Get list of players who can act this turn.

        Returns:
            List of active player IDs.
        """
        if self._turn_state is not None:
            return self._turn_state.active_players
        return self._determine_active_players()

    def get_current_sequential_player(self) -> int:
        """Get the current player in sequential mode.

        Returns:
            Player ID of the current sequential player.
        """
        if not self._active_players:
            return 0
        idx = self._current_player_idx % len(self._active_players)
        return self._active_players[idx]

    def submit_action(self, player_id: int, action: int) -> bool:
        """Record a player's action for this turn.

        Args:
            player_id: Player submitting the action.
            action: Action index chosen.

        Returns:
            True if action was accepted, False if player can't act.
        """
        if self._turn_state is None:
            logger.warning("submit_action called outside of a turn")
            return False

        if player_id not in self._turn_state.actions_submitted:
            logger.warning(
                "Player %d not active this turn (active: %s)",
                player_id,
                self._turn_state.active_players,
            )
            return False

        self._turn_state.actions_submitted[player_id] = action
        return True

    def all_actions_submitted(self) -> bool:
        """Check if all active players have submitted actions.

        Returns:
            True if all active players have acted.
        """
        if self._turn_state is None:
            return True
        return all(a is not None for a in self._turn_state.actions_submitted.values())

    def is_turn_timed_out(self) -> bool:
        """Check if the current turn has exceeded timeout.

        Returns:
            True if timeout is enabled and exceeded.
        """
        timeout = self._config.turn_timeout_seconds
        if timeout <= 0 or self._turn_state is None:
            return False
        elapsed = time.monotonic() - self._turn_state.start_time
        return bool(elapsed > timeout)

    def eliminate_player(self, player_id: int) -> None:
        """Mark a player as eliminated.

        Args:
            player_id: Player to eliminate.
        """
        if player_id in self._eliminated:
            return

        self._eliminated.add(player_id)
        self._active_players = [p for p in range(self._num_players) if p not in self._eliminated]

        logger.info("Player %d eliminated", player_id)

    def declare_war(self, player_a: int, player_b: int) -> None:
        """Record a war declaration between two players.

        In dynamic mode, this causes transitions from simultaneous
        to sequential turns.

        Args:
            player_a: First player in the war.
            player_b: Second player in the war.
        """
        self._war_pairs.add(frozenset([player_a, player_b]))

        # Dynamic mode: switch to sequential if any wars active
        if self._config.mode == "dynamic" and self._war_pairs:
            old_mode = self._current_mode
            self._current_mode = "sequential"
            if old_mode != "sequential":
                logger.info(
                    "Dynamic mode: switching to sequential (war between %d and %d)",
                    player_a,
                    player_b,
                )

    def make_peace(self, player_a: int, player_b: int) -> None:
        """Record a peace agreement between two players.

        Args:
            player_a: First player.
            player_b: Second player.
        """
        self._war_pairs.discard(frozenset([player_a, player_b]))

        # Dynamic mode: switch back to simultaneous if no wars
        if self._config.mode == "dynamic" and not self._war_pairs:
            old_mode = self._current_mode
            self._current_mode = "simultaneous"
            if old_mode != "simultaneous":
                logger.info("Dynamic mode: switching to simultaneous (all wars resolved)")

    def update_from_info(self, info: dict[str, Any]) -> None:
        """Update war/peace state from environment info dict.

        Looks for the configured war detection key in the info dict.

        Args:
            info: Environment info dict.
        """
        war_key = self._config.war_detection_key
        if war_key in info:
            war_data = info[war_key]
            if isinstance(war_data, dict):
                # Expected format: {(player_a, player_b): bool}
                for pair, at_war in war_data.items():
                    if isinstance(pair, (tuple, list)) and len(pair) == 2:
                        if at_war:
                            self.declare_war(pair[0], pair[1])
                        else:
                            self.make_peace(pair[0], pair[1])

    def reset(self) -> None:
        """Reset turn manager for a new game."""
        self._turn_number = 0
        self._current_player_idx = 0
        self._active_players = list(range(self._num_players))
        self._eliminated.clear()
        self._war_pairs.clear()
        self._turn_state = None
        self._turn_history.clear()
        self._current_mode = self._config.mode

    def _determine_active_players(self) -> list[int]:
        """Determine which players are active based on current mode.

        Returns:
            List of active player IDs.
        """
        if self._current_mode == "sequential":
            player = self.get_current_sequential_player()
            return [player]
        else:
            # Simultaneous or dynamic-simultaneous: all non-eliminated
            return list(self._active_players)

    def _advance_sequential_player(self) -> None:
        """Advance to the next player in sequential mode."""
        if not self._active_players:
            return
        self._current_player_idx = (self._current_player_idx + 1) % len(self._active_players)

    def get_state_summary(self) -> dict[str, Any]:
        """Get a summary of the current turn manager state.

        Returns:
            Dict with turn manager status information.
        """
        return {
            "turn_number": self._turn_number,
            "current_mode": self._current_mode,
            "config_mode": self._config.mode,
            "active_players": list(self._active_players),
            "eliminated": list(self._eliminated),
            "war_pairs": [list(p) for p in self._war_pairs],
            "current_sequential_player": self.get_current_sequential_player(),
        }
