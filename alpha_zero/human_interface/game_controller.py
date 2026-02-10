"""Game controller for Human vs AI games.

This module manages game flow, turn handling, and AI interaction.

Usage:
    from alpha_zero.human_interface.game_controller import GameController

    controller = GameController(config, game_interface)
    controller.start_game(human_player=0)

    while not controller.is_game_over:
        if controller.is_human_turn:
            action = get_human_action()
            result = controller.play_human_move(action)
        else:
            result = controller.play_ai_move()
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np

from alpha_zero.human_interface.config import HumanInterfaceConfig

logger = logging.getLogger(__name__)


class GamePhase(str, Enum):
    """Current phase of the game."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    HUMAN_TURN = "human_turn"
    AI_TURN = "ai_turn"
    PAUSED = "paused"
    GAME_OVER = "game_over"


@dataclass
class TurnResult:
    """Result of a turn.

    Attributes:
        action: Action taken
        player: Player who took action
        is_human: Whether player was human
        reward: Reward received
        game_over: Whether game ended
        winner: Winner if game ended
        state_before: State before action
        state_after: State after action
        ai_evaluation: AI's evaluation if applicable
        move_probabilities: Move probability distribution
        thinking_time: Time spent thinking
        timestamp: When turn occurred
    """

    action: int
    player: int
    is_human: bool
    reward: float = 0.0
    game_over: bool = False
    winner: int | None = None
    state_before: np.ndarray | None = None
    state_after: np.ndarray | None = None
    ai_evaluation: float | None = None
    move_probabilities: np.ndarray | None = None
    thinking_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action,
            "player": self.player,
            "is_human": self.is_human,
            "reward": self.reward,
            "game_over": self.game_over,
            "winner": self.winner,
            "ai_evaluation": self.ai_evaluation,
            "thinking_time": self.thinking_time,
            "timestamp": self.timestamp,
        }


@dataclass
class GameState:
    """Current game state for the controller.

    Attributes:
        turn_number: Current turn number
        current_player: Player whose turn it is
        human_player: Which player is human
        phase: Current game phase
        history: List of turn results
        start_time: When game started
        valid_actions: Currently valid actions
    """

    turn_number: int = 0
    current_player: int = 0
    human_player: int = 0
    phase: GamePhase = GamePhase.NOT_STARTED
    history: list[TurnResult] = field(default_factory=list)
    start_time: str | None = None
    valid_actions: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_number": self.turn_number,
            "current_player": self.current_player,
            "human_player": self.human_player,
            "phase": self.phase.value,
            "history": [t.to_dict() for t in self.history],
            "start_time": self.start_time,
        }


class GameController:
    """Controller for Human vs AI games.

    This class manages the game flow, handling turns for both
    human and AI players.

    Example:
        >>> config = HumanInterfaceConfig()
        >>> controller = GameController(config, game_interface, ai_agent)
        >>>
        >>> controller.start_game(human_player=0)
        >>>
        >>> while not controller.is_game_over:
        ...     if controller.is_human_turn:
        ...         action = get_human_input()
        ...         result = controller.play_human_move(action)
        ...     else:
        ...         result = controller.play_ai_move()
        ...     print(f"Turn {result.player}: Action {result.action}")
    """

    def __init__(
        self,
        config: HumanInterfaceConfig | None = None,
        game_interface: Any | None = None,
        ai_agent: Any | None = None,
    ) -> None:
        """Initialize game controller.

        Args:
            config: Human interface configuration
            game_interface: Game interface for executing actions
            ai_agent: AI agent for playing moves
        """
        self.config = config or HumanInterfaceConfig()
        self._game_interface = game_interface
        self._ai_agent = ai_agent

        self._state = GameState()
        self._observation: np.ndarray | None = None
        self._undo_stack: list[tuple[np.ndarray, TurnResult]] = []

        # Callbacks
        self._on_turn_complete: Callable[[TurnResult], None] | None = None
        self._on_game_over: Callable[[int], None] | None = None
        self._on_ai_thinking: Callable[[float], None] | None = None

        logger.info("GameController initialized")

    def set_game_interface(self, game_interface: Any) -> None:
        """Set the game interface.

        Args:
            game_interface: Game interface for executing actions
        """
        self._game_interface = game_interface

    def set_ai_agent(self, ai_agent: Any) -> None:
        """Set the AI agent.

        Args:
            ai_agent: AI agent for playing moves
        """
        self._ai_agent = ai_agent

    def start_game(
        self,
        human_player: int = 0,
        seed: int | None = None,
    ) -> GameState:
        """Start a new game.

        Args:
            human_player: Which player is human (0 or 1)
            seed: Random seed for reproducibility

        Returns:
            Initial game state.
        """
        if self._game_interface is None:
            raise RuntimeError("Game interface not set")

        # Reset game
        self._observation = self._game_interface.reset(seed)
        self._state = GameState(
            turn_number=0,
            current_player=0,
            human_player=human_player,
            phase=GamePhase.IN_PROGRESS,
            start_time=datetime.now(UTC).isoformat(),
        )
        self._undo_stack.clear()

        # Update valid actions
        self._update_valid_actions()

        # Set correct phase based on whose turn it is
        if self._state.current_player == human_player:
            self._state.phase = GamePhase.HUMAN_TURN
        else:
            self._state.phase = GamePhase.AI_TURN

        logger.info(f"Game started. Human is player {human_player}")
        return self._state

    @property
    def is_game_over(self) -> bool:
        """Check if game has ended."""
        return self._state.phase == GamePhase.GAME_OVER

    @property
    def is_human_turn(self) -> bool:
        """Check if it's the human's turn."""
        return self._state.phase == GamePhase.HUMAN_TURN or (
            self._state.phase == GamePhase.IN_PROGRESS
            and self._state.current_player == self._state.human_player
        )

    @property
    def current_state(self) -> GameState:
        """Get current game state."""
        return self._state

    @property
    def current_observation(self) -> np.ndarray | None:
        """Get current game observation."""
        return self._observation

    @property
    def valid_actions(self) -> list[int]:
        """Get list of valid actions."""
        return self._state.valid_actions

    def get_valid_actions(self) -> list[int]:
        """Get list of valid actions for current player.

        Returns:
            List of valid action indices.
        """
        return self._state.valid_actions

    def _update_valid_actions(self) -> None:
        """Update the list of valid actions."""
        if self._game_interface is not None:
            if hasattr(self._game_interface, "get_valid_actions"):
                self._state.valid_actions = self._game_interface.get_valid_actions()
            elif hasattr(self._game_interface, "valid_actions"):
                self._state.valid_actions = list(self._game_interface.valid_actions)
            else:
                # Default: all actions valid
                self._state.valid_actions = list(range(self._game_interface.action_space_size))

    def is_action_valid(self, action: int) -> bool:
        """Check if an action is valid.

        Args:
            action: Action index to check

        Returns:
            True if action is valid.
        """
        return action in self._state.valid_actions

    def play_human_move(self, action: int) -> TurnResult:
        """Play a move for the human player.

        Args:
            action: Action index to play

        Returns:
            TurnResult for the move.

        Raises:
            ValueError: If action is invalid
            RuntimeError: If not human's turn
        """
        if not self.is_human_turn:
            raise RuntimeError("Not human's turn")

        if not self.is_action_valid(action):
            raise ValueError(f"Invalid action: {action}")

        return self._execute_move(action, is_human=True)

    def play_ai_move(
        self,
        thinking_time: float | None = None,
    ) -> TurnResult:
        """Let the AI play its move.

        Args:
            thinking_time: Override AI thinking time

        Returns:
            TurnResult for the move.

        Raises:
            RuntimeError: If it's human's turn or AI not set
        """
        if self.is_human_turn:
            raise RuntimeError("Human's turn, not AI's")

        if self._ai_agent is None:
            raise RuntimeError("AI agent not set")

        thinking_time = thinking_time or self.config.thinking_time_seconds

        # Notify about AI thinking
        if self._on_ai_thinking:
            self._on_ai_thinking(thinking_time)

        start_time = time.time()

        # Get AI's action
        if hasattr(self._ai_agent, "select_action"):
            result = self._ai_agent.select_action(
                self._observation,
                valid_actions=self._state.valid_actions,
            )
            action = result["action"]
            ai_evaluation = result.get("value")
            move_probs = result.get("policy")
        elif hasattr(self._ai_agent, "get_action"):
            action = self._ai_agent.get_action(self._observation)
            ai_evaluation = None
            move_probs = None
        else:
            raise RuntimeError("AI agent has no action selection method")

        elapsed = time.time() - start_time

        return self._execute_move(
            action,
            is_human=False,
            ai_evaluation=ai_evaluation,
            move_probabilities=move_probs,
            thinking_time=elapsed,
        )

    def _execute_move(
        self,
        action: int,
        is_human: bool,
        ai_evaluation: float | None = None,
        move_probabilities: np.ndarray | None = None,
        thinking_time: float = 0.0,
    ) -> TurnResult:
        """Execute a move and update game state.

        Args:
            action: Action to execute
            is_human: Whether move is from human
            ai_evaluation: AI's position evaluation
            move_probabilities: Move probability distribution
            thinking_time: Time spent on move

        Returns:
            TurnResult for the move.
        """
        state_before = self._observation.copy() if self._observation is not None else None
        player = self._state.current_player

        # Execute action
        obs, reward, done, truncated, info = self._game_interface.step(action)

        # Create turn result
        result = TurnResult(
            action=action,
            player=player,
            is_human=is_human,
            reward=reward,
            game_over=done or truncated,
            winner=info.get("winner"),
            state_before=state_before,
            state_after=obs.copy() if obs is not None else None,
            ai_evaluation=ai_evaluation,
            move_probabilities=move_probabilities,
            thinking_time=thinking_time,
        )

        # Save for undo
        if self.config.enable_undo and state_before is not None:
            self._undo_stack.append((state_before, result))
            if len(self._undo_stack) > self.config.max_undo_depth:
                self._undo_stack.pop(0)

        # Update state
        self._observation = obs
        self._state.turn_number += 1
        self._state.history.append(result)

        if result.game_over:
            self._state.phase = GamePhase.GAME_OVER
            logger.info(f"Game over. Winner: {result.winner}")
            if self._on_game_over:
                self._on_game_over(result.winner)
        else:
            # Switch player
            self._state.current_player = 1 - self._state.current_player
            self._update_valid_actions()

            if self._state.current_player == self._state.human_player:
                self._state.phase = GamePhase.HUMAN_TURN
            else:
                self._state.phase = GamePhase.AI_TURN

        # Fire callback
        if self._on_turn_complete:
            self._on_turn_complete(result)

        return result

    def undo_move(self) -> bool:
        """Undo the last move.

        Returns:
            True if undo was successful.
        """
        if not self.config.enable_undo:
            logger.warning("Undo is disabled")
            return False

        if not self._undo_stack:
            logger.warning("No moves to undo")
            return False

        state_before, last_result = self._undo_stack.pop()

        # Restore state
        self._observation = state_before
        self._state.history.pop()
        self._state.turn_number -= 1
        self._state.current_player = last_result.player
        self._state.phase = (
            GamePhase.HUMAN_TURN
            if self._state.current_player == self._state.human_player
            else GamePhase.AI_TURN
        )
        self._update_valid_actions()

        # Restore game interface state if possible
        if hasattr(self._game_interface, "set_state"):
            self._game_interface.set_state(state_before)

        logger.debug(f"Undid move. Now at turn {self._state.turn_number}")
        return True

    def pause_game(self) -> None:
        """Pause the game."""
        if self._state.phase not in (GamePhase.NOT_STARTED, GamePhase.GAME_OVER):
            self._state.phase = GamePhase.PAUSED
            logger.info("Game paused")

    def resume_game(self) -> None:
        """Resume a paused game."""
        if self._state.phase == GamePhase.PAUSED:
            if self._state.current_player == self._state.human_player:
                self._state.phase = GamePhase.HUMAN_TURN
            else:
                self._state.phase = GamePhase.AI_TURN
            logger.info("Game resumed")

    def resign(self, player: int | None = None) -> TurnResult:
        """Resign the game.

        Args:
            player: Player who resigns (default: human)

        Returns:
            Final TurnResult.
        """
        player = player if player is not None else self._state.human_player
        winner = 1 - player

        result = TurnResult(
            action=-1,  # Resign
            player=player,
            is_human=(player == self._state.human_player),
            reward=-1.0,
            game_over=True,
            winner=winner,
        )

        self._state.history.append(result)
        self._state.phase = GamePhase.GAME_OVER

        logger.info(f"Player {player} resigned. Winner: {winner}")
        if self._on_game_over:
            self._on_game_over(winner)

        return result

    def set_on_turn_complete(self, callback: Callable[[TurnResult], None]) -> None:
        """Set callback for turn completion.

        Args:
            callback: Function to call with TurnResult
        """
        self._on_turn_complete = callback

    def set_on_game_over(self, callback: Callable[[int], None]) -> None:
        """Set callback for game over.

        Args:
            callback: Function to call with winner
        """
        self._on_game_over = callback

    def set_on_ai_thinking(self, callback: Callable[[float], None]) -> None:
        """Set callback for when AI starts thinking.

        Args:
            callback: Function to call with thinking time
        """
        self._on_ai_thinking = callback

    def get_game_summary(self) -> dict[str, Any]:
        """Get summary of the game.

        Returns:
            Dictionary with game summary.
        """
        human_moves = [t for t in self._state.history if t.is_human]
        ai_moves = [t for t in self._state.history if not t.is_human]

        return {
            "total_turns": self._state.turn_number,
            "human_moves": len(human_moves),
            "ai_moves": len(ai_moves),
            "human_player": self._state.human_player,
            "winner": self._state.history[-1].winner if self._state.history else None,
            "start_time": self._state.start_time,
            "phase": self._state.phase.value,
            "avg_human_time": (
                sum(t.thinking_time for t in human_moves) / len(human_moves) if human_moves else 0
            ),
            "avg_ai_time": (
                sum(t.thinking_time for t in ai_moves) / len(ai_moves) if ai_moves else 0
            ),
        }
