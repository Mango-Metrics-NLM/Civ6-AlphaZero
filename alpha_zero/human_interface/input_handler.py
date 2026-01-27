"""Input handling for Human vs AI games.

This module provides input parsing and validation for human players.

Usage:
    from alpha_zero.human_interface.input_handler import HumanInputHandler

    handler = HumanInputHandler(config, board_size=(10, 10))
    action = handler.get_action(valid_actions)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union

from alpha_zero.human_interface.config import HumanInterfaceConfig, InputMode

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Type of action parsed from input."""

    MOVE = "move"          # Regular game move
    UNDO = "undo"          # Undo last move
    RESIGN = "resign"      # Resign game
    PAUSE = "pause"        # Pause game
    SAVE = "save"          # Save game
    HELP = "help"          # Show help
    QUIT = "quit"          # Quit game
    INVALID = "invalid"    # Invalid input


@dataclass
class ParsedAction:
    """Result of parsing human input.

    Attributes:
        action_type: Type of action
        action_index: Game action index (for moves)
        coordinate: Human-readable coordinate (e.g., "A1")
        raw_input: Original input string
        error_message: Error message if invalid
        metadata: Additional action data
    """

    action_type: ActionType
    action_index: Optional[int] = None
    coordinate: Optional[str] = None
    raw_input: str = ""
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if action is valid."""
        return self.action_type != ActionType.INVALID

    @property
    def is_move(self) -> bool:
        """Check if action is a game move."""
        return self.action_type == ActionType.MOVE and self.action_index is not None


class ActionParser:
    """Parser for human input actions.

    This class converts various input formats into game actions.

    Example:
        >>> parser = ActionParser(board_size=(10, 10))
        >>>
        >>> # Coordinate input
        >>> result = parser.parse("A1")
        >>> print(result.action_index)  # 0
        >>>
        >>> # Index input
        >>> result = parser.parse("5")
        >>> print(result.action_index)  # 5
        >>>
        >>> # Special commands
        >>> result = parser.parse("undo")
        >>> print(result.action_type)  # ActionType.UNDO
    """

    # Special command patterns
    COMMANDS = {
        r"^(undo|u)$": ActionType.UNDO,
        r"^(resign|quit|q)$": ActionType.RESIGN,
        r"^(pause|p)$": ActionType.PAUSE,
        r"^(save|s)$": ActionType.SAVE,
        r"^(help|h|\?)$": ActionType.HELP,
        r"^(exit|x)$": ActionType.QUIT,
    }

    def __init__(
        self,
        board_size: tuple[int, int] = (10, 10),
        input_mode: InputMode = InputMode.COORDINATE,
    ):
        """Initialize action parser.

        Args:
            board_size: (rows, cols) of game board
            input_mode: How to interpret input
        """
        self.board_size = board_size
        self.input_mode = input_mode
        self._rows, self._cols = board_size

    def parse(
        self,
        input_str: str,
        valid_actions: Optional[list[int]] = None,
    ) -> ParsedAction:
        """Parse input string into an action.

        Args:
            input_str: Raw input from user
            valid_actions: List of valid action indices

        Returns:
            ParsedAction with parsed result.
        """
        raw = input_str.strip()
        normalized = raw.lower()

        # Check for special commands
        for pattern, action_type in self.COMMANDS.items():
            if re.match(pattern, normalized, re.IGNORECASE):
                return ParsedAction(
                    action_type=action_type,
                    raw_input=raw,
                )

        # Try to parse as move
        if self.input_mode == InputMode.INDEX:
            return self._parse_index(raw, valid_actions)
        else:
            return self._parse_coordinate(raw, valid_actions)

    def _parse_coordinate(
        self,
        input_str: str,
        valid_actions: Optional[list[int]] = None,
    ) -> ParsedAction:
        """Parse coordinate input (e.g., "A1", "B3").

        Args:
            input_str: Input string
            valid_actions: Valid action indices

        Returns:
            ParsedAction.
        """
        # Match patterns like A1, a1, A10, etc.
        match = re.match(r"^([A-Za-z])(\d+)$", input_str.strip())

        if not match:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_input=input_str,
                error_message=f"Invalid coordinate format. Use letter+number (e.g., A1, B3)",
            )

        col_letter = match.group(1).upper()
        row_num = int(match.group(2))

        # Convert to indices (A=0, B=1, etc. and 1-indexed rows)
        col = ord(col_letter) - ord('A')
        row = row_num - 1

        # Check bounds
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_input=input_str,
                coordinate=f"{col_letter}{row_num}",
                error_message=f"Coordinate out of bounds. Board is {self._cols}x{self._rows}",
            )

        # Convert to action index
        action_index = row * self._cols + col

        # Check if valid
        if valid_actions is not None and action_index not in valid_actions:
            return ParsedAction(
                action_type=ActionType.INVALID,
                action_index=action_index,
                coordinate=f"{col_letter}{row_num}",
                raw_input=input_str,
                error_message=f"Position {col_letter}{row_num} is not a valid move",
            )

        return ParsedAction(
            action_type=ActionType.MOVE,
            action_index=action_index,
            coordinate=f"{col_letter}{row_num}",
            raw_input=input_str,
        )

    def _parse_index(
        self,
        input_str: str,
        valid_actions: Optional[list[int]] = None,
    ) -> ParsedAction:
        """Parse index input (e.g., "0", "5").

        Args:
            input_str: Input string
            valid_actions: Valid action indices

        Returns:
            ParsedAction.
        """
        try:
            action_index = int(input_str.strip())
        except ValueError:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_input=input_str,
                error_message=f"Invalid index. Enter a number.",
            )

        # Check bounds
        total_actions = self._rows * self._cols
        if not (0 <= action_index < total_actions):
            return ParsedAction(
                action_type=ActionType.INVALID,
                action_index=action_index,
                raw_input=input_str,
                error_message=f"Index out of bounds. Valid range: 0-{total_actions - 1}",
            )

        # Check if valid
        if valid_actions is not None and action_index not in valid_actions:
            return ParsedAction(
                action_type=ActionType.INVALID,
                action_index=action_index,
                raw_input=input_str,
                error_message=f"Index {action_index} is not a valid move",
            )

        # Convert to coordinate for display
        row = action_index // self._cols
        col = action_index % self._cols
        coordinate = f"{chr(ord('A') + col)}{row + 1}"

        return ParsedAction(
            action_type=ActionType.MOVE,
            action_index=action_index,
            coordinate=coordinate,
            raw_input=input_str,
        )

    def index_to_coordinate(self, index: int) -> str:
        """Convert action index to coordinate string.

        Args:
            index: Action index

        Returns:
            Coordinate string (e.g., "A1").
        """
        row = index // self._cols
        col = index % self._cols
        return f"{chr(ord('A') + col)}{row + 1}"

    def coordinate_to_index(self, coord: str) -> Optional[int]:
        """Convert coordinate string to action index.

        Args:
            coord: Coordinate string (e.g., "A1")

        Returns:
            Action index or None if invalid.
        """
        result = self._parse_coordinate(coord)
        return result.action_index


class HumanInputHandler:
    """Handler for getting and validating human input.

    This class manages the input loop for human players,
    including prompts, validation, and confirmation.

    Example:
        >>> handler = HumanInputHandler(config, board_size=(10, 10))
        >>>
        >>> # Simple usage
        >>> action = handler.get_action(valid_actions)
        >>>
        >>> # With custom prompt
        >>> action = handler.get_action(valid_actions, prompt="Your move: ")
    """

    def __init__(
        self,
        config: Optional[HumanInterfaceConfig] = None,
        board_size: tuple[int, int] = (10, 10),
    ):
        """Initialize input handler.

        Args:
            config: Human interface configuration
            board_size: (rows, cols) of game board
        """
        self.config = config or HumanInterfaceConfig()
        self.board_size = board_size
        self._parser = ActionParser(
            board_size=board_size,
            input_mode=self.config.get_input_mode_enum(),
        )

        # Callbacks
        self._on_invalid_input: Optional[Callable[[str], None]] = None
        self._input_function: Callable[[str], str] = input

        logger.debug("HumanInputHandler initialized")

    def set_input_function(self, func: Callable[[str], str]) -> None:
        """Set custom input function (for testing).

        Args:
            func: Function that takes prompt and returns input string
        """
        self._input_function = func

    def set_on_invalid_input(
        self, callback: Callable[[str], None]
    ) -> None:
        """Set callback for invalid input.

        Args:
            callback: Function to call with error message
        """
        self._on_invalid_input = callback

    def get_action(
        self,
        valid_actions: list[int],
        prompt: str = "Enter move: ",
        timeout: Optional[float] = None,
    ) -> ParsedAction:
        """Get a valid action from the human player.

        This method loops until valid input is received or timeout occurs.

        Args:
            valid_actions: List of valid action indices
            prompt: Input prompt to display
            timeout: Override input timeout

        Returns:
            ParsedAction with validated action.
        """
        timeout = timeout or self.config.input_timeout_seconds
        start_time = time.time()

        while True:
            # Check timeout
            if time.time() - start_time > timeout:
                return ParsedAction(
                    action_type=ActionType.INVALID,
                    error_message="Input timeout",
                )

            try:
                raw_input = self._input_function(prompt)
            except (EOFError, KeyboardInterrupt):
                return ParsedAction(action_type=ActionType.QUIT)

            # Parse input
            result = self._parser.parse(raw_input, valid_actions)

            # Handle invalid moves
            if result.action_type == ActionType.INVALID:
                if self._on_invalid_input:
                    self._on_invalid_input(result.error_message or "Invalid input")
                else:
                    print(f"Invalid: {result.error_message}")
                continue

            # Handle special commands
            if result.action_type in (ActionType.UNDO, ActionType.RESIGN,
                                       ActionType.PAUSE, ActionType.SAVE,
                                       ActionType.HELP, ActionType.QUIT):
                return result

            # Confirm move if required
            if self.config.confirm_moves and result.is_move:
                if not self._confirm_move(result):
                    continue

            return result

    def _confirm_move(self, action: ParsedAction) -> bool:
        """Ask for move confirmation.

        Args:
            action: Parsed action to confirm

        Returns:
            True if confirmed.
        """
        try:
            response = self._input_function(
                f"Confirm {action.coordinate}? (y/n): "
            ).strip().lower()
            return response in ("y", "yes", "")
        except (EOFError, KeyboardInterrupt):
            return False

    def get_yes_no(
        self,
        prompt: str,
        default: bool = True,
    ) -> bool:
        """Get a yes/no response from user.

        Args:
            prompt: Question to ask
            default: Default value if empty input

        Returns:
            True for yes, False for no.
        """
        default_hint = "(Y/n)" if default else "(y/N)"

        try:
            response = self._input_function(f"{prompt} {default_hint}: ").strip().lower()

            if response == "":
                return default
            return response in ("y", "yes", "true", "1")
        except (EOFError, KeyboardInterrupt):
            return default

    def get_number(
        self,
        prompt: str,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        default: Optional[int] = None,
    ) -> Optional[int]:
        """Get a number from user.

        Args:
            prompt: Prompt to display
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            default: Default value

        Returns:
            Integer or None if cancelled.
        """
        while True:
            try:
                raw = self._input_function(prompt).strip()

                if raw == "" and default is not None:
                    return default

                value = int(raw)

                if min_val is not None and value < min_val:
                    print(f"Value must be at least {min_val}")
                    continue

                if max_val is not None and value > max_val:
                    print(f"Value must be at most {max_val}")
                    continue

                return value

            except ValueError:
                print("Please enter a valid number")
            except (EOFError, KeyboardInterrupt):
                return None

    def show_help(self) -> str:
        """Generate help text for input.

        Returns:
            Help text string.
        """
        lines = [
            "Commands:",
            "  A1, B3, etc.  - Make a move at coordinate",
            "  0, 1, 5, etc. - Make a move by index",
            "  undo (u)      - Undo last move",
            "  resign (q)    - Resign the game",
            "  pause (p)     - Pause the game",
            "  save (s)      - Save the game",
            "  help (h)      - Show this help",
            "  exit (x)      - Quit without saving",
        ]
        return "\n".join(lines)

    def show_valid_moves(
        self,
        valid_actions: list[int],
        max_display: int = 20,
    ) -> str:
        """Generate display of valid moves.

        Args:
            valid_actions: List of valid action indices
            max_display: Maximum moves to display

        Returns:
            String listing valid moves.
        """
        if not valid_actions:
            return "No valid moves available"

        coords = [self._parser.index_to_coordinate(a) for a in valid_actions[:max_display]]
        result = "Valid moves: " + ", ".join(coords)

        if len(valid_actions) > max_display:
            result += f" ... and {len(valid_actions) - max_display} more"

        return result
