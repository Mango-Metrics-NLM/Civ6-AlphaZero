"""State visualization for Human vs AI games.

This module provides ASCII visualization of game states.

Usage:
    from alpha_zero.human_interface.visualizer import StateVisualizer

    visualizer = StateVisualizer(config)
    print(visualizer.render_board(observation, valid_actions))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from alpha_zero.human_interface.config import BoardTheme, HumanInterfaceConfig

logger = logging.getLogger(__name__)


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


@dataclass
class ThemeColors:
    """Color scheme for a board theme."""

    player1: str = Colors.BLUE
    player2: str = Colors.RED
    empty: str = Colors.DIM
    valid_move: str = Colors.GREEN
    last_move: str = Colors.YELLOW
    border: str = Colors.WHITE
    text: str = Colors.WHITE
    background: str = ""


# Theme definitions
THEMES: dict[BoardTheme, ThemeColors] = {
    BoardTheme.CLASSIC: ThemeColors(
        player1=Colors.BLUE,
        player2=Colors.RED,
        empty=Colors.DIM,
        valid_move=Colors.GREEN,
        last_move=Colors.YELLOW,
    ),
    BoardTheme.DARK: ThemeColors(
        player1=Colors.CYAN,
        player2=Colors.MAGENTA,
        empty=Colors.DIM,
        valid_move=Colors.GREEN,
        last_move=Colors.YELLOW,
    ),
    BoardTheme.COLORBLIND: ThemeColors(
        player1=Colors.BLUE,
        player2=Colors.YELLOW,
        empty=Colors.DIM,
        valid_move=Colors.WHITE + Colors.BOLD,
        last_move=Colors.CYAN,
    ),
    BoardTheme.MINIMAL: ThemeColors(
        player1="",
        player2="",
        empty="",
        valid_move="",
        last_move="",
        border="",
        text="",
    ),
}


class StateVisualizer:
    """ASCII visualization of game states.

    This class renders game states as ASCII art for terminal display.

    Example:
        >>> config = HumanInterfaceConfig()
        >>> visualizer = StateVisualizer(config)
        >>>
        >>> # Render current board state
        >>> output = visualizer.render_board(observation, valid_actions)
        >>> print(output)
    """

    def __init__(self, config: HumanInterfaceConfig | None = None):
        """Initialize visualizer.

        Args:
            config: Human interface configuration
        """
        self.config = config or HumanInterfaceConfig()
        self._theme = THEMES.get(self.config.get_board_theme_enum(), THEMES[BoardTheme.CLASSIC])
        self._use_colors = True
        self._last_move: tuple[int, int] | None = None

        logger.debug(f"StateVisualizer initialized with theme {self.config.board_theme}")

    def set_theme(self, theme: str | BoardTheme) -> None:
        """Set the visualization theme.

        Args:
            theme: Theme name or enum
        """
        if isinstance(theme, str):
            theme = BoardTheme(theme)
        self._theme = THEMES.get(theme, THEMES[BoardTheme.CLASSIC])

    def disable_colors(self) -> None:
        """Disable color output."""
        self._use_colors = False

    def enable_colors(self) -> None:
        """Enable color output."""
        self._use_colors = True

    def set_last_move(self, position: tuple[int, int] | None) -> None:
        """Set the last move position for highlighting.

        Args:
            position: (row, col) of last move or None
        """
        self._last_move = position

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled.

        Args:
            text: Text to colorize
            color: Color code

        Returns:
            Colored text or original text.
        """
        if self._use_colors and color:
            return f"{color}{text}{Colors.RESET}"
        return text

    def render_board(
        self,
        observation: np.ndarray,
        valid_actions: list[int] | None = None,
        board_size: tuple[int, int] | None = None,
        player_channel: int = 0,
        last_action: int | None = None,
    ) -> str:
        """Render the game board as ASCII.

        Args:
            observation: Game observation (typically CHW format)
            valid_actions: List of valid action indices
            board_size: (rows, cols) if not inferrable from observation
            player_channel: Channel index containing player pieces
            last_action: Last action taken (for highlighting)

        Returns:
            ASCII string representation of board.
        """
        valid_actions = valid_actions or []

        # Try to infer board dimensions
        if observation.ndim == 3:
            # CHW format
            channels, height, width = observation.shape
            board = observation[player_channel]
        elif observation.ndim == 2:
            # HW format
            height, width = observation.shape
            board = observation
        else:
            # Flatten or unknown format
            if board_size:
                height, width = board_size
                board = observation.reshape(height, width)
            else:
                # Assume square
                size = int(np.sqrt(len(observation.flatten())))
                height = width = size
                board = observation.flatten()[: size * size].reshape(height, width)

        # Convert last action to position if provided
        if last_action is not None:
            self._last_move = (last_action // width, last_action % width)

        # Build the board string
        lines = []

        # Title
        if self.config.show_turn_number:
            lines.append(self._color("  Game Board", Colors.BOLD))
            lines.append("")

        # Column headers
        if self.config.show_coordinates:
            header = "    "
            for col in range(width):
                header += f" {chr(65 + col)} "
            lines.append(self._color(header, self._theme.border))
            lines.append(self._color("   " + "---" * width + "-", self._theme.border))

        # Board rows
        for row in range(height):
            line = ""
            if self.config.show_coordinates:
                line = self._color(f" {row + 1:2d}|", self._theme.border)

            for col in range(width):
                cell_value = board[row, col]
                action_idx = row * width + col

                # Determine cell display
                if cell_value > 0.5:
                    # Player 1
                    symbol = " X "
                    color = self._theme.player1
                elif cell_value < -0.5:
                    # Player 2
                    symbol = " O "
                    color = self._theme.player2
                elif action_idx in valid_actions and self.config.highlight_valid_moves:
                    # Valid move
                    symbol = " . "
                    color = self._theme.valid_move
                else:
                    # Empty
                    symbol = " . "
                    color = self._theme.empty

                # Highlight last move
                if self._last_move and (row, col) == self._last_move:
                    color = self._theme.last_move

                line += self._color(symbol, color)

            if self.config.show_coordinates:
                line += self._color("|", self._theme.border)

            lines.append(line)

        # Bottom border
        if self.config.show_coordinates:
            lines.append(self._color("   " + "---" * width + "-", self._theme.border))

        return "\n".join(lines)

    def render_move_probabilities(
        self,
        probabilities: np.ndarray,
        board_size: tuple[int, int],
        top_n: int = 5,
    ) -> str:
        """Render move probability distribution.

        Args:
            probabilities: Probability for each action
            board_size: (rows, cols) of board
            top_n: Number of top moves to show

        Returns:
            ASCII representation of probabilities.
        """
        height, width = board_size

        lines = []
        lines.append(self._color("Move Probabilities:", Colors.BOLD))
        lines.append("")

        # Get top moves
        top_indices = np.argsort(probabilities)[::-1][:top_n]

        for i, idx in enumerate(top_indices):
            prob = probabilities[idx]
            if prob < 0.001:
                continue

            row = idx // width
            col = idx % width
            coord = f"{chr(65 + col)}{row + 1}"

            # Create bar
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)

            line = f"  {i + 1}. {coord}: {bar} {prob:.1%}"
            lines.append(self._color(line, self._theme.text))

        return "\n".join(lines)

    def render_evaluation(
        self,
        evaluation: float,
        perspective: str = "current",
    ) -> str:
        """Render AI evaluation.

        Args:
            evaluation: Value from -1 to 1
            perspective: "current" or "player1"/"player2"

        Returns:
            ASCII representation of evaluation.
        """
        lines = []
        lines.append(self._color("Position Evaluation:", Colors.BOLD))

        # Convert to percentage
        win_prob = (evaluation + 1) / 2 * 100

        # Create evaluation bar
        bar_length = 30
        positive_bars = int(win_prob / 100 * bar_length)
        negative_bars = bar_length - positive_bars

        bar = self._color("█" * positive_bars, self._theme.player1) + self._color(
            "█" * negative_bars, self._theme.player2
        )

        lines.append(f"  {bar}")
        lines.append(f"  Player 1: {win_prob:.1f}% | Player 2: {100 - win_prob:.1f}%")

        return "\n".join(lines)

    def render_mcts_stats(
        self,
        stats: dict[str, Any],
    ) -> str:
        """Render MCTS search statistics.

        Args:
            stats: Dictionary with MCTS statistics

        Returns:
            ASCII representation of stats.
        """
        lines = []
        lines.append(self._color("MCTS Statistics:", Colors.BOLD))

        if "num_simulations" in stats:
            lines.append(f"  Simulations: {stats['num_simulations']}")
        if "max_depth" in stats:
            lines.append(f"  Max Depth: {stats['max_depth']}")
        if "root_value" in stats:
            lines.append(f"  Root Value: {stats['root_value']:.3f}")
        if "thinking_time" in stats:
            lines.append(f"  Thinking Time: {stats['thinking_time']:.2f}s")

        return "\n".join(lines)

    def render_game_info(
        self,
        turn_number: int,
        current_player: int,
        is_human_turn: bool,
        valid_move_count: int,
    ) -> str:
        """Render game information panel.

        Args:
            turn_number: Current turn number
            current_player: Current player (0 or 1)
            is_human_turn: Whether it's human's turn
            valid_move_count: Number of valid moves

        Returns:
            ASCII information panel.
        """
        lines = []
        lines.append(self._color("─" * 40, self._theme.border))

        turn_text = f"Turn {turn_number}"
        lines.append(self._color(turn_text, Colors.BOLD))

        player_text = f"Player {current_player + 1}"
        if is_human_turn:
            player_text += " (Your turn)"
        else:
            player_text += " (AI thinking...)"
        lines.append(player_text)

        lines.append(f"Valid moves: {valid_move_count}")
        lines.append(self._color("─" * 40, self._theme.border))

        return "\n".join(lines)

    def render_full_display(
        self,
        observation: np.ndarray,
        turn_number: int,
        current_player: int,
        is_human_turn: bool,
        valid_actions: list[int],
        ai_evaluation: float | None = None,
        move_probabilities: np.ndarray | None = None,
        mcts_stats: dict[str, Any] | None = None,
        board_size: tuple[int, int] | None = None,
    ) -> str:
        """Render complete game display.

        Args:
            observation: Game observation
            turn_number: Current turn
            current_player: Current player
            is_human_turn: Whether it's human's turn
            valid_actions: Valid action indices
            ai_evaluation: AI's position evaluation
            move_probabilities: Move probabilities from AI
            mcts_stats: MCTS statistics
            board_size: Board dimensions

        Returns:
            Complete ASCII display.
        """
        sections = []

        # Game info
        sections.append(
            self.render_game_info(turn_number, current_player, is_human_turn, len(valid_actions))
        )

        # Board
        sections.append(self.render_board(observation, valid_actions, board_size))

        # AI evaluation (if enabled and available)
        if self.config.show_ai_evaluation and ai_evaluation is not None:
            sections.append("")
            sections.append(self.render_evaluation(ai_evaluation))

        # Move probabilities (if enabled and available)
        if self.config.show_move_probabilities and move_probabilities is not None:
            sections.append("")
            board_shape = board_size or (10, 10)  # Default
            sections.append(self.render_move_probabilities(move_probabilities, board_shape))

        # MCTS stats (if enabled and available)
        if self.config.show_mcts_stats and mcts_stats:
            sections.append("")
            sections.append(self.render_mcts_stats(mcts_stats))

        return "\n".join(sections)

    def render_game_over(
        self,
        winner: int | None,
        human_player: int,
        summary: dict[str, Any],
    ) -> str:
        """Render game over screen.

        Args:
            winner: Winning player (None for draw)
            human_player: Human's player index
            summary: Game summary statistics

        Returns:
            Game over display.
        """
        lines = []
        lines.append("")
        lines.append(self._color("═" * 40, Colors.BOLD))

        if winner is None:
            lines.append(self._color("       GAME OVER - DRAW", Colors.YELLOW))
        elif winner == human_player:
            lines.append(self._color("       GAME OVER - YOU WIN!", Colors.GREEN))
        else:
            lines.append(self._color("       GAME OVER - AI WINS", Colors.RED))

        lines.append(self._color("═" * 40, Colors.BOLD))
        lines.append("")

        # Summary stats
        lines.append("Game Statistics:")
        lines.append(f"  Total turns: {summary.get('total_turns', 0)}")
        lines.append(f"  Your moves: {summary.get('human_moves', 0)}")
        lines.append(f"  AI moves: {summary.get('ai_moves', 0)}")

        if summary.get("avg_ai_time"):
            lines.append(f"  Avg AI time: {summary['avg_ai_time']:.2f}s")

        return "\n".join(lines)
