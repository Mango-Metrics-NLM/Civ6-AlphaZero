"""Serialization helpers for converting game state to JSON-safe dicts.

Handles numpy array conversion, action decoding, and structured
data formatting for API responses and template rendering.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from alpha_zero.core.civ6_actions import MicroCivActionSpace
from alpha_zero.web import DEFAULT_TOP_MOVES

if TYPE_CHECKING:
    from alpha_zero.human_interface.game_controller import GameState, TurnResult
    from alpha_zero.web.adapter import MicroCivGameAdapter

logger = logging.getLogger(__name__)

# Direction labels for human-readable action descriptions
DIRECTION_LABELS: dict[tuple[int, int], str] = {
    (0, -1): "N",
    (1, -1): "NE",
    (1, 0): "E",
    (1, 1): "SE",
    (0, 1): "S",
    (-1, 1): "SW",
    (-1, 0): "W",
    (-1, -1): "NW",
}

UNIT_TYPE_LABELS: dict[str, str] = {
    "warrior": "Warrior",
    "settler": "Settler",
    "archer": "Archer",
}


def decode_action_readable(
    action_idx: int,
    action_space: MicroCivActionSpace,
) -> str:
    """Decode an action index to a human-readable string.

    Args:
        action_idx: Flat action index.
        action_space: Action space handler for decoding.

    Returns:
        Human-readable description like "Move Warrior NE" or "End Turn".
    """
    try:
        decoded = action_space.decode_action(action_idx)
    except (IndexError, ValueError):
        return f"Action {action_idx}"

    action_type = decoded.get("type", "unknown")

    if action_type == "end_turn":
        return "End Turn"

    if action_type == "move":
        unit_idx = decoded.get("unit_idx", 0)
        direction = decoded.get("direction", (0, 0))
        dir_label = DIRECTION_LABELS.get(tuple(direction), str(direction))
        return f"Move Unit {unit_idx} {dir_label}"

    if action_type == "attack":
        unit_idx = decoded.get("unit_idx", 0)
        direction = decoded.get("direction", (0, 0))
        dir_label = DIRECTION_LABELS.get(tuple(direction), str(direction))
        return f"Attack Unit {unit_idx} {dir_label}"

    if action_type == "skip":
        unit_idx = decoded.get("unit_idx", 0)
        return f"Skip Unit {unit_idx}"

    if action_type == "found_city":
        unit_idx = decoded.get("unit_idx", 0)
        return f"Found City (Unit {unit_idx})"

    if action_type == "produce":
        city_idx = decoded.get("city_idx", 0)
        unit_type = decoded.get("unit_type", "warrior")
        label = UNIT_TYPE_LABELS.get(unit_type, unit_type)
        return f"Produce {label} (City {city_idx})"

    return f"{action_type} (Action {action_idx})"


def serialize_game_state(
    session_id: str,
    game_state: GameState,
    adapter: MicroCivGameAdapter,
    action_space: MicroCivActionSpace,
) -> dict[str, Any]:
    """Serialize full game state for API response.

    Args:
        session_id: Session identifier.
        game_state: Current GameController state.
        adapter: Game environment adapter.
        action_space: Action space for decoding.

    Returns:
        JSON-safe dictionary of the complete game state.
    """
    board_data = adapter.get_board_data()

    # Decode valid actions
    valid_actions = []
    for action_idx in game_state.valid_actions:
        idx = int(action_idx)
        valid_actions.append(
            {
                "index": idx,
                "label": decode_action_readable(idx, action_space),
            }
        )

    return {
        "session_id": session_id,
        "phase": game_state.phase.value,
        "turn_number": int(game_state.turn_number),
        "current_player": int(game_state.current_player),
        "human_player": int(game_state.human_player),
        "board": _sanitize_numpy(board_data),
        "valid_actions": valid_actions,
        "history": serialize_move_history(game_state.history, action_space),
        "game_over": game_state.phase.value == "game_over",
        "winner": _find_winner(game_state),
    }


def serialize_turn_result(
    result: TurnResult,
    action_space: MicroCivActionSpace,
) -> dict[str, Any]:
    """Serialize a turn result for API response.

    Args:
        result: TurnResult from GameController.
        action_space: Action space for decoding.

    Returns:
        JSON-safe dictionary of the turn result.
    """
    data: dict[str, Any] = {
        "action": int(result.action),
        "action_label": decode_action_readable(int(result.action), action_space),
        "player": int(result.player),
        "is_human": bool(result.is_human),
        "reward": float(result.reward),
        "game_over": bool(result.game_over),
        "winner": int(result.winner) if result.winner is not None else None,
        "thinking_time": round(float(result.thinking_time), 3),
        "timestamp": result.timestamp,
    }

    if result.ai_evaluation is not None:
        data["ai_evaluation"] = round(float(result.ai_evaluation), 4)

    if result.move_probabilities is not None:
        data["top_moves"] = _extract_top_moves(
            result.move_probabilities, action_space, top_n=DEFAULT_TOP_MOVES
        )

    return data


def serialize_move_history(
    history: list[TurnResult],
    action_space: MicroCivActionSpace,
) -> list[dict[str, Any]]:
    """Serialize move history for API response.

    Args:
        history: List of TurnResult objects.
        action_space: Action space for decoding.

    Returns:
        List of JSON-safe turn result dictionaries.
    """
    return [serialize_turn_result(turn, action_space) for turn in history]


def serialize_confidence(
    result: TurnResult,
    action_space: MicroCivActionSpace,
) -> dict[str, Any]:
    """Serialize AI confidence data for display.

    Args:
        result: TurnResult containing AI evaluation data.
        action_space: Action space for decoding.

    Returns:
        Dictionary with win probability and top moves.
    """
    data: dict[str, Any] = {
        "has_evaluation": result.ai_evaluation is not None,
        "win_probability": None,
        "win_pct": None,
        "top_moves": [],
        "thinking_time": round(result.thinking_time, 3),
    }

    if result.ai_evaluation is not None:
        # Value is in [-1, 1], convert to win probability [0, 1]
        win_prob = (float(result.ai_evaluation) + 1.0) / 2.0
        data["win_probability"] = round(win_prob, 4)
        data["win_pct"] = round(win_prob * 100, 1)

    if result.move_probabilities is not None:
        data["top_moves"] = _extract_top_moves(
            result.move_probabilities, action_space, top_n=DEFAULT_TOP_MOVES
        )

    return data


def serialize_board_for_template(
    adapter: MicroCivGameAdapter,
    valid_actions: list[int],
    action_space: MicroCivActionSpace,
    human_player: int = 0,
    last_action: int | None = None,
) -> dict[str, Any]:
    """Serialize board data specifically for HTML template rendering.

    Adds per-cell metadata (units, cities, valid move targets) for
    the Jinja2 board template to render directly.

    Args:
        adapter: Game environment adapter.
        valid_actions: List of valid action indices.
        action_space: Action space for decoding.
        human_player: Which player is human (for perspective).
        last_action: Last action taken (for highlighting).

    Returns:
        Template-ready board data dictionary.
    """
    board_data = adapter.get_board_data()
    board_size = board_data["board_size"]

    # Build lookup maps for units and cities by position
    unit_map: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for unit in board_data["units"]:
        pos = (unit["x"], unit["y"])
        unit_map.setdefault(pos, []).append(unit)

    city_map: dict[tuple[int, int], dict[str, Any]] = {}
    for city in board_data["cities"]:
        pos = (city["x"], city["y"])
        city_map[pos] = city

    # Parse valid actions to find move targets for template clickability
    move_targets: set[int] = set()
    for action_idx in valid_actions:
        try:
            action_space.decode_action(action_idx)
        except (IndexError, ValueError):
            continue
        move_targets.add(action_idx)

    # Enrich grid cells
    enriched_grid: list[list[dict[str, Any]]] = []
    for y in range(board_size):
        row: list[dict[str, Any]] = []
        for x in range(board_size):
            cell = board_data["terrain_grid"][y][x].copy()
            pos = (x, y)
            cell["units"] = unit_map.get(pos, [])
            cell["city"] = city_map.get(pos)
            cell["has_own_unit"] = any(u["player"] == human_player for u in cell["units"])
            cell["has_enemy_unit"] = any(u["player"] != human_player for u in cell["units"])
            cell["has_own_city"] = (
                cell["city"] is not None and cell["city"]["player"] == human_player
            )
            cell["has_enemy_city"] = (
                cell["city"] is not None and cell["city"]["player"] != human_player
            )
            row.append(cell)
        enriched_grid.append(row)

    result: dict[str, Any] = _sanitize_numpy(
        {
            "grid": enriched_grid,
            "board_size": board_size,
            "current_player": board_data["current_player"],
            "turn": board_data["turn"],
            "human_player": human_player,
            "valid_action_indices": valid_actions,
        }
    )
    return result


def _extract_top_moves(
    probabilities: np.ndarray,
    action_space: MicroCivActionSpace,
    top_n: int = DEFAULT_TOP_MOVES,
) -> list[dict[str, Any]]:
    """Extract top-N moves from probability distribution.

    Args:
        probabilities: Action probability array.
        action_space: Action space for decoding.
        top_n: Number of top moves to return.

    Returns:
        List of dicts with index, label, and probability.
    """
    if probabilities is None:
        return []

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.sum() == 0:
        return []

    # Get top indices
    top_indices = np.argsort(probs)[-top_n:][::-1]

    moves = []
    for idx in top_indices:
        prob = float(probs[idx])
        if prob <= 0:
            continue
        moves.append(
            {
                "index": int(idx),
                "label": decode_action_readable(int(idx), action_space),
                "probability": round(prob, 4),
                "pct": round(prob * 100, 1),
            }
        )

    return moves


def _sanitize_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types.

    Ensures all values in nested dicts/lists are JSON-serializable.

    Args:
        obj: Value that may contain numpy types.

    Returns:
        Value with all numpy types converted to Python natives.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_numpy(item) for item in obj]
    return obj


def _find_winner(game_state: GameState) -> int | None:
    """Extract winner from game state history."""
    if game_state.history and game_state.history[-1].game_over:
        return game_state.history[-1].winner
    return None
