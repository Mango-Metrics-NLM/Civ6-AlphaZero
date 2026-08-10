"""Shared bridge-state fixtures for end-to-end and policy-engine tests.

Resolves gap-analysis Important #5: ``_minimal_state`` and ``_state_dict``
were inline in :mod:`tests.integration.test_civ6_bridge_e2e`. The W2.2
``Civ6MctsPolicy`` tests will need the same helpers, and the e2e test was
manually serialising every field of :class:`GameStateData`, which means
any new schema field added to the Lua mod would silently widen the gap
before being noticed.

This module exposes two helpers:

- :func:`minimal_state` returns a structurally-valid :class:`GameStateData`
  with just enough content (one tile, two players, one ``end_turn`` legal
  action) for the loop to advance.
- :func:`state_message_dict` wraps the same payload in the JSON shape that
  :meth:`alpha_zero.bridge.file_bridge.MockFileBridge.set_mock_state`
  expects (a ``StateMessage`` envelope).

Both helpers accept ``turn_id`` / ``player_id`` / ``map_size`` overrides
so callers can compose multi-turn / multi-player scenarios without
duplicating the serialisation boilerplate.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from alpha_zero.bridge.protocol import (
    GameStateData,
    LegalAction,
    MapData,
    MessageType,
    PlayerData,
    TileData,
)

# Default scenario knobs. Module-scope so tests that want symmetric setups
# import a single constant instead of repeating literals.
DEFAULT_BOARD_SIZE: int = 8
DEFAULT_PLAYER_CIV: str = "CIVILIZATION_AMERICA"


def minimal_tile(x: int = 0, y: int = 0) -> TileData:
    """One grassland tile with vision flags set."""
    return TileData(
        x=x,
        y=y,
        terrain="TERRAIN_GRASS",
        features=[],
        is_visible=True,
        is_revealed=True,
    )


def minimal_state(
    *,
    turn_id: int = 1,
    player_id: int = 0,
    board_size: int = DEFAULT_BOARD_SIZE,
    legal_actions: list[LegalAction] | None = None,
    is_game_over: bool = False,
    winner_id: int | None = None,
) -> GameStateData:
    """Build a structurally-valid :class:`GameStateData` for tests.

    Defaults aim for the smallest payload that the loop will accept:
    one tile, two players, one ``end_turn`` legal action. Tests that need
    richer state should compose this and mutate fields directly.
    """
    if legal_actions is None:
        legal_actions = [LegalAction(action_type="end_turn", unit_id=0, city_id=0)]

    return GameStateData(
        turn_id=turn_id,
        player_id=player_id,
        map_data=MapData(width=board_size, height=board_size, tiles=[minimal_tile()]),
        units=[],
        cities=[],
        players=[
            PlayerData(id=0, civilization=DEFAULT_PLAYER_CIV, is_alive=True),
            PlayerData(id=1, civilization="CIVILIZATION_ENGLAND", is_alive=True),
        ],
        legal_actions=legal_actions,
        current_player_civ=DEFAULT_PLAYER_CIV,
        is_game_over=is_game_over,
        winner_id=winner_id,
    )


def state_message_dict(
    state: GameStateData | None = None,
    *,
    turn_id: int = 1,
    player_id: int = 0,
    board_size: int = DEFAULT_BOARD_SIZE,
) -> dict[str, Any]:
    """Wrap a :class:`GameStateData` in the StateMessage envelope dict.

    Delegates serialisation to :func:`dataclasses.asdict` so any new
    field added to the protocol propagates automatically — this is what
    the previous inline ``_state_dict`` helper failed to do (gap #5).
    The envelope adds ``type`` / ``msg_id`` / ``timestamp`` fields and
    a ``player`` summary block that the bridge protocol expects.
    """
    if state is None:
        state = minimal_state(turn_id=turn_id, player_id=player_id, board_size=board_size)

    state_payload = asdict(state)
    # GameStateData.from_dict reads the map under "map" (the Lua mod's
    # wire-format name); asdict produces the dataclass field name "map_data".
    # Rename so the round-trip preserves width/height instead of silently
    # defaulting to 0x0 and tripping GameNotStartedError downstream.
    state_payload["map"] = state_payload.pop("map_data")
    # The StateMessage schema expects a ``player`` summary block alongside
    # the ``players`` list - mirror what the Lua mod actually exports.
    state_payload["player"] = {
        "id": state.player_id,
        "civilization": state.current_player_civ,
        "is_alive": True,
    }

    return {
        "type": MessageType.STATE.value,
        "msg_id": state.turn_id,
        "timestamp": 0.0,
        "state": state_payload,
    }
