"""Builders for bridge states used by the search-mode (GM-B5) tests.

Every state is a real :class:`GameStateData`, so the tests that use it go through the
real ``StateConverter`` and nothing about the converted dict is hand-written. The map is a
square of ``size`` tiles with no scaling (``size`` equals the converter's board size), so a
game coordinate and a board coordinate are the same number.
"""

from __future__ import annotations

from collections.abc import Sequence

from alpha_zero.bridge.protocol import (
    CityData,
    GameStateData,
    LegalAction,
    PlayerData,
    TileData,
    UnitData,
)
from tests.fixtures.bridge_snapshots import minimal_state

__all__ = ["city", "make_state", "tile", "unit"]

#: Move budget given to every unit unless a test says otherwise. The simulator offers no
#: action for a unit with no moves left, so a default of zero would make a state vacuous.
DEFAULT_MOVES: float = 2.0


def tile(
    x: int,
    y: int,
    terrain: str = "TERRAIN_GRASS",
    features: Sequence[str] = (),
    yields: dict[str, int] | None = None,
) -> TileData:
    """One visible tile."""
    return TileData(
        x=x,
        y=y,
        terrain=terrain,
        features=list(features),
        is_visible=True,
        is_revealed=True,
        yields=dict(yields) if yields else {},
    )


def unit(
    unit_id: int,
    unit_type: str,
    owner_id: int,
    x: int,
    y: int,
    *,
    moves: float = DEFAULT_MOVES,
    health: int = 100,
    strength: int = 20,
) -> UnitData:
    """One unit with a full move budget."""
    return UnitData(
        id=unit_id,
        unit_type=unit_type,
        owner_id=owner_id,
        x=x,
        y=y,
        health=health,
        max_health=100,
        moves_remaining=moves,
        max_moves=DEFAULT_MOVES,
        combat_strength=strength,
    )


def city(city_id: int, owner_id: int, x: int, y: int, name: str = "City") -> CityData:
    """One city."""
    return CityData(id=city_id, name=name, owner_id=owner_id, x=x, y=y)


def make_state(
    *,
    player_id: int = 0,
    size: int = 16,
    units: Sequence[UnitData] = (),
    cities: Sequence[CityData] = (),
    tiles: Sequence[TileData] = (),
    legal_actions: Sequence[LegalAction] | None = None,
    players: Sequence[PlayerData] | None = None,
    turn_id: int = 3,
) -> GameStateData:
    """A structurally valid state with the given contents.

    Args:
        player_id: The game player id the AI plays as (``GameStateData.player_id``).
        size: Map width and height.
        units: Every unit on the map, own and other.
        cities: Every city on the map, own and other.
        tiles: Tiles that are not the converter's default plains.
        legal_actions: What the game offers; defaults to ``end_turn`` only.
        players: Player rows; defaults to the two of ``minimal_state``.
        turn_id: The game turn.
    """
    state = minimal_state(player_id=player_id, board_size=size, turn_id=turn_id)
    state.map_data.tiles = list(tiles)
    state.units = list(units)
    state.cities = list(cities)
    if legal_actions is not None:
        state.legal_actions = list(legal_actions)
    if players is not None:
        state.players = list(players)
    return state
